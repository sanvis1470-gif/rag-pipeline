"""Multi-format loader: markdown / text / HTML / PDF -> Document (plaintext segments + metadata)."""
from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .models import Document, Segment

SUPPORTED = {
    ".md": "markdown", ".markdown": "markdown",
    ".txt": "text",
    ".html": "html", ".htm": "html",
    ".pdf": "pdf",
}


def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)          # ligatures (ﬁ -> fi), nbsp, fullwidth chars
    t = t.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" ?\n ?", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _decode(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return text.replace("\r\n", "\n").replace("\r", "\n")      # Windows files use CRLF



# ------------------------------------------- heading heuristic (txt / pdf)
_URLISH = re.compile(r"[/@\[\]]|https?:|www\.")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)+|\d+[.)])\s+\S")
_SMALL = {"and", "or", "of", "the", "for", "to", "in", "on", "with", "a", "an", "vs", "&"}


def _is_numbered_heading(line: str) -> bool:
    return (bool(_NUMBERED.match(line)) and len(line) <= 90
            and line[-1] not in ".,;:!?" and not _URLISH.search(line))


def _is_plain_heading(line: str, prev_blank: bool, next_nonblank: bool, later: bool) -> bool:
    """Short standalone line without sentence punctuation. Without a blank line before it
    (PDF text has none) it must also look like a Title Case / ALL CAPS title."""
    if not later or len(line) > 60 or _URLISH.search(line) or line[-1] in ".,;:!?":
        return False
    if line.startswith(("-", "*", "\u2022")):
        return False
    words = line.split()
    if prev_blank:
        return 1 <= len(words) <= 8
    return (next_nonblank and 1 <= len(words) <= 6 and
            all(w[0].isupper() or w.lower() in _SMALL or not w[0].isalnum() for w in words))


class _HeadingSplitter:
    """Splits plain text into Segments at lines that look like headings; keeps a heading path
    (e.g. '2. CS Fundamentals > Operating Systems') that can carry over from page to page."""

    def __init__(self) -> None:
        self.stack: list[tuple[int, str]] = []
        self.last_numbered = 0

    @property
    def path(self) -> str:
        return " > ".join(t for _, t in self.stack)

    def split(self, text: str, page: int | None = None) -> list[Segment]:
        lines = [ln.strip() for ln in text.split("\n")]
        segs: list[Segment] = []
        buf: list[str] = []
        heading = self.path
        has_body = False

        def emit() -> None:
            body = normalize_text("\n".join(buf))
            if body:
                segs.append(Segment(body, heading, page))
            buf.clear()

        i = 0
        while i < len(lines):
            line = lines[i]
            level = None
            if line and any(lines[i + 1:]):
                m = _NUMBERED.match(line)
                if m and _is_numbered_heading(line):
                    while (line.count("(") > line.count(")") and i + 1 < len(lines)
                           and 0 < len(lines[i + 1]) <= 12):        # heading wrapped onto next line
                        i += 1
                        line += " " + lines[i]
                    level = m.group(1).rstrip(".)").count(".") + 1
                    self.last_numbered = level
                elif _is_plain_heading(line, i == 0 or not lines[i - 1],
                                       i + 1 < len(lines) and bool(lines[i + 1]), True):
                    level = self.last_numbered + 1 if self.last_numbered else 1
            if level:
                if has_body:                       # heading directly followed by heading: keep both
                    emit()                         # lines together instead of a heading-only chunk
                while self.stack and self.stack[-1][0] >= level:
                    self.stack.pop()
                self.stack.append((level, line))
                heading = self.path
                has_body = False
            elif line:
                has_body = True
            buf.append(line)
            i += 1
        emit()
        return segs

# ---------------------------------------------------------------- text
def parse_text(text: str) -> list[Segment]:
    body = normalize_text(text)
    if not body:
        return []
    return _HeadingSplitter().split(body) or [Segment(body)]


# ------------------------------------------------------------ markdown
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _strip_md_inline(line: str) -> str:
    line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", line)        # images
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)         # links
    line = re.sub(r"(\*\*|__)(.+?)\1", r"\2", line)              # bold
    line = re.sub(r"(?<!\w)(\*|_)(.+?)\1(?!\w)", r"\2", line)    # italics
    line = re.sub(r"`([^`]+)`", r"\1", line)                     # inline code
    return line


def _emit(segments: list[Segment], buf: list[str], heading: str, page=None) -> None:
    body = normalize_text("\n".join(buf))
    if body:
        segments.append(Segment(body, heading, page))
    buf.clear()


def parse_markdown(text: str) -> list[Segment]:
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)    # YAML front matter
    segments: list[Segment] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    heading = ""
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            _emit(segments, buf, heading)
            level, title = len(m.group(1)), _strip_md_inline(m.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            heading = " > ".join(t for _, t in stack)
            buf.append(title)                    # keep heading text inside the chunk body
        else:
            buf.append(line if in_fence else _strip_md_inline(line))
    _emit(segments, buf, heading)
    return segments


# ---------------------------------------------------------------- html
_HEADINGS = {f"h{i}" for i in range(1, 7)}
_LEAF_TAGS = sorted(_HEADINGS | {"p", "li", "pre", "td", "th", "dt", "dd", "div", "blockquote"})


def parse_html(html: str) -> list[Segment]:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "nav", "footer", "aside", "form", "svg"]):
        t.decompose()
    root = soup.body or soup
    base = soup.title.get_text(strip=True) if soup.title else ""
    segments: list[Segment] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    heading = base
    for el in root.find_all(_LEAF_TAGS):
        if el.find(_LEAF_TAGS):                  # only leaf blocks, avoids duplicated text
            continue
        text = el.get_text() if el.name == "pre" else el.get_text(" ", strip=True)
        text = text.strip()
        if not text:
            continue
        if el.name in _HEADINGS:
            _emit(segments, buf, heading)
            level = int(el.name[1])
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            heading = " > ".join(t for _, t in stack)
        buf.append(text)
    _emit(segments, buf, heading)
    return segments


# ----------------------------------------------------------------- pdf
def _pdf_outline(reader: PdfReader) -> dict[int, str]:
    """page index -> bookmark title (best effort)."""
    out: dict[int, str] = {}

    def walk(items):
        for it in items:
            if isinstance(it, list):
                walk(it)
            else:
                try:
                    out.setdefault(reader.get_destination_page_number(it), it.title)
                except Exception:
                    pass
    try:
        walk(reader.outline)
    except Exception:
        pass
    return out


def parse_pdf(data: bytes) -> list[Segment]:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        reader.decrypt("")
    outline = _pdf_outline(reader)
    segments: list[Segment] = []
    heading = ""
    splitter = _HeadingSplitter()          # only used when the PDF has no bookmarks
    for i, page in enumerate(reader.pages):
        heading = outline.get(i, heading)
        txt = page.extract_text() or ""
        txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)            # de-hyphenate line breaks
        txt = re.sub(r"\n(?=[a-z])", " ", txt)                # unwrap only mid-sentence line wraps
        txt = normalize_text(txt)
        if not txt:
            continue
        if outline:
            segments.append(Segment(txt, heading, page=i + 1))
        else:
            segments.extend(splitter.split(txt, page=i + 1))
    return segments


# ----------------------------------------------------------- public API
def parse_bytes(raw: bytes, filename: str) -> Document:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type {ext!r}; supported: {sorted(SUPPORTED)}")
    kind = SUPPORTED[ext]
    if kind == "pdf":
        segments = parse_pdf(raw)
    else:
        text = _decode(raw)
        segments = {"markdown": parse_markdown, "html": parse_html, "text": parse_text}[kind](text)
    if not segments:
        raise ValueError(f"No extractable text in {filename} (scanned PDF? it needs OCR)")
    doc_id = hashlib.sha256(raw).hexdigest()[:16]
    return Document(doc_id, Path(filename).name, kind, segments)


def load_document(path: str | Path) -> tuple[Document, bytes]:
    path = Path(path)
    raw = path.read_bytes()
    return parse_bytes(raw, path.name), raw
