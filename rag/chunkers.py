"""Three switchable chunking strategies. All return ChunkDrafts that carry heading/page metadata."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np

from .config import Config
from .embeddings import Embedder
from .models import ChunkDraft, Document

# (split regex, joiner) from coarse to fine; None = character level
SEPARATORS = [
    (r"\n\n+", "\n\n"),
    (r"\n", "\n"),
    (r"(?<=[.!?])\s+", " "),
    (r"\s+", " "),
    (None, ""),
]


def recursive_split(text: str, size: int, overlap: int = 0, level: int = 0) -> list[str]:
    """Split on the coarsest separator that works, merge pieces up to `size`, recurse on oversized pieces."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    for lvl in range(level, len(SEPARATORS)):
        pat, joiner = SEPARATORS[lvl]
        parts = list(text) if pat is None else [p for p in re.split(pat, text) if p.strip()]
        if len(parts) > 1 or pat is None:
            break

    def size_of(ps: list[str]) -> int:
        return sum(map(len, ps)) + len(joiner) * max(len(ps) - 1, 0)

    chunks: list[str] = []
    buf: list[str] = []

    def flush():
        if buf:
            chunks.append(joiner.join(buf).strip())

    for p in parts:
        if len(p) > size:                              # piece alone is too big -> go finer
            flush()
            buf.clear()
            chunks.extend(recursive_split(p, size, overlap, lvl + 1))
            continue
        if buf and size_of(buf + [p]) > size:
            flush()
            tail: list[str] = []                       # carry trailing parts over as overlap
            for q in reversed(buf):
                if size_of([q] + tail) > overlap:
                    break
                tail.insert(0, q)
            buf[:] = tail
            if buf and size_of(buf + [p]) > size:
                buf.clear()
        buf.append(p)
    flush()
    return [c for c in chunks if c]


class Chunker(ABC):
    name: str

    @abstractmethod
    def chunk(self, doc: Document) -> list[ChunkDraft]: ...


class FixedSizeChunker(Chunker):
    """Baseline: sliding window over the whole text, ignores structure."""
    name = "fixed"

    def __init__(self, size: int, overlap: int):
        self.size, self.overlap = size, overlap

    def chunk(self, doc: Document) -> list[ChunkDraft]:
        text, step, out, start = doc.text, self.size - self.overlap, [], 0
        while start < len(text):
            end = min(start + self.size, len(text))
            piece = text[start:end]
            if piece.strip():
                lead = len(piece) - len(piece.lstrip())
                seg = doc.segment_at(start + lead)
                out.append(ChunkDraft(piece.strip(), seg.heading, seg.page))
            if end == len(text):
                break
            start += step
        return out


class RecursiveHeaderChunker(Chunker):
    """Structure-aware: every section (from headings / pages) is chunked on its own,
    splitting on paragraphs -> lines -> sentences -> words only when a section is too big."""
    name = "recursive"

    def __init__(self, size: int, overlap: int):
        self.size, self.overlap = size, overlap

    def chunk(self, doc: Document) -> list[ChunkDraft]:
        return [ChunkDraft(piece, seg.heading, seg.page)
                for seg in doc.segments
                for piece in recursive_split(seg.text, self.size, self.overlap)]


_SPLIT_RE = re.compile(r"((?<=[.!?])\s+|\n{2,})")


def split_sentences(text: str) -> tuple[list[str], list[str]]:
    """Sentences plus the separator ('\\n\\n', '\\n' or ' ') that preceded each one, so chunks can be
    rebuilt with the original paragraph / line breaks."""
    sents: list[str] = []
    seps: list[str] = []
    sep = ""
    for k, part in enumerate(_SPLIT_RE.split(text)):
        if k % 2:
            sep = "\n\n" if "\n\n" in part else ("\n" if "\n" in part else " ")
        elif part.strip():
            seps.append(sep if sents else "")
            sents.append(part.strip())
    return sents, seps


def _join(sents: list[str], seps: list[str], a: int, b: int) -> str:
    return sents[a] + "".join(seps[i] + sents[i] for i in range(a + 1, b))


class SemanticChunker(Chunker):
    """Split sentences where embedding similarity between neighbouring windows drops the most."""
    name = "semantic"

    def __init__(self, embedder: Embedder, percentile: float, min_chars: int, max_chars: int, buffer: int = 1):
        self.embedder, self.percentile = embedder, percentile
        self.min_chars, self.max_chars, self.buffer = min_chars, max_chars, buffer

    def _group(self, sents: list[str], seps: list[str]) -> list[str]:
        n = len(sents)
        if n < 3:
            return [_join(sents, seps, 0, n)]
        b = self.buffer                                  # embed each sentence with its neighbours for stability
        windows = [" ".join(sents[max(0, i - b): i + b + 1]) for i in range(n)]
        emb = self.embedder.embed(windows)
        dist = 1.0 - (emb[:-1] * emb[1:]).sum(axis=1)    # dist[i] = gap between sentence i and i+1
        thr = np.percentile(dist, self.percentile)
        groups: list[str] = []
        start, cur_len = 0, len(sents[0])
        for i in range(1, n):
            topic_break = dist[i - 1] >= thr and dist[i - 1] > 0
            too_long = cur_len + len(seps[i]) + len(sents[i]) > self.max_chars
            if (topic_break and cur_len >= self.min_chars) or too_long:
                groups.append(_join(sents, seps, start, i))
                start, cur_len = i, len(sents[i])
            else:
                cur_len += len(seps[i]) + len(sents[i])
        groups.append(_join(sents, seps, start, n))
        return groups

    def chunk(self, doc: Document) -> list[ChunkDraft]:
        out: list[ChunkDraft] = []
        for seg in doc.segments:
            sents, seps = split_sentences(seg.text)
            if not sents:
                continue
            for g in self._group(sents, seps):
                for piece in recursive_split(g, self.max_chars):   # hard cap for giant "sentences"
                    out.append(ChunkDraft(piece, seg.heading, seg.page))
        return out


def make_chunker(cfg: Config, embedder: Embedder, strategy: str | None = None) -> Chunker:
    strategy = strategy or cfg.strategy
    if strategy == "fixed":
        return FixedSizeChunker(cfg.chunk_size, cfg.chunk_overlap)
    if strategy == "recursive":
        return RecursiveHeaderChunker(cfg.chunk_size, cfg.chunk_overlap)
    if strategy == "semantic":
        return SemanticChunker(embedder, cfg.semantic_percentile, cfg.semantic_min_chars, cfg.semantic_max_chars)
    raise ValueError(f"unknown strategy {strategy!r}")
