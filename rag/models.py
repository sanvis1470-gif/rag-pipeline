from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass

SEGMENT_JOINER = "\n\n"


@dataclass
class Segment:
    """A contiguous piece of a document that shares one heading / page."""
    text: str
    heading: str = ""
    page: int | None = None


@dataclass
class Document:
    """Normalized document: plaintext segments + provenance."""
    doc_id: str          # sha256(raw bytes)[:16] -> same file always maps to same id
    source: str          # original filename
    file_type: str       # markdown | text | html | pdf
    segments: list[Segment]

    def __post_init__(self):
        self._starts: list[int] = []
        pos = 0
        for s in self.segments:
            self._starts.append(pos)
            pos += len(s.text) + len(SEGMENT_JOINER)
        self.text = SEGMENT_JOINER.join(s.text for s in self.segments)

    def segment_at(self, offset: int) -> Segment:
        """Which segment does character `offset` of `self.text` belong to?"""
        i = bisect_right(self._starts, offset) - 1
        return self.segments[max(i, 0)]

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "file_type": self.file_type,
            "segments": [asdict(s) for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(d["doc_id"], d["source"], d["file_type"],
                   [Segment(**s) for s in d["segments"]])


@dataclass
class ChunkDraft:
    text: str
    heading: str = ""
    page: int | None = None


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    chunk_index: int
    text: str
    heading: str
    page: int | None
    strategy: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    def metadata(self) -> dict:
        # Chroma metadata can't hold None -> use "" / -1 sentinels
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "heading": self.heading or "",
            "page": self.page if self.page is not None else -1,
            "strategy": self.strategy,
            "char_count": self.char_count,
        }
