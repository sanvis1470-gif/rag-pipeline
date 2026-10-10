"""Keeps raw uploads + processed (normalized) docs on disk so we can re-index without re-uploading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .loaders import parse_bytes
from .models import Document

_SEP = "__"


class DocumentStore:
    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir, self.processed_dir = Path(raw_dir), Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def save(self, raw: bytes, doc: Document) -> None:
        (self.raw_dir / f"{doc.doc_id}{_SEP}{doc.source}").write_bytes(raw)
        self.save_processed(doc)

    def save_processed(self, doc: Document) -> None:
        (self.processed_dir / f"{doc.doc_id}.json").write_text(
            json.dumps(doc.to_dict(), ensure_ascii=False), encoding="utf-8")

    def iter_processed(self) -> Iterator[Document]:
        for p in sorted(self.processed_dir.glob("*.json")):
            yield Document.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def iter_reparsed(self) -> Iterator[Document]:
        """Re-run the loaders over the stored raw files (use after improving a parser)."""
        for p in sorted(self.raw_dir.iterdir()):
            _, _, name = p.name.partition(_SEP)
            doc = parse_bytes(p.read_bytes(), name)
            self.save_processed(doc)
            yield doc
