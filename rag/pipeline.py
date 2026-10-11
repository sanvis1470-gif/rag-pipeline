from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .chunkers import make_chunker
from .config import Config
from .dedup import find_duplicates
from .embeddings import Embedder, make_embedder
from .indexes import ChunkIndex
from .loaders import load_document
from .models import Chunk, Document
from .rerankers import Reranker
from .retrieval import Retriever
from .store import DocumentStore


@dataclass
class IngestReport:
    doc_id: str
    source: str
    strategy: str
    n_chunks: int
    n_inserted: int
    n_duplicates: int


class Pipeline:
    def __init__(self, cfg: Config | None = None, embedder: Embedder | None = None, auto_repair: bool = True,
                 reranker: Reranker | None = None):
        self.cfg = cfg or Config()
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self._check_embedding_model()
        self.embedder = embedder or make_embedder(self.cfg)
        self.store = DocumentStore(self.cfg.raw_dir, self.cfg.processed_dir)
        self.index = ChunkIndex(self.cfg.chroma_dir, self.cfg.bm25_path, self.cfg.collection_name, auto_repair)
        self.retriever = Retriever(self.index, self.embedder, self.cfg, reranker)

    def _check_embedding_model(self) -> None:
        """Vectors from different models can't share an index -> refuse to mix them."""
        want = {"provider": self.cfg.embedding_provider, "model": self.cfg.embedding_model}
        path = self.cfg.embedding_meta_path
        if path.exists():
            have = json.loads(path.read_text())
            if have != want:
                raise RuntimeError(
                    f"This data dir was built with {have}, but you are using {want}. "
                    f"Delete {self.cfg.data_dir} (or use another --data-dir) and re-ingest.")
        else:
            path.write_text(json.dumps(want))

    # --------------------------------------------------------------- ingest
    def ingest_file(self, path: str | Path, strategy: str | None = None) -> IngestReport:
        doc, raw = load_document(path)
        self.store.save(raw, doc)                      # raw + processed, so re-index needs no re-upload
        return self.index_document(doc, strategy)

    def ingest_many(self, paths: Iterable[str | Path], strategy: str | None = None) -> list[IngestReport]:
        return [self.ingest_file(p, strategy) for p in paths]

    def reindex(self, strategy: str | None = None, reparse: bool = False) -> list[IngestReport]:
        """Rebuild the whole index from stored docs. Starts from an empty index and goes through
        documents in alphabetical order, so for duplicate documents the same copy always survives."""
        docs = sorted(self.store.iter_reparsed() if reparse else self.store.iter_processed(),
                      key=lambda d: (d.source.lower(), d.doc_id))
        self.index.clear()
        self.cfg.dup_log.unlink(missing_ok=True)
        return [self.index_document(d, strategy) for d in docs]

    # ---------------------------------------------------------------- core
    def index_document(self, doc: Document, strategy: str | None = None) -> IngestReport:
        strategy = strategy or self.cfg.strategy
        drafts = [d for d in make_chunker(self.cfg, self.embedder, strategy).chunk(doc) if d.text.strip()]
        chunks = [Chunk(f"{doc.doc_id}:{strategy}:{i}", doc.doc_id, doc.source, i,
                        d.text, d.heading, d.page, strategy)
                  for i, d in enumerate(drafts)]
        embs = self.embedder.embed([c.text for c in chunks])      # embed BEFORE touching the index

        self.index.delete_doc(doc.doc_id)              # re-ingesting a doc replaces its old chunks
        dups = find_duplicates([c.id for c in chunks], embs, self.index, self.cfg.dedup_threshold)
        keep = [i for i, d in enumerate(dups) if d is None]
        if keep:
            self.index.add([chunks[i] for i in keep], embs[keep])
        self._log_duplicates(chunks, dups)
        return IngestReport(doc.doc_id, doc.source, strategy, len(chunks), len(keep), len(chunks) - len(keep))

    def _log_duplicates(self, chunks, dups) -> None:
        rows = [{"chunk_id": c.id, "source": c.source, "duplicate_of": d.duplicate_of,
                 "similarity": round(d.similarity, 4), "scope": d.scope, "preview": c.text[:120]}
                for c, d in zip(chunks, dups) if d is not None]
        if rows:
            with self.cfg.dup_log.open("a", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
