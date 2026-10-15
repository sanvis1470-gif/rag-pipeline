"""ChromaDB (dense) + BM25 (sparse) over the same chunks.

Chroma is the source of truth. BM25 is derived data, persisted next to it, and is rebuilt from
Chroma whenever the two id sets disagree (e.g. a crash between the two writes).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from .models import Chunk

_BATCH = 500


TOKENIZER_VERSION = 2      # bump when tokenize() changes: the stored BM25 index is then rebuilt from Chroma

# Very common English words carry no signal for keyword search ("what does a 404 ... mean" would
# otherwise match nearly every chunk on "what", "does", "a"). Kept deliberately conservative.
STOPWORDS = frozenset("""a about above after again all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have having he
her here hers him his how i if in into is it its itself just me more most my no nor not of off on once only or other
our out over own same she should so some such than that the their them then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you your""".split())


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    toks = re.findall(r"\w+", text.lower())
    return toks if keep_stopwords else [t for t in toks if t not in STOPWORDS]


class BM25Index:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.ids: list[str] = []
        self.tokens: list[list[str]] = []
        self._model: BM25Okapi | None = None
        if self.path.exists():
            d = json.loads(self.path.read_text())
            if d.get("tokenizer") == TOKENIZER_VERSION:
                self.ids, self.tokens = d["ids"], d["tokens"]
            # else: stale tokens from an older tokenizer -> start empty; ChunkIndex.repair() rebuilds from Chroma

    def _save(self) -> None:
        self._model = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"tokenizer": TOKENIZER_VERSION, "ids": self.ids, "tokens": self.tokens}))

    def add(self, ids: list[str], texts: list[str]) -> None:
        self.ids += ids
        self.tokens += [tokenize(t) for t in texts]
        self._save()

    def remove(self, ids: list[str]) -> None:
        drop = set(ids)
        keep = [i for i, x in enumerate(self.ids) if x not in drop]
        self.ids = [self.ids[i] for i in keep]
        self.tokens = [self.tokens[i] for i in keep]
        self._save()

    def replace_all(self, ids: list[str], texts: list[str]) -> None:
        self.ids, self.tokens = list(ids), [tokenize(t) for t in texts]
        self._save()

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        if self._model is None:
            self._model = BM25Okapi(self.tokens)
        q = tokenize(query) or tokenize(query, keep_stopwords=True)     # query made only of stopwords: keep them
        scores = self._model.get_scores(q)
        top = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]


class ChunkIndex:
    def __init__(self, chroma_dir: Path, bm25_path: Path, collection: str, auto_repair: bool = True):
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self.bm25 = BM25Index(bm25_path)
        if auto_repair:
            self.repair()

    # ------------------------------------------------------------ sync
    def _all(self) -> tuple[list[str], list[str]]:
        ids: list[str] = []
        docs: list[str] = []
        while True:
            r = self.col.get(include=["documents"], limit=5000, offset=len(ids))
            if not r["ids"]:
                return ids, docs
            ids += r["ids"]
            docs += r["documents"]

    def is_in_sync(self) -> bool:
        ids, _ = self._all()
        return len(ids) == len(self.bm25.ids) and set(ids) == set(self.bm25.ids)

    def repair(self) -> bool:
        """Rebuild BM25 from Chroma if they disagree. Returns True if a repair was needed."""
        if self.is_in_sync():
            return False
        ids, docs = self._all()
        self.bm25.replace_all(ids, docs)
        return True

    # ----------------------------------------------------------- write
    def add(self, chunks: list[Chunk], embs: np.ndarray) -> None:
        for i in range(0, len(chunks), _BATCH):
            part = chunks[i:i + _BATCH]
            self.col.add(
                ids=[c.id for c in part],
                embeddings=embs[i:i + _BATCH].tolist(),
                documents=[c.text for c in part],
                metadatas=[c.metadata() for c in part],
            )
        self.bm25.add([c.id for c in chunks], [c.text for c in chunks])

    def clear(self) -> None:
        """Drop every chunk from both indexes."""
        name = self.col.name
        self.client.delete_collection(name)
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self.bm25.replace_all([], [])

    def delete_doc(self, doc_id: str) -> int:
        ids = self.col.get(where={"doc_id": doc_id}, include=[])["ids"]
        if ids:
            self.col.delete(ids=ids)
            self.bm25.remove(ids)
        return len(ids)

    # ------------------------------------------------------------ read
    def count(self) -> int:
        return self.col.count()

    def nearest(self, embs: np.ndarray) -> list[tuple[str, float] | None]:
        """Nearest existing chunk (id, cosine similarity) for every row of `embs`."""
        if self.col.count() == 0 or len(embs) == 0:
            return [None] * len(embs)
        r = self.col.query(query_embeddings=embs.tolist(), n_results=1, include=["distances"])
        return [(ids[0], 1.0 - dists[0]) for ids, dists in zip(r["ids"], r["distances"])]

    def search_dense(self, q_emb: np.ndarray, k: int = 5) -> list[dict]:
        n = min(k, self.col.count())
        if n == 0:
            return []
        r = self.col.query(query_embeddings=[q_emb.tolist()], n_results=n,
                           include=["documents", "metadatas", "distances"])
        return [{"id": i, "score": 1.0 - d, "text": t, **m}
                for i, t, m, d in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]

    def search_bm25(self, query: str, k: int = 5) -> list[dict]:
        hits = self.bm25.search(query, k)
        if not hits:
            return []
        r = self.col.get(ids=[h[0] for h in hits], include=["documents", "metadatas"])
        by_id = {i: (t, m) for i, t, m in zip(r["ids"], r["documents"], r["metadatas"])}
        return [{"id": i, "score": s, "text": by_id[i][0], **by_id[i][1]} for i, s in hits if i in by_id]
