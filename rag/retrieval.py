"""Hybrid retrieval: dense (Chroma) + sparse (BM25) -> weighted Reciprocal Rank Fusion -> cross-encoder rerank."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .embeddings import Embedder
from .indexes import ChunkIndex
from .rerankers import Reranker, make_reranker

MODES = ("dense", "sparse", "hybrid")


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    heading: str
    page: int                       # -1 when the source has no pages
    chunk_index: int
    strategy: str
    dense_rank: int | None = None   # 1-based position in the dense list (None = not retrieved by dense)
    dense_score: float | None = None    # cosine similarity
    sparse_rank: int | None = None
    sparse_score: float | None = None   # BM25 score
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @classmethod
    def from_hit(cls, h: dict) -> "RetrievedChunk":
        return cls(h["id"], h["text"], h["source"], h["heading"], h["page"], h["chunk_index"], h["strategy"])


@dataclass
class RetrievalResult:
    query: str
    mode: str
    candidates: list[RetrievedChunk]   # what went into the reranker (or the final cut if no rerank)
    results: list[RetrievedChunk]      # final top-n, best first
    reranked: bool = False
    weights: tuple[float, float] = field(default=(1.0, 0.0))


def reciprocal_rank_fusion(dense: list[dict], sparse: list[dict], dense_weight: float = 0.7,
                           sparse_weight: float = 0.3, k: int = 60) -> list[RetrievedChunk]:
    """score(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d)).

    Uses ranks, not raw scores, so cosine similarities and BM25 scores (different scales) combine
    without normalization. A chunk found by both lists gets both terms and rises to the top.
    Weights are normalized to sum to 1, so 0.7/0.3 and 7/3 behave identically."""
    total = dense_weight + sparse_weight
    if total <= 0 or min(dense_weight, sparse_weight) < 0:
        raise ValueError("weights must be >= 0 and not both 0")
    dw, sw = dense_weight / total, sparse_weight / total
    pool: dict[str, RetrievedChunk] = {}
    for rank, h in enumerate(dense, 1):
        c = pool.setdefault(h["id"], RetrievedChunk.from_hit(h))
        c.dense_rank, c.dense_score = rank, h["score"]
        c.rrf_score += dw / (k + rank)
    for rank, h in enumerate(sparse, 1):
        c = pool.setdefault(h["id"], RetrievedChunk.from_hit(h))
        c.sparse_rank, c.sparse_score = rank, h["score"]
        c.rrf_score += sw / (k + rank)
    return sorted(pool.values(), key=lambda c: (-c.rrf_score, c.id))      # id breaks ties deterministically


class Retriever:
    def __init__(self, index: ChunkIndex, embedder: Embedder, cfg: Config, reranker: Reranker | None = None):
        self.index, self.embedder, self.cfg = index, embedder, cfg
        self._reranker = reranker

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:                  # loaded lazily: only when a rerank is requested
            self._reranker = make_reranker(self.cfg)
        return self._reranker

    # -- the two first-stage retrievers --------------------------------------------------
    def dense(self, query: str, k: int | None = None) -> list[dict]:
        """Top-k chunks by cosine similarity to the embedded question."""
        q_emb = self.embedder.embed([query])[0]
        return self.index.search_dense(q_emb, k or self.cfg.dense_k)

    def sparse(self, query: str, k: int | None = None) -> list[dict]:
        """Top-k chunks by BM25 score (exact keyword matches: function names, config keys, error codes)."""
        return self.index.search_bm25(query, k or self.cfg.sparse_k)

    # -- full pipeline ---------------------------------------------------------------------
    def retrieve(self, query: str, mode: str = "hybrid", rerank: bool = True, top_n: int | None = None,
                 dense_weight: float | None = None, sparse_weight: float | None = None,
                 score_margin: float | None = None) -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        cfg = self.cfg
        top_n = top_n or cfg.rerank_top_n
        dw = cfg.dense_weight if dense_weight is None else dense_weight
        sw = cfg.sparse_weight if sparse_weight is None else sparse_weight

        dense_hits = self.dense(query) if mode in ("dense", "hybrid") else []
        sparse_hits = self.sparse(query) if mode in ("sparse", "hybrid") else []

        if mode == "hybrid":
            candidates = reciprocal_rank_fusion(dense_hits, sparse_hits, dw, sw, cfg.rrf_k)[:cfg.rerank_candidates]
        else:
            candidates = []
            for rank, h in enumerate(dense_hits or sparse_hits, 1):
                c = RetrievedChunk.from_hit(h)
                if mode == "dense":
                    c.dense_rank, c.dense_score = rank, h["score"]
                else:
                    c.sparse_rank, c.sparse_score = rank, h["score"]
                candidates.append(c)

        if rerank and candidates:
            scores = self.reranker.score(query, [c.text for c in candidates])
            for c, s in zip(candidates, scores):
                c.rerank_score = float(s)
            results = sorted(candidates, key=lambda c: -c.rerank_score)[:top_n]   # stable: ties keep fusion order
            margin = cfg.rerank_margin if score_margin is None else score_margin
            if margin is not None:      # relative cutoff: keep only results close to the best score
                results = [c for c in results if c.rerank_score >= results[0].rerank_score - margin]
        else:
            results = candidates[:top_n]
        return RetrievalResult(query, mode, candidates, results, bool(rerank and candidates),
                               (dw, sw) if mode == "hybrid" else (1.0, 0.0))
