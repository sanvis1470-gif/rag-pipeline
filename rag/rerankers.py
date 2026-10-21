"""Second-pass relevance scoring. A cross-encoder reads (question, chunk) TOGETHER, so it can judge
relevance far better than the embedding comparison, which encodes each side separately."""
from __future__ import annotations

import re
from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text (higher = more relevant)."""


class CrossEncoderReranker:
    """Small local cross-encoder (default: ms-marco MiniLM, ~90 MB). No API key, runs on CPU."""
    score_kind = "logit"        # raw scores, roughly -11 (irrelevant) to +5 (answers the question)

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 16):
        from sentence_transformers import CrossEncoder      # downloads the model on first use
        self.model = CrossEncoder(model_name, max_length=512)
        self.batch_size = batch_size

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        preds = self.model.predict([(query, t) for t in texts],
                                   batch_size=self.batch_size, show_progress_bar=False)
        return [float(x) for x in preds]


class LexicalOverlapReranker:
    """Offline stand-in for tests: fraction of query words that appear in the chunk."""
    score_kind = "unit"         # already between 0 and 1

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(re.findall(r"\w+", query.lower()))
        return [len(q & set(re.findall(r"\w+", t.lower()))) / max(len(q), 1) for t in texts]


def make_reranker(cfg) -> Reranker:
    return CrossEncoderReranker(cfg.reranker_model)
