from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalized vectors."""


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.maximum(norms, 1e-12)).astype(np.float32)


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 128):
        from openai import OpenAI   # reads OPENAI_API_KEY; client retries 429/5xx itself
        self.client = OpenAI(max_retries=5)
        self.model, self.batch_size = model, batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.model, input=texts[i:i + self.batch_size])
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return _normalize(np.array(out, dtype=np.float32))


class LocalEmbedder:
    """sentence-transformers model running on your own machine: no API key, no rate limits."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer   # downloads the model on first use
        self.model = SentenceTransformer(model)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        v = self.model.encode(texts, batch_size=self.batch_size,
                              normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


class HashEmbedder:
    """Offline bag-of-words embedder for tests / dry runs. Similar words -> similar vectors."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                m[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return _normalize(m)


def make_embedder(cfg) -> Embedder:
    if cfg.embedding_provider == "openai":
        return OpenAIEmbedder(cfg.embedding_model, cfg.embed_batch_size)
    return LocalEmbedder(cfg.embedding_model)
