"""Near-duplicate detection: cosine similarity > threshold vs the index AND vs earlier chunks in the same batch."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .indexes import ChunkIndex


@dataclass
class DupInfo:
    duplicate_of: str
    similarity: float
    scope: str          # "index" (already stored) | "batch" (earlier chunk of this ingest)


def find_duplicates(ids: list[str], embs: np.ndarray, index: ChunkIndex, threshold: float) -> list[DupInfo | None]:
    existing = index.nearest(embs)
    result: list[DupInfo | None] = [None] * len(ids)
    kept: list[int] = []
    for i in range(len(ids)):
        ex = existing[i]
        if ex is not None and ex[1] > threshold:
            result[i] = DupInfo(ex[0], ex[1], "index")
            continue
        if kept:
            sims = embs[kept] @ embs[i]
            j = int(np.argmax(sims))
            if sims[j] > threshold:
                result[i] = DupInfo(ids[kept[j]], float(sims[j]), "batch")
                continue
        kept.append(i)
    return result
