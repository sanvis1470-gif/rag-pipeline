"""Measure each retrieval stage on labelled questions: dense, sparse, hybrid (RRF), hybrid + rerank."""
from __future__ import annotations

import re

from .retrieval import Retriever

STAGES = [
    ("dense", dict(mode="dense", rerank=False)),
    ("sparse", dict(mode="sparse", rerank=False)),
    ("hybrid", dict(mode="hybrid", rerank=False)),
    ("hybrid+rerank", dict(mode="hybrid", rerank=True)),
]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).lower()


def answer_rank(chunks, needle: str) -> int | None:
    """1-based rank of the first chunk whose text contains the whole answer phrase."""
    n = _norm(needle)
    for i, c in enumerate(chunks, 1):
        if n in _norm(c.text):
            return i
    return None


def run_eval(retriever: Retriever, questions: list[dict], top_n: int = 5, stages=STAGES) -> dict[str, list]:
    out: dict[str, list] = {label: [] for label, _ in stages}
    for q in questions:
        for label, kw in stages:
            res = retriever.retrieve(q["question"], top_n=top_n, **kw)
            out[label].append(answer_rank(res.results, q["answer_contains"]))
    return out


def sweep_weights(retriever: Retriever, questions: list[dict], top_n: int = 5,
                  dense_weights=(0.0, 0.3, 0.5, 0.7, 1.0)) -> dict[float, list]:
    out: dict[float, list] = {}
    for w in dense_weights:
        out[w] = [answer_rank(retriever.retrieve(q["question"], mode="hybrid", rerank=False, top_n=top_n,
                                                 dense_weight=w, sparse_weight=1 - w).results,
                              q["answer_contains"]) for q in questions]
    return out


def _metrics(ranks: list, top_n: int) -> tuple[int, int, int, float]:
    n = len(ranks)
    return (sum(r == 1 for r in ranks),
            sum(r is not None and r <= 3 for r in ranks),
            sum(r is not None for r in ranks),
            sum(1 / r for r in ranks if r) / n)


def _c(r) -> str:
    return "-" if r is None else str(r)


def format_eval(results: dict[str, list], questions: list[dict], top_n: int) -> str:
    labels = list(results)
    n = len(questions)
    w = 46
    lines = [f"Rank of the chunk containing the answer ('-' = not in top {top_n}):\n",
             "question".ljust(w) + "type".ljust(10) + "".join(l[:13].center(15) for l in labels),
             "-" * (w + 10 + 15 * len(labels))]
    for i, q in enumerate(questions):
        lines.append(q["question"][:w - 2].ljust(w) + q.get("type", "")[:9].ljust(10)
                     + "".join(_c(results[l][i]).center(15) for l in labels))
    lines += ["", "Summary:", "",
              "stage".ljust(15) + f"hit@1".rjust(7) + "hit@3".rjust(7) + f"hit@{top_n}".rjust(7) + "MRR".rjust(7)]
    for l in labels:
        h1, h3, hn, mrr = _metrics(results[l], top_n)
        lines.append(l.ljust(15) + f"{h1}/{n}".rjust(7) + f"{h3}/{n}".rjust(7) + f"{hn}/{n}".rjust(7) + f"{mrr:.2f}".rjust(7))
    types = sorted({q.get("type", "") for q in questions if q.get("type")})
    if types:
        lines += ["", "MRR by question type:", "", "type".ljust(15) + "".join(l[:13].rjust(15) for l in labels)]
        for t in types:
            idx = [i for i, q in enumerate(questions) if q.get("type") == t]
            row = "".join(f"{sum(1 / results[l][i] for i in idx if results[l][i]) / len(idx):.2f}".rjust(15)
                          for l in labels)
            lines.append(f"{t} ({len(idx)})".ljust(15) + row)
    return "\n".join(lines)


def format_sweep(sweep: dict[float, list], n: int, top_n: int) -> str:
    lines = ["Dense weight sweep (hybrid RRF, no reranker; sparse weight = 1 - dense):", "",
             "dense/sparse".ljust(15) + "hit@1".rjust(7) + f"hit@{top_n}".rjust(7) + "MRR".rjust(7)]
    for w, ranks in sweep.items():
        h1, _, hn, mrr = _metrics(ranks, top_n)
        lines.append(f"{w:.1f} / {1 - w:.1f}".ljust(15) + f"{h1}/{n}".rjust(7) + f"{hn}/{n}".rjust(7) + f"{mrr:.2f}".rjust(7))
    return "\n".join(lines)
