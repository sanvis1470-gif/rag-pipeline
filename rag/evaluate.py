"""Measure each retrieval stage on labelled questions: dense, sparse, hybrid (RRF), hybrid + rerank."""
from __future__ import annotations

import re

from .retrieval import Retriever

STAGES = [
    ("dense", dict(mode="dense", rerank=False)),
    ("sparse", dict(mode="sparse", rerank=False)),
    ("hybrid", dict(mode="hybrid", rerank=False)),
    ("hybrid+rerank", dict(mode="hybrid", rerank=True)),
    ("rerank+cutoff", dict(mode="hybrid", rerank=True)),     # score_margin is filled in by run_eval
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


def run_eval(retriever: Retriever, questions: list[dict], top_n: int = 5, stages=STAGES,
             margin: float | None = 5.0) -> dict[str, list]:
    """label -> [(rank of answer chunk or None, number of chunks returned)] per question."""
    out: dict[str, list] = {label: [] for label, _ in stages}
    for q in questions:
        for label, kw in stages:
            kw = dict(kw, score_margin=margin) if label == "rerank+cutoff" else kw
            res = retriever.retrieve(q["question"], top_n=top_n, **kw)
            out[label].append((answer_rank(res.results, q["answer_contains"]), len(res.results)))
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
    ranks = {l: [r for r, _ in results[l]] for l in labels}
    counts = {l: [n for _, n in results[l]] for l in labels}
    n = len(questions)
    w, cw = 38, 14
    lines = [f"Rank of the chunk containing the answer ('-' = not returned):\n",
             "question".ljust(w) + "type".ljust(9) + "".join(l[:cw - 1].center(cw) for l in labels),
             "-" * (w + 9 + cw * len(labels))]
    for i, q in enumerate(questions):
        lines.append(q["question"][:w - 2].ljust(w) + q.get("type", "")[:8].ljust(9)
                     + "".join(_c(ranks[l][i]).center(cw) for l in labels))
    lines += ["", "Summary:", "",
              "stage".ljust(15) + "hit@1".rjust(7) + "hit@3".rjust(7) + "answer kept".rjust(13) + "MRR".rjust(7)
              + "avg returned".rjust(14)]
    for l in labels:
        h1, h3, hn, mrr = _metrics(ranks[l], top_n)
        lines.append(l.ljust(15) + f"{h1}/{n}".rjust(7) + f"{h3}/{n}".rjust(7) + f"{hn}/{n}".rjust(13)
                     + f"{mrr:.2f}".rjust(7) + f"{sum(counts[l]) / n:.1f}".rjust(14))
    types = sorted({q.get("type", "") for q in questions if q.get("type")})
    if types:
        lines += ["", "MRR by question type:", "", "type".ljust(15) + "".join(l[:cw - 1].rjust(cw) for l in labels)]
        for t in types:
            idx = [i for i, q in enumerate(questions) if q.get("type") == t]
            row = "".join(f"{sum(1 / ranks[l][i] for i in idx if ranks[l][i]) / len(idx):.2f}".rjust(cw)
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
