"""Run the same questions against every chunking strategy and report where the right chunk ranks."""
from __future__ import annotations

import re

from .pipeline import Pipeline


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).lower()


def _rank(hits: list[dict], needle: str) -> tuple[int | None, int | None]:
    """(1-based rank, chunk length) of the first hit whose text contains `needle`."""
    n = _norm(needle)
    for i, h in enumerate(hits, 1):
        if n in _norm(h["text"]):
            return i, len(h["text"])
    return None, None


def run_comparison(pipe: Pipeline, questions: list[dict], strategies: list[str], k: int = 5) -> dict:
    """questions: [{"question": ..., "answer_contains": ...}]. A chunk counts as correct only if it
    contains the whole answer text, so an answer split across two chunks is a miss."""
    q_embs = pipe.embedder.embed([q["question"] for q in questions])
    out: dict[str, dict] = {}
    for strat in strategies:
        pipe.reindex(strat)
        rows = []
        for q, emb in zip(questions, q_embs):
            d_rank, d_len = _rank(pipe.index.search_dense(emb, k), q["answer_contains"])
            b_rank, b_len = _rank(pipe.index.search_bm25(q["question"], k), q["answer_contains"])
            rows.append({"dense": d_rank, "bm25": b_rank, "dense_len": d_len, "bm25_len": b_len})
        out[strat] = {"chunks": pipe.index.count(), "rows": rows}
    return out


def _cell(r: int | None) -> str:
    return "-" if r is None else str(r)


def format_report(results: dict, questions: list[dict], k: int) -> str:
    strategies = list(results)
    lines: list[str] = []
    w = 44
    lines.append(f"Rank of the chunk containing the answer (dense/bm25, '-' = not in top {k}):\n")
    lines.append("question".ljust(w) + "".join(s.center(14) for s in strategies))
    lines.append("-" * (w + 14 * len(strategies)))
    for i, q in enumerate(questions):
        cells = "".join(f"{_cell(results[s]['rows'][i]['dense'])}/{_cell(results[s]['rows'][i]['bm25'])}".center(14)
                        for s in strategies)
        lines.append(q["question"][:w - 2].ljust(w) + cells)

    lines.append("\nSummary:\n")
    lines.append("strategy".ljust(11) + "chunks".rjust(7) + "  method  hit@1  hit@3   MRR  avg size of correct chunk")
    n = len(questions)
    for s in strategies:
        rows = results[s]["rows"]
        for method in ("dense", "bm25"):
            ranks = [r[method] for r in rows]
            lens = [r[f"{method}_len"] for r in rows if r[f"{method}_len"]]
            hit1 = sum(1 for r in ranks if r == 1)
            hit3 = sum(1 for r in ranks if r is not None and r <= 3)
            mrr = sum(1 / r for r in ranks if r) / n
            avg = f"{sum(lens) / len(lens):.0f} chars" if lens else "-"
            first = method == "dense"
            lines.append((s if first else "").ljust(11) + (str(results[s]["chunks"]) if first else "").rjust(7)
                         + f"  {method:6s}  {hit1}/{n}   {hit3}/{n}   {mrr:.2f}  {avg}")
    return "\n".join(lines)
