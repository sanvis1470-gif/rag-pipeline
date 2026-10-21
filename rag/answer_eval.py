"""Evaluate the full answer path, including questions the documents cannot answer."""
from __future__ import annotations

from .answering import Answerer


def run_answer_eval(answerer: Answerer, questions: list[dict]) -> list[dict]:
    """questions: [{"question", "answerable": bool, "expected_keywords": [...]}]
    answerable -> correct if answered and the answer contains an expected keyword.
    not answerable -> correct if the system declined."""
    rows = []
    for q in questions:
        a = answerer.ask(q["question"])
        answerable = q.get("answerable", True)
        if answerable:
            kws = [k.lower() for k in q.get("expected_keywords", [])]
            correct = a.status == "answered" and (not kws or any(k in a.answer.lower() for k in kws))
        else:
            correct = a.status == "insufficient_evidence"
        weak = [{"status": c.status, "claim": c.claim, "cites": c.citations,
                 "reason": " | ".join(r for r in c.reasons.values() if r),
                 "evidence": " | ".join(e for e in c.evidence.values() if e),
                 "supported_by": c.supported_by}
                for c in a.claims if c.status != "verified"]
        rows.append({"question": q["question"], "answerable": answerable, "status": a.status, "correct": correct,
                     "confidence": a.confidence.composite, "retrieval": a.confidence.retrieval,
                     "coverage": a.confidence.coverage, "completeness": a.confidence.completeness,
                     "answer": a.answer, "weak": weak})
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def format_answer_eval(rows: list[dict], verbose: bool = False) -> str:
    w = 50
    lines = ["question".ljust(w) + "should".ljust(9) + "system".ljust(14) + "conf".rjust(6) + "  result",
             "-" * (w + 9 + 14 + 6 + 10)]
    for r in rows:
        lines.append(r["question"][:w - 2].ljust(w) + ("answer" if r["answerable"] else "decline").ljust(9)
                     + ("answered" if r["status"] == "answered" else "declined").ljust(14)
                     + f"{r['confidence']:.2f}".rjust(6) + ("  OK" if r["correct"] else "  WRONG"))
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    good_ans = [r for r in ans if r["correct"]]
    false_refusals = [r for r in ans if r["status"] != "answered"]
    declined = [r for r in una if r["correct"]]
    halluc = [r for r in una if r["status"] == "answered"]
    f = lambda v: "n/a" if v is None else f"{v:.2f}"
    lines += ["", "Summary:", "",
              f"answerable questions answered correctly : {len(good_ans)}/{len(ans)}",
              f"  false refusals (declined but answerable): {len(false_refusals)}",
              f"unanswerable questions correctly declined: {len(declined)}/{len(una)}",
              f"  answered anyway (hallucination risk)    : {len(halluc)}",
              f"mean confidence, answerable questions     : {f(_mean(r['confidence'] for r in ans))}",
              f"mean confidence, unanswerable questions   : {f(_mean(r['confidence'] for r in una))}",
              f"mean citation coverage on answered        : {f(_mean(r['coverage'] for r in rows if r['status'] == 'answered'))}",
              f"mean completeness on answered             : {f(_mean(r['completeness'] for r in rows if r['status'] == 'answered'))}"]
    if verbose:
        lines += ["", "Claims that were not verified (why coverage is below 1.00):", ""]
        shown = False
        for r in rows:
            if r["weak"]:
                shown = True
                lines.append(f"Q: {r['question']}")
                for w in r["weak"]:
                    sup = f"; supported by: {','.join(map(str, w['supported_by']))}" if w["supported_by"] else ""
                    lines.append(f"  [{w['status'].upper()}] {w['claim']}  (cites: {','.join(map(str, w['cites'])) or 'none'}{sup})")
                    if w["reason"]:
                        lines.append(f"      judge: {w['reason']}")
                    if w["evidence"]:
                        lines.append(f"      quoted: {w['evidence']}")
        if not shown:
            lines.append("  none: every claim was verified")
    return "\n".join(lines)
