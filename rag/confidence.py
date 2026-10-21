"""Answer confidence: retrieval relevance, citation coverage, answer completeness -> one composite score."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .llm import LLM, parse_json
from .prompts import COMPLETENESS_SYSTEM, COMPLETENESS_USER


@dataclass
class Confidence:
    retrieval: float                    # 0-1: how relevant the retrieved chunks were
    coverage: float | None              # 0-1: share of claims with verified citations (None = not checked)
    completeness: float | None          # 0-1: share of question parts addressed (None = not checked)
    composite: float
    label: str                          # high | medium | low
    missing_parts: list[str] = field(default_factory=list)


def score_to_prob(score: float, kind: str | None) -> float:
    """Cross-encoder logits -> 0..1 via sigmoid; scores that are already 0..1 are clipped."""
    if kind == "logit" or kind is None:
        return 1.0 / (1.0 + math.exp(-max(min(score, 50.0), -50.0)))
    return min(max(score, 0.0), 1.0)


def retrieval_confidence(rerank_scores: list[float], kind: str | None) -> float:
    """0.6 x best chunk + 0.4 x average of the chunks used. The best chunk dominates (one exact match is
    enough for a good answer) but a context padded with weak chunks is penalized."""
    probs = [score_to_prob(s, kind) for s in rerank_scores]
    if not probs:
        return 0.0
    return 0.6 * max(probs) + 0.4 * (sum(probs) / len(probs))


def citation_coverage(statuses: list[str]) -> float | None:
    """verified claim = 1; partially supported or cited the wrong block (but another block supports it) = 0.5;
    everything else (unsupported, uncited, unchecked) = 0."""
    if not statuses:
        return None
    points = {"verified": 1.0, "partial": 0.5, "miscited": 0.5}
    return sum(points.get(s, 0.0) for s in statuses) / len(statuses)


def composite_score(retrieval: float, coverage: float | None, completeness: float | None,
                    weights: tuple[float, float, float]) -> float:
    """Weighted mean over the components that were actually measured (weights are renormalized)."""
    parts = [(retrieval, weights[0]), (coverage, weights[1]), (completeness, weights[2])]
    used = [(v, w) for v, w in parts if v is not None and w > 0]
    total = sum(w for _, w in used)
    return sum(v * w for v, w in used) / total if total else retrieval


def label_for(score: float) -> str:
    return "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"


_MULTI_PART = re.compile(r"\b(and|also|as well as|plus|both)\b|[,;]|\?.*\?", re.I)


def is_single_part(question: str) -> bool:
    """A question with no conjunction, comma or second question mark asks for one thing."""
    return not _MULTI_PART.search(question)


def completeness_score(llm: LLM, question: str, answer: str) -> tuple[float | None, list[str]]:
    """LLM-as-judge: split the question into parts and check each is addressed. (None, []) if the judge fails."""
    data = parse_json(llm.complete(COMPLETENESS_SYSTEM,
                                   COMPLETENESS_USER.format(question=question, answer=answer), json_mode=True))
    parts = data.get("parts") if data else None
    if not isinstance(parts, list):
        return None, []
    # small models sometimes list citation markers like "[1]" as parts of the question: drop anything without words
    parts = [p for p in parts if isinstance(p, dict) and re.search(r"[A-Za-z]{3,}", str(p.get("part", "")))]
    if not parts:
        return None, []
    addressed = [bool(p.get("addressed")) for p in parts]
    if is_single_part(question):        # small models split one question into several; collapse to one verdict
        return (1.0, []) if any(addressed) else (0.0, [question])
    missing = [str(p["part"]) for p in parts if not p.get("addressed")]
    return sum(addressed) / len(addressed), missing
