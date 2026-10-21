"""Grounded answering: retrieve -> gate -> generate with [n] citations -> verify each citation -> score confidence.

If retrieval is weak (or the model says the context is insufficient) the system declines, and returns what it
found, what it could not find and which documents are worth checking, instead of a fabricated answer."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .config import Config
from .confidence import (Confidence, citation_coverage, completeness_score, composite_score, label_for,
                         retrieval_confidence, score_to_prob)
from .indexes import STOPWORDS
from .llm import LLM, make_llm, parse_json
from .prompts import GENERATION_SYSTEM, GENERATION_USER, JUDGE_SYSTEM, JUDGE_USER
from .retrieval import Retriever

CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_TRAILING_CITES = re.compile(r"([.!?])[ \t]+((?:\[\d+(?:\s*,\s*\d+)*\][ \t]*)+)")   # "Foo. [1] Bar" -> "Foo [1]. Bar"
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\"'])")
_NOT_A_CLAIM = re.compile(r"^(insufficient context|not found in the context|the context (does not|doesn't|did not)|"
                          r"there is no (information|mention)|i (do not|don't|cannot|can't))", re.I)
NO_ANSWER_TEXT = "I could not find enough information in the indexed documents to answer this question."


# ----------------------------------------------------------------------------------- data
@dataclass
class Source:
    n: int
    id: str
    source: str
    heading: str
    page: int
    relevance: float                # 0-1 (reranker score converted to a probability-like value)
    text: str


@dataclass
class Claim:
    text: str                       # sentence without citation markers
    citations: list[int]
    raw: str                        # sentence as written (with markers)


@dataclass
class ClaimCheck:
    claim: str
    citations: list[int]
    verdicts: dict[int, str]        # n -> supported | partially_supported | not_supported | invalid | unknown
    reasons: dict[int, str]
    status: str                     # verified | partial | miscited | unsupported | uncited | unchecked
    raw: str = ""
    evidence: dict[int, str] = field(default_factory=dict)      # n -> sentence the judge quoted from the block
    supported_by: list[int] = field(default_factory=list)       # blocks that DO support a claim whose cited blocks don't


@dataclass
class RagAnswer:
    question: str
    status: str                     # answered | insufficient_evidence
    answer: str
    flagged_answer: str             # answer with [n?] on unsupported and [n~] on partially supported citations
    claims: list[ClaimCheck]
    sources: list[Source]
    confidence: Confidence
    found: list[str] = field(default_factory=list)
    not_found: str = ""
    suggested_documents: list[dict] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)       # "Not found in the context: ..." statements
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ prompt + citation parsing
def build_context(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        where = s.source + (f", page {s.page}" if s.page and s.page > 0 else "") + (f", section: {s.heading}" if s.heading else "")
        blocks.append(f"[{s.n}] ({where})\n{s.text}")
    return "\n\n".join(blocks)


def _lines_of_sentences(answer: str) -> list[list[str]]:
    answer = _TRAILING_CITES.sub(
        lambda m: " " + m.group(2).rstrip() + m.group(1) + m.group(2)[len(m.group(2).rstrip()):], answer)
    out: list[list[str]] = []
    for line in answer.splitlines():
        line = _LIST_MARKER.sub("", line).strip()
        if line:
            out.append([s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip()])
    return out


def parse_claims(answer: str) -> tuple[list[Claim], list[str]]:
    """Split the answer into sentences. Sentences that make a statement become claims (with their [n] citations);
    'not found' style sentences are returned separately as missing-information statements."""
    claims: list[Claim] = []
    missing: list[str] = []
    for line in _lines_of_sentences(answer):
        for sent in line:
            cites: list[int] = []
            for group in CITE_RE.findall(sent):
                for n in re.split(r"\s*,\s*", group):
                    if int(n) not in cites:
                        cites.append(int(n))
            clean = re.sub(r"\s+([.,;:!?])", r"\1", CITE_RE.sub("", sent))
            clean = re.sub(r"\s{2,}", " ", clean).strip()
            if not re.search(r"\w", clean):
                continue                                    # nothing left but a stray citation such as "[3]"
            if clean.endswith(":") and not cites:
                continue                                    # lead-in to a list ("The resources are:"), not a claim
            if _NOT_A_CLAIM.match(clean):
                missing.append(clean)
            else:
                claims.append(Claim(clean, cites, sent))
    return claims, missing


def is_insufficient(answer: str) -> bool:
    return answer.strip().upper().startswith("INSUFFICIENT CONTEXT")


# ------------------------------------------------------------------------- verification
_WORD = re.compile(r"\w+")
_NEG_WORDS = {"not", "no", "never", "cannot", "without", "neither", "nor", "none"}


def _neg_tokens(text: str) -> tuple[list[str], list[int]]:
    """Lower-cased word tokens (n't -> not) and the sentence number each token belongs to."""
    toks: list[str] = []
    sids: list[int] = []
    for sid, sent in enumerate(re.split(r"(?<=[.!?])\s+|\n", text)):
        sent = re.sub(r"['\u2019]s\b", "", sent.lower())                  # "ByteByteGo's" -> "bytebytego"
        for t in _WORD.findall(re.sub(r"n't\b", " not", sent)):
            toks.append(t)
            sids.append(sid)
    return toks, sids


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower().strip("\"'")


def _quote_in_source(evidence: str, source: str) -> bool:
    """The judge must quote the source. Exact match, or (for small copying slips) 90% of its words are in the source."""
    ev = _norm(evidence)
    if not ev:
        return False
    if ev in _norm(source):
        return True
    ev_tokens, src_tokens = _WORD.findall(ev), set(_WORD.findall(_norm(source)))
    return len(ev_tokens) >= 3 and sum(t in src_tokens for t in ev_tokens) / len(ev_tokens) >= 0.9


def _claim_matches_evidence(claim: str, evidence: str, question: str = "") -> bool:
    """Deterministic double-check of the judge. Nearly all content words of the claim must appear in the quoted
    evidence, every number must, and negation must agree around each shared word ('could not find' must not
    match 'can find'; an unrelated 'no' elsewhere in a multi-sentence quote does not count). Words that merely
    echo the question ("status code" in "what does a 404 status code mean") add no new fact, so they are not
    required to appear in the quote."""
    asked = set(_WORD.findall(question.lower()))
    stems = {w[:5] for w in asked if len(w) >= 5}
    echoes = lambda w: not w.isdigit() and (w in asked or (len(w) >= 5 and w[:5] in stems))
    (ctoks, csid), (etoks, esid) = _neg_tokens(claim), _neg_tokens(evidence)
    words = {w for w in ctoks if len(w) > 1 and w not in STOPWORDS and w not in _NEG_WORDS and not echoes(w)}
    if len(words) < 2:
        return False

    def find(w: str) -> int | None:                           # position in the evidence, allowing 5-letter stems
        for i, e in enumerate(etoks):
            if e == w or (len(w) >= 5 and (e.startswith(w[:5]) or w.startswith(e[:5]))):
                return i
        return None

    pos = {w: find(w) for w in words}
    if sum(p is not None for p in pos.values()) / len(words) < 0.85:
        return False
    if any(w.isdigit() and pos[w] is None for w in words):
        return False
    def negated(toks: list[str], sids: list[int], i: int) -> bool:    # negation within 3 words before, same sentence
        return any(toks[j] in _NEG_WORDS for j in range(max(0, i - 3), i) if sids[j] == sids[i])

    for w, p in pos.items():
        if p is not None and negated(ctoks, csid, ctoks.index(w)) != negated(etoks, esid, p):
            return False
    return True


def _assess(data: dict | None, raw: str, claim: str, source: str, question: str = "") -> tuple[str, str, str]:
    """(verdict, reason, evidence). The judge's verdict is cross-checked against the text itself:
    'supported' needs a quote that really is in the block, and a quote that really matches the claim overrules
    a 'not supported' verdict. This protects against small-model judge errors in both directions."""
    verdict = str((data or {}).get("verdict", "")).lower().replace(" ", "_").replace("-", "_")
    reason = str((data or {}).get("reason", "")).strip()
    has_evidence_field = bool(data) and "evidence" in data
    evidence = str((data or {}).get("evidence", "") or "").strip()
    if not verdict:                                          # judge ignored the JSON format: look for the words
        low = raw.lower().replace(" ", "_")
        verdict = next((v for v in ("not_supported", "partially_supported", "supported") if v in low), "")
    verdict = ("not_supported" if verdict.startswith("not") else "partially_supported" if verdict.startswith("partial")
               else "supported" if verdict.startswith("support") else "unknown")
    if verdict == "unknown":
        return verdict, reason or "the judge returned no usable verdict", evidence
    if has_evidence_field:
        quote_ok = _quote_in_source(evidence, source)
        if verdict == "supported" and not quote_ok:
            return "partially_supported", "the judge said supported but its quote is not in the source block", evidence
        if verdict != "supported" and quote_ok and _claim_matches_evidence(claim, evidence, question):
            return "supported", "the quoted source sentence matches the claim (judge verdict overridden)", evidence
    return verdict, reason, evidence


def verify_claims(llm: LLM, claims: list[Claim], sources: dict[int, Source], question: str = "",
                  rescue: bool = True) -> list[ClaimCheck]:
    """LLM-as-judge on every (claim, cited block) pair. If none of the cited blocks supports a claim, the other
    blocks are tried too: small models often cite the wrong number, and 'supported, but by [1] not [2]' is
    more useful than 'unsupported'."""
    cache: dict[tuple[str, int], tuple[str, str, str]] = {}
    checks: list[ClaimCheck] = []
    for c in claims:
        verdicts: dict[int, str] = {}
        reasons: dict[int, str] = {}
        evidence: dict[int, str] = {}
        for n in c.citations:
            if n not in sources:
                verdicts[n], reasons[n] = "invalid", "this number does not match any context block"
                continue
            if (c.text, n) not in cache:
                raw = llm.complete(JUDGE_SYSTEM, JUDGE_USER.format(claim=c.text, source=sources[n].text), json_mode=True)
                cache[(c.text, n)] = _assess(parse_json(raw), raw, c.text, sources[n].text, question)
            verdicts[n], reasons[n], evidence[n] = cache[(c.text, n)]
        vals = set(verdicts.values())
        if not c.citations:
            status = "uncited"
        elif "supported" in vals:
            status = "verified"
        elif "partially_supported" in vals:
            status = "partial"
        elif "unknown" in vals and not vals - {"unknown", "invalid"}:
            status = "unchecked"
        else:
            status = "unsupported"
        supported_by: list[int] = []
        if rescue and status == "unsupported":
            partly: list[int] = []
            for n in sources:
                if n in c.citations:
                    continue
                if (c.text, n) not in cache:
                    raw = llm.complete(JUDGE_SYSTEM, JUDGE_USER.format(claim=c.text, source=sources[n].text), json_mode=True)
                    cache[(c.text, n)] = _assess(parse_json(raw), raw, c.text, sources[n].text, question)
                verdict = cache[(c.text, n)][0]
                if verdict == "supported":
                    supported_by.append(n)
                elif verdict == "partially_supported":
                    partly.append(n)
            if supported_by:
                status = "miscited"             # a different block fully supports it
            elif partly:
                status, supported_by = "partial", partly     # a different block supports part of it
        checks.append(ClaimCheck(c.text, c.citations, verdicts, reasons, status, c.raw, evidence, supported_by))
    return checks


def flag_answer(answer: str, checks: list[ClaimCheck]) -> str:
    """Rewrite the answer marking weak citations: [n?] = not supported / invalid, [n~] = partially supported."""
    by_raw = {c.raw: c for c in checks}

    def mark(sent: str) -> str:
        chk = by_raw.get(sent)
        if not chk:
            return sent

        def repl(m: re.Match) -> str:
            nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
            tagged = []
            for n in nums:
                v = chk.verdicts.get(n)
                tagged.append(f"{n}~" if v == "partially_supported" else f"{n}?" if v in ("not_supported", "invalid") else str(n))
            return "[" + ", ".join(tagged) + "]"
        return CITE_RE.sub(repl, sent)

    return "\n".join(" ".join(mark(s) for s in line) for line in _lines_of_sentences(answer))


# --------------------------------------------------------------------------- the answerer
class Answerer:
    def __init__(self, retriever: Retriever, llm: LLM | None, cfg: Config):
        self.retriever, self._llm, self.cfg = retriever, llm, cfg

    @property
    def llm(self) -> LLM:
        if self._llm is None:                               # created lazily so non-LLM commands never need one
            self._llm = make_llm(self.cfg)
        return self._llm

    def ask(self, question: str, verify: bool | None = None, check_completeness: bool | None = None,
            min_confidence: float | None = None) -> RagAnswer:
        cfg = self.cfg
        verify = cfg.verify_citations if verify is None else verify
        check_completeness = cfg.check_completeness if check_completeness is None else check_completeness
        threshold = cfg.min_retrieval_confidence if min_confidence is None else min_confidence

        res = self.retriever.retrieve(question, mode="hybrid", rerank=True, top_n=cfg.rerank_top_n,
                                      score_margin=cfg.answer_score_margin)
        sources = [Source(i, c.id, c.source, c.heading, c.page, score_to_prob(c.rerank_score, res.score_kind), c.text)
                   for i, c in enumerate(res.results, 1)]
        ret_conf = retrieval_confidence([c.rerank_score for c in res.results], res.score_kind)

        if not sources or ret_conf < threshold:             # gate: do not even call the LLM
            return self._decline(question, res, sources, ret_conf,
                                 note=f"Retrieval confidence {ret_conf:.2f} is below the threshold {threshold:.2f}, "
                                      "so no answer was generated.")

        raw_answer = self.llm.complete(GENERATION_SYSTEM, GENERATION_USER.format(
            context=build_context(sources), question=question)).strip()

        if is_insufficient(raw_answer):
            statement = re.sub(r"(?i)^insufficient context:?\s*", "", raw_answer).strip()
            return self._decline(question, res, sources, ret_conf, model_statement=statement,
                                 note="The model judged the retrieved context insufficient.")

        claims, missing = parse_claims(raw_answer)
        by_n = {s.n: s for s in sources}
        if verify:
            checks = verify_claims(self.llm, claims, by_n, question, cfg.rescue_miscited)
        else:
            checks = [ClaimCheck(c.text, c.citations, {}, {}, "unchecked" if c.citations else "uncited", c.raw)
                      for c in claims]
        coverage = citation_coverage([c.status for c in checks]) if verify else None

        completeness, missing_parts = (None, [])
        if check_completeness:
            completeness, missing_parts = completeness_score(self.llm, question, CITE_RE.sub("", raw_answer))

        composite = composite_score(ret_conf, coverage, completeness, cfg.confidence_weights)
        conf = Confidence(ret_conf, coverage, completeness, composite, label_for(composite), missing_parts)

        notes = []
        flagged = [c for c in checks if c.status in ("unsupported", "partial", "miscited", "uncited", "unchecked")]
        if verify and flagged:
            notes.append(f"{len(flagged)} of {len(checks)} claims have weak, missing or unverified citations; "
                         "see [n?] (unsupported or wrong block) and [n~] (partial) in the flagged answer.")
        if not claims:
            notes.append("No citable claims were found in the model's answer.")
        return RagAnswer(question, "answered", raw_answer, flag_answer(raw_answer, checks) if verify else raw_answer,
                         checks, sources, conf, missing_info=missing, notes=notes)

    # ---------------------------------------------------------------- decline gracefully
    def _decline(self, question, res, sources, ret_conf, note: str, model_statement: str | None = None) -> RagAnswer:
        def where(s: Source) -> str:
            return f"{s.source}" + (f" p{s.page}" if s.page and s.page > 0 else "") + (f" [{s.heading}]" if s.heading else "")

        found = [f"{where(s)} (relevance {s.relevance:.2f}): {re.sub(chr(10), ' ', s.text)[:140].strip()}..."
                 for s in sources[:3]]
        not_found = model_statement or f'None of the retrieved passages directly answers: "{question}"'
        docs: dict[str, dict] = {}
        for c in res.candidates:                             # all reranked candidates, best first
            rel = score_to_prob(c.rerank_score, res.score_kind) if c.rerank_score is not None else 0.0
            d = docs.setdefault(c.source, {"source": c.source, "sections": [], "best_relevance": round(rel, 3)})
            if c.heading and c.heading not in d["sections"] and len(d["sections"]) < 3:
                d["sections"].append(c.heading)
            d["best_relevance"] = max(d["best_relevance"], round(rel, 3))
        suggested = sorted(docs.values(), key=lambda d: -d["best_relevance"])[:3]
        conf = Confidence(ret_conf, None, None, ret_conf, "low")
        if not sources:
            found, not_found = [], "The index returned no passages. Is it empty? Run `python -m rag stats`."
        return RagAnswer(question, "insufficient_evidence", NO_ANSWER_TEXT, NO_ANSWER_TEXT, [], sources, conf,
                         found, not_found, suggested, notes=[note])


# ---------------------------------------------------------------------------- display
_TAG = {"verified": "OK", "partial": "PARTIAL", "miscited": "WRONG BLOCK", "unsupported": "UNSUPPORTED",
        "uncited": "NO CITATION", "unchecked": "UNCHECKED"}


def format_answer(a: RagAnswer, show_context: bool = False) -> str:
    c = a.confidence
    out = [f"Question: {a.question}", ""]
    if a.status == "answered":
        out += [f"Status: ANSWERED | confidence {c.composite:.2f} ({c.label})", "", "Answer:", a.flagged_answer, ""]
        out.append("Citation check:")
        for ck in a.claims:
            cites = ",".join(map(str, ck.citations)) or "none"
            extra = f"; supported by: {','.join(map(str, ck.supported_by))}" if ck.supported_by else ""
            out.append(f"  [{_TAG[ck.status]}] {ck.claim}  (cites: {cites}{extra})")
            if ck.status in ("unsupported", "partial"):
                for n, v in ck.verdicts.items():
                    if v != "supported" and ck.reasons.get(n):
                        out.append(f"        [{n}] {v}: {ck.reasons[n]}")
        if a.missing_info:
            out += ["", "Missing from the documents: " + " ".join(a.missing_info)]
    else:
        out += [f"Status: INSUFFICIENT EVIDENCE | retrieval confidence {c.retrieval:.2f}", "", a.answer, "",
                "What was found:"]
        out += [f"  - {f}" for f in a.found] or ["  - nothing relevant"]
        out += ["", "What was not found:", f"  {a.not_found}", "", "Documents worth checking manually:"]
        out += [f"  - {d['source']}" + (f" (sections: {'; '.join(d['sections'])})" if d["sections"] else "")
                for d in a.suggested_documents] or ["  - none"]
    cov = "n/a" if c.coverage is None else f"{c.coverage:.2f}"
    comp = "n/a" if c.completeness is None else f"{c.completeness:.2f}"
    out += ["", f"Confidence: retrieval {c.retrieval:.2f} | citation coverage {cov} | completeness {comp} | composite {c.composite:.2f}"]
    if c.missing_parts:
        out.append("  parts of the question not addressed: " + "; ".join(c.missing_parts))
    out += [f"Note: {n}" for n in a.notes]
    if a.status == "answered" or show_context:
        out += ["", "Sources:"]
        for s in a.sources:
            out.append(f"  [{s.n}] {s.source}" + (f" p{s.page}" if s.page and s.page > 0 else "")
                       + (f" [{s.heading}]" if s.heading else "") + f" (relevance {s.relevance:.2f})")
    return "\n".join(out)
