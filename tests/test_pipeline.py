import json

import pytest

from rag.chunkers import recursive_split
from rag.config import Config
from rag.embeddings import HashEmbedder
from rag.loaders import parse_bytes
from rag.pipeline import Pipeline

MD = """# Guide

Intro paragraph about the guide. It explains the layout.

## Cats

Cats are small carnivorous mammals. Cats purr when happy. Cats sleep most of the day.
Kittens play with yarn and chase mice around the house.

## Databases

A database stores structured records. Indexes speed up queries. Transactions keep data consistent.
Relational databases use SQL to join tables together.
"""

HTML = """<html><head><title>Doc</title><script>var x=1;</script></head><body>
<nav>menu</nav><h1>Top</h1><p>Hello world.</p><h2>Sub</h2><p>Details here.</p><footer>f</footer></body></html>"""


@pytest.fixture
def pipe(tmp_path):
    cfg = Config(data_dir=tmp_path / "data", chunk_size=200, chunk_overlap=40,
                 semantic_min_chars=40, semantic_max_chars=400)
    return Pipeline(cfg, embedder=HashEmbedder())


def write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content) if isinstance(content, str) else p.write_bytes(content)
    return p


def test_markdown_headings_and_html_cleanup():
    md = parse_bytes(MD.encode(), "a.md")
    assert [s.heading for s in md.segments] == ["Guide", "Guide > Cats", "Guide > Databases"]
    html = parse_bytes(HTML.encode(), "a.html")
    assert "menu" not in html.text and "var x" not in html.text
    assert html.segments[-1].heading == "Top > Sub"


def test_pdf_pages(tmp_path):
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    p = tmp_path / "x.pdf"
    c = reportlab.Canvas(str(p))
    c.drawString(72, 700, "First page text about apples.")
    c.showPage()
    c.drawString(72, 700, "Second page text about oranges.")
    c.save()
    doc = parse_bytes(p.read_bytes(), "x.pdf")
    assert [s.page for s in doc.segments] == [1, 2]


def test_recursive_split_respects_size():
    text = ("Sentence number one is here. " * 40).strip()
    parts = recursive_split(text, 100, 20)
    assert all(len(p) <= 100 for p in parts) and len(parts) > 5


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic"])
def test_each_strategy_indexes_with_metadata_and_stays_in_sync(pipe, tmp_path, strategy):
    r = pipe.ingest_file(write(tmp_path, "a.md", MD), strategy)
    assert r.n_inserted > 0
    got = pipe.index.col.get(include=["metadatas", "documents"])
    for m, d in zip(got["metadatas"], got["documents"]):
        assert m["strategy"] == strategy and m["char_count"] == len(d) and m["source"] == "a.md"
    assert pipe.index.is_in_sync()


def test_dedup_flags_near_duplicate_documents(pipe, tmp_path):
    pipe.ingest_file(write(tmp_path, "a.md", MD), "recursive")
    n_before = pipe.index.count()
    r = pipe.ingest_file(write(tmp_path, "b.md", MD + "\nOne extra closing line.\n"), "recursive")
    assert r.n_duplicates >= 2
    assert pipe.index.count() < n_before + r.n_chunks
    rows = [json.loads(l) for l in pipe.cfg.dup_log.read_text().splitlines()]
    assert rows and all(x["similarity"] > 0.95 for x in rows)
    assert pipe.index.is_in_sync()


def test_reingest_same_file_replaces_instead_of_duplicating(pipe, tmp_path):
    p = write(tmp_path, "a.md", MD)
    pipe.ingest_file(p, "recursive")
    n = pipe.index.count()
    r = pipe.ingest_file(p, "recursive")
    assert r.n_duplicates == 0 and pipe.index.count() == n


def test_reindex_without_original_file_and_strategy_switch(pipe, tmp_path):
    p = write(tmp_path, "a.md", MD)
    pipe.ingest_file(p, "recursive")
    p.unlink()                                      # original upload is gone
    reports = pipe.reindex("fixed")
    assert reports and reports[0].strategy == "fixed"
    strategies = {m["strategy"] for m in pipe.index.col.get(include=["metadatas"])["metadatas"]}
    assert strategies == {"fixed"}
    assert pipe.reindex("recursive", reparse=True)   # re-parse from stored raw bytes
    assert pipe.index.is_in_sync()


def test_bm25_is_rebuilt_when_out_of_sync(pipe, tmp_path):
    pipe.ingest_file(write(tmp_path, "a.md", MD), "recursive")
    pipe.cfg.bm25_path.write_text(json.dumps({"ids": [], "tokens": []}))   # simulate crash / corruption
    pipe2 = Pipeline(pipe.cfg, embedder=HashEmbedder(), auto_repair=False)
    assert not pipe2.index.is_in_sync()
    assert pipe2.index.repair() and pipe2.index.is_in_sync()
    assert pipe2.index.search_bm25("kittens yarn", 3)


def test_semantic_splits_on_topic_change(pipe):
    text = ("Cats purr softly. Cats chase mice. Cats nap on sofas. Cats groom their fur. Cats climb trees. "
            "Databases store records. Databases build indexes. Databases run SQL queries. "
            "Databases enforce transactions. Databases replicate data.")
    doc = parse_bytes(text.encode(), "t.txt")
    from rag.chunkers import SemanticChunker
    chunks = SemanticChunker(HashEmbedder(), 80, 20, 400).chunk(doc)
    assert len(chunks) >= 2


def test_refuses_to_mix_embedding_models(pipe):
    cfg2 = Config(data_dir=pipe.cfg.data_dir, embedding_provider="openai")
    with pytest.raises(RuntimeError, match="re-ingest"):
        Pipeline(cfg2, embedder=HashEmbedder())


def test_semantic_keeps_paragraph_breaks():
    from rag.chunkers import SemanticChunker
    text = ("First paragraph sentence one. First paragraph sentence two.\n\n"
            "Second paragraph here. It has two sentences.\n"
            "A line on its own.")
    doc = parse_bytes(text.encode(), "t.md")
    (chunk,) = SemanticChunker(HashEmbedder(), 90, 10_000, 10_000).chunk(doc)
    assert "two.\n\nSecond" in chunk.text and "sentences.\nA line" in chunk.text


def test_reindex_is_deterministic_and_keeps_the_alphabetically_first_copy(pipe, tmp_path):
    a = write(tmp_path, "a_original.md", MD)
    b = write(tmp_path, "b_copy.md", "---\ntitle: copy\n---\n" + MD)   # same content, different bytes
    pipe.ingest_file(b, "recursive")                                    # copy ingested FIRST
    pipe.ingest_file(a, "recursive")
    assert {m["source"] for m in pipe.index.col.get(include=["metadatas"])["metadatas"]} == {"b_copy.md"}
    pipe.reindex("recursive")
    assert {m["source"] for m in pipe.index.col.get(include=["metadatas"])["metadatas"]} == {"a_original.md"}
    rows = [json.loads(l) for l in pipe.cfg.dup_log.read_text().splitlines()]
    assert rows and all(r["source"] == "b_copy.md" for r in rows)
    assert pipe.index.is_in_sync()


def test_plain_text_headings_detected():
    txt = ("Study Guide Title\n\nVariables and types\nA variable is a name for a value. It is simple.\n\n"
           "Control flow\nPrograms use if and else. Loops repeat things.\n")
    doc = parse_bytes(txt.encode(), "g.txt")
    assert [s.heading for s in doc.segments] == ["Variables and types", "Control flow"]
    assert doc.segments[0].text.startswith("Study Guide Title") and "Variables and types" in doc.segments[0].text


def test_pdf_style_headings_without_blank_lines():
    from rag.loaders import parse_text
    txt = ("1. DSA (Phase 1 & 3)\nStriver Sheet [learn first] \u2014 the most followed sheet, free\n"
           "takeuforward.org/sheet\n2. CS Fundamentals \u2014 CN, OS, DBMS (Phase 1 &\n3)\nComputer Networks\n"
           "Neso Academy playlist [learn first] \u2014 builds OSI from scratch\nDBMS & SQL\n"
           "SQLBolt [practice] \u2014 interactive SQL\n")
    heads = [s.heading for s in parse_text(txt)]
    assert heads == ["1. DSA (Phase 1 & 3)",
                     "2. CS Fundamentals \u2014 CN, OS, DBMS (Phase 1 & 3) > Computer Networks",
                     "2. CS Fundamentals \u2014 CN, OS, DBMS (Phase 1 & 3) > DBMS & SQL"]


def test_windows_line_endings_are_handled():
    crlf = "---\r\ntitle: x\r\n---\r\n# Heading\r\n\r\nBody text here.\r\n"
    doc = parse_bytes(crlf.encode(), "w.md")
    assert doc.text == "Heading\n\nBody text here." or doc.text == "Heading\nBody text here."
    assert "title" not in doc.text and "\r" not in doc.text


def test_compare_runs_all_strategies(pipe, tmp_path):
    from rag.compare import format_report, run_comparison
    pipe.ingest_file(write(tmp_path, "a.md", MD), "recursive")
    qs = [{"question": "do cats purr", "answer_contains": "Cats purr when happy"},
          {"question": "what is a transaction", "answer_contains": "Transactions keep data consistent"}]
    res = run_comparison(pipe, qs, ["fixed", "recursive", "semantic"], k=5)
    assert list(res) == ["fixed", "recursive", "semantic"]
    assert all(len(v["rows"]) == 2 and v["chunks"] > 0 for v in res.values())
    assert res["recursive"]["rows"][0]["bm25"] == 1
    report = format_report(res, qs, 5)
    assert "hit@1" in report and "recursive" in report


# ------------------------------------------------------------------ phase 2: retrieval
def _hit(i, text="t"):
    return {"id": f"c{i}", "text": text, "source": "s.md", "heading": "h", "page": -1, "chunk_index": i, "strategy": "recursive"}


def test_rrf_rewards_agreement_and_respects_weights():
    from rag.retrieval import reciprocal_rank_fusion
    dense = [dict(_hit(1), score=0.9), dict(_hit(2), score=0.8), dict(_hit(3), score=0.7)]
    sparse = [dict(_hit(3), score=9.0), dict(_hit(4), score=5.0)]
    fused = reciprocal_rank_fusion(dense, sparse, 0.5, 0.5, k=60)
    assert fused[0].id == "c3"                                  # found by both lists -> wins
    c3 = fused[0]
    assert (c3.dense_rank, c3.sparse_rank) == (3, 1)
    assert abs(c3.rrf_score - (0.5 / 63 + 0.5 / 61)) < 1e-12
    dense_heavy = reciprocal_rank_fusion(dense, sparse, 0.99, 0.01)   # dense #1 now beats the "both lists" chunk
    sparse_heavy = reciprocal_rank_fusion(dense, sparse, 0.01, 0.99)
    assert dense_heavy[0].id == "c1" and sparse_heavy[0].id == "c3"
    a = reciprocal_rank_fusion(dense, sparse, 0.7, 0.3)
    b = reciprocal_rank_fusion(dense, sparse, 7, 3)             # only the ratio matters
    assert [(c.id, round(c.rrf_score, 12)) for c in a] == [(c.id, round(c.rrf_score, 12)) for c in b]


def test_rrf_rejects_bad_weights():
    from rag.retrieval import reciprocal_rank_fusion
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], [], 0, 0)
    with pytest.raises(ValueError):
        Config(dense_weight=-1)


@pytest.fixture
def retr(pipe, tmp_path):
    from rag.rerankers import LexicalOverlapReranker
    pipe.ingest_file(write(tmp_path, "a.md", MD + "\n```sql\nCREATE INDEX idx_students_email ON students (email);\n```\n"), "recursive")
    pipe.retriever._reranker = LexicalOverlapReranker()
    return pipe.retriever


def test_sparse_finds_exact_identifier(retr):
    res = retr.retrieve("idx_students_email", mode="sparse", rerank=False)
    assert res.results and "idx_students_email" in res.results[0].text
    assert res.results[0].sparse_rank == 1 and res.results[0].dense_rank is None


def test_hybrid_pipeline_sizes_and_fields(retr):
    res = retr.retrieve("how do cats behave", mode="hybrid", rerank=True)
    assert 0 < len(res.results) <= 5 <= 20 and len(res.candidates) <= 20
    assert res.reranked and all(c.rerank_score is not None for c in res.results)
    scores = [c.rerank_score for c in res.results]
    assert scores == sorted(scores, reverse=True)
    no_rr = retr.retrieve("how do cats behave", mode="hybrid", rerank=False)
    assert not no_rr.reranked and all(c.rerank_score is None for c in no_rr.results)
    assert [c.id for c in no_rr.results] == [c.id for c in no_rr.candidates][:len(no_rr.results)]


def test_reranker_can_reorder_candidates(retr):
    class Reverse:
        def score(self, query, texts):
            return list(range(len(texts)))                      # last candidate gets the highest score
    retr._reranker = Reverse()
    res = retr.retrieve("database indexes transactions", mode="hybrid", rerank=True)
    assert res.results[0].id == res.candidates[-1].id


def test_retrieval_eval_runs(retr):
    from rag.evaluate import format_eval, format_sweep, run_eval, sweep_weights
    qs = [{"type": "keyword", "question": "idx_students_email", "answer_contains": "CREATE INDEX idx_students_email"},
          {"type": "semantic", "question": "do cats purr", "answer_contains": "Cats purr when happy"}]
    out = run_eval(retr, qs)
    assert list(out) == ["dense", "sparse", "hybrid", "hybrid+rerank", "rerank+cutoff"]
    assert out["sparse"][0][0] == 1 and all(n >= 1 for rows in out.values() for _, n in rows)
    assert "MRR by question type" in format_eval(out, qs, 5)
    assert "0.7 / 0.3" in format_sweep(sweep_weights(retr, qs), len(qs), 5)


def test_tokenizer_drops_stopwords_but_keeps_identifiers():
    from rag.indexes import tokenize
    assert tokenize("what does a 404 status code mean") == ["404", "status", "code", "mean"]
    assert tokenize("idx_students_email IndentationError") == ["idx_students_email", "indentationerror"]
    assert tokenize("what is it") == [] and tokenize("what is it", keep_stopwords=True) == ["what", "is", "it"]


def test_query_of_only_stopwords_still_searches(retr):
    assert retr.sparse("what is the") is not None        # falls back to unfiltered tokens, no crash


def test_old_bm25_file_is_rebuilt_automatically(pipe, tmp_path):
    pipe.ingest_file(write(tmp_path, "a.md", MD), "recursive")
    data = json.loads(pipe.cfg.bm25_path.read_text())
    data.pop("tokenizer")                                              # looks like a file from the old tokenizer
    data["tokens"] = [["what", "does"] for _ in data["tokens"]]        # ...with stale content
    pipe.cfg.bm25_path.write_text(json.dumps(data))
    pipe2 = Pipeline(pipe.cfg, embedder=HashEmbedder())                # opening rebuilds from Chroma
    assert pipe2.index.is_in_sync() and pipe2.index.search_bm25("kittens yarn", 3)


def test_score_margin_cuts_the_irrelevant_tail(retr):
    class Fixed:
        def score(self, query, texts):
            return [5.0, 4.0, -3.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0][:len(texts)]
    retr._reranker = Fixed()
    full = retr.retrieve("database indexes transactions", mode="hybrid", rerank=True)
    cut = retr.retrieve("database indexes transactions", mode="hybrid", rerank=True, score_margin=5)
    assert len(full.results) >= 3 and len(cut.results) == 2            # keeps 5.0 and 4.0 only
    assert len(retr.retrieve("database indexes", mode="hybrid", rerank=False, score_margin=5).results) \
        == len(retr.retrieve("database indexes", mode="hybrid", rerank=False).results)   # margin needs a reranker


# ------------------------------------------------------------------ phase 3: grounded answers
def scripted_llm(gen, verdict=lambda claim, source: "supported", parts=None):
    from rag.llm import FakeLLM

    def handler(system, user, json_mode):
        if "fact-checker" in system:
            claim = user.split("CLAIM:\n", 1)[1].split("\n\nSOURCE:", 1)[0]
            source = user.split("SOURCE:\n", 1)[1]
            return json.dumps({"verdict": verdict(claim, source), "reason": "scripted"})
        if "distinct parts" in system:
            return json.dumps({"parts": parts or [{"part": "the question", "addressed": True}]})
        return gen
    return FakeLLM(handler)


def test_parse_claims_handles_citation_styles():
    from rag.answering import parse_claims
    ans = ("Cats purr when they are happy [1]. They sleep most of the day [1][2]. Kittens chase mice. [3]\n"
           "- Indexes speed up queries [2, 3].\n"
           "An uncited statement about databases.\n"
           "Not found in the context: pricing details.")
    claims, missing = parse_claims(ans)
    assert [c.citations for c in claims] == [[1], [1, 2], [3], [2, 3], []]
    assert claims[0].text == "Cats purr when they are happy."
    assert claims[2].text == "Kittens chase mice."                   # citation written after the full stop
    assert missing == ["Not found in the context: pricing details."]


def test_lenient_json_parsing():
    from rag.llm import parse_json
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here is the result: {"verdict": "supported"} Hope that helps') == {"verdict": "supported"}
    assert parse_json("no json here") is None and parse_json("[1, 2]") is None


def test_build_context_is_numbered_with_provenance():
    from rag.answering import Source, build_context
    ctx = build_context([Source(1, "a", "x.pdf", "Intro", 3, 0.9, "alpha text"), Source(2, "b", "y.md", "", -1, 0.5, "beta")])
    assert ctx.startswith("[1] (x.pdf, page 3, section: Intro)\nalpha text") and "\n\n[2] (y.md)\nbeta" in ctx


def test_confidence_math():
    from rag.confidence import citation_coverage, composite_score, label_for, retrieval_confidence, score_to_prob
    assert score_to_prob(0, "logit") == 0.5 and score_to_prob(-11, "logit") < 0.001 and score_to_prob(0.7, "unit") == 0.7
    assert retrieval_confidence([4.6], "logit") > 0.98                         # one strong chunk is enough
    assert retrieval_confidence([4.6, -10, -10], "logit") < retrieval_confidence([4.6], "logit")   # weak padding costs
    assert retrieval_confidence([], "logit") == 0.0
    assert citation_coverage(["verified", "partial", "unsupported", "uncited"]) == 0.375 and citation_coverage([]) is None
    w = (0.4, 0.4, 0.2)
    assert abs(composite_score(1.0, 0.5, 0.0, w) - 0.6) < 1e-9
    assert abs(composite_score(0.8, None, 1.0, w) - (0.8 * 0.4 + 1.0 * 0.2) / 0.6) < 1e-9   # missing part renormalized
    assert [label_for(x) for x in (0.9, 0.6, 0.2)] == ["high", "medium", "low"]


def test_answer_with_verified_citations(retr, pipe):
    from rag.answering import Answerer
    llm = scripted_llm("Cats purr when they are happy [1]. Kittens play with yarn [1].")
    ans = Answerer(retr, llm, pipe.cfg).ask("do cats purr when happy")
    assert ans.status == "answered" and [c.status for c in ans.claims] == ["verified", "verified"]
    assert ans.confidence.coverage == 1.0 and ans.confidence.completeness == 1.0 and ans.confidence.label == "high"
    assert ans.flagged_answer == ans.answer and ans.sources[0].n == 1 and not ans.notes
    assert "cats" in ans.sources[0].text.lower()


def test_unsupported_and_invalid_citations_are_flagged(retr, pipe):
    from rag.answering import Answerer, format_answer
    llm = scripted_llm("Cats purr when happy [1]. Cats can fly [1]. Dogs bark loudly [9].",
                       verdict=lambda claim, source: "not_supported" if "fly" in claim else "supported")
    ans = Answerer(retr, llm, pipe.cfg).ask("do cats purr when happy")
    # "Dogs bark loudly [9]": block 9 does not exist, but the scripted judge finds support in a real block -> miscited
    assert [c.status for c in ans.claims] == ["verified", "unsupported", "miscited"]
    assert ans.claims[2].verdicts == {9: "invalid"} and ans.claims[2].supported_by
    assert "[1?]" in ans.flagged_answer and "[9?]" in ans.flagged_answer
    assert abs(ans.confidence.coverage - 0.5) < 1e-9 and ans.notes
    assert "UNSUPPORTED" in format_answer(ans)


def test_partial_and_uncited_claims_lower_coverage(retr, pipe):
    from rag.answering import Answerer
    llm = scripted_llm("Cats purr [1]. Cats have fur.", verdict=lambda c, s: "partially_supported")
    ans = Answerer(retr, llm, pipe.cfg).ask("do cats purr when happy")
    assert [c.status for c in ans.claims] == ["partial", "uncited"] and ans.confidence.coverage == 0.25
    assert "[1~]" in ans.flagged_answer


def test_declines_without_calling_llm_when_retrieval_is_weak(retr, pipe):
    from rag.answering import NO_ANSWER_TEXT, Answerer, format_answer
    llm = scripted_llm("this must never be used")
    ans = Answerer(retr, llm, pipe.cfg).ask("zebra quantum spaceship")
    assert ans.status == "insufficient_evidence" and ans.answer == NO_ANSWER_TEXT
    assert llm.calls == []                                           # no LLM call, nothing to hallucinate with
    assert ans.suggested_documents and ans.suggested_documents[0]["source"] == "a.md"
    assert ans.found and "zebra quantum spaceship" in ans.not_found and ans.confidence.retrieval < 0.25
    text = format_answer(ans)
    assert "What was found:" in text and "What was not found:" in text and "Documents worth checking manually:" in text
    assert ans.to_dict()["status"] == "insufficient_evidence"


def test_model_declared_insufficient_context(retr, pipe):
    from rag.answering import Answerer
    llm = scripted_llm("INSUFFICIENT CONTEXT: The documents do not mention a cats' diet.")
    ans = Answerer(retr, llm, pipe.cfg).ask("do cats purr when happy")
    assert ans.status == "insufficient_evidence" and "diet" in ans.not_found
    assert [c[2] for c in llm.calls] == [False]                      # generated once, never reached the judge


def test_partial_answer_reports_missing_part(retr, pipe):
    from rag.answering import Answerer
    llm = scripted_llm("Cats purr when happy [1]. Not found in the context: how long they sleep.",
                       parts=[{"part": "do cats purr", "addressed": True}, {"part": "how long they sleep", "addressed": False}])
    ans = Answerer(retr, llm, pipe.cfg).ask("do cats purr when happy and how long do they sleep")   # a genuinely two-part question
    assert ans.status == "answered" and ans.missing_info == ["Not found in the context: how long they sleep."]
    assert ans.confidence.completeness == 0.5 and ans.confidence.missing_parts == ["how long they sleep"]


def test_unusable_judge_output_is_marked_unchecked_not_unsupported(retr, pipe):
    from rag.answering import Answerer
    from rag.llm import FakeLLM
    llm = FakeLLM(lambda s, u, j: "Cats purr when happy [1]." if "ONLY" in s else "I think so, probably!")
    ans = Answerer(retr, llm, pipe.cfg).ask("do cats purr when happy")
    assert ans.claims[0].status == "unchecked" and ans.confidence.completeness is None


def test_verification_can_be_switched_off(retr, pipe):
    from rag.answering import Answerer
    llm = scripted_llm("Cats purr when happy [1].")
    ans = Answerer(retr, llm, pipe.cfg).ask("do cats purr when happy", verify=False, check_completeness=False)
    assert ans.confidence.coverage is None and ans.confidence.completeness is None
    assert len(llm.calls) == 1 and abs(ans.confidence.composite - ans.confidence.retrieval) < 1e-9


def test_answer_eval_reports_refusals_and_hallucination_risk(retr, pipe):
    from rag.answer_eval import format_answer_eval, run_answer_eval
    from rag.answering import Answerer
    qs = [{"question": "do cats purr when happy", "answerable": True, "expected_keywords": ["purr"]},
          {"question": "zebra quantum spaceship", "answerable": False}]
    rows = run_answer_eval(Answerer(retr, scripted_llm("Cats purr when happy [1]."), pipe.cfg), qs)
    assert [r["correct"] for r in rows] == [True, True]
    report = format_answer_eval(rows)
    assert "1/1" in report and "unanswerable questions correctly declined: 1/1" in report


def test_config_validates_llm_settings():
    with pytest.raises(ValueError):
        Config(llm_provider="nope")
    with pytest.raises(ValueError):
        Config(confidence_weights=(0, 0, 0))
    assert Config().llm_model == "llama3.2:3b" and Config(llm_provider="openai").llm_model == "gpt-4o-mini"


def _judge(**reply):
    from rag.llm import FakeLLM
    return FakeLLM(lambda system, user, json_mode: json.dumps(reply))


def _check(llm, claim, source, question="what does a 404 status code mean"):
    from rag.answering import Claim, Source, verify_claims
    src = Source(1, "id", "f.html", "h", -1, 0.9, source)
    return verify_claims(llm, [Claim(claim, [1], claim)], {1: src}, question)[0]


SRC = "200 means the request succeeded. 404 means the server could not find the requested page. 500 means something went wrong."


def test_valid_evidence_quote_overrules_a_wrong_not_supported_verdict():
    chk = _check(_judge(evidence="404 means the server could not find the requested page.", verdict="not_supported",
                        reason="not explicitly stated"), "The 404 status code means the server could not find the requested page.", SRC)
    assert chk.status == "verified" and "overridden" in chk.reasons[1]


def test_negation_mismatch_is_not_overruled():
    chk = _check(_judge(evidence="404 means the server could not find the requested page.", verdict="not_supported",
                        reason="contradiction"), "The 404 status code means the server can find the requested page.", SRC)
    assert chk.status == "unsupported"


def test_number_mismatch_is_not_overruled():
    chk = _check(_judge(evidence="404 means the server could not find the requested page.", verdict="not_supported",
                        reason="wrong code"), "A 403 status code means the server could not find the requested page.", SRC)
    assert chk.status == "unsupported"


def test_supported_verdict_needs_a_real_quote():
    chk = _check(_judge(evidence="404 means the moon is made of cheese.", verdict="supported", reason="yes"),
                 "The 404 status code means the moon is made of cheese.", SRC)
    assert chk.status == "partial" and "not in the source" in chk.reasons[1]
    ok = _check(_judge(verdict="supported", reason="yes"), "The 404 status code means the server could not find the page.", SRC)
    assert ok.status == "verified"                                        # no evidence field at all: judge trusted as before


def test_completeness_ignores_citation_markers_and_bogus_parts(retr, pipe):
    from rag.answering import Answerer
    seen = []

    def handler(system, user, json_mode):
        if "fact-checker" in system:
            return json.dumps({"verdict": "supported", "reason": "ok"})
        if "distinct parts" in system:
            seen.append(user)
            return json.dumps({"parts": [{"part": "do cats purr", "addressed": True}, {"part": "[1]", "addressed": False}]})
        return "Cats purr when happy [1]."
    from rag.llm import FakeLLM
    ans = Answerer(retr, FakeLLM(handler), pipe.cfg).ask("do cats purr when happy")
    assert "[1]" not in seen[0].split("ANSWER:")[1]                       # citation markers stripped before judging
    assert ans.confidence.completeness == 1.0 and ans.confidence.missing_parts == []


def test_single_part_question_collapses_judge_noise():
    from rag.confidence import completeness_score, is_single_part
    from rag.llm import FakeLLM
    assert is_single_part("what does a 404 status code mean") and is_single_part("which free resources teach system design")
    assert not is_single_part("what is a B-tree index and why is it used") and not is_single_part("what is X, and what is Y")
    noisy = json.dumps({"parts": [{"part": "what does a 404 status code mean", "addressed": False},
                                  {"part": "the 404 code", "addressed": True}]})
    assert completeness_score(FakeLLM(lambda s, u, j: noisy), "what does a 404 status code mean", "It means not found.") == (1.0, [])
    none = json.dumps({"parts": [{"part": "what does a 404 status code mean", "addressed": False}]})
    assert completeness_score(FakeLLM(lambda s, u, j: none), "what does a 404 status code mean", "I like cats.") \
        == (0.0, ["what does a 404 status code mean"])
    multi = json.dumps({"parts": [{"part": "what is a B-tree", "addressed": True}, {"part": "why is it used", "addressed": False}]})
    assert completeness_score(FakeLLM(lambda s, u, j: multi), "what is a B-tree index and why is it used", "A B-tree is sorted.") \
        == (0.5, ["why is it used"])                                   # genuinely multi-part questions keep their parts


def test_verbose_answer_eval_lists_unverified_claims(retr, pipe):
    from rag.answer_eval import format_answer_eval, run_answer_eval
    from rag.answering import Answerer
    llm = scripted_llm("Cats purr when happy [1]. Cats can fly [1].",
                       verdict=lambda claim, source: "not_supported" if "fly" in claim else "supported")
    rows = run_answer_eval(Answerer(retr, llm, pipe.cfg), [{"question": "do cats purr when happy", "answerable": True,
                                                              "expected_keywords": ["purr"]}])
    assert rows[0]["weak"][0]["status"] == "unsupported" and rows[0]["weak"][0]["claim"] == "Cats can fly."
    report = format_answer_eval(rows, verbose=True)
    assert "Claims that were not verified" in report and "[UNSUPPORTED] Cats can fly." in report
    assert "Claims that were not verified" not in format_answer_eval(rows)


def _keyword_judge():
    """Scripted judge: a block supports a claim if the block contains the claim's first word."""
    from rag.llm import FakeLLM

    def handler(system, user, json_mode):
        claim = user.split("CLAIM:\n", 1)[1].split("\n\nSOURCE:", 1)[0]
        source = user.split("SOURCE:\n", 1)[1]
        ok = claim.split()[0].lower() in source.lower()
        return json.dumps({"verdict": "supported" if ok else "not_supported", "reason": "scripted"})
    return FakeLLM(handler)


def _two_blocks():
    from rag.answering import Source
    return {1: Source(1, "a", "x.md", "A", -1, 0.9, "Alpha facts live here."),
            2: Source(2, "b", "y.md", "B", -1, 0.8, "Beta facts live here.")}


def test_wrong_citation_number_is_detected_and_corrected():
    from rag.answering import Claim, verify_claims
    from rag.confidence import citation_coverage
    claims = [Claim("Alpha is described", [2], "Alpha is described [2]."),      # real support is block 1
              Claim("Alpha again", [3], "Alpha again [3]."),                      # block 3 does not exist
              Claim("Beta is described", [2], "Beta is described [2].")]         # correctly cited
    checks = verify_claims(_keyword_judge(), claims, _two_blocks())
    assert [c.status for c in checks] == ["miscited", "miscited", "verified"]
    assert checks[0].supported_by == [1] and checks[1].verdicts == {3: "invalid"} and checks[1].supported_by == [1]
    assert citation_coverage([c.status for c in checks]) == (0.5 + 0.5 + 1.0) / 3


def test_miscited_rescue_can_be_disabled_and_never_rescues_nonsense():
    from rag.answering import Claim, verify_claims
    off = verify_claims(_keyword_judge(), [Claim("Alpha is described", [2], "x")], _two_blocks(), rescue=False)
    assert off[0].status == "unsupported" and off[0].supported_by == []
    nonsense = verify_claims(_keyword_judge(), [Claim("Zebra facts", [2], "x")], _two_blocks())
    assert nonsense[0].status == "unsupported"                              # no block supports it, so it stays flagged


def test_list_lead_ins_are_not_claims_but_list_items_are():
    from rag.answering import parse_claims
    claims, missing = parse_claims("The free resources are:\n- ByteByteGo videos [1]\n- System Design Primer [2]")
    assert [(c.text, c.citations) for c in claims] == [("ByteByteGo videos", [1]), ("System Design Primer", [2])]


def test_multi_sentence_evidence_can_overrule_for_a_combined_claim():
    src = ("Transactions are described by the ACID properties. Atomicity means all or nothing. "
           "Consistency means valid states. Isolation means no interference. Durability means it survives a crash.")
    ev = "Atomicity means all or nothing. Consistency means valid states. Isolation means no interference. Durability means it survives a crash."
    chk = _check(_judge(evidence=ev, verdict="not_supported", reason="only atomicity defined"),
                 "ACID stands for Atomicity, Consistency, Isolation, and Durability.", src,
                 question="what does ACID stand for in database transactions")
    assert chk.status == "verified"


def test_possessive_and_fragment_claims_match_their_quote():
    src = "ByteByteGo (YouTube) short visual explainers per concept. Gaurav Sen best free beginner-friendly system design series."
    chk = _check(_judge(evidence="ByteByteGo (YouTube) short visual explainers per concept.", verdict="not_supported", reason="fragment"),
                 "ByteByteGo's YouTube explainers", src, question="which free resources teach system design")
    assert chk.status == "verified"


def test_other_block_that_only_partly_supports_gives_partial_not_unsupported():
    from rag.answering import Claim, verify_claims
    from rag.llm import FakeLLM

    def handler(system, user, json_mode):
        source = user.split("SOURCE:\n", 1)[1]
        verdict = "partially_supported" if "Alpha" in source else "not_supported"
        return json.dumps({"verdict": verdict, "reason": "scripted"})
    chk = verify_claims(FakeLLM(handler), [Claim("Alpha is large and fast", [2], "x")], _two_blocks())[0]
    assert chk.status == "partial" and chk.supported_by == [1]
