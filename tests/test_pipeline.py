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
    assert list(out) == ["dense", "sparse", "hybrid", "hybrid+rerank"] and out["sparse"][0] == 1
    assert "MRR by question type" in format_eval(out, qs, 5)
    assert "0.7 / 0.3" in format_sweep(sweep_weights(retr, qs), len(qs), 5)
