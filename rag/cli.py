from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .config import STRATEGIES, Config
from .loaders import SUPPORTED


def _expand(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED) if p.is_dir() else [p]
    return out


def _preview(text: str, limit: int = 250) -> str:
    text = " | ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."          # cut at a word boundary, mark it


def main(argv: list[str] | None = None) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser(prog="rag")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--provider", choices=["local", "openai"], default="local")
    ap.add_argument("--model", default=None, help="embedding model name (default depends on provider)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="load, chunk, embed and index files/folders")
    p.add_argument("paths", nargs="+")
    p.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=200)

    p = sub.add_parser("reindex", help="re-chunk stored docs with a (different) strategy, no re-upload")
    p.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    p.add_argument("--reparse", action="store_true", help="re-run loaders on the raw files first")

    p = sub.add_parser("search", help="sanity-check retrieval (dense and BM25 side by side)")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--full", action="store_true", help="print the whole chunk text")

    p = sub.add_parser("retrieve", help="hybrid retrieval: dense + BM25 -> RRF -> cross-encoder rerank")
    p.add_argument("query")
    p.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default="hybrid")
    p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--dense-weight", type=float, default=None)
    p.add_argument("--sparse-weight", type=float, default=None)
    p.add_argument("--trace", action="store_true", help="also show every candidate before reranking")
    p.add_argument("--full", action="store_true", help="print whole chunk text")

    p = sub.add_parser("evaluate", help="dense vs sparse vs hybrid vs hybrid+rerank on labelled questions")
    p.add_argument("--questions", default="eval_hybrid.json")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--sweep", action="store_true", help="also sweep the dense/sparse weight ratio")

    p = sub.add_parser("compare", help="run eval questions against several chunking strategies")
    p.add_argument("--questions", default="eval_questions.json")
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    p.add_argument("-k", type=int, default=5)

    sub.add_parser("stats")
    sub.add_parser("verify", help="check Chroma and BM25 hold the same chunk ids")

    a = ap.parse_args(argv)
    extra = {}
    dw, sw = getattr(a, "dense_weight", None), getattr(a, "sparse_weight", None)
    if dw is not None or sw is not None:                 # give one and the other defaults to the remainder
        extra["dense_weight"] = dw if dw is not None else 1 - sw
        extra["sparse_weight"] = sw if sw is not None else 1 - dw
    cfg = Config(data_dir=a.data_dir, **extra, embedding_provider=a.provider, embedding_model=a.model,
                 strategy=getattr(a, "strategy", "recursive"),
                 chunk_size=getattr(a, "chunk_size", 1000),
                 chunk_overlap=getattr(a, "chunk_overlap", 200))

    from .pipeline import Pipeline
    if a.cmd == "verify":
        idx = Pipeline(cfg, auto_repair=False).index
        print(f"chroma={idx.count()} bm25={len(idx.bm25.ids)} in_sync={idx.is_in_sync()}")
        return
    pipe = Pipeline(cfg)

    if a.cmd in ("ingest", "reindex"):
        reports = (pipe.ingest_many(_expand(a.paths), a.strategy) if a.cmd == "ingest"
                   else pipe.reindex(a.strategy, a.reparse))
        for r in reports:
            print(f"{r.source:40s} [{r.strategy}] chunks={r.n_chunks} inserted={r.n_inserted} dupes={r.n_duplicates}")
    elif a.cmd == "retrieve":
        res = pipe.retriever.retrieve(a.query, mode=a.mode, rerank=not a.no_rerank, top_n=a.top_n)
        dw, sw = res.weights
        stage = a.mode + ("" if a.mode != "hybrid" else f" (RRF dense {dw:.2f} / sparse {sw:.2f})")
        print(f"\nmode: {stage}{' + cross-encoder rerank' if res.reranked else ''}"
              f" | {len(res.candidates)} candidates -> top {len(res.results)}\n")
        if a.trace:
            print("-- candidates before reranking --")
            for i, c in enumerate(res.candidates, 1):
                print(f"  {i:2d}. rrf={c.rrf_score:.4f} dense#{c.dense_rank or '-'} sparse#{c.sparse_rank or '-'}"
                      f"  {c.source} [{c.heading}]")
            print("\n-- final --")
        for i, c in enumerate(res.results, 1):
            scores = f"rerank={c.rerank_score:.3f}  " if c.rerank_score is not None else ""
            body = c.text if a.full else _preview(c.text)
            print(f"{i}. {scores}rrf={c.rrf_score:.4f}  dense#{c.dense_rank or '-'}  sparse#{c.sparse_rank or '-'}  "
                  f"{c.source} p{c.page} [{c.heading}] ({len(c.text)} chars)\n    {body}\n")
    elif a.cmd == "evaluate":
        import json
        from .evaluate import format_eval, format_sweep, run_eval, sweep_weights
        questions = json.loads(Path(a.questions).read_text(encoding="utf-8"))
        print(format_eval(run_eval(pipe.retriever, questions, a.top_n), questions, a.top_n))
        if a.sweep:
            print("\n" + format_sweep(sweep_weights(pipe.retriever, questions, a.top_n), len(questions), a.top_n))
    elif a.cmd == "compare":
        import json
        from .compare import format_report, run_comparison
        questions = json.loads(Path(a.questions).read_text(encoding="utf-8"))
        results = run_comparison(pipe, questions, a.strategies, a.k)
        print(format_report(results, questions, a.k))
        print(f"\nThe index now holds the last strategy ({a.strategies[-1]}). "
              "Run `reindex --strategy <name>` to pick the one you want.")
    elif a.cmd == "search":
        q_emb = pipe.embedder.embed([a.query])[0]
        for title, hits in (("DENSE", pipe.index.search_dense(q_emb, a.k)),
                            ("BM25", pipe.index.search_bm25(a.query, a.k))):
            print(f"\n== {title} ==")
            for h in hits:
                body = h["text"] if a.full else _preview(h["text"])
                print(f"{h['score']:.3f} {h['source']} p{h['page']} [{h['heading']}] ({h['char_count']} chars)\n    {body}")
    elif a.cmd == "stats":
        got = pipe.index.col.get(include=["metadatas"])["metadatas"]
        print("chunks:", len(got))
        print("by strategy:", dict(Counter(m["strategy"] for m in got)))
        print("by source:", dict(Counter(m["source"] for m in got)))
        print("in sync:", pipe.index.is_in_sync())


if __name__ == "__main__":
    main()
