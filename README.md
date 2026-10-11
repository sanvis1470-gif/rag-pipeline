# RAG – Phases 1-2: ingestion, chunking and hybrid retrieval

```
file ──► loader ──► Document(segments: text + heading + page) ──► data/raw + data/processed
                         │
                         ▼
            chunker (fixed | recursive | semantic)
                         ▼
        embed (local bge-small, or OpenAI) ──► dedup (cos > 0.95) ──► ChromaDB  +  BM25
```

## Setup
```bash
pip install -r requirements.txt
python -m pytest -q                 # offline, uses a fake embedder
# Default embeddings are LOCAL (sentence-transformers, BAAI/bge-small-en-v1.5): no API key needed.
# The model (~130 MB) downloads automatically on the first real run.
# To use OpenAI instead:  set OPENAI_API_KEY and pass  --provider openai
```

## Use
```bash
python -m rag ingest docs/ --strategy recursive
python -m rag reindex --strategy semantic      # re-chunk stored docs, no re-upload
python -m rag reindex --reparse                # also re-run loaders on stored raw files
python -m rag search "how does X work" -k 5    # dense vs BM25 side by side
python -m rag stats
python -m rag verify                           # Chroma ids == BM25 ids ?
```
Or from Python: `Pipeline(Config(strategy="semantic")).ingest_file("a.pdf")`.

## Phase 2: hybrid retrieval

```
question ─┬─► dense  (embed question, Chroma cosine top-10) ──┐
          └─► sparse (BM25 top-10)  ───────────────────────────┴─► weighted RRF ─► top 20 ─► cross-encoder ─► top 5
```
```bash
python -m rag retrieve "what does a 404 status code mean"          # full pipeline
python -m rag retrieve "idx_students_email" --mode sparse --no-rerank
python -m rag retrieve "..." --dense-weight 0.5 --trace            # sparse weight defaults to 1 - dense; --trace shows candidates
python -m rag evaluate --sweep                                     # dense vs sparse vs hybrid vs hybrid+rerank, plus weight sweep
```
- **RRF:** `score(chunk) = w_dense/(60 + rank_dense) + w_sparse/(60 + rank_sparse)`. It uses ranks, not raw scores, so cosine and BM25 scales never need normalizing. Default weights 0.7 / 0.3 (`Config.dense_weight`, `sparse_weight`; only the ratio matters).
- **Reranker:** local cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (downloads ~90 MB on first use, no API key). It reads question and chunk together, which is why it is more precise than embedding similarity.
- **Evaluation:** `eval_hybrid.json` has keyword (exact identifiers), semantic (paraphrased) and mixed questions, so you can see where each stage helps. Add your own questions there.
- Defaults live in `Config`: `dense_k=10`, `sparse_k=10`, `rerank_candidates=20`, `rerank_top_n=5`, `rrf_k=60`.

## Layout
| file | job |
|---|---|
| `loaders.py` | md/txt/html/pdf → `Document` of `Segment(text, heading, page)`; normalization |
| `store.py` | `data/raw/<id>__<name>` + `data/processed/<id>.json` |
| `chunkers.py` | the 3 strategies behind one `Chunker.chunk(doc)` interface |
| `embeddings.py` | `LocalEmbedder` (default), `OpenAIEmbedder`, offline `HashEmbedder` for tests |
| `indexes.py` | `ChunkIndex` = Chroma (cosine) + `BM25Index`, kept in sync |
| `dedup.py` | cosine > 0.95 vs the index *and* earlier chunks in the same batch |
| `pipeline.py` | orchestration; `data/duplicates.jsonl` logs every skipped chunk |

## Design decisions worth knowing
- **doc_id = hash of file bytes.** Re-uploading the same file replaces its chunks rather than duplicating them; dedup is for *different* docs repeating content.
- **Chroma is the source of truth; BM25 is derived.** Writes go Chroma → BM25. On startup, if the id sets differ, BM25 is rebuilt from Chroma (`repair()`).
- **Embed first, then delete old chunks, then insert** — a failed API call never leaves a doc missing from the index.
- **One strategy per doc in the collection.** Switching strategy = `reindex`, which replaces that doc's chunks (otherwise fixed vs recursive chunks of the same doc would dedup against each other).
- **`reindex` rebuilds from scratch**: it clears both indexes and the duplicate log, then goes through documents alphabetically, so the same copy of a duplicate document always survives.
- **chunk_index has gaps when duplicates are skipped** (ids are `docid:strategy:i`), so you can see what was dropped.
- Metadata per chunk: `source, doc_id, chunk_index, heading, page (-1 if n/a), strategy, char_count`.

- **One embedding model per data dir.** Vector sizes differ between models, so the pipeline records the model in `data/embedding.json` and refuses to mix them.

## Known limitations
- Headings: markdown and HTML use real heading tags; PDFs use bookmarks if present. Plain text and PDFs without bookmarks use a *heuristic* (numbered lines like `1. Intro`, or short standalone lines without sentence punctuation / Title Case lines). It can misfire on numbered list items or very short one-line paragraphs; re-run `reindex --reparse` after changing it.
- Chunks never cross a segment boundary in `recursive`/`semantic` (so PDF chunks stay within one page).
- Scanned PDFs raise an error (no OCR). Tables in PDFs come out as flat text.
- BM25 is rebuilt in memory on every write; fine for thousands of chunks, not millions.
- Semantic chunking embeds every sentence window (extra API cost) and its threshold (`semantic_percentile`) needs tuning per corpus.
- Phase 2: the reranker only sees the 20 fused candidates, so a relevant chunk missing from both first-stage lists can't be recovered. Query embeddings use no instruction prefix. The evaluation sets are small.
