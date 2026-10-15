# Hybrid RAG retrieval engine

Retrieval for question answering over your own documents (RAG: retrieval-augmented generation). It ingests Markdown, text, HTML and PDF files, splits them with three switchable chunking strategies, and finds answers with **hybrid search**: embedding search plus BM25 keyword search, merged with Reciprocal Rank Fusion and re-scored by a cross-encoder. Everything runs locally with no API key, and every stage is measured on labelled questions.

## Results

14 labelled questions (6 exact-identifier, 5 paraphrased, 3 mixed) over 26 chunks, recursive chunking, local models. A hit means the chunk contains the whole answer phrase; MRR is 1.00 when the right chunk is always first.

| stage | hit@1 | hit@5 | MRR |
|---|---|---|---|
| dense (embeddings) only | 10/14 | 13/14 | 0.79 |
| BM25 only | 11/14 | 13/14 | 0.86 |
| hybrid (RRF 0.7 / 0.3) | 11/14 | 14/14 | 0.89 |
| hybrid + cross-encoder rerank | 13/14 | 14/14 | 0.96 |

- On exact-identifier questions dense search had MRR 0.70 against 1.00 for BM25 (it missed `idx_students_email` entirely). Hybrid found the right chunk in the top 5 for every question.
- The reranker lifted paraphrased-question MRR from 0.70 to 0.90.
- A dense/sparse weight sweep slightly favoured 0.5 / 0.5, but that is a one-question difference. With the reranker on, weights barely matter because it sees the whole 20-candidate pool either way.
- Measured before the stopword and score-cutoff changes below. Caveats: small question set written after reading the documents, tiny corpus, no unanswerable questions yet.

Chunking comparison (`python -m rag compare`, 10 questions): fixed, recursive and semantic all found the answer in the top 3 for every question. Recursive was chosen as the default (same top accuracy, real section boundaries, no extra embedding cost).

## How it works

```
file ──► loader ──► Document(segments: text + heading + page) ──► data/raw + data/processed
                         │
            chunker (fixed | recursive | semantic)
                         │
        embed (local bge-small, or OpenAI) ──► dedup (cos > 0.95) ──► ChromaDB + BM25

question ─┬─► dense  (embed question, Chroma cosine top-10) ──┐
          └─► sparse (BM25 top-10)  ───────────────────────────┴─► weighted RRF ─► top 20 ─► cross-encoder ─► top 5
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                              # offline, uses a fake embedder
python -m rag ingest docs/ --strategy recursive  # sample documents included
python -m rag retrieve "what does a 404 status code mean" --trace
python -m rag evaluate --sweep
```

Embeddings are local by default (sentence-transformers, `BAAI/bge-small-en-v1.5`, about 130 MB downloaded on first run); the reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2` (about 90 MB). To use OpenAI embeddings instead, set `OPENAI_API_KEY` and pass `--provider openai`.

## Commands

```bash
python -m rag ingest docs/ --strategy recursive
python -m rag reindex --strategy semantic      # re-chunk stored docs, no re-upload
python -m rag reindex --reparse                # also re-run loaders on stored raw files
python -m rag search "how does X work" -k 5    # dense vs BM25 side by side
python -m rag retrieve "..." --trace           # full hybrid pipeline, shows every stage
python -m rag retrieve "..." --margin 5        # drop results scoring >5 below the best reranker score
python -m rag evaluate --sweep                 # dense vs sparse vs hybrid vs hybrid+rerank, plus weight sweep
python -m rag compare                          # run eval_questions.json against all 3 chunking strategies
python -m rag stats
python -m rag verify                           # Chroma ids == BM25 ids ?
```

Or from Python: `Pipeline(Config(strategy="semantic")).ingest_file("a.pdf")`.

## Retrieval details

- **RRF:** `score(chunk) = w_dense/(60 + rank_dense) + w_sparse/(60 + rank_sparse)`. It uses ranks, not raw scores, so cosine and BM25 scales never need normalizing. Default weights 0.7 / 0.3 (`Config.dense_weight`, `sparse_weight`; only the ratio matters).
- **Reranker:** a cross-encoder reads the question and chunk together, which is why it is more precise than embedding similarity. It only sees the 20 fused candidates, because running it on everything would be too slow.
- **Score cutoff (optional):** the reranker always yields 5 results, but often only one is relevant. `--margin M` (or `Config.rerank_margin`) keeps only results within `M` points of the best score. A relative margin is used because an absolute threshold is too blunt: in testing, a correct chunk scored -1.9 while the top chunk scored +0.1.
- **BM25 stopwords:** common English words are removed from BM25 tokens, so "what does a 404 status code mean" no longer matches every chunk on "what" and "does". Identifiers such as `idx_students_email` are untouched. The stored index records a tokenizer version and rebuilds itself automatically when it changes.
- **Evaluation:** `eval_hybrid.json` has keyword, semantic and mixed questions so you can see where each stage helps. Add your own there.
- Defaults live in `Config`: `dense_k=10`, `sparse_k=10`, `rerank_candidates=20`, `rerank_top_n=5`, `rrf_k=60`.

## Layout

| file | job |
|---|---|
| `loaders.py` | md/txt/html/pdf → `Document` of `Segment(text, heading, page)`; normalization |
| `store.py` | `data/raw/<id>__<name>` + `data/processed/<id>.json` |
| `chunkers.py` | the 3 strategies behind one `Chunker.chunk(doc)` interface |
| `embeddings.py` | `LocalEmbedder` (default), `OpenAIEmbedder`, offline `HashEmbedder` for tests |
| `indexes.py` | `ChunkIndex` = Chroma (cosine) + `BM25Index`, kept in sync |
| `dedup.py` | cosine > 0.95 vs the index and earlier chunks in the same batch |
| `pipeline.py` | orchestration; `data/duplicates.jsonl` logs every skipped chunk |
| `compare.py` | runs `eval_questions.json` against every chunking strategy |
| `retrieval.py` | dense + sparse retrieval, weighted RRF, rerank step and score cutoff (`Retriever`) |
| `rerankers.py` | `CrossEncoderReranker` (default), offline `LexicalOverlapReranker` for tests |
| `evaluate.py` | per-stage metrics (hit@k, MRR, avg returned), per-question-type breakdown, weight sweep |

## Design decisions worth knowing

- **doc_id = hash of file bytes.** Re-uploading the same file replaces its chunks; dedup is for *different* documents repeating content.
- **Chroma is the source of truth; BM25 is derived.** Writes go Chroma → BM25. On startup, if the id sets differ, BM25 is rebuilt from Chroma.
- **Embed first, then delete old chunks, then insert** — a failed embedding call never leaves a document missing from the index.
- **One strategy per document in the collection.** Switching strategy = `reindex`, which clears and rebuilds from stored documents alphabetically, so the same copy of a duplicate document always survives.
- **`chunk_index` has gaps when duplicates are skipped** (ids are `docid:strategy:i`), so you can see what was dropped.
- **One embedding model per data dir.** Vector sizes differ between models, so the model is recorded in `data/embedding.json` and mixing is refused.
- Metadata per chunk: `source, doc_id, chunk_index, heading, page (-1 if n/a), strategy, char_count`.

## Known limitations

- Headings: markdown and HTML use real heading tags; PDFs use bookmarks if present. Plain text and PDFs without bookmarks use a heuristic (numbered lines, short standalone title lines) that can misfire on numbered list items.
- Chunks never cross a segment boundary in recursive/semantic mode (so PDF chunks stay within one page).
- Scanned PDFs raise an error (no OCR); PDF tables come out as flat text.
- BM25 is rebuilt in memory on every write: fine for thousands of chunks, not millions.
- Semantic chunking embeds every sentence window and its threshold needs tuning per corpus.
- The reranker only sees 20 candidates, so a relevant chunk missing from both first-stage lists cannot be recovered. Query embeddings use no instruction prefix. The evaluation sets are small.
