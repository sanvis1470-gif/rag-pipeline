# RAG – Phase 1: ingestion & chunking pipeline

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
python -m rag compare                          # run eval_questions.json against all 3 chunking strategies
```
Or from Python: `Pipeline(Config(strategy="semantic")).ingest_file("a.pdf")`.

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

## Known limitations (good Phase 2 / polish items)
- Headings: markdown and HTML use real heading tags; PDFs use bookmarks if present. Plain text and PDFs without bookmarks use a *heuristic* (numbered lines like `1. Intro`, or short standalone lines without sentence punctuation / Title Case lines). It can misfire on numbered list items or very short one-line paragraphs; re-run `reindex --reparse` after changing it.
- Chunks never cross a segment boundary in `recursive`/`semantic` (so PDF chunks stay within one page).
- Scanned PDFs raise an error (no OCR). Tables in PDFs come out as flat text.
- BM25 is rebuilt in memory on every write; fine for thousands of chunks, not millions.
- Semantic chunking embeds every sentence window (extra API cost) and its threshold (`semantic_percentile`) needs tuning per corpus.
