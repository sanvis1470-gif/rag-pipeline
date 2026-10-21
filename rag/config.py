from dataclasses import dataclass
from pathlib import Path

STRATEGIES = ("fixed", "recursive", "semantic")
DEFAULT_MODELS = {"local": "BAAI/bge-small-en-v1.5", "openai": "text-embedding-3-small"}
DEFAULT_LLM_MODELS = {"ollama": "llama3.2:3b", "openai": "gpt-4o-mini", "custom": None}


@dataclass
class Config:
    data_dir: Path = Path("data")
    embedding_provider: str = "local"    # local (sentence-transformers) | openai
    embedding_model: str | None = None   # None -> provider default
    collection_name: str = "chunks"

    # chunking
    strategy: str = "recursive"          # fixed | recursive | semantic
    chunk_size: int = 1000               # characters
    chunk_overlap: int = 200             # characters (fixed + recursive)
    semantic_percentile: float = 90.0    # break where similarity drop is in the top (100-x)%
    semantic_min_chars: int = 200
    semantic_max_chars: int = 1500

    # retrieval (phase 2)
    dense_k: int = 10                    # candidates from the vector store
    sparse_k: int = 10                   # candidates from BM25
    rrf_k: int = 60                      # RRF damping constant (60 is the value from the original paper)
    dense_weight: float = 0.7            # only the ratio matters; normalized internally
    sparse_weight: float = 0.3
    rerank_candidates: int = 20          # fused candidates sent to the reranker
    rerank_top_n: int = 5                # chunks kept after reranking
    rerank_margin: float | None = None   # if set, drop results scoring more than this below the best one
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # generation + citations (phase 3)
    llm_provider: str = "ollama"         # ollama (local, free) | openai | custom (any OpenAI-compatible endpoint)
    llm_model: str | None = None         # None -> provider default
    llm_base_url: str | None = None      # required for "custom"; optional override for the others
    llm_temperature: float = 0.0
    llm_max_tokens: int = 700
    answer_score_margin: float | None = 5.0     # context = chunks within this many reranker points of the best
    min_retrieval_confidence: float = 0.25      # below this the system declines to answer
    verify_citations: bool = True
    check_completeness: bool = True
    rescue_miscited: bool = True         # when a cited block doesn't support a claim, look for one that does
    confidence_weights: tuple = (0.4, 0.4, 0.2)  # retrieval, citation coverage, completeness

    # dedup / embedding
    dedup_threshold: float = 0.95        # cosine similarity
    embed_batch_size: int = 128

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        if self.embedding_provider not in DEFAULT_MODELS:
            raise ValueError(f"embedding_provider must be one of {tuple(DEFAULT_MODELS)}")
        if self.embedding_model is None:
            self.embedding_model = DEFAULT_MODELS[self.embedding_provider]
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if self.llm_provider not in DEFAULT_LLM_MODELS:
            raise ValueError(f"llm_provider must be one of {tuple(DEFAULT_LLM_MODELS)}")
        if self.llm_model is None:
            self.llm_model = DEFAULT_LLM_MODELS[self.llm_provider]
        if len(self.confidence_weights) != 3 or min(self.confidence_weights) < 0 or sum(self.confidence_weights) <= 0:
            raise ValueError("confidence_weights must be 3 non-negative numbers that do not all equal 0")
        if min(self.dense_weight, self.sparse_weight) < 0 or self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("dense_weight and sparse_weight must be >= 0 and not both 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def bm25_path(self) -> Path:
        return self.data_dir / "bm25.json"

    @property
    def embedding_meta_path(self) -> Path:
        return self.data_dir / "embedding.json"

    @property
    def dup_log(self) -> Path:
        return self.data_dir / "duplicates.jsonl"
