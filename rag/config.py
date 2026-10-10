from dataclasses import dataclass
from pathlib import Path

STRATEGIES = ("fixed", "recursive", "semantic")
DEFAULT_MODELS = {"local": "BAAI/bge-small-en-v1.5", "openai": "text-embedding-3-small"}


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
