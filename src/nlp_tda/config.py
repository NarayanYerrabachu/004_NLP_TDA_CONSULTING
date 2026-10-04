from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Johannes: all runtime storage lives under this project folder (folder itself in Docker).
# Outside Docker, cwd is typically the project folder 004_NLP_TDA_CONSULTING/.
PROJECT_ROOT = Path(".")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NLP_TDA_", env_file=".env", extra="ignore")

    # Assumed hardware target: 16–32 GB RAM until corrected.
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    sqlite_path: Path = PROJECT_ROOT / "data" / "master.db"
    chroma_path: Path = PROJECT_ROOT / "data" / "chroma"
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    exports_dir: Path = PROJECT_ROOT / "data" / "exports"
    packs_dir: Path = PROJECT_ROOT / "data" / "packs"
    # In-repo synthetic fixtures stay in fixtures/; optional local pack path can override.
    fixtures_dir: Path = Path("fixtures/synthetic_engagement")

    # Multilingual EN+DE; MiniLM fits 16 GB; swap to BAAI/bge-m3 on 32 GB if desired.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    use_hash_embeddings: bool = False  # set True for offline smoke without downloading models

    ollama_base_url: str = "http://127.0.0.1:11434"
    # 16 GB: qwen2.5:7b; 32 GB: qwen2.5:14b — assumed until hardware corrected.
    ollama_model: str = "qwen2.5:7b"
    force_mock_llm: bool = False

    host: str = "0.0.0.0"
    port: int = 38417

    chunk_size: int = 800
    chunk_overlap: int = 120
    # Extraction reads the whole pack, this many chunks per LLM call.
    extract_batch_chunks: int = 6
    # 0 = no limit. A CPU-only 7B model can need minutes per call; a limit reads that many
    # batches spread evenly over the pack, and the run reports how many chunks were read.
    extract_max_batches: int = 0
    tda_pca_dims: int = 8
    tda_max_points: int = 200

    @property
    def templates_dir(self) -> Path:
        return Path(__file__).resolve().parent / "templates"


settings = Settings()
