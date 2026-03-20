from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        os.environ.setdefault(key, value)


def _resolve_path(raw_value: str, default: Path) -> Path:
    candidate = Path(raw_value).expanduser() if raw_value else default
    if not candidate.is_absolute():
        candidate = (BASE_DIR / candidate).resolve()
    return candidate


@dataclass(slots=True)
class Settings:
    app_host: str
    app_port: int
    enable_rag: bool
    qwen_api_url: str
    qwen_api_key: str
    qwen_model: str
    qwen_max_tokens: int
    model_python_bin: str
    embedding_model_path: Path
    rerank_model_path: Path
    embedding_device: str
    embedding_batch_size: int
    rerank_batch_size: int
    enable_neural_retrieval: bool
    docs_dir: Path
    index_path: Path
    rag_top_k: int
    chunk_size: int
    chunk_overlap: int
    max_context_chars: int

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv_file(BASE_DIR / ".env")

        embedding_model_path = _resolve_path(os.getenv("EMBEDDING_MODEL_PATH", "./model/bge-m3"), BASE_DIR / "model" / "bge-m3")
        rerank_model_path = _resolve_path(
            os.getenv("RERANK_MODEL_PATH", "./model/bge-reranker-v2-m3"),
            BASE_DIR / "model" / "bge-reranker-v2-m3",
        )
        docs_dir = _resolve_path(os.getenv("DOCS_DIR", "./data/documents"), BASE_DIR / "data" / "documents")
        index_path = _resolve_path(os.getenv("INDEX_PATH", "./data/index/index.json"), BASE_DIR / "data" / "index" / "index.json")

        docs_dir.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        return cls(
            app_host=os.getenv("APP_HOST", "0.0.0.0"),
            app_port=int(os.getenv("APP_PORT", "7860")),
            enable_rag=os.getenv("ENABLE_RAG", "true").strip().lower() not in {"0", "false", "no"},
            qwen_api_url=os.getenv("QWEN_API_URL", "").strip(),
            qwen_api_key=os.getenv("QWEN_API_KEY", "").strip(),
            qwen_model=os.getenv("QWEN_MODEL", "qwen3.5-flash").strip() or "qwen3.5-flash",
            qwen_max_tokens=int(os.getenv("QWEN_MAX_TOKENS", "512")),
            model_python_bin=os.getenv("MODEL_PYTHON_BIN", "").strip(),
            embedding_model_path=embedding_model_path,
            rerank_model_path=rerank_model_path,
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cuda").strip() or "cuda",
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
            rerank_batch_size=int(os.getenv("RERANK_BATCH_SIZE", "8")),
            enable_neural_retrieval=os.getenv("ENABLE_NEURAL_RETRIEVAL", "true").strip().lower() not in {"0", "false", "no"},
            docs_dir=docs_dir,
            index_path=index_path,
            rag_top_k=int(os.getenv("RAG_TOP_K", "4")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "5000")),
        )


settings = Settings.load()
