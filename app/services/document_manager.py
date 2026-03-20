from __future__ import annotations

from pathlib import Path

from app.services.document_loader import SUPPORTED_EXTENSIONS


def sanitize_upload_filename(filename: str) -> str:
    clean_name = Path(filename or "").name.strip()
    if not clean_name:
        raise ValueError("文件名不能为空。")
    if Path(clean_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"仅支持以下格式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    return clean_name


def resolve_document_path(docs_dir: Path, relative_path: str) -> Path:
    candidate = (docs_dir / relative_path).resolve()
    docs_root = docs_dir.resolve()
    if docs_root not in candidate.parents and candidate != docs_root:
        raise ValueError("非法文档路径。")
    if not candidate.is_file():
        raise FileNotFoundError("文档不存在。")
    return candidate


def delete_document(docs_dir: Path, relative_path: str) -> str:
    target = resolve_document_path(docs_dir, relative_path)
    deleted_name = str(target.relative_to(docs_dir))
    target.unlink()

    current = target.parent
    docs_root = docs_dir.resolve()
    while current != docs_root and current.exists() and not any(current.iterdir()):
        current.rmdir()
        current = current.parent

    return deleted_name
