from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import BASE_DIR, settings
from app.models import (
    ChatRequest,
    ChatResponse,
    DocumentOperationResponse,
    DocumentSummary,
    RagSettingsRequest,
    RagSettingsResponse,
    ReindexResponse,
    SourceItem,
)
from app.services.document_manager import delete_document, sanitize_upload_filename
from app.services.llm_client import QwenClient
from app.services.neural_worker import NeuralWorkerClient
from app.services.retrieval import LocalRetriever


app = FastAPI(title="Qwen API RAG", version="0.1.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
runtime_state = {"rag_enabled": settings.enable_rag}

worker = NeuralWorkerClient(
    python_bin=settings.model_python_bin,
    embedding_model_path=settings.embedding_model_path,
    rerank_model_path=settings.rerank_model_path,
    device=settings.embedding_device,
    embedding_batch_size=settings.embedding_batch_size,
    rerank_batch_size=settings.rerank_batch_size,
) if settings.enable_neural_retrieval else None

retriever = LocalRetriever(
    docs_dir=settings.docs_dir,
    index_path=settings.index_path,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    worker=worker,
    enable_neural_retrieval=settings.enable_neural_retrieval,
)
llm_client = QwenClient(
    api_url=settings.qwen_api_url,
    api_key=settings.qwen_api_key,
    model=settings.qwen_model,
    max_context_chars=settings.max_context_chars,
    max_tokens=settings.qwen_max_tokens,
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_name": settings.qwen_model,
            "docs_dir": str(settings.docs_dir),
        },
    )


@app.get("/api/health")
async def health() -> dict[str, object]:
    return {
        "ok": True,
        "model": settings.qwen_model,
        "base_url": settings.qwen_api_url,
        "api_ready": bool(settings.qwen_api_url and settings.qwen_api_key),
        "index_ready": bool(retriever.chunks),
        "neural_retrieval": settings.enable_neural_retrieval,
        "rag_enabled": runtime_state["rag_enabled"],
    }


@app.get("/api/documents", response_model=DocumentSummary)
async def documents() -> DocumentSummary:
    return DocumentSummary(**retriever.summary())


@app.get("/api/rag", response_model=RagSettingsResponse)
async def get_rag_settings() -> RagSettingsResponse:
    return RagSettingsResponse(rag_enabled=runtime_state["rag_enabled"])


@app.post("/api/rag", response_model=RagSettingsResponse)
async def set_rag_settings(payload: RagSettingsRequest) -> RagSettingsResponse:
    runtime_state["rag_enabled"] = payload.rag_enabled
    return RagSettingsResponse(rag_enabled=runtime_state["rag_enabled"])


@app.post("/api/documents/reindex", response_model=ReindexResponse)
async def reindex_documents() -> ReindexResponse:
    summary = retriever.rebuild_index()
    return ReindexResponse(ok=True, summary=DocumentSummary(**summary))


@app.post("/api/documents/upload", response_model=DocumentOperationResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentOperationResponse:
    filename = sanitize_upload_filename(file.filename or "")
    target = settings.docs_dir / filename
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    target.write_bytes(data)
    summary = retriever.rebuild_index()
    return DocumentOperationResponse(
        ok=True,
        message=f"文档已上传: {filename}",
        summary=DocumentSummary(**summary),
    )


@app.delete("/api/documents/{doc_path:path}", response_model=DocumentOperationResponse)
async def remove_document(doc_path: str) -> DocumentOperationResponse:
    try:
        deleted_name = delete_document(settings.docs_dir, doc_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary = retriever.rebuild_index()
    return DocumentOperationResponse(
        ok=True,
        message=f"文档已删除: {deleted_name}",
        summary=DocumentSummary(**summary),
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    clean_question = payload.question.strip()
    if not clean_question:
        return ChatResponse(answer="问题不能为空。", sources=[], retrieved_chunks=0, index_ready=bool(retriever.chunks))

    hits = retriever.search(clean_question, max(1, min(payload.top_k, 8))) if runtime_state["rag_enabled"] else []

    try:
        answer = llm_client.answer(question=clean_question, retrieved_chunks=hits)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败: {exc}") from exc

    sources = [
        SourceItem(
            source=item["source"],
            chunk_id=item["chunk_id"],
            score=float(item["score"]),
            preview=str(item["text"])[:220].strip(),
        )
        for item in hits
    ]
    return ChatResponse(
        answer=answer,
        sources=sources,
        retrieved_chunks=len(hits),
        index_ready=bool(retriever.chunks),
    )


def get_app() -> FastAPI:
    return app
