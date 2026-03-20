from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.services.document_loader import RawDocument, load_documents
from app.services.neural_worker import NeuralWorkerClient

try:
    import faiss
except ModuleNotFoundError:  # pragma: no cover
    faiss = None


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
FAISS_CANDIDATE_MULTIPLIER = 8


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    source: str
    text: str
    tokens: list[str]
    length: int


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def chunk_document(document: RawDocument, chunk_size: int, chunk_overlap: int) -> list[ChunkRecord]:
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    chunks: list[ChunkRecord] = []
    text = document.text
    start = 0
    index = 0
    step = max(1, chunk_size - chunk_overlap)

    while start < len(text):
        end = min(len(text), start + chunk_size)
        slice_end = end
        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary > start + chunk_size // 3:
                slice_end = boundary

        chunk_text = text[start:slice_end].strip()
        if chunk_text:
            tokens = tokenize(chunk_text)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.source}#chunk-{index}",
                    source=document.source,
                    text=chunk_text,
                    tokens=tokens,
                    length=max(len(tokens), 1),
                )
            )
            index += 1

        if slice_end >= len(text):
            break

        start = max(start + step, slice_end - chunk_overlap)

    return chunks


class LocalRetriever:
    def __init__(
        self,
        docs_dir: Path,
        index_path: Path,
        chunk_size: int,
        chunk_overlap: int,
        worker: NeuralWorkerClient | None,
        enable_neural_retrieval: bool = True,
    ) -> None:
        self.docs_dir = docs_dir
        self.index_path = index_path
        self.vectors_path = index_path.with_name(f"{index_path.stem}_vectors.npy")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.worker = worker
        self.enable_neural_retrieval = enable_neural_retrieval and worker is not None
        self.chunks: list[ChunkRecord] = []
        self.doc_freq: dict[str, int] = {}
        self.avg_doc_len: float = 1.0
        self.indexed_at: str | None = None
        self.notes: list[str] = []
        self.vector_matrix = np.zeros((0, 0), dtype=np.float32)
        self.faiss_index: faiss.Index | None = None if faiss is not None else None
        self.load_or_build()

    def _disable_neural_retrieval(self, note: str) -> None:
        self.enable_neural_retrieval = False
        self.vector_matrix = np.zeros((0, 0), dtype=np.float32)
        self.faiss_index = None
        if note not in self.notes:
            self.notes.append(note)

    def load_or_build(self) -> None:
        if self.index_path.exists():
            try:
                payload = json.loads(self.index_path.read_text(encoding="utf-8"))
                self.indexed_at = payload.get("indexed_at")
                self.notes = payload.get("notes", [])
                self.chunks = [
                    ChunkRecord(
                        chunk_id=item["chunk_id"],
                        source=item["source"],
                        text=item["text"],
                        tokens=item["tokens"],
                        length=item["length"],
                    )
                    for item in payload.get("chunks", [])
                ]
                self._rebuild_statistics()
                if self.enable_neural_retrieval:
                    try:
                        self._load_vectors_or_rebuild()
                    except Exception as exc:  # noqa: BLE001
                        self._disable_neural_retrieval(f"本地 embedding/rerank 初始化失败，已回退到 BM25: {exc}")
                return
            except Exception:  # noqa: BLE001
                self.notes = ["索引文件损坏，已自动重建。"]

        self.rebuild_index()

    def _load_vectors_or_rebuild(self) -> None:
        if not self.vectors_path.exists():
            self.notes.append("向量文件缺失，已自动重建索引。")
            self.rebuild_index()
            return

        vectors = np.load(self.vectors_path)
        if vectors.shape[0] != len(self.chunks):
            self.notes.append("向量文件与文本块数量不一致，已自动重建索引。")
            self.rebuild_index()
            return

        self.vector_matrix = vectors.astype(np.float32)
        self._build_faiss_index()

    def _rebuild_statistics(self) -> None:
        self.doc_freq = {}
        total_len = 0
        for chunk in self.chunks:
            total_len += chunk.length
            for token in set(chunk.tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.avg_doc_len = total_len / len(self.chunks) if self.chunks else 1.0

    def _build_faiss_index(self) -> None:
        if self.vector_matrix.size == 0:
            self.faiss_index = None if faiss is None else faiss.IndexFlatIP(1)
            return

        if faiss is None:
            fallback_note = "未安装 faiss-cpu，当前向量召回退化为 NumPy 内积搜索。"
            if fallback_note not in self.notes:
                self.notes.append(fallback_note)
            self.faiss_index = None
            return

        index = faiss.IndexFlatIP(self.vector_matrix.shape[1])
        index.add(self.vector_matrix)
        self.faiss_index = index

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if not self.enable_neural_retrieval or self.worker is None:
            raise RuntimeError("当前未启用本地 embedding 检索。")
        embeddings = self.worker.embed(texts)
        return np.asarray(embeddings, dtype=np.float32)

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        if not self.enable_neural_retrieval or self.worker is None:
            return [0.0 for _ in documents]
        return self.worker.rerank(query, documents)

    def rebuild_index(self) -> dict[str, object]:
        documents, notes = load_documents(self.docs_dir)
        chunks: list[ChunkRecord] = []
        for document in documents:
            chunks.extend(chunk_document(document, self.chunk_size, self.chunk_overlap))

        self.chunks = chunks
        self.notes = notes
        self.indexed_at = datetime.now(timezone.utc).isoformat()
        self._rebuild_statistics()

        if self.enable_neural_retrieval and self.chunks:
            try:
                self.vector_matrix = self._encode_texts([chunk.text for chunk in self.chunks])
                np.save(self.vectors_path, self.vector_matrix)
                self._build_faiss_index()
            except Exception as exc:  # noqa: BLE001
                self._disable_neural_retrieval(f"本地 embedding/rerank 初始化失败，已回退到 BM25: {exc}")
        else:
            self.vector_matrix = np.zeros((0, 0), dtype=np.float32)
            self.faiss_index = None

        self._persist()
        return self.summary()

    def _persist(self) -> None:
        payload = {
            "indexed_at": self.indexed_at,
            "notes": self.notes,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def summary(self) -> dict[str, object]:
        files = sorted({chunk.source for chunk in self.chunks})
        return {
            "files": files,
            "file_count": len(files),
            "chunk_count": len(self.chunks),
            "indexed_at": self.indexed_at,
            "notes": self.notes,
            "extra": {
                "docs_dir": str(self.docs_dir),
                "index_path": str(self.index_path),
                "vectors_path": str(self.vectors_path),
                "retrieval_pipeline": "embedding -> faiss -> rerank" if self.enable_neural_retrieval else "bm25",
                "faiss_enabled": faiss is not None,
                "neural_enabled": self.enable_neural_retrieval,
            },
        }

    def _bm25_score(self, query_tokens: list[str], chunk: ChunkRecord) -> float:
        total_docs = len(self.chunks)
        if not total_docs:
            return 0.0

        k1 = 1.5
        b = 0.75
        tf: dict[str, int] = {}
        for token in chunk.tokens:
            tf[token] = tf.get(token, 0) + 1

        score = 0.0
        for token in query_tokens:
            freq = tf.get(token, 0)
            if not freq:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * chunk.length / self.avg_doc_len)
            score += idf * numerator / denominator
        return score

    def _lexical_search(self, query: str, top_k: int) -> list[dict[str, object]]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        scored_chunks: list[tuple[float, ChunkRecord]] = []
        normalized_query = query.casefold()
        unique_query_tokens = set(query_tokens)

        for chunk in self.chunks:
            bm25 = self._bm25_score(query_tokens, chunk)
            overlap = len(unique_query_tokens.intersection(chunk.tokens)) / max(len(unique_query_tokens), 1)
            phrase_bonus = 0.2 if normalized_query and normalized_query in chunk.text.casefold() else 0.0
            score = bm25 + overlap + phrase_bonus
            if score > 0:
                scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "text": chunk.text,
                "score": round(score, 4),
                "semantic_score": 0.0,
                "rerank_score": 0.0,
                "bm25_score": round(self._bm25_score(query_tokens, chunk), 4),
            }
            for score, chunk in scored_chunks[:top_k]
        ]

    def search(self, query: str, top_k: int) -> list[dict[str, object]]:
        if not self.enable_neural_retrieval or self.vector_matrix.size == 0:
            return self._lexical_search(query, top_k)

        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        query_embedding = self._encode_texts([query])
        candidate_count = min(max(top_k * FAISS_CANDIDATE_MULTIPLIER, top_k), len(self.chunks))

        if self.faiss_index is not None:
            semantic_scores, indices = self.faiss_index.search(query_embedding, candidate_count)
            candidates = [
                (int(chunk_index), float(score))
                for chunk_index, score in zip(indices[0], semantic_scores[0])
                if chunk_index >= 0
            ]
        else:
            scores = np.dot(self.vector_matrix, query_embedding[0])
            top_indices = np.argsort(scores)[::-1][:candidate_count]
            candidates = [(int(index), float(scores[index])) for index in top_indices]

        if not candidates:
            return self._lexical_search(query, top_k)

        candidate_chunks = [self.chunks[index] for index, _ in candidates]
        rerank_scores = self._rerank(query, [chunk.text for chunk in candidate_chunks])
        bm25_scores = [self._bm25_score(query_tokens, chunk) for chunk in candidate_chunks]

        max_semantic = max((score for _, score in candidates), default=1.0) or 1.0
        max_rerank = max(rerank_scores, default=1.0) or 1.0
        max_bm25 = max(bm25_scores, default=1.0) or 1.0
        normalized_query = query.casefold()
        unique_query_tokens = set(query_tokens)

        ranked: list[tuple[float, ChunkRecord, float, float, float]] = []
        for (chunk_index, semantic_score), rerank_score, bm25_score in zip(candidates, rerank_scores, bm25_scores):
            chunk = self.chunks[chunk_index]
            overlap = len(unique_query_tokens.intersection(chunk.tokens)) / max(len(unique_query_tokens), 1)
            phrase_bonus = 0.15 if normalized_query and normalized_query in chunk.text.casefold() else 0.0
            final_score = (
                0.70 * (rerank_score / max_rerank)
                + 0.20 * (semantic_score / max_semantic)
                + 0.10 * (bm25_score / max_bm25)
                + 0.05 * overlap
                + phrase_bonus
            )
            ranked.append((final_score, chunk, semantic_score, rerank_score, bm25_score))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "text": chunk.text,
                "score": round(score, 4),
                "semantic_score": round(semantic_score, 4),
                "rerank_score": round(rerank_score, 4),
                "bm25_score": round(bm25_score, 4),
            }
            for score, chunk, semantic_score, rerank_score, bm25_score in ranked[:top_k]
        ]
