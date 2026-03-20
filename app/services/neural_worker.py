from __future__ import annotations

import atexit
import json
import subprocess
import threading
from pathlib import Path

from app.core.config import BASE_DIR


class NeuralWorkerClient:
    def __init__(
        self,
        python_bin: str,
        embedding_model_path: Path,
        rerank_model_path: Path,
        device: str,
        embedding_batch_size: int,
        rerank_batch_size: int,
    ) -> None:
        self.python_bin = python_bin
        self.embedding_model_path = embedding_model_path
        self.rerank_model_path = rerank_model_path
        self.device = device
        self.embedding_batch_size = embedding_batch_size
        self.rerank_batch_size = rerank_batch_size
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        worker_script = BASE_DIR / "scripts" / "retrieval_model_worker.py"
        process = subprocess.Popen(
            [
                self.python_bin,
                str(worker_script),
                "--embedding-model-path",
                str(self.embedding_model_path),
                "--rerank-model-path",
                str(self.rerank_model_path),
                "--device",
                self.device,
                "--embedding-batch-size",
                str(self.embedding_batch_size),
                "--rerank-batch-size",
                str(self.rerank_batch_size),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._process = process

        if process.stdout is None:
            raise RuntimeError("worker stdout 不可用。")
        ready_line = process.stdout.readline()
        if not ready_line:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"worker 启动失败。{stderr.strip()}")

        payload = json.loads(ready_line)
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error", "worker 初始化失败。")))

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        with self._lock:
            self._ensure_started()
            assert self._process is not None
            if self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("worker 管道不可用。")

            self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._process.stdin.flush()

            response_line = self._process.stdout.readline()
            if not response_line:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise RuntimeError(f"worker 无响应。{stderr.strip()}")

            response = json.loads(response_line)
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error", "worker 请求失败。")))
            return response

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._request({"command": "embed", "texts": texts})
        return list(response.get("embeddings", []))

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        response = self._request({"command": "rerank", "query": query, "documents": documents})
        return [float(score) for score in response.get("scores", [])]

    def close(self) -> None:
        with self._lock:
            if self._process is None:
                return
            try:
                if self._process.poll() is None and self._process.stdin and self._process.stdout:
                    self._process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                    self._process.stdin.flush()
                    self._process.stdout.readline()
            except Exception:  # noqa: BLE001
                pass
            finally:
                if self._process.poll() is None:
                    self._process.terminate()
                self._process = None
