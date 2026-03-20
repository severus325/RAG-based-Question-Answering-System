from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model-path", required=True)
    parser.add_argument("--rerank-model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    return parser.parse_args()


def mean_pool(last_hidden_state, attention_mask, torch):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_hidden = last_hidden_state * mask
    pooled = masked_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class Worker:
    def __init__(
        self,
        embedding_model_path: str,
        rerank_model_path: str,
        device_name: str,
        embedding_batch_size: int,
        rerank_batch_size: int,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
        self.embedding_batch_size = embedding_batch_size
        self.rerank_batch_size = rerank_batch_size

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path, local_files_only=True)
        self.embedding_model = AutoModel.from_pretrained(
            embedding_model_path,
            local_files_only=True,
            weights_only=False,
        )
        self.embedding_model.to(self.device)
        self.embedding_model.eval()

        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_path, local_files_only=True)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_path, local_files_only=True)
        self.rerank_model.to(self.device)
        self.rerank_model.eval()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        outputs: list[list[float]] = []
        with self.torch.inference_mode():
            for batch in batched(texts, self.embedding_batch_size):
                encoded = self.embedding_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                model_output = self.embedding_model(**encoded)
                pooled = mean_pool(model_output.last_hidden_state, encoded["attention_mask"], self.torch)
                normalized = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.extend(normalized.detach().cpu().float().tolist())
        return outputs

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        scores: list[float] = []
        with self.torch.inference_mode():
            for batch in batched(documents, self.rerank_batch_size):
                encoded = self.rerank_tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.rerank_model(**encoded).logits.view(-1)
                scores.extend(logits.detach().cpu().float().tolist())
        return scores


def send(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = parse_args()
    try:
        worker = Worker(
            embedding_model_path=args.embedding_model_path,
            rerank_model_path=args.rerank_model_path,
            device_name=args.device,
            embedding_batch_size=args.embedding_batch_size,
            rerank_batch_size=args.rerank_batch_size,
        )
        send({"ok": True, "event": "ready"})
    except Exception as exc:  # noqa: BLE001
        send({"ok": False, "error": f"worker 初始化失败: {exc}"})
        return 1

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            command = request.get("command")
            if command == "embed":
                texts = [str(item) for item in request.get("texts", [])]
                send({"ok": True, "embeddings": worker.embed(texts)})
            elif command == "rerank":
                query = str(request.get("query", ""))
                documents = [str(item) for item in request.get("documents", [])]
                send({"ok": True, "scores": worker.rerank(query, documents)})
            elif command == "ping":
                send({"ok": True, "event": "pong"})
            elif command == "shutdown":
                send({"ok": True, "event": "shutdown"})
                return 0
            else:
                send({"ok": False, "error": f"未知命令: {command}"})
        except Exception as exc:  # noqa: BLE001
            send({"ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
