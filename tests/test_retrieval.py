import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.document_loader import RawDocument
from app.services.retrieval import LocalRetriever, chunk_document, tokenize


class RetrievalTests(unittest.TestCase):
    def test_tokenize_supports_chinese_and_ascii(self) -> None:
        tokens = tokenize("Qwen3 可以分析图像，并回答中文问题。")
        self.assertIn("qwen3", tokens)
        self.assertIn("可", tokens)
        self.assertIn("图", tokens)

    def test_chunk_document_creates_multiple_chunks(self) -> None:
        document = RawDocument(source="demo.md", text="A" * 1200)
        chunks = chunk_document(document, chunk_size=400, chunk_overlap=50)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0].chunk_id, "demo.md#chunk-0")

    def test_search_uses_faiss_candidates_and_rerank_output(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            index_path = root / "index.json"

            (docs_dir / "alpha.md").write_text("苹果 公司 产品 手机 生态 系统。", encoding="utf-8")
            (docs_dir / "beta.md").write_text("香蕉 水果 营养 丰富。", encoding="utf-8")
            (docs_dir / "gamma.md").write_text("苹果 手机 发布 会 总结 和 参数。", encoding="utf-8")

            retriever = LocalRetriever(
                docs_dir=docs_dir,
                index_path=index_path,
                chunk_size=200,
                chunk_overlap=20,
                worker=None,
                enable_neural_retrieval=False,
            )
            hits = retriever.search("苹果手机参数", top_k=2)

            self.assertGreaterEqual(len(hits), 1)
            self.assertIn("score", hits[0])
            self.assertIn("semantic_score", hits[0])
            self.assertIn("bm25_score", hits[0])
            self.assertIn("rerank_score", hits[0])
            self.assertEqual(hits[0]["source"], "gamma.md")


if __name__ == "__main__":
    unittest.main()
