import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.document_manager import delete_document, sanitize_upload_filename


class DocumentManagerTests(unittest.TestCase):
    def test_sanitize_upload_filename_keeps_supported_basename(self) -> None:
        self.assertEqual(sanitize_upload_filename("../demo.md"), "demo.md")

    def test_sanitize_upload_filename_rejects_unsupported_extension(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_upload_filename("bad.exe")

    def test_delete_document_removes_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            docs_dir = Path(tmp_dir)
            target = docs_dir / "demo.md"
            target.write_text("hello", encoding="utf-8")

            deleted = delete_document(docs_dir, "demo.md")

            self.assertEqual(deleted, "demo.md")
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
