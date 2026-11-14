"""Release links fail clearly, and only verified checkpoints become visible."""

import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "star_download_weights", ROOT / "tools/download_weights.py"
)
weights = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(weights)


class WeightDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.payload = b"test checkpoint bytes" * 100
        self.entry = {
            "file": "example/model.pth",
            "url": None,
            "size_bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }

    def serve(self, payload):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop():
            server.shutdown()
            server.server_close()
            thread.join()

        self.addCleanup(stop)
        return f"http://127.0.0.1:{server.server_port}/model.pth"

    def test_release_entries_and_paths(self):
        entries = weights.load_manifest()
        self.assertEqual(set(entries), {"university1652", "sues200_150m"})
        self.assertIsNone(entries["university1652"]["altitude_m"])
        self.assertEqual(entries["sues200_150m"]["altitude_m"], 150)
        for entry in entries.values():
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(
                weights.checkpoint_path(self.root, entry).is_relative_to(self.root)
            )

    def test_direct_download_verifies_bytes_and_reuses_existing_file(self):
        url = self.serve(self.payload)
        with contextlib.redirect_stdout(io.StringIO()):
            path = weights.download(self.entry, self.root, url)
            self.assertEqual(path.read_bytes(), self.payload)
            # No link or network is needed to reuse an already verified file.
            self.assertEqual(weights.download(self.entry, self.root, ""), path)
        self.assertFalse(list(path.parent.glob("*.part")))

    def test_corrupted_download_is_not_published(self):
        url = self.serve(b"X" * len(self.payload))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            ValueError, "SHA-256"
        ):
            weights.download(self.entry, self.root, url)
        self.assertFalse((self.root / self.entry["file"]).exists())
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_existing_unrelated_file_is_preserved(self):
        target = self.root / self.entry["file"]
        target.parent.mkdir()
        target.write_bytes(b"user data")
        with self.assertRaises(ValueError):
            weights.download(self.entry, self.root, "https://example.invalid/model.pth")
        self.assertEqual(target.read_bytes(), b"user data")

    def test_pending_all_links_fail_before_any_download(self):
        entries = {
            "first": {**self.entry, "url": "https://example.invalid/model.pth"},
            "second": self.entry,
        }
        output = self.root / "not-created"
        with patch.object(weights, "load_manifest", return_value=entries), patch.object(
            weights, "download"
        ) as download, contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as error:
                weights.main(["--model", "all", "--output-dir", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("has not been added yet", stderr.getvalue())
            download.assert_not_called()
            self.assertFalse(output.exists())

    def test_manifest_cannot_escape_checkpoint_directory(self):
        for name in ("../outside.pth", "/outside.pth"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                weights.checkpoint_path(self.root, {**self.entry, "file": name})


if __name__ == "__main__":
    unittest.main()
