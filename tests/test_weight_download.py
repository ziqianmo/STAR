"""Release links fail clearly, and only verified checkpoints become visible."""

import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
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
        self.assertEqual(set(entries), {"university1652"})
        self.assertIsNone(entries["university1652"]["altitude_m"])
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

    def test_google_drive_links_extract_the_exact_file_id(self):
        cases = {
            "https://drive.google.com/file/d/1_DNSV1A3AkkqyTi9S1CDJMV8a_zT3toT/view?usp=drive_link": "1_DNSV1A3AkkqyTi9S1CDJMV8a_zT3toT",
            "https://drive.google.com/file/d/another_example_file_id/view?usp=drive_link": "another_example_file_id",
            "https://drive.google.com/uc?export=download&id=example_id": "example_id",
            "https://drive.usercontent.google.com/download?id=example_id": "example_id",
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(weights.google_drive_file_id(url), expected)
        self.assertIsNone(weights.google_drive_file_id("https://example.com/file.pth"))
        with self.assertRaisesRegex(ValueError, "not a folder"):
            weights.google_drive_file_id(
                "https://drive.google.com/drive/folders/example"
            )

    def test_google_drive_download_publishes_only_verified_weights(self):
        def fake_download(**kwargs):
            self.assertEqual(kwargs["id"], "example_id")
            self.assertFalse(kwargs["use_cookies"])
            self.assertEqual(kwargs["user_agent"], "Mozilla/5.0")
            Path(kwargs["output"]).write_bytes(self.payload)
            return kwargs["output"]

        gdown = SimpleNamespace(download=fake_download)
        with patch.dict(sys.modules, {"gdown": gdown}), contextlib.redirect_stdout(
            io.StringIO()
        ):
            path = weights.download(
                self.entry, self.root, "https://drive.google.com/file/d/example_id/view"
            )
        self.assertEqual(path.read_bytes(), self.payload)
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_corrupted_google_drive_download_is_not_published(self):
        def fake_download(**kwargs):
            Path(kwargs["output"]).write_bytes(b"X" * len(self.payload))
            return kwargs["output"]

        gdown = SimpleNamespace(download=fake_download)
        with patch.dict(sys.modules, {"gdown": gdown}), contextlib.redirect_stdout(
            io.StringIO()
        ), self.assertRaisesRegex(ValueError, "SHA-256"):
            weights.download(
                self.entry, self.root, "https://drive.google.com/file/d/example_id/view"
            )
        self.assertFalse((self.root / self.entry["file"]).exists())
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_google_drive_access_failure_does_not_leave_a_checkpoint(self):
        def fake_download(**kwargs):
            raise RuntimeError("Permission denied")

        gdown = SimpleNamespace(download=fake_download)
        with patch.dict(sys.modules, {"gdown": gdown}), contextlib.redirect_stdout(
            io.StringIO()
        ), self.assertRaisesRegex(RuntimeError, "public link access"):
            weights.download(
                self.entry, self.root, "https://drive.google.com/file/d/example_id/view"
            )
        self.assertFalse((self.root / self.entry["file"]).exists())
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_manifest_cannot_escape_checkpoint_directory(self):
        for name in ("../outside.pth", "/outside.pth"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                weights.checkpoint_path(self.root, {**self.entry, "file": name})


if __name__ == "__main__":
    unittest.main()
