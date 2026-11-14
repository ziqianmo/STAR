"""Download a released STAR checkpoint and verify its recorded SHA-256."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlparse

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "checkpoints/weights.json"


def load_manifest(filename=MANIFEST):
    return json.loads(Path(filename).read_text(encoding="utf-8"))


def checkpoint_path(root, entry):
    root = Path(root).expanduser().resolve()
    destination = (root / entry["file"]).resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError("Checkpoint file must remain inside the output directory.")
    return destination


def sha256_file(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(filename, entry):
    filename = Path(filename)
    if filename.stat().st_size != entry["size_bytes"]:
        raise ValueError("Checkpoint size does not match the released file.")
    if sha256_file(filename) != entry["sha256"]:
        raise ValueError("Checkpoint SHA-256 does not match the released file.")


def download(entry, root, url):
    destination = checkpoint_path(root, entry)
    if destination.exists():
        verify(destination, entry)
        print("Already verified:", destination)
        return destination
    if urlparse(url).scheme not in ("https", "http"):
        raise ValueError("A direct HTTP(S) download URL is required.")
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".",
        suffix=".part",
        dir=destination.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        print("Downloading:", destination, flush=True)
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            received = 0
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    received += len(chunk)
                    if received > entry["size_bytes"]:
                        raise ValueError(
                            "Download exceeds the recorded checkpoint size."
                        )
                    stream.write(chunk)
        verify(temporary, entry)
        if destination.exists():
            raise FileExistsError(
                "Destination appeared during download; preserving it."
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print("SHA-256 verified:", destination, flush=True)
    return destination


def main(argv=None):
    entries = load_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*entries, "all"))
    parser.add_argument(
        "--list", action="store_true", help="Show checkpoints and link availability"
    )
    parser.add_argument(
        "--url", help="Override one model's URL with a direct download link"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "checkpoints")
    args = parser.parse_args(argv)
    if args.list:
        for name, entry in entries.items():
            available = "available" if entry.get("url") else "link pending"
            print(f"{name}: {entry['file']} ({available})")
        return
    if not args.model:
        parser.error("Choose --model or use --list.")
    if args.url and args.model == "all":
        parser.error("--url requires a single model name.")
    selected = list(entries) if args.model == "all" else [args.model]
    # Check every requested link before downloading any files.
    for name in selected:
        entry = entries[name]
        url = args.url or entry.get("url")
        if not url:
            destination = checkpoint_path(args.output_dir, entry)
            if destination.is_file():
                try:
                    verify(destination, entry)
                    continue
                except ValueError as error:
                    parser.error(str(error))
            parser.error(
                f"The {name} download link has not been added yet. "
                "Place the checkpoint at "
                + str(destination)
                + ", or supply --url with a direct HTTP(S) link."
            )
        if urlparse(url).scheme not in ("https", "http"):
            parser.error("A direct HTTP(S) download URL is required.")
    for name in selected:
        entry = entries[name]
        download(entry, args.output_dir, args.url or entry.get("url") or "")


if __name__ == "__main__":
    main()
