"""Download the authors' public SUES-200 V2 archive directly on the server."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
DOWNLOADS = ROOT / "data/.downloads"
ARCHIVE = DOWNLOADS / "SUES-200-512x512-V2.zip"
PART = ARCHIVE.with_suffix(".zip.part")
EXPECTED = 5698122700
URL = "https://drive.usercontent.google.com/download?id=1UyVyFJ_pRaJHIr_eBY2HL7gkS5y9UxqI&export=download&authuser=0&confirm=t"
STATE = RUNTIME / "sues200-download-status.json"


def save(state):
    state["updated_at"] = time.time()
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(STATE)


def main(argv=None):
    global ARCHIVE, PART, DOWNLOADS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARCHIVE,
        help="Destination ZIP; an interrupted download resumes from .zip.part",
    )
    args = parser.parse_args(argv)
    ARCHIVE = args.output.expanduser().resolve()
    DOWNLOADS = ARCHIVE.parent
    PART = ARCHIVE.with_suffix(".zip.part")
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(exist_ok=True)
    state = {
        "status": "downloading",
        "archive": str(ARCHIVE),
        "expected_bytes": EXPECTED,
        "bytes": PART.stat().st_size if PART.exists() else 0,
        "started_at": time.time(),
        "source_file_id": "1UyVyFJ_pRaJHIr_eBY2HL7gkS5y9UxqI",
    }
    session = requests.Session()
    session.trust_env = True
    if ARCHIVE.exists():
        if ARCHIVE.stat().st_size != EXPECTED:
            raise RuntimeError("Unexpected existing archive size")
    else:
        for attempt in range(1, 11):
            state["attempt"] = attempt
            offset = PART.stat().st_size if PART.exists() else 0
            if offset == EXPECTED:
                PART.replace(ARCHIVE)
                break
            if offset > EXPECTED:
                raise RuntimeError("Partial archive is larger than the source")
            save(state)
            try:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with session.get(
                    URL, headers=headers, stream=True, timeout=(15, 60)
                ) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        raise RuntimeError("Server did not honor the resume range")
                    if response.status_code == 206:
                        if not response.headers.get("Content-Range", "").startswith(
                            f"bytes {offset}-"
                        ):
                            raise RuntimeError("Unexpected resume offset")
                    received = offset
                    last_report = 0
                    with PART.open("ab" if offset else "wb") as output:
                        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                            if received == 0 and not chunk.startswith(b"PK\x03\x04"):
                                raise RuntimeError(
                                    "Download response is not a ZIP archive"
                                )
                            output.write(chunk)
                            received += len(chunk)
                            if received > EXPECTED:
                                raise RuntimeError("Source archive changed size")
                            if time.time() - last_report >= 10:
                                output.flush()
                                state["bytes"] = received
                                save(state)
                                print(
                                    f"Downloaded {received}/{EXPECTED} bytes",
                                    flush=True,
                                )
                                last_report = time.time()
                if PART.stat().st_size != EXPECTED:
                    raise RuntimeError("Incomplete archive response")
                PART.replace(ARCHIVE)
                break
            except Exception as error:
                state.update(
                    bytes=PART.stat().st_size if PART.exists() else 0,
                    last_error_type=type(error).__name__,
                    last_http_status=getattr(
                        getattr(error, "response", None), "status_code", None
                    ),
                )
                save(state)
                print("Download retry:", attempt, type(error).__name__, flush=True)
                if attempt == 10:
                    state["status"] = "failed"
                    save(state)
                    raise SystemExit(1) from None
                time.sleep(min(30, 3 * attempt))
    state.update(status="hashing", bytes=ARCHIVE.stat().st_size)
    save(state)
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    state.update(
        status="downloaded", sha256=digest.hexdigest(), completed_at=time.time()
    )
    save(state)
    ARCHIVE.with_suffix(".zip.sha256").write_text(
        digest.hexdigest() + "  " + ARCHIVE.name + "\n"
    )
    print("ARCHIVE_DOWNLOADED", ARCHIVE, "BYTES", ARCHIVE.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
