"""Resume and verify a complete, paginated download of official University-1652."""

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".runtime" / "huggingface"))
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import build_hf_headers

REPO = "layumi/university-1652"
REVISION = "383b6208a4c6179de75b9c635dcf3f272f629cab"
TRAIN_FOLDERS = (
    "train/satellite/",
    "train/drone/",
    "test/query_drone/",
    "test/gallery_satellite/",
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    digest = hashlib.sha256() if entry["sha256"] else hashlib.sha1()
    if not entry["sha256"]:
        digest.update(("blob " + str(entry["size"]) + "\0").encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == (entry["sha256"] or entry["blob_id"])


def manifest(runtime, revision):
    path = runtime / ("university1652-manifest-" + revision + ".json")
    if path.exists():
        return json.loads(path.read_text())["files"]
    files = []
    print("Listing the complete repository with API pagination...", flush=True)
    for item in HfApi().list_repo_tree(
        REPO, repo_type="dataset", revision=revision, recursive=True
    ):
        if not isinstance(item, RepoFile):
            continue
        if Path(item.path).is_absolute() or ".." in Path(item.path).parts:
            raise ValueError("Unsafe repository path")
        lfs = item.lfs
        sha256 = (
            lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        )
        if lfs and not sha256:
            raise ValueError("Missing LFS checksum for " + item.path)
        files.append(
            {
                "path": item.path,
                "size": item.size,
                "blob_id": item.blob_id,
                "sha256": sha256,
            }
        )
        if len(files) % 10000 == 0:
            print("Manifest files:", len(files), flush=True)
    write_json(path, {"repo_id": REPO, "revision": revision, "files": files})
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/U1652")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY", ""))
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--training-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.batch_size <= 1000:
        parser.error(
            "workers must be positive and batch-size must be between 1 and 1000"
        )
    if args.proxy:
        os.environ["HTTPS_PROXY"] = args.proxy
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["NO_PROXY"] = ",".join(
            filter(
                None,
                [
                    os.environ.get("NO_PROXY", ""),
                    "hf.co",
                    "amazonaws.com",
                    "cloudfront.net",
                ],
            )
        )
    if not get_token():
        parser.error("Run tools/login_huggingface.py with an approved account first.")
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    status_file = runtime / "dataset-download-status.json"
    started = time.time()
    status = {
        "status": "listing",
        "repo_id": REPO,
        "revision": args.revision,
        "output": str(args.output),
        "started_at": started,
        "completed_files": 0,
        "completed_bytes": 0,
        "failed_files": [],
    }
    write_json(status_file, status)
    try:
        files = manifest(runtime, args.revision)
        if args.training_only:
            files = [
                entry
                for entry in files
                if entry["path"].startswith(TRAIN_FOLDERS)
                or entry["path"] == "README.md"
            ]
        if not files:
            raise ValueError("Repository contains no selected files")

        def priority(entry):
            for rank, folder in enumerate(
                (
                    "train/satellite/",
                    "test/gallery_satellite/",
                    "train/drone/",
                    "test/query_drone/",
                )
            ):
                if entry["path"].startswith(folder):
                    return rank, entry["path"]
            return 4, entry["path"]

        files.sort(key=priority)
        status.update(
            status="downloading",
            mode="git-lfs-batch",
            workers=args.workers,
            total_files=len(files),
            total_bytes=sum(entry["size"] for entry in files),
        )
        write_json(status_file, status)
        print(
            "Download manifest:",
            status["total_files"],
            "files,",
            status["total_bytes"],
            "bytes",
            flush=True,
        )

        local = threading.local()
        known_sha = {}
        known_lock = threading.Lock()
        api_session = requests.Session()
        api_session.trust_env = False
        api_headers = build_hf_headers(token=True)
        api_headers.update(
            {
                "Accept": "application/vnd.git-lfs+json",
                "Content-Type": "application/vnd.git-lfs+json",
            }
        )
        api_proxies = {"https": args.proxy} if args.proxy else {}

        def remember(entry, path):
            if entry["sha256"]:
                with known_lock:
                    known_sha[entry["sha256"]] = path

        def actions_for(entries):
            objects = {
                entry["sha256"]: {"oid": entry["sha256"], "size": entry["size"]}
                for entry in entries
                if entry["sha256"] and entry["sha256"] not in known_sha
            }
            if not objects:
                return {}
            payload = {
                "operation": "download",
                "transfers": ["basic"],
                "objects": list(objects.values()),
                "hash_algo": "sha256",
            }
            for attempt in range(3):
                try:
                    response = api_session.post(
                        "https://huggingface.co/datasets/"
                        + REPO
                        + ".git/info/lfs/objects/batch",
                        headers=api_headers,
                        json=payload,
                        proxies=api_proxies,
                        timeout=(15, 60),
                        allow_redirects=False,
                    )
                    if response.status_code == 429:
                        time.sleep(
                            min(
                                300,
                                max(5, int(response.headers.get("Retry-After", "30"))),
                            )
                        )
                        continue
                    response.raise_for_status()
                    return {
                        obj["oid"]: obj.get("actions", {}).get("download")
                        for obj in response.json()["objects"]
                    }
                except Exception:
                    if attempt == 2:
                        raise RuntimeError(
                            "Official batch API request failed"
                        ) from None
                    time.sleep(2**attempt)
            raise RuntimeError("Official batch API rate limit; rerun to resume")

        def fetch(entry, action):
            path = args.output / entry["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".star-part")
            with known_lock:
                source = known_sha.get(entry["sha256"]) if entry["sha256"] else None
            if source is not None and source != path:
                shutil.copyfile(source, temporary)
                if verify(temporary, entry):
                    temporary.replace(path)
                    return entry["size"]
            force = False
            for attempt in range(3):
                try:
                    if entry["sha256"] and action:
                        if not hasattr(local, "cdn"):
                            local.cdn = requests.Session()
                            local.cdn.trust_env = False
                        # The account token is sent only to the official API above.
                        with local.cdn.get(
                            action["href"],
                            headers=action.get("header", {}),
                            stream=True,
                            timeout=(10, 30),
                        ) as response:
                            response.raise_for_status()
                            with temporary.open("wb") as stream:
                                for chunk in response.iter_content(
                                    chunk_size=1024 * 1024
                                ):
                                    stream.write(chunk)
                        if verify(temporary, entry):
                            temporary.replace(path)
                            remember(entry, path)
                            return entry["size"]
                    else:
                        downloaded = hf_hub_download(
                            REPO,
                            filename=entry["path"],
                            repo_type="dataset",
                            revision=args.revision,
                            local_dir=str(args.output),
                            force_download=force,
                        )
                        if verify(Path(downloaded), entry):
                            remember(entry, Path(downloaded))
                            return entry["size"]
                    if verify(path, entry):
                        remember(entry, path)
                        return entry["size"]
                    force = True
                except Exception as error:
                    if attempt == 2:
                        response = getattr(error, "response", None)
                        raise RuntimeError(
                            f'{type(error).__name__}, HTTP {getattr(response, "status_code", None)}'
                        ) from None
                time.sleep(2**attempt)
            raise RuntimeError("Checksum mismatch")

        last_report = 0

        def report(force=False):
            nonlocal last_report
            now = time.time()
            if force or now - last_report >= 30:
                status["updated_at"] = now
                status["elapsed_seconds"] = round(now - started, 1)
                write_json(status_file, status)
                print(
                    f"Verified {status['completed_files']}/{status['total_files']} files; "
                    f"{status['completed_bytes']}/{status['total_bytes']} bytes; "
                    f"failed={len(status['failed_files'])}",
                    flush=True,
                )
                last_report = now

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            batches = iter(range(0, len(files), args.batch_size))
            pending = {}
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < args.batch_size * 2:
                    start = next(batches, None)
                    if start is None:
                        exhausted = True
                        break
                    missing = []
                    for entry in files[start : start + args.batch_size]:
                        path = args.output / entry["path"]
                        if verify(path, entry):
                            remember(entry, path)
                            status["completed_files"] += 1
                            status["completed_bytes"] += entry["size"]
                        else:
                            missing.append(entry)
                    actions = actions_for(missing)
                    pending.update(
                        {
                            pool.submit(
                                fetch, entry, actions.get(entry["sha256"])
                            ): entry
                            for entry in missing
                        }
                    )
                    report()
                if not pending:
                    continue
                done, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
                for future in done:
                    entry = pending.pop(future)
                    try:
                        status["completed_bytes"] += future.result()
                        status["completed_files"] += 1
                    except Exception as error:
                        status["failed_files"].append(
                            {"path": entry["path"], "error": str(error)}
                        )
                        print("FAILED", entry["path"], str(error), flush=True)
                    report()
                report()
        report(force=True)
        if status["failed_files"]:
            raise RuntimeError("Some files failed; rerun to resume and retry them")
        status["status"] = "complete"
        status["completed_at"] = time.time()
        write_json(status_file, status)
        print("COMPLETE: all selected files match the official checksums.", flush=True)
    except Exception as error:
        status.update(
            status="failed", error_type=type(error).__name__, updated_at=time.time()
        )
        write_json(status_file, status)
        print("DOWNLOAD_STOPPED", type(error).__name__, flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
