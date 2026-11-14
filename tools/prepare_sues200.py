"""Extract the official archive and prepare STAR's existing scene split."""

import argparse
import json
import os
import shutil
import stat
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import yaml
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "data"
ARCHIVE = BASE / ".downloads/SUES-200-512x512-V2.zip"
DESTINATION = BASE / "SUES-200-512x512"
STATUS = PROJECT / ".runtime/sues200-prepare-status.json"
HEIGHTS = ("150", "200", "250", "300")
SCENES = {f"{number:04d}" for number in range(1, 201)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def save(state):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.time()
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(STATUS)


def images(directory):
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def link_directory(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"Conflicting dataset link: {target}")
    elif target.exists():
        raise RuntimeError(f"Dataset path already exists: {target}")
    else:
        target.symlink_to(
            os.path.relpath(source, target.parent), target_is_directory=True
        )
    if not target.is_dir():
        raise RuntimeError(f"Broken dataset link: {target}")


def main(argv=None):
    global BASE, ARCHIVE, DESTINATION, STATUS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument(
        "--destination",
        type=Path,
        default=DESTINATION,
        help="Raw dataset root, also used for train/test links",
    )
    parser.add_argument(
        "--link",
        type=Path,
        default=PROJECT / "data/SUES-200",
        help="Project data link; an existing matching link is accepted",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Prepare links from an already extracted dataset",
    )
    args = parser.parse_args(argv)
    ARCHIVE = args.archive.expanduser().resolve()
    DESTINATION = args.destination.expanduser().resolve()
    BASE = DESTINATION.parent
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / ".downloads").mkdir(exist_ok=True)
    if args.split_only and not DESTINATION.is_dir():
        parser.error("--split-only requires an existing raw dataset directory")
    state = {
        "status": "checking_archive",
        "archive": str(ARCHIVE),
        "destination": str(DESTINATION),
        "started_at": time.time(),
    }
    save(state)
    try:
        if DESTINATION.exists():
            if not args.split_only:
                raise RuntimeError(
                    "Destination exists; use --split-only to prepare its train/test links"
                )
            staging = DESTINATION
        else:
            if ARCHIVE.stat().st_size != 5698122700:
                raise RuntimeError("Unexpected archive size")
            with zipfile.ZipFile(ARCHIVE) as archive:
                members = archive.infolist()
                seen = set()
                for member in members:
                    name = PurePosixPath(member.filename)
                    if (
                        name.is_absolute()
                        or ".." in name.parts
                        or "\\" in member.filename
                        or not name.parts
                        or member.filename in seen
                    ):
                        raise RuntimeError(
                            f"Unsafe or duplicate ZIP entry: {member.filename}"
                        )
                    mode = member.external_attr >> 16
                    if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise RuntimeError(
                            f"Unsupported ZIP entry type: {member.filename}"
                        )
                    if member.flag_bits & 1:
                        raise RuntimeError("Encrypted ZIP entries are not supported")
                    seen.add(member.filename)
                required = sum(member.file_size for member in members)
                if shutil.disk_usage(BASE).free < required + 1024**3:
                    raise RuntimeError("Insufficient space for extraction")
                staging = Path(
                    tempfile.mkdtemp(prefix="sues200-extract-", dir=BASE / ".downloads")
                )
                state.update(
                    status="extracting",
                    staging=str(staging),
                    archive_entries=len(members),
                    uncompressed_bytes=required,
                    extracted_entries=0,
                    extracted_bytes=0,
                )
                save(state)
                reported = time.monotonic()
                for index, member in enumerate(members, 1):
                    target = staging / member.filename
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        # Reading every member to EOF verifies its ZIP CRC.
                        with archive.open(member) as source, target.open(
                            "xb"
                        ) as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                    state["extracted_entries"] = index
                    state["extracted_bytes"] += member.file_size
                    if time.monotonic() - reported >= 5:
                        save(state)
                        print("EXTRACTED", index, "/", len(members), flush=True)
                        reported = time.monotonic()

        state.update(status="checking_raw_data", zip_crc_verified=not args.split_only)
        save(state)
        satellite_candidates = [
            p for p in staging.rglob("satellite-view") if p.is_dir()
        ]
        if len(satellite_candidates) != 1:
            raise RuntimeError("Expected one satellite-view directory")
        satellite = satellite_candidates[0]
        payload = satellite.parent
        drone_candidates = [
            p for p in payload.iterdir() if p.is_dir() and "drone" in p.name.lower()
        ]
        if len(drone_candidates) != 1:
            raise RuntimeError("Expected one drone-view directory")
        drone = drone_candidates[0]
        for directory in (satellite, drone):
            actual = {p.name for p in directory.iterdir() if p.is_dir()}
            if actual != SCENES:
                raise RuntimeError(f"Unexpected scene IDs in {directory.name}")

        height_counts = Counter()
        per_scene_counts = Counter()
        dimensions = Counter()
        satellite_count = 0
        for scene in sorted(SCENES):
            satellite_images = images(satellite / scene)
            if len(satellite_images) != 1:
                raise RuntimeError(f"Expected one satellite image for scene {scene}")
            satellite_count += len(satellite_images)
            if {p.name for p in (drone / scene).iterdir() if p.is_dir()} != set(
                HEIGHTS
            ):
                raise RuntimeError(f"Unexpected drone heights for scene {scene}")
            for height in HEIGHTS:
                drone_images = images(drone / scene / height)
                if not drone_images:
                    raise RuntimeError(f"No drone images: {scene}/{height}")
                height_counts[height] += len(drone_images)
                per_scene_counts[len(drone_images)] += 1
                # Validate one decoded image for every scene and altitude.
                with Image.open(drone_images[0]) as image:
                    image.load()
                    dimensions[str(image.size)] += 1
            with Image.open(satellite_images[0]) as image:
                image.load()
                dimensions[str(image.size)] += 1

        training = yaml.safe_load(
            (PROJECT / "star/dataset/splits/sues200.yaml").read_text()
        )["index"]
        if (
            len(training) != 120
            or len(set(training)) != 120
            or not set(training) < SCENES
        ):
            raise RuntimeError("Invalid existing STAR training split")
        testing = sorted(SCENES - set(training))
        drone_name = drone.name
        if payload != DESTINATION:
            payload.rename(DESTINATION)
        if staging != payload:
            # Keep any archive extras in staging rather than deleting them.
            if not any(staging.iterdir()):
                staging.rmdir()
        state.update(
            status="preparing_split",
            drone_directory=drone_name,
            satellite_images=satellite_count,
            drone_images_by_height=dict(height_counts),
            drone_images_per_scene_height=dict(per_scene_counts),
            sample_dimensions=dict(dimensions),
        )
        save(state)
        links = 0
        for height in HEIGHTS:
            for split, view, scene_ids in (
                ("train", "drone", training),
                ("train", "satellite", training),
                ("test", "query_drone", testing),
                ("test", "query_satellite", testing),
                ("test", "gallery_drone", sorted(SCENES)),
                ("test", "gallery_satellite", sorted(SCENES)),
            ):
                for scene in scene_ids:
                    source = (
                        DESTINATION / drone_name / scene / height
                        if "drone" in view
                        else DESTINATION / "satellite-view" / scene
                    )
                    link_directory(source, DESTINATION / split / height / view / scene)
                    links += 1
        if args.link.absolute() != DESTINATION:
            link_directory(DESTINATION, args.link.expanduser().absolute())
        state.update(
            status="ready",
            training_scenes=len(training),
            test_scenes=len(testing),
            gallery_scenes=len(SCENES),
            split_directory_links=links,
            completed_at=time.time(),
        )
        save(state)
        print(json.dumps(state, indent=2), flush=True)
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        save(state)
        raise


if __name__ == "__main__":
    main()
