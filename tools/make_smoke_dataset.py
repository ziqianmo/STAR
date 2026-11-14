"""Create synthetic images for runtime checks, never for benchmark reporting."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("Output must be empty to avoid mixing real data with test images")
    rng = np.random.default_rng(42)
    for identity in range(12):
        base = rng.integers(0, 256, size=(384, 384, 3), dtype=np.uint8)
        for split, count in [
            ("train/satellite", 1),
            ("train/drone", 6),
            ("test/query_drone", 1),
            ("test/gallery_satellite", 1),
        ]:
            directory = root / split / f"{identity:04d}"
            directory.mkdir(parents=True, exist_ok=True)
            for frame in range(count):
                image = np.roll(base, frame * 3, axis=0)
                cv2.putText(
                    image,
                    f"SYNTHETIC {identity} {frame}",
                    (12, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                if not cv2.imwrite(str(directory / f"{frame:03d}.jpg"), image):
                    raise OSError("Could not write test image")
    (root / "SYNTHETIC_DATA_ONLY.txt").write_text(
        "Generated images for runtime validation. These are NOT University-1652 images.\n"
    )
    print("Created synthetic runtime dataset:", root)


if __name__ == "__main__":
    main()
