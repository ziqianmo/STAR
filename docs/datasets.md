# Dataset preparation

`--data-folder` points to the root containing `train/` and `test/`. Datasets can remain outside the repository; the loaders do not copy images.

## University-1652

```text
U1652/
├── train/
│   ├── satellite/<scene_id>/*.jpg
│   └── drone/<scene_id>/*.jpeg
└── test/
    ├── query_drone/<scene_id>/*.jpeg
    ├── gallery_satellite/<scene_id>/*.jpg
    ├── query_satellite/<scene_id>/*.jpg
    └── gallery_drone/<scene_id>/*.jpeg
```

Use the [official dataset repository](https://github.com/layumi/University1652-Baseline) for access instructions. The included downloader requires an approved account for the Hugging Face dataset:

```bash
python tools/login_huggingface.py
python tools/download_university1652.py --output /path/to/University-1652
```

The login tool requests a read token through hidden terminal input. Its local credentials are stored in `.runtime/huggingface/`, which is excluded from Git. Network proxies can be configured using environment variables.

Link an existing dataset into the default location:

```bash
mkdir -p data
ln -s /path/to/University-1652 data/U1652
```

Alternatively, use `--data-folder /path/to/University-1652` or set `UNIVERSITY1652_ROOT` before starting the Python process. Training maps matching scene IDs to contiguous classification labels; evaluation retains the original scene IDs.

## SUES-200

The [official V2 archive](https://github.com/Reza-Stu/SUES-200) contains:

```text
SUES-200-512x512/
├── drone_view_512/<0001..0200>/<150|200|250|300>/<frame>.jpg
└── satellite-view/<0001..0200>/0.png
```

Download and extract it with the included tools:

```bash
python tools/download_sues_archive.py --output /path/to/SUES-200-512x512-V2.zip
python tools/prepare_sues200.py \
  --archive /path/to/SUES-200-512x512-V2.zip \
  --destination /path/to/SUES-200-512x512
```

For an existing extraction, use:

```bash
python tools/prepare_sues200.py \
  --destination /path/to/SUES-200-512x512 --split-only
```

The preparation script preserves the raw directories and creates scene-level links:

```text
SUES-200-512x512/
├── train/<altitude>/
│   ├── satellite/<scene_id>/     # 120 training scenes
│   └── drone/<scene_id>/
└── test/<altitude>/
    ├── query_drone/<scene_id>/   # 80 query scenes
    ├── query_satellite/<scene_id>/
    ├── gallery_drone/<scene_id>/ # All 200 gallery scenes
    └── gallery_satellite/<scene_id>/
```

The fixed split in `star/dataset/splits/sues200.yaml` stores 120 training IDs as zero-padded strings. The remaining 80 IDs form the query split. All four altitudes are prepared separately. Retain this split when comparing with the existing experiments.

By default, preparation also creates `data/SUES-200`. Use `--link` to choose another link location, `--data-folder` at runtime, or the `SUES200_ROOT` environment variable. Matching links are reusable; conflicting links and occupied target directories produce an error while preserving existing content.

## Retrieval directions and frame ordering

- **D2S:** `query_drone` retrieves from `gallery_satellite`.
- **S2D:** `query_satellite` retrieves from `gallery_drone`.

Training always pairs satellite images and drone sequences. The selected direction controls validation and evaluation.

Both datasets share numeric natural sorting: `1.jpg`, `2.jpg`, `10.jpg`. The order of University-1652's zero-padded filenames is preserved. Unpadded SUES-200 frame names are read in numeric order without renaming the images.

Training uses five-frame windows with stride 1 within each scene. A scene needs at least five drone frames. Sequence-query evaluation uses the requested window length and averages the frame descriptors before final normalization.
