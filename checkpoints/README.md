# STAR model weights

The University-1652 checkpoint is available on Google Drive and listed in [`weights.json`](weights.json):

| Model ID | Training data | Local file | Download |
| --- | --- | --- | --- |
| `university1652` | University-1652 | `university1652/weights_e1_0.9535.pth` | [Google Drive](https://drive.google.com/file/d/1_DNSV1A3AkkqyTi9S1CDJMV8a_zT3toT/view?usp=drive_link) |

Paths in the table are relative to this directory. The manifest records the exact file size and SHA-256 of each checkpoint. Binary weights are distributed separately from the Git source tree.

## Download and verify

Inspect the entries:

```bash
python tools/download_weights.py --list
```

Download and verify:

```bash
python tools/download_weights.py --model university1652
```

The downloader supports Google Drive file-sharing links through `gdown`, as well as direct HTTP(S) downloads. It verifies the file size and SHA-256 before publishing a downloaded checkpoint and preserves an existing checkpoint if its contents differ. Google Drive downloads use public access without saved account cookies. The sharing setting must allow anyone with the link to download the file; quota and network restrictions may require retrying later or downloading manually.

A verified existing file is reused. To verify a manual download, run the same `--model` command. The downloader also supports `--url` for a single Google Drive file link or direct HTTP(S) link and `--output-dir` for an alternative checkpoints directory.

## Evaluation

From the repository root:

```bash
python eval_university.py \
  --checkpoint checkpoints/university1652/weights_e1_0.9535.pth \
  --direction D2S

python eval_sues200.py \
  --checkpoint /path/to/your_sues200_checkpoint.pth \
  --altitude 150 --direction D2S
```

This is the trained STAR University-1652 checkpoint. ImageNet initialization for training is controlled separately by `--pretrained` / `--no-pretrained`. Evaluation uses the selected STAR checkpoint and does not download a separate backbone.
