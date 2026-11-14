# STAR model weights

Two checkpoint entries are reserved in [`weights.json`](weights.json):

| Model ID | Training data | Local file | Download |
| --- | --- | --- | --- |
| `university1652` | University-1652 | `university1652/weights_e1_0.9535.pth` | Link to be added |
| `sues200_150m` | SUES-200, 150 m | `sues200_150m/weights_e1_0.9728.pth` | Link to be added |

Paths in the table are relative to this directory. The manifest records the exact file size and SHA-256 of each checkpoint. Binary weights are distributed separately from the Git source tree.

## Download and verify

Inspect the entries:

```bash
python tools/download_weights.py --list
```

Once the download links have been added:

```bash
python tools/download_weights.py --model university1652
python tools/download_weights.py --model sues200_150m
# Or download both:
python tools/download_weights.py --model all
```

The downloader accepts direct HTTP(S) file links, verifies the size and SHA-256, and preserves an existing checkpoint if its contents differ. A missing release link produces an explicit message; it does not start a download. Browser sharing pages can be used to download files manually to the paths above.

A verified existing file is reused. To verify a manual download, run the same `--model` command. The downloader also supports `--url` for a single direct link and `--output-dir` for an alternative checkpoints directory.

## Evaluation

From the repository root:

```bash
python eval_university.py \
  --checkpoint checkpoints/university1652/weights_e1_0.9535.pth \
  --direction D2S

python eval_sues200.py \
  --checkpoint checkpoints/sues200_150m/weights_e1_0.9728.pth \
  --altitude 150 --direction D2S
```

These are trained STAR weights. ImageNet initialization for training is controlled separately by `--pretrained` / `--no-pretrained`. Evaluation uses the selected STAR checkpoint and does not download a separate backbone.
