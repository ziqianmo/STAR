"""Shared command-line evaluation for University-1652 and SUES-200."""

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from .checkpoints import load_checkpoint
from .features import encode_images
from .metrics import evaluate_retrieval

PROJECT = Path(__file__).resolve().parents[2]


@dataclass
class EvaluationConfig:
    dataset_name: str = "university"
    sequence_query: bool = False
    data_folder: str = ""
    checkpoint_start: str = ""
    direction: str = "D2S"
    altitude: int = 300
    sequence_length: int = 5
    img_size: int = 384
    batch_size: int = 0
    num_workers: int = 0 if os.name == "nt" else 4
    device: str = ""
    verbose: bool = True
    normalize_features: bool = True
    mixed_precision: bool = True
    handcraft_model: bool = True
    pretrained: bool = False
    model: str = "convnext_base"
    nclasses: int = 0
    views: int = 2
    block: int = 2
    return_features: bool = True
    resnet: bool = False
    max_queries: int = 0
    max_gallery: int = 0
    step_size: int = 1000
    output: str = ""
    benchmark_iterations: int = 0

    def __post_init__(self):
        if self.dataset_name not in ("university", "sues200"):
            raise ValueError("Unsupported evaluation dataset.")
        university = self.dataset_name == "university"
        variable = "UNIVERSITY1652_ROOT" if university else "SUES200_ROOT"
        if not self.data_folder:
            self.data_folder = os.environ.get(
                variable,
                str(PROJECT / "data" / ("U1652" if university else "SUES-200")),
            )
        if not self.checkpoint_start:
            self.checkpoint_start = os.environ.get("STAR_CHECKPOINT", "")
        if not self.device:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if not self.batch_size:
            self.batch_size = 8 if self.sequence_query else 64
        if not self.nclasses:
            self.nclasses = 701 if university else 200
        self.set_folders()

    def set_folders(self):
        root = Path(self.data_folder)
        train = root / "train"
        test = root / "test"
        if self.dataset_name == "sues200":
            train, test = train / str(self.altitude), test / str(self.altitude)
        self.query_folder_train = str(train / "satellite")
        self.gallery_folder_train = str(train / "drone")
        query, gallery = (
            ("query_drone", "gallery_satellite")
            if self.direction == "D2S"
            else ("query_satellite", "gallery_drone")
        )
        self.query_folder_test = str(test / query)
        self.gallery_folder_test = str(test / gallery)


def configure(default, argv=None):
    config = replace(default)
    parser = argparse.ArgumentParser(
        description=f'Evaluate STAR on {config.dataset_name} ({"sequence" if config.sequence_query else "single image"})'
    )
    parser.add_argument("--data-folder", "--data-root", default=config.data_folder)
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-start",
        dest="checkpoint_start",
        required=not bool(config.checkpoint_start),
        default=config.checkpoint_start,
    )
    parser.add_argument(
        "--direction",
        choices=("D2S",) if config.sequence_query else ("D2S", "S2D"),
        default=config.direction,
    )
    if config.dataset_name == "sues200":
        parser.add_argument(
            "--altitude",
            type=int,
            choices=(150, 200, 250, 300),
            default=config.altitude,
        )
    if config.sequence_query:
        parser.add_argument(
            "--sequence-length", type=int, default=config.sequence_length
        )
    parser.add_argument("--device", default=config.device)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    parser.add_argument("--img-size", type=int, default=config.img_size)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=config.max_queries,
        help="0 evaluates all queries; a positive value is a limited check",
    )
    parser.add_argument(
        "--max-gallery",
        type=int,
        default=config.max_gallery,
        help="0 uses the full gallery; a positive value is a limited check",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=config.step_size,
        help="Query chunk size for similarity computation",
    )
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=config.mixed_precision,
    )
    parser.add_argument("--quiet", action="store_true", default=not config.verbose)
    parser.add_argument(
        "--output", default=config.output, help="Optional JSON result file"
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=config.benchmark_iterations,
        help="0 disables the optional inference timing check",
    )
    arguments = parser.parse_args(argv)
    for name, value in vars(arguments).items():
        if name != "quiet":
            setattr(config, name, value)
    config.verbose = not arguments.quiet
    if (
        min(config.batch_size, config.step_size, config.sequence_length) < 1
        or config.img_size < 32
    ):
        parser.error(
            "Batch size, step size and sequence length must be positive; image size must be at least 32."
        )
    if (
        min(
            config.num_workers,
            config.max_queries,
            config.max_gallery,
            config.benchmark_iterations,
        )
        < 0
    ):
        parser.error("Worker, sample-limit and benchmark counts must not be negative.")
    try:
        device = torch.device(config.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are cpu and cuda[:index].")
        if device.type == "cuda":
            index = device.index if device.index is not None else 0
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device {index} is unavailable; found {torch.cuda.device_count()} CUDA devices."
                )
            config.device = f"cuda:{index}"
        config.gpu_ids = (device.index or 0,) if device.type == "cuda" else ()
        config.data_folder = str(Path(config.data_folder).expanduser().resolve())
        config.checkpoint_start = str(
            Path(config.checkpoint_start).expanduser().resolve()
        )
        config.set_folders()
        if not Path(config.checkpoint_start).is_file():
            raise ValueError("Checkpoint does not exist: " + config.checkpoint_start)
        for folder in (config.query_folder_test, config.gallery_folder_test):
            if not Path(folder).is_dir():
                raise ValueError("Evaluation data directory does not exist: " + folder)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    return config


def build_loaders(config, transforms):
    if config.dataset_name == "university":
        from star.dataset.university import U1652DatasetEval as ImageDataset
        from star.dataset.university import (
            U1652DatasetEvalSequenceD2S as SequenceDataset,
        )
    else:
        from star.dataset.sues200 import SUES200DatasetEval as ImageDataset
        from star.dataset.sues200 import (
            SUES200DatasetEvalSequenceD2S as SequenceDataset,
        )
    kwargs = {
        "data_folder": config.query_folder_test,
        "mode": "query",
        "transforms": transforms,
    }
    query = (
        SequenceDataset(**kwargs, sequence_length=config.sequence_length)
        if config.sequence_query
        else ImageDataset(**kwargs)
    )
    gallery = ImageDataset(
        config.gallery_folder_test,
        mode="gallery",
        transforms=transforms,
        sample_ids=query.get_sample_ids(),
    )
    if not len(query) or not len(gallery):
        raise ValueError(
            "Evaluation query and gallery datasets must both contain samples."
        )
    query_indices = list(range(min(config.max_queries or len(query), len(query))))
    gallery_indices = list(range(len(gallery)))
    if config.max_gallery and config.max_gallery < len(gallery):
        # Retain one positive per selected query identity before filling a debug gallery.
        wanted = {query.sample_ids[i] for i in query_indices}
        chosen, covered = [], set()
        for i, scene in enumerate(gallery.sample_ids):
            if scene in wanted and scene not in covered:
                chosen.append(i)
                covered.add(scene)
                if len(chosen) == config.max_gallery:
                    break
        selected = set(chosen)
        for i in range(len(gallery)):
            if len(chosen) == config.max_gallery:
                break
            if i not in selected:
                chosen.append(i)
        gallery_indices = sorted(chosen)
    query_view, gallery_view = Subset(query, query_indices), Subset(
        gallery, gallery_indices
    )
    options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "shuffle": False,
        "pin_memory": torch.device(config.device).type == "cuda",
    }
    return DataLoader(query_view, **options), DataLoader(gallery_view, **options)


def benchmark(config, model):
    shape = (
        (1, config.sequence_length, 3, config.img_size, config.img_size)
        if config.sequence_query
        else (1, 3, config.img_size, config.img_size)
    )
    images = torch.randn(shape, device=config.device)
    times = []
    model.eval()
    with torch.inference_mode():
        for index in range(10 + config.benchmark_iterations):
            if torch.device(config.device).type == "cuda":
                torch.cuda.synchronize(config.device)
            start = time.perf_counter()
            encode_images(config, model, images, sequence=config.sequence_query)
            if torch.device(config.device).type == "cuda":
                torch.cuda.synchronize(config.device)
            if index >= 10:
                times.append(time.perf_counter() - start)
    return 1000 * sum(times) / len(times)


def run_evaluation(default, argv=None):
    config = configure(default, argv)
    from star.models import make_model
    from star.dataset.common import get_transforms

    # Checkpoint weights provide initialization; no additional backbone download.
    model = make_model(config)
    loaded = load_checkpoint(model, config.checkpoint_start)
    model = model.to(config.device)
    data_config = model.get_config()
    transforms, _, _ = get_transforms(
        (config.img_size, config.img_size), data_config["mean"], data_config["std"]
    )
    query_loader, gallery_loader = build_loaders(config, transforms)
    print(
        f"Dataset: {config.dataset_name}; direction: {config.direction}; sequence query: {config.sequence_query}",
        flush=True,
    )
    print("Query:", config.query_folder_test, flush=True)
    print("Gallery:", config.gallery_folder_test, flush=True)
    print(
        f'Checkpoint: {config.checkpoint_start}; loaded {loaded["loaded_keys"]} keys, complete backbone',
        flush=True,
    )
    print(
        f"Device: {config.device}; query samples: {len(query_loader.dataset)}; gallery samples: {len(gallery_loader.dataset)}",
        flush=True,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}", flush=True)
    metrics = evaluate_retrieval(
        config,
        model,
        query_loader,
        gallery_loader,
        step_size=config.step_size,
        query_sequence=config.sequence_query,
        return_metrics=True,
    )
    result = {
        "dataset": config.dataset_name,
        "direction": config.direction,
        "query_mode": "sequence" if config.sequence_query else "single",
        "altitude": config.altitude if config.dataset_name == "sues200" else None,
        "sequence_length": config.sequence_length if config.sequence_query else 1,
        "data_folder": config.data_folder,
        "checkpoint": config.checkpoint_start,
        "device": config.device,
        "limited_evaluation": bool(config.max_queries or config.max_gallery),
        "model_parameters": parameter_count,
        "checkpoint_loading": loaded,
        "metrics": metrics,
    }
    if config.benchmark_iterations:
        result["mean_inference_ms_per_query"] = benchmark(config, model)
        print(
            f'Mean descriptor inference time: {result["mean_inference_ms_per_query"]:.3f} ms/query',
            flush=True,
        )
    if config.output:
        destination = Path(config.output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("Results:", destination, flush=True)
    return result
