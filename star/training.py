"""Shared training setup for the two STAR experiments.

Entry scripts own experiment defaults; command-line options override a copy.
The optimization step and loss definitions live in :mod:`star.trainer`.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from star.dataset.common import get_transforms
from star.evaluate.university import evaluate
from star.loss.info_nce import InfoNCE
from star.loss.vsam import VSAMLoss
from star.models.timm_model import TimmModel
from star.trainer import train
from star.utils import Logger, setup_system

PROJECT = Path(__file__).resolve().parents[1]


@dataclass
class TrainingConfig:
    # Backbone and STAR branches. Keep these fixed for comparable experiments.
    dataset_name: str = "university"
    model: str = "convnext_base.fb_in22k_ft_in1k_384"
    pretrained: bool = False
    img_size: int = 384
    handcraft_model: bool = True
    timm_use: bool = True
    views: int = 2
    nclasses: int = 701
    block: int = 2
    return_features: bool = True
    resnet: bool = False
    grad_checkpointing: bool = False

    # Training and optimizer defaults.
    mixed_precision: bool = True
    custom_sampling: bool = True
    seed: int = 1
    epochs: int = 1
    batch_size: int = 16
    verbose: bool = True
    gpu_ids: tuple = (0,)
    clip_grad: float | None = 100.0
    decay_exclue_bias: bool = False
    label_smoothing: float = 0.1
    lr: float = 0.001
    scheduler: str = "cosine"
    warmup_epochs: float = 0.1
    lr_end: float = 0.0001
    prob_flip: float = 0.5
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False

    # Data, validation and run artifacts.
    data_folder: str = ""
    direction: str = "D2S"
    altitude: int = 300
    batch_size_eval: int = 64
    eval_every_n_epoch: int = 1
    normalize_features: bool = True
    eval_gallery_n: int = -1
    zero_shot: bool = False
    model_path: str = ""
    checkpoint_start: str | None = None
    num_workers: int = 0 if os.name == "nt" else 4
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    max_train_batches: int = 0
    eval_max_images: int = 0

    def __post_init__(self):
        if self.dataset_name not in ("university", "sues200"):
            raise ValueError("Unsupported training dataset.")
        university = self.dataset_name == "university"
        variable = "UNIVERSITY1652_ROOT" if university else "SUES200_ROOT"
        if not self.data_folder:
            self.data_folder = os.environ.get(
                variable,
                str(PROJECT / "data" / ("U1652" if university else "SUES-200")),
            )
        if not self.model_path:
            self.model_path = str(PROJECT / "outputs" / self.dataset_name)

    def set_folders(self):
        root = Path(self.data_folder)
        train_root, test_root = root / "train", root / "test"
        if self.dataset_name == "sues200":
            train_root /= str(self.altitude)
            test_root /= str(self.altitude)
        self.query_folder_train = str(train_root / "satellite")
        self.gallery_folder_train = str(train_root / "drone")
        query, gallery = (
            ("query_drone", "gallery_satellite")
            if self.direction == "D2S"
            else ("query_satellite", "gallery_drone")
        )
        self.query_folder_test = str(test_root / query)
        self.gallery_folder_test = str(test_root / gallery)


def configure(default, argv=None):
    """Apply CLI options to a copy so repeated calls retain Python defaults."""
    config = replace(default)
    parser = argparse.ArgumentParser(description=f"Train STAR on {config.dataset_name}")
    parser.add_argument("--data-folder", "--data-root", default=config.data_folder)
    parser.add_argument("--direction", choices=("D2S", "S2D"), default=config.direction)
    if config.dataset_name == "university":
        parser.add_argument(
            "--dataset",
            choices=("U1652-D2S", "U1652-S2D"),
            default=None,
            help="Compatibility alias for --direction",
        )
    else:
        parser.add_argument(
            "--altitude",
            choices=(150, 200, 250, 300),
            type=int,
            default=config.altitude,
        )
    for name in (
        "batch_size",
        "batch_size_eval",
        "epochs",
        "num_workers",
        "seed",
        "max_train_batches",
        "eval_max_images",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(config, name)
        )
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--device", default=config.device)
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=config.pretrained
    )
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-start",
        dest="checkpoint_start",
        default=config.checkpoint_start,
        help="Initialize model weights; optimizer and epoch are reset",
    )
    parser.add_argument(
        "--output-dir", "--model-path", dest="model_path", default=config.model_path
    )
    parser.add_argument("--quiet", action="store_true", default=not config.verbose)
    args = vars(parser.parse_args(argv))
    legacy_dataset = args.pop("dataset", None)
    quiet = args.pop("quiet")
    for key, value in args.items():
        setattr(config, key, value)
    if legacy_dataset:
        if (
            any(
                option.split("=")[0] == "--direction"
                for option in (argv if argv is not None else sys.argv[1:])
            )
            and config.direction != legacy_dataset.split("-")[-1]
        ):
            parser.error("--dataset and --direction disagree")
        config.direction = legacy_dataset.split("-")[-1]
    config.verbose = not quiet
    if (
        config.batch_size < 2
        or min(config.batch_size_eval, config.epochs, config.eval_every_n_epoch) < 1
    ):
        parser.error(
            "batch-size must be >= 2; evaluation batch size, epochs and evaluation interval must be positive"
        )
    if (
        min(config.num_workers, config.max_train_batches, config.eval_max_images) < 0
        or config.lr <= 0
    ):
        parser.error("Worker and limit counts cannot be negative; lr must be positive")
    try:
        device = torch.device(config.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are cpu and cuda[:index].")
        if device.type == "cuda":
            index = device.index or 0
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {index} is unavailable.")
            config.device = f"cuda:{index}"
        config.gpu_ids = (device.index or 0,) if device.type == "cuda" else ()
        config.mixed_precision = config.mixed_precision and device.type == "cuda"
        config.data_folder = str(Path(config.data_folder).expanduser().resolve())
        config.model_path = str(Path(config.model_path).expanduser().resolve())
        config.set_folders()
        if config.checkpoint_start:
            config.checkpoint_start = str(
                Path(config.checkpoint_start).expanduser().resolve()
            )
            if not Path(config.checkpoint_start).is_file():
                raise ValueError(
                    "Checkpoint does not exist: " + config.checkpoint_start
                )
        for folder in (
            config.query_folder_train,
            config.gallery_folder_train,
            config.query_folder_test,
            config.gallery_folder_test,
        ):
            if not Path(folder).is_dir():
                raise ValueError("Dataset directory does not exist: " + folder)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    return config


def run_training(default, argv=None, entry_file=None):
    config = configure(default, argv)
    run = (
        Path(config.model_path)
        / config.model
        / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    run.mkdir(parents=True, exist_ok=False)
    if entry_file:
        shutil.copyfile(entry_file, run / Path(entry_file).name)
    settings = {**asdict(config), **vars(config), "run_directory": str(run)}
    (run / "config.json").write_text(json.dumps(settings, indent=2, default=str) + "\n")
    logger = Logger(str(run / "log.txt"))
    try:
        with redirect_stdout(logger):
            print("Run directory:", run)
            print(
                "Pretrained backbone:",
                config.pretrained,
                "- checkpoint initialization:",
                config.checkpoint_start,
            )
            if config.max_train_batches or config.eval_max_images:
                print(
                    "Limited run: losses and retrieval scores are for a smoke check only."
                )
            _train_experiment(config, str(run))
    finally:
        logger.close()
    return run


def _train_experiment(config, model_path):
    if config.dataset_name == "university":
        from star.dataset.university import (
            U1652DatasetTrainSequence as TrainDataset,
            U1652DatasetEval as EvalDataset,
        )
    else:
        from star.dataset.sues200 import (
            SUES200DatasetTrainSequence as TrainDataset,
            SUES200DatasetEval as EvalDataset,
        )

    setup_system(
        seed=config.seed,
        cudnn_benchmark=config.cudnn_benchmark,
        cudnn_deterministic=config.cudnn_deterministic,
    )

    # Model.

    if config.handcraft_model is not True:
        print("\nModel: {}".format(config.model))
        model = TimmModel(
            config.model, pretrained=config.pretrained, img_size=config.img_size
        )
    else:
        from star.models import make_model

        model = make_model(config)
        print("\nModel:{}".format("adjust model: handcraft convnext-base"))

    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # Load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # Dataloader.

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(
        img_size, mean=mean, std=std
    )

    train_dataset = TrainDataset(
        query_folder=config.query_folder_train,
        gallery_folder=config.gallery_folder_train,
        transforms_query=train_sat_transforms,
        transforms_gallery=train_drone_transforms,
        prob_flip=config.prob_flip,
        shuffle_batch_size=config.batch_size,
        sequence_length=5,
    )
    if len(train_dataset) == 0:
        raise ValueError(
            "No training sequences found: each matching drone class needs at least 5 images."
        )
    if len(train_dataset.ids) < config.batch_size and config.custom_sampling:
        raise ValueError(
            "Custom sampling needs at least batch_size distinct training classes."
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=not config.custom_sampling,
        pin_memory=True,
    )

    # Reference Satellite Images
    query_dataset_test = EvalDataset(
        data_folder=config.query_folder_test,
        mode="query",
        transforms=val_transforms,
    )

    query_ids = query_dataset_test.get_sample_ids()
    if config.eval_max_images:
        query_dataset_test = torch.utils.data.Subset(
            query_dataset_test,
            range(min(config.eval_max_images, len(query_dataset_test))),
        )
    query_dataloader_test = DataLoader(
        query_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # Query Ground Images Test
    gallery_dataset_test = EvalDataset(
        data_folder=config.gallery_folder_test,
        mode="gallery",
        transforms=val_transforms,
        sample_ids=query_ids,
        gallery_n=config.eval_gallery_n,
    )
    if config.eval_max_images:
        gallery_dataset_test = torch.utils.data.Subset(
            gallery_dataset_test,
            range(min(config.eval_max_images, len(gallery_dataset_test))),
        )
    if not len(query_dataset_test) or not len(gallery_dataset_test):
        raise ValueError("Evaluation folders contain no images.")

    gallery_dataloader_test = DataLoader(
        gallery_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    # Loss.

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_fn1 = InfoNCE(
        loss_function=loss_fn,
        device=config.device,
    )

    loss_fn2 = VSAMLoss(loss_function=loss_fn, device=config.device)

    loss_functions = {"infoNCE": loss_fn1, "VSAM_loss": loss_fn2}

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.0**10)
    else:
        scaler = None

    # Optimizer.

    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Scheduler.

    steps_per_epoch = (
        min(len(train_dataloader), config.max_train_batches)
        if config.max_train_batches
        else len(train_dataloader)
    )
    train_steps = steps_per_epoch * config.epochs
    # Preserve each experiment's original warmup convention.
    warmup_steps = steps_per_epoch * config.warmup_epochs
    if config.dataset_name == "university":
        warmup_steps = int(warmup_steps)

    if config.scheduler == "polynomial":
        print(
            "\nScheduler: polynomial - max LR: {} - end LR: {}".format(
                config.lr, config.lr_end
            )
        )
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            lr_end=config.lr_end,
            power=1.5,
            num_warmup_steps=warmup_steps,
        )

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=train_steps, num_warmup_steps=warmup_steps
        )

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps
        )

    else:
        scheduler = None

    print(
        "Warmup Epochs: {} - Warmup Steps: {}".format(
            str(config.warmup_epochs).ljust(2), warmup_steps
        )
    )
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    # Zero shot.
    if config.zero_shot:
        print("\n{}[{}]{}".format(30 * "-", "Zero Shot", 30 * "-"))

        r1_test = evaluate(
            config=config,
            model=model,
            query_loader=query_dataloader_test,
            gallery_loader=gallery_dataloader_test,
            ranks=[1, 5, 10],
            step_size=1000,
            cleanup=True,
        )

    # Shuffle.
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()

    # Train.
    best_score = 0

    for epoch in range(1, config.epochs + 1):

        print("\n{}[Epoch: {}]{}".format(30 * "-", epoch, 30 * "-"))

        train_loss = train(
            config,
            model,
            dataloader=train_dataloader,
            loss_functions=loss_functions,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        print(
            "Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(
                epoch, train_loss, optimizer.param_groups[0]["lr"]
            )
        )

        # evaluate
        if (
            epoch % config.eval_every_n_epoch == 0 and epoch != 0
        ) or epoch == config.epochs:

            print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))

            r1_test = evaluate(
                config=config,
                model=model,
                query_loader=query_dataloader_test,
                gallery_loader=gallery_dataloader_test,
                ranks=[1, 5, 10],
                step_size=1000,
                cleanup=True,
            )

            if r1_test > best_score:

                best_score = r1_test

                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(
                        model.module.state_dict(),
                        "{}/weights_e{}_{:.4f}.pth".format(model_path, epoch, r1_test),
                    )
                else:
                    torch.save(
                        model.state_dict(),
                        "{}/weights_e{}_{:.4f}.pth".format(model_path, epoch, r1_test),
                    )

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), "{}/weights_end.pth".format(model_path))
    else:
        torch.save(model.state_dict(), "{}/weights_end.pth".format(model_path))
