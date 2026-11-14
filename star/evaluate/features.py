"""Descriptor extraction shared by training-time and standalone evaluation."""

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from tqdm import tqdm


def encode_images(config, model, images, sequence=False):
    device = torch.device(config.device)
    if sequence:
        if isinstance(images, (list, tuple)):
            if not images:
                raise ValueError("A query sequence must contain at least one frame.")
            images = torch.stack(images, dim=1)
        if images.ndim != 5:
            raise ValueError("Sequence input must have shape [B, T, C, H, W].")
        batch, length, channels, height, width = images.shape
        if length == 0:
            raise ValueError("A query sequence must contain at least one frame.")
        images = images.reshape(batch * length, channels, height, width)
    elif images.ndim != 4:
        raise ValueError("Single-image input must have shape [B, C, H, W].")

    images = images.to(device, non_blocking=device.type == "cuda")
    mixed_precision = device.type == "cuda" and getattr(config, "mixed_precision", True)
    context = torch.autocast(device_type="cuda") if mixed_precision else nullcontext()
    with context:
        output = model(images)
        features = output[-2] if getattr(config, "handcraft_model", True) else output
    if features.ndim != 2 or features.size(0) != images.size(0):
        raise ValueError("The retrieval model must return one descriptor per image.")
    features = features.float()
    if sequence:
        features = features.reshape(batch, length, -1).mean(dim=1)
    if getattr(config, "normalize_features", True):
        features = F.normalize(features, dim=-1)
    return features


@torch.inference_mode()
def extract_features(config, model, dataloader, sequence=False):
    model.eval()
    feature_batches, id_batches = [], []
    batches = tqdm(
        dataloader, desc="Features", disable=not getattr(config, "verbose", True)
    )
    for images, sample_ids in batches:
        features = encode_images(config, model, images, sequence=sequence)
        ids = torch.as_tensor(
            sample_ids, dtype=torch.long, device=features.device
        ).reshape(-1)
        if len(ids) != len(features):
            raise ValueError("Descriptor and sample ID counts differ.")
        feature_batches.append(features)
        id_batches.append(ids)
    if not feature_batches:
        raise ValueError("The evaluation loader contains no samples.")
    return torch.cat(feature_batches), torch.cat(id_batches)
