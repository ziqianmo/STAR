"""VSAM cosine alignment loss for paired drone and satellite features."""

import torch.nn as nn
import torch.nn.functional as F


class VSAMLoss(nn.Module):
    """Cosine alignment; constructor/forward arguments retain caller compatibility."""

    def __init__(self, loss_function=None, device=None):
        super().__init__()

    def cosine_loss(self, pred, target):
        pred_norm = F.normalize(pred, dim=1)
        target_norm = F.normalize(target, dim=1)
        return 1 - (pred_norm * target_norm).sum() / pred.size(0)

    def forward(self, image_features1, image_features2, logit_scale=None):
        # Labels and temperature are unnecessary for cosine alignment.
        batch, channels, tokens = image_features1.shape
        features1 = image_features1.transpose(2, 1).reshape(batch, channels * tokens)
        features2 = image_features2.transpose(2, 1).reshape(batch, channels * tokens)
        return self.cosine_loss(features1, features2)
