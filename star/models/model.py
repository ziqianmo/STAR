import torch.nn as nn
from .convnext import make_convnext_model
import torch
import numpy as np


class two_view_net(nn.Module):
    def __init__(
        self, class_num, block=4, return_f=False, resnet=False, pretrained=True
    ):
        super(two_view_net, self).__init__()
        self.model_1 = make_convnext_model(
            num_class=class_num,
            block=block,
            return_f=return_f,
            resnet=resnet,
            pretrained=pretrained,
        )

        # 1. temperature factor for contrastive learning
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_blocks = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.ECE_weight_D_fine_D_fine = torch.nn.Parameter(torch.ones([]) * 0.5)

        # Legacy parameters retained for loading existing STAR checkpoints.
        self.w_blocks1 = torch.nn.Parameter(torch.ones([]))
        self.w_blocks2 = torch.nn.Parameter(torch.ones([]))
        self.w_blocks3 = torch.nn.Parameter(torch.ones([]))

    def get_config(self):
        input_size = (3, 224, 224)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        config = {"input_size": input_size, "mean": mean, "std": std}
        return config

    def forward(self, x1, x2=None):
        if x2 is not None:
            y1 = self.model_1(x1, seq_len=1)
            y2 = self.model_1(x2, seq_len=5)
            return y1, y2
        else:
            y1 = self.model_1(x1, seq_len=1)
            return y1


def make_model(opt):
    if opt.views != 2:
        raise ValueError("STAR uses paired satellite/drone views (views=2).")
    # Older saved experiment scripts used triplet_loss as a feature-return flag.
    return_features = getattr(
        opt, "return_features", getattr(opt, "triplet_loss", True)
    )
    return two_view_net(
        opt.nclasses,
        block=opt.block,
        return_f=bool(return_features),
        resnet=opt.resnet,
        pretrained=getattr(opt, "pretrained", True),
    )
