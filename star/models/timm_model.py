import torch
import timm
import numpy as np
import torch.nn as nn


class TimmModel(nn.Module):

    def __init__(self, model_name, pretrained=True, img_size=384):

        super(TimmModel, self).__init__()

        self.img_size = img_size

        if "vit" in model_name:
            # 自动调整位置编码到给定 img_size
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size
            )
        else:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0
            )

        # 用于缩放 logits
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(self):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, query_img, gallery_imgs=None):
        """
        query_img: 单张卫星图像
        gallery_imgs: 无人机图像序列
        """
        if gallery_imgs is not None:
            # 对卫星图像进行特征提取
            query_features = self.model(query_img)

            # 对无人机图像序列进行特征提取并聚合（例如平均池化）
            batch_size, seq_len, c, h, w = gallery_imgs.shape
            gallery_imgs = gallery_imgs.view(
                batch_size * seq_len, c, h, w
            )  # 展平序列维度
            gallery_features = self.model(gallery_imgs)
            gallery_features = gallery_features.view(
                batch_size, seq_len, -1
            )  # 恢复序列维度

            # 聚合序列特征（这里使用平均池化作为示例）
            gallery_features = gallery_features.mean(dim=1)

            return query_features, gallery_features

        else:
            # 仅处理单张卫星图像
            query_features = self.model(query_img)
            return query_features
