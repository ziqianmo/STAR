import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model

# Import registers feature-returning ConvNeXt backbones with timm.
from .backbones.model_convnext import convnext_tiny
from .backbones.resnet import Resnet
from star import initialization as init


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        return x_out11, x_out21


class ClassBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        class_num,
        droprate,
        relu=False,
        bnorm=True,
        num_bottleneck=512,
        linear=True,
        return_f=False,
    ):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)

    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class MLP1D(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """

    def __init__(
        self,
        in_channels,
        hid_channels,
        out_channels,
        norm_layer=None,
        bias=False,
        num_mlp=2,
    ):
        super(MLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp - 1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def init_weights(self, init_linear="kaiming"):  # origin is 'normal'
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x


class MLP1D2(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """

    def __init__(
        self,
        in_channels,
        hid_channels1,
        hid_channels2,
        out_channels,
        norm_layer=None,
        bias=False,
        num_mlp=2,
    ):
        super(MLP1D2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp):
            mlps.append(nn.Conv1d(in_channels, hid_channels1, 1, bias=bias))
            mlps.append(norm_layer(hid_channels1))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels1
            hid_channels1 = hid_channels2
        mlps.append(nn.Conv1d(hid_channels2, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def init_weights(self, init_linear="kaiming"):  # origin is 'normal'
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x


from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class VSAM(nn.Module):
    def __init__(self, dim, depth, heads, in_channels=1024, patch_size=4):
        super().__init__()

        self.dim = dim
        self.in_channels = in_channels
        self.patch_size = patch_size

        patch_dim = in_channels * patch_size**2
        # 空间 Transformer
        self.space_transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=64, mlp_dim=dim * 4
        )
        # 时间 Transformer
        self.temporal_transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=64, mlp_dim=dim * 4
        )
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.Linear(patch_dim, dim),
        )

        # 伪时间补偿的可学习偏移 token（形状对齐为 [1,1,1,dim]，广播到 [b,1,n,d]）
        self.static_token = nn.Parameter(torch.randn(1, 1, 1, dim))

        self.to_output = nn.Linear(dim, in_channels)  # 可选：恢复通道数

    def forward(self, x, t):
        # 输入 x: [b, c, h, w]
        batch, c, h, w = x.shape
        # batch, c, n = x.shape
        b = batch // t
        x = x.view(b, t, c, h, w)
        x = self.to_patch_embedding(x)
        b, t_cur, n, d = x.shape
        assert t_cur == t and d == self.dim

        # 空间 Transformer
        x_sp = rearrange(x, "b t n d -> (b t) n d")  # 合并 batch 和时间维度
        x_sp = self.space_transformer(x_sp)
        x_sp = rearrange(x_sp, "(b t) n d -> b t n d", b=b)

        # 伪时间补偿（t==1时使用）
        if t == 1:
            delta = self.static_token.expand(b, 1, n, d)  # [b,1,n,d]（广播）
            x_tmp = torch.cat([x_sp, x_sp + delta], dim=1)  # [b, 2, n, d]
            t_eff = 2
        else:
            x_tmp = x_sp
            t_eff = t

        # ---------- 时间 Transformer ----------
        x_tmp = rearrange(
            x_tmp, "b t n d -> (b n) t d"
        )  # 注意：以 patch 为 batch 单位做时序
        x_tmp = self.temporal_transformer(x_tmp)
        x_tmp = rearrange(x_tmp, "(b n) t d -> b t n d", b=b)

        if t == 1:
            x_tmp = x_tmp.mean(dim=1, keepdim=True)  # [b,1,n,d]

        # ---------- 映射回通道并还原形状 ----------
        x_out = self.to_output(x_tmp)  # [b, t, n, c]
        x_out = rearrange(x_out, "b t n c -> (b t) c n")  # (b*t, c, n)
        return x_out


class DBPM(nn.Module):
    def __init__(self, block, pool="avg", num_rings=6):
        super().__init__()
        if num_rings < 3:
            raise ValueError(
                "DBPM needs at least three rings to select an interior boundary."
            )
        if pool not in ("avg", "max"):
            raise ValueError("DBPM pool must be avg or max.")
        self.block = block
        self.num_rings = num_rings

        if pool == "avg":
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == "max":
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))

    def channel_pool(self, x):
        """对通道维度进行最大池化"""
        return torch.max(x, dim=1, keepdim=True)[0]  # 保持维度

    def get_ring_features(self, x):
        """按照不同半径从中心提取特征"""
        B, C, H, W = x.size()
        c_h, c_w = H // 2, W // 2

        # 先对通道维度进行池化
        x_pooled = self.channel_pool(x)  # [B, 1, H, W]

        # 生成从1到6的半径
        radii = torch.arange(1, self.num_rings + 1)

        # 提取每个环的特征（使用通道池化后的特征）
        ring_features = []
        for i, r in enumerate(radii):
            r = int(r)
            x_curr = x_pooled[:, :, (c_h - r) : (c_h + r), (c_w - r) : (c_w + r)]

            if i > 0:  # 对于非第一个区域，需要减去前一个区域
                prev_r = int(radii[i - 1])
                x_pre = x_pooled[
                    :,
                    :,
                    (c_h - prev_r) : (c_h + prev_r),
                    (c_w - prev_r) : (c_w + prev_r),
                ]

                pad_h = r - prev_r
                pad_w = r - prev_r
                x_pad = F.pad(x_pre, (pad_w, pad_w, pad_h, pad_h), "constant", 0)
                x_curr = x_curr - x_pad

            # 池化特征
            pooled = self.pooling(x_curr)  # [B, 1, 1, 1]

            # 添加位置编码作为额外特征
            pos_encoding = torch.ones_like(pooled) * (r / self.num_rings)
            enhanced_feat = torch.cat([pooled, pos_encoding], dim=1)  # [B, 2, 1, 1]
            ring_features.append(enhanced_feat)

        return ring_features, radii

    @torch.no_grad()
    def select_boundary(self, ring_features, radii):
        """Apply paper Eqs. (7)-(9), selecting one radius per image."""
        features = torch.stack(ring_features, dim=1).flatten(start_dim=2)
        if features.dtype in (torch.float16, torch.bfloat16):
            features = features.float()

        scores = []
        for i in range(1, features.size(1) - 1):
            # Zero-based i corresponds to paper ring r=i+1. Include ring 1.
            current = features[:, i]
            inner = features[:, :i].mean(dim=1)
            outer = features[:, i + 1 :].mean(dim=1)
            inner_distance = torch.linalg.vector_norm(current - inner, dim=1)
            outer_distance = torch.linalg.vector_norm(current - outer, dim=1)
            scores.append((inner_distance - outer_distance).abs())

        # Score 0 belongs to ring index 1 (radius 2), not ring index 0.
        boundary_indices = torch.stack(scores, dim=1).argmax(dim=1) + 1
        return radii.to(features.device)[boundary_indices]

    def forward(self, x, no_overlap=True):
        if self.block != 2:
            raise ValueError("DBPM implements two foreground/background branches.")
        B, C, H, W = x.size()
        if min(H, W) < 2 * self.num_rings:
            raise ValueError("Feature maps must contain all configured DBPM rings.")
        c_h, c_w = H // 2, W // 2
        with torch.no_grad():
            ring_features, radii = self.get_ring_features(x)
            selected_radii = self.select_boundary(ring_features, radii)

        # Per-image masks avoid synchronizing selected radii back to the CPU.
        rows = torch.arange(H, device=x.device).view(1, 1, H, 1)
        cols = torch.arange(W, device=x.device).view(1, 1, 1, W)
        radius = selected_radii.view(B, 1, 1, 1)
        foreground_mask = (
            (rows >= c_h - radius)
            & (rows < c_h + radius)
            & (cols >= c_w - radius)
            & (cols < c_w + radius)
        )
        outer_radius = self.num_rings
        outer_mask = (
            (rows >= c_h - outer_radius)
            & (rows < c_h + outer_radius)
            & (cols >= c_w - outer_radius)
            & (cols < c_w + outer_radius)
        )
        background_mask = outer_mask & ~foreground_mask if no_overlap else outer_mask

        if isinstance(self.pooling, nn.AdaptiveAvgPool2d):
            foreground = (x * foreground_mask).sum(dim=(2, 3), keepdim=True)
            foreground = foreground / foreground_mask.sum(dim=(2, 3), keepdim=True)
            background = (x * background_mask).sum(dim=(2, 3), keepdim=True)
            # Preserve the existing zero-masked outer-region pooling convention.
            background = background / outer_mask.sum(dim=(2, 3), keepdim=True)
        else:
            foreground = x.masked_fill(~foreground_mask, -torch.inf).amax(
                dim=(2, 3), keepdim=True
            )
            background_values = x.masked_fill(~background_mask, 0)
            background = background_values.masked_fill(~outer_mask, -torch.inf).amax(
                dim=(2, 3), keepdim=True
            )

        return torch.cat((foreground, background), dim=2)


class build_convnext(nn.Module):
    def __init__(
        self, num_classes, block=4, return_f=False, resnet=False, pretrained=True
    ):
        super(build_convnext, self).__init__()
        self.return_f = return_f
        if resnet:
            convnext_name = "resnet101"
            print("using model_type: {} as a backbone".format(convnext_name))
            self.in_planes = 2048
            self.convnext = Resnet(pretrained=pretrained)
        else:
            convnext_name = "convnext_base"
            print("using model_type: {} as a backbone".format(convnext_name))
            if "base" in convnext_name:
                self.in_planes = 1024
            elif "large" in convnext_name:
                self.in_planes = 1536
            elif "xlarge" in convnext_name:
                self.in_planes = 2048
            else:
                self.in_planes = 768
            self.convnext = create_model(convnext_name, pretrained=pretrained)

        self.num_classes = num_classes
        self.classifier1 = ClassBlock(
            self.in_planes, num_classes, 0.5, return_f=return_f
        )
        self.block = block
        # Keep legacy module keys compatible with existing checkpoints.
        self.tri_layer = TripletAttention()
        for i in range(self.block):
            name = "classifier_heat" + str(i + 1)
            setattr(
                self,
                name,
                ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f),
            )

        # define for Domain Space Alignment Module
        in_channels = 1024
        hid_channels = 2048
        out_channels = 32  # 32
        hid_channels1 = 512
        hid_channels2 = 256
        self.out_dim = out_channels
        norm_layer = None
        num_layers = 2
        self.proj = MLP1D(
            in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers
        )
        self.proj.init_weights()
        self.proj_obj = MLP1D(
            in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers
        )
        self.proj_obj.init_weights()
        self.proj2 = MLP1D2(
            in_channels,
            hid_channels1,
            hid_channels2,
            out_channels,
            norm_layer,
            num_mlp=num_layers,
        )
        self.proj2.init_weights()
        self.scale = 1.0
        self.l2_norm = True
        self.vsam = VSAM(
            dim=self.out_dim, depth=4, heads=4, in_channels=self.out_dim, patch_size=1
        )
        self.dbpm = DBPM(block=self.block)

    def forward(self, x, seq_len):
        # -- backbone feature extractor
        gap_feature, part_features = self.convnext(x)
        b, c, h, w = part_features.shape

        # -- Training
        if self.training:

            pfeat = part_features.flatten(2)
            # W = self.proj(pfeat)
            W = self.proj2(pfeat)
            W = W.view(b, self.out_dim, h, w)
            pfeat_align = self.vsam(W, seq_len)
            pfeat_align = torch.cat([pfeat, pfeat_align], dim=1)

            convnext_feature = self.classifier1(
                gap_feature
            )  # class: (bs, 701); feature: (bs, 512)

            part_features_ = self.dbpm(part_features)

            part_features_ = part_features_.view(
                part_features_.size(0), part_features_.size(1), -1
            )
            y = self.part_classifier(
                self.block, part_features_, cls_name="classifier_heat"
            )

            # y = [convnext_feature] + [convnext_amips_feature]
            y = y + [convnext_feature]
            if (
                self.return_f
            ):  # Return logits and classification features during training.
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return pfeat_align, cls, features, gap_feature, part_features

        # -- Eval
        else:
            # ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            # y = torch.cat([y, ffeature], dim=2)
            pass

        return gap_feature, part_features

    def part_classifier(self, block, x, cls_name="classifier_heat"):
        part = {}
        predict = {}

        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y


def make_convnext_model(
    num_class, block=4, return_f=False, resnet=False, pretrained=True
):
    print("===========building convnext===========")
    model = build_convnext(
        num_class, block=block, return_f=return_f, resnet=resnet, pretrained=pretrained
    )
    return model
