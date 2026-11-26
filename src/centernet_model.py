import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
)
from torchvision.ops import deform_conv2d


BN_MOMENTUM = 0.1


# ==============================================================================
# 1. Modulated Deformable Convolution v2 & Initialization Helpers
# ==============================================================================

class ModulatedDeformConv2d(nn.Module):
    """
    Pure PyTorch/torchvision drop-in equivalent of DCNv2 (DCN class in
    xingyizhou/CenterNet/src/lib/models/networks/DCNv2/dcn_v2.py).
    Uses native torchvision.ops.deform_conv2d with learned 2D offsets + modulation masks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        deformable_groups: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            bias=True,
        )
        self.init_offset()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        dtype = x.dtype
        return deform_conv2d(
            x,
            offset.to(dtype),
            self.weight.to(dtype),
            self.bias.to(dtype),
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation),
            mask=mask.to(dtype),
        )


def fill_up_weights(up: nn.ConvTranspose2d):
    """Bilinear upsampling weight initialization (Zhou et al. resnet_dcn.py)."""
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for ch in range(1, w.size(0)):
        w[ch, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers: nn.Module):
    """Head conv initialization: N(0, 0.001) weights, zero bias (Zhou et al. resnet_dcn.py)."""
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ==============================================================================
# 2. CenterNet ResNet Backbone + Deconv/DCN Upsampler + Prediction Heads
# ==============================================================================

_RESNET_BUILDERS = {
    18: (resnet18, ResNet18_Weights.DEFAULT, 512),
    34: (resnet34, ResNet34_Weights.DEFAULT, 512),
    50: (resnet50, ResNet50_Weights.DEFAULT, 2048),
    101: (resnet101, ResNet101_Weights.DEFAULT, 2048),
}


class CenterNetResNet(nn.Module):
    """
    CenterNet (Objects as Points, Zhou et al. 2019) ResNet encoder-decoder:
      - Stride 32 ResNet backbone (18/34/50/101), optional ImageNet-1K pretrain
      - 3x upsampling stages to stride 4:
          * use_dcn=True  -> DCNv2 + Bilinear-init Deconv [256, 128, 64] (paper default)
          * use_dcn=False -> Standard Deconv [256, 256, 256] (msra_resnet.py)
      - Multi-task heads: 'hm' [C], 'wh' [2], 'reg' [2]
    """

    def __init__(
        self,
        num_layers: int = 50,
        heads: Optional[Dict[str, int]] = None,
        head_conv: int = 64,
        pretrained_backbone: bool = True,
        use_dcn: bool = True,
    ):
        super().__init__()
        if num_layers not in _RESNET_BUILDERS:
            raise ValueError(f"Unsupported ResNet depth: {num_layers}. Choose from {list(_RESNET_BUILDERS.keys())}.")

        if heads is None:
            heads = {"hm": 20, "wh": 2, "reg": 2}
        self.heads = heads
        self.use_dcn = use_dcn
        self.deconv_with_bias = False

        builder, weights_enum, backbone_out_channels = _RESNET_BUILDERS[num_layers]
        backbone = builder(weights=weights_enum if pretrained_backbone else None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.inplanes = backbone_out_channels
        if self.use_dcn:
            self.deconv_layers = self._make_deconv_layer_dcn(3, [256, 128, 64], [4, 4, 4])
            feat_dim = 64
        else:
            self.deconv_layers = self._make_deconv_layer_std(3, [256, 256, 256], [4, 4, 4])
            feat_dim = 256

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(feat_dim, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0, bias=True),
                )
                if "hm" in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(feat_dim, num_output, kernel_size=1, stride=1, padding=0, bias=True)
                if "hm" in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel: int) -> Tuple[int, int, int]:
        if deconv_kernel == 4:
            return 4, 1, 0
        elif deconv_kernel == 3:
            return 3, 1, 1
        elif deconv_kernel == 2:
            return 2, 0, 0
        raise ValueError(f"Unsupported deconv kernel size: {deconv_kernel}")

    def _make_deconv_layer_dcn(self, num_layers: int, num_filters: List[int], num_kernels: List[int]) -> nn.Sequential:
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            fc = ModulatedDeformConv2d(
                self.inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1
            )
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias,
            )
            fill_up_weights(up)

            bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
            nn.init.constant_(bn1.weight, 1)
            nn.init.constant_(bn1.bias, 0)
            bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
            nn.init.constant_(bn2.weight, 1)
            nn.init.constant_(bn2.bias, 0)

            layers.extend([fc, bn1, nn.ReLU(inplace=True), up, bn2, nn.ReLU(inplace=True)])
            self.inplanes = planes
        return nn.Sequential(*layers)

    def _make_deconv_layer_std(self, num_layers: int, num_filters: List[int], num_kernels: List[int]) -> nn.Sequential:
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            up = nn.ConvTranspose2d(
                in_channels=self.inplanes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias,
            )
            nn.init.normal_(up.weight, std=0.001)
            bn = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
            layers.extend([up, bn, nn.ReLU(inplace=True)])
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        return {head: self.__getattr__(head)(x) for head in self.heads}


# ==============================================================================
# 3. Official Tensor & Target Helpers (utils.py / image.py exact ports)
# ==============================================================================

def _gather_feat(feat: Tensor, ind: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat: Tensor, ind: Tensor) -> Tensor:
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.7) -> float:
    """CornerNet / CenterNet size-adaptive Gaussian radius."""
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1**2 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(0.0, b2**2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(0.0, b3**2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2D(shape: Tuple[int, int], sigma: float = 1.0, device=None, dtype=torch.float32) -> Tensor:
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y = torch.arange(-m, m + 1, device=device, dtype=dtype).unsqueeze(1)
    x = torch.arange(-n, n + 1, device=device, dtype=dtype).unsqueeze(0)
    h = torch.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap: Tensor, center: Tuple[int, int], radius: int, k: float = 1.0) -> Tensor:
    """Draws 2D Gaussian bump onto heatmap in-place (exact port of draw_umich_gaussian)."""
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6.0, device=heatmap.device, dtype=heatmap.dtype)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


# ==============================================================================
# 4. Official Loss Functions (losses.py / ctdet.py exact ports)
# ==============================================================================

def _neg_loss(pred: Tensor, gt: Tensor) -> Tensor:
    """Modified penalty-reduced focal loss (Zhou et al. Eq. 1 / losses.py)."""
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1.0 - gt, 4)

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, 2) * pos_inds
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_sum = pos_loss.sum()
    neg_sum = neg_loss.sum()

    if num_pos == 0:
        return -neg_sum
    return -(pos_sum + neg_sum) / num_pos


class RegL1Loss(nn.Module):
    """Sparse center-indexed L1 regression loss (losses.py RegL1Loss)."""

    def forward(self, output: Tensor, mask: Tensor, ind: Tensor, target: Tensor) -> Tensor:
        pred = _transpose_and_gather_feat(output, ind)
        mask_exp = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask_exp, target * mask_exp, reduction="sum")
        loss = loss / (mask_exp.sum() + 1e-4)
        return loss


class CenterNetLoss(nn.Module):
    """Official CenterNet ctdet objective: 1.0 * hm_loss + 0.1 * wh_loss + 1.0 * off_loss."""

    def __init__(self, hm_weight: float = 1.0, wh_weight: float = 0.1, off_weight: float = 1.0):
        super().__init__()
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight
        self.reg_l1 = RegL1Loss()

    def forward(
        self,
        outputs: Dict[str, Tensor],
        gt_hm: Tensor,
        gt_wh: Tensor,
        gt_reg: Tensor,
        ind: Tensor,
        reg_mask: Tensor,
    ) -> Dict[str, Tensor]:
        device_type = outputs["hm"].device.type
        # Run focal and L1 losses strictly in float32 for AMP stability
        with torch.amp.autocast(device_type=device_type, enabled=False):
            hm_logits = outputs["hm"].float()
            wh_pred = outputs["wh"].float()
            reg_pred = outputs["reg"].float()

            hm_prob = torch.clamp(torch.sigmoid(hm_logits), min=1e-4, max=1.0 - 1e-4)

            hm_loss = _neg_loss(hm_prob, gt_hm.float())
            wh_loss = self.reg_l1(wh_pred, reg_mask, ind, gt_wh.float())
            off_loss = self.reg_l1(reg_pred, reg_mask, ind, gt_reg.float())

        return {
            "hm_loss": self.hm_weight * hm_loss,
            "wh_loss": self.wh_weight * wh_loss,
            "off_loss": self.off_weight * off_loss,
        }


# ==============================================================================
# 5. Official Decoder (decode.py exact port)
# ==============================================================================

def _nms(heat: Tensor, kernel: int = 3) -> Tensor:
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores: Tensor, K: int = 100) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch, cat, height, width = scores.size()
    hw = height * width
    k_chan = min(K, hw)

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), k_chan)
    topk_inds = topk_inds % hw
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()

    k_all = min(K, cat * k_chan)
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k_all)
    topk_clses = (topk_ind // k_chan).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, k_all)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, k_all)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, k_all)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(
    heat: Tensor,
    wh: Tensor,
    reg: Optional[Tensor] = None,
    cat_spec_wh: bool = False,
    K: int = 100,
) -> Tensor:
    """
    Decodes CenterNet heatmaps, offsets, and sizes into [B, K, 6] detections:
    [x1, y1, x2, y2, score, cls] on feature-map coordinates.
    """
    batch, cat, height, width = heat.size()
    heat = _nms(heat, kernel=3)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    k_eff = scores.size(1)

    if reg is not None:
        reg_g = _transpose_and_gather_feat(reg, inds).view(batch, k_eff, 2)
        xs = xs.view(batch, k_eff, 1) + reg_g[:, :, 0:1]
        ys = ys.view(batch, k_eff, 1) + reg_g[:, :, 1:2]
    else:
        xs = xs.view(batch, k_eff, 1) + 0.5
        ys = ys.view(batch, k_eff, 1) + 0.5

    wh_g = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh_g = wh_g.view(batch, k_eff, cat, 2)
        clses_ind = clses.view(batch, k_eff, 1, 1).expand(batch, k_eff, 1, 2).long()
        wh_g = wh_g.gather(2, clses_ind).view(batch, k_eff, 2)
    else:
        wh_g = wh_g.view(batch, k_eff, 2)

    clses = clses.view(batch, k_eff, 1).float()
    scores = scores.view(batch, k_eff, 1)
    bboxes = torch.cat(
        [
            xs - wh_g[..., 0:1] / 2.0,
            ys - wh_g[..., 1:2] / 2.0,
            xs + wh_g[..., 0:1] / 2.0,
            ys + wh_g[..., 1:2] / 2.0,
        ],
        dim=2,
    )
    return torch.cat([bboxes, scores, clses], dim=2)


# ==============================================================================
# 6. Detector Wrapper (Standard Torchvision Detection API)
# ==============================================================================

class CenterNetDetector(nn.Module):
    """
    Torchvision-compatible wrapper for CenterNet (Objects as Points):
      - Training: forward(images, targets) -> dict of losses
      - Eval:     forward(images) -> List[{'boxes', 'scores', 'labels'}] in original resolution
    """

    def __init__(
        self,
        num_classes: int = 20,
        num_layers: int = 50,
        pretrained_backbone: bool = True,
        use_dcn: bool = True,
        input_size: Tuple[int, int] = (512, 512),
        head_conv: int = 64,
        top_k: int = 100,
        score_thresh: float = 0.0,
        flip_test: bool = False,
        hm_weight: float = 1.0,
        wh_weight: float = 0.1,
        off_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.use_dcn = use_dcn
        self.input_size = input_size
        self.stride = 4
        self.output_size = (input_size[0] // self.stride, input_size[1] // self.stride)
        self.top_k = top_k
        self.score_thresh = score_thresh
        self.flip_test = flip_test

        self.net = CenterNetResNet(
            num_layers=num_layers,
            heads={"hm": num_classes, "wh": 2, "reg": 2},
            head_conv=head_conv,
            pretrained_backbone=pretrained_backbone,
            use_dcn=use_dcn,
        )
        self.loss_fn = CenterNetLoss(
            hm_weight=hm_weight,
            wh_weight=wh_weight,
            off_weight=off_weight,
        )

        # Official ImageNet / Pascal VOC RGB normalization stats (pascal.py)
        self.register_buffer(
            "pixel_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def _letterbox_image(self, img: Tensor) -> Tuple[Tensor, float, int, int, int, int]:
        _, h, w = img.shape
        in_h, in_w = self.input_size

        if h == in_h and w == in_w:
            return img, 1.0, 0, 0, w, h

        s = max(h, w)
        pad_left = (s - w) // 2
        pad_right = s - w - pad_left
        pad_top = (s - h) // 2
        pad_bottom = s - h - pad_top

        padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        resized = F.interpolate(
            padded.unsqueeze(0), size=(in_h, in_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        scale = float(in_w) / float(s)
        return resized, scale, pad_left, pad_top, w, h

    def _prepare_targets(
        self,
        targets: List[Dict[str, Tensor]],
        scales: List[float],
        pad_xs: List[int],
        pad_ys: List[int],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = len(targets)
        out_h, out_w = self.output_size
        max_objs = max(1, max((len(t["boxes"]) for t in targets), default=0))

        gt_hm = torch.zeros((batch_size, self.num_classes, out_h, out_w), device=device, dtype=torch.float32)
        gt_wh = torch.zeros((batch_size, max_objs, 2), device=device, dtype=torch.float32)
        gt_reg = torch.zeros((batch_size, max_objs, 2), device=device, dtype=torch.float32)
        ind = torch.zeros((batch_size, max_objs), device=device, dtype=torch.int64)
        reg_mask = torch.zeros((batch_size, max_objs), device=device, dtype=torch.uint8)

        for b_idx, t in enumerate(targets):
            boxes = t["boxes"]
            labels = t["labels"]
            num_objs = len(boxes)
            if num_objs == 0:
                continue

            scale = scales[b_idx]
            px = pad_xs[b_idx]
            py = pad_ys[b_idx]

            for k in range(num_objs):
                x1, y1, x2, y2 = boxes[k].tolist()
                fx1 = max(0.0, min(float(out_w - 1), ((x1 + px) * scale) / self.stride))
                fy1 = max(0.0, min(float(out_h - 1), ((y1 + py) * scale) / self.stride))
                fx2 = max(0.0, min(float(out_w - 1), ((x2 + px) * scale) / self.stride))
                fy2 = max(0.0, min(float(out_h - 1), ((y2 + py) * scale) / self.stride))

                w_f = fx2 - fx1
                h_f = fy2 - fy1
                if h_f <= 0.0 or w_f <= 0.0:
                    continue

                if self.num_classes == 1:
                    cls_id = 0
                else:
                    cls_id = max(0, min(self.num_classes - 1, int(labels[k].item()) - 1))

                radius = max(0, int(gaussian_radius((math.ceil(h_f), math.ceil(w_f)), min_overlap=0.7)))
                ctx = (fx1 + fx2) / 2.0
                cty = (fy1 + fy2) / 2.0
                ctx_int = max(0, min(out_w - 1, int(ctx)))
                cty_int = max(0, min(out_h - 1, int(cty)))

                draw_umich_gaussian(gt_hm[b_idx, cls_id], (ctx_int, cty_int), radius)

                gt_wh[b_idx, k, 0] = w_f
                gt_wh[b_idx, k, 1] = h_f
                ind[b_idx, k] = cty_int * out_w + ctx_int
                gt_reg[b_idx, k, 0] = ctx - ctx_int
                gt_reg[b_idx, k, 1] = cty - cty_int
                reg_mask[b_idx, k] = 1

        return gt_hm, gt_wh, gt_reg, ind, reg_mask

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        device = images[0].device

        batch_tensors = []
        scales = []
        pad_xs = []
        pad_ys = []
        orig_sizes = []

        for img in images:
            padded, scale, px, py, ow, oh = self._letterbox_image(img)
            batch_tensors.append(padded)
            scales.append(scale)
            pad_xs.append(px)
            pad_ys.append(py)
            orig_sizes.append((ow, oh))

        x = torch.stack(batch_tensors, dim=0).to(device)
        x = (x - self.pixel_mean) / self.pixel_std

        if self.training and targets is not None:
            outputs = self.net(x)
            gt_hm, gt_wh, gt_reg, ind, reg_mask = self._prepare_targets(
                targets, scales, pad_xs, pad_ys, device
            )
            return self.loss_fn(outputs, gt_hm, gt_wh, gt_reg, ind, reg_mask)

        # Inference mode (supports flip_test from CtdetDetector.process)
        if self.flip_test:
            bsz = x.size(0)
            x_aug = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
            out_aug = self.net(x_aug)
            hm = (torch.sigmoid(out_aug["hm"][:bsz]) + torch.flip(torch.sigmoid(out_aug["hm"][bsz:]), dims=[3])) / 2.0
            wh = (out_aug["wh"][:bsz] + torch.flip(out_aug["wh"][bsz:], dims=[3])) / 2.0
            reg = out_aug["reg"][:bsz]
        else:
            outputs = self.net(x)
            hm = torch.sigmoid(outputs["hm"])
            wh = outputs["wh"]
            reg = outputs["reg"]

        dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=False, K=self.top_k)

        results: List[Dict[str, Tensor]] = []
        for b_idx in range(len(images)):
            scale = scales[b_idx]
            px = pad_xs[b_idx]
            py = pad_ys[b_idx]
            ow, oh = orig_sizes[b_idx]

            det = dets[b_idx]  # [K, 6]: x1, y1, x2, y2, score, cls
            scores = det[:, 4]
            keep = scores >= self.score_thresh
            det = det[keep]

            if len(det) == 0:
                results.append(
                    {
                        "boxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
                        "scores": torch.zeros((0,), device=device, dtype=torch.float32),
                        "labels": torch.zeros((0,), device=device, dtype=torch.int64),
                    }
                )
                continue

            boxes_in = det[:, :4] * self.stride
            x1 = torch.clamp(boxes_in[:, 0] / scale - px, min=0.0, max=float(ow))
            y1 = torch.clamp(boxes_in[:, 1] / scale - py, min=0.0, max=float(oh))
            x2 = torch.clamp(boxes_in[:, 2] / scale - px, min=0.0, max=float(ow))
            y2 = torch.clamp(boxes_in[:, 3] / scale - py, min=0.0, max=float(oh))
            boxes_orig = torch.stack([x1, y1, x2, y2], dim=1)

            valid = (boxes_orig[:, 2] > boxes_orig[:, 0]) & (boxes_orig[:, 3] > boxes_orig[:, 1])
            boxes_orig = boxes_orig[valid]
            scores_out = det[valid, 4]
            labels_out = det[valid, 5].long() + 1  # 1-indexed matching VOC / FCOS convention

            results.append(
                {
                    "boxes": boxes_orig.float(),
                    "scores": scores_out.float(),
                    "labels": labels_out,
                }
            )

        return results


def build_centernet_model(
    num_classes: int = 20,
    num_layers: int = 50,
    pretrained_backbone: bool = True,
    use_dcn: bool = True,
    input_size: Union[int, Tuple[int, int]] = 512,
    head_conv: int = 64,
    top_k: int = 100,
    score_thresh: float = 0.0,
    flip_test: bool = False,
    **kwargs,
) -> CenterNetDetector:
    """Factory function for CenterNet (Objects as Points, Zhou et al. 2019)."""
    if isinstance(input_size, int):
        input_size_tuple = (input_size, input_size)
    else:
        input_size_tuple = tuple(input_size)

    return CenterNetDetector(
        num_classes=num_classes,
        num_layers=num_layers,
        pretrained_backbone=pretrained_backbone,
        use_dcn=use_dcn,
        input_size=input_size_tuple,
        head_conv=head_conv,
        top_k=top_k,
        score_thresh=score_thresh,
        flip_test=flip_test,
        **kwargs,
    )
