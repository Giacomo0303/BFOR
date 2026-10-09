import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import nms


# ==============================================================================
# 1. Pure PyTorch Pooling Operations (Exact Autograd Replacement for CUDA C++)
# ==============================================================================

class TopPool(nn.Module):
    """Max-pooling from bottom to top along vertical axis."""
    def forward(self, x: Tensor) -> Tensor:
        return torch.flip(torch.cummax(torch.flip(x, dims=[-2]), dim=-2)[0], dims=[-2])


class BottomPool(nn.Module):
    """Max-pooling from top to bottom along vertical axis."""
    def forward(self, x: Tensor) -> Tensor:
        return torch.cummax(x, dim=-2)[0]


class LeftPool(nn.Module):
    """Max-pooling from right to left along horizontal axis."""
    def forward(self, x: Tensor) -> Tensor:
        return torch.flip(torch.cummax(torch.flip(x, dims=[-1]), dim=-1)[0], dims=[-1])


class RightPool(nn.Module):
    """Max-pooling from left to right along horizontal axis."""
    def forward(self, x: Tensor) -> Tensor:
        return torch.cummax(x, dim=-1)[0]


# ==============================================================================
# 2. Basic Convolutional and Residual Building Blocks
# ==============================================================================

class convolution(nn.Module):
    def __init__(self, k: int, inp_dim: int, out_dim: int, stride: int = 1, with_bn: bool = True):
        super().__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(
            inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn
        )
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class residual(nn.Module):
    def __init__(self, k: int, inp_dim: int, out_dim: int, stride: int = 1, with_bn: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = (
            nn.Sequential(
                nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(out_dim),
            )
            if stride != 1 or inp_dim != out_dim
            else nn.Sequential()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


def make_layer(k: int, inp_dim: int, out_dim: int, modules: int, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)


def make_layer_revr(k: int, inp_dim: int, out_dim: int, modules: int, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)


class MergeUp(nn.Module):
    def forward(self, up1: Tensor, up2: Tensor) -> Tensor:
        return up1 + up2


# ==============================================================================
# 3. Cascade Corner Pooling & Center Pooling Modules
# ==============================================================================

class CascadeCornerPool(nn.Module):
    """
    Cascade Corner Pooling for Top-Left and Bottom-Right corners
    (Duan et al., ICCV 2019, Section 3.2, Figure 4).
    """
    def __init__(self, dim: int, pool1_cls, pool2_cls):
        super().__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1_cls()
        self.pool2 = pool2_cls()

        self.look_conv1 = convolution(3, dim, 128)
        self.look_conv2 = convolution(3, dim, 128)
        self.P1_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)
        self.P2_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # Branch 1
        look_conv1 = self.look_conv1(x)
        p1_conv1 = self.p1_conv1(x)
        look_right = self.pool2(look_conv1)
        p1_look_conv = self.P1_look_conv(p1_conv1 + look_right)
        pool1 = self.pool1(p1_look_conv)

        # Branch 2
        look_conv2 = self.look_conv2(x)
        p2_conv1 = self.p2_conv1(x)
        look_down = self.pool1(look_conv2)
        p2_look_conv = self.P2_look_conv(p2_conv1 + look_down)
        pool2 = self.pool2(p2_look_conv)

        # Merge branches
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        return self.conv2(relu1)


class CenterPool(nn.Module):
    """
    Center Pooling (cross max-pooling) for center keypoints
    (Duan et al., ICCV 2019, Section 3.2, Figure 4).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = TopPool()
        self.pool2 = LeftPool()
        self.pool3 = BottomPool()
        self.pool4 = RightPool()

    def forward(self, x: Tensor) -> Tensor:
        p1_conv1 = self.p1_conv1(x)
        pool1 = self.pool1(p1_conv1)
        pool1 = self.pool3(pool1)

        p2_conv1 = self.p2_conv1(x)
        pool2 = self.pool2(p2_conv1)
        pool2 = self.pool4(pool2)

        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        return self.conv2(relu1)


# ==============================================================================
# 4. Hourglass-52 Recursive Module and Layers
# ==============================================================================

def make_hg_layer(kernel: int, dim0: int, dim1: int, mod: int, layer=residual):
    layers = [layer(kernel, dim0, dim1, stride=2)]
    for _ in range(mod - 1):
        layers.append(layer(kernel, dim1, dim1))
    return nn.Sequential(*layers)


class HourglassModule(nn.Module):
    """Recursive Hourglass Module (Newell et al., ECCV 2016)."""
    def __init__(self, n: int, dims: List[int], modules: List[int], layer=residual):
        super().__init__()
        self.n = n
        curr_mod = modules[0]
        next_mod = modules[1]
        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1 = make_layer(3, curr_dim, curr_dim, curr_mod, layer=layer)
        self.max1 = nn.Sequential()
        self.low1 = make_hg_layer(3, curr_dim, next_dim, curr_mod, layer=layer)
        self.low2 = (
            HourglassModule(n - 1, dims[1:], modules[1:], layer=layer)
            if n > 1
            else make_layer(3, next_dim, next_dim, next_mod, layer=layer)
        )
        self.low3 = make_layer_revr(3, next_dim, curr_dim, curr_mod, layer=layer)
        self.up2 = nn.Upsample(scale_factor=2)
        self.merge = MergeUp()

    def forward(self, x: Tensor) -> Tensor:
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return self.merge(up1, up2)


def make_kp_layer(cnv_dim: int, curr_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1)),
    )


# ==============================================================================
# 5. Full Standalone CenterNet-52 Network Architecture
# ==============================================================================

class CenterNet52Backbone(nn.Module):
    """
    1-stack Hourglass-52 CenterNet backbone with Cascade Corner Pooling
    and Center Pooling heads (Duan et al., ICCV 2019).
    """
    def __init__(self, num_classes: int = 20):
        super().__init__()
        self.num_classes = num_classes
        n = 5
        dims = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]
        cnv_dim = 256
        curr_dim = dims[0]

        # Stem (downsamples 4x)
        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2),
        )

        # Hourglass-52 stack
        self.kps = nn.ModuleList([HourglassModule(n, dims, modules, layer=residual)])
        self.cnvs = nn.ModuleList([convolution(3, curr_dim, cnv_dim)])

        # Pooling layers
        self.tl_cnvs = nn.ModuleList([CascadeCornerPool(cnv_dim, TopPool, LeftPool)])
        self.br_cnvs = nn.ModuleList([CascadeCornerPool(cnv_dim, BottomPool, RightPool)])
        self.ct_cnvs = nn.ModuleList([CenterPool(cnv_dim)])

        # Heatmaps (initialized with -2.19 bias for focal loss stability)
        self.tl_heats = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes)])
        self.br_heats = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes)])
        self.ct_heats = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, num_classes)])

        for tl_heat, br_heat, ct_heat in zip(self.tl_heats, self.br_heats, self.ct_heats):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)
            ct_heat[-1].bias.data.fill_(-2.19)

        # Associative Embedding tags (1D)
        self.tl_tags = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1)])
        self.br_tags = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 1)])

        # Sub-pixel offsets (2D)
        self.tl_regrs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2)])
        self.br_regrs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2)])
        self.ct_regrs = nn.ModuleList([make_kp_layer(cnv_dim, curr_dim, 2)])

    def forward(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        inter = self.pre(x)
        kp = self.kps[0](inter)
        cnv = self.cnvs[0](kp)

        tl_cnv = self.tl_cnvs[0](cnv)
        br_cnv = self.br_cnvs[0](cnv)
        ct_cnv = self.ct_cnvs[0](cnv)

        tl_heat = self.tl_heats[0](tl_cnv)
        br_heat = self.br_heats[0](br_cnv)
        ct_heat = self.ct_heats[0](ct_cnv)

        tl_tag = self.tl_tags[0](tl_cnv)
        br_tag = self.br_tags[0](br_cnv)

        tl_regr = self.tl_regrs[0](tl_cnv)
        br_regr = self.br_regrs[0](br_cnv)
        ct_regr = self.ct_regrs[0](ct_cnv)

        return tl_heat, br_heat, ct_heat, tl_tag, br_tag, tl_regr, br_regr, ct_regr


# ==============================================================================
# 6. CenterNet Loss Functions (Exact Formulation from Paper & Official Code)
# ==============================================================================

def _neg_loss(preds: List[Tensor], gt: Tensor) -> Tensor:
    """Gaussian Focal Loss (CornerNet / CenterNet Eq. 1)."""
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = torch.tensor(0.0, device=gt.device, dtype=gt.dtype)
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _ae_loss(tag0: Tensor, tag1: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Associative Embedding Push/Pull Loss (CornerNet Eq. 4-5)."""
    num = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze(-1)
    tag1 = tag1.squeeze(-1)

    tag_mean = (tag0 + tag1) / 2

    tag0_diff = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0_sum = tag0_diff[mask].sum()
    tag1_diff = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1_sum = tag1_diff[mask].sum()
    pull = tag0_sum + tag1_sum

    mask_pair = mask.unsqueeze(1) & mask.unsqueeze(2)
    num_exp = num.unsqueeze(2)
    num2 = (num_exp - 1) * num_exp

    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1.0 - torch.abs(dist)
    dist = F.relu(dist, inplace=False)
    dist = dist - 1.0 / (num_exp + 1e-4)
    dist = dist / (num2 + 1e-4)
    push = dist[mask_pair].sum()
    return pull, push


def _regr_loss(regr: Tensor, gt_regr: Tensor, mask: Tensor) -> Tensor:
    """Smooth L1 Offset Loss (CornerNet Eq. 3)."""
    num = mask.float().sum()
    mask_exp = mask.unsqueeze(2).expand_as(gt_regr)

    regr_sel = regr[mask_exp]
    gt_regr_sel = gt_regr[mask_exp]

    loss = F.smooth_l1_loss(regr_sel, gt_regr_sel, reduction="sum")
    return loss / (num + 1e-4)


def _gather_feat(feat: Tensor, ind: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask].view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat: Tensor, ind: Tensor) -> Tensor:
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return _gather_feat(feat, ind)


class CenterNetLoss(nn.Module):
    def __init__(self, pull_weight: float = 0.1, push_weight: float = 0.1, regr_weight: float = 1.0):
        super().__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight

    def forward(
        self,
        tl_heat: Tensor,
        br_heat: Tensor,
        ct_heat: Tensor,
        tl_tag: Tensor,
        br_tag: Tensor,
        tl_regr: Tensor,
        br_regr: Tensor,
        ct_regr: Tensor,
        tl_inds: Tensor,
        br_inds: Tensor,
        ct_inds: Tensor,
        gt_tl_heat: Tensor,
        gt_br_heat: Tensor,
        gt_ct_heat: Tensor,
        gt_mask: Tensor,
        gt_tl_regr: Tensor,
        gt_br_regr: Tensor,
        gt_ct_regr: Tensor,
    ) -> Dict[str, Tensor]:
        # Gather predicted tags and regrs at ground truth keypoint positions
        tl_tag_g = _tranpose_and_gather_feat(tl_tag, tl_inds)
        br_tag_g = _tranpose_and_gather_feat(br_tag, br_inds)

        tl_regr_g = _tranpose_and_gather_feat(tl_regr, tl_inds)
        br_regr_g = _tranpose_and_gather_feat(br_regr, br_inds)
        ct_regr_g = _tranpose_and_gather_feat(ct_regr, ct_inds)

        # Clamped sigmoids for focal loss stability
        tl_heat_sig = torch.sigmoid(tl_heat).clamp(min=1e-4, max=1.0 - 1e-4)
        br_heat_sig = torch.sigmoid(br_heat).clamp(min=1e-4, max=1.0 - 1e-4)
        ct_heat_sig = torch.sigmoid(ct_heat).clamp(min=1e-4, max=1.0 - 1e-4)

        # 1. Focal loss
        focal_tl = _neg_loss([tl_heat_sig], gt_tl_heat)
        focal_br = _neg_loss([br_heat_sig], gt_br_heat)
        focal_ct = _neg_loss([ct_heat_sig], gt_ct_heat)
        focal_loss = focal_tl + focal_br + focal_ct

        # 2. Tag push/pull loss
        pull, push = _ae_loss(tl_tag_g, br_tag_g, gt_mask)
        pull_loss = self.pull_weight * pull
        push_loss = self.push_weight * push

        # 3. Offset regression loss
        regr_tl = _regr_loss(tl_regr_g, gt_tl_regr, gt_mask)
        regr_br = _regr_loss(br_regr_g, gt_br_regr, gt_mask)
        regr_ct = _regr_loss(ct_regr_g, gt_ct_regr, gt_mask)
        regr_loss = self.regr_weight * (regr_tl + regr_br + regr_ct)

        total_loss = focal_loss + pull_loss + push_loss + regr_loss
        return {
            "loss_focal": focal_loss,
            "loss_pull": pull_loss,
            "loss_push": push_loss,
            "loss_regr": regr_loss,
        }


# ==============================================================================
# 7. Target Generation (Gaussian Heatmaps, Offsets, Tags)
# ==============================================================================

def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.7) -> float:
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


def draw_gaussian_2d(heatmap: Tensor, center: Tuple[int, int], radius: int, k: float = 1.0, delte: float = 6.0):
    """Draws 2D Gaussian bump onto heatmap in-place."""
    diameter = 2 * radius + 1
    sigma = diameter / delte
    x, y = center
    height, width = heatmap.shape

    left = min(x, radius)
    right = min(width - x, radius + 1)
    top = min(y, radius)
    bottom = min(height - y, radius + 1)

    if left + right <= 0 or top + bottom <= 0:
        return

    # Generate gaussian kernel
    y_coords = torch.arange(-top, bottom, device=heatmap.device, dtype=heatmap.dtype).unsqueeze(1)
    x_coords = torch.arange(-left, right, device=heatmap.device, dtype=heatmap.dtype).unsqueeze(0)
    g = torch.exp(-(x_coords**2 + y_coords**2) / (2 * sigma * sigma))

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    torch.maximum(masked_heatmap, g * k, out=masked_heatmap)


# ==============================================================================
# 8. Detection Decoder with Scale-Aware Central Region Verification
# ==============================================================================

def _nms(heat: Tensor, kernel: int = 3) -> Tensor:
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores: Tensor, K: int = 70) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_clses = (topk_inds // (height * width)).int()
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def decode_centernet(
    tl_heat: Tensor,
    br_heat: Tensor,
    tl_tag: Tensor,
    br_tag: Tensor,
    tl_regr: Tensor,
    br_regr: Tensor,
    ct_heat: Tensor,
    ct_regr: Tensor,
    K: int = 70,
    kernel: int = 3,
    ae_threshold: float = 0.5,
    num_dets: int = 100,
) -> Tuple[Tensor, Tensor]:
    """
    Decodes raw network heads into corner bounding boxes and center keypoints.
    Returns:
        detections: [B, num_dets, 8] -> [x1, y1, x2, y2, score, tl_score, br_score, cls]
        center: [B, K, 4] -> [x, y, cls, score]
    """
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)
    ct_heat = torch.sigmoid(ct_heat)

    # 3x3 NMS peak selection
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)
    ct_heat = _nms(ct_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk(ct_heat, K=K)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)
    ct_ys = ct_ys.view(batch, 1, K).expand(batch, K, K)
    ct_xs = ct_xs.view(batch, 1, K).expand(batch, K, K)

    # Apply sub-pixel regression offsets
    tl_regr_g = _tranpose_and_gather_feat(tl_regr, tl_inds).view(batch, K, 1, 2)
    br_regr_g = _tranpose_and_gather_feat(br_regr, br_inds).view(batch, 1, K, 2)
    ct_regr_g = _tranpose_and_gather_feat(ct_regr, ct_inds).view(batch, 1, K, 2)

    tl_xs = tl_xs + tl_regr_g[..., 0]
    tl_ys = tl_ys + tl_regr_g[..., 1]
    br_xs = br_xs + br_regr_g[..., 0]
    br_ys = br_ys + br_regr_g[..., 1]
    ct_xs = ct_xs + ct_regr_g[..., 0]
    ct_ys = ct_ys + ct_regr_g[..., 1]

    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    # Corner associative embedding distance
    tl_tag_g = _tranpose_and_gather_feat(tl_tag, tl_inds).view(batch, K, 1)
    br_tag_g = _tranpose_and_gather_feat(br_tag, br_inds).view(batch, 1, K)
    dists = torch.abs(tl_tag_g - br_tag_g)

    tl_scores_exp = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores_exp = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores = (tl_scores_exp + br_scores_exp) / 2.0

    # Reject mismatched class, distance > threshold, or inverted boxes
    tl_clses_exp = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses_exp = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = tl_clses_exp != br_clses_exp
    dist_inds = dists > ae_threshold
    width_inds = br_xs < tl_xs
    height_inds = br_ys < tl_ys

    scores[cls_inds] = -1.0
    scores[dist_inds] = -1.0
    scores[width_inds] = -1.0
    scores[height_inds] = -1.0

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, min(num_dets, scores.size(1)))
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses = tl_clses_exp.contiguous().view(batch, -1, 1)
    clses = _gather_feat(clses, inds).float()

    tl_scores_sel = tl_scores_exp.contiguous().view(batch, -1, 1)
    tl_scores_sel = _gather_feat(tl_scores_sel, inds).float()
    br_scores_sel = br_scores_exp.contiguous().view(batch, -1, 1)
    br_scores_sel = _gather_feat(br_scores_sel, inds).float()

    ct_xs_single = ct_xs[:, 0, :]
    ct_ys_single = ct_ys[:, 0, :]

    centers = torch.cat(
        [
            ct_xs_single.unsqueeze(2),
            ct_ys_single.unsqueeze(2),
            ct_clses.float().unsqueeze(2),
            ct_scores.unsqueeze(2),
        ],
        dim=2,
    )
    detections = torch.cat([bboxes, scores, tl_scores_sel, br_scores_sel, clses], dim=2)
    return detections, centers


def apply_scale_aware_central_region(
    detections: Tensor, center_points: Tensor
) -> Tensor:
    """
    Applies Duan et al. (2019) Scale-Aware Central Region Verification:
    - Small boxes (area <= 22500): n = 3
    - Large boxes (area > 22500): n = 5
    - If center of same category falls in central region:
        S = (S_tl + S_br + S_ct) / 3 = (2 * S_corner + S_ct) / 3
      Else: box is discarded (score = -1).
    """
    valid_mask = detections[:, 4] > -1.0
    valid_dets = detections[valid_mask].clone()
    if len(valid_dets) == 0:
        return torch.zeros((0, 8), device=detections.device, dtype=detections.dtype)

    bw = valid_dets[:, 2] - valid_dets[:, 0]
    bh = valid_dets[:, 3] - valid_dets[:, 1]
    s_mask = (bw * bh) <= 22500.0
    l_mask = ~s_mask

    s_dets = valid_dets[s_mask]
    l_dets = valid_dets[l_mask]

    # Process small detections (n=3)
    if len(s_dets) > 0 and len(center_points) > 0:
        s_lx = (2.0 * s_dets[:, 0] + s_dets[:, 2]) / 3.0
        s_rx = (s_dets[:, 0] + 2.0 * s_dets[:, 2]) / 3.0
        s_ty = (2.0 * s_dets[:, 1] + s_dets[:, 3]) / 3.0
        s_by = (s_dets[:, 1] + 2.0 * s_dets[:, 3]) / 3.0

        s_orig_score = s_dets[:, 4].clone()
        s_dets[:, 4] = -1.0

        cx = center_points[:, 0:1]
        cy = center_points[:, 1:2]
        c_cls = center_points[:, 2:3]

        in_box = (
            (cx > s_lx.unsqueeze(0))
            & (cx < s_rx.unsqueeze(0))
            & (cy > s_ty.unsqueeze(0))
            & (cy < s_by.unsqueeze(0))
            & (c_cls == s_dets[:, 7].unsqueeze(0))
        )
        has_match = in_box.any(dim=0)
        if has_match.any():
            best_idx = in_box[:, has_match].long().argmax(dim=0)
            c_score_matched = center_points[best_idx, 3]
            s_dets[has_match, 4] = (s_orig_score[has_match] * 2.0 + c_score_matched) / 3.0
    elif len(s_dets) > 0:
        s_dets[:, 4] = -1.0

    # Process large detections (n=5)
    if len(l_dets) > 0 and len(center_points) > 0:
        l_lx = (3.0 * l_dets[:, 0] + 2.0 * l_dets[:, 2]) / 5.0
        l_rx = (2.0 * l_dets[:, 0] + 3.0 * l_dets[:, 2]) / 5.0
        l_ty = (3.0 * l_dets[:, 1] + 2.0 * l_dets[:, 3]) / 5.0
        l_by = (2.0 * l_dets[:, 1] + 3.0 * l_dets[:, 3]) / 5.0

        l_orig_score = l_dets[:, 4].clone()
        l_dets[:, 4] = -1.0

        cx = center_points[:, 0:1]
        cy = center_points[:, 1:2]
        c_cls = center_points[:, 2:3]

        in_box = (
            (cx > l_lx.unsqueeze(0))
            & (cx < l_rx.unsqueeze(0))
            & (cy > l_ty.unsqueeze(0))
            & (cy < l_by.unsqueeze(0))
            & (c_cls == l_dets[:, 7].unsqueeze(0))
        )
        has_match = in_box.any(dim=0)
        if has_match.any():
            best_idx = in_box[:, has_match].long().argmax(dim=0)
            c_score_matched = center_points[best_idx, 3]
            l_dets[has_match, 4] = (l_orig_score[has_match] * 2.0 + c_score_matched) / 3.0
    elif len(l_dets) > 0:
        l_dets[:, 4] = -1.0

    kept = torch.cat([l_dets, s_dets], dim=0)
    kept = kept[kept[:, 4] > -1.0]
    if len(kept) > 0:
        kept = kept[torch.argsort(-kept[:, 4])]
    return kept


# ==============================================================================
# 9. CenterNet Detection Model Wrapper (Torchvision Detection Compatible)
# ==============================================================================

class CenterNetDetector(nn.Module):
    """
    CenterNet Keypoint Triplets Object Detector (Duan et al., ICCV 2019).
    Fully compatible with the codebase training and evaluation protocols:
      - Training: model(images, targets) -> loss_dict
      - Evaluation: model(images) -> list[dict['boxes', 'scores', 'labels']]
    """
    def __init__(
        self,
        num_classes: int = 20,
        input_size: Tuple[int, int] = (512, 512),
        top_k: int = 70,
        ae_threshold: float = 0.5,
        nms_kernel: int = 3,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 100,
        score_thresh: float = 0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size  # (H, W) = (512, 512)
        self.output_size = (input_size[0] // 4, input_size[1] // 4)  # (128, 128)
        self.stride = 4

        self.top_k = top_k
        self.ae_threshold = ae_threshold
        self.nms_kernel = nms_kernel
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self.score_thresh = score_thresh

        # Model backbone and heads
        self.net = CenterNet52Backbone(num_classes=num_classes)
        self.loss_fn = CenterNetLoss(pull_weight=0.1, push_weight=0.1, regr_weight=1.0)

        # CenterNet RGB image normalization buffers (converted from official BGR values)
        # BGR: [0.40789654, 0.44719302, 0.47026115], std: [0.28863828, 0.27408164, 0.27809834]
        # RGB: [0.47026115, 0.44719302, 0.40789654], std: [0.27809834, 0.27408164, 0.28863828]
        self.register_buffer(
            "pixel_mean", torch.tensor([0.47026115, 0.44719302, 0.40789654]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor([0.27809834, 0.27408164, 0.28863828]).view(1, 3, 1, 1)
        )

    def _letterbox_image(
        self, img: Tensor
    ) -> Tuple[Tensor, float, float, float, int, int]:
        """
        Pads and resizes image into fixed input_size (512, 512) preserving aspect ratio.
        Returns:
            padded_img, scale, pad_x, pad_y, orig_w, orig_h
        """
        _, orig_h, orig_w = img.shape
        target_h, target_w = self.input_size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = min(target_w, int(round(orig_w * scale)))
        new_h = min(target_h, int(round(orig_h * scale)))

        # Resize
        resized = F.interpolate(
            img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
        ).squeeze(0)

        # Pad into (target_h, target_w)
        pad_x = (target_w - new_w) / 2.0
        pad_y = (target_h - new_h) / 2.0

        pad_left = int(pad_x)
        pad_right = target_w - new_w - pad_left
        pad_top = int(pad_y)
        pad_bottom = target_h - new_h - pad_top

        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        return padded, scale, float(pad_left), float(pad_top), orig_w, orig_h

    def _prepare_targets(
        self,
        targets: List[Dict[str, Tensor]],
        scales: List[float],
        pad_xs: List[float],
        pad_ys: List[float],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generates ground truth heatmaps, tags, and offsets on feature map resolution (128x128).
        """
        B = len(targets)
        C = self.num_classes
        out_h, out_w = self.output_size
        max_tag_len = max(1, max(len(t["boxes"]) for t in targets))

        gt_tl_heat = torch.zeros((B, C, out_h, out_w), device=device, dtype=torch.float32)
        gt_br_heat = torch.zeros((B, C, out_h, out_w), device=device, dtype=torch.float32)
        gt_ct_heat = torch.zeros((B, C, out_h, out_w), device=device, dtype=torch.float32)

        gt_mask = torch.zeros((B, max_tag_len), device=device, dtype=torch.bool)
        gt_tl_regr = torch.zeros((B, max_tag_len, 2), device=device, dtype=torch.float32)
        gt_br_regr = torch.zeros((B, max_tag_len, 2), device=device, dtype=torch.float32)
        gt_ct_regr = torch.zeros((B, max_tag_len, 2), device=device, dtype=torch.float32)

        tl_inds = torch.zeros((B, max_tag_len), device=device, dtype=torch.long)
        br_inds = torch.zeros((B, max_tag_len), device=device, dtype=torch.long)
        ct_inds = torch.zeros((B, max_tag_len), device=device, dtype=torch.long)

        for b_idx, (t, scale, px, py) in enumerate(zip(targets, scales, pad_xs, pad_ys)):
            boxes = t["boxes"]
            labels = t["labels"]
            num_boxes = len(boxes)
            if num_boxes == 0:
                continue

            for k in range(num_boxes):
                x1, y1, x2, y2 = boxes[k].tolist()
                lbl = int(labels[k].item())
                # Labels are 1..20; map to category 0..19
                cat = lbl - 1 if lbl > 0 else 0
                if cat < 0 or cat >= C:
                    continue

                # Transform to 512x512 padded canvas
                x1 = x1 * scale + px
                x2 = x2 * scale + px
                y1 = y1 * scale + py
                y2 = y2 * scale + py

                # Map to 128x128 feature map
                fxtl = x1 / float(self.stride)
                fytl = y1 / float(self.stride)
                fxbr = x2 / float(self.stride)
                fybr = y2 / float(self.stride)
                fxct = (x1 + x2) / (2.0 * float(self.stride))
                fyct = (y1 + y2) / (2.0 * float(self.stride))

                xtl = int(max(0, min(out_w - 1, fxtl)))
                ytl = int(max(0, min(out_h - 1, fytl)))
                xbr = int(max(0, min(out_w - 1, fxbr)))
                ybr = int(max(0, min(out_h - 1, fybr)))
                xct = int(max(0, min(out_w - 1, fxct)))
                yct = int(max(0, min(out_h - 1, fyct)))

                box_w = max(1.0, fxbr - fxtl)
                box_h = max(1.0, fybr - fytl)

                radius = int(max(0, gaussian_radius((box_h, box_w), min_overlap=0.7)))

                draw_gaussian_2d(gt_tl_heat[b_idx, cat], (xtl, ytl), radius, delte=6.0)
                draw_gaussian_2d(gt_br_heat[b_idx, cat], (xbr, ybr), radius, delte=6.0)
                draw_gaussian_2d(gt_ct_heat[b_idx, cat], (xct, yct), radius, delte=5.0)

                gt_tl_regr[b_idx, k] = torch.tensor([fxtl - xtl, fytl - ytl], device=device)
                gt_br_regr[b_idx, k] = torch.tensor([fxbr - xbr, fybr - ybr], device=device)
                gt_ct_regr[b_idx, k] = torch.tensor([fxct - xct, fyct - yct], device=device)

                tl_inds[b_idx, k] = ytl * out_w + xtl
                br_inds[b_idx, k] = ybr * out_w + xbr
                ct_inds[b_idx, k] = yct * out_w + xct

                gt_mask[b_idx, k] = True

        return (
            gt_tl_heat,
            gt_br_heat,
            gt_ct_heat,
            gt_mask,
            gt_tl_regr,
            gt_br_regr,
            gt_ct_regr,
            tl_inds,
            br_inds,
            ct_inds,
        )

    def forward(
        self,
        images: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if isinstance(images, (list, tuple)):
            device = images[0].device
        else:
            device = images.device

        # Letterbox and batch preparation
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
        # Normalize
        x = (x - self.pixel_mean) / self.pixel_std

        # Network forward pass
        (
            tl_heat,
            br_heat,
            ct_heat,
            tl_tag,
            br_tag,
            tl_regr,
            br_regr,
            ct_regr,
        ) = self.net(x)

        # ----------------------------------------------------
        # Training Mode: compute loss
        # ----------------------------------------------------
        if self.training and targets is not None:
            (
                gt_tl_heat,
                gt_br_heat,
                gt_ct_heat,
                gt_mask,
                gt_tl_regr,
                gt_br_regr,
                gt_ct_regr,
                tl_inds,
                br_inds,
                ct_inds,
            ) = self._prepare_targets(targets, scales, pad_xs, pad_ys, device)

            loss_dict = self.loss_fn(
                tl_heat=tl_heat,
                br_heat=br_heat,
                ct_heat=ct_heat,
                tl_tag=tl_tag,
                br_tag=br_tag,
                tl_regr=tl_regr,
                br_regr=br_regr,
                ct_regr=ct_regr,
                tl_inds=tl_inds,
                br_inds=br_inds,
                ct_inds=ct_inds,
                gt_tl_heat=gt_tl_heat,
                gt_br_heat=gt_br_heat,
                gt_ct_heat=gt_ct_heat,
                gt_mask=gt_mask,
                gt_tl_regr=gt_tl_regr,
                gt_br_regr=gt_br_regr,
                gt_ct_regr=gt_ct_regr,
            )
            return loss_dict

        # ----------------------------------------------------
        # Evaluation Mode: decode detections
        # ----------------------------------------------------
        batch_dets, batch_centers = decode_centernet(
            tl_heat=tl_heat,
            br_heat=br_heat,
            tl_tag=tl_tag,
            br_tag=br_tag,
            tl_regr=tl_regr,
            br_regr=br_regr,
            ct_heat=ct_heat,
            ct_regr=ct_regr,
            K=self.top_k,
            kernel=self.nms_kernel,
            ae_threshold=self.ae_threshold,
            num_dets=self.max_detections,
        )

        results: List[Dict[str, Tensor]] = []
        for i in range(len(images)):
            dets_i = batch_dets[i]
            cents_i = batch_centers[i]
            scale = scales[i]
            px = pad_xs[i]
            py = pad_ys[i]
            ow, oh = orig_sizes[i]

            # Scale coordinates from 128x128 feature map to 512x512 canvas
            dets_i[:, :4] = dets_i[:, :4] * float(self.stride)
            cents_i[:, :2] = cents_i[:, :2] * float(self.stride)

            # Apply scale-aware central region verification
            verified_dets = apply_scale_aware_central_region(dets_i, cents_i)

            if len(verified_dets) == 0:
                results.append({
                    "boxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
                    "scores": torch.zeros((0,), device=device, dtype=torch.float32),
                    "labels": torch.zeros((0,), device=device, dtype=torch.int64),
                })
                continue

            # Unpad and scale back to original unpadded resolution
            x1 = (verified_dets[:, 0] - px) / scale
            y1 = (verified_dets[:, 1] - py) / scale
            x2 = (verified_dets[:, 2] - px) / scale
            y2 = (verified_dets[:, 3] - py) / scale

            x1 = x1.clamp(min=0.0, max=float(ow))
            x2 = x2.clamp(min=0.0, max=float(ow))
            y1 = y1.clamp(min=0.0, max=float(oh))
            y2 = y2.clamp(min=0.0, max=float(oh))

            scores = verified_dets[:, 4]
            # Map category (0..19) back to class ID (1..20)
            labels = (verified_dets[:, 7] + 1.0).long()

            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # Filter valid dimensions and score threshold
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if self.score_thresh > 0:
                valid = valid & (scores >= self.score_thresh)

            boxes = boxes[valid]
            scores = scores[valid]
            labels = labels[valid]

            # Per-class NMS to remove redundant overlapping boxes
            if len(boxes) > 0 and self.nms_iou_threshold > 0:
                keep = []
                for c in range(1, self.num_classes + 1):
                    c_mask = labels == c
                    if not c_mask.any():
                        continue
                    c_inds = torch.where(c_mask)[0]
                    c_boxes = boxes[c_inds]
                    c_scores = scores[c_inds]
                    c_keep = nms(c_boxes, c_scores, self.nms_iou_threshold)
                    keep.append(c_inds[c_keep])

                if len(keep) > 0:
                    keep = torch.cat(keep, dim=0)
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]

                # Sort by score descending and keep up to max_detections
                order = torch.argsort(-scores)[: self.max_detections]
                boxes = boxes[order]
                scores = scores[order]
                labels = labels[order]

            results.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })

        return results


def build_centernet_model(
    num_classes: int = 20,
    pretrained: bool = False,
    input_size: Tuple[int, int] = (512, 512),
    **kwargs,
) -> CenterNetDetector:
    """
    Factory function for CenterNet Keypoint Triplets model with Hourglass-52 backbone.
    Hourglass-52 is trained from scratch as per Duan et al. (2019) and Newell et al. (2016).
    """
    model = CenterNetDetector(
        num_classes=num_classes,
        input_size=input_size,
        **kwargs,
    )
    return model
