import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import nms

# Re-use proven architectural blocks from CenterNet baseline
from src.centernet_model import (
    CenterNet52Backbone,
    _ae_loss,
    _gather_feat,
    _regr_loss,
    _tranpose_and_gather_feat,
    apply_scale_aware_central_region,
    decode_centernet,
    gaussian_radius,
)


# ==============================================================================
# 1. Corner Quadrant Gaussian (Confined strictly inside Bounding Box)
# ==============================================================================

def draw_quadrant_gaussian_2d(
    heatmap: Tensor,
    center: Tuple[int, int],
    radius: int,
    quadrant: str,  # "tl" (down-right) or "br" (up-left)
    box_w: Optional[int] = None,
    box_h: Optional[int] = None,
    k: float = 1.0,
    delte: float = 6.0,
):
    """
    Draws a 2D Gaussian bump restricted strictly to the internal quadrant
    of the bounding box. Zero pixels outside the bounding box receive positive target values.
    """
    diameter = 2 * radius + 1
    sigma = diameter / delte
    x, y = center
    height, width = heatmap.shape

    if quadrant == "tl":
        # Top-Left corner: inside the box is x >= xtl, y >= ytl
        top = 0
        left = 0
        max_r_w = min(radius + 1, box_w + 1) if box_w is not None else radius + 1
        max_r_h = min(radius + 1, box_h + 1) if box_h is not None else radius + 1
        right = min(width - x, max_r_w)
        bottom = min(height - y, max_r_h)
    elif quadrant == "br":
        # Bottom-Right corner: inside the box is x <= xbr, y <= ybr
        bottom = 1
        right = 1
        max_r_w = min(radius, box_w) if box_w is not None else radius
        max_r_h = min(radius, box_h) if box_h is not None else radius
        top = min(y, max_r_h)
        left = min(x, max_r_w)
    else:
        raise ValueError(f"Unknown quadrant '{quadrant}'. Must be 'tl' or 'br'.")

    if left + right <= 0 or top + bottom <= 0:
        return

    # Generate Gaussian kernel strictly on the quadrant patch
    y_coords = torch.arange(-top, bottom, device=heatmap.device, dtype=heatmap.dtype).unsqueeze(1)
    x_coords = torch.arange(-left, right, device=heatmap.device, dtype=heatmap.dtype).unsqueeze(0)
    g = torch.exp(-(x_coords**2 + y_coords**2) / (2 * sigma * sigma))

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    torch.maximum(masked_heatmap, g * k, out=masked_heatmap)


# ==============================================================================
# 2. Masked Focal Loss (Guarantees Exactly Zero Gradient on Background)
# ==============================================================================

def _masked_neg_loss(pred_logits: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """
    Gaussian Focal Loss evaluated STRICTLY on masked pixels (inside bounding boxes).
    Background pixels (where mask == False) are completely excluded from the computation graph.
    Their autograd gradients are bit-exact 0.0.
    """
    if not mask.any():
        return torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype)

    pred = torch.sigmoid(pred_logits).clamp(min=1e-6, max=1.0 - 1e-6)

    # Strictly extract masked elements
    pred_sel = pred[mask]
    gt_sel = gt[mask]

    pos_inds = gt_sel.eq(1.0)
    neg_inds = gt_sel.lt(1.0)

    neg_weights = torch.pow(1.0 - gt_sel[neg_inds], 4)

    pos_loss = torch.log(pred_sel[pos_inds]) * torch.pow(1.0 - pred_sel[pos_inds], 2)
    neg_loss = torch.log((1.0 - pred_sel[neg_inds]).clamp(min=1e-6)) * torch.pow(pred_sel[neg_inds], 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_sum = pos_loss.sum()
    neg_sum = neg_loss.sum()

    if num_pos == 0:
        return -neg_sum
    else:
        return -(pos_sum + neg_sum) / num_pos


# ==============================================================================
# 3. Pure B-FOR Center Objectness Loss (ROS BCE + Soft-Argmax Centerness)
# ==============================================================================

def _bfor_center_loss(
    ct_logits: Tensor,
    gt_boxes_fmap: List[Tensor],  # List of [N_k, 4] in (x1, y1, x2, y2) on feature map
    alpha: float = 0.1,
    lambda_ctr: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """
    Pure B-FOR center objectness loss directly ported from BFOR_Loss (src/loss.py):
      - ros_loss: BCE between cropped prediction and anisotropic Gaussian soft target y_n
      - ctr_loss: Soft-argmax continuous centroid distance ((pred_cx - cx)^2 + (pred_cy - cy)^2) / (w * h)
    Evaluated STRICTLY on bounding box crops. Zero background pixels are involved.
    """
    ct_prob = torch.sigmoid(ct_logits)
    B, _, H, W = ct_logits.shape

    total_ros = torch.tensor(0.0, device=ct_logits.device, dtype=torch.float32)
    total_ctr = torch.tensor(0.0, device=ct_logits.device, dtype=torch.float32)
    num_boxes = 0

    for b in range(B):
        boxes = gt_boxes_fmap[b]
        for k in range(len(boxes)):
            x1, y1, x2, y2 = boxes[k].tolist()
            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            ix1 = max(0, min(W - 1, int(math.floor(x1))))
            ix2 = max(ix1 + 1, min(W, int(math.ceil(x2))))
            iy1 = max(0, min(H - 1, int(math.floor(y1))))
            iy2 = max(iy1 + 1, min(H, int(math.ceil(y2))))

            # Crop prediction strictly inside bounding box
            pred_crop = ct_prob[b, 0, iy1:iy2, ix1:ix2].clamp(min=1e-6, max=1.0 - 1e-6)

            # Meshgrid for B-FOR elliptical Gaussian soft target
            y_range = torch.arange(iy1, iy2, device=ct_logits.device, dtype=torch.float32)
            x_range = torch.arange(ix1, ix2, device=ct_logits.device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")

            # B-FOR elliptical soft target y_n
            dx = (grid_x - cx) / w
            dy = (grid_y - cy) / h
            y_n = torch.exp(-1.0 / (2.0 * alpha) * (dx**2 + dy**2))

            # 1. ROS Loss (Binary Cross-Entropy)
            ros = F.binary_cross_entropy(pred_crop, y_n, reduction="mean")

            # 2. Soft-argmax Centerness Loss
            weights = pred_crop / (torch.sum(pred_crop) + 1e-8)
            pred_cx = torch.sum(grid_x * weights)
            pred_cy = torch.sum(grid_y * weights)

            ctr = ((pred_cx - cx)**2 + (pred_cy - cy)**2) / (w * h)

            total_ros = total_ros + ros
            total_ctr = total_ctr + ctr
            num_boxes += 1

    if num_boxes == 0:
        zero_t = torch.tensor(0.0, device=ct_logits.device, dtype=torch.float32)
        return zero_t, zero_t

    return total_ros / num_boxes, total_ctr / num_boxes


# ==============================================================================
# 4. CenterNet-BFOR Loss Function
# ==============================================================================

class CenterNetBFORLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.1,
        lambda_ctr: float = 1.0,
        pull_weight: float = 0.1,
        push_weight: float = 0.1,
        regr_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.lambda_ctr = lambda_ctr
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
        tl_mask: Tensor,
        br_mask: Tensor,
        gt_mask: Tensor,
        gt_tl_regr: Tensor,
        gt_br_regr: Tensor,
        gt_ct_regr: Tensor,
        gt_boxes_fmap: List[Tensor],
    ) -> Dict[str, Tensor]:
        tl_heat = tl_heat.float()
        br_heat = br_heat.float()
        ct_heat = ct_heat.float()
        tl_tag = tl_tag.float()
        br_tag = br_tag.float()
        tl_regr = tl_regr.float()
        br_regr = br_regr.float()
        ct_regr = ct_regr.float()

        gt_tl_heat = gt_tl_heat.float()
        gt_br_heat = gt_br_heat.float()
        gt_tl_regr = gt_tl_regr.float()
        gt_br_regr = gt_br_regr.float()
        gt_ct_regr = gt_ct_regr.float()

        # Gather predicted tags and regrs at ground truth keypoint positions
        tl_tag_g = _tranpose_and_gather_feat(tl_tag, tl_inds)
        br_tag_g = _tranpose_and_gather_feat(br_tag, br_inds)

        tl_regr_g = _tranpose_and_gather_feat(tl_regr, tl_inds)
        br_regr_g = _tranpose_and_gather_feat(br_regr, br_inds)
        ct_regr_g = _tranpose_and_gather_feat(ct_regr, ct_inds)

        # 1. Masked Focal Loss on Corners (strictly inside bbox quadrants, zero bg gradient)
        focal_tl = _masked_neg_loss(tl_heat, gt_tl_heat, tl_mask)
        focal_br = _masked_neg_loss(br_heat, gt_br_heat, br_mask)
        corner_loss = focal_tl + focal_br

        # 2. Pure B-FOR Center Loss (strictly on bbox crops, zero bg gradient)
        ros_loss, ctr_loss = _bfor_center_loss(
            ct_heat, gt_boxes_fmap, alpha=self.alpha, lambda_ctr=self.lambda_ctr
        )
        center_loss = ros_loss + self.lambda_ctr * ctr_loss

        # 3. Associative Embedding push/pull loss (4D AE)
        pull, push = _ae_loss(tl_tag_g, br_tag_g, gt_mask)
        pull_loss = self.pull_weight * pull
        push_loss = self.push_weight * push

        # 4. Offset regression loss (Corners + Center sub-pixel refinement)
        regr_tl = _regr_loss(tl_regr_g, gt_tl_regr, gt_mask)
        regr_br = _regr_loss(br_regr_g, gt_br_regr, gt_mask)
        regr_ct = _regr_loss(ct_regr_g, gt_ct_regr, gt_mask)
        regr_loss = self.regr_weight * (regr_tl + regr_br + regr_ct)

        total_loss = corner_loss + center_loss + pull_loss + push_loss + regr_loss

        return {
            "loss_focal": corner_loss,
            "loss_bfor_ros": ros_loss,
            "loss_bfor_ctr": self.lambda_ctr * ctr_loss,
            "loss_pull": pull_loss,
            "loss_push": push_loss,
            "loss_regr": regr_loss,
        }


# ==============================================================================
# 5. CenterNet-BFOR Object Detector
# ==============================================================================

class CenterNetBFORDetector(nn.Module):
    """
    CenterNet Object Detector integrated with B-FOR Loss principles:
      - Class-Agnostic single objectness channel
      - 4D Associative Embedding vectors
      - Quarter-Gaussian corner Focal Loss strictly inside bounding box quadrants
      - Pure B-FOR Center Loss (ROS BCE + Soft-argmax centerness)
      - Guaranteed ZERO gradient on all background pixels outside bounding boxes
    """
    def __init__(
        self,
        alpha: float = 0.1,
        lambda_ctr: float = 1.0,
        ae_dim: int = 4,
        corner_mask_type: str = "quadrant",
        input_size: Tuple[int, int] = (512, 512),
        top_k: int = 70,
        ae_threshold: float = 0.5,
        nms_kernel: int = 3,
        nms_iou_threshold: float = 0.5,
        max_detections: int = 100,
        score_thresh: float = 0.05,
    ):
        super().__init__()
        self.num_classes = 1
        self.ae_dim = ae_dim
        self.alpha = alpha
        self.lambda_ctr = lambda_ctr
        self.corner_mask_type = corner_mask_type

        self.input_size = input_size  # (512, 512)
        self.output_size = (input_size[0] // 4, input_size[1] // 4)  # (128, 128)
        self.stride = 4

        self.top_k = top_k
        self.ae_threshold = ae_threshold
        self.nms_kernel = nms_kernel
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self.score_thresh = score_thresh

        # Backbone and prediction heads (1-channel heatmaps, 4D AE tags)
        self.net = CenterNet52Backbone(num_classes=1, ae_dim=ae_dim)
        self.loss_fn = CenterNetBFORLoss(
            alpha=alpha,
            lambda_ctr=lambda_ctr,
            pull_weight=0.1,
            push_weight=0.1,
            regr_weight=1.0,
        )

        # CenterNet RGB image normalization buffers
        self.register_buffer(
            "pixel_mean", torch.tensor([0.47026115, 0.44719302, 0.40789654]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor([0.27809834, 0.27408164, 0.28863828]).view(1, 3, 1, 1)
        )

    def _letterbox_image(
        self, img: Tensor
    ) -> Tuple[Tensor, float, float, float, int, int]:
        _, orig_h, orig_w = img.shape
        target_h, target_w = self.input_size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = min(target_w, int(round(orig_w * scale)))
        new_h = min(target_h, int(round(orig_h * scale)))

        resized = F.interpolate(
            img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
        ).squeeze(0)

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
    ):
        """
        Generates ground truth quadrant heatmaps, masks, tags, and offsets.
        Guarantees tl_mask and br_mask are strictly True inside bounding boxes,
        and False everywhere on the background.
        """
        B = len(targets)
        C = 1  # class-agnostic
        out_h, out_w = self.output_size
        max_tag_len = max(1, max(len(t["boxes"]) for t in targets))

        gt_tl_heat = torch.zeros((B, C, out_h, out_w), device=device, dtype=torch.float32)
        gt_br_heat = torch.zeros((B, C, out_h, out_w), device=device, dtype=torch.float32)

        tl_mask = torch.zeros((B, C, out_h, out_w), device=device, dtype=torch.bool)
        br_mask = torch.zeros((B, C, out_h, out_w), device=device, dtype=torch.bool)

        gt_mask = torch.zeros((B, max_tag_len), device=device, dtype=torch.bool)
        gt_tl_regr = torch.zeros((B, max_tag_len, 2), device=device, dtype=torch.float32)
        gt_br_regr = torch.zeros((B, max_tag_len, 2), device=device, dtype=torch.float32)
        gt_ct_regr = torch.zeros((B, max_tag_len, 2), device=device, dtype=torch.float32)

        tl_inds = torch.zeros((B, max_tag_len), device=device, dtype=torch.long)
        br_inds = torch.zeros((B, max_tag_len), device=device, dtype=torch.long)
        ct_inds = torch.zeros((B, max_tag_len), device=device, dtype=torch.long)

        gt_boxes_fmap: List[Tensor] = []

        for b_idx, (t, scale, px, py) in enumerate(zip(targets, scales, pad_xs, pad_ys)):
            boxes = t["boxes"]
            num_boxes = len(boxes)
            boxes_in_img = []

            if num_boxes == 0:
                gt_boxes_fmap.append(torch.zeros((0, 4), device=device, dtype=torch.float32))
                continue

            for k in range(num_boxes):
                x1, y1, x2, y2 = boxes[k].tolist()

                # Transform to 512x512 canvas
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
                ibox_w = max(1, xbr - xtl)
                ibox_h = max(1, ybr - ytl)

                radius = int(max(0, gaussian_radius((box_h, box_w), min_overlap=0.7)))

                # Draw Quarter-Gaussian confined strictly inside the box quadrant
                draw_quadrant_gaussian_2d(
                    gt_tl_heat[b_idx, 0], (xtl, ytl), radius, quadrant="tl", box_w=ibox_w, box_h=ibox_h, delte=6.0
                )
                draw_quadrant_gaussian_2d(
                    gt_br_heat[b_idx, 0], (xbr, ybr), radius, quadrant="br", box_w=ibox_w, box_h=ibox_h, delte=6.0
                )

                # Corner masks are True strictly inside the bounding box, False on all background
                if self.corner_mask_type == "quadrant":
                    # Strictly the corner quadrant patch inside the box
                    tl_mask[b_idx, 0, ytl : min(ybr + 1, ytl + radius + 1), xtl : min(xbr + 1, xtl + radius + 1)] = True
                    br_mask[b_idx, 0, max(ytl, ybr - radius) : ybr + 1, max(xtl, xbr - radius) : xbr + 1] = True
                else:
                    # The entire bounding box
                    tl_mask[b_idx, 0, ytl : ybr + 1, xtl : xbr + 1] = True
                    br_mask[b_idx, 0, ytl : ybr + 1, xtl : xbr + 1] = True

                gt_tl_regr[b_idx, k] = torch.tensor([fxtl - xtl, fytl - ytl], device=device)
                gt_br_regr[b_idx, k] = torch.tensor([fxbr - xbr, fybr - ybr], device=device)
                gt_ct_regr[b_idx, k] = torch.tensor([fxct - xct, fyct - yct], device=device)

                tl_inds[b_idx, k] = ytl * out_w + xtl
                br_inds[b_idx, k] = ybr * out_w + xbr
                ct_inds[b_idx, k] = yct * out_w + xct

                gt_mask[b_idx, k] = True
                boxes_in_img.append([fxtl, fytl, fxbr, fybr])

            if len(boxes_in_img) > 0:
                gt_boxes_fmap.append(torch.tensor(boxes_in_img, device=device, dtype=torch.float32))
            else:
                gt_boxes_fmap.append(torch.zeros((0, 4), device=device, dtype=torch.float32))

        return (
            gt_tl_heat,
            gt_br_heat,
            tl_mask,
            br_mask,
            gt_mask,
            gt_tl_regr,
            gt_br_regr,
            gt_ct_regr,
            tl_inds,
            br_inds,
            ct_inds,
            gt_boxes_fmap,
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

        # Letterbox batch preparation
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

        # Hourglass-52 Forward pass
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
        # Training Mode
        # ----------------------------------------------------
        if self.training and targets is not None:
            (
                gt_tl_heat,
                gt_br_heat,
                tl_mask,
                br_mask,
                gt_mask,
                gt_tl_regr,
                gt_br_regr,
                gt_ct_regr,
                tl_inds,
                br_inds,
                ct_inds,
                gt_boxes_fmap,
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
                tl_mask=tl_mask,
                br_mask=br_mask,
                gt_mask=gt_mask,
                gt_tl_regr=gt_tl_regr,
                gt_br_regr=gt_br_regr,
                gt_ct_regr=gt_ct_regr,
                gt_boxes_fmap=gt_boxes_fmap,
            )
            return loss_dict

        # ----------------------------------------------------
        # Evaluation Mode
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

            dets_i[:, :4] = dets_i[:, :4] * float(self.stride)
            cents_i[:, :2] = cents_i[:, :2] * float(self.stride)

            # Scale-aware central region verification
            verified_dets = apply_scale_aware_central_region(dets_i, cents_i)

            if len(verified_dets) == 0:
                results.append({
                    "boxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
                    "scores": torch.zeros((0,), device=device, dtype=torch.float32),
                    "labels": torch.zeros((0,), device=device, dtype=torch.int64),
                })
                continue

            x1 = (verified_dets[:, 0] - px) / scale
            y1 = (verified_dets[:, 1] - py) / scale
            x2 = (verified_dets[:, 2] - px) / scale
            y2 = (verified_dets[:, 3] - py) / scale

            x1 = x1.clamp(min=0.0, max=float(ow))
            x2 = x2.clamp(min=0.0, max=float(ow))
            y1 = y1.clamp(min=0.0, max=float(oh))
            y2 = y2.clamp(min=0.0, max=float(oh))

            scores = verified_dets[:, 4]
            labels = torch.ones(len(verified_dets), device=device, dtype=torch.int64)

            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if self.score_thresh > 0:
                valid = valid & (scores >= self.score_thresh)

            boxes = boxes[valid]
            scores = scores[valid]
            labels = labels[valid]

            if len(boxes) > 0 and self.nms_iou_threshold > 0:
                keep = nms(boxes, scores, self.nms_iou_threshold)
                keep = keep[: self.max_detections]
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            results.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })

        return results


def build_centernet_bfor_model(
    alpha: Optional[float] = None,
    lambda_ctr: Optional[float] = None,
    ae_dim: Optional[int] = None,
    input_size: Tuple[int, int] = (512, 512),
    **kwargs,
) -> CenterNetBFORDetector:
    """
    Factory function for CenterNet-BFOR model.
    """
    try:
        import run_config as cfg
        if alpha is None:
            alpha = getattr(cfg, "BFOR_ALPHA", 0.1)
        if lambda_ctr is None:
            lambda_ctr = getattr(cfg, "BFOR_LAMBDA_CTR", 1.0)
        if ae_dim is None:
            ae_dim = getattr(cfg, "CENTERNET_AE_DIM", 4)
        corner_mask_type = kwargs.pop("corner_mask_type", getattr(cfg, "CORNER_MASK_TYPE", "quadrant"))
    except Exception:
        if alpha is None:
            alpha = 0.1
        if lambda_ctr is None:
            lambda_ctr = 1.0
        if ae_dim is None:
            ae_dim = 4
        corner_mask_type = kwargs.pop("corner_mask_type", "quadrant")

    model = CenterNetBFORDetector(
        alpha=alpha,
        lambda_ctr=lambda_ctr,
        ae_dim=ae_dim,
        corner_mask_type=corner_mask_type,
        input_size=input_size,
        **kwargs,
    )
    return model
