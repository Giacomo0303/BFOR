import os
from random import randint

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib import patches
from torchvision import ops

import run_config as cfg
from src.datasets import PascalVOC
from src.model import BFOR_model
from src.train import load_model


def decode_predictions(
    out,
    r=9,
    K=1200,
    percentile=0.60,
    w_min=6.0,
    h_min=6.0,
    iou_thresh=0.5,
    max_detections=1000,
):
    """
    Decodes raw model outputs into ranked, class-agnostic bounding boxes
    following Section 3.2 of the B-FOR paper strictly.

    Args:
        out: Dictionary containing predictions for "sml", "med", "lrg".
             Each entry has {"obj": [1, 1, 448, 448], "w": [1, 1, 448, 448], "h": [1, 1, 448, 448]}
        r: Neighborhood kernel size for local peak extraction (default 9, from Eq. 2)
        K: Maximum number of peaks retained per scale level (default 1200)
        percentile: Score threshold quantile per scale level (default 0.60, 60th percentile)
        w_min, h_min: Minimum decoded box dimensions in pixels (default 6.0, Eq. 3)
        iou_thresh: IoU threshold for cross-scale NMS (default 0.5)
        max_detections: Maximum detections to retain after NMS (default 1000)

    Returns:
        final_boxes: Tensor of shape [N, 4] with [x1, y1, x2, y2] in [0, 448] canvas coordinates.
        final_scores: Tensor of shape [N] with objectness scores in descending order.
    """
    scales = ["sml", "med", "lrg"]
    all_boxes = []
    all_scores = []

    for scale in scales:
        # Step 1: Extract 2D maps for this scale (batch size is 1)
        obj = out[scale]["obj"][0, 0]  # [448, 448] objectness score
        w = out[scale]["w"][0, 0]  # [448, 448] normalized width
        h = out[scale]["h"][0, 0]  # [448, 448] normalized height

        # Step 2: Compute adaptive percentile threshold tau_l(I) (60th percentile)
        tau = torch.quantile(obj, percentile)

        # Step 3: Local peak extraction via max pooling
        pooled = F.max_pool2d(
            obj.unsqueeze(0).unsqueeze(0),
            kernel_size=r,
            stride=1,
            padding=r // 2,
        )[0, 0]

        # Element-wise boolean condition: local maximum AND >= tau (Eq. 2)
        is_peak = (obj == pooled) & (obj >= tau)

        # Step 4: Extract peak coordinates [y, x] and corresponding scores
        peak_coords = torch.nonzero(is_peak)  # [N_peaks, 2]
        scores = obj[is_peak]

        if len(scores) == 0:
            continue

        # Step 5: Retain top-K peaks per level ranked by score
        if len(scores) > K:
            topk_scores, topk_indices = torch.topk(scores, k=K)
            scores = topk_scores
            peak_coords = peak_coords[topk_indices]

        # Extract horizontal (x) and vertical (y) coordinates
        ys = peak_coords[:, 0].float()
        xs = peak_coords[:, 1].float()

        # Step 6: Decode spatial extent (w_q, h_q) from scale fields at peak coordinates (Eq. 3)
        w_vals = w[peak_coords[:, 0], peak_coords[:, 1]]
        h_vals = h[peak_coords[:, 0], peak_coords[:, 1]]

        box_w = torch.clamp(448.0 * w_vals, min=w_min)
        box_h = torch.clamp(448.0 * h_vals, min=h_min)

        # Convert center (xs, ys) + extent (box_w, box_h) -> [x1, y1, x2, y2]
        # and clip to the canvas boundaries [0, 448] (Eq. 3)
        x1 = torch.clamp(xs - box_w / 2.0, min=0.0, max=448.0)
        y1 = torch.clamp(ys - box_h / 2.0, min=0.0, max=448.0)
        x2 = torch.clamp(xs + box_w / 2.0, min=0.0, max=448.0)
        y2 = torch.clamp(ys + box_h / 2.0, min=0.0, max=448.0)

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        all_boxes.append(boxes)
        all_scores.append(scores)

    # If no peaks were found across any scale, return empty tensors
    if len(all_boxes) == 0:
        return torch.zeros((0, 4), device=out["sml"]["obj"].device), torch.zeros(
            (0,), device=out["sml"]["obj"].device
        )

    # Step 7: Merge candidate detections from all scales into a single pool
    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)

    # Step 8: Cross-scale Non-Maximum Suppression (NMS) with IoU threshold 0.5
    keep = ops.nms(all_boxes, all_scores, iou_threshold=iou_thresh)

    if max_detections is not None:
        keep = keep[:max_detections]

    final_boxes = all_boxes[keep]
    final_scores = all_scores[keep]

    return final_boxes, final_scores


def visualize_inference(
    img_tensor,
    pred_boxes,
    pred_scores,
    gt_boxes=None,
    save_path="inference_result.png",
    top_k=25,
    min_box_size=6.0,
):
    """
    Visualizes an image with ground-truth (green) and predicted boxes (red/yellow).
    If a prediction matches a ground-truth object with IoU >= 0.5, it is highlighted.

    Args:
        img_tensor: Tensor of shape [3, 448, 448] or [1, 3, 448, 448] in range [0, 1].
        pred_boxes: Tensor of shape [N, 4] with [x1, y1, x2, y2] in [0, 448].
        pred_scores: Tensor of shape [N] with confidence scores.
        gt_boxes: Tensor of shape [M, 4] with normalized ground truth [cx, cy, w, h] in [0, 1].
        save_path: Output file path for the plot.
        top_k: Number of highest-confidence predictions to consider (default 25).
        min_box_size: Minimum width and height (in pixels) to display, filtering tiny corner specks.
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]

    # Convert CHW torch tensor to HWC numpy array
    img_np = img_tensor.cpu().permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_np)

    # 1. Convert Ground Truth boxes to pixel corners [x1, y1, x2, y2]
    gt_corners = []
    if gt_boxes is not None and len(gt_boxes) > 0:
        for gt in gt_boxes:
            cx, cy, w, h = (gt * 448.0).cpu().numpy()
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            gt_corners.append([x1, y1, x2, y2])
            rect = patches.Rectangle(
                (x1, y1),
                w,
                h,
                linewidth=3.0,
                edgecolor="#00FF00",
                facecolor="none",
                linestyle="-",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                max(0, y1 - 4),
                "GT (Unseen)",
                color="white",
                backgroundcolor="#008800",
                fontsize=9,
                weight="bold",
            )

    # 2. Filter predictions by minimum size and take top_k
    widths = pred_boxes[:, 2] - pred_boxes[:, 0]
    heights = pred_boxes[:, 3] - pred_boxes[:, 1]
    size_mask = (widths >= min_box_size) & (heights >= min_box_size)

    filtered_boxes = pred_boxes[size_mask][:top_k]
    filtered_scores = pred_scores[size_mask][:top_k]

    # 3. Calculate IoU against GT to identify hits
    max_iou_val = 0.0
    best_rank = None
    if len(gt_corners) > 0 and len(filtered_boxes) > 0:
        gt_t = torch.tensor(gt_corners, device=pred_boxes.device, dtype=torch.float32)
        ious = ops.box_iou(filtered_boxes, gt_t)  # [N_pred, M_gt]
        max_iou_per_pred, _ = torch.max(ious, dim=1)
        max_iou_val, max_pred_idx = torch.max(max_iou_per_pred, dim=0)
        max_iou_val = max_iou_val.item()
        best_rank = max_pred_idx.item() + 1
    else:
        max_iou_per_pred = torch.zeros(len(filtered_boxes))

    # 4. Draw predicted boxes
    for i in range(len(filtered_boxes)):
        box = filtered_boxes[i].cpu().numpy()
        score = filtered_scores[i].item()
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        iou_with_gt = max_iou_per_pred[i].item() if len(max_iou_per_pred) > 0 else 0.0
        is_hit = iou_with_gt >= 0.5

        if is_hit:
            # Highlight successful detection in YELLOW / GOLD
            color = "#FFD700"
            linewidth = 3.0
            label = f"HIT (IoU={iou_with_gt:.2f}) #{i + 1}"
            bg_color = "#B8860B"
            ax.text(
                x1,
                min(440, y2 - 4),
                label,
                color="black",
                backgroundcolor=color,
                fontsize=8,
                weight="bold",
            )
        else:
            color = "#FF3333"
            linewidth = 1.2
            bg_color = "#CC0000"

        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            alpha=0.85,
        )
        ax.add_patch(rect)

        # Show score text only for top 15 or hits to prevent text clutter
        if i < 15 or is_hit:
            ax.text(
                x1,
                max(0, y1 - 3),
                f"{score:.2f}",
                color="white",
                backgroundcolor=bg_color,
                fontsize=7,
                weight="bold",
            )

    title_hit_info = (
        f" | Max IoU: {max_iou_val:.2f} (Rank #{best_rank})" if best_rank else ""
    )
    ax.set_title(
        f"B-FOR (Green = GT Unseen, Yellow = IoU>=0.5 Hit, Red = Top {len(filtered_boxes)} Proposals{title_hit_info})",
        fontsize=11,
        weight="bold",
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"--> Saved visualization to: {save_path} (Max IoU: {max_iou_val:.2f})")


def execute_inference(
    checkpoint_path="best_model.pt", save_path="inference_result.png"
):
    """
    Loads test set, runs inference on a sample containing unseen categories,
    decodes boxes, and saves the visualization.
    """
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"

    print("Loading PASCAL VOC test set...")
    test_set = PascalVOC(path=cfg.DATA_PATH, split="test")

    # Initialize model
    model = BFOR_model(n_channels=cfg.N_CHANNELS, drop_rate=cfg.DROP_RATE).to(device)

    # Load trained weights if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from: {checkpoint_path}")
        load_model(model, path=checkpoint_path)
    else:
        print(
            f"[Warning] Checkpoint '{checkpoint_path}' not found! Running with uninitialized weights."
        )

    model.eval()

    # Find a test sample that contains unseen objects (boat, cow, or tvmonitor)
    # so the visualization shows meaningful comparisons
    found_idx = None
    for _ in range(50):
        candidate_idx = randint(0, len(test_set) - 1)
        _, target = test_set[candidate_idx]
        if len(target) > 0:
            found_idx = candidate_idx
            break

    if found_idx is None:
        found_idx = randint(0, len(test_set) - 1)

    print(f"Testing on image index: {found_idx}")
    img, target = test_set[found_idx]
    img_batch = img.unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        out = model(img_batch)

    # Decode predictions strictly following the paper (Section 3.2)
    pred_boxes, pred_scores = decode_predictions(
        out=out,
        r=cfg.K,
        K=1200,
        percentile=0.60,
        w_min=6.0,
        h_min=6.0,
        iou_thresh=0.5,
        max_detections=1000,
    )

    print(f"Detections after NMS: {len(pred_boxes)} boxes")
    if len(pred_boxes) > 0:
        print(
            f"Top-1 score: {pred_scores[0]:.4f} | Lowest score: {pred_scores[-1]:.4f}"
        )

    # Visualize and save (top 25 proposals)
    visualize_inference(
        img_tensor=img,
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        gt_boxes=target,
        save_path=save_path,
        top_k=25,
        min_box_size=6.0,
    )


if __name__ == "__main__":
    execute_inference()
