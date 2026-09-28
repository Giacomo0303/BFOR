import argparse
import os
import random
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.ops import box_iou
from PIL import Image

import run_config as cfg
from src.datasets import PascalVOC
from src.inference_utils import decode_predictions
from src.model import BFOR_model
from src.train import load_model


def preprocess_image(image_input):
    """
    Preprocesses an image (file path, PIL Image, or torch Tensor) to match the model input:
    - Symmetric padding to square canvas (black background)
    - Resize to 448x448
    - Convert to normalized PyTorch tensor [1, 3, 448, 448] in [0, 1]

    Returns:
        tensor_img: [1, 3, 448, 448] torch.Tensor
        display_img: [448, 448, 3] numpy array in [0, 1] for visualization
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, torch.Tensor):
        if image_input.dim() == 3:
            tensor = image_input.unsqueeze(0)
        else:
            tensor = image_input
        display = tensor[0].permute(1, 2, 0).cpu().numpy()
        return tensor, display
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    w, h = img.size
    s = max(w, h)

    pad_left = (s - w) // 2
    pad_right = s - w - pad_left
    pad_top = (s - h) // 2
    pad_bottom = s - h - pad_top

    padded_img = F.pad(
        img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0
    )
    resized_img = F.resize(padded_img, [448, 448])
    tensor_img = F.to_tensor(resized_img).unsqueeze(0)
    display_img = np.array(resized_img, dtype=np.float32) / 255.0

    return tensor_img, display_img


def extract_all_maps(model, image_input, device=None, pixel_scale=True):
    """
    Feeds an image to the model and extracts all 9 output maps + raw output:
    For each scale ('sml', 'med', 'lrg'):
        - 'obj': Objectness / ROS heatmap [448, 448] in [0, 1]
        - 'w': Width map [448, 448] (in pixels if pixel_scale else [0, 1])
        - 'h': Height map [448, 448] (in pixels if pixel_scale else [0, 1])

    Returns:
        maps: dict with structure maps[scale][head] -> 2D numpy array [448, 448]
        display_img: [448, 448, 3] numpy array
        raw_out: dict output from model forward pass
    """
    if device is None:
        device = cfg.DEVICE if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    tensor_img, display_img = preprocess_image(image_input)
    tensor_img = tensor_img.to(device)

    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        ):
            out = model(tensor_img)

    scale_mult = 448.0 if pixel_scale else 1.0

    maps = {}
    for scale in ["sml", "med", "lrg"]:
        maps[scale] = {
            "obj": out[scale]["obj"][0, 0].cpu().to(torch.float32).numpy(),
            "w": (out[scale]["w"][0, 0].cpu().to(torch.float32).numpy()) * scale_mult,
            "h": (out[scale]["h"][0, 0].cpu().to(torch.float32).numpy()) * scale_mult,
        }

    return maps, display_img, out


def draw_gt_boxes(ax, gt_boxes, labels=None):
    """Draws ground truth boxes in bright green on the given axis."""
    if gt_boxes is None or len(gt_boxes) == 0:
        return

    for idx, box in enumerate(gt_boxes):
        cx, cy, w, h = (box * 448.0).cpu().numpy()
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0

        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=2.5,
            edgecolor="#00FF00",
            facecolor="none",
            linestyle="-",
        )
        ax.add_patch(rect)

        label_text = f"GT: {labels[idx]}" if labels and idx < len(labels) else "GT"
        ax.text(
            x1,
            max(12, y1 - 4),
            label_text,
            color="white",
            backgroundcolor="#008800",
            fontsize=8,
            weight="bold",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#008800", edgecolor="none"),
        )


def draw_predictions(ax, pred_boxes, pred_scores, gt_boxes=None, top_k=10):
    """Draws the top-k decoded proposals on the given axis."""
    if len(pred_boxes) == 0:
        ax.text(
            224,
            224,
            "No detections",
            color="white",
            ha="center",
            va="center",
            fontsize=12,
            backgroundcolor="black",
        )
        return

    num_draw = min(top_k, len(pred_boxes))
    top_boxes = pred_boxes[:num_draw].cpu()
    top_scores = pred_scores[:num_draw].cpu()

    # Precompute IoUs with GT if GT is available
    matched = [False] * num_draw
    if gt_boxes is not None and len(gt_boxes) > 0:
        gt_xyxy = []
        for box in gt_boxes:
            cx, cy, w, h = (box * 448.0).cpu()
            gt_xyxy.append([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])
        gt_tensor = torch.tensor(gt_xyxy, dtype=torch.float32)
        ious = box_iou(top_boxes, gt_tensor)
        for i in range(num_draw):
            if ious[i].max().item() >= 0.5:
                matched[i] = True

    for i in range(num_draw):
        x1, y1, x2, y2 = top_boxes[i].numpy()
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        s = top_scores[i].item()

        is_hit = matched[i]
        edge_color = "#FFD700" if is_hit else "#FF3333"  # Gold if hit, Red if false positive
        badge_bg = "#B8860B" if is_hit else "#CC0000"
        status_tag = " (HIT)" if is_hit else ""

        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=2.0,
            edgecolor=edge_color,
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            min(440, max(12, y2 - 4)),
            f"s={s:.2f}{status_tag}",
            color="white",
            fontsize=7,
            weight="bold",
            bbox=dict(boxstyle="square,pad=0.15", facecolor=badge_bg, edgecolor="none", alpha=0.85),
        )


def visualize_full_grid(
    display_img,
    maps,
    raw_out,
    gt_boxes=None,
    labels=None,
    save_path="heatmaps.png",
    colormap="jet",
    pixel_scale=True,
):
    """
    Renders a 4x3 grid:
    - Row 0: Top Overview Row
        * (0, 0): Preprocessed Input Image with Ground Truth boxes (green)
        * (0, 1): Combined Multi-Scale Objectness Heatmap Overlay on Image
        * (0, 2): Final Decoded Proposals (Top predictions after NMS)
    - Row 1: Small Scale (P3)  -> Objectness, Width, Height
    - Row 2: Medium Scale (P4) -> Objectness, Width, Height
    - Row 3: Large Scale (P5)  -> Objectness, Width, Height
    """
    scales = [("sml", "Small Scale (P3)"), ("med", "Medium Scale (P4)"), ("lrg", "Large Scale (P5)")]
    heads = [("obj", "Objectness"), ("w", "Width"), ("h", "Height")]
    unit_label = "px" if pixel_scale else "norm"

    fig, axes = plt.subplots(4, 3, figsize=(18, 22))

    # =========================================================================
    # ROW 0: Overview Plots
    # =========================================================================

    # Plot (0, 0): Input Image + GT Boxes
    axes[0, 0].imshow(display_img)
    draw_gt_boxes(axes[0, 0], gt_boxes, labels)
    lbl_text = f" ({', '.join(labels)})" if labels else ""
    axes[0, 0].set_title(f"Input Image (448x448) + Ground Truth{lbl_text}", fontsize=12, weight="bold")
    axes[0, 0].axis("off")

    # Plot (0, 1): Combined Max Objectness Heatmap Overlay
    sml_obj = maps["sml"]["obj"]
    med_obj = maps["med"]["obj"]
    lrg_obj = maps["lrg"]["obj"]
    combined_obj = np.maximum(np.maximum(sml_obj, med_obj), lrg_obj)

    axes[0, 1].imshow(display_img)
    im_comb = axes[0, 1].imshow(combined_obj, cmap=colormap, alpha=0.45, vmin=0.0, vmax=1.0)
    draw_gt_boxes(axes[0, 1], gt_boxes, labels=None)
    axes[0, 1].set_title(
        f"Combined Max Objectness Overlay\n[max: {combined_obj.max():.3f}]",
        fontsize=12,
        weight="bold",
    )
    axes[0, 1].axis("off")
    fig.colorbar(im_comb, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Plot (0, 2): Final Decoded Proposals after NMS
    axes[0, 2].imshow(display_img)
    draw_gt_boxes(axes[0, 2], gt_boxes, labels=None)

    pred_boxes, pred_scores = decode_predictions(
        out=raw_out,
        r=cfg.K,
        K=1200,
        percentile=0.60,
        w_min=6.0,
        h_min=6.0,
        iou_thresh=0.5,
        max_detections=1000,
    )
    draw_predictions(axes[0, 2], pred_boxes, pred_scores, gt_boxes=gt_boxes, top_k=10)
    axes[0, 2].set_title(
        f"Final Decoded Proposals (Top-10 after NMS)\n[Total: {len(pred_boxes)} detections]",
        fontsize=12,
        weight="bold",
    )
    axes[0, 2].axis("off")

    # =========================================================================
    # ROWS 1-3: The 9 Detailed Heatmaps (3 Scales x 3 Heads)
    # =========================================================================
    for row_offset, (scale_key, scale_name) in enumerate(scales):
        row_idx = row_offset + 1
        for col_idx, (head_key, head_name) in enumerate(heads):
            ax = axes[row_idx, col_idx]
            data = maps[scale_key][head_key]

            if head_key == "obj":
                im = ax.imshow(data, cmap=colormap, vmin=0.0, vmax=1.0)
                title = f"{scale_name} - {head_name}\n[min: {data.min():.3f}, max: {data.max():.3f}]"
            else:
                vmax = 448.0 if pixel_scale else 1.0
                im = ax.imshow(data, cmap=colormap, vmin=0.0, vmax=vmax)
                title = f"{scale_name} - {head_name} ({unit_label})\n[min: {data.min():.1f}, max: {data.max():.1f}]"

            ax.set_title(title, fontsize=11, weight="bold")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("B-FOR Model Visualization (Overview Row + 9 Multi-Scale Heatmaps)", fontsize=16, weight="bold", y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Full visualization grid successfully saved to: {save_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize B-FOR outputs: Top overview row + 9 multi-scale heatmaps."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file. If omitted, a sample from Pascal VOC test set is used.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index in the Pascal VOC test set (used if --image is not provided).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Path to trained model weights (default: best_model.pt).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="heatmaps.png",
        help="Output file path for saving the figure (default: heatmaps.png).",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Matplotlib colormap (e.g. jet, inferno, viridis, magma).",
    )
    parser.add_argument(
        "--normalized-scale",
        action="store_true",
        help="Display width and height normalized in [0, 1] instead of pixels in [0, 448].",
    )

    args = parser.parse_args()

    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"

    # 1. Initialize model and load weights
    model = BFOR_model(n_channels=cfg.N_CHANNELS, drop_rate=cfg.DROP_RATE).to(device)
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        load_model(model, path=args.checkpoint)
    else:
        print(f"[Warning] Checkpoint '{args.checkpoint}' not found. Using uninitialized weights.")

    # 2. Select image input
    gt_boxes = None
    labels = None

    if args.image:
        image_input = args.image
        print(f"Using image from file: {image_input}")
    else:
        print("Loading Pascal VOC test set...")
        test_set = PascalVOC(path=cfg.DATA_PATH, split="test")

        if args.index is not None:
            idx = args.index
            print(f"Using specified Pascal VOC test set sample at index {idx}")
        else:
            # Randomly select a test sample containing unseen objects (boat, cow, tvmonitor)
            found_idx = None
            for _ in range(100):
                cand = random.randint(0, len(test_set) - 1)
                _, target, _ = test_set[cand]
                if len(target) > 0:
                    found_idx = cand
                    break
            idx = found_idx if found_idx is not None else random.randint(0, len(test_set) - 1)
            print(f"Randomly selected Pascal VOC test set sample at index: {idx}")

        image_input, gt_boxes, labels = test_set[idx]
        print(f"Ground truth labels in image: {labels}")

    # 3. Extract all 9 maps + raw model output
    pixel_scale = not args.normalized_scale
    maps, display_img, raw_out = extract_all_maps(
        model=model,
        image_input=image_input,
        device=device,
        pixel_scale=pixel_scale,
    )

    # 4. Print stats to console
    unit = "px" if pixel_scale else "norm"
    print("\nSummary of all 9 Output Heatmaps:")
    for scale in ["sml", "med", "lrg"]:
        print(f"--- Scale: {scale.upper()} ---")
        for head in ["obj", "w", "h"]:
            m = maps[scale][head]
            print(f"  {head.upper():3s}: min = {m.min():.4f}, max = {m.max():.4f}, mean = {m.mean():.4f} {unit if head != 'obj' else ''}")

    # 5. Render the 4x3 grid
    visualize_full_grid(
        display_img=display_img,
        maps=maps,
        raw_out=raw_out,
        gt_boxes=gt_boxes,
        labels=labels,
        save_path=args.save_path,
        colormap=args.colormap,
        pixel_scale=pixel_scale,
    )


if __name__ == "__main__":
    main()
