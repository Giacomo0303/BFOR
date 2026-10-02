import argparse
import os
from random import randint

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import run_config as cfg
from src.datasets import PascalVOC
from src.inference_utils import decode_predictions, visualize_inference
from src.model import BFOR_model
from src.train import load_model


def preprocess_custom_image(image_path):
    """
    Loads any image and applies the exact B-FOR preprocessing:
    - Symmetrically pads to a square canvas (black padding)
    - Resizes to 448x448
    - Converts to normalized torch tensor [1, 3, 448, 448] in [0, 1]

    Returns:
        img_tensor: [3, 448, 448] torch.Tensor
        meta: dict with original dimensions and padding information
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    s = max(orig_w, orig_h)

    pad_left = (s - orig_w) // 2
    pad_right = s - orig_w - pad_left
    pad_top = (s - orig_h) // 2
    pad_bottom = s - h - pad_top if (s - orig_h) else 0
    pad_bottom = s - orig_h - pad_top

    padded_img = F.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0)
    resized_img = F.resize(padded_img, [448, 448])
    img_tensor = F.to_tensor(resized_img)

    meta = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "scale": 448.0 / s,
    }
    return img_tensor, meta


def run_inference(
    image_path=None,
    test_index=None,
    checkpoint_path=None,
    save_path="inference_result.png",
    top_k=20,
    iou_thresh=0.5,
    max_detections=1000,
):
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"

    # 1. Resolve checkpoint path
    if checkpoint_path is None:
        if hasattr(cfg, "SAVE_PATH") and os.path.exists(cfg.SAVE_PATH):
            checkpoint_path = cfg.SAVE_PATH
        elif os.path.exists("best_model_voc20.pt"):
            checkpoint_path = "best_model_voc20.pt"
        elif os.path.exists("best_model.pt"):
            checkpoint_path = "best_model.pt"
        else:
            checkpoint_path = "best_model.pt"

    # 2. Initialize model and load weights
    model = BFOR_model(n_channels=cfg.N_CHANNELS, drop_rate=cfg.DROP_RATE).to(device)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from: {checkpoint_path}")
        load_model(model, path=checkpoint_path)
    else:
        print(f"[Warning] Checkpoint '{checkpoint_path}' not found! Running with uninitialized weights.")

    model.eval()

    # 3. Load input image
    target = None
    labels = None

    if image_path is not None:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        print(f"Running inference on custom image: {image_path}")
        img_tensor, meta = preprocess_custom_image(image_path)
        print(f"Original image size: {meta['orig_w']}x{meta['orig_h']} --> Resized to 448x448")
    else:
        print("Loading PASCAL VOC test set...")
        test_set = PascalVOC(path=cfg.DATA_PATH, split="test")

        if test_index is not None:
            found_idx = test_index
        else:
            # Pick a random sample with ground truth objects
            found_idx = None
            for _ in range(50):
                cand = randint(0, len(test_set) - 1)
                _, tgt, _ = test_set[cand]
                if len(tgt) > 0:
                    found_idx = cand
                    break
            if found_idx is None:
                found_idx = randint(0, len(test_set) - 1)

        print(f"Testing on PASCAL VOC test set image index: {found_idx}")
        img_tensor, target, labels = test_set[found_idx]
        if labels:
            print(f"Ground truth categories present: {labels}")

    # 4. Model forward pass
    img_batch = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.float32):
            out = model(img_batch)

    # 5. Decode predictions
    pred_boxes, pred_scores = decode_predictions(
        out=out,
        r=cfg.K,
        K=1200,
        percentile=0.60,
        w_min=6.0,
        h_min=6.0,
        iou_thresh=iou_thresh,
        max_detections=max_detections,
    )

    print(f"\nDetections after NMS: {len(pred_boxes)} boxes")
    if len(pred_boxes) > 0:
        print(f"Top-1 confidence score: {pred_scores[0]:.4f} | Lowest score: {pred_scores[-1]:.4f}")
        print(f"\nTop-{min(top_k, 5)} Proposal Coordinates [x1, y1, x2, y2] (448 canvas):")
        for i in range(min(top_k, 5)):
            b = pred_boxes[i].cpu().numpy()
            s = pred_scores[i].item()
            print(f"  #{i+1}: [{b[0]:5.1f}, {b[1]:5.1f}, {b[2]:5.1f}, {b[3]:5.1f}]  score={s:.3f}")

    # 6. Visualize and save
    visualize_inference(
        img_tensor=img_tensor,
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
        gt_boxes=target,
        save_path=save_path,
        top_k=top_k,
        min_box_size=6.0,
    )
    print(f"\nVisualization saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Run B-FOR inference on an arbitrary image or test sample.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to any image file (e.g. photo.jpg). If omitted, a sample from VOC test set is used.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index in the PASCAL VOC test set (used if --image is not provided).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (defaults to best_model_voc20.pt or best_model.pt).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="inference_result.png",
        help="Output file path for the visualization (default: inference_result.png).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top proposals to draw on the image (default: 15).",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5).",
    )

    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        test_index=args.index,
        checkpoint_path=args.checkpoint,
        save_path=args.save_path,
        top_k=args.top_k,
        iou_thresh=args.iou_thresh,
    )


if __name__ == "__main__":
    main()
