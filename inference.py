import os
from random import randint

import torch

import run_config as cfg
from src.datasets import PascalVOC
from src.inference_utils import decode_predictions, visualize_inference
from src.model import BFOR_model
from src.train import load_model


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
        item = test_set[candidate_idx]
        target = item[1]
        if len(target) > 0:
            found_idx = candidate_idx
            break

    if found_idx is None:
        found_idx = randint(0, len(test_set) - 1)

    print(f"Testing on image index: {found_idx}")
    sample = test_set[found_idx]
    img, target = sample[0], sample[1]
    meta = sample[3] if len(sample) > 3 else None
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
        meta=meta,
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
