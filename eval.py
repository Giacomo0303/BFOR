from src.model import BFOR_model
from src.inference_utils import decode_predictions, boxes_padded_to_orig_xyxy
import run_config as cfg
import torch
from src.datasets import PascalVOC
from src.train import load_model
import os
from tqdm import tqdm
from torchvision.ops import box_iou, box_convert


def compute_AR(model, test_set, device, minIoU=0.5, max_detections=1000):
    stats = {
        target_class: {"hits": 0, "total": 0}
        for target_class in test_set.target_classes
    }

    model.eval()

    with torch.no_grad():
        for item in tqdm(test_set):
            x, y, labels = item[0], item[1], item[2]
            meta = item[3] if len(item) > 3 else None

            if len(labels) == 0:
                continue

            x = x.unsqueeze(0).to(device)

            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                out = model(x)

            # 1. Decode predictions on 448 canvas with letterbox padding filter
            pred_boxes, _pred_scores = decode_predictions(
                out, iou_thresh=0.5, max_detections=max_detections, meta=meta
            )

            # 2. Map predictions and GT back to original image space
            if meta is not None and "orig_boxes" in meta:
                gt_boxes = meta["orig_boxes"].to(device)
                pred_boxes_eval = boxes_padded_to_orig_xyxy(pred_boxes, meta)
            else:
                gt_boxes = box_convert(
                    y.to(device) * 448.0, in_fmt="cxcywh", out_fmt="xyxy"
                )
                pred_boxes_eval = pred_boxes

            if len(pred_boxes_eval) == 0 or len(gt_boxes) == 0:
                max_iou_per_gt = torch.zeros(len(labels), device=device)
            else:
                ious = box_iou(pred_boxes_eval, gt_boxes)
                max_iou_per_gt, _ = torch.max(ious, dim=0)

            for idx in range(len(labels)):
                stats[labels[idx]]["total"] += 1

                if max_iou_per_gt[idx] >= minIoU:
                    stats[labels[idx]]["hits"] += 1

    total_hits = 0
    total_gt = 0
    for cls in stats:
        total_hits += stats[cls]["hits"]
        total_gt += stats[cls]["total"]
        if stats[cls]["total"] > 0:
            recall = (stats[cls]["hits"] / stats[cls]["total"]) * 100
            print(f"{cls}: {recall:.1f}%")
        else:
            print(f"{cls}: N/A (0 ground truth)")

    if total_gt > 0:
      recall = (total_hits / total_gt) * 100
      print(f"Average: {recall:.1f}%")
    else:
      print("Average: N/A")


def main(checkpoint_path="best_model.pt"):

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

    compute_AR(model=model, test_set=test_set, device=cfg.DEVICE)


if __name__ == "__main__":
    main()
