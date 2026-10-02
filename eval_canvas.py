import argparse
import os
import torch
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm

import run_config as cfg
from src.datasets import COCO2017, PascalVOC
from src.inference_utils import decode_predictions
from src.model import BFOR_model
from src.train import load_model


def compute_AR(model, test_set, device, minIoU=0.5, max_detections=1000):
    stats = {
        target_class: {"hits": 0, "total": 0}
        for target_class in sorted(test_set.target_classes)
    }
    size_stats = {
        "sml": {"hits": 0, "total": 0},  # < 64px
        "med": {"hits": 0, "total": 0},  # 64px <= size < 96px
        "lrg": {"hits": 0, "total": 0},  # >= 96px
    }

    model.eval()

    with torch.no_grad():
        for x, y, labels in tqdm(test_set, desc="Evaluating"):
            if len(y) == 0:
                continue

            x = x.unsqueeze(0).to(device)

            # y is [N, 4] in cxcywh normalized in [0, 1]
            y_boxes_448 = y * 448.0
            y_xyxy = box_convert(y_boxes_448.to(device), in_fmt="cxcywh", out_fmt="xyxy")

            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                out = model(x)

            pred_boxes, _pred_scores = decode_predictions(
                out, iou_thresh=0.5, max_detections=max_detections
            )

            if len(pred_boxes) == 0:
                max_iou_per_gt = torch.zeros(len(y), device=device)
            else:
                ious = box_iou(pred_boxes, y_xyxy)
                max_iou_per_gt, _ = torch.max(ious, dim=0)

            for idx in range(len(labels)):
                lbl = labels[idx]
                if lbl not in stats:
                    continue

                stats[lbl]["total"] += 1
                is_hit = max_iou_per_gt[idx].item() >= minIoU
                if is_hit:
                    stats[lbl]["hits"] += 1

                # Size breakdown based on paper's scale definitions
                w_px = float(y_boxes_448[idx, 2])
                h_px = float(y_boxes_448[idx, 3])
                size = (w_px * h_px) ** 0.5

                if size < 64.0:
                    scale_key = "sml"
                elif size >= 96.0:
                    scale_key = "lrg"
                else:
                    scale_key = "med"

                size_stats[scale_key]["total"] += 1
                if is_hit:
                    size_stats[scale_key]["hits"] += 1

    # Formatted results table
    print("\n" + "=" * 55)
    print(f"{'Class':<16} | {'Hits':>6} / {'Total':<6} | {'Recall (AR@1000)':>16}")
    print("-" * 55)

    class_recalls = []
    total_hits = 0
    total_gt = 0
    for cls in sorted(stats.keys()):
        hits = stats[cls]["hits"]
        tot = stats[cls]["total"]
        if tot > 0:
            r = (hits / tot) * 100.0
            class_recalls.append(r)
            total_hits += hits
            total_gt += tot
            print(f"{cls:<16} | {hits:>6} / {tot:<6} | {r:>15.1f}%")
        else:
            print(f"{cls:<16} | {hits:>6} / {tot:<6} | {'N/A':>16}")

    macro_avg = sum(class_recalls) / len(class_recalls) if class_recalls else 0.0
    micro_avg = (total_hits / total_gt) * 100.0 if total_gt > 0 else 0.0

    print("=" * 55)
    print(f"{'Macro Average':<16} | {'-':>6}   {'-':<6} | {macro_avg:>15.1f}%")
    print(f"{'Overall AR':<16} | {total_hits:>6} / {total_gt:<6} | {micro_avg:>15.1f}%")
    print("=" * 55)

    print("\nSize-wise Recall Breakdown:")
    for sk, sname in [("sml", "Small (<64px)"), ("med", "Medium (64-96px)"), ("lrg", "Large (>=96px)")]:
        shits = size_stats[sk]["hits"]
        stot = size_stats[sk]["total"]
        sr = (shits / stot) * 100.0 if stot > 0 else 0.0
        print(f"  {sname:<18}: {shits:>5} / {stot:<5} ({sr:.1f}%)")
    print("=" * 55 + "\n")

    return micro_avg


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate B-FOR on Pascal VOC or MS-COCO 2017 test sets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["voc", "coco"],
        default="voc",
        help="Dataset to evaluate on: 'voc' or 'coco' (default: 'voc').",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=getattr(cfg, "DATA_PATH", "Data"),
        help="Path to datasets root folder (default: cfg.DATA_PATH).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to evaluate on (default: 'test').",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: cfg.SAVE_PATH or best_model.pt)",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        default=False,
        help="Evaluate on all classes (20 for VOC, 80 for COCO).",
    )
    parser.add_argument(
        "--unseen-only",
        action="store_true",
        default=False,
        help="Evaluate on unseen classes only (3 for VOC, 60 for COCO).",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if os.path.exists(cfg.SAVE_PATH):
            checkpoint_path = cfg.SAVE_PATH
        elif os.path.exists("best_model_voc20.pt"):
            checkpoint_path = "best_model_voc20.pt"
        elif os.path.exists("best_model.pt"):
            checkpoint_path = "best_model.pt"
        else:
            checkpoint_path = cfg.SAVE_PATH

    if args.unseen_only:
        eval_all = False
    elif args.all_classes:
        eval_all = True
    else:
        # Default behavior when neither flag is explicitly passed
        eval_all = getattr(cfg, "ALL_CLASSES", True) if args.dataset == "voc" else False

    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"

    if args.dataset == "coco":
        mode_str = (
            "All 80 Categories"
            if eval_all
            else "60 Unseen Categories (VOC20 -> COCO60)"
        )
        print(f"Loading MS-COCO 2017 {args.split} set [{mode_str}]...")
        test_set = COCO2017(
            path=args.data_path,
            split=args.split,
            all_classes=eval_all,
        )
    else:
        mode_str = (
            "All 20 Categories"
            if eval_all
            else "3 Unseen Categories (17/3 split)"
        )
        print(f"Loading PASCAL VOC {args.split} set [{mode_str}]...")
        test_set = PascalVOC(
            path=args.data_path,
            split=args.split,
            all_classes=eval_all,
        )

    model = BFOR_model(n_channels=cfg.N_CHANNELS, drop_rate=cfg.DROP_RATE).to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from: {checkpoint_path}")
        load_model(model, path=checkpoint_path)
    else:
        print(
            f"[Warning] Checkpoint '{checkpoint_path}' not found! Running with uninitialized weights."
        )

    compute_AR(model=model, test_set=test_set, device=device)


if __name__ == "__main__":
    main()
