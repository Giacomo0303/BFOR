import argparse
import copy
import json
import os
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_convert, box_iou, nms
from tqdm import tqdm

import run_config as cfg
from src.datasets import COCO2017, PascalVOC
from src.inference_utils import decode_predictions
from src.model import BFOR_model
from src.train import load_model


# =====================================================================
# Official Author Evaluation Functions (from fpn_coco_evaluation.py)
# =====================================================================

def filter_boxes_not_on_padding_448(
    boxes_xyxy_s,
    meta,
    keep_mode="overlap",
    min_valid_overlap=0.90,
):
    """
    Direct port of author's filter_boxes_not_on_padding_448.
    Removes detections whose overlap with the real image area is below threshold.
    """
    scale = float(meta["scale"])
    pad_left = float(meta["pad_left"])
    pad_top = float(meta["pad_top"])
    orig_w = float(meta["orig_w"])
    orig_h = float(meta["orig_h"])

    valid_x1 = pad_left * scale
    valid_y1 = pad_top * scale
    valid_x2 = (pad_left + orig_w) * scale
    valid_y2 = (pad_top + orig_h) * scale

    kept = []
    for x1, y1, x2, y2, s in boxes_xyxy_s:
        x1, y1, x2, y2, s = map(float, (x1, y1, x2, y2, s))
        if x2 <= x1 or y2 <= y1:
            continue

        if keep_mode == "center":
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            if valid_x1 <= cx <= valid_x2 and valid_y1 <= cy <= valid_y2:
                kept.append([x1, y1, x2, y2, s])
        elif keep_mode == "overlap":
            ix1 = max(x1, valid_x1)
            iy1 = max(y1, valid_y1)
            ix2 = min(x2, valid_x2)
            iy2 = min(y2, valid_y2)

            inter_w = max(0.0, ix2 - ix1)
            inter_h = max(0.0, iy2 - iy1)
            inter_area = inter_w * inter_h

            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if box_area > 0 and inter_area / box_area >= min_valid_overlap:
                kept.append([x1, y1, x2, y2, s])
        else:
            raise ValueError("keep_mode must be 'center' or 'overlap'")

    return kept


def boxes_padded_to_orig_xyxy(boxes_xyxy_s, meta):
    """
    Direct port of author's boxes_padded_to_orig_xyxy.
    Converts 448x448 padded box coordinates back to original unpadded image resolution.
    """
    scale = float(meta["scale"])
    pad_left = float(meta["pad_left"])
    pad_top = float(meta["pad_top"])
    orig_w = float(meta["orig_w"])
    orig_h = float(meta["orig_h"])

    out = []
    for x1, y1, x2, y2, s in boxes_xyxy_s:
        x1o = x1 / scale - pad_left
        y1o = y1 / scale - pad_top
        x2o = x2 / scale - pad_left
        y2o = y2 / scale - pad_top

        x1o = float(np.clip(x1o, 0.0, orig_w))
        x2o = float(np.clip(x2o, 0.0, orig_w))
        y1o = float(np.clip(y1o, 0.0, orig_h))
        y2o = float(np.clip(y2o, 0.0, orig_h))

        if x2o > x1o and y2o > y1o:
            out.append([x1o, y1o, x2o, y2o, float(s)])
    return out


def prepare_coco_gt(test_set):
    """
    Creates a class-agnostic ground truth COCO object containing
    only the target classes (e.g. 60 unseen categories) with category_id = 1.
    Matches instances_val2017_no_voc_classagnostic.json from the paper.
    """
    target_classes = test_set.target_classes
    target_cat_ids = {
        cat_id for cat_id, cat in test_set.coco.cats.items()
        if cat["name"] in target_classes
    }

    filtered_anns = []
    for ann in test_set.coco.dataset.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann["category_id"] in target_cat_ids:
            new_ann = copy.deepcopy(ann)
            new_ann["category_id"] = 1
            filtered_anns.append(new_ann)

    gt_dict = {
        "images": copy.deepcopy(test_set.coco.dataset.get("images", [])),
        "categories": [{"id": 1, "name": "object"}],
        "annotations": filtered_anns,
    }

    gt_coco = COCO()
    gt_coco.dataset = gt_dict
    gt_coco.createIndex()
    return gt_coco


def print_ar50_breakdown(e: COCOeval, iou=0.5, max_det=1000):
    """
    Direct port of author's print_ar50_breakdown from fpn_coco_evaluation.py lines 704-731.
    """
    recall = e.eval["recall"]
    iou_thrs = e.params.iouThrs
    t = int(np.where(np.isclose(iou_thrs, iou))[0][0])
    M = e.params.maxDets.index(max_det)

    def mean_valid(x):
        x = x[x > -1]
        return float(x.mean()) if x.size else float("nan")

    if recall.ndim == 5:  # (T,R,K,A,M)
        ar_all = mean_valid(recall[t, :, 0, 0, M])
        ar_small = mean_valid(recall[t, :, 0, 1, M])
        ar_med = mean_valid(recall[t, :, 0, 2, M])
        ar_large = mean_valid(recall[t, :, 0, 3, M])
    elif recall.ndim == 4:  # (T,R,A,M)
        ar_all = mean_valid(recall[t, :, 0, M])
        ar_small = mean_valid(recall[t, :, 1, M])
        ar_med = mean_valid(recall[t, :, 2, M])
        ar_large = mean_valid(recall[t, :, 3, M])
    else:
        raise RuntimeError(f"Unexpected recall shape: {recall.shape}")

    print("\n" + "=" * 55)
    print(f"OFFICIAL COCOeval AR@{iou} Breakdown (maxDets={max_det}):")
    print("-" * 55)
    print(f"  Overall (AROvr) : {ar_all * 100:.1f}%")
    print(f"  Small   (ARSml) : {ar_small * 100:.1f}%")
    print(f"  Medium  (ARMed) : {ar_med * 100:.1f}%")
    print(f"  Large   (ARLrg) : {ar_large * 100:.1f}%")
    print("=" * 55 + "\n")
    return ar_all


def evaluate_coco_official(
    model,
    test_set,
    device,
    iou=0.5,
    max_detections=1000,
    save_json=None,
    model_type="bfor",
):
    """
    Full official evaluation pipeline replicating fpn_coco_evaluation.py:
    - B-FOR: Letterbox 448x448 inference, padding filter, unpad to original resolution, COCOeval(useCats=0).
    - FCOS: Native resolution inference, class-agnostic NMS, COCOeval(useCats=0).
    """
    model.eval()
    print(f"Preparing Ground Truth annotations for {len(test_set.target_classes)} target categories...")
    gt_coco = prepare_coco_gt(test_set)

    results_list = []
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    with torch.no_grad():
        for idx in tqdm(
            range(len(test_set)),
            desc=f"Evaluating (Official COCO - {model_type.upper()})",
        ):
            img_id = test_set.dataset.ids[idx]

            if model_type in ["fcos", "bfor_fcos"]:
                tensor_img, _, _ = test_set[idx]
                x = [tensor_img.to(device)]

                with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                    out = model(x)

                pred_boxes = out[0]["boxes"]
                pred_scores = out[0]["scores"]

                if len(pred_boxes) == 0:
                    continue

                # Class-agnostic NMS to filter cross-class redundant proposals
                keep = nms(pred_boxes, pred_scores, iou_threshold=0.6)
                keep = keep[:max_detections]
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

                for b_idx in range(len(pred_boxes)):
                    x1, y1, x2, y2 = pred_boxes[b_idx].tolist()
                    s_val = pred_scores[b_idx].item()
                    w_box = x2 - x1
                    h_box = y2 - y1
                    if w_box <= 0.0 or h_box <= 0.0:
                        continue
                    results_list.append({
                        "image_id": int(img_id),
                        "category_id": 1,
                        "bbox": [round(x1, 2), round(y1, 2), round(w_box, 2), round(h_box, 2)],
                        "score": float(s_val),
                    })

            else:
                img_info = test_set.coco.loadImgs(img_id)[0]
                orig_w = float(img_info["width"])
                orig_h = float(img_info["height"])
                s = max(orig_w, orig_h)
                pad_left = (s - orig_w) // 2
                pad_top = (s - orig_h) // 2
                scale = 448.0 / float(s)

                meta = {
                    "orig_w": orig_w,
                    "orig_h": orig_h,
                    "square_size": s,
                    "pad_left": pad_left,
                    "pad_top": pad_top,
                    "scale": scale,
                }

                tensor_img, _, _ = test_set[idx]
                x = tensor_img.unsqueeze(0).to(device)

                with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                    out = model(x)

                pred_boxes, pred_scores = decode_predictions(
                    out, iou_thresh=0.5, max_detections=max_detections
                )

                if len(pred_boxes) == 0:
                    continue

                # [[x1, y1, x2, y2, score], ...]
                boxes_448 = []
                for b_idx in range(len(pred_boxes)):
                    box = pred_boxes[b_idx].tolist()
                    score = pred_scores[b_idx].item()
                    boxes_448.append([box[0], box[1], box[2], box[3], score])

                # Step 1: Filter boxes overlapping black padding bands
                boxes_448_filtered = filter_boxes_not_on_padding_448(
                    boxes_448, meta, keep_mode="overlap", min_valid_overlap=0.90
                )

                # Step 2: Unpad back to original image resolution
                boxes_orig = boxes_padded_to_orig_xyxy(boxes_448_filtered, meta)

                # Step 3: Format predictions for COCOeval
                for x1, y1, x2, y2, s_val in boxes_orig:
                    w_box = x2 - x1
                    h_box = y2 - y1
                    if w_box <= 0.0 or h_box <= 0.0:
                        continue
                    results_list.append({
                        "image_id": int(img_id),
                        "category_id": 1,
                        "bbox": [round(x1, 2), round(y1, 2), round(w_box, 2), round(h_box, 2)],
                        "score": float(s_val),
                    })

    if save_json:
        print(f"Saving predictions to: {save_json}")
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(results_list, f)

    if len(results_list) == 0:
        print("[Warning] No predictions produced!")
        return 0.0

    print(f"\nTotal predicted boxes across dataset: {len(results_list)}")
    res_coco = gt_coco.loadRes(results_list)
    coco_eval = COCOeval(gt_coco, res_coco, iouType="bbox")
    coco_eval.params.useCats = 0
    coco_eval.params.maxDets = [1, 10, max_detections]
    coco_eval.params.iouThrs = np.array([iou], dtype=np.float32)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return print_ar50_breakdown(coco_eval, iou=iou, max_det=max_detections)


# =====================================================================
# Canvas-Based Evaluation Function (with Per-Class Breakdown)
# =====================================================================

def compute_AR(
    model, test_set, device, minIoU=0.5, max_detections=1000, model_type="bfor"
):
    stats = {
        target_class: {"hits": 0, "total": 0}
        for target_class in sorted(test_set.target_classes)
    }
    size_stats = {
        "sml": {"hits": 0, "total": 0},  # < 32px
        "med": {"hits": 0, "total": 0},  # 32px <= size < 96px
        "lrg": {"hits": 0, "total": 0},  # >= 96px
    }

    model.eval()
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    with torch.no_grad():
        for item in tqdm(
            test_set,
            desc=f"Evaluating ({'Canvas' if model_type == 'bfor' else 'FCOS Native'})",
        ):
            if model_type in ["fcos", "bfor_fcos"]:
                tensor_img, target_dict, labels = item
                gt_boxes = target_dict["boxes"].to(device)  # [N, 4] in xyxy native
                if len(gt_boxes) == 0:
                    continue

                x = [tensor_img.to(device)]
                with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
                    out = model(x)

                pred_boxes = out[0]["boxes"]
                pred_scores = out[0]["scores"]

                if len(pred_boxes) == 0:
                    max_iou_per_gt = torch.zeros(len(gt_boxes), device=device)
                else:
                    keep = nms(pred_boxes, pred_scores, iou_threshold=0.6)
                    keep = keep[:max_detections]
                    pred_boxes = pred_boxes[keep]
                    ious = box_iou(pred_boxes, gt_boxes)
                    max_iou_per_gt, _ = torch.max(ious, dim=0)

                for idx in range(len(labels)):
                    lbl = labels[idx]
                    if lbl not in stats:
                        continue

                    stats[lbl]["total"] += 1
                    is_hit = max_iou_per_gt[idx].item() >= minIoU
                    if is_hit:
                        stats[lbl]["hits"] += 1

                    w_px = float(gt_boxes[idx, 2] - gt_boxes[idx, 0])
                    h_px = float(gt_boxes[idx, 3] - gt_boxes[idx, 1])
                    size = (w_px * h_px) ** 0.5

                    if size < 32.0:
                        scale_key = "sml"
                    elif size >= 96.0:
                        scale_key = "lrg"
                    else:
                        scale_key = "med"

                    size_stats[scale_key]["total"] += 1
                    if is_hit:
                        size_stats[scale_key]["hits"] += 1

            else:
                x, y, labels = item
                if len(y) == 0:
                    continue

                x = x.unsqueeze(0).to(device)

                # y is [N, 4] in cxcywh normalized in [0, 1]
                y_boxes_448 = y * 448.0
                y_xyxy = box_convert(
                    y_boxes_448.to(device), in_fmt="cxcywh", out_fmt="xyxy"
                )

                with torch.amp.autocast(device_type=device_type, dtype=torch.float16):
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

                    if size < 32.0:
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
    for sk, sname in [("sml", "Small (<32px)"), ("med", "Medium (32-96px)"), ("lrg", "Large (>=96px)")]:
        shits = size_stats[sk]["hits"]
        stot = size_stats[sk]["total"]
        sr = (shits / stot) * 100.0 if stot > 0 else 0.0
        print(f"  {sname:<18}: {shits:>5} / {stot:<5} ({sr:.1f}%)")
    print("=" * 55 + "\n")

    return micro_avg


# =====================================================================
# Main CLI Entrypoint
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate B-FOR or FCOS on Pascal VOC or MS-COCO 2017 test sets."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["bfor", "fcos", "bfor_fcos"],
        default=getattr(cfg, "MODEL_NAME", "bfor"),
        help="Model architecture: 'bfor', 'fcos', or 'bfor_fcos' (default: cfg.MODEL_NAME).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["voc", "coco"],
        default="voc",
        help="Dataset to evaluate on: 'voc' or 'coco' (default: 'voc').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["official", "canvas"],
        default="official",
        help="Evaluation mode: 'official' (author's COCOeval on original image resolution) or 'canvas' (448x448 canvas). Default: 'official'.",
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
        help="Path to checkpoint (default: based on model selection).",
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
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save COCO predictions JSON.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if args.model == "bfor_fcos":
            if os.path.exists("best_bfor_fcos_voc20.pt"):
                checkpoint_path = "best_bfor_fcos_voc20.pt"
            elif os.path.exists(cfg.SAVE_PATH):
                checkpoint_path = cfg.SAVE_PATH
            else:
                checkpoint_path = "best_bfor_fcos_voc20.pt"
        elif args.model == "fcos":
            if os.path.exists("best_fcos_voc20.pt"):
                checkpoint_path = "best_fcos_voc20.pt"
            elif os.path.exists(cfg.SAVE_PATH):
                checkpoint_path = cfg.SAVE_PATH
            else:
                checkpoint_path = "best_fcos_voc20.pt"
        else:
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
        print(f"Loading MS-COCO 2017 {args.split} set [{mode_str}] (model={args.model})...")
        test_set = COCO2017(
            path=args.data_path,
            split=args.split,
            all_classes=eval_all,
            model_type=args.model,
        )
    else:
        mode_str = (
            "All 20 Categories"
            if eval_all
            else "3 Unseen Categories (17/3 split)"
        )
        print(f"Loading PASCAL VOC {args.split} set [{mode_str}] (model={args.model})...")
        test_set = PascalVOC(
            path=args.data_path,
            split=args.split,
            all_classes=eval_all,
            model_type=args.model,
        )

    if args.model == "bfor_fcos":
        from src.fcos_model import build_bfor_fcos_model

        model = build_bfor_fcos_model(
            alpha=getattr(cfg, "ALPHA", 0.1),
            pretrained_backbone=False,
            score_thresh=0.0,
            detections_per_img=1000,
        ).to(device)
    elif args.model == "fcos":
        from src.fcos_model import build_fcos_model

        model = build_fcos_model(
            num_classes=getattr(cfg, "FCOS_NUM_CLASSES", 21),
            pretrained_backbone=False,
            score_thresh=0.0,
            detections_per_img=1000,
        ).to(device)
    else:
        model = BFOR_model(n_channels=cfg.N_CHANNELS, drop_rate=cfg.DROP_RATE).to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from: {checkpoint_path}")
        load_model(model, path=checkpoint_path)
    else:
        print(
            f"[Warning] Checkpoint '{checkpoint_path}' not found! Running with uninitialized weights."
        )

    if args.dataset == "coco" and args.mode == "official":
        print(f"Running OFFICIAL COCOeval Protocol ({args.model.upper()})...")
        evaluate_coco_official(
            model=model,
            test_set=test_set,
            device=device,
            iou=0.5,
            max_detections=1000,
            save_json=args.save_json,
            model_type=args.model,
        )
    else:
        compute_AR(
            model=model,
            test_set=test_set,
            device=device,
            model_type=args.model,
        )


if __name__ == "__main__":
    main()
