#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_results_metrics.py — Evaluate inference results metrics

Metrics:
  1. Detection Accuracy (normal vs. anomalous)
  2. Anomaly Type Accuracy — abnormal samples only
  3. Mean IoU — samples with predicted bbox only
  4. End-to-End Accuracy — detection + type + IoU > 0.5
  5. Per-sub-dataset breakdown
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def compute_iou(b1: List[float], b2: List[float]) -> float:
    """Compute IoU between two bounding boxes."""
    if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
        return 0.0
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
    a1 = max(0., b1[2] - b1[0]) * max(0., b1[3] - b1[1])
    a2 = max(0., b2[2] - b2[0]) * max(0., b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def normalize_bbox(bbox) -> Optional[List[float]]:
    """Normalize bbox to [0, 1] range."""
    if bbox is None:
        return None
    if isinstance(bbox, list):
        if len(bbox) == 0:
            return None
        if isinstance(bbox[0], list):
            bbox = bbox[0]
        if len(bbox) != 4:
            return None
        try:
            vals = [float(v) for v in bbox]
            # If all values <= 1, already in [0,1] range
            if all(v <= 1.0 for v in vals):
                return [max(0.0, min(1.0, v)) for v in vals]
            # Otherwise convert from [0, 1000]
            return [max(0.0, min(1.0, v / 1000.0)) for v in vals]
        except Exception:
            return None
    return None


def load_results(path: Path) -> List[Dict]:
    """Load JSONL results file."""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except Exception:
                continue
    return results


def load_gt_dataset(path: Path) -> Tuple[Dict[str, Dict], Dict[str, List[float]]]:
    """
    Load GT dataset file, returns:
      - qid -> item mapping (includes gt_answer, anomaly_type, etc.)
      - qid -> bbox mapping (normalized bboxes)

    Bboxes in the dataset are in [0, 1000] range and need to be converted to [0, 1].
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    qid_to_item = {}
    qid_to_bbox = {}

    for item in data:
        qid = item.get("qid") or item.get("id")
        if not qid:
            continue

        qid_to_item[qid] = item

        # Process bbox
        bbox = item.get("bbox")
        if bbox is not None:
            normalized = normalize_bbox(bbox)
            if normalized:
                qid_to_bbox[qid] = normalized

    return qid_to_item, qid_to_bbox


def evaluate_metrics(results: List[Dict],
                      gt_items: Optional[Dict[str, Dict]] = None,
                      gt_bboxes: Optional[Dict[str, List[float]]] = None) -> Dict:
    """
    Compute all metrics.

    Args:
        results: List of inference results
        gt_items: qid -> item mapping (loaded from GT dataset)
        gt_bboxes: qid -> bbox mapping (loaded from GT dataset)

    Returns:
        - det_acc: Detection accuracy
        - type_acc: Anomaly type accuracy (abnormal samples only)
        - mean_iou: Mean IoU (samples with bbox only)
        - e2e_acc: End-to-end accuracy (abnormal samples only)
        - loc_acc: Localization accuracy (IoU >= 0.5 among abnormal samples, regardless of type)
        - e2e_full_acc: End-to-end full accuracy (normal samples correctly classified + abnormal samples with correct detection+type+localization)
        - by_class: Per-class breakdown
    """
    total = len(results)
    if total == 0:
        return {"det_acc": 0, "type_acc": 0, "mean_iou": 0, "e2e_acc": 0, "loc_acc": 0, "e2e_full_acc": 0, "by_class": {}}

    # Counters
    det_correct = 0          # Detection correct count
    det_abnormal_correct = 0 # Abnormal samples predicted correctly
    det_normal_correct = 0   # Normal samples predicted correctly
    n_abnormal = 0           # Total abnormal samples
    n_normal = 0             # Total normal samples
    type_correct = 0         # Type classification correct (abnormal samples only)
    iou_sum = 0.0            # IoU sum
    iou_count = 0            # Samples with bbox
    e2e_correct = 0          # End-to-end correct (detection+type+localization all correct)
    loc_correct = 0          # Localization correct (IoU >= 0.5 among abnormal samples)
    e2e_full_correct = 0     # End-to-end full correct

    # Per-sub-dataset statistics
    class_stats = defaultdict(lambda: {
        "total": 0, "n_normal": 0, "n_abnormal": 0,
        "det_correct": 0, "det_abnormal_correct": 0, "det_normal_correct": 0,
        "type_correct": 0, "e2e_correct": 0,
        "loc_correct": 0, "e2e_full_correct": 0
    })

    for r in results:
        # Extract sub-dataset name from qid (format: "test_goodsad_1" -> "goodsad")
        metadata = r.get("metadata", {})
        qid = metadata.get("qid", "") or r.get("qid", "")
        
        # Parse sub-dataset name
        if qid:
            # Format: "test_goodsad_1" or "train_miad_xxx", etc.
            parts = qid.split("_")
            if len(parts) >= 2:
                # Skip test/train prefix, take second part as dataset name
                dataset_name = parts[1]
            else:
                dataset_name = qid
        else:
            dataset_name = "unknown"

        # GT info
        # Prefer GT dataset, fallback to result file
        if gt_items and qid in gt_items:
            gt_item = gt_items[qid]
            gt_present = bool(gt_item.get("gt_answer"))
            gt_type = gt_item.get("anomaly_type") or ""
            # Parse correct_answer from anomaly_list
            anomaly_list = gt_item.get("anomaly_list", [])
            gt_letter = ""
            if gt_type and anomaly_list:
                for entry in anomaly_list:
                    if isinstance(entry, str) and ". " in entry:
                        letter, desc = entry.split(". ", 1)
                        if gt_type.lower() in desc.lower() or desc.lower() in gt_type.lower():
                            gt_letter = letter.strip()
                            break
            # Get bbox from GT dataset
            gt_bbox = gt_bboxes.get(qid) if gt_bboxes else None
        else:
            # Fallback: get from result file (backward compatibility)
            gt_present = bool(r.get("gt_answer"))
            gt_type = r.get("gt_type", "") or ""
            gt_letter = r.get("correct_answer", "") or ""
            gt_bbox = normalize_bbox(r.get("gt_bbox"))

        # Prediction info
        pred_present = bool(r.get("pred_present"))
        pred_type = str(r.get("pred_type", "") or "").strip()
        pred_bbox = normalize_bbox(r.get("pred_bbox"))

        # Update sub-dataset statistics
        class_stats[dataset_name]["total"] += 1

        # 1. Detection accuracy
        if pred_present == gt_present:
            det_correct += 1
            class_stats[dataset_name]["det_correct"] += 1

        if gt_present:
            # Abnormal sample
            n_abnormal += 1
            class_stats[dataset_name]["n_abnormal"] += 1
            # Abnormal sample detected: prediction is also abnormal
            if pred_present:
                det_abnormal_correct += 1
                class_stats[dataset_name]["det_abnormal_correct"] += 1

            # Check if type classification is correct
            type_ok = False
            if gt_letter:
                # Prefer correct_answer (letter)
                type_ok = pred_type.upper() == gt_letter.upper()
            elif gt_type:
                # Fallback: fuzzy match with gt_type
                type_ok = pred_type.lower() in gt_type.lower()

            # Correct only if predicted as abnormal AND type is correct
            if pred_present and type_ok:
                type_correct += 1
                class_stats[dataset_name]["type_correct"] += 1

            # 3. IoU (requires both GT bbox and pred bbox)
            iou = 0.0
            if gt_bbox and pred_bbox:
                iou = compute_iou(pred_bbox, gt_bbox)
                iou_sum += iou
                iou_count += 1

            # 4. Localization accuracy (IoU >= 0.5 among abnormal samples, regardless of type)
            if pred_bbox and gt_bbox and iou >= 0.5:
                loc_correct += 1
                class_stats[dataset_name]["loc_correct"] += 1

            # 5. End-to-end accuracy (abnormal samples only)
            if pred_present == gt_present and type_ok and iou >= 0.5:
                e2e_correct += 1
                class_stats[dataset_name]["e2e_correct"] += 1

            # 6. End-to-end full accuracy (abnormal sample part)
            if pred_present and type_ok and iou >= 0.5:
                e2e_full_correct += 1
                class_stats[dataset_name]["e2e_full_correct"] += 1

        else:
            # Normal sample
            n_normal += 1
            class_stats[dataset_name]["n_normal"] += 1
            # Normal sample detected: prediction is also normal
            if not pred_present:
                det_normal_correct += 1
                class_stats[dataset_name]["det_normal_correct"] += 1
                e2e_full_correct += 1
                class_stats[dataset_name]["e2e_full_correct"] += 1

    # Compute metrics
    det_acc = det_correct / total if total > 0 else 0.0
    det_abnormal_acc = det_abnormal_correct / n_abnormal if n_abnormal > 0 else 0.0
    det_normal_acc = det_normal_correct / n_normal if n_normal > 0 else 0.0
    type_acc = type_correct / n_abnormal if n_abnormal > 0 else 0.0
    mean_iou = iou_sum / iou_count if iou_count > 0 else 0.0
    e2e_acc = e2e_correct / n_abnormal if n_abnormal > 0 else 0.0
    loc_acc = loc_correct / n_abnormal if n_abnormal > 0 else 0.0
    e2e_full_acc = e2e_full_correct / total if total > 0 else 0.0

    # Compute per-sub-dataset metrics
    by_class = {}
    for ds_name, stats in class_stats.items():
        ds_total = stats["total"]
        ds_abnormal = stats["n_abnormal"]
        ds_normal = stats["n_normal"]
        by_class[ds_name] = {
            "total": ds_total,
            "n_normal": ds_normal,
            "n_abnormal": ds_abnormal,
            "det_acc": stats["det_correct"] / ds_total if ds_total > 0 else 0.0,
            "det_abnormal_acc": stats["det_abnormal_correct"] / ds_abnormal if ds_abnormal > 0 else 0.0,
            "det_normal_acc": stats["det_normal_correct"] / ds_normal if ds_normal > 0 else 0.0,
            "type_acc": stats["type_correct"] / ds_abnormal if ds_abnormal > 0 else 0.0,
            "loc_acc": stats["loc_correct"] / ds_abnormal if ds_abnormal > 0 else 0.0,
            "e2e_acc": stats["e2e_correct"] / ds_abnormal if ds_abnormal > 0 else 0.0,
            "e2e_full_acc": stats["e2e_full_correct"] / ds_total if ds_total > 0 else 0.0,
        }

    return {
        "total": total,
        "n_normal": n_normal,
        "n_abnormal": n_abnormal,
        "det_correct": det_correct,
        "det_abnormal_correct": det_abnormal_correct,
        "det_normal_correct": det_normal_correct,
        "type_correct": type_correct,
        "iou_count": iou_count,
        "e2e_correct": e2e_correct,
        "loc_correct": loc_correct,
        "e2e_full_correct": e2e_full_correct,
        "det_acc": det_acc,
        "det_abnormal_acc": det_abnormal_acc,
        "det_normal_acc": det_normal_acc,
        "type_acc": type_acc,
        "mean_iou": mean_iou,
        "e2e_acc": e2e_acc,
        "loc_acc": loc_acc,
        "e2e_full_acc": e2e_full_acc,
        "by_class": by_class,
    }


def print_report(m: Dict):
    """Print evaluation report."""
    lines = [
        "=" * 60,
        "Evaluation Report",
        "=" * 60,
        f"Total samples       : {m['total']}  "
        f"(abnormal={m['n_abnormal']}, normal={m['n_normal']})",
        "",
        f"Detection Accuracy  : {m['det_correct']}/{m['total']} = {m['det_acc']*100:.2f}%",
        f"  - Abnormal recall : {m['det_abnormal_correct']}/{m['n_abnormal']} = {m['det_abnormal_acc']*100:.2f}%",
        f"  - Normal recall   : {m['det_normal_correct']}/{m['n_normal']} = {m['det_normal_acc']*100:.2f}%",
        "",
        f"Type Accuracy       : {m['type_correct']}/{m['n_abnormal']} = {m['type_acc']*100:.2f}%",
        f"  (abnormal samples only; misclassifying as normal counts as wrong)",
        "",
        f"Mean IoU            : {m['mean_iou']:.4f}  "
        f"(over {m['iou_count']} samples with bbox)",
        "",
        f"Localization Acc.   : {m['loc_correct']}/{m['n_abnormal']} = {m['loc_acc']*100:.2f}%",
        f"  (IoU >= 0.5 among abnormal samples, regardless of type)",
        "",
        f"End-to-End Acc.     : {m['e2e_correct']}/{m['n_abnormal']} = {m['e2e_acc']*100:.2f}%",
        f"  (abnormal only: detection + type + IoU >= 0.5)",
        "",
        f"E2E Full Accuracy   : {m['e2e_full_correct']}/{m['total']} = {m['e2e_full_acc']*100:.2f}%",
        f"  (normal correctly classified + abnormal detection+type+localization all correct)",
    ]

    # Per-sub-dataset statistics
    if m.get("by_class"):
        lines.extend([
            "",
            "=" * 60,
            "Per-Dataset Accuracy",
            "=" * 60,
            f"{'Dataset':<20} {'Total':>6} {'Abnrm':>6} {'Det':>8} {'Type':>8} {'Loc':>8} {'E2E':>8}",
            "-" * 74,
        ])
        # Sort by dataset name
        for ds_name in sorted(m["by_class"].keys()):
            stats = m["by_class"][ds_name]
            lines.append(
                f"{ds_name:<20} {stats['total']:>6} {stats['n_abnormal']:>6} "
                f"{stats['det_acc']*100:>7.2f}% {stats['type_acc']*100:>7.2f}% "
                f"{stats['loc_acc']*100:>7.2f}% {stats['e2e_full_acc']*100:>7.2f}%"
            )

    lines.append("=" * 60)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Evaluate inference results metrics")
    ap.add_argument("--input", "-i", required=True, help="Results JSONL file")
    ap.add_argument("--gt-json", "-g", default=None,
                    help="GT dataset JSON file (for bbox and other GT info)")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading results: {input_path}", file=sys.stderr)
    results = load_results(input_path)
    print(f"Loaded {len(results)} records", file=sys.stderr)

    # Load GT dataset (optional)
    gt_items = None
    gt_bboxes = None
    if args.gt_json:
        gt_path = Path(args.gt_json)
        if gt_path.exists():
            print(f"Loading GT dataset: {gt_path}", file=sys.stderr)
            gt_items, gt_bboxes = load_gt_dataset(gt_path)
            print(f"Loaded {len(gt_items)} GT records, {len(gt_bboxes)} bboxes", file=sys.stderr)
        else:
            print(f"Warning: GT file not found {gt_path}", file=sys.stderr)

    if not results:
        print("Error: result file is empty, no records to evaluate", file=sys.stderr)
        sys.exit(1)

    metrics = evaluate_metrics(results, gt_items, gt_bboxes)
    print_report(metrics)


if __name__ == "__main__":
    main()
