#!/usr/bin/env python3
"""
analyze_multiturn_effect.py — Analyze the effect of multi-turn tool calls.

Three statistics are computed:
  1. Normal queried: Among normal samples that called inspect(),
     the proportion that ultimately output "normal".
  2. Type correction: Anomaly samples where at least one intermediate
     query was wrong but the final answer type was correct, as a
     proportion of all multi-turn anomaly samples.
  3. IoU improvement: Anomaly samples where the final answer IoU is
     strictly greater than the intermediate crop IoU, as a
     proportion of all multi-turn anomaly samples with a GT bbox.

Usage:
    python analyze_multiturn_effect.py --input outputs/outputs_airiad-4b-zs.jsonl
    python analyze_multiturn_effect.py --input outputs/outputs_airiad-8b-os.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ===== bbox / IoU utilities =====

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
            if all(v <= 1.0 for v in vals):
                return [max(0.0, min(1.0, v)) for v in vals]
            return [max(0.0, min(1.0, v / 1000.0)) for v in vals]
        except Exception:
            return None
    return None


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


def check_type_correct(pred_type: str, gt_type: str, correct_answer: str) -> bool:
    """Check if predicted type matches the ground truth."""
    if not pred_type or pred_type.lower() == "none":
        return False
    if correct_answer:
        return pred_type.upper() == correct_answer.upper()
    if gt_type:
        return pred_type.lower() in gt_type.lower()
    return False


# ===== Tool call extraction =====

def extract_tool_calls(messages: List[Dict]) -> List[Dict]:
    """Extract inspect(crop, query) tool calls from conversation messages.

    Returns a list of dicts with keys: turn, crop, query.
    """
    tool_calls = []
    turn = 0
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if "Tool result for inspect" not in content:
            continue

        for pattern in [
            r'Tool result for inspect\(crop=\[([^\]]+)\],\s*query="([^"]+)"\)',
            r"Tool result for inspect\(crop=\[([^\]]+)\],\s*query='([^']+)'\)",
        ]:
            match = re.search(pattern, content)
            if match:
                crop_str, query_str = match.group(1), match.group(2).strip()
                try:
                    crop_vals = [float(v.strip()) for v in crop_str.split(",")]
                    if len(crop_vals) == 4:
                        turn += 1
                        tool_calls.append({"turn": turn, "crop": crop_vals, "query": query_str})
                except ValueError:
                    pass
                break
    return tool_calls


# ===== File format detection =====

def detect_format(sample: Dict) -> str:
    """Detect whether the file is in 'output' or 'result' format."""
    if "metadata" in sample and "messages" in sample:
        return "output"
    return "result"


# ===== Main analysis =====

def analyze(input_path: Path) -> Dict:
    """Run the three-category analysis on a JSONL file."""

    n_normal = 0
    n_normal_queried = 0                   # normal samples that called inspect()
    n_normal_queried_then_normal = 0       # Cat 1

    n_multi_anomaly = 0                    # multi-turn anomaly samples
    n_multi_anomaly_with_bbox = 0          # multi-turn anomaly samples with GT bbox
    n_type_correction = 0                  # Cat 2
    n_iou_improvement = 0                  # Cat 3

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            fmt = detect_format(data)

            if fmt == "output":
                # --- Output format: has metadata + messages ---
                meta = data.get("metadata", {})
                gt_present = bool(meta.get("gt_answer", False))
                gt_type = str(meta.get("anomaly_type", "") or "").strip()
                correct_answer = str(meta.get("correct_answer", "") or "").strip()
                gt_bbox = normalize_bbox(meta.get("gt_bbox"))

                tool_calls = extract_tool_calls(data.get("messages", []))
                num_tc = len(tool_calls)

                final_answer = meta.get("final_answer") or {}
                final_present = bool(final_answer.get("anomaly_present", False))
                final_type = str(final_answer.get("anomaly_type", "") or "").strip()
                final_bbox = normalize_bbox(final_answer.get("bbox"))
            else:
                # --- Result format: flat, no messages ---
                gt_present = bool(data.get("gt_answer", False))
                gt_type = str(data.get("gt_type", "") or "").strip()
                correct_answer = str(data.get("correct_answer", "") or "").strip()
                gt_bbox = normalize_bbox(data.get("gt_bbox"))

                num_tc = data.get("num_turns", 0)
                # Result format has no message-level tool call info
                tool_calls = []

                final_present = bool(data.get("pred_present", False))
                final_type = str(data.get("pred_type", "") or "").strip()
                final_bbox = normalize_bbox(data.get("pred_bbox"))

            # ----- Category 1: Normal samples -----
            if not gt_present:
                n_normal += 1
                if num_tc > 0:
                    n_normal_queried += 1
                    if not final_present:
                        n_normal_queried_then_normal += 1
                continue

            # ----- Anomaly samples (multi-turn only) -----
            if num_tc < 2:
                continue

            n_multi_anomaly += 1
            if gt_bbox is not None:
                n_multi_anomaly_with_bbox += 1

            # Category 2: Type correction
            # At least one intermediate query was wrong, but the final type is correct
            final_type_ok = final_present and check_type_correct(final_type, gt_type, correct_answer)
            if final_type_ok and tool_calls:
                any_intermediate_wrong = any(
                    not check_type_correct(tc["query"], gt_type, correct_answer)
                    for tc in tool_calls
                )
                if any_intermediate_wrong:
                    n_type_correction += 1

            # Category 3: IoU analysis
            if gt_bbox is not None and tool_calls:
                final_iou = compute_iou(final_bbox, gt_bbox) if final_bbox else 0.0
                intermediate_ious = [
                    compute_iou(normalize_bbox(tc["crop"]), gt_bbox)
                    for tc in tool_calls
                ]

                # Cat 3: final IoU >  intermediate IoU
                worst_intermediate_iou = min(intermediate_ious) if intermediate_ious else 0.0
                if final_iou > worst_intermediate_iou:
                    n_iou_improvement += 1

    return {
        "n_normal": n_normal,
        "n_normal_queried": n_normal_queried,
        "n_normal_queried_then_normal": n_normal_queried_then_normal,
        "n_multi_anomaly": n_multi_anomaly,
        "n_multi_anomaly_with_bbox": n_multi_anomaly_with_bbox,
        "n_type_correction": n_type_correction,
        "n_iou_improvement": n_iou_improvement,
    }


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n/d*100:.1f}%)" if d > 0 else "N/A"


def print_report(m: Dict) -> None:
    """Print the analysis report."""
    lines = [
        "=" * 60,
        "Multi-Turn Effect Analysis",
        "=" * 60,
        "",
        f"Normal samples: {m['n_normal']} (queried: {m['n_normal_queried']})",
        f"  Cat 1 - Queried but output normal: "
        + _pct(m["n_normal_queried_then_normal"], m["n_normal_queried"]),
        "",
        f"Multi-turn anomaly samples: {m['n_multi_anomaly']}",
        f"  Cat 2 - Type correction (intermediate wrong, final correct): "
        + _pct(m["n_type_correction"], m["n_multi_anomaly"]),
        "",
        f"Multi-turn anomaly samples with GT bbox: {m['n_multi_anomaly_with_bbox']}",
        f"  Cat 3 - IoU improvement: "
        + _pct(m["n_iou_improvement"], m["n_multi_anomaly_with_bbox"]),
        "",
        "=" * 60,
    ]
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Analyze multi-turn tool call effects")
    ap.add_argument("--input", "-i", required=True,
                    help="Outputs JSONL file (outputs_*.jsonl with message-level data)")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing: {input_path}", file=sys.stderr)
    metrics = analyze(input_path)
    print_report(metrics)


if __name__ == "__main__":
    main()
