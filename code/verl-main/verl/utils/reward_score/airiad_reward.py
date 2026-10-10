"""
IAD Agent Reward Module (SSGiGPO)

R_total = R_outcome (process reward only for SSGiGPO grouping, not in total)
"""

import json
import re
from typing import Any, Dict, List, Optional


CONFIG = {
    "enable_joint_alignment": True,
    "enable_dynamic_iou":     True,
    "iou_threshold_start":    0.1,
    "iou_threshold_end":      0.5,
}


_ANS_XML_RE = re.compile(
    r'<answer>\s*'
    r'<has_anomaly>\s*(yes|no)\s*</has_anomaly>\s*'
    r'<anomaly_type>\s*(.*?)\s*</anomaly_type>\s*'
    r'<bbox>\s*((?:\[[\d\s.,\-]+\])|null)\s*</bbox>\s*'
    r'</answer>',
    re.DOTALL | re.IGNORECASE,
)
_ANS_JSON_RE = re.compile(r'<answer>\s*(\{.*?\})\s*</answer>', re.DOTALL)


def extract_answer(solution_str: str) -> Optional[Dict[str, Any]]:
    """Extract final answer from XML or JSON format."""
    m = _ANS_XML_RE.search(solution_str)
    if m:
        return {
            "anomaly_present": m.group(1).strip().lower() == "yes",
            "anomaly_type": m.group(2).strip(),
            "bbox": json.loads(m.group(3)) if m.group(3).strip().lower() != "null" else None,
        }
    m = _ANS_JSON_RE.search(solution_str)
    if m:
        try:
            obj = json.loads(m.group(1))
            ap = obj.get("anomaly_present", obj.get("has_anomaly", False))
            if isinstance(ap, str):
                ap = ap.lower() in ("true", "yes", "1")
            return {
                "anomaly_present": ap,
                "anomaly_type": obj.get("anomaly_type") or obj.get("top_anomaly") or "none",
                "bbox": obj.get("anomaly_region") or obj.get("bbox"),
            }
        except json.JSONDecodeError:
            pass
    return None


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute IoU (coordinates in [0, 1000])."""
    x1, y1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
    x2, y2 = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1]) + (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _dynamic_threshold(progress: float) -> float:
    """Linearly interpolate IoU threshold based on training progress."""
    s, e = CONFIG["iou_threshold_start"], CONFIG["iou_threshold_end"]
    return s + (e - s) * min(1.0, max(0.0, progress))


def _iou_reward(iou: float, progress: float) -> float:
    """1.0 if IoU meets threshold, else raw IoU."""
    th = _dynamic_threshold(progress) if CONFIG["enable_dynamic_iou"] else 0.5
    return 1.0 if iou >= th else iou


# --- Outcome Reward ---

def compute_outcome_reward(
    solution_str: str,
    ground_truth: Dict[str, Any],
    training_progress: float = 1.0,
) -> Dict[str, Any]:
    answer = extract_answer(solution_str)
    info = {
        "outcome_total": 0.0,
        "acc_answer": 0.0, "acc_type": 0.0,
        "iou": None, "iou_reward": None,
        "joint_bonus": 0.0,
        "is_anomaly_sample": False,
    }
    if answer is None:
        return info

    reward = 0.0
    gt_answer = ground_truth.get("gt_answer")
    correct_answer = (ground_truth.get("correct_answer") or "").strip().upper()
    gt_bboxes = ground_truth.get("bbox", []) or []
    if gt_bboxes and isinstance(gt_bboxes[0], (int, float)):
        gt_bboxes = [gt_bboxes]

    pred_present = answer.get("anomaly_present")
    if isinstance(pred_present, str):
        pred_present = pred_present.lower() in ("true", "yes", "1")
    if isinstance(gt_answer, str):
        gt_answer = gt_answer.lower() in ("true", "yes", "1")

    is_anomaly = gt_answer is True
    info["is_anomaly_sample"] = is_anomaly

    det_correct = (pred_present == gt_answer)
    if det_correct:
        reward += 1.0  # has_anomaly correct
        info["acc_answer"] = 1.0

        pred_type = (answer.get("anomaly_type") or "none").strip()
        type_correct = (
            pred_type.lower() in ("none", "") if gt_answer is False
            else bool(correct_answer) and pred_type.upper() == correct_answer
        )
        if type_correct:
            reward += 1.0  # anomaly_type correct
            info["acc_type"] = 1.0

        type_match = 0.0
        if is_anomaly and gt_bboxes:
            pred_bbox = answer.get("bbox")
            if pred_bbox and len(pred_bbox) == 4:
                pred_bbox = [float(v) for v in pred_bbox]
                iou = max(compute_iou(pred_bbox, gt) for gt in gt_bboxes)
                iou_r = _iou_reward(iou, training_progress)
                reward += iou_r
                info["iou"] = iou
                info["iou_reward"] = iou_r

                pred_type_str = (answer.get("anomaly_type") or "").strip().upper()
                type_match = 1.0 if (correct_answer and pred_type_str == correct_answer) else 0.0
            else:
                info["iou"] = 0.0
                info["iou_reward"] = 0.0

        if CONFIG["enable_joint_alignment"] and is_anomaly:
            joint = (info["iou"] or 0.0) * type_match
            reward += joint
            info["joint_bonus"] = joint

    info["outcome_total"] = reward
    return info


# --- Process Reward ---

def compute_process_reward(
    tool_call_history: List[Dict],
    ground_truth: Dict[str, Any],
    training_progress: float = 1.0,
    det_correct: bool = True,
) -> Dict[str, Any]:
    if not tool_call_history:
        return {"process_total": 0.0, "step_rewards": [], "step_details": []}

    cfg = CONFIG
    gt_answer = ground_truth.get("gt_answer")
    correct_letter = (ground_truth.get("correct_answer") or "").strip().upper()
    is_anomaly = (gt_answer is True if isinstance(gt_answer, bool) else
                  (str(gt_answer).lower() in ("true", "yes", "1") if gt_answer else False))

    if not is_anomaly or not det_correct:
        return {"process_total": 0.0, "step_rewards": [], "step_details": []}

    threshold = _dynamic_threshold(training_progress) if cfg["enable_dynamic_iou"] else 0.5

    step_rewards, step_details = [], []

    for t, entry in enumerate(tool_call_history):
        result = entry.get("result", {})
        if not result.get("success", False):
            continue

        r = 0.0
        detail = {"t": t, "iou": 0.0, "qc": 0.0, "joint": 0.0}

        step_iou = float(result.get("step_iou", 0.0))
        step_qc = 1.0 if (correct_letter and
                          (result.get("query") or "").strip().upper() == correct_letter) else 0.0

        if step_qc == 1:
            r += 1.0
            detail["qc"] = 1.0

        if step_iou >= threshold:
            r += 1.0
            detail["iou"] = step_iou

        if cfg["enable_joint_alignment"] and step_qc == 1 and step_iou > 0:
            joint = step_iou  # type_match = 1 since qc==1
            r += joint
            detail["joint"] = joint

        detail["r_total"] = r
        step_rewards.append(r)
        step_details.append(detail)

    return {
        "process_total": sum(step_rewards),
        "step_rewards": step_rewards,
        "step_details": step_details,
    }


# --- SSGiGPO: Build anchor_obs_list ---

def _build_anchor_obs_list(tool_call_history: List[Dict]) -> List[Dict[str, Any]]:
    """Build anchor_obs_list for SSGiGPO step-level grouping.

    anchor_obs at step t = observation state *before* the decision at step t.
    step 0 → {is_initial: True}; step t≥1 → {prev_observed_type, prev_observed_bbox}.
    """
    if not tool_call_history:
        return [{"is_initial": True}]

    valid_steps = []
    for entry in tool_call_history:
        result = entry.get("result", {})
        if not result.get("success", False):
            continue
        crop = result.get("crop", [])
        query = result.get("query", "")
        step_iou = float(result.get("step_iou", 0.0))
        bbox = crop if isinstance(crop, list) and len(crop) == 4 else [0, 0, 1, 1]
        valid_steps.append({
            "query": query.strip() if query else "",
            "bbox": [float(v) for v in bbox],
            "iou": step_iou,
        })

    if not valid_steps:
        return [{"is_initial": True}]

    anchor_obs_list = []
    for t in range(len(valid_steps)):
        if t == 0:
            anchor_obs_list.append({"is_initial": True})
        else:
            prev = valid_steps[t - 1]
            anchor_obs_list.append({
                "is_initial": False,
                "prev_observed_type": prev["query"],
                "prev_observed_bbox": prev["bbox"],
            })

    return anchor_obs_list


# --- Entry Point ---

def compute_score(
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    extra_info = extra_info or {}
    training_progress = kwargs.get("training_progress",
                                    extra_info.get("training_progress", 1.0))

    outcome = compute_outcome_reward(solution_str, ground_truth, training_progress)

    tool_call_history = extra_info.get("tool_call_history", [])
    det_correct = (outcome["acc_answer"] == 1.0)
    process = compute_process_reward(tool_call_history, ground_truth, training_progress, det_correct)

    anchor_obs_list = _build_anchor_obs_list(tool_call_history)

    return {
        "score":             outcome["outcome_total"],
        "acc":               outcome["acc_answer"],
        "acc_anomaly_type":  outcome["acc_type"],
        "final_iou":         outcome["iou"],
        "iou_reward":        outcome["iou_reward"],
        "is_anomaly_sample": outcome["is_anomaly_sample"],
        "process_reward":    process["process_total"],
        "outcome_reward":    outcome["outcome_total"],
        "joint_bonus":       outcome["joint_bonus"],
        "training_progress": training_progress,
        "step_rewards":      process["step_rewards"],
        "anchor_obs_list":   anchor_obs_list,
    }
