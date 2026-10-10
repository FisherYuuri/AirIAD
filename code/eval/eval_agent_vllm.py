#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM batch evaluation script for multi-turn tool-calling anomaly detection.

Supports zero-shot and one-shot (with --oneshot for reference image).

Usage:
    python eval_agent_vllm.py \
        --model /path/to/model \
        --input test_6400.json \
        --output results.jsonl \
        --knowledge knowledge_base/knowledge.json \
        --candidates knowledge_base/candidates.json

    # one-shot mode
    python eval_agent_vllm.py \
        --oneshot \
        --model /path/to/model \
        --input test_6400.json \
        --output results.jsonl \
        --knowledge knowledge_base/knowledge.json \
        --candidates knowledge_base/candidates.json
"""

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# LLaVA-OneVision compatibility patch
try:
    from transformers.cache_utils import SlidingWindowCache
except ImportError:
    from transformers.cache_utils import DynamicCache as SlidingWindowCache
    import transformers.cache_utils
    transformers.cache_utils.SlidingWindowCache = SlidingWindowCache

import PIL.Image

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = (
    "You are an industrial anomaly detection expert equipped with a visual inspection tool.\n\n"
    "## Role\n"
    "Your task is to localize and identify defects in industrial product images. "
    "You reason iteratively: each turn you propose a spatial crop and a semantic query, "
    "then verify whether the retrieved evidence is sufficient to conclude.\n\n"
    "## Tool\n"
    "You have access to one tool:\n\n"
    "  inspect(crop, query)\n\n"
    "  Parameters:\n"
    "    crop  — [x1, y1, x2, y2], normalized floats in [0,1000], top-left to bottom-right\n"
    "    query — the LETTER (A, B, C, D, etc.) of the anomaly type from the candidate list "
    "provided in the user message; choose the letter that best matches your current visual hypothesis\n\n"
    "  Returns:\n"
    "    - A zoomed image of the cropped region\n"
    "    - Retrieved knowledge: text descriptions of defects matching the query\n\n"
    "  Call this tool when you need more visual or semantic evidence. "
    "Do NOT call it if you have already reached a conclusion.\n\n"
    "## Output Format\n"
    "At every turn, output two brief analysis lines, then either a tool call OR a final answer "
    "— never both.\n\n"
    "Analysis format (required before every action):\n"
    "<spatial>[Evaluate the current visual evidence. Where is the suspicious region? "
    "Does the crop need adjustment?]</spatial>\n"
    "<semantic>[Evaluate the retrieved knowledge against the visual. "
    "Does the selected anomaly type match what you see? Should it be refined?]</semantic>\n\n"
    "Tool call format (when more evidence needed):\n"
    "<tool_call]\n"
    "  <name>inspect</name>\n"
    "  <crop>[x1, y1, x2, y2]</crop>\n"
    "  <query>A/B/C/D/...</query>\n"
    "</tool_call]\n\n"
    "Final answer format (when conclusion is reached):\n"
    "<answer>\n"
    "  <has_anomaly>yes/no</has_anomaly>\n"
    "  <anomaly_type>A/B/C/D/... or \"none\"</anomaly_type>\n"
    "  <bbox>[x1, y1, x2, y2] or null</bbox>\n"
    "</answer>\n\n"
    "IMPORTANT:\n"
    "- Use ONLY the LETTER (A, B, C, etc.) to refer to anomaly types, NOT the full name\n"
    "- If has_anomaly is \"no\", set anomaly_type to \"none\" and bbox to null"
)

# ============================================================================
# User message templates
# ============================================================================

USER_INITIAL_TEMPLATE = (
    "Evaluate the following image from the class \"{class_name}\".\n\n"
    "Candidate anomaly types:\n{anomaly_list}\n\n"
    "Determine whether this sample is normal or abnormal. "
    "If abnormal, identify the defect type and localize it.\n\n"
    "You may call `inspect(crop, query)` to zoom into a suspicious region "
    "and retrieve relevant defect knowledge."
)

USER_ONESHOT_TEMPLATE_PART1 = (
    "Evaluate the following image from the class \"{class_name}\"."
    "\n\nCandidate anomaly types:\n{anomaly_list}\n\n"
    "A normal reference image is provided for comparison. "
    "Compare the target image with the reference to identify any visual differences. "
    "Determine whether this sample is normal or abnormal. "
    "If abnormal, identify the defect type and localize it.\n\n"
    "You may call `inspect(crop, query)` to zoom into a suspicious region "
    "and retrieve relevant defect knowledge."
)

USER_ONESHOT_TEMPLATE_PART2 = "Here is the normal reference image:"

# Tool result templates
TOOL_RESULT_TEMPLATE_PART1 = (
    "Tool result for inspect(crop={crop}, query=\"{query}\"):\n\n"
)

TOOL_RESULT_TEMPLATE_PART2 = (
    "Retrieved knowledge for \"{class_name}\":\n{knowledge}\n\n"
    "---\n"
    "Now verify your hypothesis:\n"
    "- Does the cropped image tightly cover an anomaly?\n"
    "- Does the retrieved knowledge accurately describe what you see?\n\n"
    "If YES to both → output <answer>\n"
    "If NO to either → refine and call inspect again"
)

# ============================================================================
# Regex parsers
# ============================================================================

_TC_XML_RE = re.compile(
    r'<tool_call]\s*<name>\s*inspect\s*</name>\s*'
    r'<crop>\s*(\[[\d\s.,\-]+\])\s*</crop>\s*'
    r'<query>\s*(.*?)\s*</query>\s*</tool_call]',
    re.DOTALL | re.IGNORECASE,
)
_ANS_XML_RE = re.compile(
    r'<answer>\s*<has_anomaly>\s*(yes|no)\s*</has_anomaly>\s*'
    r'<anomaly_type>\s*(.*?)\s*</anomaly_type>\s*'
    r'<bbox>\s*((?:\[[\d\s.,\-]+\])|null)\s*</bbox>\s*</answer>',
    re.DOTALL | re.IGNORECASE,
)


def parse_tool_calls(text: str) -> List[Dict]:
    calls = []
    for m in _TC_XML_RE.finditer(text):
        try:
            raw_crop = json.loads(m.group(1))
            crop = [max(0.0, min(1.0, float(v) / 1000.0)) for v in raw_crop]
            calls.append({
                "name": "inspect",
                "arguments": {"crop": crop, "query": m.group(2).strip()},
            })
        except Exception:
            pass
    return calls


def _bbox_1000_to_01(bbox: List) -> Optional[List[float]]:
    if not bbox or len(bbox) != 4:
        return None
    try:
        vals = [float(v) for v in bbox]
        if all(v <= 1.0 for v in vals):
            normed = [max(0.0, min(1.0, v)) for v in vals]
        else:
            normed = [max(0.0, min(1.0, v / 1000.0)) for v in vals]
        if normed[0] >= normed[2] or normed[1] >= normed[3]:
            return None
        return [round(v, 4) for v in normed]
    except Exception:
        return None


def parse_final_answer(text: str) -> Optional[Dict]:
    m = _ANS_XML_RE.search(text)
    if not m:
        return None
    has_anomaly = m.group(1).lower() == "yes"
    anomaly_type = m.group(2).strip()
    bbox_str = m.group(3).strip()
    bbox = None
    if bbox_str.lower() != "null":
        try:
            raw = json.loads(bbox_str)
            bbox = _bbox_1000_to_01(raw)
        except Exception:
            pass
    return {"anomaly_present": has_anomaly, "anomaly_type": anomaly_type, "bbox": bbox}


# ============================================================================
# ABCD utilities
# ============================================================================

def parse_abcd_candidates(anomaly_list: List[str]) -> Dict[str, str]:
    mapping = {}
    for i, item in enumerate(anomaly_list):
        letter = chr(ord('A') + i)
        m = re.match(r'^([A-Z])[.:\)\-]?\s*(.+)$', item.strip(), re.IGNORECASE)
        if m:
            mapping[m.group(1).upper()] = m.group(2).strip()
        else:
            mapping[letter] = item.strip()
    return mapping


def abcd_to_type(abcd: str, mapping: Dict[str, str]) -> str:
    s = abcd.strip()
    if len(s) == 1 and s.upper() in mapping:
        return mapping[s.upper()]
    return s


def query_matches_type(query: str, gt_type: str,
                       mapping: Optional[Dict[str, str]] = None) -> bool:
    if not query or not gt_type:
        return False
    actual = abcd_to_type(query, mapping or {}) if mapping else query
    q, a = actual.lower().strip(), gt_type.lower().strip()
    return q in a or a in q


# ============================================================================
# Image helpers
# ============================================================================

def pil_to_b64(image: PIL.Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def normalize_bbox(bbox: List[float]) -> List[float]:
    values = [float(v) for v in bbox]
    if any(v > 1 for v in values):
        return [v / 1000.0 for v in values]
    return values


def crop_image(image: PIL.Image.Image, bbox: List[float],
               max_aspect_ratio: float = 150) -> Optional[PIL.Image.Image]:
    try:
        x1, y1, x2, y2 = normalize_bbox(bbox)
        x1, y1 = max(0., x1), max(0., y1)
        x2, y2 = min(1., x2), min(1., y2)
        if x1 >= x2 or y1 >= y2:
            return None

        w, h = image.size
        px1, py1 = math.floor(x1 * w), math.floor(y1 * h)
        px2, py2 = max(px1 + 1, math.ceil(x2 * w)), max(py1 + 1, math.ceil(y2 * h))

        crop_w, crop_h = px2 - px1, py2 - py1
        aspect = max(crop_w, crop_h) / max(1, min(crop_w, crop_h))

        if aspect > max_aspect_ratio:
            if crop_w > crop_h:
                target_h = int(crop_w / max_aspect_ratio)
                expand = target_h - crop_h
                py1 = max(0, py1 - expand // 2)
                py2 = min(h, py2 + (expand - (py1 - max(0, py1 - expand // 2))))
            else:
                target_w = int(crop_h / max_aspect_ratio)
                expand = target_w - crop_w
                px1 = max(0, px1 - expand // 2)
                px2 = min(w, px2 + (expand - (px1 - max(0, px1 - expand // 2))))

        cropped = image.crop((px1, py1, px2, py2))
        if max(cropped.size) > 2048:
            scale = 2048 / max(cropped.size)
            cropped = cropped.resize(
                (max(1, int(cropped.width * scale)),
                 max(1, int(cropped.height * scale))),
                PIL.Image.LANCZOS,
            )
        return cropped
    except Exception:
        return None


def compute_iou(b1: List[float], b2: List[float]) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0., ix2 - ix1) * max(0., iy2 - iy1)
    a1 = max(0., b1[2] - b1[0]) * max(0., b1[3] - b1[1])
    a2 = max(0., b2[2] - b2[0]) * max(0., b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.


# ============================================================================
# Knowledge retrieval
# ============================================================================

class KnowledgeIndex:
    """Knowledge retrieval index: builds variant-to-canon mapping from candidates.json.

    Retrieval flow:
    1. Model query (e.g. "Label or cap issues") -> exact match variant in candidates[ds][cls]
    2. variant -> canon (standardized defect type name)
    3. canon -> knowledge[ds][cls][canon] to get knowledge text
    4. Returns empty string if no match found
    """
    def __init__(self, candidates_path: str, knowledge_path: str):
        with open(candidates_path, encoding='utf-8') as f:
            cands = json.load(f)
        with open(knowledge_path, encoding='utf-8') as f:
            self.know = json.load(f)
        self.variant_map: Dict[str, tuple] = {}
        for ds, classes in cands.items():
            for cls, mapping in classes.items():
                if isinstance(mapping, dict):
                    for canon, variants in mapping.items():
                        for v in [canon] + (variants or []):
                            k = v.lower().strip().rstrip(".")
                            map_key = (k, ds, cls)
                            if map_key not in self.variant_map:
                                self.variant_map[map_key] = (ds, cls, canon)
                elif isinstance(mapping, list):
                    for v in mapping:
                        k = v.lower().strip().rstrip(".")
                        map_key = (k, ds, cls)
                        if map_key not in self.variant_map:
                            self.variant_map[map_key] = (ds, cls, v)

    def get_knowledge(self, ds: str, cls: str, canon: str) -> str:
        return self.know.get(ds, {}).get(cls, {}).get(canon, "")

    def search(self, query: str, ds: str, cls: str) -> str:
        q = query.lower().strip().rstrip(".")
        map_key = (q, ds, cls)
        if map_key in self.variant_map:
            _ds, _cls, canon = self.variant_map[map_key]
            return self.get_knowledge(_ds, _cls, canon)
        return ""


def normalize_dataset(path: str) -> str:
    for ds in ['DS-MVTec', 'MVTec-LOCO', 'VisA', 'GoodsAD']:
        if ds in path:
            return {'DS-MVTec': 'MVTec'}.get(ds, ds)
    return 'MVTec'


_KNOWLEDGE_INDEX: Optional[KnowledgeIndex] = None


def get_index(candidates_path: str, knowledge_path: str) -> KnowledgeIndex:
    global _KNOWLEDGE_INDEX
    if _KNOWLEDGE_INDEX is None:
        _KNOWLEDGE_INDEX = KnowledgeIndex(candidates_path, knowledge_path)
    return _KNOWLEDGE_INDEX


def retrieve_knowledge(class_name: str, query_letter: str,
                       candidates_path: str, knowledge_path: str,
                       image_path: str,
                       candidate_mapping: Dict[str, str],
                       skip_normal: bool = False) -> str:
    actual = abcd_to_type(query_letter, candidate_mapping)
    ds = normalize_dataset(image_path)
    idx = get_index(candidates_path, knowledge_path)
    
    defect_know = idx.search(actual, ds, class_name)
    parts = []
    
    if skip_normal:
        if defect_know:
            parts.append(f"[{class_name} — {actual}]\n{defect_know}")
        else:
            parts.append(
                f"[{class_name} — {actual}] "
                f"Knowledge for this query is not yet cataloged."
            )
            good = idx.get_knowledge(ds, class_name, "good")
            if good:
                parts.append(f"[{class_name} — Normal]\n{good}")
    else:
        good = idx.get_knowledge(ds, class_name, "good")
        if good:
            parts.append(f"[{class_name} — Normal]\n{good}")
        if defect_know:
            parts.append(f"[{class_name} — {actual}]\n{defect_know}")
        else:
            parts.append(f"[{class_name} — {actual}] No description available.")
    
    return "\n\n".join(parts)


# ============================================================================
# Sample state
# ============================================================================

@dataclass
class SampleState:
    item: Dict
    messages: List[Dict]
    candidate_mapping: Dict[str, str]
    num_turns: int = 0
    final_answer: Optional[Dict] = None
    done: bool = False
    gt_answer: Optional[bool] = None
    gt_type: str = ""
    gt_bbox: Optional[List[float]] = None
    target_image: Optional[PIL.Image.Image] = None
    image_paths: List[str] = field(default_factory=list)
    sft_tool_results: List[Dict] = field(default_factory=list)


def build_initial_state(item: Dict, max_turns: int,
                        candidates_path: str, knowledge_path: str,
                        oneshot: bool = False) -> Optional[SampleState]:
    class_name = item.get("class_name", "unknown")
    anomaly_list = item.get("anomaly_list", [])
    candidate_mapping = parse_abcd_candidates(anomaly_list)
    anomaly_str = "\n".join(f"- {t}" for t in anomaly_list) or "- none"

    image_path = item.get("image") or item.get("image_path", "")
    try:
        target_image = PIL.Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[warn] Cannot load image {image_path}: {e}", file=sys.stderr)
        return None

    ref_image: Optional[PIL.Image.Image] = None
    if oneshot:
        templates = item.get("similar_templates", [])
        if templates:
            try:
                ref_image = PIL.Image.open(templates[0]).convert("RGB")
            except Exception as e:
                print(f"[warn] Cannot load ref image {templates[0]}: {e}",
                      file=sys.stderr)

    has_ref = ref_image is not None
    user_content: List[Dict] = []

    if item.get("question") and not oneshot:
        user_content.append({"type": "text", "text": item["question"]})
        user_content.append({"type": "image", "image": target_image})
    elif oneshot and has_ref:
        text_part1 = USER_ONESHOT_TEMPLATE_PART1.format(
            class_name=class_name, anomaly_list=anomaly_str)
        user_content.append({"type": "text", "text": text_part1})
        user_content.append({"type": "image", "image": target_image})
        user_content.append({"type": "text", "text": USER_ONESHOT_TEMPLATE_PART2})
        user_content.append({"type": "image", "image": ref_image})
    else:
        user_text = USER_INITIAL_TEMPLATE.format(
            class_name=class_name, anomaly_list=anomaly_str)
        user_content.append({"type": "text", "text": user_text})
        user_content.append({"type": "image", "image": target_image})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    image_path_list = [image_path]
    if ref_image is not None:
        ref_path = item.get("similar_templates", [""])[0]
        image_path_list.append(ref_path)

    return SampleState(
        item=item,
        messages=messages,
        candidate_mapping=candidate_mapping,
        gt_answer=item.get("gt_answer"),
        gt_type=item.get("anomaly_type", ""),
        gt_bbox=item.get("bbox"),
        target_image=target_image,
        image_paths=image_path_list,
    )


def _save_crop_image(image: PIL.Image.Image, crop_dir: str,
                     image_path: str, turn_idx: int,
                     crop_bbox: List[float]) -> str:
    markers = ["MMAD_resize_1024/", "MMAD/", "AgentIAD/"]
    rel = None
    for marker in markers:
        idx = image_path.find(marker)
        if idx >= 0:
            rel = image_path[idx + len(marker):]
            break
    if rel is None:
        rel = Path(image_path).name

    stem = Path(rel).stem
    parent = str(Path(rel).parent)
    bbox_str = f"[{crop_bbox[0]:.0f},{crop_bbox[1]:.0f},{crop_bbox[2]:.0f},{crop_bbox[3]:.0f}]"
    filename = f"turn{turn_idx}_{bbox_str}.png"

    save_path = Path(crop_dir) / parent / stem / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(save_path), format="PNG")
    return str(save_path)


SFT_TOOL_RESULT_TEMPLATE = (
    "Tool result for inspect(crop={crop}, query=\"{query}\"):\n\n"
    "<image>\n\n"
    "Retrieved knowledge for \"{class_name}\":\n{knowledge}\n\n"
    "---\n"
    "Now verify your hypothesis:\n"
    "- Does the cropped image tightly cover an anomaly?\n"
    "- Does the retrieved knowledge accurately describe what you see?\n\n"
    "If YES to both → output <answer>\n"
    "If NO to either → refine and call inspect again"
)


def execute_tool(state: SampleState, call: Dict,
                 candidates_path: str, knowledge_path: str,
                 max_turns: int,
                 sft_crop_dir: Optional[str] = None,
                 skip_normal: bool = False) -> None:
    args = call.get("arguments", {})
    crop = args.get("crop", [])
    query = args.get("query", "").strip()

    result_images: List[PIL.Image.Image] = []
    result_texts: List[str] = []

    if len(crop) == 4:
        cropped = crop_image(state.target_image, crop)
        if cropped:
            result_images.append(cropped)
            ktext = retrieve_knowledge(
                state.item.get("class_name", ""),
                query, candidates_path, knowledge_path,
                state.item.get("image") or state.item.get("image_path", ""),
                state.candidate_mapping,
                skip_normal=skip_normal,
            )
            result_texts.append(
                f"[inspect] Cropped region {crop} "
                f"({cropped.width}×{cropped.height}px).\n\n{ktext}"
            )
            if sft_crop_dir is not None:
                crop_path = _save_crop_image(
                    cropped, sft_crop_dir,
                    state.item.get("image") or state.item.get("image_path", ""),
                    len(state.sft_tool_results), crop,
                )
                state.image_paths.append(crop_path)
                state.sft_tool_results.append({
                    "crop": crop, "query": query,
                    "knowledge": ktext, "crop_path": crop_path,
                })
        else:
            result_texts.append(f"[inspect] Error: invalid crop {crop}.")
    else:
        result_texts.append("[inspect] Error: crop must have 4 values.")

    content: List[Dict] = []
    if result_images and len(crop) == 4:
        text_part1 = TOOL_RESULT_TEMPLATE_PART1.format(
            crop=crop, query=query,
            width=result_images[0].width, height=result_images[0].height,
        )
        text_part2 = TOOL_RESULT_TEMPLATE_PART2.format(
            class_name=state.item.get("class_name", ""), knowledge=ktext,
        )
        content.append({"type": "text", "text": text_part1})
        content.append({"type": "image", "image": result_images[0]})
        content.append({"type": "text", "text": text_part2})
    else:
        content.append({"type": "text", "text": "\n\n".join(result_texts)})
    state.messages.append({"role": "user", "content": content})


# ============================================================================
# Output writer
# ============================================================================

class ResultWriter:
    def __init__(self, path: Path):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")
        self.written = 0

    def write(self, state: SampleState):
        fa = state.final_answer or {}
        pred_type = fa.get("anomaly_type", "")
        correct_letter = state.item.get("correct_answer", "")
        type_correct = None
        if state.gt_answer:
            if correct_letter:
                type_correct = (pred_type.strip().upper()
                                == correct_letter.strip().upper())
            elif state.gt_type:
                type_correct = query_matches_type(
                    pred_type, state.gt_type, state.candidate_mapping)
        record = {
            "qid":            state.item.get("qid", state.item.get("id")),
            "image":          state.item.get("image") or state.item.get("image_path"),
            "class_name":     state.item.get("class_name"),
            "gt_answer":      state.gt_answer,
            "gt_type":        state.gt_type,
            "correct_answer": correct_letter,
            "gt_bbox":        state.gt_bbox,
            "pred_present":   fa.get("anomaly_present"),
            "pred_type":      pred_type,
            "pred_bbox":      fa.get("bbox"),
            "type_correct":   type_correct,
            "num_turns":      state.num_turns,
            "iou":            (
                compute_iou(fa["bbox"], state.gt_bbox)
                if fa.get("bbox") and state.gt_bbox
                   and len(fa["bbox"]) == 4 and len(state.gt_bbox) == 4
                else None
            ),
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.written += 1

    def close(self):
        self._fh.close()


# ============================================================================
# SFT Writer
# ============================================================================

class SFTWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = open(self.path, "a", encoding="utf-8")
        self.written = 0

    def write(self, state: SampleState):
        sft_messages = self._build_sft_messages(state)
        record = {
            "messages": sft_messages,
            "images": list(state.image_paths),
            "metadata": {
                "class_name": state.item.get("class_name"),
                "gt_answer": state.gt_answer,
                "anomaly_type": state.gt_type or None,
                "gt_bbox": state.gt_bbox,
                "qid": state.item.get("qid", state.item.get("id")),
                "num_tool_calls": len(state.sft_tool_results),
                "final_answer": state.final_answer,
                "correct_answer": state.item.get("correct_answer"),
                "anomaly_list": state.item.get("anomaly_list", []),
                "similar_templates": state.item.get("similar_templates", []),
            },
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.written += 1

    def _build_sft_messages(self, state: SampleState) -> List[Dict]:
        sft_msgs = []
        first_user_done = False
        tool_result_idx = 0

        for msg in state.messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                sft_msgs.append({"role": role, "content": content})
                continue

            n_images = sum(1 for item in content if item["type"] == "image")
            text_parts = [item["text"] for item in content if item["type"] == "text"]

            if not first_user_done:
                first_user_done = True
                if n_images == 2:
                    part1 = text_parts[0] if len(text_parts) > 0 else ""
                    part2 = text_parts[1] if len(text_parts) > 1 else "Here is the normal reference image:"
                    sft_msgs.append({
                        "role": role,
                        "content": f"{part1} <image>\n\n{part2}<image>",
                    })
                else:
                    img_tags = " ".join(["<image>"] * n_images)
                    sft_msgs.append({
                        "role": role,
                        "content": f"{'\\n\\n'.join(text_parts)} {img_tags}",
                    })
            else:
                if tool_result_idx < len(state.sft_tool_results):
                    tr = state.sft_tool_results[tool_result_idx]
                    sft_content = SFT_TOOL_RESULT_TEMPLATE.format(
                        crop=tr["crop"],
                        query=tr["query"],
                        class_name=state.item.get("class_name", ""),
                        knowledge=tr["knowledge"],
                    )
                    sft_msgs.append({"role": role, "content": sft_content})
                    tool_result_idx += 1
                else:
                    parts = []
                    for item in content:
                        if item["type"] == "image":
                            parts.append("<image>")
                        elif item["type"] == "text":
                            parts.append(item["text"])
                    sft_msgs.append({
                        "role": role,
                        "content": "\n\n".join(parts),
                    })

        return sft_msgs

    def close(self):
        self._fh.close()


# ============================================================================
# vLLM helpers
# ============================================================================

def messages_to_vllm(messages: List[Dict], processor, model_name: str = "") -> Tuple[str, List]:
    model_name_lower = model_name.lower()
    
    if "internvl" in model_name_lower:
        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(messages)
        return text, imgs or []
    elif "llava" in model_name_lower or "onevision" in model_name_lower:
        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(messages)
        return text, imgs or []
    elif "minicpm" in model_name_lower:
        from qwen_vl_utils import process_vision_info
        imgs, _ = process_vision_info(messages)
        minicpm_messages = []
        for msg in messages:
            content_parts = []
            content = msg["content"]
            if isinstance(content, str):
                content_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            content_parts.append(item.get("text", ""))
                        elif item.get("type") == "image":
                            content_parts.append("(<image>./</image>)")
                    elif isinstance(item, str):
                        content_parts.append(item)
            minicpm_messages.append({"role": msg["role"], "content": "\n".join(content_parts)})
        
        if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'apply_chat_template'):
            text = processor.tokenizer.apply_chat_template(
                minicpm_messages, tokenize=False, add_generation_prompt=True)
        else:
            text_parts = []
            for msg in minicpm_messages:
                if msg["role"] == "user":
                    text_parts.append(f"<|im_start|>user\n{msg['content']}<|im_end|>\n")
                elif msg["role"] == "assistant":
                    text_parts.append(f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n")
            text = "".join(text_parts) + "<|im_start|>assistant\n"
        return text, imgs or []
    else:
        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        imgs, _ = process_vision_info(messages)
        return text, imgs or []


# ============================================================================
# vLLM runner — multi-turn tool mode
# ============================================================================

def _run_vllm_tool_chunk(
    chunk: List[SampleState],
    llm, processor, sp,
    args, writer: ResultWriter,
    total_items: int, t0: float, done_count: List[int],
    sft_writer: Optional['SFTWriter'] = None,
):
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing

    active = list(chunk)
    sft_crop_dir = getattr(args, 'sft_crop_dir', None)
    skip_normal = getattr(args, 'no_normal_knowledge', False)
    n_workers = min(multiprocessing.cpu_count(), len(chunk), 32)

    def _build_prompt(state):
        text, imgs = messages_to_vllm(state.messages, processor, args.model)
        kw = {"prompt": text}
        if imgs:
            kw["multi_modal_data"] = {"image": imgs}
        return kw

    def _exec_tool(state_call):
        state, call = state_call
        execute_tool(state, call, args.candidates,
                     args.knowledge, args.max_turns,
                     sft_crop_dir=sft_crop_dir, skip_normal=skip_normal)
        return state

    def _finalize(state):
        writer.write(state)
        if sft_writer:
            sft_writer.write(state)
        done_count[0] += 1

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        while active:
            prompts = list(pool.map(_build_prompt, active))
            outputs = llm.generate(prompts, sp)

            need_tool: List[Tuple[SampleState, Dict]] = []
            next_active: List[SampleState] = []

            for state, output in zip(active, outputs):
                resp = output.outputs[0].text if output.outputs else ""
                state.messages.append({"role": "assistant", "content": resp})
                state.num_turns += 1

                fa = parse_final_answer(resp)
                if fa is not None:
                    state.final_answer = fa
                    state.done = True
                    _finalize(state)
                    continue

                if state.num_turns >= args.max_turns:
                    state.done = True
                    _finalize(state)
                    continue

                calls = parse_tool_calls(resp)
                if not calls:
                    state.done = True
                    _finalize(state)
                else:
                    need_tool.append((state, calls[0]))

            if need_tool:
                tool_results = list(pool.map(_exec_tool, need_tool))
                next_active.extend(tool_results)

            active = next_active
            done = done_count[0]
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"\r[vLLM] {done}/{total_items} done  "
                  f"{rate:.1f} samples/s", end="", file=sys.stderr)


def run_vllm(
    items: List[Dict], args, writer: ResultWriter,
    sft_writer: Optional['SFTWriter'] = None,
):
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    print(f"[vLLM] Loading processor from {args.model} ...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    max_imgs = 1 + (1 if args.oneshot else 0) + args.max_turns
    print(f"[vLLM] Loading model (limit_mm_per_prompt image={max_imgs}) ...",
          file=sys.stderr)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="half",
        gpu_memory_utilization=float(os.getenv("VLLM_GPU_UTIL", "0.85")),
        swap_space=int(os.getenv("VLLM_SWAP_GB", "8")),
        enforce_eager=False,
        limit_mm_per_prompt={"image": max_imgs, "video": 0},
    )

    if "internvl" in args.model.lower():
        stop_tokens = ["</s>", "<|end|>"]
    elif "llava" in args.model.lower() or "onevision" in args.model.lower():
        stop_tokens = ["<|eot_id|>", "<|end|>"]
    elif "minicpm" in args.model.lower():
        stop_tokens = ["<|im_end|>", "</s>"]
    else:
        stop_tokens = ["<|im_end|>", "<|im_start|>"]
    
    sp = SamplingParams(
        temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        stop=stop_tokens, seed=args.seed,
    )

    skip_ids = _load_done_ids(writer.path)
    all_states: List[SampleState] = []
    for item in items:
        qid = str(item.get("qid", item.get("id", "")))
        if qid in skip_ids:
            continue
        state = build_initial_state(item, args.max_turns,
                                    args.candidates, args.knowledge,
                                    oneshot=args.oneshot)
        if state:
            all_states.append(state)

    total = len(all_states)
    batch_size = args.batch_size
    print(f"[vLLM] {total} samples to process, chunk_size={batch_size}",
          file=sys.stderr)
    t0 = time.time()
    done_count = [0]

    for chunk_start in range(0, total, batch_size):
        chunk = all_states[chunk_start:chunk_start + batch_size]
        print(f"\n[vLLM] Chunk {chunk_start // batch_size + 1}"
              f" ({len(chunk)} samples)", file=sys.stderr)
        _run_vllm_tool_chunk(chunk, llm, processor, sp,
                             args, writer, total, t0, done_count,
                             sft_writer=sft_writer)

    print(file=sys.stderr)


# ============================================================================
# Resume helpers
# ============================================================================

def _load_done_ids(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                qid = r.get("qid")
                if qid is not None:
                    done.add(str(qid))
            except Exception:
                pass
    return done


# ============================================================================
# Input reading
# ============================================================================

def _normalize_bbox(bbox) -> Optional[List[float]]:
    if bbox is None:
        return None
    if isinstance(bbox, list) and bbox:
        if isinstance(bbox[0], list):
            return _bbox_1000_to_01(bbox[0])
        if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
            return _bbox_1000_to_01(bbox)
    return None


def read_items(path: Path) -> List[Dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        data = json.loads(raw)
        if isinstance(data, list) and data and "messages" in data[0]:
            out = []
            for rec in data:
                meta = rec.get("metadata", {})
                item = {
                    "qid":               meta.get("qid", rec.get("qid")),
                    "image":             rec.get("images", [None])[0],
                    "class_name":        meta.get("class_name", ""),
                    "anomaly_list":      meta.get("anomaly_list", []),
                    "gt_answer":         meta.get("gt_answer"),
                    "anomaly_type":      meta.get("anomaly_type", ""),
                    "bbox":              _normalize_bbox(meta.get("bbox")),
                    "similar_templates": meta.get("similar_templates", []),
                    "correct_answer":    meta.get("correct_answer"),
                    "question":          meta.get("question"),
                }
                out.append(item)
            return out
        for rec in data:
            rec["bbox"] = _normalize_bbox(rec.get("bbox"))
        return data
    else:
        items = [json.loads(l) for l in raw.splitlines() if l.strip()]
        for rec in items:
            rec["bbox"] = _normalize_bbox(rec.get("bbox"))
        return items


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="vLLM batch evaluation for industrial anomaly detection")
    ap.add_argument("--input",      required=True, help="Input JSON/JSONL file")
    ap.add_argument("--output",     required=True, help="Output JSONL file")
    ap.add_argument("--knowledge",  default="knowledge_base/knowledge.json")
    ap.add_argument("--candidates", default="knowledge_base/candidates.json")

    ap.add_argument("--save-sft", default=None, metavar="PATH",
                    help="Save conversations in SFT format to this JSONL file.")
    ap.add_argument("--sft-crop-dir", default=None, metavar="DIR",
                    help="Directory to save cropped images for SFT.")

    ap.add_argument("--max-turns",      type=int,   default=5)
    ap.add_argument("--max-samples",    type=int,   default=None)
    ap.add_argument("--batch-size",     type=int,   default=512)

    # vLLM arguments
    ap.add_argument("--model",                  required=True)
    ap.add_argument("--tensor-parallel-size",   type=int,   default=8)
    ap.add_argument("--pipeline-parallel-size", type=int,   default=1)
    ap.add_argument("--max-model-len",          type=int,   default=20480)
    ap.add_argument("--max-new-tokens",         type=int,   default=2048)
    ap.add_argument("--temperature",            type=float, default=0.0)
    ap.add_argument("--top-p",                  type=float, default=1.0)
    ap.add_argument("--top-k",                  type=int,   default=-1)
    ap.add_argument("--repetition-penalty",     type=float, default=1.0)
    ap.add_argument("--seed",                   type=int,   default=42)
    ap.add_argument("--no-normal-knowledge", action="store_true", default=False,
                    help="Do not return normal sample descriptions in tool results.")
    ap.add_argument("--oneshot", action="store_true",
                    help="Prepend a normal reference image from similar_templates")

    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.save_sft and not args.sft_crop_dir:
        args.sft_crop_dir = str(Path(args.save_sft).with_suffix('') / "crops").rstrip("/")
    if args.sft_crop_dir:
        Path(args.sft_crop_dir).mkdir(parents=True, exist_ok=True)

    items = read_items(input_path)
    if args.max_samples:
        items = items[:args.max_samples]
    print(f"Loaded {len(items)} items.", file=sys.stderr)

    writer = ResultWriter(output_path)

    sft_writer = None
    if args.save_sft:
        sft_writer = SFTWriter(Path(args.save_sft))
        print(f"[SFT] Saving to {args.save_sft}, "
              f"crops → {args.sft_crop_dir}", file=sys.stderr)

    skip_ids = _load_done_ids(output_path)
    if skip_ids:
        print(f"[resume] {len(skip_ids)} already done, skipping.", file=sys.stderr)

    try:
        run_vllm(items, args, writer, sft_writer=sft_writer)
    finally:
        writer.close()
        if sft_writer:
            sft_writer.close()
            print(f"[SFT] {sft_writer.written} conversations saved.", file=sys.stderr)

    print(f"\nDone. Results saved → {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
