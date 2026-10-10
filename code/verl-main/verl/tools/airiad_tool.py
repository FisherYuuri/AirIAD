# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IAD inspect(crop, query) tool with knowledge retrieval and step-level IoU."""

import json
import os
import re
from math import ceil, floor
from typing import Any, Optional
from uuid import uuid4

from qwen_vl_utils import fetch_image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse


# --- ABCD Mapping Utilities ---

def parse_abcd_candidates(anomaly_list: list) -> dict:
    """Parse ABCD format candidates into {letter: description} mapping."""
    mapping = {}
    letters = [chr(ord('A') + i) for i in range(len(anomaly_list))]

    for i, item in enumerate(anomaly_list):
        item = item.strip()
        letter = letters[i]

        match = re.match(r'^([A-Z])[.:\)\-]?\s*(.+)$', item, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            desc = match.group(2).strip()
            mapping[letter] = desc
        else:
            mapping[letter] = item
    return mapping


def abcd_to_type(abcd: str, candidate_mapping: dict) -> str:
    """Convert ABCD label to description; return original string if not a single letter."""
    if not abcd:
        return ""
    abcd_stripped = abcd.strip()
    if len(abcd_stripped) == 1 and abcd_stripped.upper() in candidate_mapping:
        return candidate_mapping[abcd_stripped.upper()]
    return abcd_stripped


# --- Dataset name normalization ---

def normalize_dataset_from_path(path: str) -> str:
    """Extract and normalize dataset name from image path."""
    for ds in ['DS-MVTec', 'MVTec-LOCO', 'VisA', 'GoodsAD']:
        if ds in path:
            return {'DS-MVTec': 'MVTec', 'MVTec-LOCO': 'MVTec-LOCO', 'VisA': 'VisA', 'GoodsAD': 'GoodsAD'}.get(ds, ds)
    return 'MVTec'


# --- KnowledgeIndex: exact-match retrieval ---

class KnowledgeIndex:
    """Exact-match knowledge index built from candidates.json.

    Query flow: model query → match variant in candidates[ds][cls] → get canon → get knowledge text.
    """

    def __init__(self, candidates_path: str, knowledge_path: str):
        with open(candidates_path, "r", encoding="utf-8") as f:
            self.candidates: dict = json.load(f)
        with open(knowledge_path, "r", encoding="utf-8") as f:
            self.knowledge: dict = json.load(f)

        # Build variant_key -> (ds, cls, canon) mapping
        self.variant_map: dict[tuple, tuple] = {}
        for dataset, classes in self.candidates.items():
            for class_name, types in classes.items():
                if isinstance(types, list):
                    for t in types:
                        k = t.lower().strip().rstrip(".")
                        map_key = (k, dataset, class_name)
                        if map_key not in self.variant_map:
                            self.variant_map[map_key] = (dataset, class_name, t)
                elif isinstance(types, dict):
                    for canonical, variants in types.items():
                        for v in [canonical] + (variants if isinstance(variants, list) else []):
                            k = v.lower().strip().rstrip(".")
                            map_key = (k, dataset, class_name)
                            if map_key not in self.variant_map:
                                self.variant_map[map_key] = (dataset, class_name, canonical)

    def _get_knowledge(self, dataset: str, class_name: str, canonical: str) -> str | None:
        return self.knowledge.get(dataset, {}).get(class_name, {}).get(canonical)

    def search(self, query: str, dataset: str, class_name: str) -> str:
        """Exact-match search: return knowledge text only if query matches a variant."""
        q = query.lower().strip().rstrip(".")
        map_key = (q, dataset, class_name)
        if map_key in self.variant_map:
            _ds, _cls, canon = self.variant_map[map_key]
            return self._get_knowledge(_ds, _cls, canon) or ""
        return ""


# --- Global caches ---

_knowledge_index_cache: dict[str, KnowledgeIndex] = {}


def get_knowledge_index(candidates_path: str, knowledge_path: str) -> KnowledgeIndex:
    """Get or create KnowledgeIndex (cached by paths)."""
    cache_key = f"{candidates_path}|{knowledge_path}"
    if cache_key not in _knowledge_index_cache:
        _knowledge_index_cache[cache_key] = KnowledgeIndex(candidates_path, knowledge_path)
    return _knowledge_index_cache[cache_key]


# --- IoU utility ---

def _compute_iou_normalized(bbox_pred: list, bbox_gt: list) -> float:
    """Compute IoU of two normalized [x1,y1,x2,y2] bounding boxes."""
    x1 = max(bbox_pred[0], bbox_gt[0])
    y1 = max(bbox_pred[1], bbox_gt[1])
    x2 = min(bbox_pred[2], bbox_gt[2])
    y2 = min(bbox_pred[3], bbox_gt[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = max(0.0, (bbox_pred[2] - bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1]))
    area2 = max(0.0, (bbox_gt[2] - bbox_gt[0]) * (bbox_gt[3] - bbox_gt[1]))
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


# --- IADTool base class ---

class IADTool(BaseTool):
    """Base class: image loading, compression, bbox validation and cropping."""

    MIN_DIMENSION = 28

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict = {}
        self.resize_crop_image = config.get("resize_crop_image", False)
        self.max_image_pixels = config.get("max_image_pixels", 1048576)
        self.min_image_pixels = config.get("min_image_pixels", 784)

    def _compress_image(self, image: Any) -> Any:
        """Resize image to fit within pixel limits."""
        import PIL.Image
        import math
        w, h = image.size
        pixels = w * h

        if pixels > self.max_image_pixels:
            resize_factor = math.sqrt(self.max_image_pixels / pixels)
            w, h = int(w * resize_factor), int(h * resize_factor)
            image = image.resize((w, h))

        if (image.width * image.height) < self.min_image_pixels:
            resize_factor = math.sqrt(self.min_image_pixels / (image.width * image.height))
            w, h = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((w, h))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _validate_bbox(self, x1, y1, x2, y2, max_val: int = 1000) -> bool:
        try:
            if not (0 <= x1 <= max_val and 0 <= y1 <= max_val
                    and 0 <= x2 <= max_val and 0 <= y2 <= max_val):
                return False
            if not (x1 < x2 and y1 < y2):
                return False
            h, w = y2 - y1, x2 - x1
            if min(h, w) == 0 or max(h, w) / min(h, w) > 100:
                return False
            return True
        except Exception:
            return False

    def _normalized_to_pixel_bbox(
        self, bbox_2d: list, image_width: int, image_height: int, max_val: int = 1000
    ) -> Optional[list]:
        x1, y1, x2, y2 = bbox_2d
        if not self._validate_bbox(x1, y1, x2, y2, max_val):
            return None
        left, top     = x1 * image_width / max_val, y1 * image_height / max_val
        right, bottom = x2 * image_width / max_val, y2 * image_height / max_val
        # Enforce minimum dimension
        if (bottom - top) < self.MIN_DIMENSION or (right - left) < self.MIN_DIMENSION:
            cx, cy = (left + right) / 2, (top + bottom) / 2
            hw = max(self.MIN_DIMENSION / 2, (right - left) / 2)
            hh = max(self.MIN_DIMENSION / 2, (bottom - top) / 2)
            left   = max(0, cx - hw)
            top    = max(0, cy - hh)
            right  = min(image_width,  left + 2 * hw)
            bottom = min(image_height, top  + 2 * hh)
        return [floor(left), floor(top), ceil(right), ceil(bottom)]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)
        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in kwargs")
        img = fetch_image({"image": image})

        anomaly_list = kwargs.get("anomaly_list", [])
        candidate_mapping = parse_abcd_candidates(anomaly_list) if anomaly_list else {}

        image_path = image if isinstance(image, str) else None
        dataset = normalize_dataset_from_path(image_path) if image_path else "MVTec"

        self._instance_dict[instance_id] = {
            "image":      img,
            "image_path": image_path,
            "class_name": kwargs.get("class_name", ""),
            "dataset":    dataset,
            "candidate_mapping": candidate_mapping,
        }
        return instance_id, ToolResponse()

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)


# --- InspectTool: inspect(crop, query) ---

INSPECT_SCHEMA = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {
        "name": "inspect",
        "description": (
            "Zoom into a specific region of the image and retrieve relevant defect "
            "descriptions. Returns a cropped image and text knowledge for the query. "
            "Query should be a LETTER (A, B, C, D, etc.) from the candidate list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "crop": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4, "maxItems": 4,
                    "description": (
                        "Bounding box [x1, y1, x2, y2] with integer values in [0, 1000]."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "The LETTER (A, B, C, D, etc.) of the anomaly type from the "
                        "candidate list provided in the user message; choose the letter "
                        "that best matches your current visual hypothesis."
                    ),
                },
            },
            "required": ["crop", "query"],
        },
    },
})


class InspectTool(IADTool):
    """inspect(crop, query): crop image region and retrieve domain knowledge.

    - crop:  bbox [x1, y1, x2, y2] in [0, 1000]
    - query: ABCD letter or defect type keyword
    - Returns step IoU for step-level credit assignment
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        super().__init__(config, tool_schema or INSPECT_SCHEMA)
        self.knowledge_path = config.get(
            "knowledge_path",
            os.getenv("VERL_DOMAIN_KNOWLEDGE_PATH",
                      "knowledge_base/knowledge.json"),
        )
        self.candidates_path = config.get(
            "candidates_path",
            os.getenv("VERL_CANDIDATES_PATH",
                      "knowledge_base/candidates.json"),
        )
        try:
            get_knowledge_index(self.candidates_path, self.knowledge_path)
        except Exception:
            pass

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        crop  = parameters.get("crop")
        query = (parameters.get("query") or "").strip()

        if not crop or len(crop) != 4:
            return (
                ToolResponse(text="Error: crop must be [x1, y1, x2, y2] normalized floats."),
                0.0,
                {"success": False, "step_iou": 0.0},
            )

        instance_data = self._instance_dict[instance_id]
        image      = instance_data["image"]
        image_path = instance_data.get("image_path", "")
        class_name = instance_data.get("class_name", "")
        dataset    = instance_data.get("dataset", "MVTec")
        candidate_mapping = instance_data.get("candidate_mapping", {})
        W, H       = image.size

        # 1. Crop image
        try:
            pixel_bbox = self._normalized_to_pixel_bbox(crop, W, H)
            if pixel_bbox is None:
                return (
                    ToolResponse(text=f"Error: crop {crop} is invalid or too small."),
                    0.0,
                    {"success": False, "step_iou": 0.0},
                )
            cropped_image = image.crop(pixel_bbox)
            if self.resize_crop_image:
                cropped_image = self._compress_image(cropped_image)
            elif cropped_image.mode != "RGB":
                cropped_image = cropped_image.convert("RGB")
        except Exception as e:
            return ToolResponse(text=f"Error cropping: {e}"), 0.0, {"success": False, "step_iou": 0.0}

        # 2. Knowledge retrieval
        actual_desc = abcd_to_type(query, candidate_mapping) if query else ""
        search_query = actual_desc if actual_desc else query

        knowledge_text = self._query_knowledge(class_name, dataset, search_query, actual_desc or query)

        # 3. Step IoU reward
        step_iou   = 0.0
        agent_data = kwargs.get("agent_data")
        if agent_data is not None:
            gt_bboxes = agent_data.extra_info.get("bbox", []) or []
            if gt_bboxes and isinstance(gt_bboxes[0], (int, float)):
                gt_bboxes = [gt_bboxes]
            if gt_bboxes:
                step_iou = max(_compute_iou_normalized(crop, gt) for gt in gt_bboxes)
            agent_data.extra_info["last_step_iou"] = step_iou

            # SSGiGPO: compute step_reward
            step_reward = 0.0

            ground_truth = agent_data.extra_info.get("ground_truth", {})
            is_anomaly = ground_truth.get("is_anomaly", True) if isinstance(ground_truth, dict) else True

            if is_anomaly:
                gt_anomaly_type = ground_truth.get("anomaly_type", "") if isinstance(ground_truth, dict) else ""
                actual_query = abcd_to_type(query, candidate_mapping) if query else ""

                if gt_anomaly_type and actual_query:
                    if gt_anomaly_type.lower() in actual_query.lower() or actual_query.lower() in gt_anomaly_type.lower():
                        step_reward += 1.0

                dynamic_iou_threshold = agent_data.extra_info.get("dynamic_iou_threshold", 0.3)
                if step_iou > dynamic_iou_threshold:
                    step_reward += 1.0

            if "step_rewards" not in agent_data.extra_fields:
                agent_data.extra_fields["step_rewards"] = []
            agent_data.extra_fields["step_rewards"].append(step_reward)

            # SSGiGPO: record observation for anchor_obs construction
            if "anchor_obs_list" not in agent_data.extra_fields:
                agent_data.extra_fields["anchor_obs_list"] = []
            anchor_obs = {
                "is_initial": False,
                "query_type": (abcd_to_type(query, candidate_mapping) if query else "").strip(),
                "bbox": [float(v) for v in crop] if isinstance(crop, list) and len(crop) == 4 else [0, 0, 1, 1],
            }
            agent_data.extra_fields["anchor_obs_list"].append(anchor_obs)

        return (
            ToolResponse(text=knowledge_text, image=[cropped_image]),
            step_iou,
            {"success": True, "step_iou": step_iou, "crop": crop,
             "query": query, "pixel_bbox": pixel_bbox},
        )

    def _query_knowledge(self, class_name: str, dataset: str,
                         search_query: str, display_name: str) -> str:
        """Query knowledge: always return normal knowledge first, then anomaly knowledge."""
        parts = []

        try:
            index = get_knowledge_index(self.candidates_path, self.knowledge_path)

            good = index._get_knowledge(dataset, class_name, "good")
            if good:
                parts.append(f"[{class_name} — Normal]\n{good}")

            know = index.search(search_query, dataset, class_name)
            if know:
                parts.append(f"[{class_name} — {display_name}]\n{know}")
            else:
                parts.append(f"[{class_name} — {display_name}] No description available.")
        except Exception:
            parts.append(f"[{class_name}] Knowledge retrieval unavailable.")

        return "\n\n".join(parts)
