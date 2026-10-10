# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
SSGiGPO (Group-in-Group Policy Optimization) — step-level advantage.

Flattens per-trajectory steps, groups by anchor_obs, normalizes step rewards,
then combines episode-level and step-level advantages.
"""

from collections import defaultdict
from typing import Optional
from uuid import uuid4

import numpy as np
import torch

from verl.trainer.ppo.core_algos import AdvantageEstimator, register_adv_est


# --- Utilities ---

def to_hashable(x):
    """Convert an object into a hashable type for clustering/grouping."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        return str(x)


def compute_iou(bbox_a: list, bbox_b: list) -> float:
    """Compute IoU between two normalized bboxes [x1, y1, x2, y2]."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]))
    area_b = max(0.0, (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


# --- Step Grouping ---

def _is_same_anchor_state(obs_a: dict, obs_b: dict, iou_threshold: float = 0.5) -> bool:
    """Check if two anchor observations represent the same state (s_t).

    Initial states: all is_initial=True steps are grouped together.
    Non-initial states: same prev_observed_type + prev_observed_bbox IoU > threshold.
    """
    if obs_a.get("is_initial", False) and obs_b.get("is_initial", False):
        return True

    if obs_a.get("is_initial", False) != obs_b.get("is_initial", False):
        return False

    if obs_a.get("prev_observed_type", "") != obs_b.get("prev_observed_type", ""):
        return False

    bbox_a = obs_a.get("prev_observed_bbox", [0, 0, 1, 1])
    bbox_b = obs_b.get("prev_observed_bbox", [0, 0, 1, 1])
    return compute_iou(bbox_a, bbox_b) > iou_threshold


def build_step_group(
    anchor_obs_list: list,
    index: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Build step groups by clustering anchor observations within each episode group.

    Args:
        anchor_obs_list: flat list of anchor observations, one per step
        index: episode group UIDs, one per step
        iou_threshold: IoU threshold for grouping similar bboxes

    Returns:
        np.ndarray of step group UIDs
    """
    step_group_uids = np.empty(len(anchor_obs_list), dtype=object)
    unique_indices = np.unique(index)

    for idx in unique_indices:
        locs = np.where(index == idx)[0]
        obs_group = [anchor_obs_list[i] for i in locs]

        clusters = []
        for i, obs in enumerate(obs_group):
            placed = False
            for cluster in clusters:
                if _is_same_anchor_state(obs, cluster["obs"], iou_threshold):
                    cluster["locs"].append(locs[i])
                    placed = True
                    break
            if not placed:
                clusters.append({"obs": obs, "locs": [locs[i]]})

        for cluster in clusters:
            uid = str(uuid4())
            for loc in cluster["locs"]:
                step_group_uids[loc] = uid

    if None in step_group_uids or np.any(step_group_uids == None):
        missing = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing}")

    return step_group_uids


# --- Episode-level Advantage (GRPO-style) ---

def episode_norm_reward(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True,
) -> torch.Tensor:
    """Compute episode-level advantage (GRPO-style).

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) episode group UIDs
        epsilon: avoid division by zero
        remove_std: if True, only subtract mean

    Returns:
        episode_advantages: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                id2std[idx] = torch.std(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"No score in prompt index: {idx}")

        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


# --- Step-level Advantage ---

def step_norm_reward(
    step_rewards: torch.Tensor,
    step_group_uids: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True,
) -> torch.Tensor:
    """Compute step-level advantage by normalizing within each step group.

    Args:
        step_rewards: (num_steps,) flat step-level rewards
        step_group_uids: (num_steps,) step group UIDs
        epsilon: avoid division by zero
        remove_std: if True, only subtract mean

    Returns:
        step_advantages: (num_steps,) per-step advantage scalars
    """
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        num_steps = scores.shape[0]
        for i in range(num_steps):
            id2score[step_group_uids[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
                id2std[idx] = torch.std(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"No score in step group: {idx}")

        for i in range(num_steps):
            if remove_std:
                scores[i] = scores[i] - id2mean[step_group_uids[i]]
            else:
                scores[i] = (scores[i] - id2mean[step_group_uids[i]]) / (id2std[step_group_uids[i]] + epsilon)

    return scores


# --- Main: SSGiGPO Advantage ---

@register_adv_est(AdvantageEstimator.SSGIGPO)
def compute_ssgigpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards_raw: list,
    anchor_obs_raw: list,
    response_mask: torch.Tensor,
    index: np.ndarray,
    step_advantage_w: float = 1.0,
    iou_threshold: float = 0.5,
    mode: str = "mean_norm",
    epsilon: float = 1e-6,
    config: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SSGiGPO advantages.

    Internal flow:
    1. Episode advantage: trajectory-level GRPO normalization
    2. Flatten steps across all trajectories
    3. Step grouping + advantage: normalize within step groups
    4. Aggregate: sum step advantages per trajectory
    5. Combine: final = episode_adv + step_advantage_w * step_adv

    Args:
        token_level_rewards: (bs, response_length)
        step_rewards_raw: list[list[float]], step rewards per trajectory
        anchor_obs_raw: list[list[dict]], anchor observations per trajectory
        response_mask: (bs, response_length)
        index: (bs,) episode group UIDs
        step_advantage_w: weight for step advantage
        iou_threshold: bbox IoU threshold for step grouping
        mode: "mean_norm" or "mean_std_norm"
        epsilon: avoid division by zero
        config: optional config dict

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    remove_std = (mode == "mean_norm")
    bs = token_level_rewards.shape[0]
    response_length = token_level_rewards.shape[-1]
    device = token_level_rewards.device

    # 1. Episode-level advantage
    episode_advantages = episode_norm_reward(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        remove_std=remove_std,
    )

    # 2. Flatten step data
    flat_step_rewards = []
    flat_anchor_obs = []
    flat_index = []
    traj_id_per_step = []

    has_step_data = step_rewards_raw is not None and anchor_obs_raw is not None

    if has_step_data:
        for traj_idx in range(bs):
            rewards = step_rewards_raw[traj_idx] if traj_idx < len(step_rewards_raw) else []
            obs_list = anchor_obs_raw[traj_idx] if traj_idx < len(anchor_obs_raw) else []

            if not isinstance(rewards, list):
                rewards = [rewards] if rewards else []
            if not isinstance(obs_list, list):
                obs_list = [obs_list] if isinstance(obs_list, dict) else []

            num_steps = max(len(rewards), len(obs_list))

            for step_idx in range(num_steps):
                r = rewards[step_idx] if step_idx < len(rewards) else 0.0
                obs = obs_list[step_idx] if step_idx < len(obs_list) else {"is_initial": True}

                flat_step_rewards.append(float(r))
                flat_anchor_obs.append(obs)
                flat_index.append(index[traj_idx])
                traj_id_per_step.append(traj_idx)

    # 3. No step data → zero step advantage
    if not has_step_data or len(flat_step_rewards) == 0:
        step_advantages_trajectory = torch.zeros(bs, device=device)
    else:
        # 4. Step grouping
        flat_index_arr = np.array(flat_index)
        step_group_uids = build_step_group(
            anchor_obs_list=flat_anchor_obs,
            index=flat_index_arr,
            iou_threshold=iou_threshold,
        )

        # 5. Step-level advantage
        flat_step_rewards_t = torch.tensor(flat_step_rewards, dtype=torch.float32, device=device)
        flat_step_advantages = step_norm_reward(
            step_rewards=flat_step_rewards_t,
            step_group_uids=step_group_uids,
            epsilon=epsilon,
            remove_std=remove_std,
        )

        # 6. Aggregate: sum step advantages per trajectory
        step_advantages_trajectory = torch.zeros(bs, device=device)
        for step_idx, traj_idx in enumerate(traj_id_per_step):
            step_advantages_trajectory[traj_idx] += flat_step_advantages[step_idx]

    # 7. Combine: final = episode_adv + step_advantage_w * step_adv
    step_adv_broadcast = step_advantages_trajectory.unsqueeze(-1).tile([1, response_length]) * response_mask
    advantages = episode_advantages + step_advantage_w * step_adv_broadcast

    return advantages, advantages
