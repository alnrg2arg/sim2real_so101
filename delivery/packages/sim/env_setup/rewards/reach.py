"""Reach reward: 30 distance stages from 30cm to 10cm."""
import torch
from ..helpers import _milestone_gate_batch


def reach_stages_30(env, object_cfg=None, ee_frame_cfg=None):
    """Reach in 30 stages: 30cm -> 10cm (~0.67cm per step). VECTORIZED."""
    from leisaac.tasks.lift_cube.mdp.rewards import _get_ee_obj_dist
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)
    device = dist.device
    N = dist.shape[0]
    M = 30

    keys = [f"reach_{i:02d}" for i in range(1, M + 1)]
    thresholds = torch.tensor(
        [0.30 - (0.30 - 0.10) * (i / 30.0) for i in range(1, M + 1)],
        device=device,
    )  # (M,)
    weights = torch.tensor([i / 30.0 for i in range(1, M + 1)], device=device)  # (M,)

    # conditions: (M, N) — dist (N,) < thresholds (M, 1)
    conditions = dist.unsqueeze(0) < thresholds.unsqueeze(1)  # (M, N)

    return _milestone_gate_batch(keys, env, conditions, weights)
