"""Shared helpers: milestone gate, sensor reading, geometry checks."""

import torch
from .config import (
    USE_DOPAMINE, DOPAMINE_ALPHA, DOPAMINE_FLOOR, EMA_BETA,
    GRIPPER_OPEN, GRIPPER_CLOSED,
)

# ── Module-level state ──
_milestones = {}        # key -> bool tensor (num_envs,)
_near_counters = {}
_current_mass = {"kg": 5.0}  # updated by CurriculumManager
_hit_rates_ema = {}     # key -> float (0~1)


def _milestone_gate(key, env, condition):
    """Returns 1.0 the FIRST time condition is met per episode.

    When USE_DOPAMINE is True, applies RPE scaling:
        R = r * (lambda + (1-lambda)(1-h_ema)^alpha)
    """
    from . import config as cfg  # re-read at call time so flag changes take effect

    num_envs = condition.shape[0]
    if key not in _milestones or _milestones[key].shape[0] != num_envs:
        _milestones[key] = torch.zeros(num_envs, device=condition.device, dtype=torch.bool)
    just_reset = env.episode_length_buf <= 1
    _milestones[key][just_reset] = False
    first_time = condition & ~_milestones[key]
    _milestones[key] |= condition

    if cfg.USE_DOPAMINE:
        h = _hit_rates_ema.get(key, 0.0)
        lam = cfg.DOPAMINE_FLOOR
        dopamine = lam + (1.0 - lam) * (1.0 - min(h, 0.99)) ** cfg.DOPAMINE_ALPHA
        return first_time.float() * dopamine

    return first_time.float()


def _milestone_gate_batch(keys, env, conditions, weights):
    """Vectorized batch milestone gate — replaces per-milestone Python loops.

    Args:
        keys: list of M milestone name strings
        env: environment (needs episode_length_buf)
        conditions: (M, N) bool tensor — condition per milestone per env
        weights: (M,) float tensor — reward weight per milestone

    Returns:
        (N,) reward tensor — weighted sum of first-time milestone fires
    """
    from . import config as cfg

    M = len(keys)
    N = conditions.shape[1]
    device = conditions.device
    just_reset = env.episode_length_buf <= 1  # (N,)

    # Ensure all milestone tensors exist and are correct size
    for key in keys:
        if key not in _milestones or _milestones[key].shape[0] != N:
            _milestones[key] = torch.zeros(N, device=device, dtype=torch.bool)

    # Stack milestone states into (M, N) tensor
    prev_states = torch.stack([_milestones[k] for k in keys], dim=0)  # (M, N)

    # Reset milestones for just-reset envs: broadcast (M, N) & (N,)
    prev_states[:, just_reset] = False

    # First time: condition is True AND milestone was not previously achieved
    first_time = conditions & ~prev_states  # (M, N)

    # Update milestone states: mark achieved
    new_states = prev_states | conditions  # (M, N)

    # Write back to _milestones dict
    for i, key in enumerate(keys):
        _milestones[key] = new_states[i]

    # Apply dopamine scaling if enabled
    if cfg.USE_DOPAMINE:
        lam = cfg.DOPAMINE_FLOOR
        alpha = cfg.DOPAMINE_ALPHA
        # Build dopamine scale per milestone: (M,)
        dopamine_scales = torch.empty(M, device=device)
        for i, key in enumerate(keys):
            h = _hit_rates_ema.get(key, 0.0)
            dopamine_scales[i] = lam + (1.0 - lam) * (1.0 - min(h, 0.99)) ** alpha
        # first_time (M, N) * weights (M, 1) * dopamine (M, 1) => (M, N), sum => (N,)
        reward = (first_time.float() * (weights * dopamine_scales).unsqueeze(1)).sum(dim=0)
    else:
        # first_time (M, N) * weights (M, 1) => (M, N), sum over milestones => (N,)
        reward = (first_time.float() * weights.unsqueeze(1)).sum(dim=0)

    return reward


def _get_counter(key, num_envs, device):
    """Get or create cumulative counter, reset on episode boundaries."""
    if key not in _near_counters or _near_counters[key].shape[0] != num_envs:
        _near_counters[key] = torch.zeros(num_envs, device=device)
    return _near_counters[key]


def _both_jaws_contact(env, min_force=0.5):
    """Pair-wise contact check via force_matrix_w (GPU native)."""
    try:
        grip_fm = env.scene["gripper_contact"].data.force_matrix_w
        jaw_fm = env.scene["jaw_contact"].data.force_matrix_w
        grip_f = grip_fm[:, 0, 0, :].norm(dim=-1)
        jaw_f = jaw_fm[:, 0, 0, :].norm(dim=-1)
        return (grip_f >= min_force) & (jaw_f >= min_force)
    except Exception as _e:
        if not hasattr(_both_jaws_contact, '_warned'):
            _both_jaws_contact._warned = True
            print(f"[WARN] _both_jaws_contact: {_e}", flush=True)
        return None


def _object_between_jaws(env, object_cfg, ee_frame_cfg, robot_cfg, max_dist=0.08):
    """Check if object is between the two jaws along the jaw-opening axis."""
    from isaaclab.assets import RigidObject
    from isaaclab.sensors import FrameTransformer
    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    obj_pos = obj.data.root_pos_w
    gripper_pos = ee.data.target_pos_w[..., 0, :]
    jaw_pos = ee.data.target_pos_w[..., 1, :]

    jaw_axis = jaw_pos - gripper_pos
    jaw_len_sq = (jaw_axis ** 2).sum(dim=1, keepdim=True).clamp(min=1e-6)
    obj_rel = obj_pos - gripper_pos
    proj = (obj_rel * jaw_axis).sum(dim=1, keepdim=True) / jaw_len_sq

    closest = gripper_pos + proj * jaw_axis
    lateral_dist = torch.norm(obj_pos - closest, dim=1)

    proj_flat = proj.reshape(-1)
    between = (proj_flat > -0.2) & (proj_flat < 1.2)
    close_to_axis = lateral_dist < max_dist
    return between & close_to_axis


def _dyn_max_force(mass_kg):
    """Dynamic max force target: heavier = more force, clamped 3N~14N."""
    return max(min(mass_kg * 30.0, 14.0), 3.0)







def _get_align_scores(env, object_cfg, ee_frame_cfg):
    import torch
    from isaaclab.assets import RigidObject
    from isaaclab.sensors import FrameTransformer
    from isaaclab.utils.math import euler_xyz_from_quat
    from leisaac.tasks.lift_cube.mdp.rewards import _get_ee_obj_dist
    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)
    # Pitch: gripper pointing downward (baseline method)
    ee_quat = ee.data.target_quat_w[:, 0, :]
    ee_roll, ee_pitch, ee_yaw = euler_xyz_from_quat(ee_quat)
    pitch_score = torch.clamp(-torch.sin(ee_pitch), min=0.0, max=1.0)
    # Yaw: perpendicular to object (baseline method)
    obj_quat = obj.data.root_quat_w
    _, _, obj_yaw = euler_xyz_from_quat(obj_quat)
    yaw_diff = obj_yaw - ee_yaw
    yaw_score = torch.cos(yaw_diff).abs()
    return dist, pitch_score, yaw_score
