"""Penalty functions: force, side approach, push, and one-shot wrappers."""
import torch
import functools
from ..helpers import _milestone_gate, _sensor_force, _current_mass


def penalty_once(original_func, penalty_name):
    """Wrap a penalty function to fire only once per episode (like milestones)."""
    def wrapper(env, **kwargs):
        raw = original_func(env, **kwargs)
        condition = raw > 0.5
        return _milestone_gate(f"penalty_{penalty_name}", env, condition)
    functools.update_wrapper(wrapper, original_func)
    return wrapper


def contact_force_penalty(env, robot_cfg=None):
    """Per-step penalty when gripper contact force exceeds mass-based limit."""
    mass_kg = _current_mass["kg"]
    force_limit = mass_kg * 30.0 + 10.0
    try:
        gs = env.scene["gripper_contact"]
        js = env.scene["jaw_contact"]
        gf = _sensor_force(gs)
        jf = _sensor_force(js)
        while gf.dim() > 1: gf = gf.max(dim=-1).values
        while jf.dim() > 1: jf = jf.max(dim=-1).values
        force = torch.max(gf, jf)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)
    excess = torch.clamp(force - force_limit, min=0.0)
    return excess / max(force_limit, 1.0)


def penalty_side_approach(env, near_dist=0.15, height_margin=0.05,
                          object_cfg=None, ee_frame_cfg=None, robot_cfg=None):
    """Penalty for crawling toward tube at tube height instead of from above."""
    from leisaac.tasks.lift_cube.mdp.rewards import _get_ee_obj_dist
    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)

    jaw_mid = (ee.data.target_pos_w[..., 0, :] + ee.data.target_pos_w[..., 1, :]) / 2
    jaw_z = jaw_mid[:, 2]
    tube_z = obj.data.root_pos_w[:, 2]

    height_above = jaw_z - tube_z
    is_near = dist < near_dist
    is_crawling = height_above < height_margin

    height_err = torch.clamp(height_margin - height_above, min=0.0) / max(height_margin, 0.01)
    near_frac = torch.clamp(1.0 - dist / near_dist, min=0.0)
    severity = height_err * near_frac
    return torch.where(is_near & is_crawling, severity, torch.zeros_like(dist))


_cube_initial_xy = {}

def penalty_push_object(env, object_cfg=None, ee_frame_cfg=None, robot_cfg=None,
                        safe_dist=0.02, max_dist=0.05):
    """Continuous penalty for pushing the object horizontally."""
    from isaaclab.assets import RigidObject
    obj: RigidObject = env.scene[object_cfg.name]
    obj_xy = obj.data.root_pos_w[:, :2]
    num_envs = obj_xy.shape[0]

    if "xy" not in _cube_initial_xy or _cube_initial_xy["xy"].shape[0] != num_envs:
        _cube_initial_xy["xy"] = obj_xy.clone()
    just_reset = env.episode_length_buf <= 1
    _cube_initial_xy["xy"][just_reset] = obj_xy[just_reset]

    push_dist = torch.norm(obj_xy - _cube_initial_xy["xy"], dim=1)
    penalty = torch.clamp((push_dist - safe_dist) / (max_dist - safe_dist), min=0.0, max=1.0)
    return penalty
