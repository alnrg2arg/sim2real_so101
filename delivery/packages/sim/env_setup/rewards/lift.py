"""Lift rewards: no achieved_X. All conditions real-time every step."""
from packages.sim.env_setup.rewards.grasp import _is_holding as _is_holding_grasp
import torch
from ..config import CUBE_INITIAL_HEIGHT
from ..helpers import (
    _milestone_gate, _milestone_gate_batch, _both_jaws_contact,
    _dyn_max_force, _current_mass,
    _read_gripper_force, _read_cube_contact,
)

_lift_counter = {}
_grasp_hold_ref = None  # will reference grasp.py's counter

def lift_progressive(env, object_cfg=None, ee_frame_cfg=None, robot_cfg=None,
                     grasp_threshold=0.26, max_height=0.20, max_hold_steps=120):
    """Lift 50mm-200mm. Conditions: is_holding + grasp_hold 3s complete."""
    from isaaclab.assets import RigidObject, Articulation
    from packages.sim.env_setup.rewards.grasp import _grasp_hold_counter, _ge_counter
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = obj.data.root_pos_w.shape[0]
    device = obj.data.root_pos_w.device

    # Real-time: is_holding + grasp conditions met
    is_hold, gripper_force = _is_holding_grasp(env, robot, 0.1, object_cfg, ee_frame_cfg, robot_cfg)

    # grasp conditions (real-time): ge done + hold 3s done
    mass_kg = _current_mass["kg"]
    dyn_mf = _dyn_max_force(mass_kg)
    ge_cnt = _ge_counter.get("cnt", torch.zeros(num_envs, device=device))
    hold_cnt = _grasp_hold_counter.get("cnt", torch.zeros(num_envs, device=device))
    grasp_complete = is_hold & (gripper_force > dyn_mf) & (ge_cnt >= 120) & (hold_cnt >= 120)

    lift_height = obj.data.root_pos_w[:, 2] - CUBE_INITIAL_HEIGHT

    # Time counter
    if "cnt" not in _lift_counter or _lift_counter["cnt"].shape[0] != num_envs:
        _lift_counter["cnt"] = torch.zeros(num_envs, device=device)
    cnt = _lift_counter["cnt"]
    cnt[env.episode_length_buf <= 1] = 0
    lifting_now = grasp_complete & (lift_height > 0.01)
    cnt[:] = torch.where(lifting_now, cnt + 1, (cnt - 3).clamp(min=0))  # grace

    # Lift milestones: 50mm to max_height — VECTORIZED
    max_mm = int(max_height * 1000)
    lift_stages = list(range(50, max_mm + 1))
    total = len(lift_stages)
    M = total

    # Build keys, thresholds, weights
    keys = [f"lift_{_mm}mm" for _mm in lift_stages]
    thresholds = torch.tensor([_mm / 1000.0 for _mm in lift_stages], device=device)  # (M,)
    weights = torch.tensor([(idx + 1) / total for idx in range(M)], device=device)   # (M,)

    # conditions: (M, N) — broadcasting: thresholds (M,1) vs lift_height (N,)
    conditions = grasp_complete.unsqueeze(0) & (lift_height.unsqueeze(0) > thresholds.unsqueeze(1))  # (M, N)

    reward = _milestone_gate_batch(keys, env, conditions, weights)

    time_score = torch.clamp(cnt / max_hold_steps, min=0.0, max=1.0)
    target_h = max_height - 0.01
    full_lift = grasp_complete & (lift_height > target_h) & (time_score >= 0.99)
    first = _milestone_gate("lift", env, full_lift)
    reward = reward + first.float()
    return reward


_lift_hold_counter = {}

def lift_hold_60(env, object_cfg=None, ee_frame_cfg=None, robot_cfg=None,
                 grasp_threshold=0.26, min_height=0.20):
    """Hold at 20cm+ for 3s. All conditions real-time."""
    from isaaclab.assets import RigidObject, Articulation
    from packages.sim.env_setup.rewards.grasp import _grasp_hold_counter, _ge_counter
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = obj.data.root_pos_w.shape[0]
    device = obj.data.root_pos_w.device

    # Real-time conditions
    is_hold, gripper_force = _is_holding_grasp(env, robot, 0.1, object_cfg, ee_frame_cfg, robot_cfg)

    mass_kg = _current_mass["kg"]
    dyn_mf = _dyn_max_force(mass_kg)
    ge_cnt = _ge_counter.get("cnt", torch.zeros(num_envs, device=device))
    hold_cnt = _grasp_hold_counter.get("cnt", torch.zeros(num_envs, device=device))
    grasp_complete = is_hold & (gripper_force > dyn_mf) & (ge_cnt >= 120) & (hold_cnt >= 120)

    lift_height = obj.data.root_pos_w[:, 2] - CUBE_INITIAL_HEIGHT
    holding_high = grasp_complete & (lift_height > min_height)

    if "cnt" not in _lift_hold_counter or _lift_hold_counter["cnt"].shape[0] != num_envs:
        _lift_hold_counter["cnt"] = torch.zeros(num_envs, device=device)
    cnt = _lift_hold_counter["cnt"]
    cnt[env.episode_length_buf <= 1] = 0
    cnt[:] = torch.where(holding_high, cnt + 1, (cnt - 3).clamp(min=0))  # grace

    # 60 milestones — VECTORIZED
    M = 60
    keys = [f"lift_hold_{i:02d}" for i in range(1, M + 1)]
    steps_needed = torch.tensor([i * 3 for i in range(1, M + 1)], device=device, dtype=cnt.dtype)  # (M,)
    weights = torch.tensor([i / 60.0 for i in range(1, M + 1)], device=device)  # (M,)

    # conditions: (M, N) — holding_high (N,) & cnt >= steps_needed (M, 1)
    conditions = holding_high.unsqueeze(0) & (cnt.unsqueeze(0) >= steps_needed.unsqueeze(1))  # (M, N)

    reward = _milestone_gate_batch(keys, env, conditions, weights)

    first = _milestone_gate("success", env, holding_high & (cnt >= 180))
    reward = reward + first.float()
    return reward
