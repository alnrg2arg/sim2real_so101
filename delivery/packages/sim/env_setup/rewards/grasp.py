"""Grasp rewards: no achieved_X variables. All conditions evaluated real-time every step."""
import torch
from ..config import CUBE_INITIAL_HEIGHT
from ..helpers import (
    _milestone_gate, _milestone_gate_batch,
    _dyn_max_force, _current_mass,
    _read_gripper_force,
)


def _is_holding(env, robot, contact_force_min=0.01, object_cfg=None, ee_frame_cfg=None, robot_cfg=None):
    """Common real-time check: gripper closed + both jaws touching cube (geometric) + between jaws."""
    from ..helpers import _object_between_jaws
    gripper_pos = robot.data.joint_pos[:, -1]
    gripper_closed = gripper_pos < 0.26
    gripper_force = _read_gripper_force(env)

    if object_cfg is not None and ee_frame_cfg is not None:
        # Geometric contact: each jaw close enough to cube
        cube_pos = env.scene[object_cfg.name].data.root_pos_w  # (N, 3)
        jaw0_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :]  # gripper body
        jaw1_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 1, :]  # jaw tip
        dist_jaw0 = torch.norm(jaw0_pos - cube_pos, dim=-1)  # (N,)
        dist_jaw1 = torch.norm(jaw1_pos - cube_pos, dim=-1)  # (N,)
        # Both jaws within contact distance of cube
        contact_dist = 0.05  # 3cm threshold
        both_touching_cube = (dist_jaw0 < contact_dist) & (dist_jaw1 < contact_dist)
        # Between jaws: cube geometrically between jaw tips
        between = _object_between_jaws(env, object_cfg, ee_frame_cfg, robot_cfg)
        if between is None:
            between = torch.ones_like(gripper_closed)
    else:
        both_touching_cube = torch.ones_like(gripper_closed)
        between = torch.ones_like(gripper_closed)

    is_hold = gripper_closed & both_touching_cube & between & (gripper_force > 0.005)
    # Debug: check which condition fails
    if not hasattr(_is_holding, "_dbg_cnt"):
        _is_holding._dbg_cnt = 0
    _is_holding._dbg_cnt += 1
    if _is_holding._dbg_cnt % 500 == 0:
        gc = gripper_closed.float().mean().item()
        bt = both_touching_cube.float().mean().item()
        bw = between.float().mean().item()
        ff = (gripper_force > 0.005).float().mean().item()
        ih = is_hold.float().mean().item()
        if gc > 0.1:  # only log when gripper is closing
            print(f"[IS_HOLD] closed={gc:.2f} touching={bt:.2f} between={bw:.2f} force={ff:.2f} => is_hold={ih:.2f}", flush=True)
    return is_hold, gripper_force

def grasp_start(env, object_cfg=None, ee_frame_cfg=None, robot_cfg=None,
                contact_force_min=0.01):
    """Both jaws + cube contact simultaneously."""
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene[robot_cfg.name]
    is_hold, _ = _is_holding(env, robot, contact_force_min, object_cfg, ee_frame_cfg, robot_cfg)
    return _milestone_gate("grasp_start", env, is_hold)


# ── Grasp enough: force + time accumulation ──

_ge_counter = {}

def grasp_enough_continuous(env, object_cfg=None, ee_frame_cfg=None, robot_cfg=None,
                            grasp_threshold=0.26):
    """Force + time accumulation. All conditions real-time. VECTORIZED."""
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    is_hold, gripper_force = _is_holding(env, robot, 0.01, object_cfg, ee_frame_cfg, robot_cfg)
    mass_kg = _current_mass["kg"]
    dyn_mf = _dyn_max_force(mass_kg)

    # Counter: accumulates while is_hold is True
    if "cnt" not in _ge_counter or _ge_counter["cnt"].shape[0] != num_envs:
        _ge_counter["cnt"] = torch.zeros(num_envs, device=device)
    cnt = _ge_counter["cnt"]
    cnt[env.episode_length_buf <= 1] = 0
    cnt[:] = torch.where(is_hold, cnt + 1, (cnt - 3).clamp(min=0))  # grace: -3 per miss

    # Progressive milestones — VECTORIZED
    # Build the full list of (key, force_frac, steps) for all milestones
    _ge_major = [
        ("ge_01",  0.0,   6),   ("ge_01b", 0.04, 12),
        ("ge_02",  0.07, 18),   ("ge_02b", 0.14, 24),
        ("ge_03",  0.21, 30),   ("ge_04",  0.36, 30),
        ("ge_05",  0.50, 30),   ("ge_06",  0.64, 30),
        ("ge_07",  0.79, 30),   ("ge_08",  1.00, 30),
        ("ge_09",  1.00, 60),   ("ge_full",1.00, 60),
    ]

    # total_stages matches original: len(_ge_major) * 6 = 72
    total_stages = len(_ge_major) * 6

    # Expand: each major + 5 sub-milestones (except last major)
    all_keys = []
    all_force_thresholds = []  # absolute force values
    all_step_thresholds = []
    all_weights = []
    stage_idx = 0

    for i, (name, frac, steps) in enumerate(_ge_major):
        force_thresh = frac * dyn_mf
        stage_idx += 1
        all_keys.append(name)
        all_force_thresholds.append(force_thresh)
        all_step_thresholds.append(float(steps))
        all_weights.append(stage_idx / total_stages)

        if i < len(_ge_major) - 1:
            nfrac, ns = _ge_major[i + 1][1], _ge_major[i + 1][2]
            for j in range(1, 6):
                t = j / 6.0
                sub_frac = frac + (nfrac - frac) * t
                sub_s = int(steps + (ns - steps) * t)
                sub_thresh = sub_frac * dyn_mf
                stage_idx += 1
                all_keys.append(f"{name}_{j}")
                all_force_thresholds.append(sub_thresh)
                all_step_thresholds.append(float(sub_s))
                all_weights.append(stage_idx / total_stages)

    M_ge = len(all_keys)

    # Build tensors
    force_thresholds_t = torch.tensor(all_force_thresholds, device=device)  # (M_ge,)
    step_thresholds_t = torch.tensor(all_step_thresholds, device=device)    # (M_ge,)
    weights_ge = torch.tensor(all_weights, device=device)                    # (M_ge,)

    # Build conditions (M_ge, N):
    # cond = is_hold & (cnt >= steps) & (gripper_force > force_thresh if force_thresh > 0.01)
    base_cond_hold_cnt = is_hold.unsqueeze(0) & (cnt.unsqueeze(0) >= step_thresholds_t.unsqueeze(1))  # (M_ge, N)
    force_check = gripper_force.unsqueeze(0) > force_thresholds_t.unsqueeze(1)  # (M_ge, N)
    needs_force = (force_thresholds_t > 0.01).unsqueeze(1)  # (M_ge, 1)
    # Where force_thresh > 0.01: must also pass force check; otherwise: just hold+cnt
    conditions_ge = base_cond_hold_cnt & (force_check | ~needs_force)  # (M_ge, N)

    reward = _milestone_gate_batch(all_keys, env, conditions_ge, weights_ge)

    # ge_hold: 20 milestones over sustained full-force hold — VECTORIZED
    M_hold = 20
    _full_cond = is_hold & (gripper_force > dyn_mf)  # (N,)
    keys_hold = [f"ge_hold_{i:02d}" for i in range(1, M_hold + 1)]
    steps_hold = torch.tensor([i * 6.0 for i in range(1, M_hold + 1)], device=device)  # (M_hold,)
    weights_hold = torch.tensor([i / 20.0 for i in range(1, M_hold + 1)], device=device)  # (M_hold,)

    # conditions (M_hold, N)
    conditions_hold = _full_cond.unsqueeze(0) & (cnt.unsqueeze(0) >= steps_hold.unsqueeze(1))

    reward = reward + _milestone_gate_batch(keys_hold, env, conditions_hold, weights_hold)

    # grasp_enough: full force + 120 steps
    _milestone_gate("grasp_enough", env, is_hold & (gripper_force > dyn_mf) & (cnt >= 120))
    return reward


# ── Grasp verified: lift 5cm + sustain 3 seconds ──

_grasp_hold_counter = {}

def grasp_contact_verified(env, object_cfg=None, ee_frame_cfg=None, robot_cfg=None,
                           grasp_threshold=0.26, contact_force_min=0.1,
                           hold_steps=120):
    """Phase 1: lift to 5cm. Phase 2: sustain 3s. VECTORIZED.
    No achieved_X — all conditions checked real-time every step."""
    from isaaclab.assets import RigidObject, Articulation
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    is_hold, gripper_force = _is_holding(env, robot, contact_force_min, object_cfg, ee_frame_cfg, robot_cfg)
    mass_kg = _current_mass["kg"]
    dyn_mf = _dyn_max_force(mass_kg)

    # grasp_enough conditions (real-time, not sticky)
    ge_cnt = _ge_counter.get("cnt", torch.zeros(num_envs, device=device))
    is_enough = is_hold & (gripper_force > dyn_mf) & (ge_cnt >= 120)

    obj_h = obj.data.root_pos_w[:, 2]

    # ── Phase 1: Progressive lift to 5cm — VECTORIZED ──
    M1 = 50
    keys_lift = [f"grasp_lift_{mm}mm" for mm in range(1, M1 + 1)]
    thresholds_lift = torch.tensor(
        [CUBE_INITIAL_HEIGHT + mm / 1000.0 for mm in range(1, M1 + 1)],
        device=device,
    )  # (50,)
    weights_lift = torch.tensor([mm / 50.0 for mm in range(1, M1 + 1)], device=device)  # (50,)

    # conditions (50, N): is_enough & obj_h > threshold
    conditions_lift = is_enough.unsqueeze(0) & (obj_h.unsqueeze(0) > thresholds_lift.unsqueeze(1))

    reward = _milestone_gate_batch(keys_lift, env, conditions_lift, weights_lift)

    # ── Phase 2: Sustain 3 seconds at 5cm+ ──
    lifted_5cm = obj_h > CUBE_INITIAL_HEIGHT + 0.05
    holding = is_enough & lifted_5cm

    # Counter RESETS if conditions fail
    if "cnt" not in _grasp_hold_counter or _grasp_hold_counter["cnt"].shape[0] != num_envs:
        _grasp_hold_counter["cnt"] = torch.zeros(num_envs, device=device)
    cnt = _grasp_hold_counter["cnt"]
    cnt[env.episode_length_buf <= 1] = 0
    cnt[:] = torch.where(holding, cnt + 1, (cnt - 3).clamp(min=0))  # grace: -3 per miss

    # 12 progressive milestones — VECTORIZED
    M2 = 12
    keys_hold = [f"grasp_hold_{i:02d}" for i in range(1, M2 + 1)]
    steps_hold = torch.tensor([i * 10.0 for i in range(1, M2 + 1)], device=device, dtype=cnt.dtype)  # (12,)
    weights_hold = torch.tensor([i / 12.0 for i in range(1, M2 + 1)], device=device)  # (12,)

    # conditions (12, N): holding & cnt >= steps
    conditions_hold = holding.unsqueeze(0) & (cnt.unsqueeze(0) >= steps_hold.unsqueeze(1))

    reward = reward + _milestone_gate_batch(keys_hold, env, conditions_hold, weights_hold)

    # GRASP = final milestone (hold_steps = 120 steps = 2 seconds)
    grasped = holding & (cnt >= hold_steps)
    first = _milestone_gate("grasp", env, grasped)
    reward = reward + first.float() * 2.0

    return reward
