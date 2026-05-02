"""Squint-exact reward for SO101 LiftCube."""

import torch
import numpy as np
from isaaclab.managers import SceneEntityCfg

# Squint SO101 start keyframe
REST_QPOS = torch.tensor(
    [0.0, 0.0, 0.0, np.pi / 2, -np.pi / 2, 60.0 * np.pi / 180.0]
)
# Stage 1 fold: easy target (shoulder_lift=-90 fully up, elbow=90 folded)
FOLD_QPOS = torch.tensor(
    [0.0, -90.0 * np.pi / 180.0, 90.0 * np.pi / 180.0, -45.0 * np.pi / 180.0, 0.0, -10.0 * np.pi / 180.0]
)
# Folded pose: shoulder_lift=-90, elbow=150, wrist_flex/roll=REST, gripper=-10
FOLDED_QPOS = torch.tensor(
    [0.0, -90.0 * np.pi / 180.0, 150.0 * np.pi / 180.0, np.pi / 2, -np.pi / 2, -10.0 * np.pi / 180.0]
)
# wrist_flex=90°(REST), wrist_roll=-90°(REST) — 접으면서 wrist 안정 유지


def _get_target_qpos(env):
    """Retrieve the target joint positions from the action manager terms."""
    am = env.action_manager
    targets = []
    for term in am._terms.values():
        if hasattr(term, "_target_qpos"):
            targets.append(term._target_qpos)
    if targets:
        return torch.cat(targets, dim=-1)
    return env.scene["robot"].data.joint_pos


def _compute_angle_between(a, b):
    """Compute angle between two vector batches."""
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.acos((a_n * b_n).sum(dim=-1).clamp(-1, 1))


def _quat_to_y_axis(q):
    """Extract y-axis direction from quaternion (w,x,y,z format).

    Given a unit quaternion q = (w, x, y, z), the y-column of the
    corresponding rotation matrix is:
        [2(xy + wz),  1 - 2(x^2 + z^2),  2(yz - wx)]
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        dim=-1,
    )


def _get_link_body_idx(robot, link_name_substr):
    """Find body index whose name contains *link_name_substr* (case-insensitive)."""
    for i, name in enumerate(robot.data.body_names):
        if link_name_substr in name.lower():
            return i
    return None


def _cube_between_jaws(env, max_tcp_dist=0.05):
    """Geometric check: cube center is BETWEEN the two jaw tips.

    Uses dot product of (fixed_tip→cube) and (moving_tip→cube).
    If cube is between jaws, these vectors point in opposite directions → dot < 0.
    If robot is pressing from one side (tip pushing), both vectors point same way → dot > 0.
    """
    ee = env.scene["ee_frame"]
    fixed_tip = ee.data.target_pos_w[:, 0, :]   # (N, 3)
    moving_tip = ee.data.target_pos_w[:, 1, :]   # (N, 3)
    cube_pos = env.scene["cube"].data.root_pos_w  # (N, 3)

    vec_to_fixed = fixed_tip - cube_pos    # cube → fixed jaw
    vec_to_moving = moving_tip - cube_pos  # cube → moving jaw

    # Dot product < 0 means cube is between the two tips
    dot = (vec_to_fixed * vec_to_moving).sum(dim=-1)
    between = dot < 0

    # Also check cube is near TCP (not far away)
    tcp = (fixed_tip + moving_tip) * 0.5
    tcp_dist = (cube_pos - tcp).norm(dim=-1)
    near = tcp_dist < max_tcp_dist

    return between & near


def _is_grasped(env, min_force=0.5, max_angle_deg=110):
    """Grasp detection: force + direction + cube-between-jaws geometric check.

    Three conditions must ALL be true:
    1. Both jaws have contact force >= min_force with cube
    2. Force direction is within max_angle of jaw y-axis
    3. Cube is geometrically between the two jaw tips (not being pushed from one side)
    """
    max_angle = max_angle_deg * np.pi / 180.0

    # ------------------------------------------------------------------
    # Contact forces (pairwise with cube)
    # ------------------------------------------------------------------
    grip_force = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :]
    jaw_force = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :]
    grip_ok = grip_force.norm(dim=-1) >= min_force
    jaw_ok = jaw_force.norm(dim=-1) >= min_force

    # ------------------------------------------------------------------
    # Grasp direction — use robot link body poses
    # ------------------------------------------------------------------
    robot = env.scene["robot"]
    grip_idx = _get_link_body_idx(robot, "gripper")
    jaw_idx = _get_link_body_idx(robot, "jaw")

    if grip_idx is not None and jaw_idx is not None:
        grip_quat = robot.data.body_quat_w[:, grip_idx, :]
        jaw_quat = robot.data.body_quat_w[:, jaw_idx, :]
        grip_y = _quat_to_y_axis(grip_quat)
        jaw_y = -_quat_to_y_axis(jaw_quat)
        g_angle_ok = _compute_angle_between(grip_y, grip_force) <= max_angle
        j_angle_ok = _compute_angle_between(jaw_y, jaw_force) <= max_angle
    else:
        ee = env.scene["ee_frame"]
        q0, q1 = ee.data.target_quat_w[:, 0, :], ee.data.target_quat_w[:, 1, :]
        g_angle_ok = _compute_angle_between(_quat_to_y_axis(q0), grip_force) <= max_angle
        j_angle_ok = _compute_angle_between(-_quat_to_y_axis(q1), jaw_force) <= max_angle

    # ------------------------------------------------------------------
    # Side-grasp check: forces must be HORIZONTAL (not vertical pushing)
    # Zero out z-component, then check anti-parallel in XY plane only
    # This rejects top/bottom contact (pushing down on cube)
    # ------------------------------------------------------------------
    grip_xy = grip_force.clone()
    grip_xy[:, 2] = 0  # zero out z
    jaw_xy = jaw_force.clone()
    jaw_xy[:, 2] = 0

    # Both forces must be predominantly horizontal (ratio, not absolute)
    grip_horiz = grip_xy.norm(dim=-1) / (grip_force.norm(dim=-1) + 1e-8) > 0.3  # >30% horizontal
    jaw_horiz = jaw_xy.norm(dim=-1) / (jaw_force.norm(dim=-1) + 1e-8) > 0.3     # >30% horizontal

    # Horizontal forces must be anti-parallel (squeezing from sides)
    grip_dir_xy = grip_xy / (grip_xy.norm(dim=-1, keepdim=True) + 1e-8)
    jaw_dir_xy = jaw_xy / (jaw_xy.norm(dim=-1, keepdim=True) + 1e-8)
    opposing = (grip_dir_xy * jaw_dir_xy).sum(dim=-1) < -0.5  # >120° apart

    return (grip_ok & g_angle_ok) & (jaw_ok & j_angle_ok) & grip_horiz & jaw_horiz & opposing


# ======================================================================
# Dopamine RPE — reward decays as hit rate increases
# ======================================================================
_hit_rates = {}
_DOPAMINE_ALPHA = 0.3   # decay exponent (moderate)
_DOPAMINE_LAMBDA = 0.3  # floor — reward never drops below 30%

def _dopamine(key, raw_reward):
    """Apply dopamine: R_eff = R_raw × [λ + (1-λ)(1-h)^α]."""
    if key not in _hit_rates:
        _hit_rates[key] = 0.0
    h_now = (raw_reward > 0.001).float().mean().item()
    _hit_rates[key] = 0.99 * _hit_rates[key] + 0.01 * h_now  # EMA
    h = _hit_rates[key]
    mult = _DOPAMINE_LAMBDA + (1 - _DOPAMINE_LAMBDA) * (1 - h) ** _DOPAMINE_ALPHA
    return raw_reward * mult


# ======================================================================
# Reward terms
# ======================================================================


# Retry tracking: detect close→open→close cycle (grasp attempt)
_was_close_to_cube = {}
_was_jaw_closed = {}
_retry_count = {}

def grasp_retry_reward(env, object_cfg=SceneEntityCfg("cube"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    """One-time reward each time agent re-opens jaw after failed grasp attempt near cube.
    Encourages: approach → close (fail) → open → approach again → close."""
    n = env.num_envs
    device = env.device
    key = "retry"
    if key not in _retry_count or _retry_count[key].shape[0] != n:
        _was_close_to_cube[key] = torch.zeros(n, dtype=torch.bool, device=device)
        _was_jaw_closed[key] = torch.zeros(n, dtype=torch.bool, device=device)
        _retry_count[key] = torch.zeros(n, dtype=torch.long, device=device)

    # Reset on new episodes
    just_reset = env.episode_length_buf <= 1
    _was_close_to_cube[key][just_reset] = False
    _was_jaw_closed[key][just_reset] = False
    _retry_count[key][just_reset] = 0

    # Current state
    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    tcp = (ee.data.target_pos_w[:, 0, :] + ee.data.target_pos_w[:, 1, :]) * 0.5
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    close = dist < 0.10
    gripper_qpos = env.scene["robot"].data.joint_pos[:, -1]
    jaw_closed = gripper_qpos < 0.3  # less than ~17°
    jaw_open = gripper_qpos > 0.5    # more than ~29°

    # Detect retry: was close+closed (attempt), now open again (retry)
    was_attempting = _was_close_to_cube[key] & _was_jaw_closed[key]
    retrying = was_attempting & jaw_open & close
    _retry_count[key][retrying] += 1

    # Update state
    _was_close_to_cube[key] = close
    _was_jaw_closed[key] = jaw_closed

    return retrying.float()


def reaching_reward(
    env,
    object_cfg=SceneEntityCfg("cube"),
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
):
    """Dense reaching reward: 1 - tanh(5 * dist(tcp, cube))."""
    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    tcp = (ee.data.target_pos_w[:, 0, :] + ee.data.target_pos_w[:, 1, :]) * 0.5
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)
    return _dopamine("reaching", 1 - torch.tanh(5 * dist))


_approach_open_fired = {}

def approach_open_reward(
    env,
    object_cfg=SceneEntityCfg("cube"),
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
):
    """One-time reward for first reaching cube with jaw open."""
    n = env.num_envs
    device = env.device
    key = "approach_open"
    if key not in _approach_open_fired or _approach_open_fired[key].shape[0] != n:
        _approach_open_fired[key] = torch.zeros(n, dtype=torch.bool, device=device)

    just_reset = env.episode_length_buf <= 1
    _approach_open_fired[key][just_reset] = False

    obj = env.scene[object_cfg.name]
    ee = env.scene[ee_frame_cfg.name]
    tcp = (ee.data.target_pos_w[:, 0, :] + ee.data.target_pos_w[:, 1, :]) * 0.5
    dist = torch.norm(obj.data.root_pos_w - tcp, dim=1)

    close = dist < 0.10
    gripper_qpos = env.scene["robot"].data.joint_pos[:, -1]
    jaw_open = gripper_qpos > 0.5

    condition = close & jaw_open & ~_approach_open_fired[key]
    _approach_open_fired[key] |= condition

    return condition.float()


def grasped_reward(
    env,
    object_cfg=SceneEntityCfg("cube"),
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
    robot_cfg=SceneEntityCfg("robot"),
):
    """Binary: 1 when grasping, 0 otherwise."""
    return _dopamine("grasped", _is_grasped(env, min_force=0.5, max_angle_deg=110).float())


def place_reward(
    env,
    object_cfg=SceneEntityCfg("cube"),
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
    robot_cfg=SceneEntityCfg("robot"),
):
    """DISABLED — replaced by fold + fold_hold."""
    return torch.zeros(env.num_envs, device=env.device)


def not_lifted_penalty(env, object_cfg=SceneEntityCfg("cube")):
    """Penalty while the cube stays on the table."""
    obj = env.scene[object_cfg.name]
    cube_z = obj.data.root_pos_w[:, 2]
    item_half_size = 0.015
    item_lifted = cube_z >= (item_half_size + 1e-3)
    return (~item_lifted).float()


def lift_hold_reward(env, object_cfg=SceneEntityCfg("cube")):
    """Reward for lifting cube 15cm + grasped (no fold requirement)."""
    obj = env.scene[object_cfg.name]
    cube_z = obj.data.root_pos_w[:, 2]
    lifted = (cube_z >= 0.15).float()
    grasped = _is_grasped(env, min_force=0.5, max_angle_deg=110).float()
    return _dopamine("lift_hold", lifted * grasped)


def _fold_3joint(env):
    """3-joint fold errors: pan(0), shoulder_lift(1), elbow(2)."""
    tgt = _get_target_qpos(env)[:, :-1]  # (N, 5)
    folded = FOLDED_QPOS[:-1].to(tgt.device)
    errs = (tgt - folded).abs()
    return errs[:, 0], errs[:, 1], errs[:, 2]  # pan, lift, elbow


def fold_reward(env, object_cfg=SceneEntityCfg("cube")):
    """3-joint continuous fold: pan + shoulder_lift + elbow. All always-on."""
    obj = env.scene[object_cfg.name]
    cube_z = obj.data.root_pos_w[:, 2]
    lifted = (cube_z >= 0.15).float()
    grasped = _is_grasped(env, min_force=0.5, max_angle_deg=110).float()
    pan_err, lift_err, elbow_err = _fold_3joint(env)

    s_pan = torch.exp(-2.0 * pan_err)        # pan (0° target)
    s_lift = torch.exp(-0.5 * lift_err)       # shoulder_lift (-90° target, big range)
    s_elbow = torch.exp(-0.5 * elbow_err)     # elbow (150° target, big range)

    return _dopamine("fold", lifted * grasped * (s_pan + s_lift + s_elbow) / 3.0)


_fh_first = [None]   # step when fold first achieved per env
_fh_held = [None]    # cumulative steps held since first fold

def fold_hold_reward(env, object_cfg=SceneEntityCfg("cube")):
    """Fold hold: pan<5° AND lift<5° AND elbow<5°, held 70%→save, 95%→big reward."""
    n = env.num_envs
    device = env.device
    if _fh_first[0] is None or _fh_first[0].shape[0] != n:
        _fh_first[0] = torch.full((n,), -1, dtype=torch.long, device=device)
        _fh_held[0] = torch.zeros(n, dtype=torch.long, device=device)

    just_reset = env.episode_length_buf <= 1
    _fh_first[0][just_reset] = -1
    _fh_held[0][just_reset] = 0

    obj = env.scene[object_cfg.name]
    cube_z = obj.data.root_pos_w[:, 2]
    lifted = (cube_z >= 0.15)
    grasped = _is_grasped(env, min_force=0.5, max_angle_deg=110)
    pan_err = _fold_3joint(env)[0]
    holding = lifted & grasped & (pan_err < 0.262)  # pan<15°

    first_time = holding & (_fh_first[0] < 0)
    _fh_first[0][first_time] = env.episode_length_buf[first_time]
    _fh_held[0][holding] += 1

    started = _fh_first[0] >= 0
    elapsed = (env.episode_length_buf - _fh_first[0]).clamp(min=1)
    ratio = (_fh_held[0].float() / elapsed.float()).clamp(max=1.0)

    above_50 = started & (ratio >= 0.50) & (_fh_held[0] >= 5)
    reward = torch.zeros(n, device=device)
    held_frac = (_fh_held[0].float() / env.max_episode_length).clamp(max=1.0)
    reward[above_50] = held_frac[above_50] ** 2
    reward[started & ~above_50] = ratio[started & ~above_50] * 1.0

    return _dopamine("fold_hold", reward)


def _touching_table(env):
    """Detect gripper/jaw touching anything other than the cube (i.e. the table)."""
    gn = env.scene["gripper_contact"].data.net_forces_w[:, 0, :]
    gc = env.scene["gripper_contact"].data.force_matrix_w[:, 0, 0, :]
    jn = env.scene["jaw_contact"].data.net_forces_w[:, 0, :]
    jc = env.scene["jaw_contact"].data.force_matrix_w[:, 0, 0, :]
    return ((gn - gc).norm(dim=-1) >= 0.01) | ((jn - jc).norm(dim=-1) >= 0.01)


def table_collision_penalty(
    env,
    table_height=0.04,
    ee_frame_cfg=SceneEntityCfg("ee_frame"),
):
    """Penalty for touching the table surface."""
    return _touching_table(env).float()


# ── Smoothness penalties ──
_prev_action = [None]
_prev_prev_action = [None]

def action_rate_penalty(env):
    """||a_t - a_{t-1}||² — penalize sudden action changes."""
    cur = env.action_manager.action
    if _prev_action[0] is None or _prev_action[0].shape != cur.shape:
        _prev_action[0] = cur.clone()
    penalty = ((cur - _prev_action[0]) ** 2).sum(dim=-1)
    _prev_action[0] = cur.clone()
    return penalty


def action_jerk_penalty(env):
    """||a_t - 2*a_{t-1} + a_{t-2}||² — penalize acceleration changes (tremor)."""
    cur = env.action_manager.action
    if _prev_action[0] is None or _prev_action[0].shape != cur.shape:
        _prev_action[0] = cur.clone()
    if _prev_prev_action[0] is None or _prev_prev_action[0].shape != cur.shape:
        _prev_prev_action[0] = cur.clone()
    jerk = cur - 2 * _prev_action[0] + _prev_prev_action[0]
    penalty = (jerk ** 2).sum(dim=-1)
    _prev_prev_action[0] = _prev_action[0].clone()
    _prev_action[0] = cur.clone()
    return penalty
