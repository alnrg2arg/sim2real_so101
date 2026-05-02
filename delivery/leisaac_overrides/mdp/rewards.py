"""Reward functions for lift cube task.

Reward stages (촘촘한 단계):
  1. reaching_coarse  - ee가 물체 20cm 이내 접근
  2. reaching_fine    - ee가 물체 5cm 이내 정밀 접근
  3. gripper_open_near - 물체 근처에서 그리퍼 벌리기
  4. gripper_closing   - 물체 가까이서 그리퍼 닫기 (연속)
  5. grasp_verified    - 실제 잡기 성공 (물체가 초기높이보다 올라감)
  6. lift_low          - 약간 들기 (초기+3cm 이상)
  7. lift_high         - 많이 들기 (초기+10cm 이상)
  8. hold_stable       - 잡은 채 유지 (높이 유지)
  9. drop_penalty      - 떨어뜨리기 벌점
"""
from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

# Initial cube height on table (measured: ~0.056m)
CUBE_INITIAL_HEIGHT = 0.056


def _get_ee_obj_dist(env, object_cfg, ee_frame_cfg):
    """Helper: compute ee-to-object distance."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj_pos = obj.data.root_pos_w
    ee_pos = ee.data.target_pos_w[..., 1, :]  # jaw (inner gripper tip)
    return torch.norm(obj_pos - ee_pos, dim=1), obj_pos, ee_pos


# ── Stage 1: Coarse reaching (far away → within 20cm) ──
def reaching_coarse(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)
    return 1 - torch.tanh(dist / std)


# ── Stage 2: Fine reaching (within 5cm, tight kernel) ──
def reaching_fine(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)
    return 1 - torch.tanh(dist / std)


# ── Stage 2.5: Align gripper with object orientation ──
def gripper_align_object(
    env: ManagerBasedRLEnv,
    near_dist: float = 0.15,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Alignment reward: yaw match + pitch pointing downward.
    
    Gripper must:
    1. Match object yaw (rotation on table) - cube is symmetric so abs(cos) ok
    2. Point downward (pitch close to -90deg) to grasp from above
    """
    from isaaclab.utils.math import euler_xyz_from_quat
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)

    # Yaw alignment (cube is symmetric, so abs is fine)
    obj_quat = obj.data.root_quat_w
    _, _, obj_yaw = euler_xyz_from_quat(obj_quat)
    ee_quat = ee.data.target_quat_w[:, 0, :]
    ee_roll, ee_pitch, ee_yaw = euler_xyz_from_quat(ee_quat)
    yaw_diff = obj_yaw - ee_yaw
    yaw_score = torch.cos(yaw_diff).abs()  # 0~1, symmetric for cube

    # Pitch alignment: gripper should point downward (pitch ~ -pi/2)
    # cos(pitch - (-pi/2)) = cos(pitch + pi/2) = -sin(pitch)
    # When pitch = -pi/2 (pointing down): -sin(-pi/2) = 1.0 (perfect)
    # When pitch = 0 (horizontal): -sin(0) = 0.0 (bad)
    pitch_score = torch.clamp(-torch.sin(ee_pitch), min=0.0, max=1.0)

    # Combined: both yaw and pitch must be good
    alignment = yaw_score * 0.4 + pitch_score * 0.6  # pitch is more important

    near = dist < near_dist
    return torch.where(near, alignment, torch.zeros_like(alignment))


# ── Stage 3: Open gripper when approaching object ──
def gripper_open_near_object(
    env: ManagerBasedRLEnv,
    near_dist: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -1]
    near = dist < near_dist
    gripper_open = gripper_pos > 0.4
    return torch.where(torch.logical_and(near, gripper_open), 1.0, 0.0)


# ── Stage 4: Close gripper when very close (continuous) ──
def gripper_closing_near(
    env: ManagerBasedRLEnv,
    near_dist: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -1]
    very_near = dist < near_dist
    close_reward = torch.clamp(1.0 - gripper_pos * 2.0, min=0.0, max=1.0)
    return torch.where(very_near, close_reward, torch.zeros_like(close_reward))


# ── Stage 5: Verified grasp (object LIFTED above initial height) ──
def grasp_contact(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    dist_threshold: float = 0.04,
    grasp_threshold: float = 0.26,
    lift_threshold: float = 0.005,  # Must be lifted 0.5cm above initial
) -> torch.Tensor:
    """Grasp verified by: ee close + gripper closed + object lifted above initial height."""
    obj: RigidObject = env.scene[object_cfg.name]
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -1]

    close_enough = dist < dist_threshold
    gripper_closed = gripper_pos < grasp_threshold
    # Object must be above its initial resting height
    obj_lifted = obj.data.root_pos_w[:, 2] > (CUBE_INITIAL_HEIGHT + lift_threshold)

    grasped = close_enough & gripper_closed & obj_lifted
    return torch.where(grasped, 1.0, 0.0)


# ── Stage 6: Low lift (>3cm above initial) ──
def lift_low(
    env: ManagerBasedRLEnv,
    min_height: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    grasp_threshold: float = 0.26,
) -> torch.Tensor:
    """Reward for lifting object while grasping."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)

    gripper_closed = robot.data.joint_pos[:, -1] < grasp_threshold
    close_enough = dist < 0.05
    # Must be lifted above initial height + min_height
    obj_above = obj.data.root_pos_w[:, 2] > (CUBE_INITIAL_HEIGHT + min_height)

    lifted = gripper_closed & close_enough & obj_above
    return torch.where(lifted, 1.0, 0.0)


# ── Stage 7: High lift (continuous height reward, only counts above initial) ──
def lift_high(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_base_name: str = "base",
) -> torch.Tensor:
    """Continuous reward proportional to how high above INITIAL position."""
    obj: RigidObject = env.scene[object_cfg.name]
    cube_h = obj.data.root_pos_w[:, 2]
    # Only reward height above initial resting position
    height_gain = cube_h - CUBE_INITIAL_HEIGHT
    return torch.clamp(height_gain, min=0.0, max=0.3)


# ── Stage 8: Hold stable (grasped + lifted + maintaining) ──
def hold_stable(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.10,  # 10cm above ground (well above table)
    grasp_threshold: float = 0.26,
) -> torch.Tensor:
    """Reward for maintaining grasp while object is clearly lifted."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)

    # Must be well above initial height (not just sitting on table)
    cube_above = obj.data.root_pos_w[:, 2] > height_threshold
    gripper_closed = robot.data.joint_pos[:, -1] < grasp_threshold
    close_enough = dist < 0.05

    holding = cube_above & gripper_closed & close_enough
    return torch.where(holding, 1.0, 0.0)


# ── Stage 9: Drop penalty ──
def object_dropped_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    height_threshold: float = 0.08,
    dist_threshold: float = 0.10,
) -> torch.Tensor:
    """Penalty if object was lifted but is now falling."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_pos = obj.data.root_pos_w
    ee_pos = ee.data.target_pos_w[..., 1, :]  # jaw (inner gripper tip)

    cube_above = obj_pos[:, 2] > height_threshold
    ee_dist = torch.norm(obj_pos - ee_pos, dim=1)
    ee_far = ee_dist > dist_threshold
    obj_vel_z = obj.data.root_lin_vel_w[:, 2]
    falling = obj_vel_z < -0.1

    dropped = cube_above & ee_far & falling
    return torch.where(dropped, 1.0, 0.0)


# ── Stage 6 (NEW): Lift to 20cm (continuous) ──
def lift_low(
    env: ManagerBasedRLEnv,
    min_height: float = 0.05,
    max_height: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    grasp_threshold: float = 0.26,
) -> torch.Tensor:
    """Continuous reward for lifting 5~10cm while grasping."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)
    gripper_closed = robot.data.joint_pos[:, -1] < grasp_threshold
    close_enough = dist < 0.05
    height_gain = obj.data.root_pos_w[:, 2] - CUBE_INITIAL_HEIGHT
    normalized = torch.clamp((height_gain - min_height) / (max_height - min_height), min=0.0, max=1.0)
    grasping = gripper_closed & close_enough & (height_gain > min_height)
    return torch.where(grasping, normalized, torch.zeros_like(normalized))


def lift_high(
    env: ManagerBasedRLEnv,
    min_height: float = 0.10,
    max_height: float = 0.25,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    grasp_threshold: float = 0.26,
) -> torch.Tensor:
    """Continuous reward for lifting 10~25cm while grasping."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)
    gripper_closed = robot.data.joint_pos[:, -1] < grasp_threshold
    close_enough = dist < 0.05
    height_gain = obj.data.root_pos_w[:, 2] - CUBE_INITIAL_HEIGHT
    normalized = torch.clamp((height_gain - min_height) / (max_height - min_height), min=0.0, max=1.0)
    grasping = gripper_closed & close_enough & (height_gain > min_height)
    return torch.where(grasping, normalized, torch.zeros_like(normalized))



# ── Velocity-based grasp detection ──
def grasp_velocity_match(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    grasp_threshold: float = 0.26,
    dist_threshold: float = 0.06,
) -> torch.Tensor:
    """Reward when object moves with gripper = actually grasped.
    Checks: gripper closed + ee close to object + object above table."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)

    gripper_closed = robot.data.joint_pos[:, -1] < grasp_threshold
    close_enough = dist < dist_threshold

    # Object must have moved up from initial position (not just sitting on table)
    obj_above = obj.data.root_pos_w[:, 2] > (CUBE_INITIAL_HEIGHT + 0.003)  # 3mm above rest

    # Object velocity should be non-zero if being moved by gripper
    obj_vel = torch.norm(obj.data.root_lin_vel_w, dim=1)
    obj_moving = obj_vel > 0.01  # moving at all

    # Either object is above rest OR object is moving while gripper holds it
    grasped = gripper_closed & close_enough & (obj_above | obj_moving)
    return torch.where(grasped, 1.0, 0.0)


# ── Penalty: object too far from EE ──
def object_out_of_reach(
    env: ManagerBasedRLEnv,
    max_dist: float = 0.50,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty when cube is too far from EE."""
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)
    return torch.where(dist > max_dist, 1.0, 0.0)


# ── Penalty: gripper collides with table ──
def gripper_table_collision(
    env: ManagerBasedRLEnv,
    table_height: float = 0.05,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty when gripper goes below table surface."""
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = ee.data.target_pos_w[..., 1, 2]  # jaw z
    return torch.where(ee_z < table_height, 1.0, 0.0)


# ── Penalty: joint velocity too fast ──
def joint_velocity_excess(
    env: ManagerBasedRLEnv,
    threshold: float = 2.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for joint velocities exceeding threshold (rad/s)."""
    robot: Articulation = env.scene[robot_cfg.name]
    vel = robot.data.joint_vel[:, :-1]  # exclude gripper
    excess = torch.clamp(vel.abs() - threshold, min=0.0)
    return excess.sum(dim=1)


# ── Penalty: gripper outside ±20cm corridor around object ──
def gripper_lateral_deviation(
    env: ManagerBasedRLEnv,
    corridor_half_width: float = 0.20,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty when gripper exits a ±20cm corridor around the object.

    Draws parallel lines at cube_x ± corridor and cube_y ± corridor.
    Penalizes proportional to how far the gripper is outside the box.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_xy = ee.data.target_pos_w[..., 1, :2]  # jaw XY
    obj_xy = obj.data.root_pos_w[:, :2]
    # Per-axis deviation beyond corridor
    dx = torch.clamp((ee_xy[:, 0] - obj_xy[:, 0]).abs() - corridor_half_width, min=0.0)
    dy = torch.clamp((ee_xy[:, 1] - obj_xy[:, 1]).abs() - corridor_half_width, min=0.0)
    return dx + dy
