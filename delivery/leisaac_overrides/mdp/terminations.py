from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def cube_height_above_base(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    robot_base_name: str = "base",
    height_threshold: float = 0.20,
) -> torch.Tensor:
    """Determine if the cube is above the robot base.

    This function checks whether all success conditions for the task have been met:
    1. cube is above the robot base

    Args:
        env: The RL environment instance.
        cube_cfg: Configuration for the cube entity.
        robot_cfg: Configuration for the robot entity.
        robot_base_name: Name of the robot base.
        height_threshold: Threshold for the cube height above the robot base.
    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_height = cube.data.root_pos_w[:, 2]
    base_index = robot.data.body_names.index(robot_base_name)
    robot_base_height = robot.data.body_pos_w[:, base_index, 2]
    above_base = cube_height - robot_base_height > height_threshold
    done = torch.logical_and(done, above_base)

    return done


def object_out_of_reach_terminate(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    max_dist: float = 0.50,
) -> torch.Tensor:
    """Terminate if cube is too far from EE (lenient, 50cm)."""
    from isaaclab.sensors import FrameTransformer
    cube: RigidObject = env.scene[cube_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    dist = torch.norm(cube.data.root_pos_w - ee.data.target_pos_w[..., 1, :], dim=1)
    return dist > max_dist


def object_dropped_terminate(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    height_threshold: float = 0.08,
    dist_threshold: float = 0.10,
) -> torch.Tensor:
    """Terminate if object was lifted but is now falling (dropped)."""
    cube: RigidObject = env.scene[cube_cfg.name]
    from isaaclab.sensors import FrameTransformer
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_above = cube.data.root_pos_w[:, 2] > height_threshold
    ee_dist = torch.norm(cube.data.root_pos_w - ee.data.target_pos_w[..., 1, :], dim=1)
    ee_far = ee_dist > dist_threshold
    falling = cube.data.root_lin_vel_w[:, 2] < -0.1
    return cube_above & ee_far & falling


def gripper_table_collision_terminate(
    env: ManagerBasedRLEnv | DirectRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    table_height: float = 0.05,
) -> torch.Tensor:
    """Terminate if gripper goes below table surface."""
    from isaaclab.sensors import FrameTransformer
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee.data.target_pos_w[..., 1, 2] < table_height
