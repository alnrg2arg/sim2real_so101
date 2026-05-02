"""Gripper open reward: 10 stages while approaching."""
import torch
from ..config import GRIPPER_OPEN
from ..helpers import _milestone_gate


def gripper_open_stages_10(env, object_cfg=None, ee_frame_cfg=None, robot_cfg=None):
    """Gripper open in 10 stages: 0->1.05, with distance conditions."""
    from isaaclab.assets import Articulation
    from leisaac.tasks.lift_cube.mdp.rewards import _get_ee_obj_dist
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -1]
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)

    reward = torch.zeros_like(gripper_pos)

    # Stages 1-5: gripper opening while within 20cm
    for i in range(1, 6):
        grip_thresh = 0.21 * i
        cond = (gripper_pos > grip_thresh) & (dist < 0.20)
        first = _milestone_gate(f"open_{i:02d}", env, cond)
        reward = reward + first.float() * (i / 10.0)

    # Stages 6-10: fully open + getting closer (15cm->10cm)
    for i in range(6, 11):
        dist_thresh = 0.15 - (0.15 - 0.10) * ((i - 5) / 5.0)
        cond = (gripper_pos > GRIPPER_OPEN) & (dist < dist_thresh)
        first = _milestone_gate(f"open_{i:02d}", env, cond)
        reward = reward + first.float() * (i / 10.0)

    return reward
