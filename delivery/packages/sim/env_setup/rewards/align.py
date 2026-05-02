"""Alignment rewards: distance + between_jaws progression."""
import torch
from ..config import GRIPPER_OPEN
from ..helpers import _milestone_gate, _object_between_jaws


def align_stages_30(env, object_cfg=None, ee_frame_cfg=None, robot_cfg=None,
                    jaw_dist_threshold=0.015):
    """Align in 30 stages with progressive between_jaws.
    Stages 1-10: dist 15cm->10cm, gripper open (approach)
    Stages 11-20: dist 10cm->5cm, gripper open + between_jaws (align)
    Stages 21-30: dist < 5cm + between_jaws + jaw precision"""
    from leisaac.tasks.lift_cube.mdp.rewards import _get_ee_obj_dist
    robot = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -1]
    is_open = gripper_pos > GRIPPER_OPEN
    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)

    between = _object_between_jaws(env, object_cfg, ee_frame_cfg, robot_cfg)
    if between is None:
        between = torch.ones_like(gripper_pos, dtype=torch.bool)

    reward = torch.zeros_like(gripper_pos)

    # Stages 1-10: distance 15cm -> 10cm (approach, no between required)
    for i in range(1, 11):
        threshold = 0.15 - (0.15 - 0.10) * (i / 10.0)
        cond = is_open & (dist < threshold)
        first = _milestone_gate(f"align_s{i:02d}", env, cond)
        reward = reward + first.float() * (i / 30.0)

    # Stages 11-20: distance 10cm -> 5cm (must be between jaws)
    for i in range(11, 21):
        threshold = 0.10 - (0.10 - 0.05) * ((i - 10) / 10.0)
        cond = is_open & (dist < threshold) & between
        first = _milestone_gate(f"align_s{i:02d}", env, cond)
        reward = reward + first.float() * (i / 30.0)

    # Stages 21-30: dist < 5cm + between jaws + tightening precision
    for i in range(21, 31):
        jaw_max = jaw_dist_threshold + (0.06 - jaw_dist_threshold) * (30 - i) / 9.0
        between_tight = _object_between_jaws(env, object_cfg, ee_frame_cfg, robot_cfg, max_dist=jaw_max)
        cond = is_open & (dist < 0.05) & between_tight
        first = _milestone_gate(f"align_s{i:02d}", env, cond)
        reward = reward + first.float() * (i / 30.0)

    return reward
