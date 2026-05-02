"""Close reward: 18 stages. 1-9: approach close, 10-18: fine close 0.35→0.26."""
import torch
from ..config import GRIPPER_OPEN, GRIPPER_CLOSED
from ..helpers import _milestone_gate, _milestones, _object_between_jaws


def close_stages_10(env, object_cfg=None, ee_frame_cfg=None, robot_cfg=None):
    """Close in 18 stages.
    Stages 1-9: gripper 1.05→0.34 (coarse close, between_jaws)
    Stages 10-18: gripper 0.34→0.26 (fine close, between_jaws + near)
    Gate: align_s20"""
    from isaaclab.assets import Articulation
    from leisaac.tasks.lift_cube.mdp.rewards import _get_ee_obj_dist
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -1]
    num_envs = gripper_pos.shape[0]
    device = gripper_pos.device

    dist, _, _ = _get_ee_obj_dist(env, object_cfg, ee_frame_cfg)
    near_object = dist < 0.05
    between = _object_between_jaws(env, object_cfg, ee_frame_cfg, robot_cfg)
    if between is None:
        between = torch.ones(num_envs, device=device, dtype=torch.bool)

    was_aligned = _milestones.get("align_s20",
        torch.zeros(num_envs, device=device, dtype=torch.bool))

    reward = torch.zeros_like(gripper_pos)

    # Stages 1-9: coarse close 1.05 → 0.34
    for i in range(1, 10):
        threshold = GRIPPER_OPEN - (GRIPPER_OPEN - 0.34) * (i / 9.0)
        cond = was_aligned & near_object & between & (gripper_pos < threshold)
        first = _milestone_gate(f"close_{i:02d}", env, cond)
        reward = reward + first.float() * (i / 18.0)

    # Stages 10-18: fine close 0.34 → 0.26
    for i in range(10, 19):
        threshold = 0.34 - (0.34 - GRIPPER_CLOSED) * ((i - 9) / 9.0)
        cond = was_aligned & near_object & between & (gripper_pos < threshold)
        first = _milestone_gate(f"close_{i:02d}", env, cond)
        reward = reward + first.float() * (i / 18.0)

    return reward
