"""Custom action: delta accumulated on previous target (ManiSkill use_target=True)."""
import torch
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass


class TargetDeltaJointPositionAction(JointAction):
    """Delta added to PREVIOUS TARGET, not current qpos.
    
    ManiSkill pd_joint_target_delta_pos equivalent:
        target = previous_target + clip(action, -1, 1) * scale
    """
    
    cfg: "TargetDeltaJointPositionActionCfg"
    
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.use_zero_offset:
            self._offset = 0.0
        self._target_qpos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
    
    def reset(self, env_ids):
        self._target_qpos[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
    
    def apply_actions(self):
        # ManiSkill exact: clamp RAW action to [-1,1] THEN scale
        # self.raw_actions = network output (before scale)
        # self.processed_actions = offset + scale * raw_actions (already scaled)
        # We need: clamp(raw, -1, 1) * scale = clamp(raw, -1, 1) * self._scale
        raw_clamped = self.raw_actions.clamp(-1.0, 1.0)
        delta = raw_clamped * self._scale
        # Accumulate on previous target
        self._target_qpos = self._target_qpos + delta
        # Set as position target (PD controller tracks this)
        self._asset.set_joint_position_target(self._target_qpos, joint_ids=self._joint_ids)


@configclass
class TargetDeltaJointPositionActionCfg(JointActionCfg):
    """Config for TargetDeltaJointPositionAction."""
    class_type = TargetDeltaJointPositionAction
    use_zero_offset: bool = True
