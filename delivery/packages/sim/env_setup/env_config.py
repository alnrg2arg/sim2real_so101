"""Environment configuration — Squint-exact for SO101 LiftCube.

Matches Squint (https://github.com/aalmuzairee/squint) exactly:
  - Delta joint position control: arm 0.1 rad, gripper 0.2 rad
  - Observations: qpos, qvel, target_qpos, dist_to_rest, obj_pose, tcp_pos, tcp_to_obj, is_grasped
  - Rewards: reaching + grasped + place*grasped - 3*table - 1*not_lifted, /3
  - Episode: 5.0s (50 steps @ 10Hz)
  - rest_qpos = SO101 start keyframe [0,0,0,pi/2,-pi/2,60deg]
"""

import torch
import numpy as np


# ── Squint SO101 rest_qpos (start keyframe) ──
_REST_QPOS = torch.tensor([0.0, 0.0, 0.0, np.pi / 2, -np.pi / 2, 60.0 * np.pi / 180.0])


# ── Custom observation functions ──

def tcp_pos_w(env, ee_frame_cfg=None):
    """TCP position in world frame. Shape: (N, 3)."""
    from isaaclab.managers import SceneEntityCfg
    cfg = ee_frame_cfg or SceneEntityCfg("ee_frame")
    ee = env.scene[cfg.name]
    jaw0 = ee.data.target_pos_w[:, 0, :]
    jaw1 = ee.data.target_pos_w[:, 1, :]
    return (jaw0 + jaw1) * 0.5


def tcp_to_obj_pos(env, ee_frame_cfg=None, object_cfg=None):
    """Relative position from TCP to object. Shape: (N, 3)."""
    from isaaclab.managers import SceneEntityCfg
    ee_cfg = ee_frame_cfg or SceneEntityCfg("ee_frame")
    obj_cfg = object_cfg or SceneEntityCfg("cube")
    ee = env.scene[ee_cfg.name]
    obj = env.scene[obj_cfg.name]
    tcp = (ee.data.target_pos_w[:, 0, :] + ee.data.target_pos_w[:, 1, :]) * 0.5
    return obj.data.root_pos_w - tcp


def is_grasped_obs(env, robot_cfg=None, object_cfg=None, ee_frame_cfg=None):
    """Binary grasp signal via force_matrix_w. Shape: (N, 1)."""
    from packages.sim.env_setup.maniskill_rewards import _is_grasped
    return _is_grasped(env, min_force=0.5, max_angle_deg=110).float().unsqueeze(-1)


def joint_vel_obs(env, robot_cfg=None):
    """Joint velocities. Shape: (N, 6). Squint includes qvel in obs."""
    from isaaclab.managers import SceneEntityCfg
    cfg = robot_cfg or SceneEntityCfg("robot")
    return env.scene[cfg.name].data.joint_vel


def configure_env(env_cfg, cfg, mdp, RewTerm, DoneTerm, SceneEntityCfg):
    """Configure env to match Squint SO101LiftCube-v1 exactly."""
    from packages.sim.env_setup.maniskill_rewards import (
        reaching_reward, approach_open_reward, grasp_retry_reward, grasped_reward,
        table_collision_penalty, not_lifted_penalty, lift_hold_reward, fold_reward, fold_hold_reward,
    )

    # ══════════════════════════════════════════
    #  EPISODE & SIMULATION (Squint-exact)
    # ══════════════════════════════════════════
    # Squint: max_episode_steps=50, sim_freq=100, control_freq=10 => 5.0s
    env_cfg.episode_length_s = 10.0  # 100 steps at 10Hz
    env_cfg.sim.dt = 0.01           # 100Hz sim
    env_cfg.decimation = 10         # 100Hz / 10 = 10Hz control
    env_cfg.sim.physx.solver_position_iteration_count = 15
    env_cfg.sim.physx.solver_velocity_iteration_count = 1
    env_cfg.sim.physx.contact_offset = 0.02
    env_cfg.sim.physx.rest_offset = 0.0
    env_cfg.sim.physx.solve_articulation_contact_last = True  # gripper penetration fix
    env_cfg.sim.render_interval = cfg.get("sim_render_interval", 2)
    env_cfg.recorders = None
    env_cfg.scene.robot.spawn.activate_contact_sensors = True

    # ══════════════════════════════════════════
    #  ee_frame offsets — Squint URDF exact
    # ══════════════════════════════════════════
    from isaaclab.sensors import OffsetCfg
    # Jaw INNER contact surface (from STL mesh analysis):
    # Fixed jaw inner face: y shifted 10mm inward from Squint tip
    # Moving jaw inner face: y shifted ~10mm toward fixed jaw
    # This prevents false reaching reward from jaw closing
    env_cfg.scene.ee_frame.target_frames[0].offset = OffsetCfg(
        pos=(0.000, -0.010, -0.092))   # fixed jaw inner contact surface
    env_cfg.scene.ee_frame.target_frames[1].offset = OffsetCfg(
        pos=(-0.01, -0.067, 0.02))     # moving jaw inner contact surface

    # ══════════════════════════════════════════
    #  Camera — wrist only (Squint-exact, 142 v28 config)
    # ══════════════════════════════════════════
    from isaaclab.sensors import TiledCameraCfg
    import isaaclab.sim as sim_utils_cam

    # Disable front/side to save VRAM
    if hasattr(env_cfg.scene, 'front'):
        env_cfg.scene.front = None
    if hasattr(env_cfg.scene, 'side'):
        env_cfg.scene.side = None

    env_cfg.scene.wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/fp_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.05, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
        data_types=["rgb"], spawn=sim_utils_cam.PinholeCameraCfg(focal_length=2.5, focus_distance=200.0, horizontal_aperture=24.0, vertical_aperture=18.0, clipping_range=(0.003, 50.0), lock_camera=True),
        width=128, height=128, update_period=0)  # 128x128 for learning, downsampled to 16x16

    # ══════════════════════════════════════════
    #  CUBE SPAWN (Squint: ±10cm random)
    # ══════════════════════════════════════════
    try:
        for ev_cfg in getattr(env_cfg.events, '__dict__', {}).values():
            if hasattr(ev_cfg, 'params') and 'pose_range' in getattr(ev_cfg, 'params', {}):
                ev_cfg.params["pose_range"]["x"] = (-0.15, 0.15)
                ev_cfg.params["pose_range"]["y"] = (-0.15, 0.15)
                ev_cfg.params["pose_range"]["yaw"] = (-3.14159, 3.14159)
    except Exception:
        pass

    # Always disable domain_randomize_1 (references front camera which we removed)
    if hasattr(env_cfg, 'events') and hasattr(env_cfg.events, 'domain_randomize_1'):
        env_cfg.events.domain_randomize_1 = None

    if hasattr(env_cfg.observations, 'subtask_terms'):
        env_cfg.observations.subtask_terms = None
    env_cfg.observations.policy.concatenate_terms = True
    env_cfg.observations.policy.concatenate_dim = -1

    # ══════════════════════════════════════════
    #  OBSERVATIONS (Squint state-mode exact)
    # ══════════════════════════════════════════
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.envs.mdp import joint_pos, root_pos_w, root_quat_w

    # Clear ALL existing obs terms
    for attr in list(vars(env_cfg.observations.policy)):
        if not attr.startswith('_') and attr not in ('concatenate_terms', 'concatenate_dim', 'enable_corruption', 'corruption_cfg'):
            setattr(env_cfg.observations.policy, attr, None)

    # 1. Joint positions (6D)
    env_cfg.observations.policy.qpos = ObsTerm(
        func=joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 2. Joint velocities (6D) — Squint includes qvel
    env_cfg.observations.policy.qvel = ObsTerm(
        func=joint_vel_obs,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # 3. Controller target_qpos (6D)
    def _target_qpos(env, robot_cfg=None):
        import torch as _t
        am = env.action_manager
        targets = []
        for term in am._terms.values():
            if hasattr(term, '_target_qpos'):
                targets.append(term._target_qpos)
        if targets:
            return _t.cat(targets, dim=-1)
        return env.scene["robot"].data.joint_pos
    env_cfg.observations.policy.target_qpos = ObsTerm(
        func=_target_qpos,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # 4. dist_to_rest_qpos (5D, exclude gripper) — Squint exact
    def _dist_to_rest_qpos(env, robot_cfg=None):
        import torch as _t
        am = env.action_manager
        targets = []
        for term in am._terms.values():
            if hasattr(term, '_target_qpos'):
                targets.append(term._target_qpos)
        if targets:
            tgt = _t.cat(targets, dim=-1)[:, :-1]
        else:
            tgt = env.scene["robot"].data.joint_pos[:, :-1]
        rest = _REST_QPOS[:-1].to(tgt.device)
        return tgt - rest
    env_cfg.observations.policy.dist_to_rest = ObsTerm(
        func=_dist_to_rest_qpos,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    # 5. Object position (3D)
    env_cfg.observations.policy.obj_pos = ObsTerm(
        func=root_pos_w,
        params={"asset_cfg": SceneEntityCfg("cube")},
    )
    # 6. Object orientation (4D)
    env_cfg.observations.policy.obj_quat = ObsTerm(
        func=root_quat_w,
        params={"asset_cfg": SceneEntityCfg("cube")},
    )
    # 7. TCP position (3D)
    env_cfg.observations.policy.tcp_pos = ObsTerm(
        func=tcp_pos_w,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )
    # 8. TCP-to-object (3D)
    env_cfg.observations.policy.tcp_to_obj = ObsTerm(
        func=tcp_to_obj_pos,
        params={"ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube")},
    )
    # 9. Is grasped (1D)
    env_cfg.observations.policy.is_grasped = ObsTerm(
        func=is_grasped_obs,
        params={"robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("cube"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    # ══════════════════════════════════════════
    #  ACTIONS — Squint-exact delta scale
    # ══════════════════════════════════════════
    from packages.sim.env_setup.target_delta_action import TargetDeltaJointPositionActionCfg
    # Squint SO101: arm [-0.1, 0.1], gripper [-0.2, 0.2]
    env_cfg.actions.arm_action = TargetDeltaJointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        scale=0.1,  # Squint: 0.1 rad (was 0.05)
    )
    env_cfg.actions.gripper_action = TargetDeltaJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=0.2,  # Squint: 0.2 rad
    )

    # ══════════════════════════════════════════
    #  TERMINATIONS — time_out only
    # ══════════════════════════════════════════
    if hasattr(env_cfg, 'terminations'):
        for attr in list(vars(env_cfg.terminations)):
            if not attr.startswith('_') and attr != 'time_out':
                setattr(env_cfg.terminations, attr, None)

    # ══════════════════════════════════════════
    #  REWARDS — Squint-exact (5 terms)
    #  Isaac Lab multiplies reward * weight * dt internally.
    #  dt = decimation * sim_dt = 10 * 0.01 = 0.1
    #  Squint does NOT multiply by dt. To match: weight_isaaclab = weight_squint / dt
    #  Squint normalized reward = raw / 3, so weight_squint = 1/3
    #  => weight_isaaclab = (1/3) / 0.1 = 10/3
    # ══════════════════════════════════════════
    _DT_COMP = 10.0  # 1 / dt = 1 / 0.1

    # 1. Reaching: 1 - tanh(5 * dist)
    env_cfg.rewards.reaching = RewTerm(
        func=reaching_reward,
        params={"object_cfg": SceneEntityCfg("cube"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=1.0 / 3 * _DT_COMP)

    # 2. Grasped: binary
    env_cfg.rewards.grasped = RewTerm(
        func=grasped_reward,
        params={"object_cfg": SceneEntityCfg("cube"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_cfg": SceneEntityCfg("robot")},
        weight=1.0 / 3 * _DT_COMP)


    # 4. Table collision: Squint raw = -3, normalized = -3/3 = -1
    env_cfg.rewards.table_penalty = RewTerm(
        func=table_collision_penalty,
        params={"table_height": 0.04,
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=-3.0 / 3 * _DT_COMP)

    # 5. Not-lifted: Squint raw = -1, normalized = -1/3
    env_cfg.rewards.not_lifted = RewTerm(
        func=not_lifted_penalty,
        params={"object_cfg": SceneEntityCfg("cube")},
        weight=-1.0 / 3 * _DT_COMP)

    # 6. Lift: reward for lifting 15cm + grasped
    env_cfg.rewards.lift_hold = RewTerm(
        func=lift_hold_reward,
        params={"object_cfg": SceneEntityCfg("cube")},
        weight=1.0 / 3 * _DT_COMP)

    # 6b. Fold: reward for moving toward folded pose while lifted + grasped
    env_cfg.rewards.fold = RewTerm(
        func=fold_reward,
        params={"object_cfg": SceneEntityCfg("cube")},
        weight=1.0 / 3 * _DT_COMP)

    # 6c. Fold Hold: reward for MAINTAINING folded pose (every step = longer hold = more reward)
    env_cfg.rewards.fold_hold = RewTerm(
        func=fold_hold_reward,
        params={"object_cfg": SceneEntityCfg("cube")},
        weight=1.0 / 3 * _DT_COMP)

    # 7. Approach with open jaw — bridge between reaching and grasping (~30% of reaching)
    env_cfg.rewards.approach_open = RewTerm(
        func=approach_open_reward,
        params={"object_cfg": SceneEntityCfg("cube"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=0.1 * _DT_COMP)  # 1.0 — about 30% of reaching(3.33)

    # 8. Grasp retry bonus (one-time per retry attempt)
    env_cfg.rewards.grasp_retry = RewTerm(
        func=grasp_retry_reward,
        params={"object_cfg": SceneEntityCfg("cube"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=0.03 * _DT_COMP)  # 0.3

    # Remove all other default rewards
    keep = {"reaching", "approach_open", "grasped", "table_penalty", "not_lifted", "lift_hold", "fold", "fold_hold", "grasp_retry"}
    for name in list(vars(env_cfg.rewards)):
        if not name.startswith("_") and name not in keep:
            setattr(env_cfg.rewards, name, None)

    return env_cfg


def apply_motor_limits(env, cfg):
    """Set PD gains — STS3215 real servo matching."""
    try:
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        count = 0
        for prim in stage.Traverse():
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                drive.GetStiffnessAttr().Set(20.0)    # STS3215: ~20 Nm/rad
                drive.GetDampingAttr().Set(2.0)        # STS3215: ~2 Nm·s/rad
                drive.GetMaxForceAttr().Set(3.5)       # STS3215: 3.5 Nm stall torque
                count += 1
        try:
            print("[Motor] velocity: unlimited", flush=True)
        except Exception as ve:
            print(f"[Motor] velocity limit failed: {ve}", flush=True)
        if count > 0:
            print(f"[Motor] PD: stiffness=20 damping=2 force=3.5Nm STS3215-real ({count} joints)", flush=True)
    except Exception as e:
        print(f"[Motor] Error: {e}", flush=True)

    # Gripper friction = 2.0 (Squint-exact: SO101.py urdf_config)
    try:
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        grip_count = 0
        for prim in stage.Traverse():
            name = prim.GetName().lower()
            if "gripper" in name or "jaw" in name or "finger" in name:
                mat = UsdPhysics.MaterialAPI.Get(stage, prim.GetPath())
                if not mat:
                    mat = UsdPhysics.MaterialAPI.Apply(prim)
                if mat:
                    mat.GetStaticFrictionAttr().Set(2.0)
                    mat.GetDynamicFrictionAttr().Set(2.0)
                    mat.GetRestitutionAttr().Set(0.0)
                    grip_count += 1
        print(f"[Gripper] friction=2.0 on {grip_count} prims (Squint-exact)", flush=True)
    except Exception as e:
        print(f"[Gripper] Friction error: {e}", flush=True)

    # Robot link mass readout
    try:
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        total_robot_mass = 0.0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "/env_0/" not in path or "cube" in prim.GetName().lower():
                continue
            mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
            if mass_api:
                m = mass_api.GetMassAttr().Get()
                if m and m > 0:
                    total_robot_mass += m
                    print(f"[Robot] {prim.GetName():25s} mass={m:.4f} kg", flush=True)
        print(f"[Robot] Total robot mass (env_0): {total_robot_mass:.4f} kg", flush=True)
    except Exception as e:
        print(f"[Robot] Mass readout error: {e}", flush=True)

    # Cube physics: set mass=50g + friction
    try:
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        cube_count = 0
        for prim in stage.Traverse():
            if "cube" in prim.GetName().lower():
                mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
                if not mass_api:
                    mass_api = UsdPhysics.MassAPI.Apply(prim)
                if mass_api:
                    mass_api.GetMassAttr().Set(0.05)  # 50g
                    cube_count += 1
                mat = UsdPhysics.MaterialAPI.Get(stage, prim.GetPath())
                if mat:
                    mat.GetStaticFrictionAttr().Set(0.3)
                    mat.GetDynamicFrictionAttr().Set(0.3)
                    mat.GetRestitutionAttr().Set(0.0)
        # Verify
        for prim in stage.Traverse():
            if "cube" in prim.GetName().lower():
                mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
                if mass_api:
                    m = mass_api.GetMassAttr().Get()
                    if "/env_0/" in str(prim.GetPath()):
                        print(f"[Cube] mass={m:.4f} kg ({m*1000:.1f}g) friction=0.3", flush=True)
                break
        print(f"[Cube] Set {cube_count} cubes to 50g", flush=True)
    except Exception as e:
        print(f"[Cube] Physics error: {e}", flush=True)
