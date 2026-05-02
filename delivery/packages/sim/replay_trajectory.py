"""Replay saved success trajectories in Isaac Lab and record video."""
import argparse
import os
import glob
import numpy as np
import yaml

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", "/workspace/leisaac/assets")

parser = argparse.ArgumentParser()
parser.add_argument("--traj-dir", type=str, default="/data/rl_sac_state_v10/episodes")
parser.add_argument("--output-dir", type=str, default="/data/replay_videos")
parser.add_argument("--max-episodes", type=int, default=5)
parser.add_argument("--config", type=str, default="configs/reward_config.yaml")
args = parser.parse_args()

CFG = {}
if os.path.exists(args.config):
    with open(args.config) as f:
        CFG = yaml.safe_load(f)
CFG["enable_cameras"] = True

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"enable_cameras": True})
simulation_app = app_launcher.app

import torch
import torch.nn.functional as F
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import isaaclab.sim as sim_utils
import leisaac.tasks.lift_cube
from leisaac.tasks.lift_cube import mdp
from packages.sim.env_setup.env_config import configure_env, apply_motor_limits

DEVICE = "cuda:0"

# Create env with 1 env + cameras
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=1)
env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg)

_cube_prim = env_cfg.scene.cube.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper", update_period=0.0, history_length=4,
    filter_prim_paths_expr=[_cube_prim])
env_cfg.scene.jaw_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/jaw", update_period=0.0, history_length=4,
    filter_prim_paths_expr=[_cube_prim])
env_cfg.scene.cube_contact = ContactSensorCfg(
    prim_path=_cube_prim, update_period=0.0, history_length=4,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/gripper", "{ENV_REGEX_NS}/Robot/jaw"])
env_cfg.scene.cube.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True

# Ensure wrist camera
if not hasattr(env_cfg.scene, 'wrist') or env_cfg.scene.wrist is None:
    from isaaclab.sensors import TiledCameraCfg
    env_cfg.scene.wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/fp_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.05, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=2.5, focus_distance=200.0, horizontal_aperture=24.0, vertical_aperture=18.0, clipping_range=(0.003, 50.0), lock_camera=True),
        width=640, height=480, update_period=0)

# Add front camera for recording
from isaaclab.sensors import TiledCameraCfg
env_cfg.scene.front = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(-0.03141, -0.5301, 0.43648), rot=(0.92171, 0.38687, -0.02715, -0.00607), convention="opengl"),
    data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=13.0, focus_distance=400.0, horizontal_aperture=20.955, vertical_aperture=15.2908, clipping_range=(0.01, 50.0), lock_camera=True),
    width=640, height=480, update_period=0)

if hasattr(env_cfg.observations, 'subtask_terms'):
    env_cfg.observations.subtask_terms = None

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, CFG)
print(f"[Replay] Env created", flush=True)

# Find trajectories
traj_dirs = sorted(glob.glob(os.path.join(args.traj_dir, "success_*")))
if not traj_dirs:
    print(f"[Replay] No trajectories found in {args.traj_dir}")
    exit(1)
print(f"[Replay] Found {len(traj_dirs)} success episodes", flush=True)

os.makedirs(args.output_dir, exist_ok=True)

# Try to import video writer
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[Replay] cv2 not available, saving frames as images", flush=True)

from PIL import Image

for ep_idx, traj_dir in enumerate(traj_dirs[:args.max_episodes]):
    traj_file = os.path.join(traj_dir, "trajectory.npz")
    if not os.path.exists(traj_file):
        continue

    data = np.load(traj_file, allow_pickle=True)
    traj_actions = torch.tensor(data["actions"], device=DEVICE, dtype=torch.float32)
    traj_qpos = data["qpos"]
    ep_len = int(data["ep_length"])
    ep_name = os.path.basename(traj_dir)

    print(f"\n[Replay] Episode {ep_idx+1}/{args.max_episodes}: {ep_name} ({ep_len} steps)", flush=True)

    # Reset env
    env.reset()

    # Set initial joint positions from trajectory
    robot = env.scene["robot"]
    init_qpos = torch.tensor(traj_qpos[0], device=DEVICE, dtype=torch.float32).unsqueeze(0)
    robot.set_joint_position_target(init_qpos)

    # Also set action manager targets (arm=5 joints, gripper=1 joint)
    for term_name, term in env.action_manager._terms.items():
        if hasattr(term, '_target_qpos'):
            n_joints = term._target_qpos.shape[-1]
            if n_joints == 5:  # arm
                term._target_qpos[:] = init_qpos[:, :5]
            elif n_joints == 1:  # gripper
                term._target_qpos[:] = init_qpos[:, 5:]
            else:
                term._target_qpos[:] = init_qpos[:, :n_joints]

    traj_target = torch.tensor(data["target_qpos"], device=DEVICE, dtype=torch.float32)
    frames_front = []
    frames_wrist = []

    for step in range(ep_len):
        # Directly set target_qpos from trajectory (bypass action network)
        target = traj_target[step].unsqueeze(0)
        for term_name, term in env.action_manager._terms.items():
            if hasattr(term, '_target_qpos'):
                n_j = term._target_qpos.shape[-1]
                if n_j == 5:
                    term._target_qpos[:] = target[:, :5]
                elif n_j == 1:
                    term._target_qpos[:] = target[:, 5:]
        robot = env.scene["robot"]
        robot.set_joint_position_target(target)

        # Step with zero action (target already set)
        action = torch.zeros(1, traj_actions.shape[-1], device=DEVICE)
        env.step(action)

        # Capture frames
        try:
            front_cam = env.scene["front"]
            front_rgb = front_cam.data.output["rgb"][0, :, :, :3]
            if front_rgb.dtype == torch.float32:
                front_rgb = (front_rgb * 255).clamp(0, 255).to(torch.uint8)
            frames_front.append(front_rgb.cpu().numpy())
        except:
            pass

        try:
            wrist_cam = env.scene["wrist"]
            wrist_rgb = wrist_cam.data.output["rgb"][0, :, :, :3]
            if wrist_rgb.dtype == torch.float32:
                wrist_rgb = (wrist_rgb * 255).clamp(0, 255).to(torch.uint8)
            frames_wrist.append(wrist_rgb.cpu().numpy())
        except:
            pass

        cube_z = env.scene["cube"].data.root_pos_w[0, 2].item()
        if step % 10 == 0:
            print(f"  step {step}/{ep_len} cube_z={cube_z:.3f}", flush=True)

    # Save video
    if frames_front:
        if HAS_CV2:
            h, w = frames_front[0].shape[:2]
            out_path = os.path.join(args.output_dir, f"{ep_name}_front.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
            for f in frames_front:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"  Saved: {out_path}", flush=True)
        else:
            # Save as GIF
            out_path = os.path.join(args.output_dir, f"{ep_name}_front.gif")
            imgs = [Image.fromarray(f) for f in frames_front]
            imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
            print(f"  Saved: {out_path}", flush=True)

    if frames_wrist:
        if HAS_CV2:
            h, w = frames_wrist[0].shape[:2]
            out_path = os.path.join(args.output_dir, f"{ep_name}_wrist.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
            for f in frames_wrist:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"  Saved: {out_path}", flush=True)
        else:
            out_path = os.path.join(args.output_dir, f"{ep_name}_wrist.gif")
            imgs = [Image.fromarray(f) for f in frames_wrist]
            imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
            print(f"  Saved: {out_path}", flush=True)

print(f"\n[Replay] Done! Videos saved to {args.output_dir}", flush=True)
env.close()
simulation_app.close()
