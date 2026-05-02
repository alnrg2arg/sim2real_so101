"""Replay success trajectories and record GIF (front + side + wrist cameras)."""
import os, glob, numpy as np, yaml

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", "/workspace/leisaac/assets")

CFG = {}
if os.path.exists("configs/reward_config.yaml"):
    with open("configs/reward_config.yaml") as f:
        CFG = yaml.safe_load(f)
CFG["enable_cameras"] = True

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"enable_cameras": True})
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import isaaclab.sim as sim_utils
import leisaac.tasks.lift_cube
from leisaac.tasks.lift_cube import mdp
from packages.sim.env_setup.env_config import configure_env, apply_motor_limits
from PIL import Image

DEVICE = "cuda:0"
TRAJ_DIR = "/data/rl_sac_state_v17/episodes"
OUTPUT = "/data/replay_v17"
MAX_REPLAYS = 1

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

# 3 cameras
env_cfg.scene.front = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(-0.03141, -0.5301, 0.43648), rot=(0.92171, 0.38687, -0.02715, -0.00607), convention="opengl"),
    data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=13.0, focus_distance=400.0, horizontal_aperture=20.955, vertical_aperture=15.2908, clipping_range=(0.01, 50.0), lock_camera=True),
    width=640, height=480, update_period=0)
env_cfg.scene.side = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/side_camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(-0.4806, -0.56229, 0.26344), rot=(0.71926, 0.51278, -0.29271, -0.36614), convention="opengl"),
    data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=17.0, focus_distance=400.0, horizontal_aperture=20.955, vertical_aperture=15.2908, clipping_range=(0.01, 50.0), lock_camera=True),
    width=640, height=480, update_period=0)
if not hasattr(env_cfg.scene, 'wrist') or env_cfg.scene.wrist is None:
    env_cfg.scene.wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/fp_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.05, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=2.5, focus_distance=200.0, horizontal_aperture=24.0, vertical_aperture=18.0, clipping_range=(0.003, 50.0), lock_camera=True),
        width=640, height=480, update_period=0)

if hasattr(env_cfg.observations, 'subtask_terms'):
    env_cfg.observations.subtask_terms = None

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, CFG)
print(f"[Replay] Env ready", flush=True)

os.makedirs(OUTPUT, exist_ok=True)
traj_files = sorted(glob.glob(os.path.join(TRAJ_DIR, "success_*/trajectory.npz")))[:MAX_REPLAYS]
print(f"[Replay] {len(traj_files)} trajectories to replay", flush=True)

def grab_frame(cam_name):
    try:
        cam = env.scene[cam_name]
        rgb = cam.data.output["rgb"][0, :, :, :3]
        if rgb.dtype == torch.float32:
            rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
        return rgb.cpu().numpy()
    except:
        return None

for ri, traj_file in enumerate(traj_files):
    data = np.load(traj_file, allow_pickle=True)
    ep_len = int(data["ep_length"])
    traj_name = os.path.basename(os.path.dirname(traj_file))
    actions = torch.tensor(data["actions"], device=DEVICE, dtype=torch.float32)
    target_qpos = torch.tensor(data["target_qpos"], device=DEVICE, dtype=torch.float32)
    print(f"\n[Replay] {ri+1}/{len(traj_files)}: {traj_name} ({ep_len} steps, final_z={data['final_cube_z']:.3f})", flush=True)

    env.reset()
    # Set initial target
    init_tgt = target_qpos[0].unsqueeze(0)
    for term in env.action_manager._terms.values():
        if hasattr(term, '_target_qpos'):
            n_j = term._target_qpos.shape[-1]
            if n_j == 5:
                term._target_qpos[:] = init_tgt[:, :5]
            elif n_j == 1:
                term._target_qpos[:] = init_tgt[:, 5:]

    frames_front, frames_side, frames_wrist = [], [], []

    traj_qpos = torch.tensor(data["qpos"], device=DEVICE, dtype=torch.float32)
    traj_cube = torch.tensor(data["cube_pos"], device=DEVICE, dtype=torch.float32)
    traj_cube_q = torch.tensor(data.get("cube_quat", np.tile([1,0,0,0], (ep_len,1))), device=DEVICE, dtype=torch.float32)

    for step in range(ep_len):
        # Set robot joint state directly
        qpos = traj_qpos[step].unsqueeze(0)
        qvel = torch.zeros_like(qpos)
        env.scene["robot"].write_joint_state_to_sim(qpos, qvel)
        env.scene["robot"].set_joint_position_target(qpos)

        # Set cube position (trajectory stores env-local coords, add env_origin for world frame)
        cube_pos_local = traj_cube[step].unsqueeze(0)
        cube_quat = traj_cube_q[step].unsqueeze(0)
        env_origin = env.scene.env_origins[:1]  # (1, 3)
        cube_pos_world = cube_pos_local + env_origin
        env.scene["cube"].write_root_pose_to_sim(torch.cat([cube_pos_world, cube_quat], dim=-1))
        env.scene["cube"].write_root_velocity_to_sim(torch.zeros(1, 6, device=DEVICE))

        # Write to sim and step
        env.scene.write_data_to_sim()
        env.sim.step(render=True)
        env.scene.update(dt=env.cfg.sim.dt * env.cfg.decimation)

        f = grab_frame("front")
        if f is not None: frames_front.append(f)
        f = grab_frame("side")
        if f is not None: frames_side.append(f)
        f = grab_frame("wrist")
        if f is not None: frames_wrist.append(f)

        cube_z = env.scene["cube"].data.root_pos_w[0, 2].item()
        if step % 10 == 0:
            print(f"  step {step}/{ep_len} z={cube_z:.3f}", flush=True)

    # Save GIFs
    for cam_name, frames in [("front", frames_front), ("side", frames_side), ("wrist", frames_wrist)]:
        if frames:
            out = os.path.join(OUTPUT, f"{traj_name}_{cam_name}.gif")
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=100, loop=0)
            print(f"  Saved: {out}", flush=True)

print(f"\n[Done] All replays saved to {OUTPUT}", flush=True)
env.close()
simulation_app.close()
