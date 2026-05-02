"""Run trained SAC policy and record video."""
import os
import numpy as np
import yaml

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", "/workspace/leisaac/assets")
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

CFG = {}
if os.path.exists("configs/reward_config.yaml"):
    with open("configs/reward_config.yaml") as f:
        CFG = yaml.safe_load(f)
CFG["enable_cameras"] = True

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"enable_cameras": True})
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.nn.functional as F
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import isaaclab.sim as sim_utils
import leisaac.tasks.lift_cube
from leisaac.tasks.lift_cube import mdp
from packages.sim.env_setup.env_config import configure_env, apply_motor_limits

DEVICE = "cuda:0"
CKPT = "/data/rl_sac_state_v41/best_ckpt.pt"
OUTPUT = "/data/infer_v41_30fps"
NUM_EPISODES = 5

# Env
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=1)
env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg)
# 30fps render: decimation=3 → 100Hz/3 ≈ 33Hz → ~333 steps in 10s
env_cfg.decimation = 3
env_cfg.sim.render_interval = 1
# Wider spawn range for diverse demos
for ev_cfg in getattr(env_cfg.events, '__dict__', {}).values():
    if hasattr(ev_cfg, 'params') and 'pose_range' in getattr(ev_cfg, 'params', {}):
        ev_cfg.params["pose_range"]["x"] = (-0.15, 0.15)
        ev_cfg.params["pose_range"]["y"] = (-0.15, 0.15)

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

if not hasattr(env_cfg.scene, 'wrist') or env_cfg.scene.wrist is None:
    env_cfg.scene.wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/fp_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.05, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=2.5, focus_distance=200.0, horizontal_aperture=24.0, vertical_aperture=18.0, clipping_range=(0.003, 50.0), lock_camera=True),
        width=640, height=480, update_period=0)

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

if hasattr(env_cfg.observations, 'subtask_terms'):
    env_cfg.observations.subtask_terms = None

env_cfg.sim.physx.solver_position_iteration_count = 15
env_cfg.sim.physx.solver_velocity_iteration_count = 1
env_cfg.sim.physx.rest_offset = 0.0
env_cfg.sim.physx.solve_articulation_contact_last = True  # gripper penetration fix
env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, CFG)

obs_dim = env.observation_manager.compute()["policy"].shape[-1]
act_dim = env.action_manager.total_action_dim
print(f"[Infer] obs={obs_dim}, act={act_dim}", flush=True)

# Load actor from checkpoint (same architecture as train_sac_fast.py)
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None: m.bias.data.fill_(0.0)

class StateProjection(nn.Module):
    def __init__(self, n_state, device=None):
        super().__init__()
        self.repr_dim = 256
        self.state_proj = nn.Sequential(
            nn.Linear(n_state, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU())
    def forward(self, state):
        return self.state_proj(state)

class Actor(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.proj = StateProjection(n_obs, device=device)
        self.fc = nn.Sequential(
            nn.Linear(self.proj.repr_dim, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU(),
            nn.Linear(256, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU(),
            nn.Linear(256, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU())
        self.fc_mean = nn.Linear(256, n_act, device=device)
        self.fc_logstd = nn.Linear(256, n_act, device=device)
        self.register_buffer("action_scale", torch.ones(n_act, device=device))
        self.register_buffer("action_bias", torch.zeros(n_act, device=device))
        self.LOG_STD_MAX = 2; self.LOG_STD_MIN = -5
        self.apply(weight_init)

    def get_eval_action(self, state):
        x = self.proj(state)
        x = self.fc(x)
        mean = self.fc_mean(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias

actor = Actor(n_obs=obs_dim, n_act=act_dim, device=DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
actor.load_state_dict(ckpt["actor"])
actor.eval()
print(f"[Infer] Loaded: {CKPT}", flush=True)

os.makedirs(OUTPUT, exist_ok=True)
from PIL import Image

def grab_cam(cam_name):
    try:
        cam = env.scene[cam_name]
        rgb = cam.data.output["rgb"][0, :, :, :3]
        if rgb.dtype == torch.float32:
            rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
        return rgb.cpu().numpy()
    except:
        return None

from packages.sim.env_setup.maniskill_rewards import _fold_3joint, _is_grasped
import math

# 5 diverse cube spawn configs: (x_offset, y_offset, yaw_degrees)
SPAWN_CONFIGS = [
    (0.00,  0.00,    0),   # center, facing forward
    (0.08, -0.05,   45),   # right-front, 45° rotated
    (-0.08,  0.05,  -60),  # left-back, -60° rotated
    (0.05,  0.08,   90),   # front-right, perpendicular
    (-0.05, -0.08, 135),   # back-left, 135° rotated
]

total_success = 0
for ep in range(NUM_EPISODES):
    obs_dict = env.reset()[0]
    obs = obs_dict["policy"]

    # Log random spawn position/orientation
    _origin = env.scene.env_origins[0]
    _cpos = env.scene["cube"].data.root_pos_w[0] - _origin
    _cquat = env.scene["cube"].data.root_quat_w[0]
    _yaw = math.degrees(2 * math.atan2(_cquat[3].item(), _cquat[0].item()))
    print(f"  Cube spawn: x={_cpos[0]:.3f} y={_cpos[1]:.3f} yaw={_yaw:.0f}°", flush=True)
    ep_reward = 0
    frames_front, frames_side, frames_wrist = [], [], []
    ep_data = {"qpos": [], "actions": [], "cube_pos": [], "cube_quat": [], "target_qpos": []}
    hold_first = -1
    hold_steps = 0

    for step in range(333):  # 10s at 33Hz (30fps render)
        with torch.no_grad():
            action = actor.get_eval_action(obs)
        obs_dict, rewards, dones, truncs, infos = env.step(action)
        obs = obs_dict["policy"]
        ep_reward += rewards.item()

        # Record data
        ep_data["qpos"].append(env.scene["robot"].data.joint_pos[0].cpu().numpy())
        ep_data["actions"].append(action[0].cpu().numpy())
        _origins = env.scene.env_origins[0]
        ep_data["cube_pos"].append((env.scene["cube"].data.root_pos_w[0] - _origins).cpu().numpy())
        ep_data["cube_quat"].append(env.scene["cube"].data.root_quat_w[0].cpu().numpy())
        am = env.action_manager
        tgts = []
        for term in am._terms.values():
            if hasattr(term, '_target_qpos'):
                tgts.append(term._target_qpos[0])
        if tgts:
            ep_data["target_qpos"].append(torch.cat(tgts, dim=-1).cpu().numpy())

        # 3 camera frames
        f = grab_cam("front")
        if f is not None: frames_front.append(f)
        f = grab_cam("side")
        if f is not None: frames_side.append(f)
        f = grab_cam("wrist")
        if f is not None: frames_wrist.append(f)

        cube_z = env.scene["cube"].data.root_pos_w[0, 2].item()
        # Hold tracking: lifted(15cm) + grasped + pan<5°
        pan_err = _fold_3joint(env)[0][0].item()
        gr = _is_grasped(env)[0].item()
        holding = (cube_z >= 0.15) and (gr > 0.5) and (pan_err < 0.262)
        if holding:
            if hold_first < 0:
                hold_first = step
            hold_steps += 1

        if step % 20 == 0:
            print(f"  ep{ep} step={step} z={cube_z:.3f} pan={pan_err:.3f} gr={int(gr)} hold={hold_steps} r={ep_reward:.1f}", flush=True)
        if (dones | truncs).any():
            break

    # Success/fail: 80% hold ratio (333 steps at 33Hz = 10s)
    MAX_STEPS = 333
    remaining = max(MAX_STEPS - hold_first, 1) if hold_first >= 0 else 1
    ratio = hold_steps / remaining if hold_first >= 0 else 0
    success = hold_first >= 0 and hold_steps >= 5 and remaining > 5 and ratio >= 0.80
    tag = "SUCCESS" if success else "FAIL"
    if success:
        total_success += 1
    print(f"  [{tag}] ep{ep} ret={ep_reward:.0f} ratio={ratio:.0%} hold={hold_steps}/{remaining}", flush=True)

    # Save episode dir
    ep_dir = os.path.join(OUTPUT, f"{tag}_ep{ep:02d}_x{_cpos[0]:.0f}y{_cpos[1]:.0f}yaw{_yaw:.0f}_ret{ep_reward:.0f}")
    os.makedirs(ep_dir, exist_ok=True)

    # Save trajectory npz
    np.savez_compressed(os.path.join(ep_dir, "trajectory.npz"),
        qpos=np.array(ep_data["qpos"]),
        actions=np.array(ep_data["actions"]),
        cube_pos=np.array(ep_data["cube_pos"]),
        cube_quat=np.array(ep_data["cube_quat"]),
        target_qpos=np.array(ep_data["target_qpos"]),
        ep_length=len(ep_data["qpos"]),
        total_reward=ep_reward)

    # Save 3 camera frames as npz (for VLA training)
    if frames_front:
        np.savez_compressed(os.path.join(ep_dir, "cam_front.npz"), frames=np.array(frames_front))
    if frames_side:
        np.savez_compressed(os.path.join(ep_dir, "cam_side.npz"), frames=np.array(frames_side))
    if frames_wrist:
        np.savez_compressed(os.path.join(ep_dir, "cam_wrist.npz"), frames=np.array(frames_wrist))

    # Save MP4s (30fps)
    import cv2
    def save_mp4(frs, path):
        if frs:
            h, w = frs[0].shape[:2]
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
            for f in frs:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
    save_mp4(frames_front, os.path.join(ep_dir, "front.mp4"))
    save_mp4(frames_side, os.path.join(ep_dir, "side.mp4"))
    save_mp4(frames_wrist, os.path.join(ep_dir, "wrist.mp4"))

    print(f"[Infer] ep{ep}: ret={ep_reward:.0f} frames={len(frames_front)} SAVED to {ep_dir}", flush=True)

print(f"\n[Done] {NUM_EPISODES} episodes, {total_success} SUCCESS ({total_success/NUM_EPISODES*100:.0f}%) saved to {OUTPUT}", flush=True)
env.close()
simulation_app.close()
