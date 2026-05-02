"""Replay top 20 success + 10 checkpoint inference episodes."""
import os, glob, numpy as np, yaml, sys

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
import torch.nn as nn
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
EPISODE_DIR = "/data/rl_sac_state_v41/episodes"
CKPT = "/data/rl_sac_state_v41/best_ckpt.pt"
OUTPUT = "/data/replay_v41_final"
TOP_N = 10
INFER_N = 0

# ── Env ──
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=1)
env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg)
_cube_prim = env_cfg.scene.cube.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", update_period=0.0, history_length=4, filter_prim_paths_expr=[_cube_prim])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/jaw", update_period=0.0, history_length=4, filter_prim_paths_expr=[_cube_prim])
env_cfg.scene.cube_contact = ContactSensorCfg(prim_path=_cube_prim, update_period=0.0, history_length=4, filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/gripper", "{ENV_REGEX_NS}/Robot/jaw"])
env_cfg.scene.cube.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True

env_cfg.scene.front = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(-0.03141, -0.5301, 0.43648), rot=(0.92171, 0.38687, -0.02715, -0.00607), convention="opengl"),
    data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=13.0, focus_distance=400.0, horizontal_aperture=20.955, vertical_aperture=15.2908, clipping_range=(0.01, 50.0), lock_camera=True),
    width=640, height=480, update_period=0)
env_cfg.scene.side = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/base/side_camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(-0.4806, -0.56229, 0.26344), rot=(0.71926, 0.51278, -0.29271, -0.36614), convention="opengl"),
    data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=17.0, focus_distance=400.0, horizontal_aperture=20.955, vertical_aperture=15.2908, clipping_range=(0.01, 50.0), lock_camera=True),
    width=640, height=480, update_period=0)
if not hasattr(env_cfg.scene, 'wrist') or env_cfg.scene.wrist is None:
    env_cfg.scene.wrist = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper/fp_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.05, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=2.5, focus_distance=200.0, horizontal_aperture=24.0, vertical_aperture=18.0, clipping_range=(0.003, 50.0), lock_camera=True),
        width=640, height=480, update_period=0)
if hasattr(env_cfg.observations, 'subtask_terms'):
    env_cfg.observations.subtask_terms = None

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, CFG)
obs_dim = env.observation_manager.compute()["policy"].shape[-1]
act_dim = env.action_manager.total_action_dim
os.makedirs(OUTPUT, exist_ok=True)

def grab_frame(cam_name):
    try:
        cam = env.scene[cam_name]
        rgb = cam.data.output["rgb"][0, :, :, :3]
        if rgb.dtype == torch.float32:
            rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
        return rgb.cpu().numpy()
    except:
        return None

def save_gif(frames, path):
    if frames:
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100, loop=0)

# ═══════════════════════════════════════
# Part 1: Top 20 success trajectory replay
# ═══════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"Part 1: Top {TOP_N} success trajectory replay", flush=True)
print(f"{'='*60}", flush=True)

# Find all trajectories and sort by pan_ratio
all_trajs = []
for d in glob.glob(os.path.join(EPISODE_DIR, "*/trajectory.npz")):
    try:
        data = np.load(d, allow_pickle=True)
        cz = float(data.get("final_cube_z", 0))
        ratio = float(data.get("pan_ratio", data.get("fold_ratio", 0)))
        ret = float(data["rewards"].sum()) if "rewards" in data else 0
        all_trajs.append((ret, d))
    except:
        pass
all_trajs.sort(reverse=True)
print(f"Found {len(all_trajs)} total successes", flush=True)

for ri, (ret_val, traj_file) in enumerate(all_trajs[:TOP_N]):
    data = np.load(traj_file, allow_pickle=True)
    ep_len = int(data["ep_length"])
    traj_name = os.path.basename(os.path.dirname(traj_file))
    traj_qpos = torch.tensor(data["qpos"], device=DEVICE, dtype=torch.float32)
    traj_cube = torch.tensor(data["cube_pos"], device=DEVICE, dtype=torch.float32)
    traj_cube_q = torch.tensor(data.get("cube_quat", np.tile([1,0,0,0], (ep_len,1))), device=DEVICE, dtype=torch.float32)

    print(f"\n[Replay {ri+1}/{TOP_N}] {traj_name} ret={ret_val:.0f} ({ep_len} steps)", flush=True)
    env.reset()

    # Set cube to recorded initial position + orientation
    cube = env.scene["cube"]
    init_pos = traj_cube[0].unsqueeze(0) + env.scene.env_origins  # local → world
    init_quat = traj_cube_q[0].unsqueeze(0)
    cube.write_root_pose_to_sim(torch.cat([init_pos, init_quat], dim=-1))
    print(f"  Cube set to: {traj_cube[0].cpu().numpy()}", flush=True)

    traj_tgt = torch.tensor(data["target_qpos"], device=DEVICE, dtype=torch.float32)

    # Disable action manager apply_actions — we set targets manually
    _orig_apply = {}
    for name, term in env.action_manager._terms.items():
        _orig_apply[name] = term.apply_actions
        term.apply_actions = lambda: None

    from packages.sim.env_setup.maniskill_rewards import _fold_3joint, _is_grasped
    hold_first = -1
    hold_steps = 0
    frames_front, frames_side, frames_wrist = [], [], []

    for step in range(ep_len):
        # Set target_qpos directly from recorded data (bypass action manager)
        tgt = traj_tgt[step].unsqueeze(0)
        for term in env.action_manager._terms.values():
            if hasattr(term, '_target_qpos'):
                n_j = term._target_qpos.shape[-1]
                if n_j == 5:
                    term._target_qpos[:] = tgt[:, :5]
                elif n_j == 1:
                    term._target_qpos[:] = tgt[:, 5:]
        env.scene["robot"].set_joint_position_target(tgt)
        zero_action = torch.zeros(1, act_dim, device=DEVICE)
        env.step(zero_action)

        f = grab_frame("front")
        if f is not None: frames_front.append(f)
        f = grab_frame("side")
        if f is not None: frames_side.append(f)
        f = grab_frame("wrist")
        if f is not None: frames_wrist.append(f)

        cz = env.scene["cube"].data.root_pos_w[0, 2].item()
        pan_err = _fold_3joint(env)[0][0].item()
        gr = _is_grasped(env)[0].item()
        holding = (cz >= 0.15) and (gr > 0.5) and (pan_err < 0.087)
        if holding:
            if hold_first < 0: hold_first = step
            hold_steps += 1

        if step % 20 == 0:
            print(f"  step {step}/{ep_len} z={cz:.3f} gr={int(gr)} hold={hold_steps}", flush=True)

    # Restore apply_actions
    for name, term in env.action_manager._terms.items():
        term.apply_actions = _orig_apply[name]

    # Success/fail
    remaining = max(ep_len - hold_first, 1) if hold_first >= 0 else 1
    ratio = hold_steps / remaining if hold_first >= 0 else 0
    success = hold_first >= 0 and hold_steps >= 5 and remaining > 5 and ratio >= 0.90
    tag = "SUCCESS" if success else "FAIL"
    print(f"  [{tag}] ratio={ratio:.0%} hold={hold_steps}/{remaining}", flush=True)

    ep_dir = os.path.join(OUTPUT, f"{tag}_top{ri+1:02d}_{traj_name}")
    os.makedirs(ep_dir, exist_ok=True)
    save_gif(frames_front, os.path.join(ep_dir, "front.gif"))
    save_gif(frames_side, os.path.join(ep_dir, "side.gif"))
    save_gif(frames_wrist, os.path.join(ep_dir, "wrist.gif"))
    print(f"  Saved {ep_dir}", flush=True)

# ═══════════════════════════════════════
# Part 2: Checkpoint inference (10 episodes)
# ═══════════════════════════════════════
print(f"\n{'='*60}", flush=True)
print(f"Part 2: Checkpoint inference ({INFER_N} episodes)", flush=True)
print(f"{'='*60}", flush=True)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None: m.bias.data.fill_(0.0)

class StateProjection(nn.Module):
    def __init__(self, n_state, device=None):
        super().__init__()
        self.repr_dim = 256
        self.state_proj = nn.Sequential(nn.Linear(n_state, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU())
    def forward(self, state): return self.state_proj(state)

class Actor(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.proj = StateProjection(n_obs, device=device)
        self.fc = nn.Sequential(
            nn.Linear(256, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU(),
            nn.Linear(256, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU(),
            nn.Linear(256, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU())
        self.fc_mean = nn.Linear(256, n_act, device=device)
        self.fc_logstd = nn.Linear(256, n_act, device=device)
        self.register_buffer("action_scale", torch.ones(n_act, device=device))
        self.register_buffer("action_bias", torch.zeros(n_act, device=device))
        self.apply(weight_init)
    def get_eval_action(self, state):
        x = self.proj(state); x = self.fc(x)
        return torch.tanh(self.fc_mean(x)) * self.action_scale + self.action_bias

actor = Actor(n_obs=obs_dim, n_act=act_dim, device=DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
actor.load_state_dict(ckpt["actor"])
actor.eval()
print(f"Loaded: {CKPT}", flush=True)

for ep in range(INFER_N):
    obs_dict = env.reset()[0]
    obs = obs_dict["policy"]
    frames = []
    ep_reward = 0
    max_z = 0

    for step in range(100):
        with torch.no_grad():
            action = actor.get_eval_action(obs)
        obs_dict, rewards, dones, truncs, infos = env.step(action)
        obs = obs_dict["policy"]
        ep_reward += rewards.item()
        cz = env.scene["cube"].data.root_pos_w[0, 2].item()
        max_z = max(max_z, cz)
        f = grab_frame("front")
        if f is not None: frames.append(f)
        if step % 20 == 0:
            print(f"  ep{ep} step={step} z={cz:.3f} r={ep_reward:.0f}", flush=True)
        if (dones | truncs).any(): break

    save_gif(frames, os.path.join(OUTPUT, f"infer_ep{ep}_ret{ep_reward:.0f}_z{max_z*100:.0f}cm.gif"))
    print(f"[Infer] ep{ep}: ret={ep_reward:.0f} max_z={max_z:.3f}", flush=True)

print(f"\n[Done] All saved to {OUTPUT}", flush=True)
env.close()
simulation_app.close()
