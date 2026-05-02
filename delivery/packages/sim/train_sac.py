"""SAC training for SO101 LiftCube using stable-baselines3.

Key differences from PPO:
- Off-policy: replay buffer stores experiences for re-use
- Entropy maximization: automatic exploration
- Much more sample efficient: 1-2M steps vs 25M+
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np
import torch

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", "/workspace/leisaac/assets")

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--total-timesteps", type=int, default=5_000_000)
parser.add_argument("--http-port", type=int, default=8888)
parser.add_argument("--save-dir", type=str, default="/data/rl_sac")
parser.add_argument("--config", type=str, default="configs/reward_config.yaml")
args, _ = parser.parse_known_args()

CFG = {}
if os.path.exists(args.config):
    with open(args.config) as f:
        CFG = yaml.safe_load(f)

# Isaac Lab init
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"enable_cameras": CFG.get("enable_cameras", False)})
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import isaaclab.sim as sim_utils
import leisaac.tasks.lift_cube
from leisaac.tasks.lift_cube import mdp

from packages.sim.env_setup.env_config import configure_env, add_fingertip_pads, apply_motor_limits
from packages.sim import dashboard

import gymnasium as gym
from stable_baselines3 import SAC
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

DEVICE = "cuda:0"

# ── Env setup ──
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=args.num_envs)
env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg)

# Contact sensors with filter
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

# Remove cameras only if disabled
if not CFG.get("enable_cameras", False):
    for cam in ["front", "side", "wrist"]:
        if hasattr(env_cfg.scene, cam):
            setattr(env_cfg.scene, cam, None)
    for attr in list(vars(env_cfg.observations.policy)):
        if attr in ('front', 'wrist', 'side') or 'image' in attr.lower():
            setattr(env_cfg.observations.policy, attr, None)
if hasattr(env_cfg.observations, 'subtask_terms'):
    env_cfg.observations.subtask_terms = None

env = ManagerBasedRLEnv(cfg=env_cfg)
print(f"[Env] Created: obs={env.observation_space}, act={env.action_space}", flush=True)

# Apply motor limits + fingertip pads
apply_motor_limits(env, CFG)
pass  # fingertip pads removed

# ── SB3 Wrapper ──
# SB3 needs a gym-compatible env. Isaac Lab env returns tensors on GPU.
# We need a wrapper that converts to numpy.

# ── SB3 Wrapper ──
sb3_env = Sb3VecEnvWrapper(env)
print(f"[SB3] Wrapped: obs={sb3_env.observation_space}, act={sb3_env.action_space}", flush=True)

# Setup cameras
_cam_map = {}
for _cn in ["front","side","wrist"]:
    try:
        _cam_map[_cn] = env.scene[_cn]
        print(f"[Camera] {_cn}: OK", flush=True)
    except:
        pass

# ── Dashboard ──
dashboard.train_stats = {
    "status": "training", "iteration": 0, "max_iterations": args.total_timesteps,
    "mean_reward": 0, "max_reward": 0, "alltime_max_reward": 0,
    "success_count": 0, "episode_count": 0, "best_lift_cm": 0,
    "reward_history": [], "recent_episodes": [],
}
dashboard.start(args.http_port)

# ── SAC Training ──
os.makedirs(args.save_dir, exist_ok=True)

print(f"[SAC] Starting training: {args.total_timesteps} steps, {args.num_envs} envs", flush=True)
print(f"[SAC] obs={sb3_env.observation_space.shape}, act={sb3_env.action_space.shape}", flush=True)

model = SAC(
    "MlpPolicy",
    sb3_env,
    learning_rate=3e-4,
    buffer_size=1_000_000,  # Squint: 10x larger
    batch_size=512,  # Squint: 2x larger
    gamma=0.9,  # Squint: short horizon for 50-step eps
    tau=0.01,  # Squint: faster target update
    ent_coef="auto",
    train_freq=1,  # every env step
    gradient_steps=256,  # Squint: high UTD
    learning_starts=5000,  # Squint: more random exploration first
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256, 256], qf=[512, 512, 512]),  # Squint: wider critic
        use_sde=False,
    ),
    verbose=1,
    tensorboard_log=os.path.join(args.save_dir, "tb"),
    device="cuda",
)

from stable_baselines3.common.callbacks import BaseCallback

class DashboardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._ep_rewards = []
        self._ep_reward = 0
        self._ep_count = 0
        self._best_reward = float('-inf')
        self._success_count = 0
        self._ep_frames = []  # current episode frames
        self._save_dir = os.path.join(args.save_dir, "episodes")
        os.makedirs(self._save_dir, exist_ok=True)
    
    def _on_step(self):
        reward = self.locals.get("rewards", [0])[0] if isinstance(self.locals.get("rewards"), (list,np.ndarray)) else self.locals.get("rewards", 0)
        self._ep_reward += float(reward) if not isinstance(reward, (list,np.ndarray)) else float(reward)
        
        # Record frame data for episode
        try:
            frame = {
                "joint_pos": env.scene["robot"].data.joint_pos[0].cpu().numpy().copy(),
                "cube_pos": env.scene["cube"].data.root_pos_w[0].cpu().numpy().copy(),
                "ee_pos": env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy().copy(),
            }
            # Get action from locals
            act = self.locals.get("actions")
            if act is not None:
                if isinstance(act, torch.Tensor):
                    frame["action"] = act[0].cpu().numpy().copy()
                elif isinstance(act, np.ndarray):
                    frame["action"] = act[0].copy() if act.ndim > 1 else act.copy()
            self._ep_frames.append(frame)
        except:
            pass
        
        done = self.locals.get("dones", [False])[0] if isinstance(self.locals.get("dones"), (list,np.ndarray)) else self.locals.get("dones", False)
        
        # Check success: cube lifted + grasped
        is_success = False
        try:
            cube_z = env.scene["cube"].data.root_pos_w[0, 2].item()
            is_success = cube_z > 0.08  # lifted above 8cm
        except:
            pass
        
        if done:
            self._ep_count += 1
            self._ep_rewards.append(self._ep_reward)
            self._ep_rewards = self._ep_rewards[-100:]
            if self._ep_reward > self._best_reward:
                self._best_reward = self._ep_reward
            
            if is_success:
                self._success_count += 1
                # Save success episode
                try:
                    ep_dir = os.path.join(self._save_dir, f"success_ep{self._ep_count:06d}_r{self._ep_reward:.1f}")
                    os.makedirs(ep_dir, exist_ok=True)
                    for i, fr in enumerate(self._ep_frames):
                        np.savez_compressed(os.path.join(ep_dir, f"frame_{i:04d}.npz"), **fr)
                    import json
                    with open(os.path.join(ep_dir, "meta.json"), "w") as mf:
                        json.dump({"reward": self._ep_reward, "steps": len(self._ep_frames), 
                                   "success": True, "episode": self._ep_count, 
                                   "timestep": self.num_timesteps}, mf)
                    print(f"[SUCCESS #{self._success_count}] ep={self._ep_count} reward={self._ep_reward:.2f} steps={len(self._ep_frames)} saved to {ep_dir}", flush=True)
                except Exception as se:
                    print(f"[SUCCESS] Save error: {se}", flush=True)
            
            # Also save best episodes (even if not success)
            if self._ep_reward > self._best_reward * 0.9 and len(self._ep_frames) > 5:
                try:
                    ep_dir = os.path.join(self._save_dir, f"best_ep{self._ep_count:06d}_r{self._ep_reward:.1f}")
                    os.makedirs(ep_dir, exist_ok=True)
                    for i, fr in enumerate(self._ep_frames):
                        np.savez_compressed(os.path.join(ep_dir, f"frame_{i:04d}.npz"), **fr)
                except:
                    pass
            
            self._ep_reward = 0
            self._ep_frames = []
        
        # Update cameras
        if self.num_timesteps % 10 == 0 and _cam_map:
            dashboard.update_cameras(_cam_map)
        
        if self.num_timesteps % 100 == 0:
            # Get live reward terms from Isaac Lab env directly
            try:
                rm = env.unwrapped.reward_manager
                rt = {}
                for name in rm.active_terms:
                    cfg = rm.get_term_cfg(name)
                    try:
                        val = rm._term_values[name].mean().item()
                    except:
                        try:
                            raw = cfg.func(env.unwrapped, **cfg.params)
                            val = raw.mean().item() * cfg.weight
                        except:
                            val = 0.0
                    rt[name] = {"weight": cfg.weight, "value": round(val, 4)}
            except Exception as _e:
                rt = {"_error": str(_e)}
            
            with dashboard.stats_lock:
                dashboard.train_stats["iteration"] = self.num_timesteps
                dashboard.train_stats["episode_count"] = self._ep_count
                dashboard.train_stats["success_count"] = self._success_count
                dashboard.train_stats["status"] = "training"
                dashboard.train_stats["reward_terms"] = rt
                if self._ep_rewards:
                    dashboard.train_stats["mean_reward"] = float(np.mean(self._ep_rewards[-20:]))
                    dashboard.train_stats["alltime_max_reward"] = self._best_reward
                    dashboard.train_stats["reward_history"].append({
                        "iter": self.num_timesteps,
                        "mean": float(np.mean(self._ep_rewards[-20:])),
                        "max": float(max(self._ep_rewards[-20:]))
                    })
                    dashboard.train_stats["reward_history"] = dashboard.train_stats["reward_history"][-500:]
                    # Recent episodes
                    dashboard.train_stats["recent_episodes"] = [
                        {"ep": i+1, "reward": r} for i, r in enumerate(self._ep_rewards[-15:])]
        
        if self.num_timesteps % 1000 == 0:
            mean_r = np.mean(self._ep_rewards[-20:]) if self._ep_rewards else 0
            print(f"[SAC] step={self.num_timesteps} eps={self._ep_count} mean_r={mean_r:.2f} best={self._best_reward:.2f} success={self._success_count}", flush=True)
        
        # Save checkpoint every 50K steps
        if self.num_timesteps % 50000 == 0 and self.num_timesteps > 0:
            try:
                ckpt_path = os.path.join(args.save_dir, f"sac_ckpt_{self.num_timesteps}")
                self.model.save(ckpt_path)
            except:
                pass
        
        return True

model.learn(
    total_timesteps=args.total_timesteps,
    log_interval=10,
    callback=DashboardCallback(),
)

model.save(os.path.join(args.save_dir, "sac_final"))
print("[SAC] Training complete!", flush=True)

env.close()
simulation_app.close()
