#!/usr/bin/env python3
"""Replay demo episodes in Isaac Lab env → collect full 43-dim observations for BC."""

import argparse
import glob
import os

import numpy as np
import torch
import yaml

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", os.environ.get("LEISAAC_ASSETS_ROOT", "/workspace/leisaac/assets"))

parser = argparse.ArgumentParser()
parser.add_argument("--demo-dir", type=str, required=True)
parser.add_argument("--save-path", type=str, default="/data/rl_output/demo_full_obs.npz")
parser.add_argument("--config", type=str, default="configs/reward_config.yaml")
args = parser.parse_args()

# Load config
CFG = {}
if os.path.exists(args.config):
    with open(args.config) as f:
        CFG = yaml.safe_load(f)

# Isaac Lab init
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"enable_cameras": True})
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
import isaaclab.sim as sim_utils
import leisaac.tasks.lift_cube
from leisaac.tasks.lift_cube import mdp

from packages.sim.env_setup import configure_env

DEVICE = "cuda:0"

# Create env with 1 env
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=1)
env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg)
env = ManagerBasedRLEnv(cfg=env_cfg)
print(f"[Replay] Env created, obs groups: {list(env.observation_manager.group_obs_term_dim.keys())}")

# Get obs dim
obs = env.observation_manager.compute()
rl_obs = obs["rl_policy"]
obs_dim = rl_obs.shape[-1]
print(f"[Replay] rl_policy obs dim: {obs_dim}")

# Load demos
episodes = sorted(glob.glob(os.path.join(args.demo_dir, "episode_*")))
print(f"[Replay] Found {len(episodes)} demo episodes")

all_obs = []
all_actions = []

for ep_idx, ep_dir in enumerate(episodes):
    npz_path = os.path.join(ep_dir, "data.npz")
    if not os.path.exists(npz_path):
        continue

    data = np.load(npz_path)
    demo_actions_raw = data["actions"]  # (T, 6) — raw servo ticks with homing offset
    T = demo_actions_raw.shape[0]

    # Convert servo ticks → radians
    # STS3215: 4096 ticks = 360 degrees, so tick / 2048 * π = radians
    demo_actions = demo_actions_raw / 2048.0 * np.pi
    demo_actions = demo_actions.astype(np.float32)

    # Reset env
    obs_dict, _ = env.reset()
    ep_obs = []
    ep_act = []

    for t in range(T):
        # Get full rl_policy observation
        rl_obs = env.observation_manager.compute()["rl_policy"]
        ep_obs.append(rl_obs[0].cpu().numpy())

        # Apply demo action (now in radians)
        action = torch.tensor(demo_actions[t:t+1], dtype=torch.float32, device=DEVICE)
        obs_dict, _, _, _, _ = env.step(action)
        ep_act.append(demo_actions[t])

    ep_obs = np.array(ep_obs)
    ep_act = np.array(ep_act)
    all_obs.append(ep_obs)
    all_actions.append(ep_act)

    print(f"  Episode {ep_idx+1}/{len(episodes)}: {T} steps, obs={ep_obs.shape}, act={ep_act.shape}")

# Concatenate all
all_obs = np.concatenate(all_obs, axis=0)
all_actions = np.concatenate(all_actions, axis=0)

print(f"\n[Replay] Total: {all_obs.shape[0]} samples")
print(f"  obs: {all_obs.shape}")
print(f"  actions: {all_actions.shape}")

# Save
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
np.savez(args.save_path, observations=all_obs, actions=all_actions)
print(f"[Replay] Saved: {args.save_path}")

simulation_app.close()
