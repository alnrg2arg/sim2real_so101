#!/usr/bin/env python3
"""Replay saved episodes by feeding recorded actions into the environment.
Uses GPU:1 so training on GPU:0 continues undisturbed.

Usage: /isaac-sim/python.sh replay_episode.py --episode /data/replay_episodes/lift9cm_ep_001030
       /isaac-sim/python.sh replay_episode.py --all /data/replay_episodes
"""
import argparse
import os
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=str, default=None, help="Single episode dir")
parser.add_argument("--all", type=str, default=None, help="Dir containing multiple episode dirs")
parser.add_argument("--output-dir", type=str, default="/data/replay_output", help="Output directory for mp4s")
parser.add_argument("--gpu", type=int, default=1, help="Physical GPU index")
args = parser.parse_args()

DEVICE = f"cuda:{args.gpu}"

# Collect episodes to replay
episodes = []
if args.all:
    for d in sorted(os.listdir(args.all)):
        ep_dir = os.path.join(args.all, d)
        if os.path.isdir(ep_dir) and os.path.exists(os.path.join(ep_dir, "meta.json")):
            episodes.append(ep_dir)
elif args.episode:
    episodes.append(args.episode)

if not episodes:
    print("No episodes found!")
    sys.exit(1)

# Sort by lift height descending
def get_lift(ep_dir):
    try:
        with open(os.path.join(ep_dir, "meta.json")) as f:
            return json.load(f).get("lift_cm", 0)
    except:
        return 0
episodes.sort(key=get_lift, reverse=True)

print(f"Will replay {len(episodes)} episodes:")
for ep in episodes:
    print(f"  {os.path.basename(ep)}: {get_lift(ep):.1f}cm")

import numpy as np
import torch

# Setup Isaac Sim on GPU:1
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"enable_cameras": True})
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import TiledCameraCfg

# Use same imports as training
sys.path.insert(0, "/workspace/delivery")
import leisaac.tasks.lift_cube  # registers the task
from packages.sim.env_setup import configure_env, set_cube_mass
from leisaac.tasks.lift_cube import mdp
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
import yaml

config_path = "/workspace/delivery/configs/reward_config.yaml"
CFG = {}
if os.path.exists(config_path):
    with open(config_path) as f:
        CFG = yaml.safe_load(f) or {}

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=1)

env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg)
env_cfg.sim.render_interval = 1

env = ManagerBasedRLEnv(cfg=env_cfg)
print(f"Env ready: act={env.action_space}")

os.makedirs(args.output_dir, exist_ok=True)

# Replay each episode
for ep_idx, ep_dir in enumerate(episodes):
    ep_name = os.path.basename(ep_dir)
    out_path = os.path.join(args.output_dir, f"{ep_name}.mp4")

    # Load frames
    frames_data = []
    fi = 0
    while True:
        fp = os.path.join(ep_dir, f"frame_{fi:04d}.npz")
        if not os.path.exists(fp):
            break
        frames_data.append(np.load(fp))
        fi += 1

    if not frames_data:
        print(f"  {ep_name}: no frames, skip")
        continue

    actions = [f["action"] for f in frames_data]
    has_step = "step" in frames_data[0] if frames_data else False
    steps = [int(f["step"]) for f in frames_data] if has_step else list(range(0, len(frames_data) * 4, 4))
    step_gaps = [steps[i+1] - steps[i] for i in range(len(steps)-1)]
    gap = max(1, int(np.median(step_gaps))) if step_gaps else 4

    print(f"\n[{ep_idx+1}/{len(episodes)}] {ep_name}: {len(actions)} frames, gap={gap}, lift={get_lift(ep_dir):.1f}cm")

    # Reset env
    env.reset()
    for _ in range(3):
        env.step(torch.zeros(1, env.action_space.shape[-1], device=DEVICE))

    # Replay and capture
    cam = env.scene["front"]
    video_frames = []

    for i, act in enumerate(actions):
        act_tensor = torch.tensor(act, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        for _ in range(gap):
            env.step(act_tensor)

        rgb = cam.data.output.get("rgb")
        if rgb is not None:
            img = rgb[0, :, :, :3].cpu().numpy()
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            video_frames.append(img)

    # Save mp4
    if video_frames:
        import tempfile, shutil
        tmp = tempfile.mkdtemp()
        for i, fr in enumerate(video_frames):
            from PIL import Image
            Image.fromarray(fr).save(os.path.join(tmp, f"{i:04d}.png"))
        os.system(f"ffmpeg -y -framerate 15 -i {tmp}/%04d.png -c:v libx264 -pix_fmt yuv420p {out_path} 2>/dev/null")
        shutil.rmtree(tmp)
        sz = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        print(f"  Saved: {out_path} ({sz//1024}KB, {len(video_frames)} frames)")
    else:
        print(f"  No video frames captured!")

print(f"\nDone! All videos in {args.output_dir}")
env.close()
