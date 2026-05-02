#!/usr/bin/env python3
"""Convert replay_episodes.py output (npz frames) → LeRobot v2 dataset.

Usage:
    python convert_replay_to_lerobot.py \
        --replay-dir /data/rl_output/replay \
        --output-dir /data/lerobot_dataset/sim_lift \
        --repo-id local/sim_lift \
        --fps 30 \
        --cameras front side
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--replay-dir", type=str, default="/data/rl_output/replay")
    p.add_argument("--output-dir", type=str, default="/data/lerobot_dataset/sim_lift")
    p.add_argument("--repo-id", type=str, default="local/sim_lift")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cameras", nargs="+", default=["front", "side"],
                   help="Which cameras to include (front, side, wrist)")
    p.add_argument("--image-size", type=int, nargs=2, default=[256, 256],
                   help="Resize images to [H, W]")
    p.add_argument("--min-reward", type=float, default=0.0,
                   help="Only include episodes with total reward > this")
    return p.parse_args()


def build_features(cameras, image_size, state_dim=6, action_dim=6):
    """Build LeRobot feature dict for dataset creation."""
    features = {}

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, image_size[0], image_size[1]),
            "names": ["channels", "height", "width"],
        }

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": ["joint_positions"],
    }

    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": ["joint_actions"],
    }

    return features


def load_episode_frames(ep_dir):
    """Load all npz frames from an episode directory, sorted by frame number."""
    frame_files = sorted(ep_dir.glob("frame_*.npz"))
    frames = []
    for f in frame_files:
        data = dict(np.load(f, allow_pickle=True))
        frames.append(data)
    return frames


def main():
    args = parse_args()

    replay_dir = Path(args.replay_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find episodes
    episode_dirs = sorted([d for d in replay_dir.iterdir()
                           if d.is_dir() and d.name.startswith("episode_")])
    print(f"Found {len(episode_dirs)} episodes in {replay_dir}")

    # Filter by reward
    valid_episodes = []
    for ep_dir in episode_dirs:
        meta_path = ep_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("total_reward", 0) > args.min_reward:
                valid_episodes.append(ep_dir)
                print(f"  {ep_dir.name}: reward={meta['total_reward']:.1f} lift={meta.get('final_lift_cm', 0):.1f}cm -> OK")
            else:
                print(f"  {ep_dir.name}: reward={meta['total_reward']:.1f} -> SKIP (below {args.min_reward})")
        else:
            valid_episodes.append(ep_dir)
            print(f"  {ep_dir.name}: no meta.json -> including")

    print(f"\n{len(valid_episodes)} episodes pass filter")
    if not valid_episodes:
        print("No valid episodes. Exiting.")
        return

    # Probe first frame for dimensions
    probe = dict(np.load(next(valid_episodes[0].glob("frame_*.npz"))))
    state_dim = probe["joint_pos"].shape[0]
    action_dim = probe["action"].shape[0]
    print(f"state_dim={state_dim}, action_dim={action_dim}")

    # Create dataset
    features = build_features(args.cameras, args.image_size, state_dim, action_dim)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=output_dir,
        robot_type="so101_follower",
        features=features,
    )

    # Process episodes
    for ep_idx, ep_dir in enumerate(valid_episodes):
        frames = load_episode_frames(ep_dir)
        print(f"\nEpisode {ep_idx}: {ep_dir.name} ({len(frames)} frames)")

        for fi, frame_data in enumerate(frames):
            row = {}

            # State
            row["observation.state"] = torch.from_numpy(
                frame_data["joint_pos"].astype(np.float32)
            )

            # Action
            row["action"] = torch.from_numpy(
                frame_data["action"].astype(np.float32)
            )

            # Camera images
            for cam in args.cameras:
                key = f"cam_{cam}"
                if key in frame_data:
                    img = frame_data[key][:, :, :3]  # RGB, drop alpha if present
                    img_pil = Image.fromarray(img.astype(np.uint8))
                    img_pil = img_pil.resize(
                        (args.image_size[1], args.image_size[0]),
                        Image.BILINEAR,
                    )
                    row[f"observation.images.{cam}"] = img_pil

            dataset.add_frame(row)

        dataset.save_episode(task="lift_toothpaste")
        print(f"  Saved episode {ep_idx}")

    # Finalize
    dataset.consolidate()
    print(f"\nDataset saved to {output_dir}")
    print(f"  Episodes: {len(valid_episodes)}")
    print(f"  Repo ID: {args.repo_id}")


if __name__ == "__main__":
    main()
