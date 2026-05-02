"""Episode data saver — saves camera/state frames and metadata to disk."""

import json
import shutil
from pathlib import Path

import numpy as np


def save_episode_data(save_dir, episode_count, env_id, tracker, iteration, config_path):
    """Save episode data (frames + metadata) for a completed episode.

    Args:
        save_dir: Path to base episodes directory.
        episode_count: Global episode counter (used in folder name).
        env_id: Environment index.
        tracker: EpisodeTracker instance holding per-env buffers.
        iteration: Current training iteration.
        config_path: Path to the reward config YAML to copy alongside data.
    """
    was_success = tracker._success_marked[env_id].item()
    ep_term_fired = tracker.ep_term_fired[env_id]
    ep_cam_buffer = tracker.ep_cam_buffer[env_id]
    ep_state_buffer = tracker.ep_state_buffer[env_id]

    _has_grasp = "grasp_enough" in ep_term_fired
    _has_data = len(ep_cam_buffer) > 0 or len(ep_state_buffer) > 0
    _should_save = (was_success or _has_grasp) and _has_data
    if not _should_save:
        return

    try:
        _tag = "success" if was_success else (
            "lift" if any(k.startswith("lift_") for k in ep_term_fired) else "grasp"
        )
        ep_save_dir = Path(save_dir) / f"{_tag}_ep_{episode_count:06d}"
        ep_save_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        meta = {
            "episode": episode_count,
            "reward": tracker.ep_reward_buf[env_id].item(),
            "lift_cm": tracker.ep_max_lift[env_id].item(),
            "success": was_success,
            "steps": tracker.ep_step_buf[env_id].item(),
            "milestones": list(ep_term_fired),
            "iteration": iteration,
        }
        with open(ep_save_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        # Save config for replay
        shutil.copy2(config_path, str(ep_save_dir / "reward_config.yaml"))

        # Save frames (joint_pos, action, cam images or state-only)
        _frames = ep_cam_buffer if ep_cam_buffer else ep_state_buffer
        for fi, frame in enumerate(_frames):
            frame_path = ep_save_dir / f"frame_{fi:04d}.npz"
            save_dict = {}
            for k, v in frame.items():
                if isinstance(v, np.ndarray):
                    save_dict[k] = v
                else:
                    save_dict[k] = np.array(v)
            np.savez_compressed(str(frame_path), **save_dict)

        _mode = "cam+state" if ep_cam_buffer else "state-only"
        print(f"[DATA] Saved {_tag} episode {episode_count} ({len(_frames)} frames, {_mode}) -> {ep_save_dir}", flush=True)
    except Exception as e:
        print(f"[DATA] Save error: {e}", flush=True)
