"""Episode tracker — vectorized tensor operations for fast multi-env processing."""

import torch
import numpy as np

from packages.sim.env_setup import CUBE_INITIAL_HEIGHT


class EpisodeTracker:
    """Tracks per-environment episode state using vectorized tensor ops."""

    REACH_DIST_M = 0.05
    REACH_HOLD_STEPS = 60

    def __init__(self, num_envs, stats, dc_cfg, device="cuda:0"):
        self.num_envs = num_envs
        self.stats = stats
        self.device = device

        # Data collection thresholds
        self.HOLD_STABLE_HEIGHT = dc_cfg.get("hold_stable_height", 0.10)
        self.HOLD_STABLE_GRASP_THR = dc_cfg.get("grasp_threshold", 0.26)
        self.HOLD_STABLE_DIST_THR = 0.05
        self.HOLD_STABLE_MIN_STEPS = dc_cfg.get("hold_stable_min_steps", 10)
        self.CONTACT_FORCE_MIN = dc_cfg.get("contact_force_min", 0.1)

        # ── Tensor buffers (GPU) ──
        self.ep_reward_buf = torch.zeros(num_envs, device=device)
        self.ep_hold_count = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.ep_step_buf = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.ep_near_steps = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.ep_max_lift = torch.zeros(num_envs, device=device)
        self.ep_push_penalty = torch.zeros(num_envs, device=device)
        self._cube_initial_xy = torch.zeros(num_envs, 2, device=device)
        self._cube_xy_initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self._success_marked = torch.zeros(num_envs, device=device, dtype=torch.bool)

        # Per-env sets (must stay Python — variable-length per env)
        self.ep_term_fired = {i: set() for i in range(num_envs)}

        # Data collection buffers (Python — variable-length lists)
        self.ep_cam_buffer = {i: [] for i in range(num_envs)}
        self.ep_state_buffer = {i: [] for i in range(num_envs)}

        # Global counters
        self.success_count = stats.get("success_count", 0)
        self.episode_count = stats.get("episode_count", 0)
        self.best_lift = stats.get("best_lift_cm", 0.0)
        self._alltime_max = stats.get("alltime_max_reward", float('-inf'))
        self._alltime_best_mean = stats.get("alltime_best_mean", float('-inf'))
        self._recent_ep_rewards = []
        self._stage_counts = stats.get("stage_counts", {
            "reach": 0, "align": 0, "close": 0, "grasp_start": 0,
            "grasp_enough": 0, "grasp": 0, "lift": 0, "success": 0,
        })
        stats.setdefault("stage_counts", self._stage_counts)
        stats.setdefault("stage_rates", {"reach": 0, "grasp": 0, "lift": 0, "success": 0})
        stats.setdefault("alltime_max_reward", 0)
        stats.setdefault("alltime_best_mean", 0)

        # Penalty totals
        self._push_penalty_total = stats.get("push_penalty_total", 0.0)
        self._push_penalty_episodes = stats.get("push_penalty_episodes", 0)

        # Per-term episode counting
        self._term_ep_counts = stats.get("term_ep_counts", {})
        stats["term_ep_counts"] = self._term_ep_counts
        self._term_reward_sums = {}
        self._term_reward_counts = {}

        # Sensor references
        self.has_contact_sensor = False
        self.contact_sensor = None
        self.jaw_sensor = None

        # ── Best-of-N episode saving ──
        self.SAVE_EVERY_N_EPS = 300
        self._window_ep_count = 0
        self._window_best_lift = -1.0
        self._window_best_data = None  # {env_id, frames: [...], reward, lift_cm, milestones}
        # Per-env GPU frame buffer: accumulate during episode
        self._ep_frame_buf = {i: [] for i in range(num_envs)}

    def set_sensors(self, contact_sensor, jaw_sensor, has_contact_sensor):
        self.contact_sensor = contact_sensor
        self.jaw_sensor = jaw_sensor
        self.has_contact_sensor = has_contact_sensor

    # ── Vectorized contact helpers ──

    # ── Scalar contact helpers (GPU contact filter) ──

    def get_contact_force(self, env_id):
        """Get pair-wise contact force from force_matrix_w."""
        return 0.0  # Not used in ManiSkill reward system

    def is_both_jaws_contact(self, env_id, min_force=0.5):
        """Not used in ManiSkill reward system."""
        return False

    # ── Exploration noise ──

    def apply_exploration_noise(self, runner, stats, iteration, start_iteration):
        _explore_scale = stats.pop("explore_noise_scale", None)
        if _explore_scale is None:
            return
        try:
            _p = runner.alg.get_policy() if hasattr(runner.alg, 'get_policy') else getattr(runner.alg, 'policy', None)
            if _p and hasattr(_p, 'std'):
                _base_std = getattr(_p, '_base_std', None)
                if _base_std is None:
                    _p._base_std = _p.std.data.clone()
                    _base_std = _p._base_std
                _p.std.data[:] = (_base_std * _explore_scale).clamp(min=0.01, max=10.0)
                if not hasattr(_p, '_last_explore_log') or abs(_p._last_explore_log - _explore_scale) > 0.05:
                    _p._last_explore_log = _explore_scale
                    print(f"\n[Explore] noise={_explore_scale:.2f}x (std={_p.std.data.mean().item():.3f})", flush=True)
        except Exception as _e:
            if iteration == start_iteration:
                print(f"[Explore] skip: {_e}", flush=True)

    # ── VECTORIZED per-step processing ──

    def process_step_vectorized(self, env, rewards, step, iteration, start_iteration, dashboard):
        """Process ALL envs in one call using tensor operations. No Python for-loop."""
        N = self.num_envs

        # Accumulate rewards (all envs at once)
        self.ep_reward_buf += rewards.squeeze()
        self.ep_step_buf += 1

        # Get all positions as tensors (already batched in Isaac Lab)
        cube_pos = env.scene["cube"].data.root_pos_w          # (N, 3)
        ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 1] # (N, 3)
        gripper = env.scene["robot"].data.joint_pos[:, -1]      # (N,)

        # Distances and heights (vectorized)
        diff = cube_pos - ee_pos                                # (N, 3)
        dist = diff.norm(dim=-1)                                # (N,)
        cube_h = cube_pos[:, 2]                                 # (N,)
        lift_cm = (cube_h - CUBE_INITIAL_HEIGHT) * 100          # (N,)

        # Pair-wise contact forces via force_matrix_w (GPU native)
        try:
            grip_fm = env.scene["gripper_contact"].data.force_matrix_w
            jaw_fm = env.scene["jaw_contact"].data.force_matrix_w
            grip_force = grip_fm[:, 0, 0, :].norm(dim=-1)
            jaw_force = jaw_fm[:, 0, 0, :].norm(dim=-1)
            both_contact = (grip_force >= 0.5) & (jaw_force >= 0.5)
            gripper_force = torch.max(grip_force, jaw_force)
        except Exception:
            both_contact = torch.zeros(N, device=self.device, dtype=torch.bool)
            gripper_force = torch.zeros(N, device=self.device)

        # ── Max lift tracking ──
        # Only update for envs that have grasp milestones
        # (ep_term_fired check must stay Python, but we batch the update)
        # Vectorized grasp milestone check — no Python per-env loop
        try:
            from packages.sim.env_setup import _milestones
            grasp_mm_keys = [k for k in _milestones if k.startswith('grasp_') and k.endswith('mm')]
            if grasp_mm_keys:
                has_grasp = torch.zeros(N, device=self.device, dtype=torch.bool)
                for k in grasp_mm_keys:
                    v = _milestones[k]
                    if v.numel() >= N:
                        has_grasp |= v[:N]
            else:
                has_grasp = torch.zeros(N, device=self.device, dtype=torch.bool)
        except Exception:
            has_grasp = torch.zeros(N, device=self.device, dtype=torch.bool)
            for i in range(N):
                if any(k.startswith('grasp_') and k.endswith('mm') for k in self.ep_term_fired[i]):
                    has_grasp[i] = True
        self.ep_max_lift = torch.where(has_grasp, torch.max(self.ep_max_lift, lift_cm), self.ep_max_lift)

        # ── Side approach penalty (vectorized) ──

        # ── Push penalty (vectorized) ──
        cube_xy = cube_pos[:, :2]                               # (N, 2)
        first_step = self.ep_step_buf <= 1
        needs_init = first_step | ~self._cube_xy_initialized
        if needs_init.any():
            self._cube_initial_xy[needs_init] = cube_xy[needs_init].clone()
            self._cube_xy_initialized[needs_init] = True
        push_d = (cube_xy - self._cube_initial_xy).norm(dim=-1)  # (N,)
        push_mask = push_d > 0.02
        if push_mask.any():
            push_sev = ((push_d - 0.02) / 0.03).clamp(0, 1)
            self.ep_push_penalty += torch.where(push_mask, push_sev * -0.01, torch.zeros_like(push_sev))

        # ── Near steps ──
        self.ep_near_steps += (dist < self.REACH_DIST_M).int()

        # ── Milestone tracking (Python — variable-length sets) ──
        try:
            from packages.sim.env_setup import _milestones
            for key, val in _milestones.items():
                if val.numel() >= N:
                    fired_envs = val[:N].nonzero(as_tuple=True)[0]
                    for eid in fired_envs.tolist():
                        self.ep_term_fired[eid].add(key)
        except Exception as _me:
            if iteration == start_iteration and step == 0:
                print(f"[WARN] Milestone tracking error: {_me}", flush=True)

        # ── Success — ManiSkill exact ──
        import numpy as _np
        cube_lifted = cube_h >= 0.016  # cube_half_size + 1mm
        _rest = torch.tensor([0, 0, 0, _np.pi/2, _np.pi/2], device=self.device)
        _cur_qpos = env.scene["robot"].data.joint_pos[:, :-1]
        _dist_rest = torch.linalg.norm(_cur_qpos - _rest, dim=-1)
        _reached_rest = _dist_rest < 0.2
        new_success = (cube_lifted & both_contact & _reached_rest) & ~self._success_marked

        if new_success.any():
            n_new = new_success.sum().item()
            self.success_count += n_new
            self._success_marked |= new_success
            self.ep_hold_count = torch.where(new_success,
                                             torch.full_like(self.ep_hold_count, -9999),
                                             self.ep_hold_count)
            with dashboard.stats_lock:
                self.stats["success_count"] = self.success_count

        # ── Dashboard update (last env only, avoid lock contention) ──
        last_id = N - 1
        with dashboard.stats_lock:
            self.stats["current_phase"] = "training"
            self.stats["current_lift_cm"] = lift_cm[last_id].item()
            self.stats["current_contact_force"] = gripper_force[last_id].item()

        # ── Debug log (only significant rewards, sampled) ──
        abs_rew = rewards.squeeze().abs()
        big_mask = abs_rew > 1.0
        if False:  # REW_DBG disabled
            # Log only the first one to avoid spam
            eid = big_mask.nonzero(as_tuple=True)[0][0].item()
            print(f"[REW_DBG] env={eid} step_rew={rewards[eid].item():.2f} "
                  f"ep_total={self.ep_reward_buf[eid].item():.2f}", flush=True)

    def collect_state_data_batch(self, env, actions, cam_map):
        """Buffer state as GPU tensors for ALL envs. No GPU->CPU transfer.
        Best episode is saved to disk every SAVE_EVERY_N_EPS episodes."""
        try:
            # Store as GPU tensor slices (no CPU transfer = fast)
            frame = {
                "joint_pos": env.scene["robot"].data.joint_pos.clone(),      # (N, J)
                "joint_vel": env.scene["robot"].data.joint_vel.clone(),      # (N, J)
                "action": actions.clone(),                                    # (N, A)
                "cube_pos": env.scene["cube"].data.root_pos_w.clone(),       # (N, 3)
                "ee_pos": env.scene["ee_frame"].data.target_pos_w[:, 1].clone(),  # (N, 3)
            }
            for env_id in range(self.num_envs):
                self._ep_frame_buf[env_id].append({
                    k: v[env_id] for k, v in frame.items()
                })
        except Exception:
            pass

    # ── Episode end (per-env, called only for done envs) ──

    def end_episode(self, env_id, dashboard):
        ep_rew = self.ep_reward_buf[env_id].item()
        lift_cm_end = self.ep_max_lift[env_id].item()
        was_success = self._success_marked[env_id].item()

        ep_info = {
            "reward": ep_rew,
            "lift_cm": lift_cm_end,
            "hold_steps": max(0, self.ep_hold_count[env_id].item()) if not was_success else self.HOLD_STABLE_MIN_STEPS,
            "success": was_success,
        }
        self.episode_count += 1
        if lift_cm_end > self.best_lift:
            self.best_lift = lift_cm_end

        if ep_rew > self._alltime_max:
            self._alltime_max = ep_rew
        self._recent_ep_rewards.append(ep_rew)
        self._recent_ep_rewards[:] = self._recent_ep_rewards[-20:]
        rolling_mean = sum(self._recent_ep_rewards) / len(self._recent_ep_rewards)
        if rolling_mean > self._alltime_best_mean:
            self._alltime_best_mean = rolling_mean

        for name in self.ep_term_fired[env_id]:
            self._term_ep_counts[name] = self._term_ep_counts.get(name, 0) + 1

        with dashboard.stats_lock:
            self.stats["recent_episodes"].append(ep_info)
            self.stats["recent_episodes"] = self.stats["recent_episodes"][-30:]
            self.stats["episode_count"] = self.episode_count
            self.stats["best_lift_cm"] = self.best_lift
            self.stats["success_rate_pct"] = (self.success_count / max(self.episode_count, 1)) * 100
            self.stats["term_ep_counts"] = self._term_ep_counts
            ec = max(self.episode_count, 1)
            self.stats["term_ep_rates"] = {k: round(v / ec * 100, 1)
                                           for k, v in self._term_ep_counts.items()}
            # Push penalty stats
            push_p = self.ep_push_penalty[env_id].item()
            if push_p < -0.001:
                self._push_penalty_episodes += 1
            self._push_penalty_total += push_p
            self.stats["push_penalty_total"] = round(self._push_penalty_total, 2)
            self.stats["push_penalty_episodes"] = self._push_penalty_episodes
            self.stats["push_penalty_ep_avg"] = round(self._push_penalty_total / max(self.episode_count, 1), 3)
            self.stats["push_penalty_last_ep"] = round(push_p, 3)

        # ── Best-of-N tracking ──
        self._window_ep_count += 1
        if lift_cm_end > self._window_best_lift:
            self._window_best_lift = lift_cm_end
            # Save frames (GPU→CPU) only for new best
            frames_cpu = []
            for f in self._ep_frame_buf.get(env_id, []):
                frames_cpu.append({k: v.cpu().numpy().copy() for k, v in f.items()})
            self._window_best_data = {
                "frames": frames_cpu,
                "reward": ep_rew,
                "lift_cm": lift_cm_end,
                "success": was_success,
                "milestones": list(self.ep_term_fired[env_id]),
                "episode": self.episode_count,
                "cube_initial_pos": self._ep_frame_buf.get(env_id, [{}])[0].get("cube_pos", torch.zeros(3)).cpu().numpy().tolist() if len(self._ep_frame_buf.get(env_id, [])) > 0 else [0,0,0],
                "cube_mass_kg": self.stats.get("cube_mass_kg", 0),
                "cube_range_cm": self.stats.get("cube_range_cm", 0),
                "iteration": self.stats.get("iteration", 0),
            }

        if self._window_ep_count >= self.SAVE_EVERY_N_EPS:
            self._save_best_episode()

        # Track per-term reward averages
        try:
            rt = self.stats.get("reward_terms", {})
            for k, v in rt.items():
                val = v.get("value", 0)
                if k not in self._term_reward_sums:
                    self._term_reward_sums[k] = 0.0
                    self._term_reward_counts[k] = 0
                self._term_reward_sums[k] += val
                self._term_reward_counts[k] += 1
            self.stats["term_reward_avgs"] = {
                k: round(self._term_reward_sums[k] / max(self._term_reward_counts[k], 1), 4)
                for k in self._term_reward_sums
            }
        except Exception:
            pass
        return ep_rew, ep_info

    def _save_best_episode(self):
        """Save best episode of the window to disk and reset."""
        import json, os
        import numpy as np
        if self._window_best_data is None or not self._window_best_data["frames"]:
            self._window_ep_count = 0
            self._window_best_lift = -1.0
            self._window_best_data = None
            return
        try:
            d = self._window_best_data
            save_dir = f"/data/rl_dopamine/episodes/best_ep_{d['episode']:06d}_lift{d['lift_cm']:.0f}mm"
            os.makedirs(save_dir, exist_ok=True)
            # Meta
            meta = {k: v for k, v in d.items() if k != "frames"}
            meta["window_size"] = self.SAVE_EVERY_N_EPS
            with open(os.path.join(save_dir, "meta.json"), "w") as f:
                json.dump(meta, f)
            # Frames
            for i, frame in enumerate(d["frames"]):
                np.savez_compressed(os.path.join(save_dir, f"frame_{i:04d}.npz"), **frame)
            print(f"\n[BEST] Saved best of {self.SAVE_EVERY_N_EPS} eps: "
                  f"lift={d['lift_cm']:.1f}cm reward={d['reward']:.1f} "
                  f"({len(d['frames'])} frames) → {save_dir}", flush=True)
        except Exception as e:
            print(f"[BEST] Save error: {e}", flush=True)
        # Reset window
        self._window_ep_count = 0
        self._window_best_lift = -1.0
        self._window_best_data = None

    def reset_episode(self, env_id):
        self.ep_reward_buf[env_id] = 0.0
        self.ep_step_buf[env_id] = 0
        self.ep_hold_count[env_id] = 0
        self.ep_near_steps[env_id] = 0
        self.ep_max_lift[env_id] = 0.0
        self.ep_push_penalty[env_id] = 0.0
        self._cube_initial_xy[env_id] = 0.0
        self._cube_xy_initialized[env_id] = False
        self._success_marked[env_id] = False
        self.ep_term_fired[env_id] = set()
        self.ep_cam_buffer[env_id] = []
        self.ep_state_buffer[env_id] = []
        self._ep_frame_buf[env_id] = []
