"""Iteration logger — CSV logging, console output, and checkpoint saving."""

import csv
import json
import os
import shutil

import numpy as np
import torch


class IterLogger:
    """Handles per-iteration CSV logging, console printing, and checkpoint saving."""

    ITER_LOG_BASE_FIELDS = [
        "iter", "episodes", "mean_rew", "max_rew", "alltime_max", "alltime_best_mean",
        "best_lift_cm", "success_count", "success_rate",
        "loss_mean", "curriculum_stage", "cube_mass_kg", "cube_range_cm",
        "push_penalty_total", "push_penalty_last_ep",
        "reach_30", "align_s30", "close_10",
        "grasp_start", "grasp_enough", "grasp",
        "lift_30mm", "lift_100mm", "lift_200mm",
    ]

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self._iter_log_path = os.path.join(log_dir, "iter_log.csv")
        self._iter_log_file = None
        self._iter_csv = None
        self._iter_log_fields = None
        self._milestone_cols = []
        print(f"[Log] Iter log: {self._iter_log_path}", flush=True)

    def build_stats_snapshot(self, iteration, stats, ep_rewards_this_iter,
                             recent_ep_rewards, alltime_max, alltime_best_mean,
                             mean_loss, loss_dict, env, term_snap):
        """Update the shared stats dict with iteration-level info.

        Args:
            iteration: Current iteration index.
            stats: The shared dashboard.train_stats dict.
            ep_rewards_this_iter: List of episode rewards collected this iteration.
            recent_ep_rewards: Rolling list of recent episode rewards.
            alltime_max: All-time max single-episode reward.
            alltime_best_mean: All-time best rolling mean reward.
            mean_loss: Scalar mean loss for this iteration.
            loss_dict: Raw loss dict from PPO update.
            env: The environment (for reward manager access).
            term_snap: Per-term reward snapshot dict.
        """
        mean_rew = np.mean(ep_rewards_this_iter) if ep_rewards_this_iter else 0
        max_rew = np.max(ep_rewards_this_iter) if ep_rewards_this_iter else 0

        stats["iteration"] = iteration + 1
        stats["mean_reward"] = float(sum(recent_ep_rewards) / len(recent_ep_rewards)) if recent_ep_rewards else 0
        stats["max_reward"] = float(max_rew)
        stats["alltime_max_reward"] = float(alltime_max) if alltime_max != float('-inf') else 0
        stats["alltime_best_mean"] = float(alltime_best_mean) if alltime_best_mean != float('-inf') else 0
        stats["rolling_mean_20"] = float(sum(recent_ep_rewards) / len(recent_ep_rewards)) if recent_ep_rewards else 0

        if ep_rewards_this_iter:
            stats["reward_history"].append({"iter": iteration + 1, "mean": float(mean_rew), "max": float(max_rew)})
            stats["reward_history"] = stats["reward_history"][-500:]

        # Enrich reward_terms with milestone hit rates for live display
        _ter = stats.get("term_ep_rates", {})
        _group_map = {
            "reach_stages": "reach_",
            "open_stages": "open_",
            "align_stages": "align_s",
            "close_stages": "close_",
            "lift_hold": "lift_hold_",
        }
        for term_name, prefix in _group_map.items():
            if term_name in term_snap:
                ms_vals = [v for k, v in _ter.items() if k.startswith(prefix)]
                term_snap[term_name]["value"] = round(max(ms_vals, default=0) / 100, 4)
        # Single milestone reward terms: use exact match from term_ep_rates
        for k in term_snap:
            if k in _ter:
                term_snap[k]["value"] = round(_ter[k] / 100, 4)
        stats["reward_terms"] = term_snap

        # Milestone hit counts
        try:
            from packages.sim.env_setup.maniskill_rewards import get_milestone_stats
            stats["milestone_hits"] = get_milestone_stats()
        except Exception:
            stats["milestone_hits"] = {}

        # v5: expose the full loss breakdown from the loss_dict
        if isinstance(loss_dict, dict):
            stats["loss_dict"] = {k: round(float(v), 5) for k, v in loss_dict.items()
                                  if isinstance(v, (int, float))}

    def compute_term_snapshot(self, env):
        """Compute per-term reward snapshot from the environment's reward manager.

        Returns:
            dict mapping term name -> {"weight": float, "value": float}.
        """
        _term_snap = {}
        try:
            rm = env.unwrapped.reward_manager
            for idx, name in enumerate(rm.active_terms):
                cfg = rm.get_term_cfg(name)
                val = 0.0
                try:
                    val = rm._term_values[name].mean().item()
                except Exception:
                    try:
                        val = rm._term_values[:, idx].mean().item() if hasattr(rm._term_values, 'shape') else 0.0
                    except Exception:
                        try:
                            raw = cfg.func(env.unwrapped, **cfg.params)
                            val = (raw.mean().item() * cfg.weight)
                        except Exception:
                            val = 0.0
                _term_snap[name] = {"weight": cfg.weight, "value": round(val, 4)}
        except Exception:
            pass
        return _term_snap

    def write_csv_row(self, iteration, stats, ep_rewards_this_iter,
                      recent_ep_rewards, alltime_max, alltime_best_mean,
                      best_lift, success_count, episode_count, mean_loss):
        """Write one row to the iter log CSV. Re-creates writer when milestone columns change."""
        max_rew = np.max(ep_rewards_this_iter) if ep_rewards_this_iter else 0
        _ter = stats.get("term_ep_rates", {})

        _iter_row = {
            "iter": iteration + 1,
            "episodes": episode_count,
            "mean_rew": round(float(sum(recent_ep_rewards) / len(recent_ep_rewards)) if recent_ep_rewards else 0, 2),
            "max_rew": round(float(max_rew), 2),
            "alltime_max": round(float(alltime_max) if alltime_max != float('-inf') else 0, 2),
            "alltime_best_mean": round(float(alltime_best_mean) if alltime_best_mean != float('-inf') else 0, 2),
            "best_lift_cm": round(best_lift, 2),
            "success_count": success_count,
            "success_rate": round((success_count / max(episode_count, 1)) * 100, 1),
            "loss_mean": round(mean_loss, 5),
            "curriculum_stage": stats.get("curriculum_stage", "stage1"),
            "cube_mass_kg": stats.get("cube_mass_kg", 0),
            "cube_range_cm": stats.get("cube_range_cm", 0),
            "push_penalty_total": round(stats.get("push_penalty_total", 0), 3),
            "push_penalty_last_ep": round(stats.get("push_penalty_last_ep", 0), 3),
        }
        # Add all milestone rates to row
        for mk in sorted(_ter.keys()):
            _iter_row[mk] = _ter.get(mk, 0)

        # CSV: reopen with updated header when new milestones appear
        _current_ms_cols = sorted(_ter.keys())
        if self._iter_csv is None or _current_ms_cols != self._milestone_cols:
            self._milestone_cols = _current_ms_cols
            self._iter_log_fields = self.ITER_LOG_BASE_FIELDS + self._milestone_cols
            if self._iter_log_file is not None:
                self._iter_log_file.close()
            _need_header = not os.path.exists(self._iter_log_path) or self._iter_csv is None
            self._iter_log_file = open(self._iter_log_path, "a", newline="")
            self._iter_csv = csv.DictWriter(self._iter_log_file, fieldnames=self._iter_log_fields, extrasaction="ignore")
            if _need_header:
                self._iter_csv.writeheader()
        self._iter_csv.writerow(_iter_row)
        if (iteration + 1) % 10 == 0:
            self._iter_log_file.flush()

    def console_log(self, iteration, max_iterations, mean_rew, alltime_max,
                    best_lift, success_count, episode_count, stats):
        """Print a summary line to console every 5 iterations."""
        if (iteration + 1) % 5 != 0:
            return
        _am = alltime_max if alltime_max != float('-inf') else 0
        # Show per-term reward values (ManiSkill style)
        _rt = stats.get('reward_terms', {})
        _reach = _rt.get('reaching', {}).get('value', 0) if isinstance(_rt.get('reaching'), dict) else 0
        _grasp = _rt.get('grasped', {}).get('value', 0) if isinstance(_rt.get('grasped'), dict) else 0
        _tbl = _rt.get('table_penalty', {}).get('value', 0) if isinstance(_rt.get('table_penalty'), dict) else 0
        _place = _rt.get('place', {}).get('value', 0) if isinstance(_rt.get('place'), dict) else 0
        print(f"[RL] Iter {iteration+1}/{max_iterations} | "
              f"R={mean_rew:.1f} | max={_am:.1f} | lift={best_lift:.1f}cm | "
              f"success={success_count} | eps={episode_count} | "
              f"reach={_reach:.3f} grasp={_grasp:.3f} place={_place:.3f} tbl={_tbl:.3f}",
              flush=True)

    def save_checkpoint(self, iteration, runner, stats):
        """Save model checkpoint and stats sidecar every 100 iterations."""
        if (iteration + 1) % 100 != 0:
            return
        try:
            ckpt_path = os.path.join(self.log_dir, f"model_{iteration+1}.pt")
            latest_path = os.path.join(self.log_dir, "model_latest.pt")
            # Direct save to avoid Logger bug
            _policy = runner.alg.get_policy() if hasattr(runner.alg, "get_policy") else runner.alg.policy
            torch.save({
                "actor_state_dict": _policy.state_dict(),
                "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                "iter": iteration + 1,
                "infos": {},
            }, ckpt_path)
            # Copy to model_latest.pt (avoid runner.save twice - Logger bug)
            shutil.copy2(ckpt_path, latest_path)
            # Save stats sidecar (JSON) for resume
            _save_stats = {k: v for k, v in stats.items()
                          if not k.startswith("current_")}
            for sp in [ckpt_path.replace(".pt", "_stats.json"),
                       latest_path.replace(".pt", "_stats.json")]:
                with open(sp, "w") as f:
                    json.dump(_save_stats, f)
            print(f"[RL] Saved: {ckpt_path}", flush=True)
        except Exception as e:
            print(f"[RL] Checkpoint error: {e}", flush=True)
