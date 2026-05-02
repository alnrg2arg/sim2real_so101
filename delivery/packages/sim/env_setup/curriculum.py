"""Curriculum manager: position, mass, effort, exploration, episode length."""

from .config import (
    DEG2RAD, REAL_EFFORT_NM, ARM_SELF_WEIGHT_NM, EMA_BETA, USE_DOPAMINE,
)
from .helpers import _current_mass, _hit_rates_ema


class CurriculumManager:
    """Manages position randomization, mass curriculum, effort scheduling."""

    def __init__(self, cfg, env):
        self.env = env
        self._pos_idx = -1
        self._mass_idx = -1
        self._lateral_disabled = False
        self._episode_length = cfg.get("episode_length_s", 15.0)

        # Plateau detection + entropy boost
        self._plateau_window = 100  # check every 100 iters
        self._plateau_last_max = float('-inf')
        self._plateau_stuck_count = 0
        self._entropy_boosted = False
        self._entropy_boost_until = 0
        self._base_entropy = cfg.get("ppo", {}).get("entropy_coef", 0.01)

        self.pos_stages = []
        for s in cfg.get("position_curriculum", []):
            self.pos_stages.append((s["until_iter"], s["xy_cm"], s.get("yaw_deg", 10), s["label"]))

        self.mass_stages = []
        for s in cfg.get("mass_curriculum", []):
            self.mass_stages.append((s["until_iter"], s["mass"], s["label"]))

    def step(self, iteration, stats):
        self._update_hit_rates(stats)
        self._apply_position(iteration, stats)
        self._apply_mass(iteration, stats)
        self._apply_effort(stats)
        self._apply_exploration(stats)
        self._apply_episode_length(stats)
        self._check_plateau(iteration, stats)

    # ── Episode length ──

    def _apply_episode_length(self, stats):
        """Set episode length from config."""
        target = self._episode_length
        current = self.env.cfg.episode_length_s
        if abs(current - target) > 0.1:
            self.env.cfg.episode_length_s = target
            print(f"\n[Episode Length] {current:.0f}s -> {target:.0f}s (fixed)", flush=True)

    # ── Position curriculum ──

    def _set_pose_range(self, xy_cm, yaw_deg):
        xy_m = xy_cm / 100.0
        yaw_rad = yaw_deg * DEG2RAD
        try:
            for term in self.env.event_manager._mode_term_cfgs.get("reset", []):
                if hasattr(term, 'params') and 'pose_range' in getattr(term, 'params', {}):
                    term.params["pose_range"]["x"] = (-xy_m, xy_m)
                    term.params["pose_range"]["y"] = (-xy_m, xy_m)
                    term.params["pose_range"]["yaw"] = (-yaw_rad, yaw_rad)
        except Exception:
            pass

    def _apply_position(self, iteration, stats):
        if not self.pos_stages:
            return
        # Success-based curriculum: advance after 50 successes per stage
        SUCCESSES_PER_STAGE = 50
        if not hasattr(self, '_pos_stage_success'):
            self._pos_stage_success = 0
            self._pos_last_success = stats.get("success_count", 0)
            self._pos_idx = 0
            # Apply first stage
            _, xy_cm, yaw_deg, label = self.pos_stages[0]
            self._set_pose_range(xy_cm, yaw_deg)
            print(f"\n[Position] Stage 1: {label} (0/{SUCCESSES_PER_STAGE} successes)", flush=True)
            stats["curriculum_stage"] = "stage1"
            stats["cube_range_cm"] = xy_cm

        current_success = stats.get("success_count", 0)
        new_suc = current_success - self._pos_last_success
        if new_suc > 0:
            self._pos_stage_success += new_suc
            self._pos_last_success = current_success

        # Advance stages
        while self._pos_stage_success >= SUCCESSES_PER_STAGE and self._pos_idx < len(self.pos_stages) - 1:
            self._pos_stage_success -= SUCCESSES_PER_STAGE
            self._pos_idx += 1
            _, xy_cm, yaw_deg, label = self.pos_stages[self._pos_idx]
            self._set_pose_range(xy_cm, yaw_deg)
            print(f"\n[Position] Stage {self._pos_idx+1}: {label} (50 successes reached!)", flush=True)
            stats["curriculum_stage"] = f"stage{self._pos_idx+1}"
            stats["cube_range_cm"] = xy_cm

    # ── Hit rates ──

    def _update_hit_rates(self, stats):
        """Update EMA hit rates for dopamine scaling and dashboard."""
        from . import config as cfg
        rates = stats.get("term_ep_rates", {})
        if cfg.USE_DOPAMINE:
            beta = EMA_BETA
            for k, v in rates.items():
                current = v / 100.0
                prev = _hit_rates_ema.get(k, current)
                _hit_rates_ema[k] = beta * prev + (1.0 - beta) * current
        else:
            for k, v in rates.items():
                _hit_rates_ema[k] = v / 100.0

    # ── Mass curriculum ──

    def _lifts_for_mass(self, mass_kg):
        """Heavier=fewer lifts (10kg->10), lighter=more lifts (0.2kg->100)."""
        return int(10 + (10.0 - mass_kg) / (10.0 - 0.2) * 90)

    def _apply_mass(self, iteration, stats):
        if not self.mass_stages:
            return
        from .env_config import set_cube_mass, _set_robot_effort
        if not hasattr(self, '_mass_lift_count'):
            self._mass_lift_count = 0
            self._mass_last_success = 0
        _, cur_mass, _ = self.mass_stages[self._mass_idx]
        lifts_per_mass = self._lifts_for_mass(cur_mass)
        current_success = stats.get("success_count", 0)
        new_lifts = current_success - self._mass_last_success
        if new_lifts > 0:
            self._mass_lift_count += new_lifts
            self._mass_last_success = current_success

        next_idx = self._mass_idx
        while self._mass_lift_count >= lifts_per_mass and next_idx < len(self.mass_stages) - 1:
            self._mass_lift_count -= lifts_per_mass
            next_idx += 1
        if next_idx != self._mass_idx:
            self._mass_idx = next_idx
            _, mass_kg, label = self.mass_stages[self._mass_idx]
            set_cube_mass(self.env, mass_kg)
            _current_mass["kg"] = mass_kg
            self._mass_random = False
            print(f"\n[Mass Curriculum] {label} (after {lifts_per_mass} lifts)", flush=True)
            stats["cube_mass_kg"] = mass_kg
        elif self._mass_idx == len(self.mass_stages) - 1 and self._mass_lift_count >= lifts_per_mass:
            if not getattr(self, '_mass_random', False):
                self._mass_random = True
                print(f"\n[Mass Curriculum] RANDOM 0.2~10kg", flush=True)
            import random
            _masses  = [0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            _weights = [30,  20, 15, 10, 6, 4, 3, 3, 3, 2, 2, 2]
            rand_mass = random.choices(_masses, weights=_weights, k=1)[0]
            set_cube_mass(self.env, rand_mass)
            _current_mass["kg"] = rand_mass
            full_effort = max(REAL_EFFORT_NM, rand_mass * 9.81 * 0.25 + REAL_EFFORT_NM)
            _set_robot_effort(self.env, full_effort)
            stats["cube_mass_kg"] = rand_mass
        elif self._mass_idx == 0 and not hasattr(self, '_mass_init_done'):
            self._mass_init_done = True
            _, mass_kg, label = self.mass_stages[0]
            set_cube_mass(self.env, mass_kg)
            _current_mass["kg"] = mass_kg
            self._mass_random = False
            print(f"\n[Mass Curriculum] {label}", flush=True)
            stats["cube_mass_kg"] = mass_kg

    # ── Effort curriculum ──

    def _apply_effort(self, stats):
        return  # DISABLED — ManiSkill uses fixed 100N
        """Stage-based effort based on milestone progression."""
        from .env_config import _set_robot_effort
        if not hasattr(self, '_effort_stage'):
            self._effort_stage = 0
        _stages = [
            ("close_01",     0.10, "align"),
            ("close_05",     0.25, "close"),
            ("grasp_start",  0.40, "grip"),
            ("ge_04",        0.55, "ge_mid"),
            ("grasp_enough", 0.65, "grasp"),
            ("grasp_5mm",    0.80, "lift"),
            ("lift_100mm",   0.90, "lift_high"),
            (None,           1.00, "lift_hold"),
        ]
        rates = stats.get("term_ep_rates", {})

        while self._effort_stage < len(_stages) - 1:
            ms_key = _stages[self._effort_stage][0]
            if ms_key and rates.get(ms_key, 0) > 0.5:
                self._effort_stage += 1
            else:
                break

        ratio = _stages[min(self._effort_stage, len(_stages)-1)][1]
        label = _stages[min(self._effort_stage, len(_stages)-1)][2]
        mass_kg = _current_mass["kg"]
        full_effort = max(REAL_EFFORT_NM, mass_kg * 9.81 * 0.25 + REAL_EFFORT_NM)
        gripper_effort = max(REAL_EFFORT_NM, full_effort * ratio)

        if label in ("align", "close"):
            arm_min = ARM_SELF_WEIGHT_NM
        elif label in ("grip", "ge_mid"):
            arm_min = mass_kg * 9.81 * 0.35 + ARM_SELF_WEIGHT_NM
        else:
            arm_min = mass_kg * 9.81 * 0.50 + ARM_SELF_WEIGHT_NM
        arm_effort = max(gripper_effort, arm_min)

        if not hasattr(self, '_last_effort_ratio') or self._last_effort_ratio != ratio:
            self._last_effort_ratio = ratio
            _set_robot_effort(self.env, gripper_effort, arm_effort)
            print(f"\n[Effort] grip={gripper_effort:.1f}Nm arm={arm_effort:.1f}Nm ({ratio*100:.0f}%) stage={label}", flush=True)

    # ── Exploration noise ──

    def _apply_exploration(self, stats):
        """Scale noise for first mass stage (5kg) only: 1.3x -> 1.0x."""
        if not hasattr(self, '_mass_lift_count'):
            return
        if self._mass_idx > 0:
            if hasattr(self, '_last_explore_scale') and self._last_explore_scale != 1.0:
                self._last_explore_scale = 1.0
                stats["explore_noise_scale"] = 1.0
            return
        _, cur_mass, _ = self.mass_stages[0]
        lifts_needed = self._lifts_for_mass(cur_mass)
        if lifts_needed <= 0:
            return
        progress = min(self._mass_lift_count / lifts_needed, 1.0)
        scale = 1.0 if progress >= 0.9 else 1.3 - 0.3 * (progress / 0.9)
        if not hasattr(self, '_last_explore_scale') or abs(self._last_explore_scale - scale) > 0.05:
            self._last_explore_scale = scale
            stats["explore_noise_scale"] = scale

    # ── Plateau detection + entropy boost ──

    def _check_plateau(self, iteration, stats):
        """If max_reward doesn't improve for 100 iters, boost entropy 2x for 200 iters."""
        if iteration % self._plateau_window != 0 or iteration == 0:
            return

        current_max = stats.get("alltime_max_reward", 0)

        # Check if improved
        if current_max > self._plateau_last_max + 1.0:
            # Improved — reset stuck count
            self._plateau_stuck_count = 0
            self._plateau_last_max = current_max
            # If boosted, restore entropy
            if self._entropy_boosted and iteration >= self._entropy_boost_until:
                self._entropy_boosted = False
                try:
                    self.env.unwrapped.cfg = self.env.unwrapped.cfg  # trigger recompute
                    runner_alg = stats.get("_runner_alg", None)
                    if runner_alg and hasattr(runner_alg, 'entropy_coef'):
                        runner_alg.entropy_coef = self._base_entropy
                        print(f"\n[Plateau] Entropy restored to {self._base_entropy}", flush=True)
                except Exception:
                    pass
        else:
            self._plateau_stuck_count += 1

        # Plateau detected: stuck for 100 iters
        if self._plateau_stuck_count >= 1 and not self._entropy_boosted:
            self._entropy_boosted = True
            self._entropy_boost_until = iteration + 200
            boosted = self._base_entropy * 2.0
            stats["entropy_boost"] = boosted
            print(f"\n[Plateau] Detected at iter {iteration} (max_rew={current_max:.0f}). "
                  f"Entropy {self._base_entropy} -> {boosted} for 200 iters", flush=True)

        # Auto-restore after 200 iters
        if self._entropy_boosted and iteration >= self._entropy_boost_until:
            self._entropy_boosted = False
            self._plateau_stuck_count = 0
            stats["entropy_boost"] = None
            print(f"\n[Plateau] Entropy restored to {self._base_entropy} at iter {iteration}", flush=True)

