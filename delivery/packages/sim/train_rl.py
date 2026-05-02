#!/usr/bin/env python3
"""RL Training for Lift Cube with live dashboard + success data collection.

rsl_rl v5 API:
  - obs is TensorDict (from tensordict package), not a plain tensor
  - alg.compute_returns(obs) must be called before alg.update()
  - alg.update() returns loss_dict (dict), not scalar
  - alg.train_mode() / alg.eval_mode() on the algorithm
  - runner.load(path, load_cfg=None, strict=True)
  - Policy accessed via alg.get_policy() (not alg.policy)
  - Env name: LeIsaac-SO101-LiftCube-v0 (not -RL-v0)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT",
    "/workspace/leisaac/assets")

# ── Args ──
parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--max-iterations", type=int, default=10_000_000)
parser.add_argument("--http-port", type=int, default=8888)
parser.add_argument("--save-dir", type=str, default="/data/rl_output")
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--config", type=str, default="configs/reward_config.yaml")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
args, _ = parser.parse_known_args()

# Set random seed if specified
if args.seed is not None:
    import random
    random.seed(args.seed)
    import numpy as _np
    _np.random.seed(args.seed)
    import torch as _torch
    _torch.manual_seed(args.seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(args.seed)
    print(f"[Seed] Set to {args.seed}", flush=True)

# ── Config ──
CFG = {}
if os.path.exists(args.config):
    with open(args.config) as f:
        CFG = yaml.safe_load(f)
    print(f"[Config] Loaded: {args.config}", flush=True)

# ── Isaac Lab init (must be before other isaac imports) ──
from isaaclab.app import AppLauncher
_enable_cams = CFG.get("enable_cameras", True)
app_launcher = AppLauncher({"enable_cameras": _enable_cams})
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
import isaaclab.sim as sim_utils
import leisaac.tasks.lift_cube
from leisaac.tasks.lift_cube import mdp
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from packages.sim import dashboard
from packages.sim.env_setup import (
    configure_env, build_ppo_config, set_cube_mass, apply_motor_limits,
    CurriculumManager,
)

from packages.sim.env_builder import create_env, create_runner, install_std_clamping, resume_from_checkpoint
from packages.sim.episode_tracker import EpisodeTracker
from packages.sim.data_saver import save_episode_data
from packages.sim.iter_logger import IterLogger

DEVICE = "cuda:0"

# ── Data collection thresholds (from config) ──
dc = CFG.get("data_collection", {})

# ======================================================
#  Environment Setup
# ======================================================
print("[Env] Setting up LiftCube RL environment (rsl_rl v5)...", flush=True)
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=args.num_envs)

env, cam_map, sensors = create_env(
    env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg,
    TiledCameraCfg, ContactSensorCfg, sim_utils, ManagerBasedRLEnv,
    parse_env_cfg, configure_env, DEVICE, args.num_envs,
)

# Sync DEVICE with env (may be cpu if PhysX CPU mode)
DEVICE = str(env.device)

# ======================================================
#  PPO Runner (rsl_rl v5)
# ======================================================
wrapped_env = RslRlVecEnvWrapper(env)

# Append seed to save dir for multi-seed runs
_save_dir = args.save_dir
if args.seed is not None:
    _save_dir = os.path.join(args.save_dir, f"seed_{args.seed}")
log_dir = os.path.join(_save_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

runner, train_cfg = create_runner(wrapped_env, CFG, args, DEVICE, log_dir)
install_std_clamping(runner)
start_iteration, _resumed_stats = resume_from_checkpoint(runner, args, log_dir)

# ======================================================
#  Dashboard + Curriculum
# ======================================================
_default_stats = {
    "iteration": 0, "max_iterations": args.max_iterations,
    "mean_reward": 0, "max_reward": 0, "alltime_max_reward": 0,
    "alltime_best_mean": 0, "success_count": 0, "episode_count": 0,
    "best_lift_cm": 0, "status": "initializing", "current_phase": "---",
    "current_lift_cm": 0, "current_hold_steps": 0,
    "recent_episodes": [], "success_rate_pct": 0,
    "current_contact_force": 0, "curriculum_stage": "stage1",
    "cube_range_cm": 2, "reward_history": [], "cube_mass_kg": 30.0,
    "reach_pct": 0, "grasp_pct": 0, "lift_pct": 0, "hold_pct": 0,
}
if _resumed_stats:
    _default_stats.update(_resumed_stats)
    _default_stats["status"] = "initializing"
dashboard.train_stats = _default_stats
stats = dashboard.train_stats

save_dir = Path(_save_dir) / "episodes"
save_dir.mkdir(parents=True, exist_ok=True)
dashboard.start(args.http_port)

# Initial mass (respect resume iteration for curriculum)
mass_cfg = CFG.get("mass_curriculum", [])
initial_mass = mass_cfg[0].get("mass", 30.0) if mass_cfg else 30.0
for stage in mass_cfg:
    if start_iteration < stage.get("until_iter", 0):
        initial_mass = stage["mass"]
        break
set_cube_mass(env, initial_mass)

# Warm up

apply_motor_limits(env, CFG)

# Step 1: Add fingertip collision pads for stable grasping
try:
    pass  # fingertip pads removed
except Exception as e:
    print(f"[FingertipPad] Skip: {e}", flush=True)
try:
    import omni.usd
    from pxr import UsdGeom
    _stage = omni.usd.get_context().get_stage()
    for _p in _stage.Traverse():
        if "cube" in _p.GetName().lower():
            _bbox = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
            _b = _bbox.ComputeWorldBound(_p)
            _r = _b.GetRange()
            if not _r.IsEmpty():
                _sz = _r.GetSize()
                print(f"[BBOX] {_p.GetPath()} x={_sz[0]*100:.1f}cm y={_sz[1]*100:.1f}cm z={_sz[2]*100:.1f}cm", flush=True)
except Exception as _e:
    print(f"[BBOX] Error: {_e}", flush=True)
dashboard.update_cameras(cam_map)
stats["status"] = "training"

curriculum = CurriculumManager(CFG, env)

# ── Tracker + Logger ──
tracker = EpisodeTracker(args.num_envs, stats, dc, device=DEVICE)
tracker.set_sensors(sensors["contact_sensor"], sensors["jaw_sensor"], sensors["has_contact_sensor"])
logger = IterLogger(log_dir)

print(f"\n[RL] === Training Start (iter {start_iteration}) -- rsl_rl v5 API ===", flush=True)

# ======================================================
#  Training Loop
# ======================================================
# v5: train_mode() is on the algorithm, not the runner
runner.alg.train_mode()

# v5: get_observations() returns TensorDict -- no manual .to(DEVICE) needed
obs = wrapped_env.get_observations()

for iteration in range(start_iteration, args.max_iterations):
    runner.current_learning_iteration = iteration
    curriculum.step(iteration, stats)

    # Apply entropy boost from plateau detection
    _eboost = stats.pop("entropy_boost", None)
    if _eboost is not None:
        try:
            runner.alg.entropy_coef = _eboost
        except Exception:
            pass

    # Exploration noise scaling from curriculum
    tracker.apply_exploration_noise(runner, stats, iteration, start_iteration)

    ep_rewards_this_iter = []

    # v5: ensure train mode at start of each rollout
    runner.alg.train_mode()

    for step in range(train_cfg["num_steps_per_env"]):
        # v5: obs is TensorDict; alg.act() handles it natively
        actions = runner.alg.act(obs)

        # v5: wrapped_env.step() returns TensorDict obs


        obs, rewards, dones, extras = wrapped_env.step(actions)

        # Camera update (periodic)
        if True:  # every step for smooth video
            dashboard.update_cameras(cam_map)

        # Collect state data every 4 steps (batch GPU->CPU transfer)
        if step % 4 == 0:
            tracker.collect_state_data_batch(env, actions, cam_map)

        # Vectorized step processing (all envs at once, no Python for-loop)
        tracker.process_step_vectorized(env, rewards, step, iteration,
                                        start_iteration, dashboard)

        # Episode ends (only loop over done envs)
        done_envs = dones.nonzero(as_tuple=True)[0]
        for env_id in done_envs.tolist():
            ep_rew, ep_info = tracker.end_episode(env_id, dashboard)
            ep_rewards_this_iter.append(ep_rew)

            save_episode_data(save_dir, tracker.episode_count, env_id,
                              tracker, iteration, args.config)

            tracker.reset_episode(env_id)

        # v5: process_env_step still stores transitions per step
        runner.alg.process_env_step(obs, rewards, dones, extras)

    # ── v5 PPO update sequence ──
    # 1. Compute returns from the final obs BEFORE calling update
    runner.alg.compute_returns(obs)

    # 2. Enter train mode and run PPO update
    runner.alg.train_mode()
    # v5: update() returns a loss_dict (dict of scalars), not a single scalar
    loss_dict = runner.alg.update()

    # Extract a summary loss value for logging
    if isinstance(loss_dict, dict):
        _loss_vals = [v for v in loss_dict.values() if isinstance(v, (int, float))]
        mean_loss = np.mean(_loss_vals) if _loss_vals else 0.0
    else:
        mean_loss = float(loss_dict)

    mean_rew = np.mean(ep_rewards_this_iter) if ep_rewards_this_iter else 0

    # Per-term reward snapshot
    term_snap = logger.compute_term_snapshot(env)

    with dashboard.stats_lock:
        logger.build_stats_snapshot(
            iteration, stats, ep_rewards_this_iter,
            tracker._recent_ep_rewards, tracker._alltime_max,
            tracker._alltime_best_mean, mean_loss, loss_dict, env, term_snap,
        )

    # ── Write iter log CSV ──
    logger.write_csv_row(
        iteration, stats, ep_rewards_this_iter,
        tracker._recent_ep_rewards, tracker._alltime_max,
        tracker._alltime_best_mean, tracker.best_lift,
        tracker.success_count, tracker.episode_count, mean_loss,
    )

    logger.console_log(
        iteration, args.max_iterations, mean_rew,
        tracker._alltime_max, tracker.best_lift,
        tracker.success_count, tracker.episode_count, stats,
    )

    # Checkpoint
    logger.save_checkpoint(iteration, runner, stats)
