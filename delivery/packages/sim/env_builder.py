"""Environment builder — camera setup, contact sensors, env creation, PPO runner, resume logic."""

import json
import os
import types

import torch


def create_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg,
               TiledCameraCfg, ContactSensorCfg, sim_utils, ManagerBasedRLEnv,
               parse_env_cfg, configure_env, DEVICE, num_envs):
    """Create the Isaac Lab RL environment with cameras and contact sensors.

    Returns:
        (env, cam_map, sensors_dict) where sensors_dict has keys:
        'has_contact_sensor', 'contact_sensor', 'jaw_sensor'.
    """
    _enable_cams = CFG.get("enable_cameras", True)

    env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg)

    # Cameras: remove base config cameras if disabled, override if enabled
    if not _enable_cams:
        for _cam_name in ["front", "side", "wrist"]:
            if hasattr(env_cfg.scene, _cam_name):
                setattr(env_cfg.scene, _cam_name, None)

    if _enable_cams:
        env_cfg.scene.front = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(-0.03141, -0.5301, 0.43648),
                rot=(0.92171, 0.38687, -0.02715, -0.00607),
                convention="opengl",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=13.0, focus_distance=400.0,
                horizontal_aperture=20.955, vertical_aperture=15.2908,
                clipping_range=(0.01, 50.0), lock_camera=True,
            ),
            width=640, height=480, update_period=0,
        )
        # side & wrist enabled for 3-camera view
        env_cfg.scene.side = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, -0.5, 0.35),
                rot=(0.9239, 0.3827, 0.0, 0.0),
                convention="opengl",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=13.0, focus_distance=400.0,
                horizontal_aperture=20.955, vertical_aperture=15.2908,
                clipping_range=(0.01, 50.0), lock_camera=True,
            ),
            width=320, height=240, update_period=0,
        )

        env_cfg.scene.wrist = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist/wrist_camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.05),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="opengl",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=5.0, focus_distance=400.0,
                horizontal_aperture=20.955, vertical_aperture=15.2908,
                clipping_range=(0.01, 50.0), lock_camera=True,
            ),
            width=320, height=240, update_period=0,
        )

    # Get cube prim path for contact filtering
    _cube_prim = env_cfg.scene.cube.prim_path

    # Contact sensors with pair-wise filtering (GPU native)
    # No filter (detect ALL contacts including cube)
    env_cfg.scene.gripper_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper", update_period=0.0,
        history_length=4, debug_vis=False,
        filter_prim_paths_expr=[_cube_prim],  # pair-wise: gripper→cube only
    )
    env_cfg.scene.jaw_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw", update_period=0.0,
        history_length=4, debug_vis=False,
        filter_prim_paths_expr=[_cube_prim],  # pair-wise: jaw→cube only
    )
    # Cube contact: use cube's own prim path
    try:
        _cube_prim = env_cfg.scene.cube.prim_path
        env_cfg.scene.cube_contact = ContactSensorCfg(
            prim_path=_cube_prim, update_period=0.0,
            history_length=4, debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/gripper", "{ENV_REGEX_NS}/Robot/jaw"],
        )
        env_cfg.scene.cube.spawn.activate_contact_sensors = True
    except Exception as _e:
        print(f"[WARN] cube_contact setup failed: {_e}", flush=True)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"[Env] Created: obs={env.observation_space}, act={env.action_space}", flush=True)

    # Camera/sensor refs
    cam_map = {}
    for name in ["front", "side", "wrist"]:
        try:
            cam_map[name] = env.scene[name]
            print(f"[Camera] {name}: OK", flush=True)
        except Exception as e:
            print(f"[Camera] {name}: FAILED ({e})", flush=True)

    has_contact_sensor = False
    contact_sensor = None
    jaw_sensor = None
    try:
        contact_sensor = env.scene["gripper_contact"]
        jaw_sensor = env.scene["jaw_contact"]
        has_contact_sensor = True
        print(f"[Sensor] Gripper contact: OK", flush=True)
        print(f"[Sensor] Jaw contact: OK", flush=True)
    except Exception as e:
        print(f"[Sensor] Contact setup: {e}", flush=True)

    sensors = {
        "has_contact_sensor": has_contact_sensor,
        "contact_sensor": contact_sensor,
        "jaw_sensor": jaw_sensor,
    }

    return env, cam_map, sensors


def create_runner(wrapped_env, CFG, args, DEVICE, log_dir):
    """Create the OnPolicyRunner with PPO config.

    Args:
        wrapped_env: RslRlVecEnvWrapper around the env.
        CFG: Full config dict.
        args: Parsed command-line args.
        DEVICE: Device string.
        log_dir: Path to log directory.

    Returns:
        (runner, train_cfg) tuple.
    """
    from rsl_rl.runners import OnPolicyRunner
    from packages.sim.env_setup import build_ppo_config

    train_cfg = build_ppo_config(CFG, args.max_iterations, DEVICE)
    runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=log_dir, device=DEVICE)
    
    # Apply orthogonal weight init (ManiSkill ppo.py exact)
    import math
    def _ortho_init(module, std=math.sqrt(2), bias=0.0):
        for name, param in module.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                torch.nn.init.orthogonal_(param, std)
            elif 'bias' in name:
                torch.nn.init.constant_(param, bias)
    try:
        policy = runner.alg.get_policy() if hasattr(runner.alg, 'get_policy') else getattr(runner.alg, 'policy', None)
        if policy is not None:
            _ortho_init(policy)
            print("[Init] Applied orthogonal init (ManiSkill)", flush=True)
    except Exception as e:
        print(f"[Init] Orthogonal init skipped: {e}", flush=True)
    
    return runner, train_cfg


def install_std_clamping(runner):
    """Install std clamping hook on the policy to prevent NaN crash."""
    try:
        _policy = None
        if hasattr(runner.alg, 'get_policy'):
            _policy = runner.alg.get_policy()
        elif hasattr(runner.alg, 'policy'):
            _policy = runner.alg.policy

        if _policy is not None and hasattr(_policy, 'update_distribution'):
            _orig_update_dist = _policy.update_distribution

            def _safe_update(self, obs):
                _orig_update_dist(obs)
                if hasattr(self, 'distribution') and hasattr(self.distribution, 'scale'):
                    self.distribution.scale = self.distribution.scale.clamp(min=0.01, max=10.0)

            _policy.update_distribution = types.MethodType(_safe_update, _policy)
            print("[RL] Std clamping hook installed", flush=True)
        else:
            print("[RL] Std clamping: no compatible policy found, skipping", flush=True)
    except Exception as e:
        print(f"[RL] Std clamping skipped: {e}", flush=True)


def resume_from_checkpoint(runner, args, log_dir):
    """Resume training from a checkpoint.

    Returns:
        (start_iteration, resumed_stats) tuple.
    """
    start_iteration = 0
    _resumed_stats = None

    if not args.resume:
        return start_iteration, _resumed_stats

    print(f"[RL] Resuming: {args.resume}", flush=True)
    # v5 load: check if checkpoint has critic before loading
    _ckpt_keys = torch.load(args.resume, map_location="cpu", weights_only=False).keys()
    _has_critic = "critic_state_dict" in _ckpt_keys
    # Skip optimizer if BC checkpoint (optimizer param groups won't match PPO)
    _is_bc = "bc" in args.resume.lower() or not _has_critic
    _load_cfg = {"actor": True, "critic": _has_critic, "optimizer": not _is_bc, "iteration": True}
    if _is_bc:
        print("[RL] BC checkpoint detected -- loading actor only (fresh optimizer + critic)", flush=True)
    elif not _has_critic:
        print("[RL] Checkpoint missing critic_state_dict -- loading actor + optimizer only", flush=True)
    runner.load(args.resume, load_cfg=_load_cfg, strict=True)

    # Restore iteration from checkpoint
    start_iteration = runner.current_learning_iteration
    # If checkpoint had iter=0 (old format), infer from filename
    if start_iteration == 0 and "model_" in args.resume:
        try:
            start_iteration = int(args.resume.split("model_")[-1].split(".pt")[0])
        except ValueError:
            pass

    # Load training stats from sidecar file
    stats_path = args.resume.replace(".pt", "_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            _resumed_stats = json.load(f)
        print(f"[RL] Restored stats from {stats_path}", flush=True)
    elif os.path.exists(os.path.join(log_dir, "model_latest_stats.json")):
        with open(os.path.join(log_dir, "model_latest_stats.json")) as f:
            _resumed_stats = json.load(f)
        print(f"[RL] Restored stats from latest_stats.json", flush=True)

    if start_iteration > 0:
        print(f"[RL] Resumed at iteration {start_iteration}", flush=True)

    return start_iteration, _resumed_stats
