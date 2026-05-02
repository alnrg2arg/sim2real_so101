"""Debug inference: log detailed force/grasp state every step."""
import os, numpy as np, yaml
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", "/workspace/leisaac/assets")

CFG = {}
if os.path.exists("configs/reward_config.yaml"):
    with open("configs/reward_config.yaml") as f:
        CFG = yaml.safe_load(f)
CFG["enable_cameras"] = True

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"enable_cameras": True})
simulation_app = app_launcher.app

import torch
import torch.nn as nn
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import isaaclab.sim as sim_utils
import leisaac.tasks.lift_cube
from leisaac.tasks.lift_cube import mdp
from packages.sim.env_setup.env_config import configure_env, apply_motor_limits
from packages.sim.env_setup.maniskill_rewards import _is_grasped, _quat_to_y_axis, _get_link_body_idx

DEVICE = "cuda:0"
CKPT = "/data/rl_sac_state_v17/ckpt_99944448.pt"

env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=1)
env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg)
_cube_prim = env_cfg.scene.cube.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", update_period=0.0, history_length=4, filter_prim_paths_expr=[_cube_prim])
env_cfg.scene.jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/jaw", update_period=0.0, history_length=4, filter_prim_paths_expr=[_cube_prim])
env_cfg.scene.cube.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True
if hasattr(env_cfg.observations, 'subtask_terms'):
    env_cfg.observations.subtask_terms = None
if not hasattr(env_cfg.scene, 'wrist') or env_cfg.scene.wrist is None:
    env_cfg.scene.wrist = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper/fp_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.05, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=2.5, focus_distance=200.0, horizontal_aperture=24.0, vertical_aperture=18.0, clipping_range=(0.003, 50.0), lock_camera=True),
        width=128, height=128, update_period=0)

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, CFG)
obs_dim = env.observation_manager.compute()["policy"].shape[-1]
act_dim = env.action_manager.total_action_dim

# Load actor
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None: m.bias.data.fill_(0.0)

class StateProjection(nn.Module):
    def __init__(self, n, device=None):
        super().__init__()
        self.repr_dim = 256
        self.state_proj = nn.Sequential(nn.Linear(n, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU())
    def forward(self, s): return self.state_proj(s)

class Actor(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.proj = StateProjection(n_obs, device=device)
        self.fc = nn.Sequential(
            nn.Linear(256, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU(),
            nn.Linear(256, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU(),
            nn.Linear(256, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU())
        self.fc_mean = nn.Linear(256, n_act, device=device)
        self.fc_logstd = nn.Linear(256, n_act, device=device)
        self.register_buffer("action_scale", torch.ones(n_act, device=device))
        self.register_buffer("action_bias", torch.zeros(n_act, device=device))
        self.apply(weight_init)
    def get_eval_action(self, s):
        x = self.proj(s); x = self.fc(x)
        return torch.tanh(self.fc_mean(x)) * self.action_scale + self.action_bias

actor = Actor(n_obs=obs_dim, n_act=act_dim, device=DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
actor.load_state_dict(ckpt["actor"])
actor.eval()
print(f"[Debug] Loaded {CKPT}", flush=True)

# Run 3 episodes with detailed logging
for ep in range(3):
    obs_dict = env.reset()[0]
    obs = obs_dict["policy"]
    print(f"\n{'='*80}", flush=True)
    print(f"Episode {ep}", flush=True)
    print(f"{'='*80}", flush=True)

    for step in range(100):
        with torch.no_grad():
            action = actor.get_eval_action(obs)
        obs_dict, rewards, dones, truncs, infos = env.step(action)
        obs = obs_dict["policy"]

        # Get detailed state
        cube_pos = env.scene["cube"].data.root_pos_w[0]
        ee = env.scene["ee_frame"]
        tcp = (ee.data.target_pos_w[0, 0, :] + ee.data.target_pos_w[0, 1, :]) * 0.5
        tcp_dist = (cube_pos - tcp).norm().item()
        cube_z = cube_pos[2].item()

        # Contact forces
        grip_f = env.scene["gripper_contact"].data.force_matrix_w[0, 0, 0, :]
        jaw_f = env.scene["jaw_contact"].data.force_matrix_w[0, 0, 0, :]
        grip_mag = grip_f.norm().item()
        jaw_mag = jaw_f.norm().item()

        # Force directions
        grip_xy = grip_f.clone(); grip_xy[2] = 0
        jaw_xy = jaw_f.clone(); jaw_xy[2] = 0
        grip_xy_mag = grip_xy.norm().item()
        jaw_xy_mag = jaw_xy.norm().item()

        # Anti-parallel
        if grip_xy_mag > 0.01 and jaw_xy_mag > 0.01:
            dot_xy = (grip_xy / (grip_xy.norm() + 1e-8) * jaw_xy / (jaw_xy.norm() + 1e-8)).sum().item()
        else:
            dot_xy = 0.0

        # Z component ratio
        grip_z_ratio = abs(grip_f[2].item()) / (grip_mag + 1e-8)
        jaw_z_ratio = abs(jaw_f[2].item()) / (jaw_mag + 1e-8)

        # Grasp check
        is_grasp = _is_grasped(env)[0].item()

        # Robot TCP height vs cube height
        tcp_z = tcp[2].item()

        if step % 5 == 0 or grip_mag > 0.1 or jaw_mag > 0.1:
            print(f"  step={step:3d} cube_z={cube_z:.3f} tcp_z={tcp_z:.3f} tcp_dist={tcp_dist:.3f} "
                  f"grip={grip_mag:.2f}N(z%={grip_z_ratio:.0%}) jaw={jaw_mag:.2f}N(z%={jaw_z_ratio:.0%}) "
                  f"xy_dot={dot_xy:+.2f} grasp={'Y' if is_grasp else 'N'} "
                  f"r={rewards.item():.1f}", flush=True)

        if (dones | truncs).any():
            break

print("\n[Debug] Done", flush=True)
env.close()
simulation_app.close()
