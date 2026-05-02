"""Fast SAC for Isaac Lab — Squint-exact with 3-camera vision (front+side+wrist).

Matches Squint (https://github.com/aalmuzairee/squint) exactly:
- C51 Distributional Critic (101 atoms)
- vmap Q-network ensemble via from_modules
- CudaGraphModule on top of torch.compile
- bfloat16 autocast
- bootstrap_at_done="always" (always bootstrap, never cut)
- torchrl ReplayBuffer with LazyTensorStorage + TensorDict
- Delayed policy updates (policy_freq=4)
- Alpha update inside critic step (Squint ordering)
"""

import argparse
import os
import math
import random
import time
import json
import glob
import numpy as np
import yaml

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("LEISAAC_ASSETS_ROOT", "/workspace/leisaac/assets")
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--image-size", type=int, default=16)
parser.add_argument("--total-timesteps", type=int, default=100_000_000)
parser.add_argument("--http-port", type=int, default=8888)
parser.add_argument("--save-dir", type=str, default="/data/rl_sac_fast")
parser.add_argument("--config", type=str, default="configs/reward_config.yaml")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt file to resume from")
parser.add_argument("--no-cudagraphs", action="store_true", help="Disable CudaGraphModule")
args, _ = parser.parse_known_args()

CFG = {}
if os.path.exists(args.config):
    with open(args.config) as f:
        CFG = yaml.safe_load(f)
CFG["enable_cameras"] = True  # force cameras on for vision

# ── Isaac Lab init ──
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"enable_cameras": True})
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule
from torchrl.data import LazyTensorStorage, ReplayBuffer

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
import isaaclab.sim as sim_utils
import leisaac.tasks.lift_cube
from leisaac.tasks.lift_cube import mdp

from packages.sim.env_setup.env_config import configure_env, apply_motor_limits
from packages.sim import dashboard

DEVICE = "cuda:0"
NUM_ENVS = args.num_envs

# ── Env setup ──
env_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-v0", device=DEVICE, num_envs=NUM_ENVS)
env_cfg = configure_env(env_cfg, CFG, mdp, RewTerm, DoneTerm, SceneEntityCfg)

_cube_prim = env_cfg.scene.cube.prim_path
env_cfg.scene.gripper_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper", update_period=0.0, history_length=4,
    filter_prim_paths_expr=[_cube_prim])
env_cfg.scene.jaw_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/jaw", update_period=0.0, history_length=4,
    filter_prim_paths_expr=[_cube_prim])
env_cfg.scene.cube_contact = ContactSensorCfg(
    prim_path=_cube_prim, update_period=0.0, history_length=4,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/gripper", "{ENV_REGEX_NS}/Robot/jaw"])
env_cfg.scene.cube.spawn.activate_contact_sensors = True
env_cfg.scene.robot.spawn.activate_contact_sensors = True

if hasattr(env_cfg.observations, 'subtask_terms'):
    env_cfg.observations.subtask_terms = None

# Ensure wrist camera exists (may be removed by configure_env)
if not hasattr(env_cfg.scene, 'wrist') or env_cfg.scene.wrist is None:
    from isaaclab.sensors import TiledCameraCfg
    env_cfg.scene.wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/fp_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.05, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=2.5, focus_distance=200.0, horizontal_aperture=24.0, vertical_aperture=18.0, clipping_range=(0.003, 50.0), lock_camera=True),
        width=640, height=480, update_period=0)

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, CFG)

obs_dim = env.observation_manager.compute()["policy"].shape[-1]
act_dim = env.action_manager.total_action_dim

IMAGE_SIZE = args.image_size
CAMERA_NAMES = ["wrist"]  # Squint: wrist only
N_CHANNELS = 3  # single camera RGB
print(f"[Env] {NUM_ENVS} envs, obs_state={obs_dim}, act={act_dim}, image={IMAGE_SIZE}x{IMAGE_SIZE}x{N_CHANNELS}", flush=True)

# ── Seeding ──
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ── Hyperparams (Squint-exact) ──
GAMMA = 0.9
TAU = 0.01
BATCH_SIZE = 1024
BUFFER_SIZE = 1_000_000
NUM_UPDATES = 256
POLICY_FREQ = 4
TARGET_NET_FREQ = 1
LEARNING_STARTS = 5000
POLICY_LR = 3e-4
Q_LR = 3e-4
ALPHA_LR = 3e-4
NUM_ATOMS = 101
V_MIN = -20.0
V_MAX = 20.0
NUM_Q = 2
USE_COMPILE = True
USE_CUDAGRAPHS = not args.no_cudagraphs
BOOTSTRAP_AT_DONE = "always"  # Squint default

# ── Camera reading ──
def read_cameras(env, target_size=IMAGE_SIZE):
    """Read 3 cameras, downsample, concat → (N, H, W, 9) uint8."""
    frames = []
    for cam_name in CAMERA_NAMES:
        cam = env.scene[cam_name]
        rgb = cam.data.output["rgb"][:, :, :, :3]
        if rgb.dtype == torch.float32:
            rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
        rgb_f = rgb.float().permute(0, 3, 1, 2)
        rgb_small = F.interpolate(rgb_f, size=(target_size, target_size), mode='bilinear', align_corners=False)
        frames.append(rgb_small)
    return torch.cat(frames, dim=1).permute(0, 2, 3, 1).to(torch.uint8)

# ── Networks (Squint-exact with vision) ──
def weight_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        gain = nn.init.calculate_gain('relu') if isinstance(m, nn.Conv2d) else 1.0
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class CNNEncoder(nn.Module):
    """Squint-exact CNN encoder, 9-channel input (3 cameras x RGB)."""
    def __init__(self, n_channels=N_CHANNELS, image_size=IMAGE_SIZE, device=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, 4, stride=2, device=device), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=1, device=device), nn.ReLU(),
            nn.Flatten()
        )
        dummy = torch.zeros(1, n_channels, image_size, image_size, device=device)
        self.repr_dim = self.conv(dummy).shape[-1]
        self.apply(weight_init)
        self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, obs):
        # obs: (B, H, W, C) float [0-255]
        obs = obs.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        return self.conv(obs / 255.0 - 0.5)


class Projection(nn.Module):
    """Squint-exact: CNN features → 50D + state → 256D = 306D."""
    def __init__(self, n_cnn_features, n_state, device=None):
        super().__init__()
        self.repr_dim = 50 + 256
        self.rgb_proj = nn.Sequential(
            nn.Linear(n_cnn_features, 50, device=device), nn.LayerNorm(50, device=device), nn.Tanh(),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(n_state, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU(),
        )
        self.apply(weight_init)

    def forward(self, rgb_feat, state):
        return torch.cat([self.rgb_proj(rgb_feat), self.state_proj(state)], dim=-1)


class StateProjection(nn.Module):
    """Kept for Critic internal projection (not used directly, replaced by Projection)."""
    def __init__(self, n_state, device=None):
        super().__init__()
        self.repr_dim = 256
        self.state_proj = nn.Sequential(
            nn.Linear(n_state, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU(),
        )

    def forward(self, state):
        return self.state_proj(state)


class Actor(nn.Module):
    def __init__(self, n_proj, n_act, device=None):
        """n_proj = Projection output dim (306)."""
        super().__init__()
        hidden_dim = 256

        self.fc = nn.Sequential(
            nn.Linear(n_proj, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, n_act, device=device)
        self.fc_logstd = nn.Linear(hidden_dim, n_act, device=device)

        self.register_buffer("action_scale", torch.ones(n_act, device=device))
        self.register_buffer("action_bias", torch.zeros(n_act, device=device))

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.apply(weight_init)

    def forward(self, proj_features, get_log_std=False):
        x = self.fc(proj_features)
        mean = self.fc_mean(x)
        if get_log_std:
            log_std = self.fc_logstd(x)
            log_std = torch.tanh(log_std)
            log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
            return mean, log_std
        return mean

    def get_eval_action(self, state):
        mean = self.forward(state)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, state):
        mean, log_std = self.forward(state, get_log_std=True)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Critic(nn.Module):
    """Distributional C51 Ensemble-Q-network critic with vmap (Squint-exact).
    Receives projected features (306D) from external Projection module."""
    def __init__(self, n_proj, n_act, num_atoms, v_min, v_max, num_q=2, device=None):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_q = num_q
        self.v_min = v_min
        self.v_max = v_max
        self.q_support = torch.linspace(v_min, v_max, num_atoms, device=device)

        q_input_dim = n_proj + n_act

        # Build Q-networks, apply weight init, then stack into q_params
        q_nets = [self._build_q_network(q_input_dim, num_atoms, device=device) for _ in range(num_q)]
        for qn in q_nets:
            qn.apply(weight_init)

        # q_params: registered stacked parameter container (what optimizer + vmap both use)
        self.q_params = from_modules(*q_nets, as_module=True)

        # Meta-device template for vmap dispatch (hidden from parameters()/state_dict())
        object.__setattr__(self, '_q_meta', self._build_q_network(q_input_dim, num_atoms, device="meta"))
        object.__setattr__(self, '_q_repr', repr(q_nets[0]))

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for i in range(self.num_q):
            lines.append(f"  (q{i}): {self._q_repr}")
        lines.append(")")
        return "\n".join(lines)

    def _build_q_network(self, input_dim, num_atoms, device=None):
        hidden_dim = 512
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms, device=device)
        )

    def _vmap_q(self, params, x):
        with params.to_module(self._q_meta):
            return self._q_meta(x)

    def forward(self, proj_features, actions):
        x = torch.cat([proj_features, actions], dim=-1)
        return torch.vmap(self._vmap_q, (0, None))(self.q_params, x)

    def get_q_values(self, proj_features, actions, detach_critic=False):
        """Expected Q-values: [num_q, batch]. detach_critic freezes critic but keeps action grad."""
        if detach_critic:
            x = torch.cat([proj_features.detach(), actions], dim=-1)
            logits = torch.vmap(self._vmap_q, (0, None))(self.q_params.data, x)
        else:
            logits = self.forward(proj_features, actions)
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.q_support, dim=-1)

    def categorical(self, proj_features, actions, rewards, bootstrap, discount):
        """C51 categorical projection: [num_q, batch, num_atoms]."""
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        device = rewards.device

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount * self.q_support
        target_z = target_z.clamp(self.v_min, self.v_max)

        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower = torch.where(torch.logical_and(lower > 0, is_integer), lower - 1, lower)
        upper = torch.where(torch.logical_and(lower == 0, is_integer), upper + 1, upper)

        logits = self.forward(proj_features, actions)  # [num_q, batch, atoms]
        next_dists = F.softmax(logits, dim=-1)

        total_batch = self.num_q * batch_size
        next_dists_flat = next_dists.reshape(-1, self.num_atoms)
        offset = torch.arange(total_batch, device=device).unsqueeze(1) * self.num_atoms

        lower_exp = lower.unsqueeze(0).expand(self.num_q, -1, -1).reshape(total_batch, self.num_atoms)
        upper_exp = upper.unsqueeze(0).expand(self.num_q, -1, -1).reshape(total_batch, self.num_atoms)
        b_exp = b.unsqueeze(0).expand(self.num_q, -1, -1).reshape(total_batch, self.num_atoms)

        max_index = total_batch * self.num_atoms - 1
        lower_indices = torch.clamp((lower_exp + offset).view(-1), 0, max_index)
        upper_indices = torch.clamp((upper_exp + offset).view(-1), 0, max_index)

        proj_dist_flat = torch.zeros_like(next_dists_flat)
        proj_dist_flat.view(-1).index_add_(0, lower_indices, (next_dists_flat * (upper_exp.float() - b_exp)).view(-1))
        proj_dist_flat.view(-1).index_add_(0, upper_indices, (next_dists_flat * (b_exp - lower_exp.float())).view(-1))

        return proj_dist_flat.reshape(self.num_q, batch_size, self.num_atoms)


# ── Init ──
# ── Vision modules ──
encoder = CNNEncoder(n_channels=N_CHANNELS, image_size=IMAGE_SIZE, device=DEVICE)
projection = Projection(n_cnn_features=encoder.repr_dim, n_state=obs_dim, device=DEVICE)
proj_dim = projection.repr_dim  # 306

actor = Actor(n_proj=proj_dim, n_act=act_dim, device=DEVICE)
critic = Critic(n_proj=proj_dim, n_act=act_dim,
                num_atoms=NUM_ATOMS, v_min=V_MIN, v_max=V_MAX,
                num_q=NUM_Q, device=DEVICE)

# Target critic (Squint-exact)
critic_target = Critic(n_proj=proj_dim, n_act=act_dim,
                       num_atoms=NUM_ATOMS, v_min=V_MIN, v_max=V_MAX,
                       num_q=NUM_Q, device=DEVICE)
critic_target.load_state_dict(critic.state_dict())
critic_online_params = list(critic.parameters())
critic_target_params = list(critic_target.parameters())

# Detached copies for inference (weight-sharing)
encoder_detach = CNNEncoder(n_channels=N_CHANNELS, image_size=IMAGE_SIZE, device=DEVICE)
from_module(encoder).data.to_module(encoder_detach)
projection_detach = Projection(n_cnn_features=encoder.repr_dim, n_state=obs_dim, device=DEVICE)
from_module(projection).data.to_module(projection_detach)
actor_detach = Actor(n_proj=proj_dim, n_act=act_dim, device=DEVICE)
from_module(actor).data.to_module(actor_detach)

# Auto entropy (Squint-exact)
target_entropy = -float(act_dim)
log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
alpha = log_alpha.detach().exp()

# Optimizers (Squint-exact: critic optimizer includes encoder + projection)
critic_optimizer = optim.Adam(
    list(critic.parameters()) + list(encoder.parameters()) + list(projection.parameters()),
    lr=Q_LR, capturable=USE_CUDAGRAPHS and not USE_COMPILE)
actor_optimizer = optim.Adam(actor.parameters(), lr=POLICY_LR,
                             capturable=USE_CUDAGRAPHS and not USE_COMPILE)
alpha_optimizer = optim.Adam([log_alpha], lr=ALPHA_LR,
                             capturable=USE_CUDAGRAPHS and not USE_COMPILE)

print(f"[SAC-Vision] CNNEncoder repr_dim={encoder.repr_dim}, Projection repr_dim={proj_dim}", flush=True)

# ── Replay Buffer (torchrl — Squint-exact) ──
rb = ReplayBuffer(storage=LazyTensorStorage(BUFFER_SIZE, device=DEVICE))

# ── Update functions (Squint-exact ordering) ──

def update_main(data):
    """Critic + encoder + projection + alpha update (Squint-exact)."""
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            # Encode next obs (no grad — target)
            next_rgb_feat = encoder(data["next_obs_rgb"].float())
            next_proj = projection(next_rgb_feat, data["next_observations"])
            next_actions, next_log_pi, _ = actor.get_action(next_proj)

            bootstrap = (~data["dones"]).float()
            discount = GAMMA
            rewards = data["rewards"].flatten()

            entropy_bonus = alpha * next_log_pi.flatten()
            rewards_with_entropy = rewards - bootstrap.flatten() * discount * entropy_bonus

            target_distributions = critic_target.categorical(
                next_proj, next_actions,
                rewards_with_entropy, bootstrap, discount
            )

        # Encode current obs (WITH grad — trains encoder + projection)
        rgb_feat = encoder(data["obs_rgb"].float())
        proj = projection(rgb_feat, data["observations"])

        q_outputs = critic(proj, data["actions"])
        q_log_probs = F.log_softmax(q_outputs, dim=-1)

        q_losses = -torch.sum(target_distributions * q_log_probs, dim=-1).mean(dim=-1)
        critic_loss = q_losses.sum()

        with torch.no_grad():
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = torch.sum(q_probs * critic.q_support, dim=-1)
            q_max = q_values.max()
            q_min = q_values.min()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Alpha update (inside critic step — Squint ordering)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            _, log_pi, _ = actor.get_action(proj.detach())
        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()
    alpha.copy_(log_alpha.detach().exp())

    return TensorDict(critic_loss=critic_loss.detach(), q_max=q_max, q_min=q_min,
                      alpha=alpha.detach(), alpha_loss=alpha_loss.detach(),
                      encoded_proj=proj.detach())


def update_actor(data, encoded_proj):
    """Actor update with detached critic (Squint-exact)."""
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        proj = encoded_proj
        pi, log_pi, _ = actor.get_action(proj)
        q_values = critic.get_q_values(proj, pi, detach_critic=True)

        critic_value = q_values.mean(dim=0)
        actor_loss = (alpha * log_pi - critic_value).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return TensorDict(actor_loss=actor_loss.detach())


def get_rollout_action(obs_rgb, obs_state):
    rgb_feat = encoder_detach(obs_rgb.float())
    proj = projection_detach(rgb_feat, obs_state)
    action, _, _ = actor_detach.get_action(proj)
    return action


# ── Compile & CudaGraphs (Squint-exact) ──
if USE_COMPILE:
    update_main = torch.compile(update_main)
    update_actor = torch.compile(update_actor)
    get_rollout_action = torch.compile(get_rollout_action)
    print("[Opt] torch.compile enabled", flush=True)

if USE_CUDAGRAPHS:
    update_main = CudaGraphModule(update_main)
    update_actor = CudaGraphModule(update_actor)
    print("[Opt] CudaGraphModule enabled", flush=True)

# ── Resume ──
resume_step = 0
resume_ep_count = 0
resume_best_return = 0.0
resume_success_count = 0

if args.resume and os.path.exists(args.resume):
    print(f"[Resume] Loading: {args.resume}", flush=True)
    ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    critic_target.load_state_dict(ckpt.get("critic_target", ckpt["critic"]))
    from_module(actor).data.to_module(actor_detach)
    from_module(actor).data.to_module(actor_eval_net)
    if "actor_opt" in ckpt:
        actor_optimizer.load_state_dict(ckpt["actor_opt"])
    if "critic_opt" in ckpt:
        critic_optimizer.load_state_dict(ckpt["critic_opt"])
    if "alpha_opt" in ckpt:
        alpha_optimizer.load_state_dict(ckpt["alpha_opt"])
    if "log_alpha" in ckpt:
        with torch.no_grad():
            la = ckpt["log_alpha"]
            log_alpha.copy_(la.data if isinstance(la, torch.Tensor) else torch.tensor([la]))
            alpha.copy_(log_alpha.exp())
    resume_step = ckpt.get("global_step", 0)
    resume_ep_count = ckpt.get("ep_count", 0)
    resume_best_return = ckpt.get("best_return", 0.0)
    resume_success_count = ckpt.get("success_count", 0)
    print(f"[Resume] step={resume_step} eps={resume_ep_count} best={resume_best_return:.2f} suc={resume_success_count}", flush=True)
elif args.resume:
    print(f"[Resume] Not found: {args.resume}, starting fresh", flush=True)

# ── Dashboard ──
dashboard.train_stats = {
    "status": "training", "iteration": resume_step, "max_iterations": args.total_timesteps,
    "mean_reward": 0, "alltime_max_reward": resume_best_return, "success_count": resume_success_count,
    "episode_count": resume_ep_count, "best_lift_cm": 0, "reward_terms": {},
    "reward_history": [], "recent_episodes": [],
}
dashboard.start(args.http_port)

# ── Training loop ──
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(os.path.join(args.save_dir, "episodes"), exist_ok=True)

print(f"[SAC-Vision] Starting: {args.total_timesteps} steps, {NUM_ENVS} envs, UTD={NUM_UPDATES}", flush=True)
print(f"[SAC-Squint] C51 atoms={NUM_ATOMS}, v_min={V_MIN}, v_max={V_MAX}, num_q={NUM_Q}", flush=True)
print(f"[SAC-Squint] bootstrap_at_done={BOOTSTRAP_AT_DONE}", flush=True)
for mod in [actor, critic]:
    print(mod, flush=True)

obs_dict = env.reset()[0]
obs = obs_dict["policy"]
obs_rgb = read_cameras(env, target_size=IMAGE_SIZE)

global_step = resume_step
ep_count = resume_ep_count
ep_rewards = torch.zeros(NUM_ENVS, device=DEVICE)
recent_returns = []
best_return = resume_best_return
success_count = resume_success_count
best_lift_cm = 0.0
best_dist_to_rest = 999.0
start_time = time.time()

num_iterations = args.total_timesteps // NUM_ENVS
start_iteration = resume_step // NUM_ENVS

for iteration in range(start_iteration, num_iterations):
    # ── Collect ──
    if global_step < LEARNING_STARTS:
        actions = torch.rand(NUM_ENVS, act_dim, device=DEVICE) * 2 - 1
    else:
        with torch.no_grad():
            actions = get_rollout_action(obs_rgb, obs)

    obs_dict, rewards, terminations, truncations, infos = env.step(actions)
    next_obs = obs_dict["policy"]
    next_obs_rgb = read_cameras(env, target_size=IMAGE_SIZE)

    # Capture reward terms right after env.step (before they get overwritten)
    try:
        rm = env.unwrapped.reward_manager
        last_reward_terms = {}
        for term_idx, (name, term_cfg) in enumerate(zip(rm._term_names, rm._term_cfgs)):
            val = rm._step_reward[:, term_idx].mean().item()
            last_reward_terms[name] = {"weight": term_cfg.weight, "value": round(val, 4)}
    except Exception:
        last_reward_terms = {}

    # Track best lift height (only when grasped) and dist_to_rest
    try:
        from packages.sim.env_setup.maniskill_rewards import _is_grasped, _get_target_qpos, REST_QPOS
        _grasp_mask = _is_grasped(env)
        if _grasp_mask.any():
            _cz_grasped = env.scene["cube"].data.root_pos_w[_grasp_mask, 2].max().item()
            if _cz_grasped * 100 > best_lift_cm:
                best_lift_cm = _cz_grasped * 100
        _tgt = _get_target_qpos(env)[:, :-1]
        _rest = REST_QPOS[:-1].to(_tgt.device)
        _d2r = torch.linalg.norm(_tgt - _rest, dim=-1).min().item()
        if _d2r < best_dist_to_rest:
            best_dist_to_rest = _d2r
    except Exception:
        pass

    # Bootstrap handling (Squint-exact: bootstrap_at_done="always")
    if BOOTSTRAP_AT_DONE == "always":
        dones = torch.zeros_like(terminations, dtype=torch.bool)
    elif BOOTSTRAP_AT_DONE == "on_truncation":
        dones = terminations
    else:  # "never"
        dones = terminations | truncations

    # Store transition with images (torchrl TensorDict)
    transition = TensorDict(
        observations=obs,
        obs_rgb=obs_rgb,
        next_observations=next_obs,
        next_obs_rgb=next_obs_rgb,
        actions=actions.float(),
        rewards=rewards.float(),
        dones=dones,
        batch_size=rewards.shape[0],
        device=DEVICE,
    )
    rb.extend(transition)

    # Track episodes
    ep_rewards += rewards
    done_mask = (terminations | truncations)
    if done_mask.any():
        done_ids = done_mask.nonzero(as_tuple=True)[0]
        for eid in done_ids:
            ret = ep_rewards[eid].item()
            ep_count += 1
            recent_returns.append(ret)
            recent_returns = recent_returns[-100:]
            if ret > best_return:
                best_return = ret
            # Check success (Squint-exact: item_lifted & is_grasped & reached_rest)
            try:
                cube_z = env.scene["cube"].data.root_pos_w[eid, 2].item()
                item_lifted = cube_z >= (0.015 + 1e-3)  # half_size + epsilon
                from packages.sim.env_setup.maniskill_rewards import _is_grasped, _get_target_qpos, REST_QPOS
                is_grasp = _is_grasped(env)[eid].item()
                tgt = _get_target_qpos(env)[eid, :-1]
                rest = REST_QPOS[:-1].to(tgt.device)
                d2r = torch.linalg.norm(tgt - rest).item()
                reached_rest = d2r < 0.2
                if item_lifted and is_grasp and reached_rest:
                    success_count += 1
                    suc_dir = os.path.join(args.save_dir, "episodes", f"success_{success_count}_{global_step}")
                    os.makedirs(suc_dir, exist_ok=True)
                    np.savez_compressed(os.path.join(suc_dir, "data.npz"),
                        obs=obs[eid].cpu().numpy(), action=actions[eid].cpu().numpy(),
                        reward=ret, cube_z=cube_z, dist_to_rest=d2r)
                    print(f"[SUCCESS #{success_count}] step={global_step} ep={ep_count} ret={ret:.2f} z={cube_z:.3f} d2r={d2r:.3f}", flush=True)
            except:
                pass
        ep_rewards[done_mask] = 0

    obs = next_obs
    obs_rgb = next_obs_rgb
    global_step += NUM_ENVS

    # ── Train (Squint-exact loop) ──
    if global_step > LEARNING_STARTS:
        for grad_step in range(NUM_UPDATES):
            data = rb.sample(BATCH_SIZE)

            # Update critic + alpha
            out_main = update_main(data)
            encoded_proj = out_main.get("encoded_proj", None)

            # Update actor (delayed)
            if grad_step % POLICY_FREQ == 0:
                out_main.update(update_actor(data, encoded_proj))

            # Update target networks (_foreach_lerp_ — Squint-exact)
            if grad_step % TARGET_NET_FREQ == 0:
                with torch.no_grad():
                    torch._foreach_lerp_(critic_target_params, critic_online_params, TAU)

        # Sync detached inference copies after each update batch
        from_module(encoder).data.to_module(encoder_detach)
        from_module(projection).data.to_module(projection_detach)
        from_module(actor).data.to_module(actor_detach)

    # ── Log ──
    if iteration % 5 == 0:
        mean_r = np.mean(recent_returns[-20:]) if recent_returns else 0
        sps = (global_step - resume_step) / max(time.time() - start_time, 1)

        rt = last_reward_terms

        with dashboard.stats_lock:
            dashboard.train_stats.update({
                "iteration": global_step, "episode_count": ep_count,
                "mean_reward": mean_r, "alltime_max_reward": best_return,
                "success_count": success_count, "status": "training",
                "best_lift_cm": round(best_lift_cm, 2),
                "best_dist_to_rest": round(best_dist_to_rest, 3),
                "reward_terms": rt,
                "recent_episodes": [{"ep": i, "reward": r} for i, r in enumerate(recent_returns[-15:])],
            })
            dashboard.train_stats["reward_history"].append(
                {"iter": global_step, "mean": mean_r, "max": best_return})
            dashboard.train_stats["reward_history"] = dashboard.train_stats["reward_history"][-500:]

    if iteration % 20 == 0 and recent_returns:
        mean_r = np.mean(recent_returns[-20:]) if recent_returns else 0
        sps = (global_step - resume_step) / max(time.time() - start_time, 1)
        print(f"[SAC] step={global_step} eps={ep_count} mean_r={mean_r:.2f} best={best_return:.2f} suc={success_count} sps={sps:.0f}", flush=True)

    # ── Checkpoint (full state) ──
    if iteration % 200 == 0 and iteration > start_iteration:
        ckpt_path = os.path.join(args.save_dir, f"ckpt_{global_step}.pt")
        torch.save({
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "critic_target": critic_target.state_dict(),
            "actor_opt": actor_optimizer.state_dict(),
            "critic_opt": critic_optimizer.state_dict(),
            "alpha_opt": alpha_optimizer.state_dict(),
            "log_alpha": log_alpha,
            "global_step": global_step,
            "ep_count": ep_count,
            "best_return": best_return,
            "success_count": success_count,
        }, ckpt_path)
        ckpts = sorted(glob.glob(os.path.join(args.save_dir, "ckpt_*.pt")))
        for old in ckpts[:-3]:
            os.remove(old)
        print(f"[Checkpoint] {ckpt_path}", flush=True)

# Final save
torch.save({
    "actor": actor.state_dict(),
    "critic": critic.state_dict(),
    "critic_target": critic_target.state_dict(),
    "actor_opt": actor_optimizer.state_dict(),
    "critic_opt": critic_optimizer.state_dict(),
    "alpha_opt": alpha_optimizer.state_dict(),
    "log_alpha": log_alpha,
    "global_step": global_step,
    "ep_count": ep_count,
    "best_return": best_return,
    "success_count": success_count,
}, os.path.join(args.save_dir, "sac_final.pt"))
print(f"[SAC] Complete! {global_step} steps, {ep_count} eps, best={best_return:.2f}, success={success_count}", flush=True)

env.close()
simulation_app.close()
