"""Fast SAC for Isaac Lab — Squint-exact implementation (state-only, no vision).

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

# ── Isaac Lab init ──
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"enable_cameras": CFG.get("enable_cameras", False)})
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

if not CFG.get("enable_cameras", False):
    for cam in ["front", "side", "wrist"]:
        if hasattr(env_cfg.scene, cam):
            setattr(env_cfg.scene, cam, None)
    for attr in list(vars(env_cfg.observations.policy)):
        if attr in ('front', 'wrist', 'side') or 'image' in attr.lower():
            setattr(env_cfg.observations.policy, attr, None)
if hasattr(env_cfg.observations, 'subtask_terms'):
    env_cfg.observations.subtask_terms = None

env = ManagerBasedRLEnv(cfg=env_cfg)
apply_motor_limits(env, CFG)

obs_dim = env.observation_manager.compute()["policy"].shape[-1]
act_dim = env.action_manager.total_action_dim

print(f"[Env] {NUM_ENVS} envs, obs={obs_dim}, act={act_dim}", flush=True)

# ── Seeding ──
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# ── Hyperparams (Squint-exact) ──
GAMMA = 0.9
TAU = 0.01
BATCH_SIZE = 512
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

# ── Networks (Squint-exact) ──
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class StateProjection(nn.Module):
    """State-only projection (replaces Squint's Projection which has CNN+state)."""
    def __init__(self, n_state, device=None):
        super().__init__()
        self.repr_dim = 256
        self.state_proj = nn.Sequential(
            nn.Linear(n_state, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU(),
        )

    def forward(self, state):
        return self.state_proj(state)


class Actor(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        hidden_dim = 256

        self.proj = StateProjection(n_obs, device=device)
        self.fc = nn.Sequential(
            nn.Linear(self.proj.repr_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, n_act, device=device)
        self.fc_logstd = nn.Linear(hidden_dim, n_act, device=device)

        # Action space [-1, 1] for Isaac Lab
        self.register_buffer("action_scale", torch.ones(n_act, device=device))
        self.register_buffer("action_bias", torch.zeros(n_act, device=device))

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.apply(weight_init)

    def forward(self, state, get_log_std=False):
        x = self.proj(state)
        x = self.fc(x)
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
    """Distributional C51 Ensemble-Q-network critic with vmap (Squint-exact)."""
    def __init__(self, n_obs, n_act, num_atoms, v_min, v_max, num_q=2, device=None):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_q = num_q
        self.v_min = v_min
        self.v_max = v_max
        self.q_support = torch.linspace(v_min, v_max, num_atoms, device=device)

        self.proj = StateProjection(n_obs, device=device)
        self.proj.apply(weight_init)

        q_input_dim = self.proj.repr_dim + n_act

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
        lines.append(f"  (proj): {self.proj}")
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

    def forward(self, state, actions):
        proj = self.proj(state)
        x = torch.cat([proj, actions], dim=-1)
        return torch.vmap(self._vmap_q, (0, None))(self.q_params, x)

    def get_q_values(self, state, actions, detach_critic=False):
        """Expected Q-values: [num_q, batch]. detach_critic freezes critic but keeps action grad."""
        if detach_critic:
            with torch.no_grad():
                proj = self.proj(state)
            x = torch.cat([proj, actions], dim=-1)
            logits = torch.vmap(self._vmap_q, (0, None))(self.q_params.data, x)
        else:
            logits = self.forward(state, actions)
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.q_support, dim=-1)

    def categorical(self, state, actions, rewards, bootstrap, discount):
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

        logits = self.forward(state, actions)  # [num_q, batch, atoms]
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
actor = Actor(n_obs=obs_dim, n_act=act_dim, device=DEVICE)
critic = Critic(n_obs=obs_dim, n_act=act_dim,
                num_atoms=NUM_ATOMS, v_min=V_MIN, v_max=V_MAX,
                num_q=NUM_Q, device=DEVICE)

# Target critic (Squint-exact)
critic_target = Critic(n_obs=obs_dim, n_act=act_dim,
                       num_atoms=NUM_ATOMS, v_min=V_MIN, v_max=V_MAX,
                       num_q=NUM_Q, device=DEVICE)
critic_target.load_state_dict(critic.state_dict())
critic_online_params = list(critic.parameters())
critic_target_params = list(critic_target.parameters())

# Inference copies (Squint: weight-sharing via from_module for detached inference)
actor_detach = Actor(n_obs=obs_dim, n_act=act_dim, device=DEVICE)
actor_eval_net = Actor(n_obs=obs_dim, n_act=act_dim, device=DEVICE).eval()
from_module(actor).data.to_module(actor_detach)
from_module(actor).data.to_module(actor_eval_net)

# Auto entropy (Squint-exact)
target_entropy = -float(act_dim)
log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
alpha = log_alpha.detach().exp()

# Optimizers (Squint-exact: critic optimizer includes proj params)
critic_optimizer = optim.Adam(critic.parameters(), lr=Q_LR,
                              capturable=USE_CUDAGRAPHS and not USE_COMPILE)
actor_optimizer = optim.Adam(actor.parameters(), lr=POLICY_LR,
                             capturable=USE_CUDAGRAPHS and not USE_COMPILE)
alpha_optimizer = optim.Adam([log_alpha], lr=ALPHA_LR,
                             capturable=USE_CUDAGRAPHS and not USE_COMPILE)

# ── Replay Buffer (torchrl — Squint-exact) ──
rb = ReplayBuffer(storage=LazyTensorStorage(BUFFER_SIZE, device=DEVICE))

# ── Update functions (Squint-exact ordering) ──

def update_main(data):
    """Critic + alpha update (Squint-exact)."""
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            next_state = data["next_observations"]
            next_state_actions, next_state_log_pi, _ = actor.get_action(next_state)

            bootstrap = (~data["dones"]).float()
            discount = GAMMA
            rewards = data["rewards"].flatten()

            entropy_bonus = alpha * next_state_log_pi.flatten()
            rewards_with_entropy = rewards - bootstrap.flatten() * discount * entropy_bonus

            target_distributions = critic_target.categorical(
                next_state, next_state_actions,
                rewards_with_entropy, bootstrap, discount
            )

        state = data["observations"]

        # Shape: [num_q, batch, num_atoms]
        q_outputs = critic(state, data["actions"])
        q_log_probs = F.log_softmax(q_outputs, dim=-1)

        # Cross-entropy: sum over num_atoms, mean over batch -> [num_q]
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
            _, log_pi, _ = actor.get_action(state)
        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()
    alpha.copy_(log_alpha.detach().exp())

    return TensorDict(critic_loss=critic_loss.detach(), q_max=q_max, q_min=q_min,
                      alpha=alpha.detach(), alpha_loss=alpha_loss.detach(),
                      encoded_state=state.detach())


def update_actor(data, encoded_state):
    """Actor update with detached critic (Squint-exact)."""
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        state = encoded_state
        pi, log_pi, _ = actor.get_action(state)
        q_values = critic.get_q_values(state, pi, detach_critic=True)

        # Mean (No CDQ — Squint-exact)
        critic_value = q_values.mean(dim=0)
        actor_loss = (alpha * log_pi - critic_value).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return TensorDict(actor_loss=actor_loss.detach())


def get_rollout_action(state):
    action, _, _ = actor_detach.get_action(state)
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
resume_saved_count = 0

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
    resume_saved_count = ckpt.get("saved_count", 0)
    print(f"[Resume] step={resume_step} eps={resume_ep_count} best={resume_best_return:.2f} suc={resume_saved_count}", flush=True)
elif args.resume:
    print(f"[Resume] Not found: {args.resume}, starting fresh", flush=True)

# ── Dashboard ──
dashboard.train_stats = {
    "status": "training", "iteration": resume_step, "max_iterations": args.total_timesteps,
    "mean_reward": 0, "alltime_max_reward": resume_best_return, "saved_count": resume_saved_count,
    "episode_count": resume_ep_count, "best_lift_cm": 0, "reward_terms": {},
    "reward_history": [], "recent_episodes": [],
}
dashboard.start(args.http_port)

# ── Training loop ──
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(os.path.join(args.save_dir, "episodes"), exist_ok=True)

print(f"[SAC-Squint] Starting: {args.total_timesteps} steps, {NUM_ENVS} envs, UTD={NUM_UPDATES}", flush=True)
print(f"[SAC-Squint] C51 atoms={NUM_ATOMS}, v_min={V_MIN}, v_max={V_MAX}, num_q={NUM_Q}", flush=True)
print(f"[SAC-Squint] bootstrap_at_done={BOOTSTRAP_AT_DONE}", flush=True)
for mod in [actor, critic]:
    print(mod, flush=True)

obs_dict = env.reset()[0]
obs = obs_dict["policy"]

global_step = resume_step
ep_count = resume_ep_count
ep_rewards = torch.zeros(NUM_ENVS, device=DEVICE)
recent_returns = []
best_return = resume_best_return
saved_count = resume_saved_count
best_lift_cm = 0.0
best_dist_to_rest = 999.0
start_time = time.time()

# ── Trajectory buffers for episode recording ──
MAX_EP_STEPS = 100  # 10s at 10Hz
traj_qpos = torch.zeros(NUM_ENVS, MAX_EP_STEPS, 6, device=DEVICE)
traj_actions = torch.zeros(NUM_ENVS, MAX_EP_STEPS, act_dim, device=DEVICE)
traj_rewards = torch.zeros(NUM_ENVS, MAX_EP_STEPS, device=DEVICE)
traj_cube_pos = torch.zeros(NUM_ENVS, MAX_EP_STEPS, 3, device=DEVICE)
traj_tcp_pos = torch.zeros(NUM_ENVS, MAX_EP_STEPS, 3, device=DEVICE)
traj_target_qpos = torch.zeros(NUM_ENVS, MAX_EP_STEPS, 6, device=DEVICE)
traj_is_grasped = torch.zeros(NUM_ENVS, MAX_EP_STEPS, device=DEVICE)
traj_cube_quat = torch.zeros(NUM_ENVS, MAX_EP_STEPS, 4, device=DEVICE)
traj_joint_vel = torch.zeros(NUM_ENVS, MAX_EP_STEPS, 6, device=DEVICE)
traj_step_idx = torch.zeros(NUM_ENVS, dtype=torch.long, device=DEVICE)

# Lift hold tracking: lifted(15cm) + grasped + pan<5°
lh_counter = torch.zeros(NUM_ENVS, dtype=torch.long, device=DEVICE)         # consecutive lift hold steps
fh_first = torch.full((NUM_ENVS,), -1, dtype=torch.long, device=DEVICE) # step when lift+pan first achieved
fh_held = torch.zeros(NUM_ENVS, dtype=torch.long, device=DEVICE)        # cumulative steps with lift+pan

# Server-side cumulative counters (persist across dashboard refresh)
cum_reaching = 0
cum_open = 0
cum_retry = 0
cum_grasped = 0
cum_lifted = 0
cum_pan_ok = 0      # lifted(15cm) + grasped + pan < 5°

num_iterations = args.total_timesteps // NUM_ENVS
start_iteration = resume_step // NUM_ENVS

for iteration in range(start_iteration, num_iterations):
    # ── Collect ──
    if global_step < LEARNING_STARTS:
        actions = torch.rand(NUM_ENVS, act_dim, device=DEVICE) * 2 - 1
    else:
        with torch.no_grad():
            actions = get_rollout_action(obs)

    obs_dict, rewards, terminations, truncations, infos = env.step(actions)
    next_obs = obs_dict["policy"]

    # Record trajectory for every env (clamped to MAX_EP_STEPS)
    try:
        t_idx = traj_step_idx.clamp(max=MAX_EP_STEPS - 1)
        env_ids = torch.arange(NUM_ENVS, device=DEVICE)
        traj_qpos[env_ids, t_idx] = env.scene["robot"].data.joint_pos
        traj_actions[env_ids, t_idx] = actions
        traj_rewards[env_ids, t_idx] = rewards
        _env_origins = env.scene.env_origins  # (N, 3)
        traj_cube_pos[env_ids, t_idx] = env.scene["cube"].data.root_pos_w - _env_origins
        ee = env.scene["ee_frame"]
        traj_tcp_pos[env_ids, t_idx] = (ee.data.target_pos_w[:, 0, :] + ee.data.target_pos_w[:, 1, :]) * 0.5 - _env_origins
        from packages.sim.env_setup.maniskill_rewards import _get_target_qpos, _is_grasped
        traj_target_qpos[env_ids, t_idx] = _get_target_qpos(env)
        traj_is_grasped[env_ids, t_idx] = _is_grasped(env).float()
        traj_cube_quat[env_ids, t_idx] = env.scene["cube"].data.root_quat_w
        traj_joint_vel[env_ids, t_idx] = env.scene["robot"].data.joint_vel
        traj_step_idx += 1
    except Exception:
        pass

    # Capture reward terms right after env.step (before they get overwritten)
    try:
        rm = env.unwrapped.reward_manager
        last_reward_terms = {}
        for term_idx, (name, term_cfg) in enumerate(zip(rm._term_names, rm._term_cfgs)):
            val = rm._step_reward[:, term_idx].mean().item()
            last_reward_terms[name] = {"weight": term_cfg.weight, "value": round(val, 4)}
    except Exception:
        last_reward_terms = {}

    # ── Cumul + lift hold tracking (no try/except) ──
    from packages.sim.env_setup.maniskill_rewards import _is_grasped as _check_grasp, _fold_3joint, _get_target_qpos, REST_QPOS
    _ee = env.scene["ee_frame"]
    _tcp = (_ee.data.target_pos_w[:, 0, :] + _ee.data.target_pos_w[:, 1, :]) * 0.5
    _obj_pos = env.scene["cube"].data.root_pos_w
    _dists = torch.norm(_obj_pos - _tcp, dim=1)
    _cz = _obj_pos[:, 2]
    _gr = _check_grasp(env)
    _pe, _le, _ee = _fold_3joint(env)
    _gripper_qpos = env.scene["robot"].data.joint_pos[:, -1]

    cum_reaching += int((_dists < 0.10).sum().item())
    cum_open += int(((_dists < 0.10) & (_gripper_qpos > 0.5)).sum().item())
    _retry_val = last_reward_terms.get("grasp_retry", {}).get("value", 0) if isinstance(last_reward_terms.get("grasp_retry"), dict) else 0
    cum_retry += int(_retry_val > 0.001)
    cum_grasped += int(_gr.sum().item())
    cum_lifted += int(((_cz >= 0.15) & _gr).sum().item())
    cum_pan_ok += int(((_cz >= 0.15) & _gr & (_pe < 0.262)).sum().item())

    # Fold hold tracking: pan<5°
    _lh_holding = (_cz >= 0.15) & _gr & (_pe < 0.262)
    lh_counter[_lh_holding] += 1
    lh_counter[~_lh_holding] = 0
    _first_time = _lh_holding & (fh_first < 0)
    fh_first[_first_time] = env.episode_length_buf[_first_time]
    fh_held[_lh_holding] += 1

    # Best lift height
    if _gr.any():
        _cz_grasped = _cz[_gr].max().item()
        if _cz_grasped * 100 > best_lift_cm:
            best_lift_cm = _cz_grasped * 100
    _tgt = _get_target_qpos(env)[:, :-1]
    _rest = REST_QPOS[:-1].to(_tgt.device)
    _d2r = torch.linalg.norm(_tgt - _rest, dim=-1).min().item()
    if _d2r < best_dist_to_rest:
        best_dist_to_rest = _d2r

    # Per-step info for dashboard
    _fold_stage_info = {
        "fold_ok": int(_lh_holding.sum().item()),
        "pan_err": round(_pe[_gr].mean().item(), 3) if _gr.any() else 0,
        "pan_err": round(_pe[_gr].mean().item(), 3) if _gr.any() else 0,
    }

    # Bootstrap handling (Squint-exact: bootstrap_at_done="always")
    if BOOTSTRAP_AT_DONE == "always":
        dones = torch.zeros_like(terminations, dtype=torch.bool)
    elif BOOTSTRAP_AT_DONE == "on_truncation":
        dones = terminations
    else:  # "never"
        dones = terminations | truncations

    # Store transition (torchrl TensorDict — Squint-exact)
    transition = TensorDict(
        observations=obs,
        next_observations=next_obs,
        actions=actions.float(),
        rewards=rewards.float(),
        dones=dones,
        batch_size=rewards.shape[0],
        device=DEVICE,
    )
    rb.extend(transition)

    # ── Status log (every 100 iterations) ──
    if iteration % 100 == 0:
        n_gr = int(_gr.sum().item())
        n_lh = int(_lh_holding.sum().item())
        max_lh = int(lh_counter.max().item())
        mpe = _pe[_gr].min().item() if _gr.any() else 99.0
        print(f"[Status] grasp={n_gr} fold={n_lh} max_fh={max_lh} pan={mpe:.3f}/0.262 saved={saved_count} cg={cum_grasped} cfold={cum_pan_ok}", flush=True)

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
            # Lift hold + pan → save trajectory (use train-side tracking)
            # Lift hold + pan ratio check
            _fs = fh_first[eid].item()
            _fh = fh_held[eid].item()
            _ep_len = MAX_EP_STEPS  # 100 (env resets before we read, so use fixed max)
            _remaining = max(_ep_len - _fs, 1) if _fs >= 0 else 1
            _ratio = _fh / _remaining if _fs >= 0 else 0
            print(f"[EP_END] env={eid.item()} fs={_fs} fh={_fh} rem={_remaining} ratio={_ratio:.0%} ret={ret:.0f}", flush=True)
            if _fs >= 0 and _fh >= 5 and _remaining > 5 and _ratio >= 0.80:
                saved_count += 1
                ep_len = min(traj_step_idx[eid].item(), MAX_EP_STEPS)
                try:
                    _cz_save = env.scene["cube"].data.root_pos_w[eid, 2].item()
                    save_dir = os.path.join(args.save_dir, "episodes", f"lift_hold_{saved_count}_{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    np.savez_compressed(os.path.join(save_dir, "trajectory.npz"),
                        qpos=traj_qpos[eid, :ep_len].cpu().numpy(),
                        actions=traj_actions[eid, :ep_len].cpu().numpy(),
                        rewards=traj_rewards[eid, :ep_len].cpu().numpy(),
                        cube_pos=traj_cube_pos[eid, :ep_len].cpu().numpy(),
                        cube_quat=traj_cube_quat[eid, :ep_len].cpu().numpy(),
                        tcp_pos=traj_tcp_pos[eid, :ep_len].cpu().numpy(),
                        target_qpos=traj_target_qpos[eid, :ep_len].cpu().numpy(),
                        joint_vel=traj_joint_vel[eid, :ep_len].cpu().numpy(),
                        is_grasped=traj_is_grasped[eid, :ep_len].cpu().numpy(),
                        ep_length=ep_len, final_cube_z=_cz_save, pan_ratio=_ratio)
                    print(f"[SAVED #{saved_count}] step={global_step} env={eid.item()} z={_cz_save:.3f} ratio={_ratio:.0%} ({_fh}/{_remaining}steps) ret={ret:.0f}", flush=True)
                except Exception as _se:
                    print(f"[SAVE ERROR] {_se}", flush=True)

            if ret > best_return:
                best_return = ret
                # Save best checkpoint
                try:
                    torch.save({
                        "actor": actor.state_dict(), "critic": critic.state_dict(),
                        "critic_target": critic_target.state_dict(),
                        "log_alpha": log_alpha, "global_step": global_step,
                        "ep_count": ep_count, "best_return": best_return,
                        "saved_count": saved_count,
                    }, os.path.join(args.save_dir, "best_ckpt.pt"))
                except:
                    pass
        # Reset trajectory buffers + hold counter for done envs
        traj_step_idx[done_mask] = 0
        lh_counter[done_mask] = 0
        fh_first[done_mask] = -1
        fh_held[done_mask] = 0
        ep_rewards[done_mask] = 0

    obs = next_obs
    global_step += NUM_ENVS

    # Stop if 10000 episodes saved
    if saved_count >= 10000:
        print(f"[DONE] {saved_count} fold_hold episodes saved. Stopping.", flush=True)
        break

    # ── Train (Squint-exact loop) ──
    if global_step > LEARNING_STARTS:
        for grad_step in range(NUM_UPDATES):
            data = rb.sample(BATCH_SIZE)

            # Update critic + alpha
            out_main = update_main(data)
            encoded_state = out_main.get("encoded_state", None)

            # Update actor (delayed)
            if grad_step % POLICY_FREQ == 0:
                out_main.update(update_actor(data, encoded_state))

            # Update target networks (_foreach_lerp_ — Squint-exact)
            if grad_step % TARGET_NET_FREQ == 0:
                with torch.no_grad():
                    torch._foreach_lerp_(critic_target_params, critic_online_params, TAU)

    # ── Log ──
    if iteration % 5 == 0:
        mean_r = np.mean(recent_returns[-20:]) if recent_returns else 0
        sps = (global_step - resume_step) / max(time.time() - start_time, 1)

        rt = last_reward_terms

        with dashboard.stats_lock:
            dashboard.train_stats.update({
                "iteration": global_step, "episode_count": ep_count, "num_envs": NUM_ENVS,
                "mean_reward": mean_r, "alltime_max_reward": best_return,
                "saved_count": saved_count, "status": "training",
                "best_lift_cm": round(best_lift_cm, 2),
                "best_dist_to_rest": round(best_dist_to_rest, 3),
                "reward_terms": rt,
                "fold_stage": _fold_stage_info,
                "recent_episodes": [{"ep": i, "reward": r} for i, r in enumerate(recent_returns[-15:])],
                "cumulative": {
                    "reaching": cum_reaching, "open": cum_open, "retry": cum_retry,
                    "grasped": cum_grasped, "lifted": cum_lifted,
                    "fold_ok": cum_pan_ok,
                    "saved": saved_count,
                    "saved": saved_count,
                },
            })
            dashboard.train_stats["reward_history"].append(
                {"iter": global_step, "mean": mean_r, "max": best_return})
            dashboard.train_stats["reward_history"] = dashboard.train_stats["reward_history"][-500:]

    if iteration % 20 == 0 and recent_returns:
        mean_r = np.mean(recent_returns[-20:]) if recent_returns else 0
        sps = (global_step - resume_step) / max(time.time() - start_time, 1)
        print(f"[SAC] step={global_step} eps={ep_count} mean_r={mean_r:.2f} best={best_return:.2f} suc={saved_count} sps={sps:.0f}", flush=True)

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
            "saved_count": saved_count,
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
    "saved_count": saved_count,
}, os.path.join(args.save_dir, "sac_final.pt"))
print(f"[SAC] Complete! {global_step} steps, {ep_count} eps, best={best_return:.2f}, saved={saved_count}", flush=True)

env.close()
simulation_app.close()
