#!/usr/bin/env python3
"""BC pretrain using rsl_rl v5 MLPModel → directly loadable as PPO checkpoint."""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Pure PyTorch — no rsl_rl/Isaac imports needed for BC training

parser = argparse.ArgumentParser()
parser.add_argument("--demo-dir", type=str, default=None, help="Raw demo dir (6-dim states)")
parser.add_argument("--replay-data", type=str, default=None, help="Replayed full obs data (.npz from replay_demos.py)")
parser.add_argument("--save-path", type=str, default="/data/rl_output/logs/bc_for_ppo.pt")
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--obs-dim", type=int, default=43)
parser.add_argument("--act-dim", type=int, default=6)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load demo data ──
if args.replay_data and os.path.exists(args.replay_data):
    # Full obs from replay_demos.py (correct 43-dim)
    data = np.load(args.replay_data)
    states = data["observations"]
    actions = data["actions"]
    episodes = []
    print(f"Loaded replay data: {states.shape[0]} samples, obs={states.shape[1]}, act={actions.shape[1]}")
    args.obs_dim = states.shape[1]
    args.act_dim = actions.shape[1]
elif args.demo_dir:
    # Raw demo (6-dim only, will pad — NOT recommended)
    all_states = []
    all_actions = []
    episodes = sorted(glob.glob(os.path.join(args.demo_dir, "episode_*")))
    print(f"Found {len(episodes)} episodes (raw 6-dim, padding to {args.obs_dim})")
    for ep_dir in episodes:
        npz_path = os.path.join(ep_dir, "data.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        all_states.append(data["states"])
        all_actions.append(data["actions"])
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    if states.shape[1] < args.obs_dim:
        pad = np.zeros((states.shape[0], args.obs_dim - states.shape[1]), dtype=np.float32)
        states = np.concatenate([states, pad], axis=1)
else:
    raise ValueError("Provide --replay-data or --demo-dir")

print(f"Total samples: {states.shape[0]}, obs: {states.shape}, act: {actions.shape}")

X = torch.tensor(states, dtype=torch.float32, device=device)
Y = torch.tensor(actions, dtype=torch.float32, device=device)
loader = DataLoader(TensorDataset(X, Y), batch_size=args.batch_size, shuffle=True)

# ── Build MLP matching rsl_rl v5 actor architecture ──
# Architecture: [256, 128, 64] with ELU, same as PPO actor
layers = []
dims = [args.obs_dim] + [256, 128, 64]
for i in range(len(dims) - 1):
    layers.append(nn.Linear(dims[i], dims[i+1]))
    layers.append(nn.ELU())
layers.append(nn.Linear(dims[-1], args.act_dim))
actor_mlp = nn.Sequential(*layers).to(device)

print(f"Actor MLP: {actor_mlp}")

# ── Train ──
optimizer = torch.optim.Adam(actor_mlp.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()

print(f"\nTraining BC for {args.epochs} epochs...")
for epoch in range(args.epochs):
    total_loss = 0
    count = 0
    for xb, yb in loader:
        pred = actor_mlp(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        count += xb.size(0)

    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/count:.6f}")

# ── Map to rsl_rl v5 actor_state_dict format ──
# Sequential indices: 0=Linear, 1=ELU, 2=Linear, 3=ELU, 4=Linear, 5=ELU, 6=Linear
# rsl_rl v5 MLPModel keys: mlp.0.weight, mlp.0.bias, mlp.2.weight, ...
bc_state = actor_mlp.state_dict()
actor_state = {}

# obs_normalizer (initialized to identity)
actor_state["obs_normalizer._mean"] = torch.zeros(1, args.obs_dim)
actor_state["obs_normalizer._var"] = torch.ones(1, args.obs_dim)
actor_state["obs_normalizer._std"] = torch.ones(1, args.obs_dim)
actor_state["obs_normalizer.count"] = torch.tensor(1.0)

# distribution std
actor_state["distribution.std_param"] = torch.zeros(args.act_dim)

# MLP weights — Sequential key "0.weight" → "mlp.0.weight"
for k, v in bc_state.items():
    actor_state[f"mlp.{k}"] = v

# Critic (untrained, same arch but output_dim=1)
critic_layers = []
for i in range(len(dims) - 1):
    critic_layers.append(nn.Linear(dims[i], dims[i+1]))
    critic_layers.append(nn.ELU())
critic_layers.append(nn.Linear(dims[-1], 1))
critic_mlp = nn.Sequential(*critic_layers).to(device)

critic_state = {}
critic_state["obs_normalizer._mean"] = torch.zeros(1, args.obs_dim)
critic_state["obs_normalizer._var"] = torch.ones(1, args.obs_dim)
critic_state["obs_normalizer._std"] = torch.ones(1, args.obs_dim)
critic_state["obs_normalizer.count"] = torch.tensor(1.0)
for k, v in critic_mlp.state_dict().items():
    critic_state[f"mlp.{k}"] = v

# ── Save as rsl_rl v5 checkpoint ──
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

checkpoint = {
    "actor_state_dict": actor_state,
    "critic_state_dict": critic_state,
    "optimizer_state_dict": {"state": {}, "param_groups": []},
    "iter": 0,
    "infos": {"bc_pretrained": True, "demo_samples": len(X)},
}

torch.save(checkpoint, args.save_path)
print(f"\nSaved rsl_rl v5 compatible checkpoint: {args.save_path}")
print(f"Use with: --resume {args.save_path}")
