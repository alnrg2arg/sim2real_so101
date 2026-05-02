#!/usr/bin/env python3
"""Behavior Cloning pretrain from demo episodes → save checkpoint for PPO resume."""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument("--demo-dir", type=str, required=True, help="Path to demo episodes (each has data.npz)")
parser.add_argument("--save-path", type=str, default="/data/rl_output/logs/bc_pretrained.pt")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])
parser.add_argument("--obs-dim", type=int, default=43, help="RL policy obs dim (joint states etc)")
parser.add_argument("--act-dim", type=int, default=6, help="Action dim (6 joints)")
args = parser.parse_args()

# Load all demo episodes
all_states = []
all_actions = []

episodes = sorted(glob.glob(os.path.join(args.demo_dir, "episode_*")))
print(f"Found {len(episodes)} episodes")

for ep_dir in episodes:
    npz_path = os.path.join(ep_dir, "data.npz")
    if not os.path.exists(npz_path):
        continue
    data = np.load(npz_path)
    states = data["states"]   # (T, 6) joint positions
    actions = data["actions"]  # (T, 6) joint actions

    # For BC: state → action mapping
    # States are 6-dim (joint pos), but RL obs is 43-dim
    # We'll pad with zeros for the extra dims and the MLP will learn
    # Or better: just use joint_pos as a subset of obs
    all_states.append(states)
    all_actions.append(actions)

states = np.concatenate(all_states, axis=0)
actions = np.concatenate(all_actions, axis=0)
print(f"Total samples: {states.shape[0]}")
print(f"States shape: {states.shape}, Actions shape: {actions.shape}")

# Pad states to match RL obs dim (43) if needed
# RL obs: joint_pos(6) + joint_vel(6) + joint_pos_rel(6) + joint_vel_rel(6) + actions(6) + ee_frame(7) + joint_pos_target(6) = 43
# Demo only has joint_pos(6). Pad rest with zeros.
if states.shape[1] < args.obs_dim:
    pad = np.zeros((states.shape[0], args.obs_dim - states.shape[1]), dtype=np.float32)
    states_padded = np.concatenate([states, pad], axis=1)
else:
    states_padded = states

# Convert to tensors
device = "cuda" if torch.cuda.is_available() else "cpu"
X = torch.tensor(states_padded, dtype=torch.float32, device=device)
Y = torch.tensor(actions, dtype=torch.float32, device=device)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Simple MLP (same architecture as PPO actor)
layers = []
in_dim = args.obs_dim
for h in args.hidden_dims:
    layers.append(nn.Linear(in_dim, h))
    layers.append(nn.ELU())
    in_dim = h
layers.append(nn.Linear(in_dim, args.act_dim))
model = nn.Sequential(*layers).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()

# Train
print(f"Training BC for {args.epochs} epochs...")
for epoch in range(args.epochs):
    total_loss = 0
    count = 0
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        count += xb.size(0)
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/count:.6f}")

# Save as a state dict that can be loaded into rsl_rl actor
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

# Map to rsl_rl MLP naming: mlp.0, mlp.1, ...
rsl_state = {}
for i, (k, v) in enumerate(model.state_dict().items()):
    rsl_state[f"mlp.{k.split('.')[0]}.{k.split('.')[1]}"] = v

torch.save({
    "bc_model": model.state_dict(),
    "rsl_actor_mlp": rsl_state,
    "obs_dim": args.obs_dim,
    "act_dim": args.act_dim,
    "hidden_dims": args.hidden_dims,
    "demo_episodes": len(episodes),
    "demo_samples": states.shape[0],
}, args.save_path)

print(f"\nSaved BC checkpoint: {args.save_path}")
print(f"Next: use --resume {args.save_path} with PPO training")
