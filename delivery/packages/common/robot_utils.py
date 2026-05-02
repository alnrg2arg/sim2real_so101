"""Pure action conversion utilities (no Isaac Sim dependency)."""

import numpy as np
import torch

from .constants import SO101_MOTOR_LIMITS, SO101_USD_JOINT_LIMITS


def leisaac_action_to_lerobot(action: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert action from Isaac Lab (radians) to LeRobot (motor degrees)."""
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    processed = np.zeros_like(action)
    for idx, joint_name in enumerate(SO101_USD_JOINT_LIMITS):
        jl = SO101_USD_JOINT_LIMITS[joint_name]
        ml = SO101_MOTOR_LIMITS[joint_name]
        deg = action[:, idx] / np.pi * 180.0
        processed[:, idx] = (deg - jl[0]) / (jl[1] - jl[0]) * (ml[1] - ml[0]) + ml[0]

    return processed


def lerobot_action_to_leisaac(action: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert action from LeRobot (motor degrees) to Isaac Lab (radians)."""
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    processed = np.zeros_like(action)
    for idx, joint_name in enumerate(SO101_USD_JOINT_LIMITS):
        jl = SO101_USD_JOINT_LIMITS[joint_name]
        ml = SO101_MOTOR_LIMITS[joint_name]
        deg = (action[:, idx] - ml[0]) / (ml[1] - ml[0]) * (jl[1] - jl[0]) + jl[0]
        processed[:, idx] = deg / 180.0 * np.pi

    return processed
