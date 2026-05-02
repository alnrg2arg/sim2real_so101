"""Constants and global configuration for the RL reward system."""

import math

# ── Physical constants ──
CUBE_INITIAL_HEIGHT = 0.056  # meters, cube resting height on table
DEG2RAD = math.pi / 180.0

# ── Gripper joint limits ──
GRIPPER_OPEN = 1.05
GRIPPER_CLOSED = 0.26

# ── Reward scaling ──
DT_COMP = 60  # 1/dt compensation so milestone rewards have correct magnitude

# ── Effort limits ──
REAL_EFFORT_NM = 3.5
ARM_SELF_WEIGHT_NM = 8.0

# ── Dopamine RPE scaling ──
# Set USE_DOPAMINE = True via configure_env(cfg={"use_dopamine": True})
USE_DOPAMINE = False
DOPAMINE_ALPHA = 0.5   # exponent in (1-h)^alpha
DOPAMINE_FLOOR = 0.1   # lambda: minimum reward fraction
EMA_BETA = 0.99        # EMA smoothing (~100 episode window)
