#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] setup_rl_env.sh starting"

export LEISAAC_SRC=/workspace/leisaac/source/leisaac/leisaac
export DELIVERY_DIR=/workspace/delivery
export SAVE_DIR=/data/rl_output
export ISAACLAB_RSL_WRAPPER=/workspace/leisaac/dependencies/IsaacLab/source/isaaclab_rl/isaaclab_rl

export DISPLAY=:1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_ENV_PRIVACY_CONSENT=Y
export LEISAAC_ASSETS_ROOT=/workspace/leisaac/assets

echo "[INFO] checking required paths"
test -d "$LEISAAC_SRC"
test -d "$DELIVERY_DIR"
test -f "$DELIVERY_DIR/packages/sim/train_rl.py"
test -f "$DELIVERY_DIR/configs/reward_config.yaml"
test -f "$DELIVERY_DIR/leisaac_overrides/lerobot.py"
test -f "$DELIVERY_DIR/leisaac_overrides/mdp/rewards.py"
test -f "$DELIVERY_DIR/leisaac_overrides/mdp/terminations.py"

# ── 1. Apply overrides FIRST (before anything else) ──
echo "[INFO] applying leisaac overrides"
cp "$DELIVERY_DIR/leisaac_overrides/lerobot.py" \
   "$LEISAAC_SRC/assets/robots/lerobot.py"

cp "$DELIVERY_DIR/leisaac_overrides/mdp/rewards.py" \
   "$LEISAAC_SRC/tasks/lift_cube/mdp/rewards.py"

cp "$DELIVERY_DIR/leisaac_overrides/mdp/terminations.py" \
   "$LEISAAC_SRC/tasks/lift_cube/mdp/terminations.py"

# Patch __init__.py to import rewards (original image doesn't have it)
if ! grep -q 'from .rewards import' "$LEISAAC_SRC/tasks/lift_cube/mdp/__init__.py"; then
  echo 'from .rewards import *' >> "$LEISAAC_SRC/tasks/lift_cube/mdp/__init__.py"
  echo "  __init__.py: added rewards import"
fi

# Clear stale bytecode cache
echo "[INFO] clearing __pycache__"
find "$LEISAAC_SRC" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "[INFO] verifying overrides"
grep -q "object_out_of_reach"    "$LEISAAC_SRC/tasks/lift_cube/mdp/rewards.py"      && echo "  rewards.py: object_out_of_reach OK"
grep -q "lateral_deviation"      "$LEISAAC_SRC/tasks/lift_cube/mdp/rewards.py"      && echo "  rewards.py: lateral_deviation OK"
grep -q "joint_velocity_excess"  "$LEISAAC_SRC/tasks/lift_cube/mdp/rewards.py"      && echo "  rewards.py: joint_velocity_excess OK"
grep -q "gripper_table_collision" "$LEISAAC_SRC/tasks/lift_cube/mdp/rewards.py"     && echo "  rewards.py: gripper_table_collision OK"
grep -q "object_dropped_penalty" "$LEISAAC_SRC/tasks/lift_cube/mdp/rewards.py"      && echo "  rewards.py: object_dropped_penalty OK"

# ── 2. Output dir ──
echo "[INFO] creating output dir"
mkdir -p "$SAVE_DIR"

# ── 3. Python deps ──
echo "[INFO] installing base python deps"
/isaac-sim/python.sh -m pip install -q pyyaml tensordict tensorboard

# ── 4. Xvfb ──
echo "[INFO] installing xvfb"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq xvfb git > /dev/null

echo "[INFO] starting xvfb if needed"
if ! pgrep -f "Xvfb :1" >/dev/null 2>&1; then
  Xvfb :1 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 &
  sleep 2
fi

# ── 5. rsl_rl ──
echo "[INFO] cloning and installing upstream rsl_rl if missing"
if ! /isaac-sim/python.sh -c "import rsl_rl" >/dev/null 2>&1; then
  if [ ! -d /workspace/rsl_rl ]; then
    git clone https://github.com/leggedrobotics/rsl_rl.git /workspace/rsl_rl
  fi
  cd /workspace/rsl_rl
  /isaac-sim/python.sh -m pip install -e .
fi

echo "[INFO] verifying rsl_rl import"
export PYTHONPATH="$ISAACLAB_RSL_WRAPPER:${PYTHONPATH:-}"
/isaac-sim/python.sh -c "from rsl_rl.runners import OnPolicyRunner; print('  rsl_rl OK')"

# ── 6. Env name patch ──
echo "[INFO] patching env name in train_rl.py"
sed -i 's/LeIsaac-SO101-LiftCube-RL-v0/LeIsaac-SO101-LiftCube/g' \
  "$DELIVERY_DIR/packages/sim/train_rl.py"

echo "[INFO] setup complete"
