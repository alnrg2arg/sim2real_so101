#!/usr/bin/env bash
set -euo pipefail

export DELIVERY_DIR=/workspace/delivery
export SAVE_DIR=/data/rl_output
export ISAACLAB_RSL_WRAPPER=/workspace/leisaac/dependencies/IsaacLab/source/isaaclab_rl/isaaclab_rl

export DISPLAY=:1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export OMNI_ENV_PRIVACY_CONSENT=Y
export PYTHONPATH="$DELIVERY_DIR:$ISAACLAB_RSL_WRAPPER:${PYTHONPATH:-}"

cd "$DELIVERY_DIR"

exec /isaac-sim/python.sh packages/sim/train_rl.py \
  --config configs/reward_config.yaml \
  --save-dir "$SAVE_DIR" \
  "$@"
