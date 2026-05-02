#!/bin/bash
cd /workspace/delivery
export LEISAAC_ASSETS_ROOT=/workspace/leisaac/assets
export PYTHONPATH=/workspace/delivery:/workspace/leisaac/dependencies/IsaacLab/source/isaaclab_rl/isaaclab_rl
export DISPLAY=:1
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y

while true; do
  LATEST=$(ls /data/rl_output/logs/model_*.pt 2>/dev/null | sed "s/.*model_//;s/\.pt//" | sort -n | tail -1)
  if [ -n "$LATEST" ]; then
    RESUME_PT=/data/rl_output/logs/model_${LATEST}.pt
  else
    RESUME_PT=/data/rl_output/logs/bc_for_ppo.pt
  fi
  echo "[AUTO] Resuming from $RESUME_PT at $(date)" >> /tmp/train_171.log
  /isaac-sim/python.sh packages/sim/train_rl.py     --config configs/reward_config.yaml     --save-dir /data/rl_output     --num-envs 1     --resume $RESUME_PT     >> /tmp/train_171.log 2>&1
  echo "[AUTO] Crashed (exit $?) at $(date). Restarting in 10s..." >> /tmp/train_171.log
  sleep 10
done
