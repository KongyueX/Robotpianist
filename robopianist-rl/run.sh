#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNS_DIR="${SCRIPT_DIR}/robopianist_runs"
ROOT_DIR="${RUNS_DIR}/rl"
WANDB_ROOT="${RUNS_DIR}/wandb"

mkdir -p "$ROOT_DIR" "$WANDB_ROOT"

WANDB_DIR="$WANDB_ROOT" \
WANDB_MODE=offline \
MUJOCO_GL=egl \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDA_VISIBLE_DEVICES=0 \
MUJOCO_EGL_DEVICE_ID=0 \
python train.py \
  --root-dir "$ROOT_DIR" \
  --mode offline \
  --name "SAC-Twinkle-offline-seed42" \
  --warmstart-steps 5000 \
  --max-steps 5000000 \
  --discount 0.8 \
  --agent-config.critic-dropout-rate 0.01 \
  --agent-config.critic-layer-norm \
  --agent-config.hidden-dims 256 256 256 \
  --trim-silence \
  --gravity-compensation \
  --control-timestep 0.05 \
  --n-steps-lookahead 10 \
  --environment-name "RoboPianist-debug-TwinkleTwinkleRousseau-v0" \
  --action-reward-observation \
  --primitive-fingertip-collisions \
  --eval-episodes 1 \
  --eval-interval 1000000 \
  --camera-id "piano/back" \
  --tqdm-bar
