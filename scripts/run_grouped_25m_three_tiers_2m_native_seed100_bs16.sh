#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root regardless of the caller's current directory.
MACFLOW_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACFLOW_PROJECT_DIR="$(cd "${MACFLOW_SCRIPT_DIR}/.." && pwd)"
MACFLOW_NATIVE_ROOT="${MACFLOW_PROJECT_DIR}/../mac-flow/data/native_smac/25m"
MACFLOW_CONDA_ENV="${MACFLOW_CONDA_ENV:-macflow_gpu_grouped}"

cd "${MACFLOW_PROJECT_DIR}"
mkdir -p logs

# Use the coherent system CUDA 12.8 stack that passed the grouped GPU smoke test.
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export WANDB_MODE="${WANDB_MODE:-disabled}"
export SC2PATH="${SC2PATH:-/home/msi/StarCraftII/StarCraftII}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

MACFLOW_REQUIRED_FILES=(
  obs.npy
  states.npy
  actions.npy
  rewards.npy
  legals.npy
  terminals.npy
  truncations.npy
  filled.npy
  path_lengths.npy
)

for MACFLOW_TIER in Good Medium Poor; do
  for MACFLOW_FILE in "${MACFLOW_REQUIRED_FILES[@]}"; do
    if [[ ! -r "${MACFLOW_NATIVE_ROOT}/${MACFLOW_TIER}/${MACFLOW_FILE}" ]]; then
      echo "[preflight] missing dataset file: ${MACFLOW_NATIVE_ROOT}/${MACFLOW_TIER}/${MACFLOW_FILE}" >&2
      exit 1
    fi
  done
done

echo "[preflight] checking NVIDIA driver"
nvidia-smi -L

echo "[preflight] checking JAX GPU backend in ${MACFLOW_CONDA_ENV}"
conda run --no-capture-output -n "${MACFLOW_CONDA_ENV}" python -c \
  "import jax; assert jax.default_backend() == 'gpu', jax.devices(); print('[jax]', jax.devices())"

# These values reproduce the flags from the user's completed native-SMAC 2M
# baseline experiments.  Only the grouped innovation options differ.
for MACFLOW_TIER in Good Medium Poor; do
  MACFLOW_RUN_GROUP="25m_${MACFLOW_TIER}_2m_grouped_freq50k_seed100_bs16"
  echo "[serial] starting tier=${MACFLOW_TIER} time=$(date --iso-8601=seconds)"

  conda run --no-capture-output -n "${MACFLOW_CONDA_ENV}" python smac_main.py \
    --run_group="${MACFLOW_RUN_GROUP}" \
    --project_name=macflow_25m_grouped \
    --save_dir=exp/ \
    --seed=100 \
    --source=og_marl \
    --env=smac_v1 \
    --scenario=25m \
    --dataset="${MACFLOW_TIER}" \
    --agent_name=grouped_macflow \
    --data_dir=./data \
    --native_data_dir="${MACFLOW_NATIVE_ROOT}/${MACFLOW_TIER}" \
    --offline_steps=2000000 \
    --sequence_length=20 \
    --sample_period=1 \
    --batch_size=16 \
    --buffer_size=320000 \
    --log_interval=1000 \
    --eval_interval=200000 \
    --save_interval=200000 \
    --num_eval_episodes=32 \
    --agent.lr=0.0003 \
    --agent.discount=0.99 \
    --agent.tau=0.005 \
    --agent.alpha=3.0 \
    --agent.flow_steps=10 \
    --agent.normalize_q_loss=false \
    --agent.q_agg=mean \
    --agent.use_lstm=true \
    --agent.lstm_hidden_dim=64 \
    --agent.lstm_layers=1 \
    --agent.lstm_layer_norm=true \
    --agent.sd_alpha=0.05 \
    --agent.change_group_value=0.8 \
    --agent.group_update_freq=50000 \
    --agent.group_attention_dim=64 \
    --agent.inter_group_top_k=2 \
    --agent.decentralized_execution=false

  # Match the user's original 2M protocol: the training loop evaluates 32
  # episodes at 2M, then the saved 2M checkpoint is evaluated independently
  # for another 100 episodes.
  MACFLOW_CHECKPOINT_PATH="$(
    find "exp/${MACFLOW_RUN_GROUP}" -mindepth 2 -maxdepth 2 \
      -type f -name 'params_2000000.pkl' -printf '%T@ %p\n' \
      | sort -n | tail -n 1 | cut -d' ' -f2-
  )"
  if [[ -z "${MACFLOW_CHECKPOINT_PATH}" ]]; then
    echo "[final-eval] missing 2M checkpoint for ${MACFLOW_TIER}" >&2
    exit 1
  fi
  MACFLOW_RUN_DIR="$(dirname "${MACFLOW_CHECKPOINT_PATH}")"

  echo "[final-eval] starting tier=${MACFLOW_TIER} episodes=100 checkpoint=${MACFLOW_CHECKPOINT_PATH}"
  conda run --no-capture-output -n "${MACFLOW_CONDA_ENV}" python eval_smac_checkpoint.py \
    --checkpoint_dir="${MACFLOW_RUN_DIR}" \
    --checkpoint_step=2000000 \
    --seed=100 \
    --env=smac_v1 \
    --source=og_marl \
    --scenario=25m \
    --dataset="${MACFLOW_TIER}" \
    --agent_name=grouped_macflow \
    --data_dir=./data \
    --native_data_dir="${MACFLOW_NATIVE_ROOT}/${MACFLOW_TIER}" \
    --num_eval_episodes=100 \
    --sequence_length=20 \
    --sample_period=1 \
    --batch_size=16 \
    --buffer_size=320000 \
    --output_json="${MACFLOW_RUN_DIR}/eval_final_2000000_100.json" \
    --agent.lr=0.0003 \
    --agent.discount=0.99 \
    --agent.tau=0.005 \
    --agent.alpha=3.0 \
    --agent.flow_steps=10 \
    --agent.normalize_q_loss=false \
    --agent.q_agg=mean \
    --agent.use_lstm=true \
    --agent.lstm_hidden_dim=64 \
    --agent.lstm_layers=1 \
    --agent.lstm_layer_norm=true \
    --agent.sd_alpha=0.05 \
    --agent.change_group_value=0.8 \
    --agent.group_update_freq=50000 \
    --agent.group_attention_dim=64 \
    --agent.inter_group_top_k=2 \
    --agent.decentralized_execution=false
  echo "[final-eval] completed tier=${MACFLOW_TIER} output=${MACFLOW_RUN_DIR}/eval_final_2000000_100.json"

  echo "[serial] completed tier=${MACFLOW_TIER} time=$(date --iso-8601=seconds)"
done

echo "[serial] all tiers completed time=$(date --iso-8601=seconds)"
