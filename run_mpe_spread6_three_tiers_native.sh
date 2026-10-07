#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

MACFLOW_CONDA_ENV="${MACFLOW_CONDA_ENV:-macflow_gpu_grouped}"
SEED="${SEED:-100}"
TIERS="${TIERS:-good medium poor}"
OFFLINE_STEPS="${OFFLINE_STEPS:-2000000}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BUFFER_SIZE="${BUFFER_SIZE:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200000}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-32}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-100}"
NUM_AGENTS="${NUM_AGENTS:-6}"
SCENARIO="${SCENARIO:-simple_spread${NUM_AGENTS}}"
PROJECT_NAME="${PROJECT_NAME:-grouped_macflow_mpe_spread${NUM_AGENTS}}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/outputs/mpe_spread6_omar_raw}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

mkdir -p logs

REQUIRED_FILES=(
  obs.npy
  states.npy
  actions.npy
  rewards.npy
  legals.npy
  terminals.npy
  truncations.npy
  path_lengths.npy
)

for tier in ${TIERS}; do
  lower_tier="$(echo "${tier}" | tr '[:upper:]' '[:lower:]')"
  native_dir="${DATA_ROOT}/${lower_tier}"

  for required_file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -r "${native_dir}/${required_file}" ]]; then
      echo "[preflight] missing dataset file: ${native_dir}/${required_file}" >&2
      exit 1
    fi
  done
done

for tier in ${TIERS}; do
  lower_tier="$(echo "${tier}" | tr '[:upper:]' '[:lower:]')"
  native_dir="${DATA_ROOT}/${lower_tier}"
  run_group="mpe_spread${NUM_AGENTS}_${tier}_${OFFLINE_STEPS}_grouped_macflow_seed${SEED}_bs${BATCH_SIZE}"
  log_file="logs/${run_group}.log"

  echo "[grouped-mac-flow] tier=${tier} native_data_dir=${native_dir} log=${log_file}"
  checkpoint_dir="$(
    for root in "exp/${PROJECT_NAME}/${run_group}" "exp/${run_group}"; do
      if [[ -d "${root}" ]]; then
        find "${root}" -maxdepth 1 -type d -name "sd*_*" -printf '%T@ %p\n'
      fi
    done | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [[ -n "${checkpoint_dir}" && -r "${checkpoint_dir}/params_${OFFLINE_STEPS}.pkl" ]]; then
    echo "[grouped-mac-flow] found completed training checkpoint=${checkpoint_dir}/params_${OFFLINE_STEPS}.pkl; skipping train"
  else
    conda run --no-capture-output -n "${MACFLOW_CONDA_ENV}" python smac_main.py \
      --project_name="${PROJECT_NAME}" \
      --run_group="${run_group}" \
      --save_dir=exp/ \
      --agent_name=grouped_macflow \
      --env=mpe \
      --source=omar \
      --scenario="${SCENARIO}" \
      --dataset="${tier}" \
      --seed="${SEED}" \
      --native_data_dir="${native_dir}" \
      --offline_steps="${OFFLINE_STEPS}" \
      --sequence_length="${SEQUENCE_LENGTH}" \
      --sample_period=1 \
      --buffer_size="${BUFFER_SIZE}" \
      --batch_size="${BATCH_SIZE}" \
      --log_interval="${LOG_INTERVAL}" \
      --eval_interval="${EVAL_INTERVAL}" \
      --save_interval="${SAVE_INTERVAL}" \
      --num_eval_episodes="${NUM_EVAL_EPISODES}" \
      > "${log_file}" 2>&1

    checkpoint_dir="$(
      for root in "exp/${PROJECT_NAME}/${run_group}" "exp/${run_group}"; do
        if [[ -d "${root}" ]]; then
          find "${root}" -maxdepth 1 -type d -name "sd*_*" -printf '%T@ %p\n'
        fi
      done | sort -nr | head -n 1 | cut -d' ' -f2-
    )"
  fi
  if [[ -z "${checkpoint_dir}" ]]; then
    echo "[grouped-mac-flow] ERROR: checkpoint directory not found for run_group=${run_group}" >&2
    exit 1
  fi

  echo "[grouped-mac-flow] final_eval tier=${tier} checkpoint=${checkpoint_dir} episodes=${FINAL_EVAL_EPISODES}"
  conda run --no-capture-output -n "${MACFLOW_CONDA_ENV}" python -u eval_smac_checkpoint.py \
    --checkpoint_dir="${checkpoint_dir}" \
    --checkpoint_step="${OFFLINE_STEPS}" \
    --seed="${SEED}" \
    --env=mpe \
    --source=omar \
    --scenario="${SCENARIO}" \
    --dataset="${tier}" \
    --agent_name=grouped_macflow \
    --native_data_dir="${native_dir}" \
    --num_eval_episodes="${FINAL_EVAL_EPISODES}" \
    --sequence_length="${SEQUENCE_LENGTH}" \
    --sample_period=1 \
    --buffer_size="${BUFFER_SIZE}" \
    --batch_size="${BATCH_SIZE}" \
    --output_json="${checkpoint_dir}/eval_final_${OFFLINE_STEPS}_${FINAL_EVAL_EPISODES}.json" \
    >> "${log_file}" 2>&1
  echo "[grouped-mac-flow] completed tier=${tier}"
done

echo "[grouped-mac-flow] all tiers completed"
