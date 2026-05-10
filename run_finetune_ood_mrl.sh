#!/bin/bash

# PILOT-UTR MRL reward-guided diffusion finetuning.
#
# Before running, place/download the required external assets and update the
# paths below:
#   PRETRAINED_DIFFUSION_CKPT: pretrained discrete diffusion checkpoint.
#   ENFORMER_UTRLM_ORACLE_CKPT: vanilla or cap-calibrated Enformer/UTR oracle.
#   UTRLM_EVAL_CKPT: UTR-LM checkpoint used only for eval reward logging.
#   TOKEN_LENGTH_DISTRIBUTION: empirical target-length distribution.
#   EVAL_KMER_REFERENCE_CSV: reference high-reward sequences for k-mer logging.

ALPHAS=(0.0015)
BETAS=(0.0015)

SEED=9
GPU=3
REWARD_TYPE="Enformer_UTRLM"
CKPT_SAVE_FOLDER_NAME="finetune_ood_mrl_top10_3mer"

PILOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${PILOT_ROOT}/checkpoints"
DATA_DIR="${PILOT_ROOT}/data"
OUTPUT_DIR="${PILOT_ROOT}/outputs"
cd "${PILOT_ROOT}"

PRETRAINED_DIFFUSION_CKPT="${CHECKPOINT_DIR}/pretrained_4_base_50nt_no_eos.ckpt"
ENFORMER_UTRLM_ORACLE_CKPT="${CHECKPOINT_DIR}/vanilla_or_cap_enformer_utrlm.ckpt"
UTRLM_EVAL_CKPT="${CHECKPOINT_DIR}/utrlm_eval_mrl.pt"
TOKEN_LENGTH_DISTRIBUTION="${DATA_DIR}/token_length_distribution_50_utr_4_base.txt"
EVAL_KMER_REFERENCE_CSV="${DATA_DIR}/train_dataset_top_10pct.csv"
CKPT_SAVE_BASE_DIR="${OUTPUT_DIR}/${CKPT_SAVE_FOLDER_NAME}"

for required_path in \
  "${PRETRAINED_DIFFUSION_CKPT}" \
  "${ENFORMER_UTRLM_ORACLE_CKPT}" \
  "${UTRLM_EVAL_CKPT}" \
  "${TOKEN_LENGTH_DISTRIBUTION}" \
  "${EVAL_KMER_REFERENCE_CSV}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing required file: ${required_path}" >&2
    exit 1
  fi
done

for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    echo "=========================================="
    echo "Running alpha=${alpha} beta=${beta}"
    echo "Saving checkpoints under ${CKPT_SAVE_BASE_DIR}/seed${SEED}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=${GPU} python finetune_reward_mrl.py \
      --alpha ${alpha} \
      --beta ${beta} \
      --seed ${SEED} \
      --name "grid_alpha${alpha}_beta${beta}" \
      --reward_type "${REWARD_TYPE}" \
      --pretrained_ckpt_path "${PRETRAINED_DIFFUSION_CKPT}" \
      --oracle_ckpt_path "${ENFORMER_UTRLM_ORACLE_CKPT}" \
      --utrlm_eval_checkpoint_path "${UTRLM_EVAL_CKPT}" \
      --token_length_distribution "${TOKEN_LENGTH_DISTRIBUTION}" \
      --num_epochs 500 \
      --num_accum_steps 4 \
      --batch_size 32 \
      --truncate_steps 50 \
      --total_num_steps 128 \
      --learning_rate 1e-3 \
      --gumbel_temp 1.0 \
      --gradnorm_clip 1 \
      --log_base_dir "${CKPT_SAVE_BASE_DIR}" \
      --save_every_n_epochs 50 \
      --eval_kmer_reference_csv "${EVAL_KMER_REFERENCE_CSV}" \
      --eval_kmer_seq_col utr \
      --eval_kmer_k 3

    echo "Finished alpha=${alpha} beta=${beta}"
    echo ""
  done
done

echo "Grid search complete."
