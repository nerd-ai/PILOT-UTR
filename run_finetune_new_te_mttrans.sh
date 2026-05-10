#!/bin/bash

# PILOT-UTR TE reward-guided diffusion finetuning with MTTrans.
#
# Before running, place/download the required external assets and update the
# paths below:
#   PRETRAINED_DIFFUSION_CKPT: pretrained discrete diffusion checkpoint.
#   MTTRANS_CHECKPOINT: MTTrans/UTRGAN TE reward checkpoint.
#   TOKEN_LENGTH_DISTRIBUTION: empirical RP-PC3 target-length distribution.
#   EVAL_KMER_REFERENCE_CSV: reference high-TE sequences for k-mer logging.

# ALPHAS=(0.001 0.0015 0.003)
ALPHAS=(0.0005)
BETAS=(0.0005)
LEARNING_RATES=(3e-5)

SEED=9
GPU=1
FORWARD_KL_ON_OLD_XT=False
SAVE_FOLDER_NAME="mdlm_mrl_25_100_mttrans_augmentation_rp_pc3"
EVAL_KMER_SEQ_COL="utr"
EVAL_KMER_K=3
SAVE_BEST_METRIC="reward_eval"
SAVE_BEST_MIN_EVAL_KMER_CORRS=(0.70)
MTTRANS_INPUT_LENGTH=105

PILOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${PILOT_ROOT}/checkpoints"
DATA_DIR="${PILOT_ROOT}/data"
OUTPUT_DIR="${PILOT_ROOT}/outputs"
cd "${PILOT_ROOT}"

PRETRAINED_DIFFUSION_CKPT="/your_path/pretrained_4_base_rp_pc3_te.ckpt"
MTTRANS_CHECKPOINT="/your_path/mttrans_rp_pc3_model_best_cv1.pth"
TOKEN_LENGTH_DISTRIBUTION="/your_path/token_length_distribution_RP_PC3_te_no_eos.txt"
EVAL_KMER_REFERENCE_CSV="/your_path/RP_PC3_te_train_top5pct_by_te.csv"
LOG_BASE_ROOT="/your_path/te_mttrans_finetune_outputs"

for required_path in \
  "${PRETRAINED_DIFFUSION_CKPT}" \
  "${MTTRANS_CHECKPOINT}" \
  "${TOKEN_LENGTH_DISTRIBUTION}" \
  "${EVAL_KMER_REFERENCE_CSV}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing required file: ${required_path}" >&2
    exit 1
  fi
done

for learning_rate in "${LEARNING_RATES[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    for beta in "${BETAS[@]}"; do
      for save_best_min_eval_kmer_corr in "${SAVE_BEST_MIN_EVAL_KMER_CORRS[@]}"; do
        LOG_BASE_DIR="${LOG_BASE_ROOT}/${SAVE_FOLDER_NAME}_bestkmer${save_best_min_eval_kmer_corr}"
        echo "=========================================="
        echo "Running alpha=${alpha} beta=${beta} lr=${learning_rate} save_best_min_eval_kmer_corr=${save_best_min_eval_kmer_corr}"
        echo "Saving under ${LOG_BASE_DIR}"
        echo "=========================================="

        CUDA_VISIBLE_DEVICES=${GPU} python finetune_reward_te_mttrans.py \
        --alpha ${alpha} \
        --beta ${beta} \
        --seed ${SEED} \
        --name "grid_alpha${alpha}_beta${beta}_lr${learning_rate}_bestkmer${save_best_min_eval_kmer_corr}" \
        --log_base_dir "${LOG_BASE_DIR}" \
        --eval_kmer_reference_csv "${EVAL_KMER_REFERENCE_CSV}" \
        --eval_kmer_seq_col "${EVAL_KMER_SEQ_COL}" \
        --eval_kmer_k ${EVAL_KMER_K} \
        --save_best_metric "${SAVE_BEST_METRIC}" \
        --save_best_min_eval_kmer_corr ${save_best_min_eval_kmer_corr} \
        --reward_type mttrans \
        --mttrans_checkpoint "${MTTRANS_CHECKPOINT}" \
        --mttrans_input_length ${MTTRANS_INPUT_LENGTH} \
        --pretrained_ckpt_path "${PRETRAINED_DIFFUSION_CKPT}" \
        --token_length_distribution "${TOKEN_LENGTH_DISTRIBUTION}" \
        --reward_padding_side left \
        --num_epochs 500 \
        --num_accum_steps 4 \
        --batch_size 32 \
        --truncate_steps 50 \
        --total_num_steps 128 \
        --learning_rate ${learning_rate} \
        --gumbel_temp 1.0 \
        --gradnorm_clip 1 \
        --forward_kl_on_old_xt ${FORWARD_KL_ON_OLD_XT} \
        --save_every_n_epochs 5 \
        --gradient_type base_soft \
        --mask_base_ce_coeff 0.0 \
        --mask_base_agg_method global \
        --mask_base_divergence ce \
        --mask_base_truncate_steps 50

        echo "Finished alpha=${alpha} beta=${beta} lr=${learning_rate} save_best_min_eval_kmer_corr=${save_best_min_eval_kmer_corr}"
        echo ""
      done
    done
  done
done

echo "Grid search complete."
