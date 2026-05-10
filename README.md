# PILOT-UTR

PILOT-UTR is a discrete diffusion framework for UTR sequence optimization with calibrated reward models. This release keeps the main training workflows clear and explicit:

- MRL diffusion pretraining: `main_gosai.py`
- MRL reward/oracle training: `train_oracle_vanilla.py`, `train_oracle_cap.py`
- MRL reward-guided diffusion finetuning: `run_finetune_ood_mrl.sh`
- TE reward-guided diffusion finetuning with MTTrans: `run_finetune_new_te_mttrans.sh`

This project is built on the DRAKES codebase: https://github.com/ChenyuWang-Monica/DRAKES.git

The commands below use the MRL task as the end-to-end example.

## Environment

Create the Conda environment from the exported environment file:

```bash
cd /your_path/PILOT-UTR
conda env create -f environment.yaml
conda activate sedd
```

If you rename the environment in `environment.yaml`, activate that name instead.

## Data

For the MRL example, the expected CSV files are:

```text
data_and_model/train_dataset_mrl.csv
data_and_model/val_dataset_mrl.csv
```

Each file should contain at least:

```text
utr,rl
```

where `utr` is the UTR sequence and `rl` is the measured MRL/reward label.

The 4-base tokenizer vocabulary is:

```text
data_and_model/data/4_base_token_vocab_no_eos.json
```

It maps:

```json
{"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
```

## 1. Pretrain The Diffusion Model

The pretraining config is:

```text
configs_gosai/config_gosai_pretrain.yaml
```

For the MRL example, this config sets:

- `model.length: 50`
- `data.train_csv_path: ${cwd:}/data_and_model/train_dataset_mrl.csv`
- `data.valid_csv_path: ${cwd:}/data_and_model/val_dataset_mrl.csv`
- `data.tokenizer_vocab_path: ${cwd:}/data_and_model/data/4_base_token_vocab_no_eos.json`
- `checkpointing.save_dir: ${cwd:}/outputs/pretrain_mrl`

Run pretraining:

```bash
cd /your_path/PILOT-UTR
python main_gosai.py
```

The best checkpoint is saved by the Lightning checkpoint callback under:

```text
outputs/pretrain_mrl/checkpoints/best.ckpt
```

For later finetuning, either edit `PRETRAINED_DIFFUSION_CKPT` in `run_finetune_ood_mrl.sh` to this path, or copy/link it to:

```text
checkpoints/pretrained_4_base_50nt_no_eos.ckpt
```

## 2. Train A Vanilla Reward Model

Train the standard regression reward model:

```bash
cd /your_path/PILOT-UTR
ENABLE_WANDB=0 python train_oracle_vanilla.py \
  --train-csv /your_path/PILOT-UTR/data_and_model/train_dataset_mrl.csv \
  --val-csv /your_path/PILOT-UTR/data_and_model/val_dataset_mrl.csv \
  --save-dir /your_path/PILOT-UTR/outputs/oracle_vanilla_mrl \
  --sequence-column utr \
  --target-column rl \
  --seq-len 50 \
  --pad-side right
```

The best validation-loss checkpoint is saved as:

```text
outputs/oracle_vanilla_mrl/vanilla_best.ckpt
```

## 3. Train A Cap-Calibrated Reward Model

The cap-calibrated reward model uses the labeled MRL data plus negative/OOD generated sequences. The negative CSV should contain a sequence column, by default `seq`.

Example:

```bash
cd /your_path/PILOT-UTR
ENABLE_WANDB=0 python train_oracle_cap.py \
  --train-csv /your_path/PILOT-UTR/data_and_model/train_dataset_mrl.csv \
  --val-csv /your_path/PILOT-UTR/data_and_model/val_dataset_mrl.csv \
  --negative-csv /your_path/negative_sequences.csv \
  --save-dir /your_path/PILOT-UTR/outputs/oracle_cap_mrl \
  --sequence-column utr \
  --target-column rl \
  --negative-sequence-column seq \
  --seq-len 50 \
  --pad-side right \
  --cap-threshold 6.5 \
  --cap-lambda 3.0
```

The checkpoint is saved as:

```text
outputs/oracle_cap_mrl/cap_best.ckpt
```

This checkpoint can be used as the finetuning reward model.

## 4. Prepare Finetuning Assets

The MRL finetuning launcher is:

```text
run_finetune_ood_mrl.sh
```

Before running it, edit the path block near the top of the file:

```bash
PRETRAINED_DIFFUSION_CKPT="${CHECKPOINT_DIR}/pretrained_4_base_50nt_no_eos.ckpt"
ENFORMER_UTRLM_ORACLE_CKPT="${CHECKPOINT_DIR}/vanilla_or_cap_enformer_utrlm.ckpt"
UTRLM_EVAL_CKPT="${CHECKPOINT_DIR}/utrlm_eval_mrl.pt"
TOKEN_LENGTH_DISTRIBUTION="${DATA_DIR}/token_length_distribution_50_utr_4_base.txt"
EVAL_KMER_REFERENCE_CSV="${DATA_DIR}/train_dataset_top_10pct.csv"
```

Required files:

- `PRETRAINED_DIFFUSION_CKPT`: pretrained diffusion checkpoint from step 1.
- `ENFORMER_UTRLM_ORACLE_CKPT`: vanilla or cap-calibrated reward checkpoint from step 2 or 3.
- `UTRLM_EVAL_CKPT`: external UTR-LM checkpoint used only for evaluation reward logging.
- `TOKEN_LENGTH_DISTRIBUTION`: empirical target-length distribution used during finetuning.
- `EVAL_KMER_REFERENCE_CSV`: high-reward reference sequences for k-mer correlation logging.

The length distribution file should contain a Python-style dictionary, for example:

```text
{50: 1.0}
```

or a full empirical distribution such as:

```text
{45: 0.02, 46: 0.03, ..., 50: 0.20}
```

## 5. Run MRL Reward-Guided Finetuning

After the paths in `run_finetune_ood_mrl.sh` are set:

```bash
cd /your_path/PILOT-UTR
bash run_finetune_ood_mrl.sh
```

The launcher runs:

```bash
python finetune_reward_mrl.py \
  --reward_type Enformer_UTRLM \
  --pretrained_ckpt_path "${PRETRAINED_DIFFUSION_CKPT}" \
  --oracle_ckpt_path "${ENFORMER_UTRLM_ORACLE_CKPT}" \
  --utrlm_eval_checkpoint_path "${UTRLM_EVAL_CKPT}" \
  --token_length_distribution "${TOKEN_LENGTH_DISTRIBUTION}" \
  --alpha 0.0015 \
  --beta 0.0015
```

The finetuning objective combines:

- reward maximization from the calibrated reward model
- reverse KL regularization
- forward KL regularization
- entropy regularization
- k-mer correlation monitoring

Checkpoints are saved under:

```text
outputs/finetune_ood_mrl_top10_3mer/
```

The best model is written as:

```text
best_model.ckpt
```

## TE MTTrans Workflow

The TE workflow uses the trimmed MTTrans-only script:

```text
finetune_reward_te_mttrans.py
```

and the launcher:

```text
run_finetune_new_te_mttrans.sh
```

Before running, edit these paths:

```bash
PRETRAINED_DIFFUSION_CKPT="/your_path/pretrained_4_base_rp_pc3_te.ckpt"
MTTRANS_CHECKPOINT="/your_path/mttrans_rp_pc3_model_best_cv1.pth"
TOKEN_LENGTH_DISTRIBUTION="/your_path/token_length_distribution_RP_PC3_te_no_eos.txt"
EVAL_KMER_REFERENCE_CSV="/your_path/RP_PC3_te_train_top5pct_by_te.csv"
LOG_BASE_ROOT="/your_path/te_mttrans_finetune_outputs"
```

Then run:

```bash
cd /your_path/PILOT-UTR
bash run_finetune_new_te_mttrans.sh
```

This TE path uses left-padded MTTrans reward inputs with `MTTRANS_INPUT_LENGTH=105`.

## Notes

- Checkpoints are not included in the repository. Place them in the paths specified by the launchers.
- If you do not want Weights & Biases logging for oracle training, use `ENABLE_WANDB=0`.
- For diffusion training/finetuning, edit the launcher or script arguments to set `--wandb False` when running without W&B.
- Run commands from the repository root so `${cwd:}` paths in Hydra configs resolve correctly.
