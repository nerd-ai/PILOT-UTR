import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import torch

import oracle_new


def one_hot_encode(
    sequences: Sequence[str],
    base_order: str = "ACGT",
) -> torch.Tensor:
  """Convert sequences into a channel-first one-hot tensor [B, 4, L]."""
  idx = {b: i for i, b in enumerate(base_order)}
  max_len = max(len(seq) for seq in sequences)
  tensor = torch.zeros(len(sequences), len(base_order), max_len, dtype=torch.float32)
  for i, seq in enumerate(sequences):
    seq = seq.upper().replace("U", "T")
    for j, ch in enumerate(seq):
      if ch not in idx:
        raise ValueError(f"Unexpected base '{ch}' in sequence {i} at position {j}")
      tensor[i, idx[ch], j] = 1.0
  return tensor


def load_sequences(csv_path: Path, column: str = "seq") -> List[str]:
  df = pd.read_csv(csv_path)
  if column not in df.columns:
    raise ValueError(f"Column '{column}' not found in {csv_path}. Available: {df.columns}")
  seqs = df[column].astype(str).tolist()
  if not seqs:
    raise ValueError(f"No sequences found in column '{column}' of {csv_path}")
  return seqs


def main():
  parser = argparse.ArgumentParser(description="Run UTR-LM oracle on sequences (hard argmax path).")
  parser.add_argument("--input-csv", type=Path, required=True, help="CSV file containing sequences.")
  parser.add_argument("--seq-column", type=str, default="seq", help="Column name with sequences.")
  parser.add_argument("--output-csv", type=Path, required=True, help="Where to write CSV with predictions.")
  parser.add_argument("--checkpoint-path", type=Path, default="/home/xli263/xli/utr_design/UTR-LM/Model/Downstream/MRL/MJ3_seed1337_ESM2SISS_FS4.1.ep93.1e-2.dr5_unmod_1_utr_10folds_rl_LabelScalerFalse_LabelLog2False_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_nodes40_dropout30.5_finetuneTrue_huberlossTrue_lr0.01_fold0_epoch299.pt", help="Path to a UTR-LM checkpoint .pt file.")
  parser.add_argument(
      "--checkpoint-root",
      type=Path,
      default=None,
      help="Root directory containing checkpoints (defaults to the checkpoint's parent).")
  parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu device for inference.")
  parser.add_argument("--seq-trim-len", type=int, default=100, help="Trim length passed to UTR-LM oracle.")
  parser.add_argument(
      "--dataset-patterns",
      type=str,
      nargs="*",
      default=("utr",),
      help="Dataset name filters if you rely on auto-discovery instead of explicit checkpoint.")

  args = parser.parse_args()

  checkpoint_root = args.checkpoint_root if args.checkpoint_root else args.checkpoint_path.parent
  sequences = load_sequences(args.input_csv, column=args.seq_column)
  one_hot = one_hot_encode(sequences)  # [B, 4, L]

  oracle = oracle_new.get_utrlm_oracle(
      checkpoint_root=str(checkpoint_root),
      checkpoint_paths=[str(args.checkpoint_path)],
      device=args.device,
      dataset_patterns=tuple(args.dataset_patterns),
      seq_trim_len=args.seq_trim_len)

  oracle.eval()
  with torch.no_grad():
    preds = oracle(one_hot, soft_input=False).squeeze(-1).cpu().numpy()

  df = pd.read_csv(args.input_csv)
  df["utrlm_pred"] = preds
  args.output_csv.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(args.output_csv, index=False)
  print(f"Wrote predictions to {args.output_csv} with {len(df)} rows.")


if __name__ == "__main__":
  main()
