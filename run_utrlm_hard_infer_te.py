import argparse
from pathlib import Path
from typing import List, Sequence

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


def discover_te_checkpoints(te_ckpt_root: Path, dataset_filters: Sequence[str]) -> List[str]:
    wanted = tuple(f.lower() for f in dataset_filters)
    ckpt_paths = sorted(
        str(path) for path in te_ckpt_root.glob("*.pt")
        if (
            "te_" in path.name.lower()
            and any(dataset_name in path.name.lower() for dataset_name in wanted)
        )
    )
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No TE checkpoints found in {te_ckpt_root} for filters {dataset_filters}"
        )
    return ckpt_paths


def main():
    parser = argparse.ArgumentParser(
        description="Run TE UTR-LM oracle on sequences (hard argmax path)."
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="CSV file containing sequences.")
    parser.add_argument("--seq-column", type=str, default="seq", help="Column name with sequences.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Where to write CSV with predictions.")
    parser.add_argument(
        "--te-ckpt-root",
        type=Path,
        default=Path("/home/xli263/xli/utr_design/UTR-LM/Model/Downstream/TE_EL"),
        help="Directory containing TE .pt checkpoints.",
    )
    parser.add_argument(
        "--dataset-filters",
        type=str,
        nargs="+",
        default=("pc3"),
        help="One or more substrings used to select TE checkpoints from --te-ckpt-root.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu device for inference.")
    parser.add_argument("--seq-trim-len", type=int, default=100, help="Trim length passed to UTR-LM oracle.")
    parser.add_argument("--pred-column", type=str, default="te_pred", help="Prediction column name.")
    args = parser.parse_args()

    sequences = load_sequences(args.input_csv, column=args.seq_column)
    one_hot = one_hot_encode(sequences)  # [B, 4, L]
    te_eval_ckpt_paths = discover_te_checkpoints(args.te_ckpt_root, args.dataset_filters)

    oracle = oracle_new.get_utrlm_oracle(
        checkpoint_root=str(args.te_ckpt_root),
        checkpoint_paths=te_eval_ckpt_paths,
        device=args.device,
        seq_trim_len=args.seq_trim_len,
    )

    oracle.eval()
    with torch.no_grad():
        preds = oracle(one_hot, soft_input=False).squeeze(-1).cpu().numpy()

    df = pd.read_csv(args.input_csv)
    df[args.pred_column] = preds
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote predictions to {args.output_csv} with {len(df)} rows.")


if __name__ == "__main__":
    main()
