"""
Predict rl values for a CSV dataset and compare against ground truth.

Example:
python script/predict_new.py \
    --config log/Backbone/RL_hard_share/3M/small_repective_filed_strides1113.ini \
    --checkpoint small_repective_filed_strides1113-model_best.pth \
    --input data/MPA_H_train_val.csv \
    --task MPA_H \
    --seq-col utr \
    --label-col rl \
    --out predictions_mpa_h_with_labels.csv
"""
import argparse
import os
import sys
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.popen import Auto_popen
from utils import load_model
from models.reader import one_hot, pad_zeros
from evaluation.evaluation_func import reverse_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MTtrans on a CSV file and report prediction accuracy."
    )
    parser.add_argument("--config", required=True, help="Path to the MTtrans .ini file.")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pth checkpoint.")
    parser.add_argument("--input", required=True, help="CSV file containing sequences and labels.")
    parser.add_argument("--task", default="MPA_V", choices=["MPA_U", "MPA_H", "MPA_V"])
    parser.add_argument("--seq-col", default="utr", help="Column containing the input sequence.")
    parser.add_argument("--label-col", default="rl", help="Column containing the ground truth label.")
    parser.add_argument("--id-col", default="id", help="Optional ID column (defaults to row_i when missing).")
    parser.add_argument("--trim-len", type=int, default=100, help="Trim/pad sequences to this length before encoding.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--denormalize", action="store_true", help="Reverse the z-score transform on predictions.")
    parser.add_argument("--out", default="predictions_with_labels.csv", help="Output CSV path.")
    return parser.parse_args()


def canonicalize(seq: str) -> str:
    return seq.upper().replace("U", "T")


def trim_to_n(seq: str, trim_len: int) -> str:
    seq = canonicalize(seq)
    return ("N" * trim_len + seq)[-trim_len:]


def encode_sequence(seq: str, pad_to: int) -> torch.Tensor:
    oh = one_hot(seq).astype(np.float32)
    padded = pad_zeros(oh, pad_to)
    return torch.tensor(padded, dtype=torch.float32)


def batch_iter(records: List[Tuple[str, torch.Tensor]], batch_size: int) -> Iterable[Tuple[List[str], torch.Tensor]]:
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        ids, tensors = zip(*chunk)
        yield list(ids), torch.stack(tensors, dim=0)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    cfg = Auto_popen(args.config)
    model = cfg.Model_Class(*cfg.model_args)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    if isinstance(checkpoint.get("state_dict"), dict):
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = checkpoint["state_dict"]
    model.to(device)
    model.eval()

    pad_to = cfg.pad_to

    # Load data
    df = pd.read_csv(args.input)
    if args.seq_col not in df.columns:
        raise ValueError(f"Missing sequence column '{args.seq_col}' in {args.input}")
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column '{args.label_col}' in {args.input}")

    ids = (
        df[args.id_col].astype(str).tolist()
        if args.id_col in df.columns
        else [f"row_{i}" for i in range(len(df))]
    )
    seqs = df[args.seq_col].astype(str).tolist()
    labels = df[args.label_col].astype(float).tolist()

    trimmed = [trim_to_n(seq, args.trim_len) for seq in seqs]
    encoded = [(sid, encode_sequence(seq, pad_to)) for sid, seq in zip(ids, trimmed)]

    # Predict
    preds = []
    with torch.no_grad():
        for batch_ids, batch in batch_iter(encoded, args.batch_size):
            batch = batch.to(device)
            model.task = args.task
            out = model(batch).squeeze(-1).cpu().numpy()
            if args.denormalize:
                out = reverse_transform(out, args.task)
            preds.extend(zip(batch_ids, out))

    pred_map = dict(preds)
    pred_vals = [pred_map[sid] for sid in ids]

    # Metrics
    y_true = np.array(labels, dtype=float)
    y_pred = np.array(pred_vals, dtype=float)
    pearson_val = stats.pearsonr(y_true, y_pred)
    pearson = pearson_val[0] if isinstance(pearson_val, tuple) else pearson_val.statistic
    spearman_val = stats.spearmanr(y_true, y_pred)
    spearman = spearman_val[0] if isinstance(spearman_val, tuple) else spearman_val.statistic
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    nmae = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8))
    print(f"Samples: {len(y_true)}")
    print(f"Pearson r: {pearson:.4f}")
    print(f"Spearman r: {spearman:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"NMAE: {nmae:.4f}")
    print(f"MSE: {mse:.4f}")

    # Save merged output
    out_df = pd.DataFrame(
        {
            "id": ids,
            "sequence": trimmed,
            f"true_{args.label_col}": y_true,
            f"pred_{args.label_col}": y_pred,
            "abs_error": np.abs(y_true - y_pred),
        }
    )
    out_df.to_csv(args.out, index=False)
    print(f"Wrote predictions with labels to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
