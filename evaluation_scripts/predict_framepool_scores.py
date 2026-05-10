#!/usr/bin/env python3
"""Generate per-sequence FramePool predictions for a CSV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

UTRGAN_ROOT = Path("/home/xli263/xli/utr_design/UTRGAN")
if str(UTRGAN_ROOT) not in sys.path:
    sys.path.insert(0, str(UTRGAN_ROOT))

from src.mrl_te_optimization.framepool import load_framepool
from src.mrl_te_optimization.util import encode_seq_framepool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict FramePool scores for sequences in a CSV.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV file containing sequences.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV with framepool predictions added.",
    )
    parser.add_argument(
        "--seq-col",
        type=str,
        default="utr",
        help="Sequence column name.",
    )
    parser.add_argument(
        "--pred-col",
        type=str,
        default="framepool_pred",
        help="Prediction column name to write.",
    )
    parser.add_argument(
        "--ckpt-h5",
        type=Path,
        default=Path("/home/xli263/xli/utr_design/UTRGAN/models/utr_model_combined_residual_new.h5"),
        help="FramePool checkpoint path.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=128,
        help="Left-padding length for FramePool encoding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Prediction batch size.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if not args.ckpt_h5.exists():
        raise FileNotFoundError(f"FramePool checkpoint not found: {args.ckpt_h5}")

    df = pd.read_csv(args.input_csv)
    if args.seq_col not in df.columns:
        raise ValueError(
            f"Sequence column '{args.seq_col}' not found in {args.input_csv}. "
            f"Available: {list(df.columns)}"
        )

    seqs = df[args.seq_col].astype(str).tolist()
    x = np.array([encode_seq_framepool(seq, max_len=args.max_len) for seq in seqs], dtype=np.float32)
    model = load_framepool(str(args.ckpt_h5))
    preds = model.predict(x, batch_size=args.batch_size, verbose=0).reshape(-1)

    out_df = df.copy()
    out_df[args.pred_col] = preds
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote FramePool predictions to {args.output_csv} with {len(out_df)} rows.")


if __name__ == "__main__":
    main()
