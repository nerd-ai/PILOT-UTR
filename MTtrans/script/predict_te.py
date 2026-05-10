"""
Batch TE inference with MTtrans across multiple checkpoints.

This script loads a single MTtrans architecture from a .ini file, then
iterates over checkpoint files (e.g., the 3R CV seeds) and averages the
predictions across tasks and seeds.
"""
import argparse
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evaluation.evaluation_func import reverse_transform
from models.popen import Auto_popen
from models.reader import one_hot, pad_zeros


# ---------------------- #
# CLI
# ---------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average MTtrans TE predictions across tasks and CV seeds.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the MTtrans .ini file that defines the architecture (e.g., log/Backbone/RL_hard_share/3R/schedule_MTL.ini).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing checkpoint .pth files (e.g., checkpoint/RL_hard_share_MTL/3R).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file: FASTA (.fasta, .fa) or CSV (.csv) with a 'seq' column.",
    )
    parser.add_argument(
        "--out",
        default="mttrans_te_predictions.csv",
        help="Where to save the averaged predictions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--denormalize",
        action="store_true",
        help="Use evaluation.reverse_transform to map predictions back to original TE scale.",
    )
    parser.add_argument(
        "--len-col",
        default=None,
        help="Optional column name in CSV providing per-row trim length.",
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help="Optional column name in CSV to use as sequence ID (defaults to 'id' or row_#).",
    )
    return parser.parse_args()


# ---------------------- #
# Sequence helpers
# ---------------------- #
def canonicalize(seq: str) -> str:
    return seq.upper().replace("U", "T")


def trim_to_n(seq: str, trim_len: int = 100) -> str:
    return ("N" * trim_len + canonicalize(seq))[-trim_len:]


def encode_sequence(seq: str, pad_to: int) -> torch.Tensor:
    oh = one_hot(seq).astype(np.float32)
    padded = pad_zeros(oh, pad_to)
    return torch.tensor(padded, dtype=torch.float32)


def batch_iter(records: List[Tuple[str, torch.Tensor]], batch_size: int) -> Iterable[Tuple[List[str], torch.Tensor]]:
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        ids, tensors = zip(*chunk)
        yield list(ids), torch.stack(tensors, dim=0)


# ---------------------- #
# Input readers
# ---------------------- #
def read_fasta(path: str, trim_len: int = 100) -> List[Tuple[str, str]]:
    records = []
    for rec in SeqIO.parse(path, "fasta"):
        records.append((rec.id, trim_to_n(str(rec.seq), trim_len)))
    return records


def read_csv(path: str, len_col: str = None, id_col: str = None, default_trim: int = 100) -> List[Tuple[str, str]]:
    df = pd.read_csv(path)
    if "utr" not in df.columns:
        raise ValueError("CSV input must contain a 'seq' column.")

    # IDs
    if id_col and id_col in df.columns:
        ids = df[id_col].astype(str).tolist()
    elif "id" in df.columns:
        ids = df["id"].astype(str).tolist()
    else:
        ids = [f"row_{i}" for i in range(len(df))]

    # Per-row trim lengths
    if len_col and len_col in df.columns:
        trim_lengths = df[len_col].fillna(default_trim).astype(int).tolist()
    else:
        trim_lengths = [default_trim] * len(df)

    records = []
    for sid, seq, tlen in zip(ids, df["utr"].astype(str), trim_lengths):
        records.append((sid, trim_to_n(seq, tlen)))
    return records


# ---------------------- #
# Checkpoint discovery
# ---------------------- #
DEFAULT_TASK_MAP = {
    "V_293": "RP_293T",
    "V_muscle": "RP_muscle",
    "V_PC3": "RP_PC3",
}


def infer_task_from_name(name: str, task_map: Dict[str, str]) -> str:
    lower = name.lower()
    for key, task in task_map.items():
        if key.lower() in lower:
            return task
    raise ValueError(f"Cannot infer task for checkpoint '{name}'. Please extend DEFAULT_TASK_MAP.")


def discover_checkpoints(ckpt_dir: str, task_map: Dict[str, str]) -> List[Tuple[str, str]]:
    """Return list of (checkpoint_path, task_name) pairs."""
    paths = []
    for fname in sorted(os.listdir(ckpt_dir)):
        if not fname.endswith(".pth"):
            continue
        # Only keep checkpoints we can map to a task
        try:
            task = infer_task_from_name(fname, task_map)
        except ValueError:
            continue
        paths.append((os.path.join(ckpt_dir, fname), task))
    if not paths:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")
    return paths


# ---------------------- #
# Prediction
# ---------------------- #
def predict_batch(
    model: torch.nn.Module,
    encoded: List[Tuple[str, torch.Tensor]],
    batch_size: int,
    device: torch.device,
    task: str,
    denormalize: bool,
) -> Dict[str, float]:
    preds: Dict[str, float] = {}
    with torch.no_grad():
        for ids, batch in batch_iter(encoded, batch_size):
            batch = batch.to(device)
            model.task = task
            out = model(batch).squeeze(-1).cpu().numpy()
            if denormalize:
                out = reverse_transform(out, task)
            preds.update(zip(ids, out))
    return preds


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load architecture/config
    cfg = Auto_popen(args.config)
    base_model = cfg.Model_Class(*cfg.model_args)
    base_model.to(device)
    base_model.eval()

    # List checkpoints and tasks
    ckpt_task_pairs = discover_checkpoints(args.checkpoint_dir, DEFAULT_TASK_MAP)
    print(f"Found {len(ckpt_task_pairs)} checkpoints: {[os.path.basename(p) for p, _ in ckpt_task_pairs]}")

    # Read input
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".csv":
        seq_records = read_csv(args.input, args.len_col, args.id_col)
    else:
        seq_records = read_fasta(args.input)

    encoded = [(sid, encode_sequence(seq, cfg.pad_to)) for sid, seq in seq_records]
    ids = [sid for sid, _ in encoded]

    # Accumulate predictions
    sum_preds = {sid: 0.0 for sid in ids}
    n_preds = {sid: 0 for sid in ids}

    for ckpt_path, task in ckpt_task_pairs:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint["state_dict"], dict):
            base_model.load_state_dict(checkpoint["state_dict"])
        else:
            # stored whole model
            base_model = checkpoint["state_dict"]
            base_model.to(device)
            base_model.eval()

        ckpt_preds = predict_batch(
            base_model,
            encoded,
            args.batch_size,
            device,
            task,
            args.denormalize,
        )
        for sid, pred in ckpt_preds.items():
            sum_preds[sid] += float(pred)
            n_preds[sid] += 1

    averaged = np.array([sum_preds[sid] / max(n_preds[sid], 1) for sid in ids], dtype=float)

    # Save
    np.savetxt(
        args.out,
        np.column_stack([ids, averaged]),
        fmt="%s",
        header="id,prediction",
        comments="",
    )
    print(f"Averaged predictions from {len(ckpt_task_pairs)} checkpoints saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
