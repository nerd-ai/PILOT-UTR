# import argparse
# import os
# import sys
# from typing import Iterable, List, Tuple
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# import numpy as np
# import torch
# from Bio import SeqIO

# from models.popen import Auto_popen
# from utils import load_model
# from models.reader import one_hot, pad_zeros
# from evaluation.evaluation_func import reverse_transform


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Run MTtrans on a FASTA file.")
#     parser.add_argument("--config", required=True, help="Path to the MTtrans .ini file.")
#     parser.add_argument("--checkpoint", required=True, help="Path to the .pth checkpoint.")
#     parser.add_argument("--fasta", required=True, help="FASTA file with 5' UTR sequences.")
#     parser.add_argument("--task", default="MPA_H",
#                         choices=["MPA_U", "MPA_H", "MPA_V"],
#                         help="Which MTtrans head to use for prediction (MPA_H suits human UTRs).")
#     parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference.")
#     parser.add_argument("--out", default="mttrans_predictions.csv", help="Where to save the outputs.")
#     parser.add_argument("--denormalize", action="store_true",
#                         help="Use evaluation.reverse_transform to map predictions back to original TE scale.")
#     return parser.parse_args()


# def trim_to_100_nt(seq: str) -> str:
#     seq = seq.upper().replace("U", "T")
#     return ("N" * 100 + seq)[-100:]


# def encode_sequence(seq: str, pad_to: int) -> torch.Tensor:
#     oh = one_hot(seq).astype(np.float32)
#     tensor = torch.tensor(oh, dtype=torch.float32)
#     padded = pad_zeros(oh, pad_to)
#     # return padded if isinstance(padded, torch.Tensor) else torch.tensor(padded, dtype=torch.float32)
#     return padded


# def batch_iter(records: List[Tuple[str, torch.Tensor]], batch_size: int) -> Iterable[Tuple[List[str], torch.Tensor]]:
#     for i in range(0, len(records), batch_size):
#         chunk = records[i : i + batch_size]
#         ids, tensors = zip(*chunk)
#         batch = torch.stack(tensors, dim=0)
#         yield list(ids), batch


# def main() -> None:
#     args = parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     cfg = Auto_popen(args.config)
#     # Point Auto_popen at the exact checkpoint supplied via CLI
#     # def _patched_path(self):
#     #     return args.checkpoint
#     # Auto_popen.vae_pth_path.fget = lambda self: args.checkpoint  # type: ignore

#     model = cfg.Model_Class(*cfg.model_args)
#     model = load_model(cfg, model)
#     model.to(device)
#     model.eval()

#     pad_to = cfg.pad_to
#     fasta_records = list(SeqIO.parse(args.fasta, "fasta"))
#     encoded = [
#         (rec.id, encode_sequence(trim_to_100_nt(str(rec.seq)), pad_to))
#         for rec in fasta_records
#     ]

#     preds = {}
#     with torch.no_grad():
#         for ids, batch in batch_iter(encoded, args.batch_size):
#             batch = batch.to(device)
#             model.task = args.task
#             out = model(batch).squeeze(-1).cpu().numpy()
#             if args.denormalize:
#                 out = reverse_transform(out, args.task)
#             preds.update(zip(ids, out))

#     np.savetxt(args.out,
#                 np.column_stack([list(preds.keys()), list(preds.values())]),
#                 fmt="%s",
#                 header="id,prediction",
#                 comments="")
#     print(f"Wrote {len(preds)} predictions to {os.path.abspath(args.out)}")


# if __name__ == "__main__":
#     main()



import argparse
import os
import sys
from typing import Iterable, List, Tuple
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from models.popen import Auto_popen
from utils import load_model
from models.reader import one_hot, pad_zeros
from evaluation.evaluation_func import reverse_transform


# ---------------------- #
# Argument Parsing
# ---------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MTtrans on a FASTA or CSV file.")
    parser.add_argument("--config", required=True, help="Path to the MTtrans .ini file.")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pth checkpoint.")
    parser.add_argument("--input", required=True,
                        help="Input file: FASTA (.fasta, .fa) or CSV (.csv) with a 'seq' column.")
    parser.add_argument("--task", default="MPA_H",
                        choices=["MPA_U", "MPA_H", "MPA_V"],
                        help="Which MTtrans head to use for prediction (MPA_H suits human UTRs).")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference.")
    parser.add_argument("--out", default="mpa_h_optimized.csv", help="Where to save the outputs.")
    parser.add_argument("--denormalize", action="store_true",
                        help="Use evaluation.reverse_transform to map predictions back to original TE scale.")
    parser.add_argument("--len-col", default=None,
                        help="Optional column name in CSV providing per-row trim length (e.g. nt_length).")
    parser.add_argument("--id-col", default=None,
                        help="Optional column name in CSV to use as sequence ID (defaults to 'id' or row_#).")
    return parser.parse_args()


# ---------------------- #
# Sequence Processing
# ---------------------- #
def canonicalize(seq: str) -> str:
    """Uppercase and map U→T."""
    return seq.upper().replace("U", "T")


def trim_to_n(seq: str, trim_len: int = 100) -> str:
    """Keep only the last `trim_len` nt; pad left with N if shorter."""
    seq = canonicalize(seq)
    return ("N" * trim_len + seq)[-trim_len:]


def encode_sequence(seq: str, pad_to: int) -> torch.Tensor:
    """One-hot encode and pad a sequence."""
    oh = one_hot(seq).astype(np.float32)
    padded = pad_zeros(oh, pad_to)
    return torch.tensor(padded, dtype=torch.float32)


def batch_iter(records: List[Tuple[str, torch.Tensor]], batch_size: int) -> Iterable[Tuple[List[str], torch.Tensor]]:
    """Yield batches of (ids, tensor_batch)."""
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        ids, tensors = zip(*chunk)
        yield list(ids), torch.stack(tensors, dim=0)


# ---------------------- #
# Input Readers
# ---------------------- #
def read_fasta(path: str, trim_len: int = 100) -> List[Tuple[str, str]]:
    """Read a FASTA file and return [(id, trimmed_seq)]."""
    records = []
    for rec in SeqIO.parse(path, "fasta"):
        trimmed = trim_to_n(str(rec.seq), trim_len)
        records.append((rec.id, trimmed))
    return records


def read_csv(path: str, len_col: str = None, id_col: str = None, default_trim: int = 100) -> List[Tuple[str, str]]:
    """Read a CSV with columns: seq (required), optional id, len_col."""
    df = pd.read_csv(path)
    if "seq" not in df.columns:
        raise ValueError("CSV input must contain a 'seq' column.")

    # Determine IDs
    if id_col and id_col in df.columns:
        ids = df[id_col].astype(str).tolist()
    elif "id" in df.columns:
        ids = df["id"].astype(str).tolist()
    else:
        ids = [f"row_{i}" for i in range(len(df))]

    # Determine per-row trim lengths
    if len_col and len_col in df.columns:
        trim_lengths = df[len_col].fillna(default_trim).astype(int).tolist()
    else:
        trim_lengths = [default_trim] * len(df)

    records = []
    for sid, seq, tlen in zip(ids, df["seq"].astype(str), trim_lengths):
        trimmed = trim_to_n(seq, tlen)
        records.append((sid, trimmed))
    return records


# ---------------------- #
# Main Inference Logic
# ---------------------- #
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model configuration and weights
    cfg = Auto_popen(args.config)
    model = cfg.Model_Class(*cfg.model_args)
    model = load_model(cfg, model)
    model.to(device)
    model.eval()

    pad_to = cfg.pad_to

    # --- Read input ---
    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".csv":
        seq_records = read_csv(args.input, args.len_col, args.id_col)
    else:
        seq_records = read_fasta(args.input)

    # --- Encode ---
    encoded = [(sid, encode_sequence(seq, pad_to)) for sid, seq in seq_records]

    # --- Predict ---
    preds = {}
    with torch.no_grad():
        for ids, batch in batch_iter(encoded, args.batch_size):
            batch = batch.to(device)
            model.task = args.task
            out = model(batch).squeeze(-1).cpu().numpy()
            if args.denormalize:
                out = reverse_transform(out, args.task)
            preds.update(zip(ids, out))

    # --- Save output ---
    np.savetxt(
        args.out,
        np.column_stack([list(preds.keys()), list(preds.values())]),
        fmt="%s",
        header="id,prediction",
        delimiter=",",         # <--- ADD THIS LINE
        comments=""
    )
    print(f"Wrote {len(preds)} predictions to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
