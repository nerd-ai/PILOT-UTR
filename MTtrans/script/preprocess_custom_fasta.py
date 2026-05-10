# import argparse
# from pathlib import Path

# import pandas as pd
# from Bio import SeqIO


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Prepare FASTA sequences for MTtrans.")
#     parser.add_argument(
#         "--input",
#         required=True,
#         help="Path to the input FASTA file (e.g. utrdb_filter.fasta).",
#     )
#     parser.add_argument(
#         "--output",
#         required=True,
#         help="Destination CSV (e.g. data/custom_utrdb.csv).",
#     )
#     parser.add_argument(
#         "--trim-len",
#         type=int,
#         default=100,
#         help="Number of nucleotides to retain from the 3′ end of each UTR (default: 100).",
#     )
#     return parser.parse_args()


# def canonicalize(seq: str) -> str:
#     """Uppercase and map U→T."""
#     return seq.upper().replace("U", "T")


# def truncate_suffix(seq: str, trim_len: int) -> str:
#     """
#     Keep only the last `trim_len` nucleotides of the sequence.
#     Left-pad with Ns so the result is always exactly `trim_len` long.
#     """
#     return ("N" * trim_len + seq)[-trim_len:]


# def main() -> None:
#     args = parse_args()
#     fasta_path = Path(args.input)
#     out_path = Path(args.output)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     rows = []
#     for record in SeqIO.parse(fasta_path, "fasta"):
#         raw_seq = canonicalize(str(record.seq))
#         trimmed = truncate_suffix(raw_seq, args.trim_len)
#         rows.append(
#             {
#                 "id": record.id,
#                 "utr_raw": raw_seq,
#                 "utr_len": len(raw_seq),
#                 "utr_trimmed": trimmed,
#             }
#         )

#     df = pd.DataFrame(rows)
#     df.to_csv(out_path, index=False)
#     print(f"Saved {len(df)} sequences to {out_path.resolve()}")


# if __name__ == "__main__":
#     main()





import argparse
from pathlib import Path
from typing import Iterable, Tuple, Optional

import pandas as pd
from Bio import SeqIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UTR sequences for MTtrans from FASTA or CSV.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input FASTA or CSV (e.g. utrdb_filter.fasta or utrdb_filter.csv).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination CSV (e.g. data/custom_utrdb.csv).",
    )
    parser.add_argument(
        "--trim-len",
        type=int,
        default=100,
        help="Default nucleotides to retain from the 3′ end (used if --len-col is not set).",
    )
    parser.add_argument(
        "--len-col",
        choices=["nt_length", "token_length"],
        default=None,
        help="Optional column name in CSV providing per-row trim length (overrides --trim-len).",
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help="Optional column name in CSV to use as sequence ID. If omitted, uses 'id' if present; otherwise row_#.",
    )
    return parser.parse_args()


def canonicalize(seq: str) -> str:
    """Uppercase and map U→T."""
    return seq.upper().replace("U", "T")


def truncate_suffix(seq: str, trim_len: int) -> str:
    """
    Keep only the last `trim_len` nucleotides of the sequence.
    Left-pad with Ns so the result is always exactly `trim_len` long.
    """
    return ("N" * trim_len + seq)[-trim_len:]


def read_from_fasta(path: Path, default_trim_len: int) -> Iterable[Tuple[str, str, int]]:
    """
    Yields (id, canonical_seq, trim_len) from a FASTA file,
    using the global default trim length.
    """
    for record in SeqIO.parse(path, "fasta"):
        seq = canonicalize(str(record.seq))
        yield record.id, seq, default_trim_len


def read_from_csv(
    path: Path,
    default_trim_len: int,
    len_col: Optional[str],
    id_col: Optional[str],
) -> Iterable[Tuple[str, str, int]]:
    """
    Yields (id, canonical_seq, trim_len) from a CSV.
    Expects a 'seq' column. Optionally uses len_col for per-row trim length.
    """
    df = pd.read_csv(path)

    if "seq" not in df.columns:
        raise ValueError("CSV must contain a 'seq' column.")

    # Determine ID column
    if id_col is not None:
        if id_col not in df.columns:
            raise ValueError(f"--id-col '{id_col}' not found in CSV columns: {list(df.columns)}")
        id_series = df[id_col].astype(str)
    elif "id" in df.columns:
        id_series = df["id"].astype(str)
    else:
        id_series = pd.Series([f"row_{i}" for i in range(len(df))])

    # Determine per-row trim length
    if len_col is not None:
        if len_col not in df.columns:
            raise ValueError(f"--len-col '{len_col}' not found in CSV columns: {list(df.columns)}")
        # Coerce to integer and validate
        trim_series = pd.to_numeric(df[len_col], errors="coerce").fillna(default_trim_len).astype(int)
        if (trim_series <= 0).any():
            bad = trim_series[trim_series <= 0]
            raise ValueError(f"Non-positive values found in --len-col '{len_col}': {bad.unique()[:5]}")
    else:
        trim_series = pd.Series([default_trim_len] * len(df))

    seq_series = df["seq"].astype(str).map(canonicalize)

    for sid, sseq, slen in zip(id_series, seq_series, trim_series):
        yield sid, sseq, int(slen)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Detect input type by extension
    ext = in_path.suffix.lower()
    if ext == ".csv":
        iterator = read_from_csv(
            in_path,
            default_trim_len=args.trim_len,
            len_col=args.len_col,
            id_col=args.id_col,
        )
    else:
        # Treat anything else as FASTA (supports .fa, .fasta, .fna, etc.)
        iterator = read_from_fasta(in_path, default_trim_len=args.trim_len)

    # Build rows
    rows = []
    count = 0
    for sid, raw_seq, trim_len in iterator:
        trimmed = truncate_suffix(raw_seq, trim_len)
        rows.append(
            {
                "id": sid,
                "utr_raw": raw_seq,
                "utr_len": len(raw_seq),
                "trim_source_len": trim_len,   # new: records the length actually used
                "utr_trimmed": trimmed,
            }
        )
        count += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} sequences to {out_path.resolve()}")


if __name__ == "__main__":
    main()
