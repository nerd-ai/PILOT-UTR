#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_INPUT = Path(
    "/home/xli263/xli/utr_design/DRAKES/drakes_rna/baseline_generated_sequences/generated_cfg_high_te.csv"
)


def myers_levenshtein(a: str, b: str) -> int:
    """Bit-parallel Levenshtein distance.

    This is efficient for the short DNA/RNA strings used here and avoids an
    external dependency such as rapidfuzz or python-Levenshtein.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a

    m = len(a)
    bitmask = (1 << m) - 1
    top_bit = 1 << (m - 1)

    peq: dict[str, int] = {}
    for i, char in enumerate(a):
        peq[char] = peq.get(char, 0) | (1 << i)

    pv = bitmask
    mv = 0
    score = m

    for char in b:
        eq = peq.get(char, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | ~(xh | pv)
        mh = pv & xh

        if ph & top_bit:
            score += 1
        elif mh & top_bit:
            score -= 1

        ph = ((ph << 1) | 1) & bitmask
        mh = (mh << 1) & bitmask
        pv = (mh | ~(xv | ph)) & bitmask
        mv = ph & xv

    return score


def normalized_levenshtein(a: str, b: str) -> float:
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return myers_levenshtein(a, b) / denom


def load_sequences(path: Path, seq_column: str, unique: bool) -> list[str]:
    df = pd.read_csv(path)
    if seq_column not in df.columns:
        raise ValueError(f"Sequence column '{seq_column}' not found in {path}")
    seqs = df[seq_column].dropna().astype(str).str.upper().tolist()
    if unique:
        seqs = list(dict.fromkeys(seqs))
    if len(seqs) < 2:
        raise ValueError("Need at least two sequences to compute nearest-neighbor distance.")
    return seqs


def mean_nearest_neighbor_normalized_levenshtein(
    seqs: Iterable[str],
) -> tuple[float, list[float]]:
    seqs = list(seqs)
    n = len(seqs)
    lengths = [len(seq) for seq in seqs]
    nearest = [math.inf] * n

    for i in range(n - 1):
        seq_i = seqs[i]
        len_i = lengths[i]
        for j in range(i + 1, n):
            len_j = lengths[j]
            denom = max(len_i, len_j)
            if denom == 0:
                dist_norm = 0.0
            else:
                lower_bound = abs(len_i - len_j) / denom
                if lower_bound >= nearest[i] and lower_bound >= nearest[j]:
                    continue
                dist_norm = myers_levenshtein(seq_i, seqs[j]) / denom
            if dist_norm < nearest[i]:
                nearest[i] = dist_norm
            if dist_norm < nearest[j]:
                nearest[j] = dist_norm

    mean_nn = sum(nearest) / n
    return mean_nn, nearest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean nearest-neighbor normalized Levenshtein distance for generated sequences."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seq-column", default="seq")
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Deduplicate sequences before computing the metric. By default duplicates are kept.",
    )
    parser.add_argument(
        "--save-per-seq",
        type=Path,
        default=None,
        help="Optional CSV path to save each sequence and its nearest-neighbor distance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seqs = load_sequences(args.input_csv, args.seq_column, unique=args.unique)
    mean_nn, nearest = mean_nearest_neighbor_normalized_levenshtein(seqs)

    print(f"Input: {args.input_csv}")
    print(f"Sequence column: {args.seq_column}")
    print(f"Sequences used: {len(seqs)}")
    print(f"Unique mode: {args.unique}")
    print(f"Mean nearest-neighbor normalized Levenshtein distance: {mean_nn:.6f}")
    print(f"Min nearest-neighbor distance: {min(nearest):.6f}")
    print(f"Max nearest-neighbor distance: {max(nearest):.6f}")

    if args.save_per_seq is not None:
        out = pd.DataFrame(
            {
                "seq": seqs,
                "nearest_neighbor_normalized_levenshtein": nearest,
            }
        )
        args.save_per_seq.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.save_per_seq, index=False)
        print(f"Saved per-sequence nearest-neighbor distances to: {args.save_per_seq}")


if __name__ == "__main__":
    main()
