#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate sequence diversity from a CSV file."
    )
    parser.add_argument("csv_path", help="Path to the input CSV file.")
    parser.add_argument(
        "--seq-column",
        default="seq",
        help="CSV column containing sequences. Default: seq",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size used for exact pairwise Hamming computation. Default: 256",
    )
    parser.add_argument(
        "--max-exact-pairs",
        type=int,
        default=20_000_000,
        help="Use exact pairwise Hamming if total within-length pairs stay below this limit. Default: 20000000",
    )
    parser.add_argument(
        "--sample-pairs",
        type=int,
        default=200_000,
        help="Number of random within-length pairs to sample when exact computation is disabled. Default: 200000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for pair sampling. Default: 0",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Optional path to write the results as JSON.",
    )
    return parser.parse_args()


def load_sequences(csv_path, seq_column):
    sequences = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or seq_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(
                f"Column '{seq_column}' not found in {csv_path}. Available columns: {available}"
            )
        for row in reader:
            seq = row[seq_column].strip().upper()
            if seq:
                sequences.append(seq)
    if not sequences:
        raise ValueError("No non-empty sequences were found.")
    return sequences


def shannon_entropy(chars):
    total = len(chars)
    counts = Counter(chars)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def encode_sequences(sequences):
    return np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8).reshape(
        len(sequences), len(sequences[0])
    )


def exact_pairwise_hamming(encoded, chunk_size):
    n = encoded.shape[0]
    distances = []
    nearest = np.full(n, encoded.shape[1], dtype=np.int32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        left = encoded[start:end]

        intra = (left[:, None, :] != left[None, :, :]).sum(axis=2)
        if end - start > 1:
            iu = np.triu_indices(end - start, k=1)
            distances.append(intra[iu].astype(np.int16, copy=False))
            masked = intra.copy()
            np.fill_diagonal(masked, encoded.shape[1] + 1)
            nearest[start:end] = np.minimum(nearest[start:end], masked.min(axis=1))

        if end < n:
            right = encoded[end:]
            inter = (left[:, None, :] != right[None, :, :]).sum(axis=2)
            distances.append(inter.reshape(-1).astype(np.int16, copy=False))
            nearest[start:end] = np.minimum(nearest[start:end], inter.min(axis=1))
            nearest[end:] = np.minimum(nearest[end:], inter.min(axis=0))

    if distances:
        all_distances = np.concatenate(distances)
    else:
        all_distances = np.array([], dtype=np.int16)
    return all_distances, nearest


def sample_pairwise_hamming(encoded, sample_pairs, rng):
    n, seq_len = encoded.shape
    if n < 2:
        return np.array([], dtype=np.int16), np.array([], dtype=np.int16)

    first = rng.integers(0, n, size=sample_pairs)
    second = rng.integers(0, n - 1, size=sample_pairs)
    second = second + (second >= first)

    distances = (encoded[first] != encoded[second]).sum(axis=1).astype(np.int16, copy=False)

    nearest = np.full(n, seq_len, dtype=np.int16)
    sampled = min(sample_pairs, max(10_000, 5 * n))
    first = rng.integers(0, n, size=sampled)
    second = rng.integers(0, n - 1, size=sampled)
    second = second + (second >= first)
    nn_dist = (encoded[first] != encoded[second]).sum(axis=1).astype(np.int16, copy=False)
    np.minimum.at(nearest, first, nn_dist)
    np.minimum.at(nearest, second, nn_dist)

    return distances, nearest


def summarize_distances(distances):
    if len(distances) == 0:
        return None
    return {
        "count": int(len(distances)),
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "min": int(np.min(distances)),
        "q25": float(np.percentile(distances, 25)),
        "median": float(np.median(distances)),
        "q75": float(np.percentile(distances, 75)),
        "max": int(np.max(distances)),
    }


def summarize_nearest(nearest):
    if len(nearest) == 0:
        return None
    return {
        "mean": float(np.mean(nearest)),
        "std": float(np.std(nearest)),
        "min": int(np.min(nearest)),
        "median": float(np.median(nearest)),
        "max": int(np.max(nearest)),
    }


def analyze_group(sequences, chunk_size, max_exact_pairs, sample_pairs, rng):
    encoded = encode_sequences(sequences)
    n = len(sequences)
    pairs = n * (n - 1) // 2
    method = "exact" if pairs <= max_exact_pairs else "sampled"
    if method == "exact":
        distances, nearest = exact_pairwise_hamming(encoded, chunk_size)
    else:
        distances, nearest = sample_pairwise_hamming(encoded, sample_pairs, rng)

    position_entropies = [
        shannon_entropy(encoded[:, idx].tobytes().decode("ascii"))
        for idx in range(encoded.shape[1])
    ]

    counts = Counter(sequences)
    duplicate_clusters = sum(1 for count in counts.values() if count > 1)
    max_duplicate_count = max(counts.values())

    return {
        "count": n,
        "sequence_length": len(sequences[0]),
        "unique_sequences": len(counts),
        "unique_fraction": len(counts) / n,
        "duplicate_clusters": duplicate_clusters,
        "max_duplicate_count": max_duplicate_count,
        "method": method,
        "total_pairs": int(pairs),
        "pairwise_hamming": summarize_distances(distances),
        "nearest_neighbor_hamming": summarize_nearest(nearest),
        "position_entropy": {
            "mean": float(np.mean(position_entropies)),
            "std": float(np.std(position_entropies)),
            "min": float(np.min(position_entropies)),
            "max": float(np.max(position_entropies)),
        },
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    sequences = load_sequences(args.csv_path, args.seq_column)
    by_length = {}
    for seq in sequences:
        by_length.setdefault(len(seq), []).append(seq)

    analyzed_pairs = sum(len(group) * (len(group) - 1) // 2 for group in by_length.values())
    total_pairs = len(sequences) * (len(sequences) - 1) // 2
    skipped_cross_length_pairs = total_pairs - analyzed_pairs

    results = {
        "csv_path": str(Path(args.csv_path).resolve()),
        "sequence_column": args.seq_column,
        "total_sequences": len(sequences),
        "length_distribution": {str(k): len(v) for k, v in sorted(by_length.items())},
        "analyzed_within_length_pairs": int(analyzed_pairs),
        "skipped_cross_length_pairs": int(skipped_cross_length_pairs),
        "groups": [
            analyze_group(group, args.chunk_size, args.max_exact_pairs, args.sample_pairs, rng)
            for _, group in sorted(by_length.items())
        ],
    }

    print(f"File: {results['csv_path']}")
    print(f"Total sequences: {results['total_sequences']}")
    print(f"Length distribution: {results['length_distribution']}")
    if skipped_cross_length_pairs:
        print(f"Skipped cross-length pairs: {skipped_cross_length_pairs}")

    for group in results["groups"]:
        print()
        print(
            f"Length {group['sequence_length']}: {group['count']} sequences, "
            f"{group['unique_sequences']} unique ({group['unique_fraction']:.4f})"
        )
        print(
            f"Pairwise Hamming method: {group['method']} over {group['total_pairs']} within-length pairs"
        )
        pairwise = group["pairwise_hamming"]
        if pairwise:
            print(
                "Pairwise Hamming: "
                f"mean={pairwise['mean']:.3f}, std={pairwise['std']:.3f}, "
                f"min={pairwise['min']}, q25={pairwise['q25']:.3f}, "
                f"median={pairwise['median']:.3f}, q75={pairwise['q75']:.3f}, max={pairwise['max']}"
            )
        nearest = group["nearest_neighbor_hamming"]
        if nearest:
            print(
                "Nearest-neighbor Hamming: "
                f"mean={nearest['mean']:.3f}, std={nearest['std']:.3f}, "
                f"min={nearest['min']}, median={nearest['median']:.3f}, max={nearest['max']}"
            )
        entropy = group["position_entropy"]
        print(
            "Per-position entropy (bits): "
            f"mean={entropy['mean']:.3f}, std={entropy['std']:.3f}, "
            f"min={entropy['min']:.3f}, max={entropy['max']:.3f}"
        )
        print(
            f"Duplicate clusters: {group['duplicate_clusters']}, "
            f"largest duplicate count: {group['max_duplicate_count']}"
        )

    if args.json_path:
        with open(args.json_path, "w") as handle:
            json.dump(results, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
