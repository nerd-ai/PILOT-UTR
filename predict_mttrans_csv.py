import argparse
from pathlib import Path

import pandas as pd
import torch

import mttrans


BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
DEFAULT_INPUT_CSV = Path(
    "/home/xli263/xli/utr_design/DRAKES/drakes_rna/generated_sequences_human_mrl_hacking/generated_negative_sample_RP_PC3_te_mttrans_3e-5.csv"
)
DEFAULT_OUTPUT_CSV = Path(
    "/home/xli263/xli/utr_design/DRAKES/drakes_rna/generated_sequences_human_mrl_hacking/generated_negative_sample_RP_PC3_te_mttrans_3e-5_augment.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MTTrans TE inference on a CSV and append prediction scores."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--seq-column", default="seq")
    parser.add_argument("--score-column", default="mttrans_pred")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=mttrans.DEFAULT_MTTRANS_CKPT,
        help="Path to the serialized MTTrans checkpoint.",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=105,
        help="Fixed MTTrans input length used for left padding / last-N truncation.",
    )
    parser.add_argument(
        "--task",
        choices=["RP_293T", "RP_muscle", "RP_PC3"],
        default=None,
        help=(
            "MTTrans task head to use. By default, use the task stored in the "
            "serialized checkpoint."
        ),
    )
    parser.add_argument(
        "--train-mode",
        action="store_true",
        help="Run the MTTrans model in train() mode during inference to match training-style forward behavior.",
    )
    return parser.parse_args()


def sequences_to_base_tensor(seqs: list[str], input_length: int) -> torch.Tensor:
    x = torch.zeros(len(seqs), 4, input_length, dtype=torch.float32)
    for batch_idx, seq in enumerate(seqs):
        seq = seq.upper()
        if len(seq) > input_length:
            seq = seq[-input_length:]
        start = input_length - len(seq)
        for pos, base in enumerate(seq):
            base_idx = BASE_TO_IDX.get(base)
            if base_idx is not None:
                x[batch_idx, base_idx, start + pos] = 1.0
    return x


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    if args.seq_column not in df.columns:
        raise ValueError(f"Sequence column '{args.seq_column}' not found in {args.input_csv}")

    sequences = df[args.seq_column].fillna("").astype(str).tolist()
    oracle = mttrans.get_mttrans_oracle(
        checkpoint_path=args.checkpoint,
        map_location=args.device,
        input_length=args.input_length,
        task=args.task,
    )
    oracle.to(args.device)
    if args.train_mode:
        oracle.train()
        oracle.model.train()
    else:
        oracle.eval()
        oracle.model.eval()

    preds = []
    with torch.no_grad():
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start : start + args.batch_size]
            batch_x = sequences_to_base_tensor(batch, args.input_length).to(args.device)
            batch_preds = oracle(batch_x).squeeze(-1).detach().cpu().tolist()
            preds.extend(float(x) for x in batch_preds)

    out_df = df.copy()
    out_df[args.score_column] = preds
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"Scored {len(out_df)} sequences -> {args.output_csv}")
    print(f"MTTrans task: {getattr(oracle.model, 'task', None)}")
    print(f"Average {args.score_column}: {out_df[args.score_column].mean():.6f}")


if __name__ == "__main__":
    main()
