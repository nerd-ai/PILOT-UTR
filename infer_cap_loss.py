import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import os

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from grelu.lightning import LightningModel
from grelu.sequence.format import strings_to_one_hot


DEFAULT_CHECKPOINT = Path(
    "/home/xli263/xli/utr_design/DRAKES/drakes_rna/experiment/single_run_new/hybrid_best.ckpt"
)
DEFAULT_INPUT = Path(
    "/home/xli263/xli/utr_design/DRAKES/drakes_rna/data_and_model/train_50%_low_augmentation_70%_G_C.csv"
)
DEFAULT_OUTPUT = Path(
    "/home/xli263/xli/utr_design/DRAKES/drakes_rna/data_and_model/train_50%_low_augmentation_70%_G_C_cap_scores_new.csv"
)


class SequenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.rows = frame.to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def collate_sequences(batch: list[dict]) -> tuple[list[dict], torch.Tensor]:
    seqs = [row["mutated_utr"].strip().upper() for row in batch]
    return batch, strings_to_one_hot(seqs)


def reduce_to_scalar_scores(preds: torch.Tensor) -> torch.Tensor:
    if preds.ndim == 1:
        return preds

    flat = preds.reshape(preds.shape[0], -1)
    if flat.shape[1] != 1:
        raise ValueError(
            f"Expected one scalar prediction per sequence, got shape {tuple(preds.shape)}."
        )
    return flat[:, 0]


def build_model(device: torch.device) -> LightningModel:
    model_params = {
        "model_type": "EnformerPretrainedModel",
        "n_tasks": 1,
        "n_transformers": 3,
    }
    train_params = {
        "task": "regression",
        "loss": "MSE",
        "logger": None,
        "devices": "cpu",
        "checkpoint": False,
    }
    model = LightningModel(model_params=model_params, train_params=train_params)
    return model.to(device)


def load_checkpoint(model: LightningModel, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score mutated sequences with a checkpoint and compute hinge cap loss."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=6.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    frame = pd.read_csv(args.input_csv)
    if "mutated_utr" not in frame.columns:
        raise ValueError("Input CSV must contain a 'mutated_utr' column.")

    loader = DataLoader(
        SequenceDataset(frame),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
    )

    model = build_model(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    pred_scores = []
    cap_losses = []

    with torch.no_grad():
        for rows, seq_x in loader:
            seq_x = seq_x.to(device)
            scores = reduce_to_scalar_scores(model(seq_x, logits=True)).detach().cpu()
            caps = torch.clamp(scores - args.threshold, min=0.0)

            pred_scores.extend(scores.tolist())
            cap_losses.extend(caps.tolist())

    output = frame.copy()
    output["pred_score"] = pred_scores
    output["cap_loss"] = cap_losses
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)

    mean_cap_loss = float(torch.tensor(cap_losses, dtype=torch.float32).mean().item()) if cap_losses else float("nan")
    frac_capped = (
        float((torch.tensor(cap_losses, dtype=torch.float32) > 0).float().mean().item())
        if cap_losses
        else float("nan")
    )

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Scored rows: {len(output)}")
    print(f"Threshold T: {args.threshold}")
    print(f"Mean cap loss: {mean_cap_loss:.6f}")
    print(f"Fraction above threshold: {frac_capped:.6f}")
    print(f"Saved output to: {args.output_csv}")


if __name__ == "__main__":
    main()
