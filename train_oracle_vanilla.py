import argparse
import os
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from grelu.lightning import LightningModel
from grelu.sequence.format import strings_to_one_hot


DEFAULT_TRAIN_CSV = Path(
    "/your_path/train_dataset_mrl.csv"
)
DEFAULT_VAL_CSV = Path(
    "/your_path/val_dataset_mrl.csv"
)
DEFAULT_SAVE_DIR = Path(
    "/your_path/oracle_vanilla_mrl"
)

SEED = 42
BATCH_SIZE = 512
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
MAX_EPOCHS = 15
USE_WANDB = os.environ.get("ENABLE_WANDB", "1") == "1"
DEFAULT_SEQUENCE_COLUMN = "utr"
DEFAULT_TARGET_COLUMN = "rl"
DEFAULT_SEQ_LEN = 50
PAD_BASE = "N"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the vanilla oracle on a regression target from explicit train/validation CSVs."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help="Training CSV path, for example /your_path/train_dataset_mrl.csv.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=DEFAULT_VAL_CSV,
        help="Validation CSV path, for example /your_path/val_dataset_mrl.csv.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="Directory where vanilla_best.ckpt will be saved.",
    )
    parser.add_argument("--sequence-column", default=DEFAULT_SEQUENCE_COLUMN)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument(
        "--pad-side",
        choices=["right", "left"],
        default="right",
        help="Pad shorter sequences on the right or left before one-hot encoding.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


class RegressionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        sequence_column: str,
        target_column: str,
    ) -> None:
        self.rows = [
            (row[sequence_column].strip().upper(), float(row[target_column]))
            for row in frame[[sequence_column, target_column]].to_dict("records")
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[str, float]:
        return self.rows[idx]


def normalize_sequence_length(seq: str, seq_len: int, pad_side: str = "right") -> str:
    seq = seq[:seq_len]
    if len(seq) < seq_len:
        pad = PAD_BASE * (seq_len - len(seq))
        if pad_side == "left":
            seq = pad + seq
        else:
            seq = seq + pad
    return seq


def collate_regression(
    batch: list[tuple[str, float]],
    seq_len: int,
    pad_side: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    seqs, labels = zip(*batch)
    seq_x = strings_to_one_hot(
        [normalize_sequence_length(seq, seq_len, pad_side=pad_side) for seq in seqs]
    )
    label_y = torch.tensor(labels, dtype=torch.float32)
    return seq_x, label_y


def reduce_to_scalar_scores(preds: torch.Tensor) -> torch.Tensor:
    if preds.ndim == 1:
        return preds

    flat = preds.reshape(preds.shape[0], -1)
    if flat.shape[1] != 1:
        raise ValueError(
            f"Expected one scalar prediction per sequence, got shape {tuple(preds.shape)}."
        )
    return flat[:, 0]


def pearson_corr(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.float()
    targets = targets.float()
    preds_centered = preds - preds.mean()
    targets_centered = targets - targets.mean()
    denom = torch.sqrt(preds_centered.pow(2).sum() * targets_centered.pow(2).sum())
    if denom.item() == 0:
        return float("nan")
    return ((preds_centered * targets_centered).sum() / denom).item()


def choose_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    gpu_index = 7
    return torch.device(f"cuda:{gpu_index}")


def build_model(device: torch.device) -> LightningModel:
    model_params = {
        "model_type": "EnformerModel",
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


def evaluate_regression(
    model: LightningModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    losses = []
    preds_all = []
    targets_all = []

    model.eval()
    with torch.no_grad():
        for seq_x, labels in loader:
            seq_x = seq_x.to(device)
            labels = labels.to(device)

            preds = reduce_to_scalar_scores(model(seq_x, logits=True))
            loss = F.mse_loss(preds, labels)

            batch_size = labels.shape[0]
            losses.append(loss.item() * batch_size)
            preds_all.append(preds.detach().cpu())
            targets_all.append(labels.detach().cpu())

    model.train()

    if len(loader.dataset) == 0:
        return float("nan"), float("nan")

    mean_loss = sum(losses) / len(loader.dataset)
    pearson = pearson_corr(torch.cat(preds_all), torch.cat(targets_all))
    return mean_loss, pearson


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()

    train_df = pd.read_csv(args.train_csv)[[args.sequence_column, args.target_column]].dropna()
    val_df = pd.read_csv(args.val_csv)[[args.sequence_column, args.target_column]].dropna()

    regression_train_loader = DataLoader(
        RegressionDataset(
            train_df.reset_index(drop=True),
            sequence_column=args.sequence_column,
            target_column=args.target_column,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=lambda batch: collate_regression(batch, args.seq_len, args.pad_side),
    )
    regression_val_loader = DataLoader(
        RegressionDataset(
            val_df.reset_index(drop=True),
            sequence_column=args.sequence_column,
            target_column=args.target_column,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda batch: collate_regression(batch, args.seq_len, args.pad_side),
    )

    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    checkpoint_path = args.save_dir / "vanilla_best.ckpt"
    best_val_loss = float("inf")

    run = None
    if USE_WANDB:
        run = wandb.init(
            entity=None,
            project="UTR-design",
            job_type="training",
            group="single-split-direct-te",
            name="train-single-run-vanilla-direct-te",
            config={
                "train_path": str(args.train_csv),
                "val_path": str(args.val_csv),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "sequence_column": args.sequence_column,
                "target_column": args.target_column,
                "seq_len": args.seq_len,
                "batch_size": BATCH_SIZE,
                "lr": LEARNING_RATE,
                "max_epochs": MAX_EPOCHS,
                "loss_type": "mse",
                "seed": args.seed,
            },
            reinit=True,
        )

    print(
        f"Starting vanilla regression training on {device}, checkpoint dir: {args.save_dir}, "
        f"train_rows={len(train_df)}, val_rows={len(val_df)}, "
        f"target={args.target_column}, seq_len={args.seq_len}, seed={args.seed}"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss_total = 0.0
        n_examples = 0

        for seq_x, labels in regression_train_loader:
            seq_x = seq_x.to(device)
            labels = labels.to(device)

            preds = reduce_to_scalar_scores(model(seq_x, logits=True))
            loss = F.mse_loss(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            train_loss_total += loss.item() * batch_size
            n_examples += batch_size

        train_loss = train_loss_total / n_examples
        val_loss, val_pearson = evaluate_regression(
            model=model,
            loader=regression_val_loader,
            device=device,
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pearson": val_pearson,
        }
        print(
            f"Epoch {epoch}/{MAX_EPOCHS} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_pearson={val_pearson:.6f}"
        )
        if run is not None:
            wandb.log(metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_params": model.model_params,
                    "train_params": model.train_params,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_pearson": val_pearson,
                    "train_csv": str(args.train_csv),
                    "val_csv": str(args.val_csv),
                    "sequence_column": args.sequence_column,
                    "target_column": args.target_column,
                    "seq_len": args.seq_len,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

    if run is not None:
        wandb.finish()

    print(f"Saved best vanilla checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
