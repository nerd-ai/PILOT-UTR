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
DEFAULT_NEGATIVE_CSV = Path(
    "/your_path/negative_sequences.csv"
)
DEFAULT_SAVE_DIR = Path(
    "/your_path/oracle_cap_mrl"
)

SEED = 42
BATCH_SIZE = 512
NEGATIVE_BATCH_SIZE = 128
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
MAX_EPOCHS = 15
CAP_THRESHOLD = 6.5
CAP_LAMBDA = float(os.environ.get("CAP_LOSS_LAMBDA", "3.0"))
USE_WANDB = os.environ.get("ENABLE_WANDB", "1") == "1"
DEFAULT_SEQUENCE_COLUMN = "utr"
DEFAULT_TARGET_COLUMN = "rl"
DEFAULT_NEGATIVE_SEQUENCE_COLUMN = "seq"
DEFAULT_SEQ_LEN = 50
PAD_BASE = "N"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the vanilla oracle with standard regression MSE plus a "
            "negative-sample cap loss. Best checkpoint is selected by standard "
            "unweighted validation MSE only."
        )
    )
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_VAL_CSV)
    parser.add_argument("--negative-csv", type=Path, default=DEFAULT_NEGATIVE_CSV)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--sequence-column", default=DEFAULT_SEQUENCE_COLUMN)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--negative-sequence-column", default=DEFAULT_NEGATIVE_SEQUENCE_COLUMN)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument(
        "--pad-side",
        choices=["right", "left"],
        default="right",
        help="Pad shorter sequences on the right or left before one-hot encoding.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--negative-batch-size", type=int, default=NEGATIVE_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--cap-threshold", type=float, default=CAP_THRESHOLD)
    parser.add_argument("--cap-lambda", type=float, default=CAP_LAMBDA)
    return parser.parse_args()


class RegressionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        sequence_column: str,
        target_column: str,
    ) -> None:
        self.rows = [
            (str(row[sequence_column]).strip().upper(), float(row[target_column]))
            for row in frame[[sequence_column, target_column]].to_dict("records")
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[str, float]:
        return self.rows[idx]


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[str]) -> None:
        self.rows = sequences

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> str:
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


def collate_sequences(
    batch: list[str],
    seq_len: int,
    pad_side: str,
) -> torch.Tensor:
    return strings_to_one_hot(
        [normalize_sequence_length(seq, seq_len, pad_side=pad_side) for seq in batch]
    )


def reduce_to_scalar_scores(preds: torch.Tensor) -> torch.Tensor:
    if preds.ndim == 1:
        return preds

    flat = preds.reshape(preds.shape[0], -1)
    if flat.shape[1] != 1:
        raise ValueError(
            f"Expected one scalar prediction per sequence, got shape {tuple(preds.shape)}."
        )
    return flat[:, 0]


def cap_loss(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.clamp(scores - threshold, min=0.0).mean()


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


def evaluate_cap(
    model: LightningModel,
    loader: DataLoader,
    threshold: float,
    device: torch.device,
) -> float:
    losses = []

    model.eval()
    with torch.no_grad():
        for seq_x in loader:
            seq_x = seq_x.to(device)
            scores = reduce_to_scalar_scores(model(seq_x, logits=True))
            loss = cap_loss(scores, threshold)
            losses.append(loss.item() * seq_x.shape[0])

    model.train()

    if len(loader.dataset) == 0:
        return float("nan")
    return sum(losses) / len(loader.dataset)


def load_negative_sequences(csv_path: Path, sequence_column: str) -> list[str]:
    frame = pd.read_csv(csv_path)[[sequence_column]].dropna()
    sequences = [
        str(row[sequence_column]).strip().upper()
        for row in frame.to_dict("records")
    ]
    if not sequences:
        raise ValueError(f"No negative sequences found in {csv_path}")
    return sequences


def main() -> None:
    args = parse_args()
    if args.cap_lambda < 0:
        raise ValueError("--cap-lambda must be non-negative.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()

    train_df = pd.read_csv(args.train_csv)[[args.sequence_column, args.target_column]].dropna()
    val_df = pd.read_csv(args.val_csv)[[args.sequence_column, args.target_column]].dropna()
    negative_sequences = load_negative_sequences(
        args.negative_csv,
        sequence_column=args.negative_sequence_column,
    )

    regression_train_loader = DataLoader(
        RegressionDataset(
            train_df.reset_index(drop=True),
            sequence_column=args.sequence_column,
            target_column=args.target_column,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_regression(batch, args.seq_len, args.pad_side),
    )
    regression_val_loader = DataLoader(
        RegressionDataset(
            val_df.reset_index(drop=True),
            sequence_column=args.sequence_column,
            target_column=args.target_column,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_regression(batch, args.seq_len, args.pad_side),
    )
    negative_train_loader = DataLoader(
        SequenceDataset(negative_sequences),
        batch_size=args.negative_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_sequences(batch, args.seq_len, args.pad_side),
    )
    negative_eval_loader = DataLoader(
        SequenceDataset(negative_sequences),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_sequences(batch, args.seq_len, args.pad_side),
    )

    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint_path = args.save_dir / "cap_best.ckpt"
    best_val_loss = float("inf")

    run = None
    if USE_WANDB:
        run = wandb.init(
            entity=None,
            project="UTR-design",
            job_type="training",
            group="single-split-cap",
            name="train-single-run-cap-direct-te",
            config={
                "train_path": str(args.train_csv),
                "val_path": str(args.val_csv),
                "negative_path": str(args.negative_csv),
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "negative_rows": len(negative_sequences),
                "sequence_column": args.sequence_column,
                "target_column": args.target_column,
                "negative_sequence_column": args.negative_sequence_column,
                "seq_len": args.seq_len,
                "pad_side": args.pad_side,
                "batch_size": args.batch_size,
                "negative_batch_size": args.negative_batch_size,
                "lr": args.learning_rate,
                "max_epochs": args.max_epochs,
                "loss_type": "standard_mse_plus_negative_cap",
                "cap_lambda": args.cap_lambda,
                "cap_threshold": args.cap_threshold,
                "best_metric": "standard_unweighted_val_mse",
                "seed": args.seed,
            },
            reinit=True,
        )

    print(
        f"Starting cap-loss regression training on {device}, checkpoint dir: {args.save_dir}, "
        f"train_rows={len(train_df)}, val_rows={len(val_df)}, "
        f"negative_rows={len(negative_sequences)}, target={args.target_column}, "
        f"seq_len={args.seq_len}, cap_threshold={args.cap_threshold}, "
        f"cap_lambda={args.cap_lambda}, seed={args.seed}"
    )

    for epoch in range(1, args.max_epochs + 1):
        train_regression_loss_total = 0.0
        train_cap_loss_total = 0.0
        train_total_loss_total = 0.0
        n_regression_examples = 0
        n_negative_examples = 0
        n_steps = 0

        negative_iter = iter(negative_train_loader)
        for regression_x, labels in regression_train_loader:
            regression_x = regression_x.to(device)
            labels = labels.to(device)

            try:
                negative_x = next(negative_iter)
            except StopIteration:
                negative_iter = iter(negative_train_loader)
                negative_x = next(negative_iter)
            negative_x = negative_x.to(device)

            all_x = torch.cat([regression_x, negative_x], dim=0)
            all_scores = reduce_to_scalar_scores(model(all_x, logits=True))

            reg_batch_size = labels.shape[0]
            negative_batch_current = negative_x.shape[0]
            regression_scores = all_scores[:reg_batch_size]
            negative_scores = all_scores[reg_batch_size:]

            regression_loss = F.mse_loss(regression_scores, labels)
            negative_cap = cap_loss(negative_scores, args.cap_threshold)
            total_loss = regression_loss + args.cap_lambda * negative_cap

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_regression_loss_total += regression_loss.item() * reg_batch_size
            train_cap_loss_total += negative_cap.item() * negative_batch_current
            train_total_loss_total += total_loss.item()
            n_regression_examples += reg_batch_size
            n_negative_examples += negative_batch_current
            n_steps += 1

        train_regression_loss = train_regression_loss_total / n_regression_examples
        train_cap_loss = train_cap_loss_total / n_negative_examples
        train_total_loss = train_total_loss_total / n_steps

        val_loss, val_pearson = evaluate_regression(
            model=model,
            loader=regression_val_loader,
            device=device,
        )
        val_cap_loss = evaluate_cap(
            model=model,
            loader=negative_eval_loader,
            threshold=args.cap_threshold,
            device=device,
        )

        metrics = {
            "epoch": epoch,
            "train_total_loss": train_total_loss,
            "train_regression_loss": train_regression_loss,
            "train_cap_loss": train_cap_loss,
            "val_loss": val_loss,
            "val_pearson": val_pearson,
            "val_cap_loss": val_cap_loss,
        }
        print(
            f"Epoch {epoch}/{args.max_epochs} "
            f"train_total_loss={train_total_loss:.6f} "
            f"train_regression_loss={train_regression_loss:.6f} "
            f"train_cap_loss={train_cap_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_cap_loss={val_cap_loss:.6f} "
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
                    "best_val_cap_loss": val_cap_loss,
                    "cap_lambda": args.cap_lambda,
                    "cap_threshold": args.cap_threshold,
                    "train_csv": str(args.train_csv),
                    "val_csv": str(args.val_csv),
                    "negative_csv": str(args.negative_csv),
                    "sequence_column": args.sequence_column,
                    "target_column": args.target_column,
                    "negative_sequence_column": args.negative_sequence_column,
                    "seq_len": args.seq_len,
                    "pad_side": args.pad_side,
                    "seed": args.seed,
                    "best_metric": "standard_unweighted_val_mse",
                },
                checkpoint_path,
            )

    if run is not None:
        wandb.finish()

    print(f"Saved best cap-loss checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
