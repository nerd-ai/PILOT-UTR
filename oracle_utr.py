import glob
import os

import torch
from grelu.lightning import LightningModel

# Folder with 5 fold checkpoints produced by train_oracle.py
CKPT_DIR = "/home/xli263/xli/utr_design/DRAKES/drakes_rna/data_and_model/utr_oracle"
CKPT_GLOB = "reward_utr_rl_fold*.ckpt"


def _load_oracle_model(ckpt_path: str, map_location: str):
    checkpoint = torch.load(ckpt_path, map_location=map_location)

    # Lightning checkpoints include framework metadata and can be restored directly.
    if isinstance(checkpoint, dict) and "pytorch-lightning_version" in checkpoint:
        model = LightningModel.load_from_checkpoint(ckpt_path, map_location=map_location)
    # Hybrid checkpoints from train_oracle.py are plain torch.save() dicts.
    elif isinstance(checkpoint, dict) and "model_params" in checkpoint and "train_params" in checkpoint:
        train_params = dict(checkpoint["train_params"])
        train_params["logger"] = None
        train_params["checkpoint"] = False
        model = LightningModel(
            model_params=checkpoint["model_params"],
            train_params=train_params,
        )
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model = model.to(map_location)
    else:
        raise ValueError(f"Unsupported oracle checkpoint format: {ckpt_path}")

    model.train_params["logger"] = None
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class SoftMinEnsembleUTROracle(torch.nn.Module):
    """UTR oracle ensemble with uncertainty penalty: mean - gamma * variance."""

    def __init__(self, models, gamma: float = 1.0):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.gamma = float(gamma)

    def forward(self, x):
        # Collect per-model scalar predictions as [K, B].
        preds = []
        for model in self.models:
            y = model(x)
            # Normalize common scalar output shapes to [B].
            # Supported: [B], [B,1], [B,1,1].
            if y.dim() == 1:
                pass
            elif y.dim() == 2 and y.shape[-1] == 1:
                y = y[:, 0]
            elif y.dim() == 3 and y.shape[-1] == 1 and y.shape[-2] == 1:
                y = y[:, 0, 0]
            else:
                raise ValueError(f"Unexpected oracle output shape: {tuple(y.shape)}")
            preds.append(y)
        stacked = torch.stack(preds, dim=0)
        mean_score = stacked.mean(dim=0)
        # Use population variance (unbiased=False) for stable gradients with small K.
        var_score = stacked.var(dim=0, unbiased=False)
        robust_score = mean_score - self.gamma * var_score
        return robust_score.unsqueeze(-1)


class MeanEnsembleUTROracle(torch.nn.Module):
    """UTR oracle ensemble that returns the simple mean prediction."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        preds = []
        for model in self.models:
            y = model(x)
            if y.dim() == 1:
                pass
            elif y.dim() == 2 and y.shape[-1] == 1:
                y = y[:, 0]
            elif y.dim() == 3 and y.shape[-1] == 1 and y.shape[-2] == 1:
                y = y[:, 0, 0]
            else:
                raise ValueError(f"Unexpected oracle output shape: {tuple(y.shape)}")
            preds.append(y)
        mean_score = torch.stack(preds, dim=0).mean(dim=0)
        return mean_score.unsqueeze(-1)


def get_utr_oracle(
    map_location: str = "cuda",
    uncertainty_gamma: float = 1.0,
    oracle_ckpt_path: str = None,
    oracle_ckpt_paths = None,
):
    """
    Load oracle checkpoint(s) and return an uncertainty-penalized ensemble oracle.

    Args:
        map_location: Device mapping for loading checkpoints (e.g., "cuda" or "cpu").
        uncertainty_gamma: Penalty strength for ensemble variance.
            Score = mean(preds) - uncertainty_gamma * var(preds).
        oracle_ckpt_path: Optional path to a single checkpoint. If provided,
            only this checkpoint is loaded and used as the oracle.
        oracle_ckpt_paths: Optional explicit list of checkpoints to ensemble.
    """
    if oracle_ckpt_paths is not None:
        ckpt_paths = []
        for ckpt_path in oracle_ckpt_paths:
            resolved = ckpt_path
            if not os.path.isabs(resolved):
                resolved = os.path.join(CKPT_DIR, resolved)
            if not os.path.exists(resolved):
                raise FileNotFoundError(f"Oracle checkpoint not found: {resolved}")
            ckpt_paths.append(resolved)
    elif oracle_ckpt_path is not None:
        if not os.path.isabs(oracle_ckpt_path):
            oracle_ckpt_path = os.path.join(CKPT_DIR, oracle_ckpt_path)
        if not os.path.exists(oracle_ckpt_path):
            raise FileNotFoundError(f"Oracle checkpoint not found: {oracle_ckpt_path}")
        ckpt_paths = [oracle_ckpt_path]
    else:
        ckpt_paths = sorted(glob.glob(os.path.join(CKPT_DIR, CKPT_GLOB)))
        if len(ckpt_paths) != 5:
            raise FileNotFoundError(
                f"Expected 5 oracle checkpoints in {CKPT_DIR} matching {CKPT_GLOB}, "
                f"found {len(ckpt_paths)}."
            )

    models = []
    for ckpt_path in ckpt_paths:
        models.append(_load_oracle_model(ckpt_path, map_location))

    if oracle_ckpt_paths is not None:
        return MeanEnsembleUTROracle(models=models)
    return SoftMinEnsembleUTROracle(models=models, gamma=uncertainty_gamma)
