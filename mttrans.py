from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


# DEFAULT_MTTRANS_CKPT = Path(
#     "/home/xli263/xli/utr_design/UTRGAN/src/mrl_te_optimization/script/checkpoint/RL_hard_share_MTL/3R/schedule_MTL-model_best_cv1.pth"
# )

DEFAULT_MTTRANS_CKPT = Path(
    "/home/xli263/xli/utr_design/UTRGAN/src/mrl_te_optimization/script/checkpoint_cap_rp_pc3/RL_hard_share_MTL/3R_cap_l2_pr058_top3_bs64/schedule_MTL_cap_l2_pr058_top3_bs64-model_best_cv1.pth"
)
UTRGAN_MTTRANS_SRC = Path("/home/xli263/xli/utr_design/UTRGAN/src/mrl_te_optimization")
UTRGAN_MTTRANS_MODELS = UTRGAN_MTTRANS_SRC / "models"


def _load_serialized_mttrans(checkpoint_path: str | Path, map_location: str | torch.device):
    checkpoint_path = str(checkpoint_path)

    old_path = list(sys.path)
    old_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "models" or name.startswith("models.")
    }
    for name in list(old_modules):
        del sys.modules[name]

    try:
        sys.path.insert(0, str(UTRGAN_MTTRANS_SRC))
        namespace_pkg = types.ModuleType("models")
        namespace_pkg.__path__ = [str(UTRGAN_MTTRANS_MODELS)]
        namespace_pkg.__package__ = "models"
        sys.modules["models"] = namespace_pkg
        importlib.invalidate_caches()
        importlib.import_module("models.Modules")
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    finally:
        sys.path[:] = old_path
        for name in [n for n in list(sys.modules) if n == "models" or n.startswith("models.")]:
            del sys.modules[name]
        sys.modules.update(old_modules)

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Unexpected MTTrans checkpoint format at {checkpoint_path}")
    return ckpt["state_dict"]


class MTTransOracle(torch.nn.Module):
    """Wrapper around the serialized UTRGAN MTTrans TE model.

    The original UTRGAN TE train/eval path right-aligns valid sequence near the
    start codon by left-padding shorter inputs and cropping longer inputs to the
    last `pad_to` nt. This wrapper reproduces that preprocessing for [B, 4, L]
    inputs before passing them to the serialized TE checkpoint.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_length: int = 105,
        task: str | None = None,
    ):
        super().__init__()
        self.model = model
        self.input_length = int(input_length)
        self.task = task
        if self.task is not None:
            if hasattr(self.model, "tower") and self.task not in self.model.tower:
                raise ValueError(
                    f"Task '{self.task}' is not available in MTTrans checkpoint. "
                    f"Available tasks: {list(self.model.tower.keys())}"
                )
            self.model.task = self.task

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 4:
            raise ValueError(f"MTTrans expects [B, 4, L], got shape {tuple(x.shape)}")

        x = x.float()
        seq_len = x.shape[-1]
        if seq_len > self.input_length:
            x = x[:, :, -self.input_length :]
        elif seq_len < self.input_length:
            x = F.pad(x, (self.input_length - seq_len, 0), mode="constant", value=0.0)
        return x

    def forward(self, x: torch.Tensor, soft_input: bool = False) -> torch.Tensor:
        del soft_input
        device = next(self.model.parameters()).device
        model_input = self._prepare_input(x).to(device)
        if self.task is not None:
            self.model.task = self.task
        was_training = self.model.training
        self.model.eval()
        if model_input.requires_grad:
            # cuDNN GRU backward requires training-mode forward; disable cuDNN
            # so we can preserve eval-mode reward values while keeping input gradients.
            with torch.backends.cudnn.flags(enabled=False):
                preds = self.model(model_input)
        else:
            preds = self.model(model_input)
        if was_training:
            self.model.train()
        if preds.dim() == 1:
            preds = preds.unsqueeze(-1)
        return preds


def get_mttrans_oracle(
    checkpoint_path: str | Path = DEFAULT_MTTRANS_CKPT,
    map_location: str | torch.device = "cpu",
    input_length: int = 105,
    task: str | None = None,
) -> MTTransOracle:
    model = _load_serialized_mttrans(checkpoint_path, map_location=map_location)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return MTTransOracle(model=model, input_length=input_length, task=task)
