#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

import dataloader_gosai
import diffusion_gosai_update as diffusion
import mttrans
import oracle_new
import oracle_utr
from utils import set_seed

BASES = ("A", "C", "G", "T")
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
DEFAULT_LENGTH_SOURCE_CSV = Path(
    "/home/xli263/xli/utr_design/DRAKES/drakes_rna/data_te/RP_PC3_te_train.csv"
)


class UTROracleWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, soft_input: bool = False):
        del soft_input
        device = next(self.model.parameters()).device
        preds = self.model(x.to(device))
        if preds.dim() == 1:
            preds = preds.unsqueeze(-1)
        return preds


class _FramePoolAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, oracle):
        y_np, meta = oracle._forward_numpy(x.detach())
        y = torch.from_numpy(y_np).to(device=x.device, dtype=x.dtype)
        ctx.oracle = oracle
        ctx.meta = meta
        ctx.save_for_backward(x.detach())
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_np = ctx.oracle._backward_numpy(
            x.detach(),
            grad_output.detach(),
            ctx.meta,
        )
        grad_x = torch.from_numpy(grad_np).to(device=x.device, dtype=x.dtype)
        return grad_x, None


class FramePoolOracleWrapper(torch.nn.Module):
    def __init__(self, model_path: str, max_len: int = 128):
        super().__init__()
        self.model_path = model_path
        self.max_len = int(max_len)

        import sys
        import tensorflow as tf

        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from UTRGAN.src.mrl_te_optimization.framepool import load_framepool

        self.tf = tf
        self.tf_model = load_framepool(model_path)
        self.tf_model.trainable = False

    def _prepare_for_framepool(self, x: torch.Tensor):
        x_l4 = x.transpose(1, 2).contiguous()
        x_l4 = x_l4.clamp_min(1e-8)
        x_l4 = x_l4 / x_l4.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        batch_size, seq_len, channels = x_l4.shape
        if channels != 4:
            raise ValueError(f"FramePool expects 4 channels, got shape {tuple(x.shape)}")

        pad_left = 0
        if self.max_len > 0 and seq_len < self.max_len:
            pad_left = self.max_len - seq_len
            pad = torch.zeros(batch_size, pad_left, 4, device=x_l4.device, dtype=x_l4.dtype)
            x_l4 = torch.cat([pad, x_l4], dim=1)
        return x_l4, pad_left

    def _forward_numpy(self, x: torch.Tensor):
        x_l4, pad_left = self._prepare_for_framepool(x)
        x_np = x_l4.detach().cpu().numpy().astype(np.float32, copy=False)
        y_np = self.tf_model(x_np, training=False).numpy().astype(np.float32, copy=False)
        meta = {"pad_left": int(pad_left)}
        return y_np, meta

    def _backward_numpy(self, x: torch.Tensor, grad_output: torch.Tensor, meta):
        x_l4, pad_left = self._prepare_for_framepool(x)
        x_np = x_l4.detach().cpu().numpy().astype(np.float32, copy=False)
        go_np = grad_output.detach().cpu().numpy().astype(np.float32, copy=False)

        xt = self.tf.convert_to_tensor(x_np, dtype=self.tf.float32)
        got = self.tf.convert_to_tensor(go_np, dtype=self.tf.float32)
        with self.tf.GradientTape() as tape:
            tape.watch(xt)
            y = self.tf_model(xt, training=False)
        gx = tape.gradient(y, xt, output_gradients=got).numpy().astype(np.float32, copy=False)
        if pad_left > 0:
            gx = gx[:, pad_left:, :]
        gx = np.transpose(gx, (0, 2, 1))
        return gx

    def forward(self, x, soft_input: bool = False):
        del soft_input
        return _FramePoolAutogradFn.apply(x, self)


def load_config():
    root = Path("/home/xli263/xli/utr_design/DRAKES/drakes_rna/configs_gosai")
    base = OmegaConf.load(root / "config_gosai_pretrain.yaml")
    model_cfg = OmegaConf.load(root / "model" / "dnaconv.yaml")
    noise_cfg = OmegaConf.load(root / "noise" / "loglinear.yaml")
    strategy_cfg = OmegaConf.load(root / "strategy" / "ddp.yaml")
    lr_cfg = OmegaConf.load(root / "lr_scheduler" / "constant_warmup.yaml")
    return OmegaConf.merge(
        base,
        OmegaConf.create({"model": model_cfg, "noise": noise_cfg, "strategy": strategy_cfg, "lr_scheduler": lr_cfg}),
    )


def build_tokenizer(cfg):
    tokenizer_type = cfg.data.get("tokenizer_type", "csv_motif")
    if tokenizer_type == "csv_motif":
        tokenizer = dataloader_gosai.MotifAwareTokenizer(
            vocab_json_path=cfg.data.motif_vocab_path,
            pad_token=cfg.data.get("pad_token", "N"),
            eos_token=cfg.data.get("eos_token", "EOS"),
            base_tokens=cfg.data.get("motif_base_tokens", ("A", "C", "G", "T")),
            max_length=cfg.model.length,
            trim_to=cfg.data.get("motif_trim_len"),
        )
        pad_id = tokenizer.pad_token_id
        eos_id = getattr(tokenizer, "eos_token_id", None)
    else:
        tokenizer, pad_id, eos_id = dataloader_gosai.build_simple_tokenizer(
            cfg.data.tokenizer_vocab_path,
            pad_token=cfg.data.get("pad_token", "N"),
            eos_token=cfg.data.get("eos_token", None),
            unk_token=cfg.data.get("unk_token", None),
        )

    return tokenizer, pad_id, eos_id


def load_model(checkpoint: Path, cfg, device: torch.device):
    model = diffusion.Diffusion(config=cfg)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict strict=False missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    return model


def load_reward_model(args, device: torch.device):
    if args.reward_type in ("utrlm", "utrlm_te"):
        reward_model = UTROracleWrapper(
            oracle_utr.get_utr_oracle(
                map_location=str(device),
                oracle_ckpt_path=args.oracle_ckpt_path,
            )
        )
    elif args.reward_type == "rnafm":
        reward_model = oracle_new.get_rnafm_oracle(
            predictor_checkpoint=args.rnafm_predictor_checkpoint,
            backbone_path=args.rnafm_backbone_path,
            fm_root=args.rnafm_fm_root,
            device=str(device),
            seq_trim_len=args.rnafm_seq_trim_len,
        )
    elif args.reward_type == "framepool":
        reward_model = FramePoolOracleWrapper(
            model_path=args.framepool_model_path,
            max_len=args.framepool_max_len,
        )
    elif args.reward_type == "mttrans":
        reward_model = mttrans.get_mttrans_oracle(
            checkpoint_path=args.mttrans_checkpoint,
            map_location=device,
            input_length=args.mttrans_input_length,
        )
    else:
        raise ValueError(f"Unknown reward_type {args.reward_type}")

    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
    return reward_model


def trim_token_ids(token_ids: Sequence[int], pad_id: int, eos_id: Optional[int]) -> List[int]:
    out = []
    for token_id in token_ids:
        token_id = int(token_id)
        if eos_id is not None and token_id == int(eos_id):
            break
        if token_id == int(pad_id):
            break
        out.append(token_id)
    return out


def decode_sequences(token_batch: torch.Tensor, tokenizer, pad_id: int, eos_id: Optional[int]) -> Tuple[List[str], List[List[int]]]:
    seqs = []
    trimmed_ids = []
    for token_ids in token_batch.detach().cpu().tolist():
        trimmed = trim_token_ids(token_ids, pad_id=pad_id, eos_id=eos_id)
        seq = tokenizer.decode(trimmed).replace(" ", "")
        seqs.append(seq)
        trimmed_ids.append(trimmed)
    return seqs, trimmed_ids


def sequences_to_base_tensor(seqs: Sequence[str], device: torch.device, input_length: Optional[int] = None):
    max_len = max((len(seq) for seq in seqs), default=0)
    tensor_len = int(input_length) if input_length is not None else max_len
    x = torch.zeros(len(seqs), 4, tensor_len, device=device, dtype=torch.float32)
    for batch_idx, seq in enumerate(seqs):
        seq = str(seq).upper()
        if input_length is not None and len(seq) > tensor_len:
            seq = seq[-tensor_len:]
        start = tensor_len - len(seq) if input_length is not None else 0
        for pos, base in enumerate(seq):
            base_idx = BASE_TO_IDX.get(base)
            if base_idx is not None:
                x[batch_idx, base_idx, start + pos] = 1.0
    return x


def score_sequences(seqs: Sequence[str], reward_model, device: torch.device) -> List[float]:
    if not seqs:
        return []
    x = sequences_to_base_tensor(
        seqs,
        device=device,
        input_length=getattr(reward_model, "input_length", None),
    )
    with torch.no_grad():
        preds = reward_model(x, soft_input=False)
    return preds[..., 0].detach().cpu().view(-1).tolist()


def parse_target_length(args, eos_id: Optional[int]):
    if args.target_nt_length is not None and args.target_token_length is not None:
        raise ValueError("Specify only one of --target-nt-length or --target-token-length.")
    if args.target_nt_length is not None:
        return int(args.target_nt_length)
    if args.target_token_length is not None:
        token_length = int(args.target_token_length)
        return max(0, token_length - 1) if eos_id is not None else token_length
    return None


def load_target_token_lengths(
    csv_path: Path,
    seq_column: str,
    tokenizer,
    eos_id: Optional[int],
    max_model_length: int,
) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    if seq_column not in df.columns:
        raise ValueError(f"Missing sequence column '{seq_column}' in {csv_path}")

    target_lengths = []
    max_model_length = int(max_model_length)
    for seq in df[seq_column].astype(str):
        enc = tokenizer.encode(seq)
        token_length = int(sum(enc.attention_mask))
        # diffusion_gosai_update target_length means the index at which EOS/PAD
        # is forced. If token_length includes EOS, pass the decoded-token count.
        target_length = token_length - 1 if eos_id is not None else token_length
        target_length = max(0, min(target_length, max_model_length))
        if target_length > 0:
            target_lengths.append(target_length)

    if not target_lengths:
        raise ValueError(f"No positive target token lengths found in {csv_path}")
    return torch.tensor(target_lengths, dtype=torch.long)


def save_csv(output_path: Path, rows: List[dict]):
    if not rows:
        print("No rows to save; skipping CSV write.")
        return
    fieldnames = ["seq", "reward", "target_length", "token_length", "nt_length", "token_ids"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser("Generate sequences with classifier guidance over a pretrained diffusion model.")
    parser.add_argument("--checkpoint", type=Path, default="/home/xli263/xli/utr_design/DRAKES/drakes_rna/mdlm/pretrained_utr_ckpt/pretrained_4_base_rp_pc3_te.ckpt", help="Path to the pretrained diffusion checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=None, help="Override cfg.sampling.steps.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-nt-length", type=int, default=None,
                        help="Target decoded nucleotide length. If EOS is in the vocab, EOS is placed immediately after this many bases.")
    parser.add_argument("--target-token-length", type=int, default=None,
                        help="Target token length including EOS when EOS is in the vocab.")
    parser.add_argument("--sample-target-lengths", action=argparse.BooleanOptionalAction, default=True,
                        help="Sample target token lengths from --length-source-csv when no fixed target length is set.")
    parser.add_argument("--length-source-csv", type=Path, default=DEFAULT_LENGTH_SOURCE_CSV)
    parser.add_argument("--length-seq-column", type=str, default="utr")
    parser.add_argument("--reward-type", type=str, choices=["utrlm", "utrlm_te", "rnafm", "framepool", "mttrans"], default="utrlm")
    parser.add_argument("--oracle-ckpt-path", type=str,
                        default="/home/xli263/xli/utr_design/DRAKES/drakes_rna/experiment/single_run_vanilla_mrl_50nt/vanilla_best.ckpt")
    parser.add_argument("--rnafm-predictor-checkpoint", type=str, default=None)
    parser.add_argument("--rnafm-backbone-path", type=str, default=None)
    parser.add_argument("--rnafm-fm-root", type=str, default=None)
    parser.add_argument("--rnafm-seq-trim-len", type=int, default=50)
    parser.add_argument("--framepool-model-path", type=str, default=None)
    parser.add_argument("--framepool-max-len", type=int, default=128)
    parser.add_argument("--mttrans-checkpoint", type=str,
                        default=str(mttrans.DEFAULT_MTTRANS_CKPT))
    parser.add_argument("--mttrans-input-length", type=int, default=105,
                        help="MTTrans fixed input length. The mttrans.py wrapper left-pads shorter inputs and keeps the last N nt for longer inputs.")
    args = parser.parse_args()

    set_seed(args.seed, use_cuda=torch.cuda.is_available())
    cfg = load_config()
    tokenizer, pad_id, eos_id = build_tokenizer(cfg)
    target_length = parse_target_length(args, eos_id=eos_id)
    target_length_values = None
    if target_length is None and args.sample_target_lengths:
        target_length_values = load_target_token_lengths(
            args.length_source_csv,
            args.length_seq_column,
            tokenizer=tokenizer,
            eos_id=eos_id,
            max_model_length=cfg.model.length,
        )
        print(
            f"Loaded {len(target_length_values)} target token lengths from {args.length_source_csv}; "
            f"min={int(target_length_values.min())}, max={int(target_length_values.max())}, "
            f"mean={float(target_length_values.float().mean()):.2f}"
        )

    device = torch.device(args.device)
    model = load_model(args.checkpoint, cfg, device=device)
    reward_model = load_reward_model(args, device=device)

    rows = []
    remaining = int(args.num_samples)
    while remaining > 0:
        batch_size = min(int(args.batch_size), remaining)
        if target_length is not None:
            batch_target_length = int(target_length)
            row_target_lengths = [batch_target_length] * batch_size
        elif target_length_values is not None:
            idx = torch.randint(0, len(target_length_values), (batch_size,))
            row_target_lengths = target_length_values[idx].tolist()
            batch_target_length = row_target_lengths
        else:
            batch_target_length = None
            row_target_lengths = [None] * batch_size
        samples = model.controlled_sample_CG(
            reward_model=reward_model,
            guidance_scale=float(args.guidance_scale),
            num_steps=args.num_steps,
            eval_sp_size=batch_size,
            target_length=batch_target_length,
        )
        seqs, trimmed_ids = decode_sequences(samples, tokenizer, pad_id=pad_id, eos_id=eos_id)
        rewards = score_sequences(seqs, reward_model, device=device)

        for seq, reward, token_ids, row_target_length in zip(seqs, rewards, trimmed_ids, row_target_lengths):
            rows.append({
                "seq": seq,
                "reward": float(reward),
                "target_length": row_target_length,
                "token_length": len(token_ids) + (1 if eos_id is not None and row_target_length is not None else 0),
                "nt_length": len(seq),
                "token_ids": " ".join(str(token_id) for token_id in token_ids),
            })
        remaining -= batch_size
        print(f"Generated {len(rows)}/{args.num_samples} sequences")

    save_csv(args.output, rows)
    print(f"Wrote {len(rows)} sequences to {args.output}")


if __name__ == "__main__":
    main()
