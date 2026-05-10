import argparse
import ast
import csv
import datetime
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
# PyTorch 2.6+ defaults weights_only=True which breaks legacy checkpoints
# containing omegaconf objects. Force weights_only=False for this script.
_orig_torch_load = torch.load
torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, 'weights_only': False})
import torch.nn.functional as F
import wandb
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

import diffusion_gosai_update
import dataloader_gosai
import oracle_new
import oracle_utr
from utils import set_seed, str2bool

import math
from collections import Counter
BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
DIFFUSION_BASE_ORDER = ('A', 'C', 'G', 'T')


class UTROracleWrapper(torch.nn.Module):
  """Adapter to run the UTR oracle checkpoint with the finetuning loop signature."""

  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, x, soft_input: bool = False):
    # `x` is expected to be [B, 4, L] (soft or hard one-hot).
    device = next(self.model.parameters()).device
    preds = self.model(x.to(device))
    # Keep a predictable trailing dimension for downstream `[..., 0]` access.
    if preds.dim() == 1:
      preds = preds.unsqueeze(-1)
    return preds


class _FramePoolAutogradFn(torch.autograd.Function):
  """Torch autograd bridge for TensorFlow FramePool inference."""

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
  """
  TensorFlow FramePool wrapped as a differentiable PyTorch reward model.
  Expects x as [B, 4, L] (soft/hard one-hot) and returns [B, 1].
  """

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
    # x: [B, 4, L] -> [B, L, 4], normalized over A/C/G/T.
    x_l4 = x.transpose(1, 2).contiguous()
    x_l4 = x_l4.clamp_min(1e-8)
    x_l4 = x_l4 / x_l4.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    B, L, C = x_l4.shape
    if C != 4:
      raise ValueError(f"FramePool expects 4 channels, got shape {tuple(x.shape)}")

    pad_left = 0
    if self.max_len > 0 and L < self.max_len:
      pad_left = self.max_len - L
      pad = torch.zeros(B, pad_left, 4, device=x_l4.device, dtype=x_l4.dtype)
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
    gx = tape.gradient(y, xt, output_gradients=got).numpy().astype(np.float32, copy=False)  # [B, Lp, 4]
    if pad_left > 0:
      gx = gx[:, pad_left:, :]
    gx = np.transpose(gx, (0, 2, 1))  # [B, 4, L]
    return gx

  def forward(self, x, soft_input: bool = False):
    # Keep signature compatible with existing reward models.
    del soft_input
    return _FramePoolAutogradFn.apply(x, self)


def _motif_sample_to_base(sample, model):
  """Map motif-level (ST) vectors to base-level (ST) vectors via the stencil."""
  if not hasattr(model, "motif2base_stencil"):
    raise AttributeError("motif2base_stencil buffer not found on model.")
  if not hasattr(model, "motif_lengths"):
    raise AttributeError("motif_lengths buffer not found on model.")
  if sample.dim() != 3:
    raise ValueError(f"Expected motif sample of shape [B, T, V], got {tuple(sample.shape)}")

  B, T, _ = sample.shape
  stencil = model.motif2base_stencil.to(device=sample.device, dtype=torch.float32)
  motif_lengths = model.motif_lengths.to(device=sample.device)
  eos_id = getattr(model, "eos_token_id", None)
  pad_id = getattr(model, "pad_token_id", None)

  motif_ids_hard = sample.argmax(dim=-1)
  start = torch.zeros(B, T, dtype=torch.long, device=sample.device)
  base_len = torch.zeros(B, dtype=torch.long, device=sample.device)
  valid_T = torch.zeros(B, dtype=torch.long, device=sample.device)
  max_base_len = 0

  for b in range(B):
    cur = 0
    t_eff = 0
    for t in range(T):
      v = int(motif_ids_hard[b, t].item())
      if (eos_id is not None and v == eos_id) or (pad_id is not None and v == pad_id):
        break
      start[b, t] = cur
      L_t = int(motif_lengths[v].item())
      cur += L_t
      t_eff = t + 1
    base_len[b] = cur
    valid_T[b] = t_eff
    if cur > max_base_len:
      max_base_len = cur

  base_soft = torch.zeros(B, max_base_len, 4, device=sample.device, dtype=torch.float32)

  # Use the (ST) motif sample directly; linear projection keeps gradients.
  probs = sample
  for b in range(B):
    Teff = int(valid_T[b].item())
    for t in range(Teff):
      v_hard = int(motif_ids_hard[b, t].item())
      L_t = int(motif_lengths[v_hard].item())
      if L_t == 0:
        continue
      s = int(start[b, t].item())
      local_soft = torch.einsum('v, vkc -> kc', probs[b, t], stencil)
      base_soft[b, s:s + L_t] += local_soft[:L_t]

  # if max_base_len > 0:
  #   row_sums = base_soft.sum(dim=-1)
  #   if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4):
  #     raise AssertionError("Base projection rows are not one-hot (sum != 1).")
  #   for b in range(B):
  #     end = int(base_len[b].item())
  #     if end > 0 and (row_sums[b, :end] == 0).any():
  #       raise AssertionError("Found [0,0,0,0] row before end of sequence.")
  return base_soft


def _token_probs_to_base_probs(token_probs, model, gradient_type):
  """
  Convert token-space probabilities [B, T, V] to base-space probs [B, L, 4].
  """
  if gradient_type == "motif_soft" and getattr(model, "vocab_size", 0) > 7:
    hard_ids = token_probs.argmax(dim=-1)
    hard_onehot = torch.nn.functional.one_hot(
      hard_ids, num_classes=token_probs.shape[-1]).to(torch.float32)
    base_probs = _motif_sample_to_base(hard_onehot, model)
  else:
    if token_probs.shape[-1] < 4:
      raise ValueError(f"Expected at least 4 base channels, got shape {tuple(token_probs.shape)}")
    base_probs = token_probs[:, :, :4]

  base_probs = base_probs.clamp_min(1e-12)
  base_probs = base_probs / base_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
  return base_probs


def _decode_base_sequence(base_probs_row, valid_len=None):
  """
  Decode one [L,4] probability row to A/C/G/T string by argmax.
  """
  if valid_len is None:
    valid_len = base_probs_row.shape[0]
  valid_len = int(max(0, min(valid_len, base_probs_row.shape[0])))
  idx = base_probs_row[:valid_len].argmax(dim=-1).tolist()
  return ''.join(DIFFUSION_BASE_ORDER[i] for i in idx)


def _base_comp_str(base_probs_row, valid_len=None):
  if valid_len is None:
    valid_len = base_probs_row.shape[0]
  valid_len = int(max(0, min(valid_len, base_probs_row.shape[0])))
  if valid_len == 0:
    return "A:0.000 C:0.000 G:0.000 T:0.000"
  mean_comp = base_probs_row[:valid_len].mean(dim=0)
  return (
    f"A:{mean_comp[0].item():.3f} "
    f"C:{mean_comp[1].item():.3f} "
    f"G:{mean_comp[2].item():.3f} "
    f"T:{mean_comp[3].item():.3f}"
  )


def _base_comp_str_on_masked(last_x_row, base_probs_row, mask_idx, valid_len=None):
  """
  Average A/C/G/T probabilities over masked xt positions only.
  last_x_row: [T, V], base_probs_row: [L, 4]
  """
  if valid_len is None:
    valid_len = min(last_x_row.shape[0], base_probs_row.shape[0])
  valid_len = int(max(0, min(valid_len, last_x_row.shape[0], base_probs_row.shape[0])))
  if valid_len == 0:
    return "A:0.000 C:0.000 G:0.000 T:0.000"
  xt_ids = last_x_row[:valid_len].argmax(dim=-1)
  masked = (xt_ids == int(mask_idx))
  if masked.sum().item() == 0:
    return "A:0.000 C:0.000 G:0.000 T:0.000"
  mean_comp = base_probs_row[:valid_len][masked].mean(dim=0)
  return (
    f"A:{mean_comp[0].item():.3f} "
    f"C:{mean_comp[1].item():.3f} "
    f"G:{mean_comp[2].item():.3f} "
    f"T:{mean_comp[3].item():.3f}"
  )


def _decode_xt_argmax(last_x_row, model, valid_len=None):
  """
  Decode one [T, V] last_x row via argmax, preserving explicit mask token display.
  """
  if valid_len is None:
    valid_len = last_x_row.shape[0]
  valid_len = int(max(0, min(valid_len, last_x_row.shape[0])))
  ids = last_x_row[:valid_len].argmax(dim=-1).tolist()

  id2token = getattr(model, "id2token", None)
  mask_idx = int(getattr(model, "mask_index", -1))
  toks = []
  for i in ids:
    if i == mask_idx:
      toks.append("[MASK]")
    elif id2token is not None and 0 <= i < len(id2token):
      toks.append(str(id2token[i]))
    elif 0 <= i < len(DIFFUSION_BASE_ORDER):
      toks.append(DIFFUSION_BASE_ORDER[i])
    else:
      toks.append(f"<{i}>")
  return ' '.join(toks)


def _masked_fill_a_stats(last_x_row, mask_idx, decoded_seq):
  """
  Compute how many masked positions in xt are decoded as 'A' in decoded_seq.
  last_x_row: [T, V], decoded_seq: plain A/C/G/T string with length <= T.
  """
  ids = last_x_row.argmax(dim=-1)
  valid_len = min(len(decoded_seq), ids.shape[0])
  mask_pos = [i for i in range(valid_len) if int(ids[i].item()) == mask_idx]
  if not mask_pos:
    return 0, 0, 0.0
  a_cnt = sum(1 for i in mask_pos if decoded_seq[i] == 'A')
  total = len(mask_pos)
  return a_cnt, total, a_cnt / max(1, total)


def _masked_avg_vocab_probs(probs_row, last_x_row, mask_idx, token_valid_len=None):
  """
  Average token-vocab probabilities over masked token positions only.
  probs_row: [T, V], last_x_row: [T, V]
  """
  if probs_row.dim() != 2 or last_x_row.dim() != 2:
    return None, 0
  T = probs_row.shape[0]
  if token_valid_len is not None:
    T = min(T, int(max(0, token_valid_len)))
  T = min(T, last_x_row.shape[0])
  if T <= 0:
    return None, 0
  probs = probs_row[:T].clamp_min(1e-12)
  probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
  ids = last_x_row[:T].argmax(dim=-1)
  masked = (ids == int(mask_idx))
  masked_cnt = int(masked.sum().item())
  if masked_cnt <= 0:
    return torch.zeros(probs.shape[-1], device=probs.device, dtype=probs.dtype), 0
  avg = probs[masked].mean(dim=0)
  avg = avg / avg.sum().clamp_min(1e-12)
  return avg, masked_cnt


def _format_vocab_distribution(prob_vec, model):
  """Format full token-vocab distribution as token:prob comma-separated pairs."""
  if prob_vec is None:
    return ""
  id2token = getattr(model, "id2token", None)
  pairs = []
  for tok_id in range(int(prob_vec.shape[0])):
    tok_prob = float(prob_vec[tok_id].item())
    if id2token is not None and 0 <= tok_id < len(id2token):
      tok_name = str(id2token[tok_id])
    else:
      tok_name = str(tok_id)
    pairs.append(f"{tok_name}:{tok_prob:.6f}")
  return ",".join(pairs)


def _motif_mask_to_base_mask(last_x_row, mask_idx, model, base_len, token_valid_len=None):
  """
  Map token-space mask positions to base-space positions for motif tokenization.
  last_x_row: [T, V], returns bool mask [base_len].
  """
  base_len = int(max(0, base_len))
  out = torch.zeros(base_len, dtype=torch.bool, device=last_x_row.device)
  if base_len == 0:
    return out
  ids = last_x_row.argmax(dim=-1)
  T = int(ids.shape[0] if token_valid_len is None else max(0, min(int(token_valid_len), int(ids.shape[0]))))
  motif_lengths = getattr(model, "motif_lengths", None)
  eos_id = getattr(model, "eos_token_id", None)
  pad_id = getattr(model, "pad_token_id", None)
  cur = 0
  for t in range(T):
    tok_id = int(ids[t].item())
    if (eos_id is not None and tok_id == int(eos_id)) or (pad_id is not None and tok_id == int(pad_id)):
      break
    L_t = 1
    if motif_lengths is not None and 0 <= tok_id < int(motif_lengths.shape[0]):
      L_t = int(motif_lengths[tok_id].item())
      # Keep masked-span accounting robust even if special tokens have 0 length.
      if L_t <= 0:
        L_t = 1
    nxt = min(base_len, cur + max(1, L_t))
    if tok_id == int(mask_idx):
      out[cur:nxt] = True
    cur = nxt
    if cur >= base_len:
      break
  return out


def _masked_stats_motif_aware(last_x_row, base_probs_row, decoded_seq, mask_idx, model, token_valid_len=None):
  """
  Compute masked-base composition and masked->A fill stats with motif-aware token->base alignment.
  """
  valid_base_len = int(min(base_probs_row.shape[0], len(decoded_seq)))
  if valid_base_len <= 0:
    return "A:0.000 C:0.000 G:0.000 T:0.000", 0, 0, 0.0
  base_mask = _motif_mask_to_base_mask(
    last_x_row, mask_idx, model, base_len=valid_base_len, token_valid_len=token_valid_len)
  masked_cnt = int(base_mask.sum().item())
  if masked_cnt <= 0:
    return "A:0.000 C:0.000 G:0.000 T:0.000", 0, 0, 0.0
  mean_comp = base_probs_row[:valid_base_len][base_mask].mean(dim=0)
  comp_str = (
    f"A:{mean_comp[0].item():.3f} "
    f"C:{mean_comp[1].item():.3f} "
    f"G:{mean_comp[2].item():.3f} "
    f"T:{mean_comp[3].item():.3f}"
  )
  a_cnt = sum(1 for i in range(valid_base_len) if bool(base_mask[i].item()) and decoded_seq[i] == 'A')
  return comp_str, a_cnt, masked_cnt, a_cnt / max(1, masked_cnt)


def _sample_to_base_probs(sample, model, gradient_type):
  """Convert sample output to base probs [B, L, 4] for sequence-level metrics."""
  if gradient_type == "motif_soft" and getattr(model, "vocab_size", 0) > 7:
    probs = sample.clamp_min(1e-12)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return _token_probs_to_base_probs(probs, model, gradient_type=gradient_type)
  if sample.shape[-1] < 4:
    raise ValueError(f"Expected at least 4 channels in sample, got {tuple(sample.shape)}")
  base_probs = sample[:, :, :4].clamp_min(1e-12)
  return base_probs / base_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _get_valid_lens(target_length, batch_size, max_len):
  if target_length is None:
    return [int(max_len)] * int(batch_size)
  if isinstance(target_length, torch.Tensor):
    vals = target_length.detach().cpu().tolist()
  else:
    vals = list(target_length)
  out = []
  for v in vals[:batch_size]:
    out.append(int(max(0, min(int(v), int(max_len)))))
  if len(out) < batch_size:
    out.extend([int(max_len)] * (batch_size - len(out)))
  return out


def _hamming_norm(s1: str, s2: str) -> float:
  n = min(len(s1), len(s2))
  if n == 0:
    return 0.0
  return sum(1 for i in range(n) if s1[i] != s2[i]) / float(n)


def _base_comp_vec(seq: str) -> torch.Tensor:
  counts = torch.zeros(4, dtype=torch.float32)
  if not seq:
    return counts
  for ch in seq:
    if ch in BASE_TO_IDX:
      counts[BASE_TO_IDX[ch]] += 1.0
  denom = counts.sum().clamp_min(1.0)
  return counts / denom


def _kmer_dist_l1(s1: str, s2: str, k: int = 3) -> float:
  if k <= 0:
    return 0.0
  if len(s1) < k and len(s2) < k:
    return 0.0
  c1 = Counter(s1[i:i+k] for i in range(max(0, len(s1) - k + 1)))
  c2 = Counter(s2[i:i+k] for i in range(max(0, len(s2) - k + 1)))
  keys = set(c1.keys()) | set(c2.keys())
  if not keys:
    return 0.0
  n1 = float(sum(c1.values()))
  n2 = float(sum(c2.values()))
  l1 = 0.0
  for kk in keys:
    p1 = (c1.get(kk, 0) / n1) if n1 > 0 else 0.0
    p2 = (c2.get(kk, 0) / n2) if n2 > 0 else 0.0
    l1 += abs(p1 - p2)
  return float(l1)


def _pearson_corr(x_vals, y_vals) -> float:
  if len(x_vals) != len(y_vals) or len(x_vals) == 0:
    return 0.0
  x_mean = sum(x_vals) / float(len(x_vals))
  y_mean = sum(y_vals) / float(len(y_vals))
  num = 0.0
  den_x = 0.0
  den_y = 0.0
  for x, y in zip(x_vals, y_vals):
    dx = float(x) - x_mean
    dy = float(y) - y_mean
    num += dx * dy
    den_x += dx * dx
    den_y += dy * dy
  if den_x <= 0.0 or den_y <= 0.0:
    return 0.0
  return float(num / math.sqrt(den_x * den_y))


def _kmer_freq_vector(seqs, k: int = 3):
  if k <= 0:
    raise ValueError(f"k must be positive, got {k}")
  kmers = [''.join(chars) for chars in itertools.product(DIFFUSION_BASE_ORDER, repeat=k)]
  counts = Counter()
  total = 0
  for seq in seqs:
    seq = str(seq).upper()
    if len(seq) < k:
      continue
    for i in range(len(seq) - k + 1):
      kmer = seq[i:i + k]
      if all(ch in BASE_TO_IDX for ch in kmer):
        counts[kmer] += 1
        total += 1
  if total <= 0:
    return kmers, [0.0 for _ in kmers]
  return kmers, [counts[km] / float(total) for km in kmers]


def _load_csv_sequences(csv_path: str, seq_col: str = "utr"):
  seqs = []
  with open(csv_path, newline='') as handle:
    reader = csv.DictReader(handle)
    if seq_col not in (reader.fieldnames or []):
      raise ValueError(f'Sequence column "{seq_col}" not found in {csv_path}. Columns: {reader.fieldnames}')
    for row in reader:
      seq = str(row[seq_col]).strip().upper()
      if seq:
        seqs.append(seq)
  return seqs


def _at_fraction(seqs) -> float:
  a_count = 0
  t_count = 0
  total_count = 0
  for seq in seqs:
    seq = str(seq).upper()
    a_count += seq.count('A')
    t_count += seq.count('T')
    total_count += sum(seq.count(base) for base in ('A', 'C', 'G', 'T'))
  if total_count <= 0:
    return 0.0
  return float((a_count + t_count) / float(total_count))


def _base_ratios(seqs):
  counts = Counter()
  total_count = 0
  for seq in seqs:
    seq = str(seq).upper()
    for base in DIFFUSION_BASE_ORDER:
      count = seq.count(base)
      counts[base] += count
      total_count += count
  if total_count <= 0:
    return {base: 0.0 for base in DIFFUSION_BASE_ORDER}
  return {base: counts[base] / float(total_count) for base in DIFFUSION_BASE_ORDER}


def _state_mask_ratio(x_state, mask_idx: int, valid_mask_2d=None) -> float:
  """
  Mean mask probability/occupancy over valid positions.
  x_state: [B, T, V] one-hot/prob state.
  """
  if x_state.dim() != 3:
    return 0.0
  mask_chan = x_state[:, :, int(mask_idx)].to(torch.float32)
  if valid_mask_2d is None:
    return float(mask_chan.mean().item())
  denom = valid_mask_2d.sum().clamp_min(1.0)
  return float((mask_chan * valid_mask_2d).sum().item() / denom.item())


def _masked_base_ce_loss(
  log_p_x0,
  log_p_x0_old,
  last_x,
  mask_idx,
  gradient_type,
  model_new,
  model_old,
  length_mask_2d=None,
  agg_method="global",
  divergence="ce",
):
  """
  Masked-token base divergence between old/new models.
  agg_method:
    - global: mean base probs over masked positions per sample, then divergence.
    - position: divergence per masked position, then mean over masked positions.
  divergence:
    - ce: CE(old||new) = -sum old*log(new)
    - kl: KL(old||new) = sum old*(log(old)-log(new))
  """
  probs_new = log_p_x0.exp()
  probs_new = probs_new / probs_new.sum(dim=-1, keepdim=True).clamp_min(1e-12)
  probs_old = log_p_x0_old.exp()
  probs_old = probs_old / probs_old.sum(dim=-1, keepdim=True).clamp_min(1e-12)
  base_new = _token_probs_to_base_probs(probs_new, model_new, gradient_type=gradient_type)  # [B, L, 4]
  base_old = _token_probs_to_base_probs(probs_old, model_old, gradient_type=gradient_type)  # [B, L, 4]

  mask_tok = (last_x.argmax(dim=-1) == int(mask_idx))  # [B, T]
  if length_mask_2d is not None:
    mask_tok = mask_tok & (length_mask_2d > 0)

  L = min(base_new.shape[1], base_old.shape[1], mask_tok.shape[1])
  if L <= 0:
    return torch.tensor(0.0, device=log_p_x0.device)
  mask = mask_tok[:, :L].to(base_new.dtype)  # [B, L]
  base_new = base_new[:, :L, :]
  base_old = base_old[:, :L, :]

  mask_count = mask.sum()
  if mask_count.item() <= 0:
    return torch.tensor(0.0, device=log_p_x0.device)

  p_new = base_new.clamp_min(1e-8)
  p_new = p_new / p_new.sum(dim=-1, keepdim=True).clamp_min(1e-8)
  p_old = base_old.clamp_min(1e-8)
  p_old = p_old / p_old.sum(dim=-1, keepdim=True).clamp_min(1e-8)
  if agg_method == "global":
    per_sample_mask = mask.sum(dim=1)  # [B]
    valid = per_sample_mask > 0
    if not valid.any():
      return torch.tensor(0.0, device=log_p_x0.device)
    denom = per_sample_mask.clamp_min(1.0).unsqueeze(-1)  # [B, 1]
    p_new_s = (p_new * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, 4]
    p_old_s = (p_old * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, 4]
    p_new_s = p_new_s.clamp_min(1e-8)
    p_new_s = p_new_s / p_new_s.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    p_old_s = p_old_s.clamp_min(1e-8)
    p_old_s = p_old_s / p_old_s.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if divergence == "kl":
      loss_s = (p_old_s * (p_old_s.log() - p_new_s.log())).sum(dim=-1)
    else:
      loss_s = -(p_old_s * p_new_s.log()).sum(dim=-1)
    return loss_s[valid].mean()

  # position-wise mode
  if divergence == "kl":
    loss_pos = (p_old * (p_old.log() - p_new.log())).sum(dim=-1)  # [B, L]
  else:
    loss_pos = -(p_old * p_new.log()).sum(dim=-1)  # [B, L]
  return (loss_pos * mask).sum() / mask_count.clamp_min(1.0)


def _js_div_from_log_probs(log_p, log_q, eps: float = 1e-12):
  """
  Jensen-Shannon divergence between categorical distributions represented
  by log-prob tensors of shape [..., V].
  Returns an elementwise tensor with the same shape as the input.
  """
  p = log_p.exp()
  p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
  q = log_q.exp()
  q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
  m = 0.5 * (p + q)
  m = m / m.sum(dim=-1, keepdim=True).clamp_min(eps)
  log_m = torch.log(m.clamp_min(eps))
  return 0.5 * (p * (log_p - log_m) + q * (log_q - log_m))

class LengthSamplerStable:
    def __init__(
        self,
        length_values,
        p_data,
        device,
        min_count=8,
        update_every=None,
        update_every_epochs=None,
        ucb_c=0.0,
        # New stability knobs
        ema_alpha=0.02,        # EMA step for reward stats (recency)
        eta=1.0,               # exponent step size
        beta=0.05,             # trust-region smoothing on p
        A_clip=3.0,            # clip normalized advantages
        p_min=1e-3,            # floor prob to avoid starvation
    ):
        self.length_values = list(length_values)
        self.L2i = {L: i for i, L in enumerate(self.length_values)}
        self.device = device

        self.p_data = p_data.detach().clone().to(device)
        self.p_data = self.p_data / self.p_data.sum()
        self.p = self.p_data.clone()

        nL = len(self.length_values)
        self.count = torch.zeros(nL, device=device)
        # EMA mean reward per length (recency-weighted)
        self.ema_mean = torch.zeros(nL, device=device)
        # Global EMA mean for cold start
        self.global_ema = 0.0

        self.min_count = min_count
        self.update_every = update_every
        self.update_every_epochs = update_every_epochs
        self.ucb_c = ucb_c

        self.ema_alpha = ema_alpha
        self.eta = eta
        self.beta = beta
        self.A_clip = A_clip
        self.p_min = p_min

        self.t = 0

    def sample_lengths(self, B: int):
        idx = torch.multinomial(self.p, B, replacement=True)
        return [self.length_values[i] for i in idx.tolist()]

    @torch.no_grad()
    def update(self, sampled_lengths, rewards_tensor, epoch_num=None):
        self.t += 1
        r = rewards_tensor.detach().to(self.device)
        B = len(sampled_lengths)

        # global EMA baseline for cold start
        batch_mean = r.mean().item()
        self.global_ema = (1 - self.ema_alpha) * self.global_ema + self.ema_alpha * batch_mean

        # update per-length EMA mean
        for L, rv in zip(sampled_lengths, r.tolist()):
            i = self.L2i[L]
            self.count[i] += 1
            self.ema_mean[i] = (1 - self.ema_alpha) * self.ema_mean[i] + self.ema_alpha * rv

        if self.update_every_epochs is not None and epoch_num is not None:
            if (epoch_num + 1) % self.update_every_epochs == 0:
                self._recompute_probs()
        elif self.t % self.update_every == 0:
            self._recompute_probs()

    @torch.no_grad()
    def _recompute_probs(self):
        n = self.count

        # score: EMA mean where we have enough samples; otherwise use global EMA
        base = torch.full_like(self.ema_mean, float(self.global_ema))
        score = torch.where(n >= self.min_count, self.ema_mean, base)

        # optional exploration bonus
        if self.ucb_c > 0:
            bonus = self.ucb_c * torch.sqrt(torch.log(torch.tensor(float(self.t + 1), device=self.device)) / (n + 1.0))
            score = score + bonus

        # baseline b: expected score under current p (reduces variance)
        b = (self.p * score).sum()

        # advantage-like signal
        A = score - b

        # normalize + clip (stability)
        sd = A.std(unbiased=False).clamp_min(1e-6)
        A_norm = (A / sd).clamp(-self.A_clip, self.A_clip)

        # anchored exponentiated update: log p_data + eta * A
        logits = torch.log(self.p_data + 1e-12) + self.eta * A_norm
        p_new = torch.softmax(logits, dim=0)

        # trust-region smoothing on p (prevents distribution shock)
        p_smooth = (1 - self.beta) * self.p + self.beta * p_new

        # floor to avoid starvation
        p_smooth = torch.clamp(p_smooth, min=self.p_min)
        self.p = p_smooth / p_smooth.sum()





def load_token_length_distribution(path: str):
  text = Path(path).read_text().strip()
  if not text:
    raise ValueError(f"Token length distribution file is empty: {path}")
  start = text.find("{")
  end = text.rfind("}")
  if start == -1 or end == -1 or end <= start:
    raise ValueError(f"Could not find dict payload in token length distribution: {path}")
  dist = ast.literal_eval(text[start:end + 1])
  if not isinstance(dist, dict) or not dist:
    raise ValueError(f"Token length distribution is not a non-empty dict: {path}")
  lengths = sorted(dist.keys())
  probs = torch.tensor([float(dist[L]) for L in lengths], dtype=torch.float)
  probs_sum = probs.sum().item()
  if probs_sum <= 0:
    raise ValueError(f"Token length distribution has non-positive total probability: {path}")
  probs = probs / probs_sum
  return lengths, probs


def save_token_length_distribution(path: str, lengths, probs):
  probs = probs.detach().cpu().tolist()
  dist = {int(L): float(p) for L, p in zip(lengths, probs)}
  with open(path, 'w') as f:
    f.write(str(dist) + '\n')


def parse_base_probs(probs_str: str):
  vals = [float(x.strip()) for x in probs_str.split(",") if x.strip()]
  if len(vals) != 4:
    raise ValueError(
      f"--natural_base_probs must contain 4 comma-separated values (A,C,G,T), got: {probs_str}"
    )
  probs = torch.tensor(vals, dtype=torch.float32)
  if (probs < 0).any():
    raise ValueError(f"--natural_base_probs must be non-negative, got: {probs_str}")
  s = probs.sum().item()
  if s <= 0:
    raise ValueError(f"--natural_base_probs sum must be > 0, got: {probs_str}")
  return probs / s



def _build_sft_dataloader(cfg, args):
  tokenizer_type = str(cfg.data.get('tokenizer_type', 'csv_motif')).lower()
  if tokenizer_type in ('simple_vocab', 'simple'):
    vocab_json = cfg.data.get('tokenizer_vocab_path')
    if vocab_json is None:
      raise ValueError('tokenizer_type="simple_vocab" expects data.tokenizer_vocab_path in config.')
    with open(vocab_json, 'r') as fp:
      vocab_dict = json.load(fp)
    tokenizer = dataloader_gosai.SimpleVocabTokenizer(
      vocab_dict,
      pad_token=cfg.data.get('pad_token', 'N'),
      eos_token=cfg.data.get('eos_token', 'EOS'),
      unk_token=cfg.data.get('unk_token', None),
      normalize_case=cfg.data.get('normalize_case', True),
    )
    pad_id = tokenizer.pad_token_id
  elif tokenizer_type == 'csv_motif':
    vocab_json = cfg.data.get('motif_vocab_path')
    if vocab_json is None:
      raise ValueError('tokenizer_type="csv_motif" expects data.motif_vocab_path in config.')
    tokenizer = dataloader_gosai.MotifAwareTokenizer(
      vocab_json_path=vocab_json,
      pad_token=cfg.data.get('pad_token', 'N'),
      eos_token=cfg.data.get('eos_token', 'EOS'),
      base_tokens=cfg.data.get('motif_base_tokens', ("A", "C", "G", "T")),
      max_length=cfg.model.length,
      trim_to=cfg.data.get('motif_trim_len'),
    )
    pad_id = tokenizer.pad_token_id
  else:
    raise ValueError(f'Unsupported tokenizer_type for SFT regularization: {tokenizer_type}')

  dataset = dataloader_gosai.UTRDataset(
    csv_path=args.sft_dataset_csv,
    tokenizer=tokenizer,
    max_length=cfg.model.length,
    pad_id=pad_id,
    seq_col=args.sft_seq_col,
    label_col=None,
  )
  return torch.utils.data.DataLoader(
    dataset,
    batch_size=args.sft_batch_size,
    shuffle=True,
    num_workers=args.sft_num_workers,
    pin_memory=True,
    drop_last=True,
  )


def _next_sft_batch(sft_iter, sft_loader):
  try:
    batch = next(sft_iter)
  except StopIteration:
    sft_iter = iter(sft_loader)
    batch = next(sft_iter)
  return batch, sft_iter


def _normalize_target_length(target_length, device, seq_len):
  if target_length is None:
    return None
  if isinstance(target_length, torch.Tensor):
    target_len = target_length.to(device=device, dtype=torch.long)
  else:
    target_len = torch.tensor(target_length, device=device, dtype=torch.long)
  return target_len.clamp(min=0, max=seq_len)


def _build_length_mask(target_length, device, seq_len, dtype=torch.float32):
  target_len = _normalize_target_length(target_length, device, seq_len)
  if target_len is None:
    return None
  seq_idx = torch.arange(seq_len, device=device).unsqueeze(0)
  return (seq_idx < target_len.unsqueeze(1)).to(dtype)


def _get_ood_last_k_steps(args):
  return int(args.truncate_steps if args.ood_last_k_steps <= 0 else args.ood_last_k_steps)


def fine_tune(new_model, new_model_y, new_model_y_eval, old_model, args, eps_ood=None, eps=1e-5, sft_loader=None):
  # torch.autograd.set_detect_anomaly(True)
  with open(log_path, 'w') as f:
    f.write(args.__repr__() + '\n')
  if args.log_denoise_trajectory:
    with open(trajectory_log_path, 'a') as f:
      f.write(f"# fine_tune start total_num_steps={args.total_num_steps}\n")

  ref_kmer_freq = None
  ref_kmer_count = 0
  if args.eval_kmer_reference_csv:
    ref_sequences = _load_csv_sequences(args.eval_kmer_reference_csv, seq_col=args.eval_kmer_seq_col)
    _, ref_kmer_freq = _kmer_freq_vector(ref_sequences, k=args.eval_kmer_k)
    ref_kmer_count = len(ref_sequences)
    print(
      f"[eval_kmer] loaded {ref_kmer_count} reference sequences from "
      f"{args.eval_kmer_reference_csv} using k={args.eval_kmer_k}"
    )

  length_values = None
  length_probs = None
  if args.token_length_distribution:
    length_values, length_probs = load_token_length_distribution(args.token_length_distribution)
  length_sampler = None
  if length_values is not None:
      length_sampler = LengthSamplerStable(
          length_values=length_values,
          p_data=length_probs,                 # original P_data(L)
          device=length_probs.device,
          min_count=8,
          update_every_epochs=10,
          ucb_c=0.0,
          # New stability knobs
          ema_alpha=0.02,        # EMA step for reward stats (recency)
          eta=0.8,               # exponent step size
          beta=0.07,             # trust-region smoothing on p
          A_clip=3.0,            # clip normalized advantages
          p_min=1e-3,            # floor prob to avoid starvation
      )
  # if args.ood_enable and eps_ood is None:
  #   raise ValueError("OOD gating is enabled but eps_ood is None. Provide calibrated or overridden threshold.")
  # if args.ood_enable:
  #   ood_k_steps = _get_ood_last_k_steps(args)
  #   print(f"[ood] enabled: eps_ood={eps_ood:.6f} last_k={ood_k_steps} r_min={args.ood_r_min}")

  new_model.config.finetuning.truncate_steps = args.truncate_steps
  new_model.config.finetuning.gumbel_softmax_temp = args.gumbel_temp
  dt = (1 - eps) / args.total_num_steps
  # Keep generation deterministic for OOD scoring comparisons.
  # We explicitly run in eval mode to avoid train-time stochasticity.
  new_model.eval()
  torch.set_grad_enabled(True)
  optim = torch.optim.Adam(new_model.parameters(), lr=args.learning_rate)
  scheduler = None
  if args.lr_cosine_decay:
    T = 100
    lr_min = args.lr_min
    lr0 = args.learning_rate

    def _lr_lambda(epoch):
      return (lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * epoch / T))) / lr0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=_lr_lambda)
  best_metric = 0.0
  best_checkpoint_path = os.path.join(save_path, 'best_model.ckpt')
  batch_losses = []
  batch_rewards = []
  sft_enabled = (args.sft_reg_coeff > 0 and sft_loader is not None)
  sft_iter = iter(sft_loader) if sft_enabled else None
  target_base_probs = None
  if args.base_comp_coeff > 0:
    target_base_probs = parse_base_probs(args.natural_base_probs).to(new_model.device)

  lambda_atg = 0

  for epoch_num in range(args.num_epochs):
    rewards = []
    rewards_eval = []
    losses = []
    reward_losses = []
    sft_losses = [] if sft_enabled else None
    kl_losses = []
    kl_losses_for = []
    base_comp_losses = []
    mask_base_ce_losses = []
    entropy_losses = []
    acgt_means = []
    ood_frac_mean = 0.0
    ood_score_mean = 0.0
    drift_final_mask_new_mean = 0.0
    drift_final_mask_old_mean = 0.0
    drift_mask_curve_l1_mean = 0.0
    drift_seq_hamming_mean = 0.0
    drift_seq_basecomp_l1_mean = 0.0
    drift_seq_kmer_l1_mean = 0.0
    eval_kmer_corr = None
    eval_gen_at_fraction = None
    eval_gen_base_ratios = None
    tot_grad_norm = 0.0
    # Keep eval mode throughout this routine for stable OOD score scale.
    new_model.eval()
    skipped_steps = 0
    epoch_sampled_lengths = []
    epoch_reward_1d = []
    epoch_generated_sequences = []
    epoch_traj_sample_idx = None
    for _step in range(args.num_accum_steps):
      target_length = None
      # if length_values is not None:
      #   sampled_idx = torch.multinomial(length_probs, args.batch_size, replacement=True).tolist()
      #   sampled_lengths = [length_values[i] for i in sampled_idx]
      #   target_length = sampled_lengths
      if length_sampler is not None:
        sampled_lengths = length_sampler.sample_lengths(args.batch_size)
        target_length = sampled_lengths
      sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list, kl_x_list, p_vocab, log_p_x0_last = new_model._sample_finetune_gradient(
        eval_sp_size=args.batch_size,
        copy_flag_temp=args.copy_flag_temp,
        target_length=target_length,
        gradient_type=args.gradient_type)
      with torch.no_grad():
        base_probs_metric = _sample_to_base_probs(sample.detach(), new_model, args.gradient_type)
        valid_lens_metric = _get_valid_lens(target_length, batch_size=base_probs_metric.shape[0], max_len=base_probs_metric.shape[1])
        for bi, valid_len in enumerate(valid_lens_metric):
          epoch_generated_sequences.append(
            _decode_base_sequence(base_probs_metric[bi], valid_len=int(valid_len))
          )
      old_last_x_list_for = None
      old_condt_list_for = None
      old_move_chance_t_list_for = None
      old_copy_flag_list_for = None
      if (not args.js) and args.forward_kl_on_old_xt and args.beta > 0:
        with torch.no_grad():
          _, old_last_x_list_for, old_condt_list_for, old_move_chance_t_list_for, old_copy_flag_list_for, _, _, _ = old_model._sample_finetune_gradient(
            eval_sp_size=args.batch_size,
            copy_flag_temp=args.copy_flag_temp,
            target_length=target_length,
            gradient_type=args.gradient_type)
      if args.enable_drift_monitor and (_step % max(1, args.drift_monitor_every) == 0):
        with torch.no_grad():
          old_sample, old_last_x_list, _, _, _, old_kl_x_list, old_p_vocab, _ = old_model._sample_finetune_gradient(
            eval_sp_size=args.batch_size,
            copy_flag_temp=args.copy_flag_temp,
            target_length=target_length,
            gradient_type=args.gradient_type)

          # # 1) Final-step mask ratio in post-update x (from p_vocab/old_p_vocab).
          # seq_len_post = p_vocab.shape[1]
          # valid_mask_post = _build_length_mask(
          #   target_length, device=p_vocab.device, seq_len=seq_len_post, dtype=torch.float32)
          # new_final_mask = (1.0 - p_vocab.sum(dim=-1)).clamp(0.0, 1.0)
          # old_final_mask = (1.0 - old_p_vocab.sum(dim=-1)).clamp(0.0, 1.0)
          # if valid_mask_post is None:
          #   drift_final_mask_new_mean = float(new_final_mask.mean().item())
          #   drift_final_mask_old_mean = float(old_final_mask.mean().item())
          # else:
          #   denom = valid_mask_post.sum().clamp_min(1.0)
          #   drift_final_mask_new_mean = float(((new_final_mask * valid_mask_post).sum() / denom).item())
          #   drift_final_mask_old_mean = float(((old_final_mask * valid_mask_post).sum() / denom).item())

          # 2) Trajectory-level divergence via mask-ratio curves over t.
          # Tm = min(len(kl_x_list), len(old_kl_x_list), int(args.total_num_steps))
          # if Tm > 0:
          #   seq_len_curve = kl_x_list[0].shape[1]
          #   valid_mask_curve = _build_length_mask(
          #     target_length, device=kl_x_list[0].device, seq_len=seq_len_curve, dtype=torch.float32)
          #   new_curve = []
          #   old_curve = []
          #   mask_idx = int(getattr(new_model, "mask_index", -1))
          #   old_mask_idx = int(getattr(old_model, "mask_index", -1))
          #   for tt in range(Tm):
          #     new_curve.append(_state_mask_ratio(kl_x_list[tt], mask_idx, valid_mask_curve))
          #     old_curve.append(_state_mask_ratio(old_kl_x_list[tt], old_mask_idx, valid_mask_curve))
          #   curve_l1 = sum(abs(a - b) for a, b in zip(new_curve, old_curve)) / float(Tm)
          #   drift_mask_curve_l1_mean = float(curve_l1)

          # 3) Sequence-level distances between new/old final rollouts.
      #     base_probs_new = _sample_to_base_probs(sample.detach(), new_model, args.gradient_type)
      #     base_probs_old = _sample_to_base_probs(old_sample.detach(), old_model, args.gradient_type)
      #     B = int(base_probs_new.shape[0])
      #     max_len = min(int(base_probs_new.shape[1]), int(base_probs_old.shape[1]))
      #     valid_lens = _get_valid_lens(target_length, B, max_len)
      #     hamming_vals = []
      #     comp_vals = []
      #     kmer_vals = []
      #     for bi in range(B):
      #       L = int(max(0, min(valid_lens[bi], max_len)))
      #       s_new = _decode_base_sequence(base_probs_new[bi], valid_len=L)
      #       s_old = _decode_base_sequence(base_probs_old[bi], valid_len=L)
      #       hamming_vals.append(_hamming_norm(s_new, s_old))
      #       comp_vals.append(float(torch.abs(_base_comp_vec(s_new) - _base_comp_vec(s_old)).sum().item()))
      #       kmer_vals.append(_kmer_dist_l1(s_new, s_old, k=max(1, int(args.drift_kmer_k))))
      #     drift_seq_hamming_mean = float(sum(hamming_vals) / max(1, len(hamming_vals)))
      #     drift_seq_basecomp_l1_mean = float(sum(comp_vals) / max(1, len(comp_vals)))
      #     drift_seq_kmer_l1_mean = float(sum(kmer_vals) / max(1, len(kmer_vals)))
      # if args.log_denoise_trajectory and _step == 0:
      #   epoch_traj_sample_idx = int(max(0, min(args.trajectory_sample_index, sample.shape[0] - 1)))
      #   with open(trajectory_log_path, 'a') as f:
      #     f.write(f"\n# epoch={epoch_num} selected_sample={epoch_traj_sample_idx}\n")
      # if args.log_motif_stats and (_step % args.log_motif_stats_every == 0):
      #   if _step == 0:
      #     print(f"[motif_stats] log_p_x0_last is None: {log_p_x0_last is None}")
      #   with torch.no_grad():
      #     probs = p_vocab if log_p_x0_last is None else log_p_x0_last.exp()
      #     if probs.dim() == 3:
      #       probs = probs.clamp_min(1e-8)
      #       probs = probs / probs.sum(dim=-1, keepdim=True)
      #       mean_entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
      #       avg_prob = probs.mean(dim=(0, 1))
      #       avg_prob = avg_prob / avg_prob.sum()
      #       topk = min(5, avg_prob.numel())
      #       top_vals, top_idx = torch.topk(avg_prob, k=topk)
      #       top_pairs = ", ".join(
      #         f"{int(i)}:{float(v):.4f}" for i, v in zip(top_idx, top_vals)
      #       )
      #       msg = (
      #         f"[motif_stats] epoch {epoch_num} step {_step} "
      #         f"mean_entropy {mean_entropy:.4f} top{topk} {top_pairs}"
      #       )
      #       print(msg)
      #       with open(log_path, 'a') as f:
      #         f.write(msg + "\n")
      if args.gradient_type == "motif_soft" and new_model.vocab_size > 7:
        if not hasattr(new_model, "motif2base_stencil"):
          raise ValueError("gradient_type=motif_soft requires motif2base_stencil on the model.")
        base_probs = _motif_sample_to_base(sample, new_model)
      else:
        base_probs = sample
      base_probs = base_probs.transpose(1, 2)


      use_soft = args.reward_type in ('utrlm', 'rnafm', 'framepool')
      preds = new_model_y(base_probs, soft_input=use_soft)
      reward = preds[..., 0]

      with torch.no_grad():
        preds_eval = new_model_y_eval(base_probs, soft_input=False)


      reward_argmax_eval = preds_eval[..., 0]
      rewards_eval.append(reward_argmax_eval.detach().mean().cpu().item())

      base_comp_loss = torch.tensor(0.0, device=base_probs.device)
      if target_base_probs is not None:
        B, _, L = base_probs.shape
        if target_length is not None:
          if isinstance(target_length, torch.Tensor):
            tgt_len = target_length.to(base_probs.device, dtype=torch.long)
          else:
            tgt_len = torch.tensor(target_length, device=base_probs.device, dtype=torch.long)
          tgt_len = tgt_len.clamp(min=0, max=L)
          pos = torch.arange(L, device=base_probs.device).unsqueeze(0)
          valid_mask = (pos < tgt_len.unsqueeze(1)).to(base_probs.dtype)  # [B, L]
          denom = valid_mask.sum().clamp_min(1.0)
          batch_base_probs = (base_probs * valid_mask.unsqueeze(1)).sum(dim=(0, 2)) / denom
        else:
          batch_base_probs = base_probs.mean(dim=(0, 2))
        batch_base_probs = batch_base_probs.clamp_min(1e-8)
        batch_base_probs = batch_base_probs / batch_base_probs.sum()
        if args.base_comp_loss_type == "kl":
          # KL(P_opt || P_natural) over A/C/G/T marginals.
          base_comp_loss = torch.sum(
            batch_base_probs * (torch.log(batch_base_probs) - torch.log(target_base_probs))
          )
        else:
          base_comp_loss = F.mse_loss(batch_base_probs, target_base_probs)

      total_kl = []
      total_kl_for = []
      total_mask_base_ce = []
      seq_len = last_x_list[0].shape[1]
      length_mask_2d = _build_length_mask(target_length, device=new_model.device, seq_len=seq_len, dtype=torch.float32)
      length_mask = None if length_mask_2d is None else length_mask_2d.unsqueeze(-1)
      if old_last_x_list_for is not None:
        seq_len_for = old_last_x_list_for[0].shape[1]
        length_mask_for_2d = _build_length_mask(target_length, device=new_model.device, seq_len=seq_len_for, dtype=torch.float32)
        length_mask_for = None if length_mask_for_2d is None else length_mask_for_2d.unsqueeze(-1)
      else:
        seq_len_for = seq_len
        length_mask_for_2d = length_mask_2d
        length_mask_for = length_mask
      # ood_sum_nll = torch.zeros(reward.shape[0], device=reward.device, dtype=torch.float32)
      # ood_cnt = torch.zeros(reward.shape[0], device=reward.device, dtype=torch.float32)
      # ood_start_t = max(0, args.total_num_steps - _get_ood_last_k_steps(args))
      mask_base_start_t = (
        0 if args.mask_base_truncate_steps <= 0
        else max(0, args.total_num_steps - int(args.mask_base_truncate_steps))
      )
      for random_t in range(args.total_num_steps):
        if args.truncate_kl and random_t < args.total_num_steps - args.truncate_steps:
          continue
        last_x = kl_x_list[random_t]
        # print(last_x.requires_grad)
        condt = condt_list[random_t]
        move_chance_t = move_chance_t_list[random_t]
        copy_flag = copy_flag_list[random_t]
        log_p_x0 = new_model.forward(last_x, condt)[:, :, :-1]
        log_p_x0_old = old_model.forward(last_x, condt)[:, :, :-1]

        # Optional denoising trajectory logging: one selected sample per epoch.
        if (
          args.log_denoise_trajectory
          and _step == 0
          and (random_t % args.trajectory_step_stride == 0)
          and (args.trajectory_max_steps <= 0 or random_t < args.trajectory_max_steps)
        ):
          with torch.no_grad():
            probs_t = log_p_x0.exp()
            probs_t = probs_t / probs_t.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            probs_t_old = log_p_x0_old.exp()
            probs_t_old = probs_t_old / probs_t_old.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            b_idx = epoch_traj_sample_idx if epoch_traj_sample_idx is not None else 0
            if target_length is None:
              valid_len_tok = last_x.shape[1]
            elif isinstance(target_length, torch.Tensor):
              valid_len_tok = int(target_length[b_idx].item())
            else:
              valid_len_tok = int(target_length[b_idx])
            mask_idx = int(getattr(new_model, "mask_index", -1))
            avg_new, mask_cnt = _masked_avg_vocab_probs(
              probs_t[b_idx], last_x[b_idx], mask_idx, token_valid_len=valid_len_tok)
            avg_old, _ = _masked_avg_vocab_probs(
              probs_t_old[b_idx], last_x[b_idx], mask_idx, token_valid_len=valid_len_tok)
            avg_new_str = _format_vocab_distribution(avg_new, new_model)
            avg_old_str = _format_vocab_distribution(avg_old, old_model)
            traj_msg = (
              f"epoch={epoch_num}\tstep={_step}\tt={random_t}\tsample={b_idx}\tlen={valid_len_tok}\t"
              f"num_masking_token={mask_cnt}\t"
              f"motif_masked_avg_new={avg_new_str}\t"
              f"motif_masked_avg_old={avg_old_str}"
            )
            with open(trajectory_log_path, 'a') as f:
              f.write(traj_msg + "\n")

        p_x0 = log_p_x0.exp()
        p_x0_old = log_p_x0_old.exp()
        # print(log_p_x0.requires_grad)
        if args.js:
          js_div = copy_flag * _js_div_from_log_probs(log_p_x0, log_p_x0_old) / move_chance_t[0, 0, 0]
          if length_mask is not None:
            js_div = js_div * length_mask
          js_div = (js_div * last_x[:, :, :-1]).sum((1, 2))
          total_kl.append(js_div)
        else:
          kl_div = copy_flag * (-p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)) / move_chance_t[0, 0, 0]

          if old_last_x_list_for is not None:
            last_x_for = old_last_x_list_for[random_t]
            condt_for = old_condt_list_for[random_t]
            move_chance_t_for = old_move_chance_t_list_for[random_t]
            copy_flag_for = old_copy_flag_list_for[random_t]
            log_p_x0_for = new_model.forward(last_x_for, condt_for)[:, :, :-1]
            log_p_x0_old_for = old_model.forward(last_x_for, condt_for)[:, :, :-1]
            p_x0_for = log_p_x0_for.exp()
            p_x0_old_for = log_p_x0_old_for.exp()
            kl_div_for = copy_flag_for * (-p_x0_old_for + p_x0_for + p_x0_old_for * (log_p_x0_old_for - log_p_x0_for)) / move_chance_t_for[0, 0, 0]
          else:
            last_x_for = last_x
            kl_div_for = copy_flag * (-p_x0_old + p_x0 + p_x0_old * (log_p_x0_old - log_p_x0)) / move_chance_t[0, 0, 0]

          if length_mask is not None:
            kl_div = kl_div * length_mask
          if length_mask_for is not None:
            kl_div_for = kl_div_for * length_mask_for
          kl_div = (kl_div * last_x[:, :, :-1]).sum((1, 2))
          kl_div_for = (kl_div_for * last_x_for[:, :, :-1]).sum((1, 2))
          total_kl.append(kl_div)
          total_kl_for.append(kl_div_for)
        if args.mask_base_ce_coeff > 0 and random_t >= mask_base_start_t:
          step_mask_ce = _masked_base_ce_loss(
            log_p_x0=log_p_x0,
            log_p_x0_old=log_p_x0_old,
            last_x=last_x,
            mask_idx=int(getattr(new_model, "mask_index", -1)),
            gradient_type=args.gradient_type,
            model_new=new_model,
            model_old=old_model,
            length_mask_2d=length_mask_2d,
            agg_method=args.mask_base_agg_method,
            divergence=args.mask_base_divergence,
          )
          total_mask_base_ce.append(step_mask_ce)
        # OOD per-step metrics disabled for speed.
        # if args.ood_enable and random_t >= ood_start_t:
        #   a0_hat_new = log_p_x0.argmax(dim=-1)  # [B, T]
        #   nll = -log_p_x0_old.gather(-1, a0_hat_new.unsqueeze(-1)).squeeze(-1)  # [B, T]
        #   if length_mask_2d is not None:
        #     ood_sum_nll += (nll * length_mask_2d).sum(dim=-1)
        #     ood_cnt += length_mask_2d.sum(dim=-1)
        #   else:
        #     ood_sum_nll += nll.sum(dim=-1)
        #     ood_cnt += float(nll.shape[1])

      # OOD hard-gating disabled for speed.
      # if args.ood_enable:
      #   ood_scores = ood_sum_nll / ood_cnt.clamp_min(1.0)
      #   ood_mask = ood_scores > eps_ood
      #   reward = torch.where(ood_mask, torch.full_like(reward, args.ood_r_min), reward)
      #   ood_frac_list.append(float(ood_mask.float().mean().item()))
      #   ood_score_mean_list.append(float(ood_scores.mean().item()))
      # else:
      #   ood_scores = None
      rewards.append(reward.detach().mean().cpu().item())

      current_step = epoch_num * args.num_accum_steps + _step + 1
      if args.kl_coeff_schedule_warmup and current_step < args.kl_coeff_schedule_warmup:
        current_alpha = current_step / args.kl_coeff_schedule_warmup * args.alpha
        current_beta = current_step / args.kl_coeff_schedule_warmup * args.beta
      elif args.alpha_schedule_warmup and epoch_num < args.alpha_schedule_warmup:
        current_alpha = (epoch_num + 1) / args.alpha_schedule_warmup * args.alpha
        current_beta = (epoch_num + 1) / args.alpha_schedule_warmup * args.beta
      else:
        current_alpha = args.alpha
        current_beta = args.beta

      kl_loss = torch.stack(total_kl, 1).sum(1).mean()
      if args.js:
        kl_loss_for = torch.tensor(0.0, device=kl_loss.device)
      else:
        kl_loss_for = torch.stack(total_kl_for, 1).sum(1).mean()
      if len(total_mask_base_ce) > 0:
        mask_base_ce_loss = torch.stack(total_mask_base_ce).mean()
      else:
        mask_base_ce_loss = torch.tensor(0.0, device=kl_loss.device)
      entropy_loss = torch.tensor(0.0, device=kl_loss.device)
      probs = None
      if args.entropy:
        if log_p_x0_last is None:
          probs = p_vocab
        else:
          probs = log_p_x0_last.exp()
        probs = probs.clamp_min(1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        entropy_loss = -(probs * probs.log()).sum(dim=-1).mean()
        if args.entropy_coeff_schedule_warmup and current_step < args.entropy_coeff_schedule_warmup:
          current_entropy_coeff = current_step / args.entropy_coeff_schedule_warmup * args.entropy_coeff
        else:
          current_entropy_coeff = args.entropy_coeff
      else:
        current_entropy_coeff = 0.0
      ## normalization for the reward
      r = reward
      mu_reward = r.detach().mean()
      std_reward = r.detach().std()
      reward_norm = (r - mu_reward) / (std_reward+1e-10)
      # reward_loss = - torch.mean(reward_norm)
      reward_loss = - torch.mean(reward)
      sft_loss = torch.tensor(0.0, device=reward_loss.device)
      if sft_enabled:
        sft_batch, sft_iter = _next_sft_batch(sft_iter, sft_loader)
        sft_x0 = sft_batch['seqs'].to(new_model.device)
        sft_attention_mask = sft_batch['attention_mask'].to(new_model.device, dtype=torch.float32)
        sft_loss = new_model._loss(sft_x0, sft_attention_mask).loss
      if probs is None:
        if log_p_x0_last is None:
          probs = p_vocab
        else:
          probs = log_p_x0_last.exp()
        probs = probs.clamp_min(1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
      acgt_mean = probs[:, :, :4].mean()
      # if epoch_num > 0 and args.entropy:
      loss = (
        reward_loss
        + kl_loss * current_alpha
        + kl_loss_for * (0.0 if args.js else current_beta)
        + args.base_comp_coeff * base_comp_loss
        + args.mask_base_ce_coeff * mask_base_ce_loss
      )
      if sft_enabled:
        loss = loss + args.sft_reg_coeff * sft_loss
      # else:
      #   loss = reward_loss + kl_loss * current_alpha
      loss = loss / args.num_accum_steps

      loss.backward()
      ## update length sampler
      # reward_1d = reward.squeeze(-1)
      # if length_sampler is not None:
      #   epoch_sampled_lengths.extend(sampled_lengths)
      #   epoch_reward_1d.append(reward_1d.detach())
      ## update length sampler
      if (_step + 1) % args.num_accum_steps == 0:
        norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), args.gradnorm_clip)
      #   tot_grad_norm += norm
      #   optim.step()
      #   optim.zero_grad()
        if norm > args.grad_skip_threshold:
                min_move = min(t.min().item() for t in move_chance_t_list) if move_chance_t_list else float('nan')
                print(
                  f"WARNING: Skipped step at epoch {epoch_num} due to exploding gradient: {norm} | "
                  f"reward_mean {reward.mean().item():.6f} reward_std {reward.std().item():.6f} | "
                  f"kl {kl_loss.item():.6f} entropy {float(entropy_loss):.6f} | "
                  f"min_move_chance {min_move:.6f}"
                )
                optim.zero_grad() # Throw away these gradients
                skipped_steps += 1
        else:
            # Only step if gradients are healthy
            tot_grad_norm += norm
            optim.step()
            optim.zero_grad()

      batch_losses.append(loss.item())
      batch_rewards.append(torch.mean(reward).item())
      losses.append(loss.item() * args.num_accum_steps)
      reward_losses.append(reward_loss.item())
      if sft_enabled:
        sft_losses.append(float(sft_loss))
      kl_losses.append(kl_loss.item())
      kl_losses_for.append(kl_loss_for.item())
      base_comp_losses.append(float(base_comp_loss))
      mask_base_ce_losses.append(float(mask_base_ce_loss))
      # entropy_losses.append(float(entropy_loss))
      # acgt_means.append(float(acgt_mean))
  
    rewards = torch.tensor(rewards)
    rewards_eval = torch.tensor(rewards_eval)
    losses = torch.tensor(losses)
    reward_losses = torch.tensor(reward_losses)
    # sft_losses = torch.tensor(sft_losses) if sft_enabled else None
    # mean_sft_loss = sft_losses.float().mean().item() if sft_losses is not None else 0.0
    kl_losses = torch.tensor(kl_losses)
    kl_losses_for = torch.tensor(kl_losses_for)
    # base_comp_losses = torch.tensor(base_comp_losses)
    mask_base_ce_losses = torch.tensor(mask_base_ce_losses)
    # entropy_losses = torch.tensor(entropy_losses)
    # acgt_means = torch.tensor(acgt_means)
    # reward_mean_std = rewards.float().std(unbiased=False).item()
    # if length_sampler is not None:
    #   epoch_rewards_tensor = torch.cat(epoch_reward_1d, dim=0) if epoch_reward_1d else torch.tensor([], device=new_model.device)
    #   if len(epoch_sampled_lengths) == len(epoch_rewards_tensor):
    #     length_sampler.update(epoch_sampled_lengths, epoch_rewards_tensor, epoch_num=epoch_num)
    if ref_kmer_freq is not None:
      _, gen_kmer_freq = _kmer_freq_vector(epoch_generated_sequences, k=args.eval_kmer_k)
      eval_kmer_corr = _pearson_corr(gen_kmer_freq, ref_kmer_freq)
      # eval_gen_at_fraction = _at_fraction(epoch_generated_sequences)
    # eval_gen_base_ratios = _base_ratios(epoch_generated_sequences)
    print_items = [
      "Epoch %d" % epoch_num,
      "Mean reward %f" % rewards.float().mean().item(),
      "Mean reward eval %f" % rewards_eval.float().mean().item(),
      # "Mean reward std %f" % rewards.float().std(unbiased=False).item(),
      "Mean grad norm %f" % tot_grad_norm,
      "Mean loss %f" % losses.float().mean().item(),
      "Mean reward loss %f" % reward_losses.float().mean().item(),
      # "Mean sft loss %f" % mean_sft_loss,
      "Mean kl loss %f" % kl_losses.float().mean().item(),
      "Mean kl loss for %f" % kl_losses_for.float().mean().item(),
      # "Mean base comp loss %f" % base_comp_losses.float().mean().item(),
      "Mean mask base CE loss %f" % mask_base_ce_losses.float().mean().item(),
      # "Mean entropy loss %f" % entropy_losses.float().mean().item(),
      # "Mean ACGT prob %f" % acgt_means.float().mean().item(),
      # "OOD frac %f" % ood_frac_mean,
      # "OOD score mean %f" % ood_score_mean,
      # "Mask final(new) %f" % drift_final_mask_new_mean,
      # "Mask final(old) %f" % drift_final_mask_old_mean,
      # "Mask curve L1 %f" % drift_mask_curve_l1_mean,
      # "Seq Hamming %f" % drift_seq_hamming_mean,
      # "Seq basecomp L1 %f" % drift_seq_basecomp_l1_mean,
      # "Seq kmer L1 %f" % drift_seq_kmer_l1_mean
    ]
    if eval_kmer_corr is not None:
      print_items.append("Eval %d-mer corr %f" % (args.eval_kmer_k, eval_kmer_corr))
    # if eval_gen_at_fraction is not None:
    #   print_items.append("Gen A+T fraction %f" % eval_gen_at_fraction)
    # if eval_gen_base_ratios is not None:
    #   print_items.extend(
    #     "Gen %s ratio %f" % (base, eval_gen_base_ratios[base])
    #     for base in DIFFUSION_BASE_ORDER
    #   )
    print(*print_items)
    if args.name != 'debug':
      log_payload = {
        "epoch": epoch_num,
        "mean_reward": rewards.float().mean().item(),
        "mean_reward_eval": rewards_eval.float().mean().item(),
        # "mean reward std": rewards.float().std(unbiased=False).item(),
        "mean_grad_norm": tot_grad_norm,
        "mean_loss": losses.float().mean().item(),
        "mean_reward_loss": reward_losses.float().mean().item(),
        # "mean_sft_loss": mean_sft_loss,
        "mean_kl_loss": kl_losses.float().mean().item(),
        "mean_kl_loss_for": kl_losses_for.float().mean().item(),
        # "mean_base_comp_loss": base_comp_losses.float().mean().item(),
        "mean_mask_base_ce_loss": mask_base_ce_losses.float().mean().item(),
        # "mean_entropy_loss": entropy_losses.float().mean().item(),
        # "mean_acgt_prob": acgt_means.float().mean().item(),
        # "ood_frac": ood_frac_mean,
        # "ood_score_mean": ood_score_mean,
        # "drift_final_mask_new": drift_final_mask_new_mean,
        # "drift_final_mask_old": drift_final_mask_old_mean,
        # "drift_mask_curve_l1": drift_mask_curve_l1_mean,
        # "drift_seq_hamming": drift_seq_hamming_mean,
        # "drift_seq_basecomp_l1": drift_seq_basecomp_l1_mean,
        # "drift_seq_kmer_l1": drift_seq_kmer_l1_mean,
      }
      if eval_kmer_corr is not None:
        log_payload["eval_kmer_corr"] = eval_kmer_corr
      # if eval_gen_at_fraction is not None:
      #   log_payload["eval_gen_at_fraction"] = eval_gen_at_fraction
      # if eval_gen_base_ratios is not None:
      #   for base in DIFFUSION_BASE_ORDER:
      #     log_payload[f"gen_base_ratio_{base}"] = eval_gen_base_ratios[base]
      wandb.log(log_payload)
    with open(log_path, 'a') as f:
      log_line = (
        f"Epoch {epoch_num} Mean reward {rewards.float().mean().item()} Mean reward eval {rewards_eval.float().mean().item()} "
        f"Mean grad norm {tot_grad_norm} Mean loss {losses.float().mean().item()} "
        f"Mean reward loss {reward_losses.float().mean().item()}"
        f"Mean kl loss {kl_losses.float().mean().item()} "
        f"Mean kl loss for {kl_losses_for.float().mean().item()} "
        # f"Mean base comp loss {base_comp_losses.float().mean().item()} "
        f"Mean mask base CE loss {mask_base_ce_losses.float().mean().item()} "
      )
      if eval_kmer_corr is not None:
        log_line += f"Eval {args.eval_kmer_k}-mer corr {eval_kmer_corr}"
      # if eval_gen_at_fraction is not None:
      #   log_line += f" Gen A+T fraction {eval_gen_at_fraction}"
      # if eval_gen_base_ratios is not None:
      #   for base in DIFFUSION_BASE_ORDER:
      #     log_line += f" Gen {base} ratio {eval_gen_base_ratios[base]}"
      log_line += "\n"
      f.write(
        log_line
        # f"Mean entropy loss {entropy_losses.float().mean().item()} "
        # f"Mean ACGT prob {acgt_means.float().mean().item()}\n "
        # f"OOD frac {ood_frac_mean} "
        # f"OOD score mean {ood_score_mean} "
        # f"drift_final_mask_new {drift_final_mask_new_mean} "
        # f"drift_final_mask_old {drift_final_mask_old_mean} "
        # f"drift_mask_curve_l1 {drift_mask_curve_l1_mean} "
        # f"drift_seq_hamming {drift_seq_hamming_mean} "
        # f"drift_seq_basecomp_l1 {drift_seq_basecomp_l1_mean} "
        # f"drift_seq_kmer_l1 {drift_seq_kmer_l1_mean}\n"
      )

    # Track and save best checkpoint by combined metric
    mean_reward = rewards.float().mean().item()
    mean_reward_eval = rewards_eval.float().mean().item()
    # metric_value = 0.5 * (mean_reward + mean_reward_eval)
    metric_value = losses.float().mean().item()
    if metric_value < best_metric:
      best_metric = metric_value
      torch.save(new_model.state_dict(), best_checkpoint_path)
      print(f"New best (total_loss={best_metric:.4f}) saved to {best_checkpoint_path}")

    if (epoch_num + 1) % args.save_every_n_epochs == 0:
      model_path = os.path.join(save_path, f'model_{epoch_num}.ckpt')
      torch.save(new_model.state_dict(), model_path)
      print(f"Model saved at epoch {epoch_num}")

    if scheduler is not None:
      scheduler.step()

  if args.name != 'debug':
    wandb.finish()

  if length_sampler is not None:
    out_path = args.length_distribution_out or os.path.join(save_path, 'length_distribution.txt')
    save_token_length_distribution(out_path, length_sampler.length_values, length_sampler.p)
    print(f"Saved length distribution to {out_path}")

  return batch_losses


argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--base_path', type=str, default='/home/xli263/xli/utr_design/DRAKES/data_and_model/')
argparser.add_argument('--log_base_dir', type=str, default=None,
                       help='Optional base output directory for logs/checkpoints. Overrides the built-in default layout when provided.')
argparser.add_argument('--learning_rate', type=float, default=1e-3)
argparser.add_argument('--lr_cosine_decay', type=str2bool, default=False,
                       help='Enable cosine LR decay with T=100 epochs.')
argparser.add_argument('--lr_min', type=float, default=1e-5,
                       help='Minimum LR for cosine decay.')
argparser.add_argument('--num_epochs', type=int, default=300)
argparser.add_argument('--num_accum_steps', type=int, default=4)
argparser.add_argument('--truncate_steps', type=int, default=50)
argparser.add_argument("--truncate_kl", type=str2bool, default=False)
argparser.add_argument('--gumbel_temp', type=float, default=1.0)
# argparser.add_argument('--gumbel_temp', type=float, default=0.5)
argparser.add_argument('--gradnorm_clip', type=float, default=1)
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--name', type=str, default='run_mrl_base')
argparser.add_argument('--reward_type', type=str, choices=['utrlm', 'utrlm_te', 'rnafm', 'framepool'], default='utrlm')
argparser.add_argument('--gradient_type', type=str, choices=['base_soft', 'motif_soft'], default='base_soft',
                       help='Use motif-level straight-through gradients and map to base space outside when set to motif_soft.')
argparser.add_argument('--total_num_steps', type=int, default=128)
argparser.add_argument('--copy_flag_temp', type=float, default=None)
argparser.add_argument('--save_every_n_epochs', type=int, default=50)
argparser.add_argument('--alpha', type=float, default=0.001)
argparser.add_argument('--beta', type=float, default=0.001,
                       help='Coefficient for forward KL regularization KL(old||new).')
argparser.add_argument('--js', type=str2bool, default=False,
                       help='If true, replace reverse/forward KL regularization with a single JS divergence term weighted by alpha. Beta and forward_kl_on_old_xt are ignored.')
argparser.add_argument('--forward_kl_on_old_xt', type=str2bool, default=True,
                       help='If true, evaluate KL(old||new) on x_t sampled from old_model instead of new_model.')
# argparser.add_argument('--alpha', type=float, default=0.005)
argparser.add_argument('--sft_reg_coeff', type=float, default=0,
                       help='Coefficient for supervised diffusion regularization on pretraining data.')
argparser.add_argument('--sft_dataset_csv', type=str,
                       default='/home/xli263/xli/utr_design/DRAKES/drakes_rna/data_and_model/train_dataset.csv',
                       help='CSV path for SFT-style regularization data.')
argparser.add_argument('--sft_seq_col', type=str, default='utr',
                       help='Sequence column name in the SFT CSV.')
argparser.add_argument('--sft_batch_size', type=int, default=32,
                       help='Batch size for SFT regularization loader.')
argparser.add_argument('--sft_num_workers', type=int, default=0,
                       help='DataLoader workers for SFT regularization loader.')
argparser.add_argument('--alpha_schedule_warmup', type=int, default=0)
argparser.add_argument('--kl_coeff_schedule_warmup', type=int, default=0,
                       help='Warmup steps for KL coefficient (alpha). Overrides epoch warmup when > 0.')
argparser.add_argument('--entropy', type=str2bool, default=True,
                       help='Enable entropy bonus term in the loss.')
argparser.add_argument('--entropy_coeff', type=float, default=1e-5,
                       help='Coefficient for entropy bonus.')
argparser.add_argument('--entropy_coeff_schedule_warmup', type=int, default=0,
                       help='Warmup steps for entropy coefficient.')
argparser.add_argument("--seed", type=int, default=9)
argparser.add_argument('--grad_skip_threshold', type=float, default=1000.0)
argparser.add_argument('--base_comp_coeff', type=float, default=0.0,
                       help='Coefficient for A/C/G/T composition constraint. 0 disables it.')
argparser.add_argument('--base_comp_loss_type', type=str, choices=['kl', 'l2'], default='kl',
                       help='Loss type for base composition constraint.')
argparser.add_argument('--natural_base_probs', type=str, default='0.312771,0.216887,0.249919,0.220423',
                       help='Natural A,C,G,T base distribution, comma-separated.')
argparser.add_argument('--save_best_metric', type=str, choices=['reward', 'reward_eval'], default='reward_eval',
                       help='Metric used to track and save the best checkpoint.')
argparser.add_argument('--utrlm_checkpoint_root', type=str, default="/home/xli263/xli/utr_design/UTR-LM/Model/Downstream/MRL",
                       help='Path to UTR-LM TE checkpoint directory.')
argparser.add_argument('--pretrained_ckpt_path', type=str, default=None,
                       help='Optional path to the pretrained diffusion checkpoint. Overrides the built-in default when provided.')
argparser.add_argument('--oracle_ckpt_path', type=str,
                       default="/home/xli263/xli/utr_design/DRAKES/drakes_rna/experiment/single_run_negative_samples_cap_6.5/negative_cap_best.ckpt",
                       help='Single UTR oracle checkpoint path for reward guidance. Supports manual hybrid checkpoints.')
argparser.add_argument('--utrlm_seq_trim_len', type=int, default=100,
                       help='Sequence length used by UTR-LM TE predictors.')
argparser.add_argument('--rnafm_predictor_checkpoint', type=str,
                       default="/home/xli263/xli/utr_design/RNA-FM/tutorials/utr-function-prediction/result/CNN_emb-rnafm50nt_best_utr_predictor.pth",
                       help='Path to the RNA-FM UTR predictor checkpoint.')
argparser.add_argument('--rnafm_backbone_path', type=str, default=None,
                       help='Optional path to RNA-FM pretrained .pth file.')
argparser.add_argument('--rnafm_fm_root', type=str, default="/home/xli263/xli/utr_design/RNA-FM",
                       help='Path to RNA-FM repo root (for local import fallback).')
argparser.add_argument('--rnafm_seq_trim_len', type=int, default=100,
                       help='Sequence length used by RNA-FM oracle.')
argparser.add_argument('--framepool_model_path', type=str,
                       default='/home/xli263/xli/utr_design/UTRGAN/models/utr_model_combined_residual_new.h5',
                       help='Path to FramePool .h5 weights file.')
argparser.add_argument('--framepool_max_len', type=int, default=128,
                       help='Left-pad to this length for FramePool input (<=0 disables padding).')
argparser.add_argument('--use_motif', action='store_true', help='Whether to use motif-based input representation for reward model.')

argparser.add_argument('--target_len', type=int, default=51,
                       help='Target length for UTR sequences')
argparser.add_argument(
  '--token_length_distribution',
  type=str,
  default='/home/xli263/xli/utr_design/DRAKES/drakes_rna/data_and_model/token_length_distribution_50_utr_4_base.txt',
  help='Path to token length distribution file for sampling target lengths.'
)
argparser.add_argument('--length_distribution_out', type=str, default=None,
                       help='Optional output path for the final length distribution.')
argparser.add_argument('--log_motif_stats', type=str2bool, default=False,
                       help='Log motif-vocab probability statistics during finetuning.')
argparser.add_argument('--log_motif_stats_every', type=int, default=50,
                       help='Log motif stats every N steps when enabled.')
argparser.add_argument('--log_denoise_trajectory', type=str2bool, default=False,
                       help='Print decoded sequence trajectory along denoising steps.')
argparser.add_argument('--trajectory_every', type=int, default=1,
                       help='Log trajectory every N optimization steps.')
argparser.add_argument('--trajectory_max_steps', type=int, default=128,
                       help='Max denoising timesteps to print per logged batch; <=0 means all steps.')
argparser.add_argument('--trajectory_step_stride', type=int, default=1,
                       help='Only print timesteps where t % stride == 0.')
argparser.add_argument('--trajectory_sample_index', type=int, default=0,
                       help='Batch index to print for trajectory logs.')
argparser.add_argument('--ood_enable', type=str2bool, default=True,
                       help='Enable OOD score computation and reward hard-gating.')
argparser.add_argument('--ood_last_k_steps', type=int, default=5,
                       help='Use last K denoising steps for OOD score. <=0 means use truncate_steps.')
argparser.add_argument('--ood_score_threshold', type=float, default=3.0,
                       help='Fixed OOD score threshold (eps_ood). Required when --ood_enable=true.')
argparser.add_argument('--ood_r_min', type=float, default=0,
                       help='Pessimistic reward used for hard-gated OOD samples.')
argparser.add_argument('--mask_base_ce_coeff', type=float, default=0.0,
                       help='Coefficient for masked-position base distribution CE(old||new) regularization.')
argparser.add_argument('--mask_base_agg_method', type=str, choices=['global', 'position'], default='global',
                       help='Masked-base divergence aggregation: global mean probs first, or per-position.')
argparser.add_argument('--mask_base_divergence', type=str, choices=['ce', 'kl'], default='ce',
                       help='Masked-base divergence type: CE(old||new) or KL(old||new).')
argparser.add_argument('--mask_base_truncate_steps', type=int, default=50,
                       help='If >0, compute/average masked-base CE/KL over only the last K denoising steps. 0 means all steps.')
argparser.add_argument('--enable_drift_monitor', type=str2bool, default=False,
                       help='Log rollout drift monitors against old_model during finetuning.')
argparser.add_argument('--drift_monitor_every', type=int, default=1,
                       help='Compute drift monitors every N accumulation steps.')
argparser.add_argument('--drift_kmer_k', type=int, default=3,
                       help='k for k-mer L1 distance in sequence drift monitor.')
argparser.add_argument('--eval_kmer_reference_csv', type=str, default=None,
                       help='Optional reference CSV used for epoch-level generated-vs-reference k-mer correlation logging.')
argparser.add_argument('--eval_kmer_seq_col', type=str, default='utr',
                       help='Sequence column name in --eval_kmer_reference_csv.')
argparser.add_argument('--eval_kmer_k', type=int, default=3,
                       help='k used for epoch-level generated-vs-reference k-mer correlation logging.')
                       
args = argparser.parse_args()
print(args)

CKPT_PATH = args.pretrained_ckpt_path or '/home/xli263/xli/utr_design/DRAKES/drakes_rna/mdlm/pretrained_utr_ckpt/pretrained_4_base_50nt_no_eos.ckpt'
# log_base_dir = os.path.join(args.base_path, 'mdlm/reward_mrl_optimized_motif_50_utr_target_base_len')
diver_tag = f"js_diver{args.alpha}" if args.js else f"alpha{args.alpha}_beta{args.beta}"
if args.log_base_dir is not None:
  log_base_dir = os.path.join(args.log_base_dir, f"seed{args.seed}")
else:
  log_base_dir = os.path.join(
      args.base_path,
      f"/home/xli263/xli/utr_design/DRAKES/drakes_rna/mdlm/reward_4_base_new_50utr_lr_{args.reward_type}_{args.learning_rate}_{diver_tag}_t{args.truncate_steps}/"
      f"seed{args.seed}_ce_coeff{args.mask_base_ce_coeff}_agg_method{args.mask_base_agg_method}_divergence{args.mask_base_divergence}_t_ce{args.mask_base_truncate_steps}"
  )

GlobalHydra.instance().clear()
initialize(config_path="configs_gosai", job_name="TE_optimization")
cfg = compose(config_name="config_gosai_pretrain")
cfg.eval.checkpoint_path = CKPT_PATH
curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if args.name == 'debug':
  print("Debug mode")
  save_path = os.path.join(log_base_dir, args.name)
  os.makedirs(save_path, exist_ok=True)
  log_path = os.path.join(save_path, f'log_seed{args.seed}_{curr_time}.txt')
  trajectory_log_path = os.path.join(save_path, f'trajectory_seed{args.seed}_{curr_time}.txt')
else:
  run_diver_tag = f'js_diver{args.alpha}' if args.js else f'alpha{args.alpha}_beta{args.beta}'
  run_name = (
    f'{run_diver_tag}_accum{args.num_accum_steps}_bsz{args.batch_size}_'
    f'truncate{args.truncate_steps}_temp{args.gumbel_temp}_clip{args.gradnorm_clip}_{args.name}_{curr_time}')
  save_path = os.path.join(log_base_dir, run_name)
  os.makedirs(save_path, exist_ok=True)
  wandb.init(project='reward_mrl_4_base_ce_new', name=run_name, config=args, dir=save_path)
  log_path = os.path.join(save_path, f'log_seed{args.seed}_{curr_time}.txt')
  trajectory_log_path = os.path.join(save_path, f'trajectory_seed{args.seed}_{curr_time}.txt')

if args.log_denoise_trajectory:
  with open(trajectory_log_path, 'w') as f:
    f.write(args.__repr__() + '\n')
    f.write(f"# total_num_steps={args.total_num_steps} trajectory_max_steps={args.trajectory_max_steps}\n")

set_seed(args.seed, use_cuda=True)

new_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg,strict=False)
old_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg,strict=False)
old_model.eval()
for p in old_model.parameters():
  p.requires_grad = False

eps_ood = None
# OOD initialization/logging disabled for speed.
# if args.ood_enable:
#   if args.ood_score_threshold is None:
#     raise ValueError("--ood_score_threshold is required when --ood_enable=true.")
#   eps_ood = float(args.ood_score_threshold)
#   print(f"[ood] using fixed threshold eps_ood={eps_ood:.6f}")

if args.reward_type == 'utrlm':
  print("Using UTR_Oracle")
  oracle_device = str(getattr(new_model, "device", "cuda"))
  reward_model = UTROracleWrapper(
      oracle_utr.get_utr_oracle(
          map_location=oracle_device,
          oracle_ckpt_path=args.oracle_ckpt_path))
  # reward_model = UTROracleWrapper(
  #   oracle_utr.get_utr_oracle(map_location=oracle_device))
  ckpt = "/home/xli263/xli/utr_design/UTR-LM/Model/Downstream/MRL/MJ3_seed1337_ESM2SISS_FS4.1.ep93.1e-2.dr5_unmod_1_utr_10folds_rl_LabelScalerFalse_LabelLog2False_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_nodes40_dropout30.5_finetuneTrue_huberlossTrue_lr0.01_fold0_epoch299.pt"
  reward_model_eval = oracle_new.get_utrlm_oracle(
      checkpoint_root=args.utrlm_checkpoint_root,
      checkpoint_paths=[ckpt],
      device=oracle_device,
      seq_trim_len=args.utrlm_seq_trim_len)
  # reward_model_eval = reward_model
  # reward_model = reward_model_eval
elif args.reward_type == 'utrlm_te':
  oracle_device = str(getattr(new_model, "device", "cuda"))
  te_ckpt_root = "/home/xli263/xli/utr_design/UTR-LM/Model/Downstream/TE_EL"
  te_ckpt_paths = sorted(
      str(path) for path in Path(te_ckpt_root).glob("*.pt")
      # if ("te_" in path.name.lower())
      if ("pc3" in path.name.lower() and "te_" in path.name.lower())
  )
  if not te_ckpt_paths:
    raise FileNotFoundError(f"No PC3 checkpoints found in {te_ckpt_root}")
  reward_model = UTROracleWrapper(
      oracle_utr.get_utr_oracle(
          map_location=oracle_device,
          oracle_ckpt_path=args.oracle_ckpt_path))
  reward_model_eval = oracle_new.get_utrlm_oracle(
      checkpoint_root=te_ckpt_root,
      checkpoint_paths=te_ckpt_paths,
      device=oracle_device,
      seq_trim_len=args.utrlm_seq_trim_len)
  # reward_model_eval = reward_model
elif args.reward_type == 'rnafm':
  oracle_device = str(getattr(new_model, "device", "cuda"))
  reward_model = oracle_new.get_rnafm_oracle(
      predictor_checkpoint=args.rnafm_predictor_checkpoint,
      backbone_path=args.rnafm_backbone_path,
      fm_root=args.rnafm_fm_root,
      device=oracle_device,
      seq_trim_len=args.rnafm_seq_trim_len)
  # reward_model_eval = reward_model
  ckpt = "/home/xli263/xli/utr_design/UTR-LM/Model/Downstream/MRL/MJ3_seed1337_ESM2SISS_FS4.1.ep93.1e-2.dr5_unmod_1_utr_10folds_rl_LabelScalerFalse_LabelLog2False_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_nodes40_dropout30.5_finetuneTrue_huberlossTrue_lr0.01_fold0_epoch299.pt"
  reward_model_eval = oracle_new.get_utrlm_oracle(
      checkpoint_root=args.utrlm_checkpoint_root,
      checkpoint_paths=[ckpt],
      device=oracle_device,
      seq_trim_len=args.utrlm_seq_trim_len)
elif args.reward_type == 'framepool':
  if str(getattr(new_model, "device", "cuda")) != "cpu":
    print("[framepool] TensorFlow reward executes on CPU bridge; this can be slower than pure torch oracles.")
    oracle_device = str(getattr(new_model, "device", "cuda"))
  reward_model = FramePoolOracleWrapper(
      model_path=args.framepool_model_path,
      max_len=args.framepool_max_len)
  # reward_model_eval = reward_model
  ckpt = "/home/xli263/xli/utr_design/UTR-LM/Model/Downstream/MRL/MJ3_seed1337_ESM2SISS_FS4.1.ep93.1e-2.dr5_unmod_1_utr_10folds_rl_LabelScalerFalse_LabelLog2False_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_nodes40_dropout30.5_finetuneTrue_huberlossTrue_lr0.01_fold0_epoch299.pt"
  reward_model_eval = oracle_new.get_utrlm_oracle(
      checkpoint_root=args.utrlm_checkpoint_root,
      checkpoint_paths=[ckpt],
      device=oracle_device,
      seq_trim_len=args.utrlm_seq_trim_len)
else:
  raise ValueError(f'Unknown reward_type {args.reward_type}')

reward_model.eval()
for p in reward_model.parameters():
  p.requires_grad = False
reward_model_eval.eval()

sft_loader = None
if args.sft_reg_coeff > 0:
  if args.sft_batch_size <= 0:
    args.sft_batch_size = args.batch_size
  sft_loader = _build_sft_dataloader(cfg, args)
  print(
    f"[sft] enabled coeff={args.sft_reg_coeff} "
    f"csv={args.sft_dataset_csv} batch_size={args.sft_batch_size} "
    f"num_workers={args.sft_num_workers}"
  )

# if args.ood_enable and args.name != 'debug':
#   wandb.log({
#     "ood_eps_ood": float(eps_ood),
#     "ood_r_min": float(args.ood_r_min),
#     "ood_last_k_steps": int(_get_ood_last_k_steps(args)),
#   })

fine_tune(new_model, reward_model, reward_model_eval, old_model, args, eps_ood=eps_ood, sft_loader=sft_loader)
