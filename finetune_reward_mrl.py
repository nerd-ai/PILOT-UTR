import argparse
import ast
import datetime
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

import diffusion_gosai_update
import oracle_new
import oracle_utr
from utils import set_seed, str2bool


BASES = ("A", "C", "G", "T")


class UTROracleWrapper(torch.nn.Module):
  """Adapter for GReLU/Enformer-style UTR reward checkpoints."""

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
  lengths = sorted(int(k) for k in dist.keys())
  probs = torch.tensor([float(dist[L]) for L in lengths], dtype=torch.float32)
  probs_sum = probs.sum().item()
  if probs_sum <= 0:
    raise ValueError(f"Token length distribution has non-positive total probability: {path}")
  return lengths, probs / probs_sum


def save_token_length_distribution(path: str, lengths, probs):
  probs = probs.detach().cpu().tolist()
  dist = {int(length): float(prob) for length, prob in zip(lengths, probs)}
  with open(path, "w") as handle:
    handle.write(str(dist) + "\n")


def sample_target_lengths(length_values, length_probs, batch_size, device):
  idx = torch.multinomial(length_probs.to(device), batch_size, replacement=True)
  return torch.tensor(
    [int(length_values[int(i)]) for i in idx.detach().cpu().tolist()],
    device=device,
    dtype=torch.long,
  )


def _get_valid_lens(target_length, batch_size, max_len):
  if target_length is None:
    return [int(max_len)] * int(batch_size)
  if isinstance(target_length, torch.Tensor):
    vals = target_length.detach().cpu().view(-1).tolist()
  else:
    vals = list(target_length)
  return [int(max(0, min(int(v), int(max_len)))) for v in vals]


def _decode_base_sequence(base_probs_row, valid_len=None):
  if valid_len is None:
    valid_len = base_probs_row.shape[0]
  valid_len = int(max(0, min(valid_len, base_probs_row.shape[0])))
  idx = base_probs_row[:valid_len].argmax(dim=-1).detach().cpu().tolist()
  return "".join(BASES[int(i)] for i in idx)


def _sample_to_base_probs(sample, model):
  del model
  if sample.dim() != 3 or sample.shape[-1] < 4:
    raise ValueError(f"Expected base-soft sample [B, L, >=4], got {tuple(sample.shape)}")
  base_probs = sample[:, :, :4]
  return base_probs / base_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _kmer_freq_vector(seqs, k: int = 3):
  counts = {}
  total = 0
  for seq in seqs:
    seq = str(seq).upper()
    if len(seq) < k:
      continue
    for i in range(len(seq) - k + 1):
      kmer = seq[i:i + k]
      if all(base in "ACGT" for base in kmer):
        counts[kmer] = counts.get(kmer, 0) + 1
        total += 1
  kmers = ["".join(chars) for chars in np.array(np.meshgrid(*(["ACGT"] * k))).T.reshape(-1, k)]
  vec = np.array([counts.get(kmer, 0) for kmer in kmers], dtype=np.float64)
  if total > 0:
    vec /= float(total)
  return kmers, vec


def _pearson_corr(x_vals, y_vals) -> float:
  x = np.asarray(x_vals, dtype=np.float64)
  y = np.asarray(y_vals, dtype=np.float64)
  if x.shape != y.shape or x.size == 0:
    return float("nan")
  x = x - x.mean()
  y = y - y.mean()
  denom = np.sqrt((x * x).sum() * (y * y).sum())
  if denom == 0:
    return float("nan")
  return float((x * y).sum() / denom)


def _load_csv_sequences(csv_path: str, seq_col: str = "utr"):
  import pandas as pd

  df = pd.read_csv(csv_path)
  if seq_col not in df.columns:
    raise ValueError(f"Missing sequence column '{seq_col}' in {csv_path}")
  return df[seq_col].dropna().astype(str).tolist()


def _build_length_mask(target_length, device, seq_len, dtype=torch.float32):
  if target_length is None:
    return None
  if isinstance(target_length, torch.Tensor):
    target_len = target_length.to(device=device, dtype=torch.long).view(-1)
  else:
    target_len = torch.tensor(target_length, device=device, dtype=torch.long).view(-1)
  target_len = target_len.clamp(min=0, max=seq_len)
  seq_idx = torch.arange(seq_len, device=device).unsqueeze(0)
  return (seq_idx < target_len.unsqueeze(1)).to(dtype)


def configure_output(args):
  diver_tag = f"alpha{args.alpha}_beta{args.beta}"
  if args.log_base_dir:
    log_base_dir = os.path.join(args.log_base_dir, f"seed{args.seed}")
  else:
    log_base_dir = os.path.join(
      args.output_dir,
      f"reward_mrl_{args.learning_rate}_{diver_tag}_t{args.truncate_steps}",
      f"seed{args.seed}",
    )

  curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  run_name = (
    f"{diver_tag}_accum{args.num_accum_steps}_bsz{args.batch_size}_"
    f"truncate{args.truncate_steps}_temp{args.gumbel_temp}_clip{args.gradnorm_clip}_"
    f"{args.name}_{curr_time}"
  )
  save_path = os.path.join(log_base_dir, run_name)
  os.makedirs(save_path, exist_ok=True)
  log_path = os.path.join(save_path, f"log_seed{args.seed}_{curr_time}.txt")
  return save_path, log_path, run_name


def load_diffusion_models(args):
  GlobalHydra.instance().clear()
  initialize(config_path="configs_gosai", job_name="PILOT_UTR_MRL_finetune")
  cfg = compose(config_name=args.config_name)
  cfg.eval.checkpoint_path = args.pretrained_ckpt_path

  new_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(
    cfg.eval.checkpoint_path, config=cfg, strict=False)
  old_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(
    cfg.eval.checkpoint_path, config=cfg, strict=False)
  old_model.eval()
  for param in old_model.parameters():
    param.requires_grad = False
  return cfg, new_model, old_model


def load_reward_models(args, device):
  if args.reward_type != "Enformer_UTRLM":
    raise ValueError("finetune_reward_mrl.py supports only --reward_type Enformer_UTRLM")

  reward_model = UTROracleWrapper(
    oracle_utr.get_utr_oracle(
      map_location=str(device),
      oracle_ckpt_path=args.oracle_ckpt_path,
    )
  )
  reward_model_eval = oracle_new.get_utrlm_oracle(
    checkpoint_root=args.utrlm_checkpoint_root,
    checkpoint_paths=[args.utrlm_eval_checkpoint_path],
    device=str(device),
    seq_trim_len=args.utrlm_seq_trim_len,
  )
  reward_model.eval()
  reward_model_eval.eval()
  for param in reward_model.parameters():
    param.requires_grad = False
  for param in reward_model_eval.parameters():
    param.requires_grad = False
  return reward_model, reward_model_eval


def finetune(new_model, old_model, reward_model, reward_model_eval, args, save_path, log_path):
  with open(log_path, "w") as handle:
    handle.write(repr(args) + "\n")

  length_values = None
  length_probs = None
  if args.token_length_distribution:
    length_values, length_probs = load_token_length_distribution(args.token_length_distribution)

  ref_kmer_freq = None
  if args.eval_kmer_reference_csv:
    ref_sequences = _load_csv_sequences(args.eval_kmer_reference_csv, seq_col=args.eval_kmer_seq_col)
    _, ref_kmer_freq = _kmer_freq_vector(ref_sequences, k=args.eval_kmer_k)
    print(
      f"[eval_kmer] loaded {len(ref_sequences)} reference sequences from "
      f"{args.eval_kmer_reference_csv} using k={args.eval_kmer_k}"
    )

  new_model.config.finetuning.truncate_steps = args.truncate_steps
  new_model.config.finetuning.gumbel_softmax_temp = args.gumbel_temp
  new_model.eval()
  torch.set_grad_enabled(True)

  optimizer = torch.optim.Adam(new_model.parameters(), lr=args.learning_rate)
  best_metric = float("inf")
  best_checkpoint_path = os.path.join(save_path, "best_model.ckpt")

  for epoch_num in range(args.num_epochs):
    rewards = []
    rewards_eval = []
    losses = []
    reward_losses = []
    kl_losses = []
    kl_losses_for = []
    entropy_losses = []
    epoch_generated_sequences = []
    total_grad_norm = 0.0
    skipped_steps = 0

    new_model.eval()
    optimizer.zero_grad()

    for accum_step in range(args.num_accum_steps):
      target_length = None
      if length_values is not None:
        target_length = sample_target_lengths(
          length_values, length_probs, args.batch_size, device=new_model.device)

      sample, last_x_list, condt_list, move_chance_t_list, copy_flag_list, kl_x_list, p_vocab, log_p_x0_last = (
        new_model._sample_finetune_gradient(
          eval_sp_size=args.batch_size,
          copy_flag_temp=args.copy_flag_temp,
          target_length=target_length,
          gradient_type="base_soft",
        )
      )

      with torch.no_grad():
        base_probs_metric = _sample_to_base_probs(sample.detach(), new_model)
        valid_lens = _get_valid_lens(
          target_length,
          batch_size=base_probs_metric.shape[0],
          max_len=base_probs_metric.shape[1],
        )
        for batch_i, valid_len in enumerate(valid_lens):
          epoch_generated_sequences.append(
            _decode_base_sequence(base_probs_metric[batch_i], valid_len=valid_len)
          )

      base_probs = _sample_to_base_probs(sample, new_model).transpose(1, 2)
      preds = reward_model(base_probs, soft_input=True)
      reward = preds[..., 0]

      with torch.no_grad():
        preds_eval = reward_model_eval(base_probs, soft_input=False)
      reward_eval = preds_eval[..., 0]

      seq_len = last_x_list[0].shape[1]
      length_mask_2d = _build_length_mask(
        target_length, device=new_model.device, seq_len=seq_len, dtype=torch.float32)
      length_mask = None if length_mask_2d is None else length_mask_2d.unsqueeze(-1)

      total_kl = []
      total_kl_for = []
      for random_t in range(args.total_num_steps):
        if args.truncate_kl and random_t < args.total_num_steps - args.truncate_steps:
          continue

        last_x = kl_x_list[random_t]
        condt = condt_list[random_t]
        move_chance_t = move_chance_t_list[random_t]
        copy_flag = copy_flag_list[random_t]

        log_p_x0 = new_model.forward(last_x, condt)[:, :, :-1]
        log_p_x0_old = old_model.forward(last_x, condt)[:, :, :-1]
        p_x0 = log_p_x0.exp()
        p_x0_old = log_p_x0_old.exp()

        kl_div = copy_flag * (
          -p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)
        ) / move_chance_t[0, 0, 0]
        kl_div_for = copy_flag * (
          -p_x0_old + p_x0 + p_x0_old * (log_p_x0_old - log_p_x0)
        ) / move_chance_t[0, 0, 0]

        if length_mask is not None:
          kl_div = kl_div * length_mask
          kl_div_for = kl_div_for * length_mask

        kl_div = (kl_div * last_x[:, :, :-1]).sum((1, 2))
        kl_div_for = (kl_div_for * last_x[:, :, :-1]).sum((1, 2))
        total_kl.append(kl_div)
        total_kl_for.append(kl_div_for)

      kl_loss = torch.stack(total_kl, dim=1).sum(dim=1).mean()
      kl_loss_for = torch.stack(total_kl_for, dim=1).sum(dim=1).mean()

      if log_p_x0_last is None:
        probs = p_vocab
      else:
        probs = log_p_x0_last.exp()
      probs = probs.clamp_min(1e-8)
      probs = probs / probs.sum(dim=-1, keepdim=True)
      entropy = -(probs * probs.log()).sum(dim=-1).mean()

      reward_loss = -reward.mean()
      loss = (
        reward_loss
        + args.alpha * kl_loss
        + args.beta * kl_loss_for
        - args.entropy_coeff * entropy
      )
      (loss / args.num_accum_steps).backward()

      if (accum_step + 1) % args.num_accum_steps == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), args.gradnorm_clip)
        if grad_norm > args.grad_skip_threshold:
          print(
            f"WARNING: skipped step at epoch {epoch_num}; grad_norm={float(grad_norm):.6f}"
          )
          optimizer.zero_grad()
          skipped_steps += 1
        else:
          total_grad_norm += float(grad_norm)
          optimizer.step()
          optimizer.zero_grad()

      rewards.append(float(reward.detach().mean().cpu()))
      rewards_eval.append(float(reward_eval.detach().mean().cpu()))
      losses.append(float(loss.detach().cpu()))
      reward_losses.append(float(reward_loss.detach().cpu()))
      kl_losses.append(float(kl_loss.detach().cpu()))
      kl_losses_for.append(float(kl_loss_for.detach().cpu()))
      entropy_losses.append(float(entropy.detach().cpu()))

    eval_kmer_corr = None
    if ref_kmer_freq is not None:
      _, gen_kmer_freq = _kmer_freq_vector(epoch_generated_sequences, k=args.eval_kmer_k)
      eval_kmer_corr = _pearson_corr(gen_kmer_freq, ref_kmer_freq)

    metrics = {
      "epoch": epoch_num,
      "mean_reward": float(np.mean(rewards)),
      "mean_reward_eval": float(np.mean(rewards_eval)),
      "mean_grad_norm": total_grad_norm,
      "mean_loss": float(np.mean(losses)),
      "mean_reward_loss": float(np.mean(reward_losses)),
      "mean_kl_loss": float(np.mean(kl_losses)),
      "mean_kl_loss_for": float(np.mean(kl_losses_for)),
      "mean_entropy": float(np.mean(entropy_losses)),
      "skipped_steps": skipped_steps,
    }
    if eval_kmer_corr is not None:
      metrics["eval_kmer_corr"] = eval_kmer_corr

    print_items = [
      f"Epoch {epoch_num}",
      f"Mean reward {metrics['mean_reward']:.6f}",
      f"Mean reward eval {metrics['mean_reward_eval']:.6f}",
      f"Mean grad norm {metrics['mean_grad_norm']:.6f}",
      f"Mean loss {metrics['mean_loss']:.6f}",
      f"Mean reward loss {metrics['mean_reward_loss']:.6f}",
      f"Mean kl loss {metrics['mean_kl_loss']:.6f}",
      f"Mean kl loss for {metrics['mean_kl_loss_for']:.6f}",
      f"Mean entropy {metrics['mean_entropy']:.6f}",
    ]
    if eval_kmer_corr is not None:
      print_items.append(f"Eval {args.eval_kmer_k}-mer corr {eval_kmer_corr:.6f}")
    print(*print_items)

    with open(log_path, "a") as handle:
      handle.write(" ".join(print_items) + "\n")

    if args.wandb:
      wandb.log(metrics)

    if metrics["mean_loss"] < best_metric:
      best_metric = metrics["mean_loss"]
      torch.save(new_model.state_dict(), best_checkpoint_path)
      print(f"New best (mean_loss={best_metric:.6f}) saved to {best_checkpoint_path}")

    if (epoch_num + 1) % args.save_every_n_epochs == 0:
      model_path = os.path.join(save_path, f"model_{epoch_num}.ckpt")
      torch.save(new_model.state_dict(), model_path)
      print(f"Model saved at epoch {epoch_num}")

  if length_values is not None:
    out_path = args.length_distribution_out or os.path.join(save_path, "length_distribution.txt")
    save_token_length_distribution(out_path, length_values, length_probs)


def parse_args():
  parser = argparse.ArgumentParser(
    description="PILOT-UTR MRL reward-guided diffusion finetuning."
  )
  parser.add_argument("--config_name", default="config_gosai_pretrain")
  parser.add_argument("--pretrained_ckpt_path", required=True)
  parser.add_argument("--oracle_ckpt_path", required=True)
  parser.add_argument("--utrlm_eval_checkpoint_path", required=True)
  parser.add_argument("--utrlm_checkpoint_root", default="")
  parser.add_argument("--utrlm_seq_trim_len", type=int, default=100)
  parser.add_argument("--reward_type", choices=["Enformer_UTRLM"], default="Enformer_UTRLM")
  parser.add_argument("--token_length_distribution", default=None)
  parser.add_argument("--length_distribution_out", default=None)
  parser.add_argument("--eval_kmer_reference_csv", default=None)
  parser.add_argument("--eval_kmer_seq_col", default="utr")
  parser.add_argument("--eval_kmer_k", type=int, default=3)
  parser.add_argument("--output_dir", default="outputs")
  parser.add_argument("--log_base_dir", default=None)
  parser.add_argument("--name", default="pilot_utr_mrl")
  parser.add_argument("--seed", type=int, default=9)
  parser.add_argument("--num_epochs", type=int, default=500)
  parser.add_argument("--num_accum_steps", type=int, default=4)
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--truncate_steps", type=int, default=50)
  parser.add_argument("--truncate_kl", type=str2bool, default=False)
  parser.add_argument("--total_num_steps", type=int, default=128)
  parser.add_argument("--learning_rate", type=float, default=1e-3)
  parser.add_argument("--gumbel_temp", type=float, default=1.0)
  parser.add_argument("--gradnorm_clip", type=float, default=1.0)
  parser.add_argument("--grad_skip_threshold", type=float, default=1000.0)
  parser.add_argument("--alpha", type=float, default=0.0015)
  parser.add_argument("--beta", type=float, default=0.0015)
  parser.add_argument("--entropy_coeff", type=float, default=1e-5)
  parser.add_argument("--copy_flag_temp", type=float, default=None)
  parser.add_argument("--save_every_n_epochs", type=int, default=50)
  parser.add_argument("--wandb", type=str2bool, default=True)
  return parser.parse_args()


def main():
  args = parse_args()
  save_path, log_path, run_name = configure_output(args)

  if args.wandb:
    wandb.init(project="PILOT-UTR", name=run_name, config=args, dir=save_path)

  set_seed(args.seed, use_cuda=True)
  _, new_model, old_model = load_diffusion_models(args)
  reward_model, reward_model_eval = load_reward_models(args, device=new_model.device)
  finetune(new_model, old_model, reward_model, reward_model_eval, args, save_path, log_path)

  if args.wandb:
    wandb.finish()


if __name__ == "__main__":
  main()
