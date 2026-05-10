import argparse
import ast
import datetime
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

import diffusion_gosai_update
import oracle_new
import oracle_utr
from utils import set_seed, str2bool

import math
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



def fine_tune(new_model, new_model_y, new_model_y_eval, old_model, args, eps=1e-5):
  torch.autograd.set_detect_anomaly(True)
  with open(log_path, 'w') as f:
    f.write(args.__repr__() + '\n')

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

  new_model.config.finetuning.truncate_steps = args.truncate_steps
  new_model.config.finetuning.gumbel_softmax_temp = args.gumbel_temp
  dt = (1 - eps) / args.total_num_steps
  new_model.train()
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
  best_metric = float('-inf')
  best_checkpoint_path = os.path.join(save_path, 'best_model.ckpt')
  batch_losses = []
  batch_rewards = []
  target_base_probs = None
  if args.base_comp_coeff > 0:
    target_base_probs = parse_base_probs(args.natural_base_probs).to(new_model.device)

  lambda_atg = 0

  for epoch_num in range(args.num_epochs):
    rewards = []
    rewards_eval = []
    losses = []
    reward_losses = []
    kl_losses = []
    base_comp_losses = []
    entropy_losses = []
    acgt_means = []
    tot_grad_norm = 0.0
    new_model.train()
    skipped_steps = 0
    epoch_sampled_lengths = []
    epoch_reward_1d = []
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
      if args.log_motif_stats and (_step % args.log_motif_stats_every == 0):
        if _step == 0:
          print(f"[motif_stats] log_p_x0_last is None: {log_p_x0_last is None}")
        with torch.no_grad():
          probs = p_vocab if log_p_x0_last is None else log_p_x0_last.exp()
          if probs.dim() == 3:
            probs = probs.clamp_min(1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            mean_entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
            avg_prob = probs.mean(dim=(0, 1))
            avg_prob = avg_prob / avg_prob.sum()
            topk = min(5, avg_prob.numel())
            top_vals, top_idx = torch.topk(avg_prob, k=topk)
            top_pairs = ", ".join(
              f"{int(i)}:{float(v):.4f}" for i, v in zip(top_idx, top_vals)
            )
            msg = (
              f"[motif_stats] epoch {epoch_num} step {_step} "
              f"mean_entropy {mean_entropy:.4f} top{topk} {top_pairs}"
            )
            print(msg)
            with open(log_path, 'a') as f:
              f.write(msg + "\n")
      if args.gradient_type == "motif_soft" and new_model.vocab_size > 7:
        if not hasattr(new_model, "motif2base_stencil"):
          raise ValueError("gradient_type=motif_soft requires motif2base_stencil on the model.")
        base_probs = _motif_sample_to_base(sample, new_model)
      else:
        base_probs = sample
      base_probs = base_probs.transpose(1, 2)


      use_soft = args.reward_type in ('utrlm', 'rnafm')
      preds = new_model_y(base_probs, soft_input=use_soft)
      reward = preds[..., 0]

      with torch.no_grad():
        preds_eval = new_model_y_eval(base_probs, soft_input=False)


      reward_argmax_eval = preds_eval[..., 0]
      rewards_eval.append(reward_argmax_eval.detach().mean().cpu().item())
      rewards.append(reward.detach().mean().cpu().item())

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
      length_mask = None
      if target_length is not None:
        if isinstance(target_length, torch.Tensor):
          target_len = target_length.to(new_model.device, dtype=torch.long)
        else:
          target_len = torch.tensor(target_length, device=new_model.device, dtype=torch.long)
        seq_len = last_x_list[0].shape[1]
        target_len = target_len.clamp(min=0, max=seq_len)
        seq_idx = torch.arange(seq_len, device=new_model.device).unsqueeze(0)
        # Mask valid tokens before the target length; EOS/PAD tails are excluded.
        length_mask = (seq_idx < target_len.unsqueeze(1)).unsqueeze(-1).to(torch.float32)
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

        p_x0 = log_p_x0.exp()
        p_x0_old = log_p_x0_old.exp()
        # print(log_p_x0.requires_grad)
        kl_div = copy_flag * (-p_x0 + p_x0_old + p_x0 * (log_p_x0 - log_p_x0_old)) / move_chance_t[0, 0, 0]
        if length_mask is not None:
          kl_div = kl_div * length_mask
        kl_div = (kl_div * last_x[:, :, :-1]).sum((1, 2))
        total_kl.append(kl_div)

      current_step = epoch_num * args.num_accum_steps + _step + 1
      if args.kl_coeff_schedule_warmup and current_step < args.kl_coeff_schedule_warmup:
        current_alpha = current_step / args.kl_coeff_schedule_warmup * args.alpha
      elif args.alpha_schedule_warmup and epoch_num < args.alpha_schedule_warmup:
        current_alpha = (epoch_num + 1) / args.alpha_schedule_warmup * args.alpha
      else:
        current_alpha = args.alpha

      kl_loss = torch.stack(total_kl, 1).sum(1).mean()
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
      if probs is None:
        if log_p_x0_last is None:
          probs = p_vocab
        else:
          probs = log_p_x0_last.exp()
        probs = probs.clamp_min(1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
      acgt_mean = probs[:, :, :4].mean()
      # if epoch_num > 0 and args.entropy:
      loss = reward_loss + kl_loss * current_alpha + args.base_comp_coeff * base_comp_loss
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
      kl_losses.append(kl_loss.item())
      base_comp_losses.append(float(base_comp_loss))
      entropy_losses.append(float(entropy_loss))
      acgt_means.append(float(acgt_mean))
  
    rewards = torch.tensor(rewards)
    rewards_eval = torch.tensor(rewards_eval)
    losses = torch.tensor(losses)
    reward_losses = torch.tensor(reward_losses)
    kl_losses = torch.tensor(kl_losses)
    base_comp_losses = torch.tensor(base_comp_losses)
    entropy_losses = torch.tensor(entropy_losses)
    acgt_means = torch.tensor(acgt_means)
    # reward_mean_std = rewards.float().std(unbiased=False).item()
    # if length_sampler is not None:
    #   epoch_rewards_tensor = torch.cat(epoch_reward_1d, dim=0) if epoch_reward_1d else torch.tensor([], device=new_model.device)
    #   if len(epoch_sampled_lengths) == len(epoch_rewards_tensor):
    #     length_sampler.update(epoch_sampled_lengths, epoch_rewards_tensor, epoch_num=epoch_num)
    print("Epoch %d" % epoch_num,
          "Mean reward %f" % rewards.float().mean().item(),
          "Mean reward eval %f" % rewards_eval.float().mean().item(),
          "Mean reward std %f" % rewards.float().std(unbiased=False).item(),
          "Mean grad norm %f" % tot_grad_norm,
          "Mean loss %f" % losses.float().mean().item(),
          "Mean reward loss %f" % reward_losses.float().mean().item(),
          "Mean kl loss %f" % kl_losses.float().mean().item(),
          "Mean base comp loss %f" % base_comp_losses.float().mean().item(),
          "Mean entropy loss %f" % entropy_losses.float().mean().item(),
          "Mean ACGT prob %f" % acgt_means.float().mean().item())
    if args.name != 'debug':
      wandb.log({
        "epoch": epoch_num,
        "mean_reward": rewards.float().mean().item(),
        "mean_reward_eval": rewards_eval.float().mean().item(),
        "mean reward std": rewards.float().std(unbiased=False).item(),
        "mean_grad_norm": tot_grad_norm,
        "mean_loss": losses.float().mean().item(),
        "mean_reward_loss": reward_losses.float().mean().item(),
        "mean_kl_loss": kl_losses.float().mean().item(),
        "mean_base_comp_loss": base_comp_losses.float().mean().item(),
        "mean_entropy_loss": entropy_losses.float().mean().item(),
        "mean_acgt_prob": acgt_means.float().mean().item(),
      })
    with open(log_path, 'a') as f:
      f.write(
        f"Epoch {epoch_num} Mean reward {rewards.float().mean().item()} Mean reward eval {rewards_eval.float().mean().item()} "
        f"Mean grad norm {tot_grad_norm} Mean loss {losses.float().mean().item()} "
        f"Mean reward loss {reward_losses.float().mean().item()} Mean kl loss {kl_losses.float().mean().item()} "
        f"Mean base comp loss {base_comp_losses.float().mean().item()} "
        f"Mean entropy loss {entropy_losses.float().mean().item()} "
        f"Mean ACGT prob {acgt_means.float().mean().item()}\n")

    # Track and save best checkpoint by combined metric
    mean_reward = rewards.float().mean().item()
    mean_reward_eval = rewards_eval.float().mean().item()
    metric_value = 0.5 * (mean_reward + mean_reward_eval)
    if metric_value > best_metric:
      best_metric = metric_value
      torch.save(new_model.state_dict(), best_checkpoint_path)
      print(f"New best (avg_reward={best_metric:.4f}) saved to {best_checkpoint_path}")

    # model_path = os.path.join(save_path, f'epoch_{epoch_num}.ckpt')
    # torch.save(new_model.state_dict(), model_path)
    # print(f"Model saved at epoch {epoch_num} -> {model_path}")

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
argparser.add_argument('--learning_rate', type=float, default=1e-3)
argparser.add_argument('--lr_cosine_decay', type=str2bool, default=False,
                       help='Enable cosine LR decay with T=100 epochs.')
argparser.add_argument('--lr_min', type=float, default=1e-5,
                       help='Minimum LR for cosine decay.')
argparser.add_argument('--num_epochs', type=int, default=100)
argparser.add_argument('--num_accum_steps', type=int, default=4)
argparser.add_argument('--truncate_steps', type=int, default=50)
argparser.add_argument("--truncate_kl", type=str2bool, default=False)
argparser.add_argument('--gumbel_temp', type=float, default=1.0)
# argparser.add_argument('--gumbel_temp', type=float, default=0.5)
argparser.add_argument('--gradnorm_clip', type=float, default=1)
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--name', type=str, default='run_mrl_base')
argparser.add_argument('--reward_type', type=str, choices=['utrlm', 'utrlm_te', 'rnafm'], default='utrlm')
argparser.add_argument('--gradient_type', type=str, choices=['base_soft', 'motif_soft'], default='base_soft',
                       help='Use motif-level straight-through gradients and map to base space outside when set to motif_soft.')
argparser.add_argument('--total_num_steps', type=int, default=128)
argparser.add_argument('--copy_flag_temp', type=float, default=None)
argparser.add_argument('--save_every_n_epochs', type=int, default=100)
argparser.add_argument('--alpha', type=float, default=0.006)
# argparser.add_argument('--alpha', type=float, default=0.005)
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
argparser.add_argument('--base_comp_coeff', type=float, default=0,
                       help='Coefficient for A/C/G/T composition constraint. 0 disables it.')
argparser.add_argument('--base_comp_loss_type', type=str, choices=['kl', 'l2'], default='kl',
                       help='Loss type for base composition constraint.')
argparser.add_argument('--natural_base_probs', type=str, default='0.312771,0.216887,0.249919,0.220423',
                       help='Natural A,C,G,T base distribution, comma-separated.')
argparser.add_argument('--save_best_metric', type=str, choices=['reward', 'reward_eval'], default='reward_eval',
                       help='Metric used to track and save the best checkpoint.')
argparser.add_argument('--utrlm_checkpoint_root', type=str, default="/home/xli263/xli/utr_design/UTR-LM/Model/Downstream/MRL",
                       help='Path to UTR-LM TE checkpoint directory.')
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
                       
args = argparser.parse_args()
print(args)

CKPT_PATH = os.path.join(args.base_path, '/home/xli263/xli/utr_design/DRAKES/drakes_rna/mdlm/pretrained_utr_ckpt/pretrained_4_base_new.ckpt')
# log_base_dir = os.path.join(args.base_path, 'mdlm/reward_mrl_optimized_motif_50_utr_target_base_len')
log_base_dir = os.path.join(
    args.base_path,
    f"/home/xli263/xli/utr_design/DRAKES/drakes_rna/mdlm/reward_4_base_50utr/"
    f"lr{args.learning_rate}_alpha{args.alpha}_t{args.truncate_steps}_base_comp{args.base_comp_coeff}_seed{args.seed}"
)

GlobalHydra.instance().clear()
initialize(config_path="configs_gosai", job_name="MRL_optimization")
cfg = compose(config_name="config_gosai.yaml")
cfg.eval.checkpoint_path = CKPT_PATH
curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if args.name == 'debug':
  print("Debug mode")
  save_path = os.path.join(log_base_dir, args.name)
  os.makedirs(save_path, exist_ok=True)
  log_path = os.path.join(save_path, f'log_seed{args.seed}_{curr_time}.txt')
else:
  run_name = (
    f'alpha{args.alpha}_accum{args.num_accum_steps}_bsz{args.batch_size}_'
    f'truncate{args.truncate_steps}_temp{args.gumbel_temp}_clip{args.gradnorm_clip}_{args.name}_{curr_time}')
  save_path = os.path.join(log_base_dir, run_name)
  os.makedirs(save_path, exist_ok=True)
  wandb.init(project='reward_mrl_4_base_uncertainty', name=run_name, config=args, dir=save_path)
  log_path = os.path.join(save_path, f'log_seed{args.seed}_{curr_time}.txt')

set_seed(args.seed, use_cuda=True)

new_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg,strict=False)
old_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg,strict=False)

if args.reward_type == 'utrlm':
  print("Using UTR_Oracle")
  oracle_device = str(getattr(new_model, "device", "cuda"))
  reward_model = UTROracleWrapper(
      oracle_utr.get_utr_oracle(map_location=oracle_device))
  ckpt = "/home/xli263/xli/utr_design/UTR-LM/Model/Downstream/MRL/MJ3_seed1337_ESM2SISS_FS4.1.ep93.1e-2.dr5_unmod_1_utr_10folds_rl_LabelScalerFalse_LabelLog2False_AvgEmbFalse_BosEmbTrue_CNNlayer0_epoch300_nodes40_dropout30.5_finetuneTrue_huberlossTrue_lr0.01_fold0_epoch299.pt"
  reward_model_eval = oracle_new.get_utrlm_oracle(
      checkpoint_root=args.utrlm_checkpoint_root,
      checkpoint_paths=[ckpt],
      device=oracle_device,
      seq_trim_len=args.utrlm_seq_trim_len)
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
      oracle_utr.get_utr_oracle(map_location=oracle_device))
  reward_model_eval = oracle_new.get_utrlm_oracle(
      checkpoint_root=te_ckpt_root,
      checkpoint_paths=te_ckpt_paths,
      device=oracle_device,
      seq_trim_len=args.utrlm_seq_trim_len)
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
else:
  raise ValueError(f'Unknown reward_type {args.reward_type}')

reward_model.eval()
for p in reward_model.parameters():
  p.requires_grad = False
reward_model_eval.eval()

fine_tune(new_model, reward_model, reward_model_eval, old_model, args)
