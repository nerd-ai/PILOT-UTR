import itertools
import math
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor

import dataloader_gosai
import models
import noise_schedule
import utils
import oracle
from scipy.stats import wasserstein_distance, pearsonr

import json
LOG2 = math.log(2)
LOGGER = utils.get_logger(__name__)


def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)

# def _sample_categorical_gradient(categorical_probs, temp = 1.0):
#   gumbel_norm = (
#     1e-10
#     - (torch.rand_like(categorical_probs) + 1e-10).log())
#   output = torch.nn.functional.softmax((torch.log(categorical_probs)-torch.log(gumbel_norm))/temp, 2)
#   return output
def _sample_categorical_gradient(categorical_probs, temp):
    eps = 1e-20

    # Ensure it's a valid probability distribution
    categorical_probs = categorical_probs.clamp(min=eps)
    categorical_probs = categorical_probs / categorical_probs.sum(dim=-1, keepdim=True)

    # Sample Gumbel noise
    u = torch.rand_like(categorical_probs).clamp(min=eps, max=1.0 - eps)
    gumbel = -torch.log(-torch.log(u))  # Gumbel(0,1)

    # Standard Gumbel-softmax
    logits = (categorical_probs.log() + gumbel) / temp
    output = torch.nn.functional.softmax(logits, dim=-1)
    return output


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    eval=False):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.pad_token_id = self.config.data.pad_token_id
    self.eos_token_id = getattr(self.config.data, 'eos_token_id', None)
    self.vocab_size = self.config.data.vocab_size
    self.mask_index = self.vocab_size
    self.vocab_size += 1
    self.sampler = self.config.sampling.predictor
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    self.parameterization = self.config.parameterization
    ## initialize motif stencil if using motif-aware tokenizer
    tokenizer_type = getattr(self.config.data, 'tokenizer_type', None)
    with open(self.config.data.motif_vocab_path, "r") as f:
      token2id = json.load(f)  # {"A":0, "C":1, ..., "N":64, "EOS":65}

    self.token2id = token2id
    # Build id -> token list
    id2token = [None] * (self.vocab_size-1)
    for tok, idx in token2id.items():
        id2token[idx] = tok
    self.id2token = id2token
    
    if tokenizer_type == 'csv_motif':
      # Get path from config (fallback to tokenizer_vocab_path if motif_vocab_path missing)
      stencil = self._build_motif2base_stencil()
      self.register_buffer('motif2base_stencil', stencil)
      with torch.no_grad():
        nonzero = (stencil.abs().sum(dim=-1) > 0)          # [66, k_max]
        motif_lengths = nonzero.sum(dim=-1).to(torch.long) # [66]
      self.register_buffer("motif_lengths", motif_lengths)



    if self.config.backbone == 'cnn':
      self.backbone = models.dnaconv.CNNModel(
        self.config.model, alphabet_size=self.vocab_size, num_cls=3) # num_cls is not used since classifier is always set to False
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')
    self.default_target_length = getattr(self.config.sampling,
                                        'target_length',
                                        self.config.model.length)
    if (self.pad_token_id is not None and
        not (0 <= self.pad_token_id < self.config.data.vocab_size)):
        raise ValueError(
            f'pad_token_id {self.pad_token_id} must be < vocab_size '
            f'{self.config.data.vocab_size}.')
    self._active_mask = None  # used during sampling to clamp pads
    self.T = self.config.T
    self.subs_masking = self.config.subs_masking

    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

    # subset of data for evaluation
    if eval:
      self.eval_sets_sp = oracle.subset_for_eval(n=config.eval.subset_size) 
      self.eval_sets_sp_clss = oracle.subset_eval_groundtruth(self.eval_sets_sp)
      self.eval_sets_sp_preds = oracle.subset_eval_preds(self.eval_sets_sp) 
      self.eval_sets_sp_kmers = oracle.subset_eval_kmers(self.eval_sets_sp) 
      self.emb_pca = oracle.cal_emb_pca(oracle.subset_for_eval(n=40000), n_components=50)
      self.eval_sets_sp_embs_pca = oracle.subset_eval_embs_pca(self.eval_sets_sp, self.emb_pca) 



  def _is_motif_token(self, token: str) -> bool:
    return all(ch in "ACGT" for ch in token)

  def _build_motif2base_stencil(self) -> torch.Tensor:
    vocab = self.id2token
    V_full = len(vocab)
    max_len = 0
    for v, tok in enumerate(vocab):
        if tok in ("N", "EOS"):
            continue
        if self._is_motif_token(tok):
            max_len = max(max_len, len(tok))
    if max_len == 0:
        raise ValueError("No motif tokens found in the vocabulary.")
    k_max = max_len
    stencil = torch.zeros(V_full, k_max, 4, dtype=torch.float32)
    base2idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    # 3. Fill motif rows
    for v, tok in enumerate(vocab):
        # leave N/EOS as zero rows
        if tok in ("N", "EOS"):
            continue
        if not self._is_motif_token(tok):
            # Unknown specials -> leave zeros too
            continue
        for r, ch in enumerate(tok):
          if r >= k_max:
              break
          stencil[v, r, base2idx[ch]] = 1.0
    return stencil

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    assert self.parameterization == 'subs'

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    
    print('distributed:', distributed)
    # TODO: need to check these two functions
    if distributed:
      sampler_cls = dataloader_gosai.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader_gosai.RandomFaultTolerantSampler
    
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _subs_parameterization(self, logits, xt):
    logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    if xt.ndim > 2 and xt.shape[-1] == self.vocab_size:
      # this is for finetuning setting when the input is one-hot encoded or probs
      xt = xt.argmax(dim=-1)
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma):
    """Returns log score."""
    sigma = self._process_sigma(sigma)

    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, sigma)
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    return logits

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(batch['seqs'], attention_mask)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  def on_validation_epoch_end(self):
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
      all_samples, all_detoeknized_samples = [], []
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples = self._sample().detach().cpu().numpy()
        detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples)
        all_samples.append(samples)
        all_detoeknized_samples.extend(detokenized_samples)
      all_samples = np.concatenate(all_samples, axis=0)
      ws_distance_dict = self.cal_wasserstein_distance(all_detoeknized_samples)
      pearsonr_list = self.cal_kmer_pearsonr(all_detoeknized_samples)
      ws_embpca_list = self.cal_ws_distance_embpca(all_detoeknized_samples)
      
      current_step = self.trainer.global_step
      LOGGER.info(f'Current step: {current_step}')
      LOGGER.info(f'Wasserstein distance: {ws_distance_dict}')
      LOGGER.info(f'3mer Pearsonr: {pearsonr_list}')
      LOGGER.info(f'Wasserstein distance embpca: {ws_embpca_list}')
      self.log('val/3mer_pearsonr', pearsonr_list, on_step=False, on_epoch=True, sync_dist=True)
      self.log('val/ws_embpca', ws_embpca_list, on_step=False, on_epoch=True, sync_dist=True)

      for key in ws_distance_dict:
        for cell_type in ws_distance_dict[key]:
          metric_values = ws_distance_dict[key][cell_type]
          if metric_values:  # Check if the list is not empty
              # Assuming metric_values contains [train_metric, valid_metric, test_metric]
              self.log(f'val/{key}_{cell_type}', metric_values[0], on_step=False, on_epoch=True, sync_dist=True)

    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))
      
  def cal_wasserstein_distance(self, seqs):
    generated_preds = oracle.cal_gosai_pred_new(seqs)
    ws_distance_dict = {'truth': {'hepg2': [], 'k562': [], 'sknsh': []}, 
                        'preds': {'hepg2': [], 'k562': [], 'sknsh': []}} 
    ws_distance_dict['truth']['hepg2'].append(wasserstein_distance(generated_preds[:, 0], self.eval_sets_sp_clss[:, 0]))
    ws_distance_dict['truth']['k562'].append(wasserstein_distance(generated_preds[:, 1], self.eval_sets_sp_clss[:, 1]))
    ws_distance_dict['truth']['sknsh'].append(wasserstein_distance(generated_preds[:, 2], self.eval_sets_sp_clss[:, 2]))   
    ws_distance_dict['preds']['hepg2'].append(wasserstein_distance(generated_preds[:, 0], self.eval_sets_sp_preds[:, 0]))
    ws_distance_dict['preds']['k562'].append(wasserstein_distance(generated_preds[:, 1], self.eval_sets_sp_preds[:, 1]))
    ws_distance_dict['preds']['sknsh'].append(wasserstein_distance(generated_preds[:, 2], self.eval_sets_sp_preds[:, 2])) 
    return ws_distance_dict

  def cal_ws_distance_embpca(self, seqs):
    generated_embs = oracle.cal_gosai_emb(seqs)
    generated_embs_pca = self.emb_pca.transform(generated_embs.reshape(generated_embs.shape[0], -1))
    return oracle.get_wasserstein_dist(generated_embs_pca, self.eval_sets_sp_embs_pca)
  
  def compare_kmer(self, kmer1, kmer2, n_sp1, n_sp2):
    kmer_set = set(kmer1.keys()) | set(kmer2.keys())
    counts = np.zeros((len(kmer_set), 2))
    for i, kmer in enumerate(kmer_set):
        if kmer in kmer1:
            counts[i][1] = kmer1[kmer] * n_sp2 / n_sp1
        if kmer in kmer2:
            counts[i][0] = kmer2[kmer]
    return pearsonr(counts[:, 0], counts[:, 1])[0]

  def cal_kmer_pearsonr(self, seqs):
    generated_kmer = oracle.count_kmers(seqs)
    return self.compare_kmer(self.eval_sets_sp_kmers, generated_kmer, self.config.eval.subset_size, len(seqs))


  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  def q_xt(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    if p_x0 is None:
      p_x0 = self.forward(x, sigma_t).exp()
      # if self.pad_token_id is not None and self._active_mask is not None:
      #   logits = self.forward(x, sigma_t).exp()
      #   # logits = self._suppress_pad_logits(logits)
      #   p_x0 = logits.exp()
    
    assert move_chance_t.ndim == p_x0.ndim
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  # enforce pad tokens after  sampling
  # def _enforce_pad_tail_onehot(self, x, eos_id, pad_id):
  #   if eos_id is None or pad_id is None:
  #       return x
  #   ids = x.argmax(dim=-1)  # hard EOS decision
  #   eos = (ids == eos_id)
  #   after_eos = eos.cumsum(dim=1) > 0
  #   after_eos = after_eos & (~eos)

  #   pad_onehot = torch.zeros_like(x)
  #   pad_onehot[..., pad_id] = 1.0
  #   # overwrite positions after EOS with PAD onehot
  #   x = torch.where(after_eos.unsqueeze(-1), pad_onehot, x)
  #   return x
  # def _enforce_pad_tail_ids(self, x, eos_id, pad_id):
  #   if self.eos_token_id is None or self.pad_token_id is None:
  #     return x
  #   eos = (x == self.eos_token_id)
  #   after = (eos.cumsum(dim=1) > 0) & (~eos)
  #   return torch.where(after, torch.full_like(x, self.pad_token_id), x)
  def _enforce_target_length_onehot(self, x_onehot, target_length):
    # x_onehot: [B, T, V], soft one-hot
    B, T, V = x_onehot.shape
    seq_indices = torch.arange(T, device=x_onehot.device).unsqueeze(0)
    target_length_exp = target_length.unsqueeze(1)
    eos_mask = (seq_indices == target_length_exp)  # [B, T]
    if self.eos_token_id is None:
      pad_mask = (seq_indices >= target_length_exp)
    else:
      pad_mask = (seq_indices > target_length_exp)

    # Build fixed one-hot rows for EOS and PAD
    if self.eos_token_id is not None:
        eos_onehot = torch.zeros(B, T, V, device=x_onehot.device, dtype=x_onehot.dtype)
        eos_onehot[:, :, self.eos_token_id] = 1.0
        x_onehot = torch.where(eos_mask.unsqueeze(-1), eos_onehot, x_onehot)

    if self.pad_token_id is not None:
        pad_onehot = torch.zeros(B, T, V, device=x_onehot.device, dtype=x_onehot.dtype)
        pad_onehot[:, :, self.pad_token_id] = 1.0
        x_onehot = torch.where(pad_mask.unsqueeze(-1), pad_onehot, x_onehot)

    return x_onehot

  def _enforce_target_length_ids(self, x_ids, target_length):
    # x_ids: [B, T], token ids
    B, T = x_ids.shape
    seq_indices = torch.arange(T, device=x_ids.device).unsqueeze(0)
    target_length_exp = target_length.unsqueeze(1)
    eos_mask = (seq_indices == target_length_exp)
    if self.eos_token_id is None:
      pad_mask = (seq_indices >= target_length_exp)
    else:
      pad_mask = (seq_indices > target_length_exp)

    if self.eos_token_id is not None:
      x_ids = torch.where(eos_mask, self.eos_token_id, x_ids)
    if self.pad_token_id is not None:
      x_ids = torch.where(pad_mask, self.pad_token_id, x_ids)
    return x_ids

  def _enforce_target_length(self, x, target_length):
    if x.ndim == 2:
      return self._enforce_target_length_ids(x, target_length)
    if x.ndim == 3:
      return self._enforce_target_length_onehot(x, target_length)
    return x



  def _ddpm_update(self, x, t, dt, return_process=False):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t) # t
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    # log_p_x0 = self._suppress_pad_logits(log_p_x0)
    assert move_chance_t.ndim == log_p_x0.ndim
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    copy_flag = (x != self.mask_index).to(x.dtype)
    x_next = copy_flag * x + (1 - copy_flag) * _x
    # x_next = self._enforce_pad_tail_ids(x_next, self.eos_token_id, self.pad_token_id)
    if return_process:
      return copy_flag * x + (1 - copy_flag) * _x, x, unet_conditioning, move_chance_t, copy_flag
    else:
      # return copy_flag * x + (1 - copy_flag) * _x
      return x_next
  
  def _ar_sampler(self, bsz):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))
    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y
    return x

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5, eval_sp_size=None,target_length=None):
    """Generate samples from the model."""
    if eval_sp_size is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    else:
      batch_size_per_gpu = eval_sp_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    # if target_length is None:
    #   target_length = self.default_target_length
    # target_length = min(target_length, self.config.model.length)
    # active_mask = torch.arange(self.config.model.length, device=self.device) < target_length
    # prev_active_mask = self._active_mask
    # self._active_mask = active_mask

    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    
    ## target length enforcement can be added here

    if target_length is not None:
      if isinstance(target_length, int):
                  target_length = torch.full((batch_size_per_gpu,), target_length, device=self.device, dtype=torch.long)
      elif isinstance(target_length, (list, tuple, np.ndarray)):
          target_length = torch.tensor(target_length, device=self.device, dtype=torch.long)
      seq_indices = torch.arange(self.config.model.length, device=self.device).unsqueeze(0)
      target_length_exp = target_length.unsqueeze(1)

      # Mask for EOS: Exactly at target_length
      eos_mask = (seq_indices == target_length_exp)
      if self.eos_token_id is None:
        pad_mask = (seq_indices >= target_length_exp)
      else:
        pad_mask = (seq_indices > target_length_exp)
        
      # Force Constraints
      # The model sees these as "observed" data immediately
      if self.eos_token_id is not None:
          x = torch.where(eos_mask, self.eos_token_id, x)
      
      if self.pad_token_id is not None:
          x = torch.where(pad_mask, self.pad_token_id, x)

    # x = self._enforce_pad_tail(x)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x = self._ddpm_update(x, t, dt)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
        x = x_next
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        logits = self.forward(x, unet_conditioning)
        x = logits[:, :, :-1].argmax(dim=-1)
      # x = self._enforce_pad_tail(x)
    return x
  
  def _ddpm_update_finetune_gradient(self, x, t, dt, copy_flag_temp, return_process=False):
    
    if x.ndim == 2 or x.shape[-1] != self.vocab_size:
      x = F.one_hot(x, num_classes=self.vocab_size).to(torch.float32)

    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t) # (1-eps)*t
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical_gradient(q_xs, temp=self.config.finetuning.gumbel_softmax_temp)
    
    if copy_flag_temp is not None:
      copy_flag_prob = 1 - x[:, :, self.mask_index].unsqueeze(-1)
      soft_copy_flag = torch.nn.functional.sigmoid(copy_flag_prob/copy_flag_temp)
    else:
      soft_copy_flag = 1 - x[:, :, self.mask_index].unsqueeze(-1)

    ## enforce padding tokens after the first eos token
    x_next = soft_copy_flag * x + (1 - soft_copy_flag) * _x

    # x_next = self._enforce_pad_tail_onehot(x_next, self.eos_token_id, self.pad_token_id)

    if return_process:
      return (
        soft_copy_flag * x + (1 - soft_copy_flag) * _x,
        x,
        unet_conditioning,
        move_chance_t,
        soft_copy_flag,
        log_p_x0,
      )
    else:
      # return soft_copy_flag * x + (1 - soft_copy_flag) * _x
      return x_next
    
   
  def _sample_finetune_gradient(
    self,
    num_steps=None,
    eps=1e-5,
    eval_sp_size=None,
    copy_flag_temp=None,
    target_length=None,
    gradient_type=None,
  ):
    """Generate samples from the model."""
    assert self.parameterization == 'subs' and self.sampler == 'ddpm'
    if eval_sp_size is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    else:
      batch_size_per_gpu = eval_sp_size
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)

    if target_length is not None:
      if isinstance(target_length, int):
        target_length = torch.full((batch_size_per_gpu,), target_length, device=self.device, dtype=torch.long)
      elif isinstance(target_length, (list, tuple, np.ndarray)):
        target_length = torch.tensor(target_length, device=self.device, dtype=torch.long)
      seq_indices = torch.arange(self.config.model.length, device=self.device).unsqueeze(0)
      target_length_exp = target_length.unsqueeze(1)
      eos_mask = (seq_indices == target_length_exp)
      if self.eos_token_id is None:
        pad_mask = (seq_indices >= target_length_exp)
      else:
        pad_mask = (seq_indices > target_length_exp)
      if self.eos_token_id is not None:
        x = torch.where(eos_mask, self.eos_token_id, x)
      if self.pad_token_id is not None:
        x = torch.where(pad_mask, self.pad_token_id, x)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    last_x_list = []
    kl_x_list = []
    condt_list = []
    move_chance_t_list = []
    copy_flag_list = []
    log_p_x0_last = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        if i < num_steps - self.config.finetuning.truncate_steps:
          x, last_x, condt, move_chance_t, copy_flag = self._ddpm_update(x, t, dt, return_process=True)
          if target_length is not None:
            x = self._enforce_target_length(x, target_length)
          x = x.detach()
          copy_flag = copy_flag.unsqueeze(-1)
          last_x = F.one_hot(last_x, num_classes=self.vocab_size).to(torch.float32).detach()
          kl_x = last_x
        else: 
          x, last_x, condt, move_chance_t, copy_flag, log_p_x0 = self._ddpm_update_finetune_gradient(
            x, t, dt, copy_flag_temp, return_process=True
          )
          if target_length is not None:
            x = self._enforce_target_length(x, target_length)
          kl_x = last_x
          log_p_x0_last = log_p_x0
      # if target_length is not None:
      #   x = self._enforce_target_length_onehot(x, target_length)
      last_x_list.append(last_x)
      kl_x_list.append(kl_x)
      condt_list.append(condt)
      move_chance_t_list.append(move_chance_t)
      copy_flag_list.append(copy_flag)

    logits = x[:, :, :-1]
    p_vocab = x[:, :, :-1]
    motif_ids_hard = logits.argmax(dim=-1)
    x_argmax = torch.nn.functional.one_hot(
      logits.argmax(dim=-1),
      num_classes=self.vocab_size - 1).to(torch.float32)
    # print(self.vocab_size)
    if gradient_type == "motif_soft":
      # Return straight-through motif-level vectors; mapping to base space happens outside.
      sample_out = logits + (x_argmax - logits).detach()
    elif self.vocab_size > 7 and hasattr(self, 'motif2base_stencil'):
      B, T, V_logits = logits.shape
      stencil = self.motif2base_stencil.to(device=logits.device, dtype=torch.float32)
      V_stencil, k_max, C = stencil.shape
      assert C == 4
      motif_lengths = self.motif_lengths
      start = torch.zeros(B, T, dtype=torch.long, device=logits.device)
      base_len = torch.zeros(B, dtype=torch.long, device=logits.device)
      valid_T = torch.zeros(B, dtype=torch.long, device=logits.device)
      max_base_len = 0

      for b in range(B):
          cur = 0
          t_eff = 0
          for t in range(T):
              v = int(motif_ids_hard[b, t].item())
              # Truncate at first eos or pad
              if v == self.eos_token_id or v == self.pad_token_id:
                  break
              start[b, t] = cur
              L_t = int(motif_lengths[v].item())
              cur += L_t
              t_eff = t + 1
          base_len[b] = cur
          valid_T[b] = t_eff
          if cur > max_base_len:
              max_base_len = cur
      # with torch.no_grad():
      #   stop = (motif_ids_hard == self.eos_token_id) | (motif_ids_hard == self.pad_token_id)
      #   # first stop index per sequence (T if none)
      #   first_stop = torch.where(
      #       stop.any(dim=1),
      #       stop.float().argmax(dim=1),
      #       torch.full((stop.size(0),), stop.size(1), device=stop.device)
      #   )
      #   print("first_stop mean:", first_stop.float().mean().item(),
      #         "min:", first_stop.min().item(),
      #         "max:", first_stop.max().item())
      #   print("base_len mean:", base_len.float().mean().item(),
      #         "min:", base_len.min().item(),
      #         "max:", base_len.max().item())

      # -----------------------
      # 4) Scatter soft & hard bases into [B, max_base_len, 4]
      # -----------------------
      base_soft = torch.zeros(
          B, max_base_len, 4, device=logits.device, dtype=torch.float32
      )
      base_hard = torch.zeros_like(base_soft)

      probs = logits  # [B, T, V_logits]

      for b in range(B):
          Teff = int(valid_T[b].item())
          for t in range(Teff):
              v_hard = int(motif_ids_hard[b, t].item())
              L_t = int(motif_lengths[v_hard].item())
              if L_t == 0:
                  continue

              s = int(start[b, t].item())

              # Expected base pattern at this motif slot: [k_max, 4]
              local_soft = torch.einsum('v, vkc -> kc', probs[b, t], stencil)  # [k_max, 4]
              local_hard = stencil[v_hard]                                     # [k_max, 4]
              # p_star = probs[b, t, v_hard]  # [V_logits]
              # local_soft = p_star * local_hard  # [k_max, 4]
              # Only first L_t offsets are "real" bases
              base_soft[b, s:s + L_t] += local_soft[:L_t]
              base_hard[b, s:s + L_t] += local_hard[:L_t]

      # Normalize soft bases to probabilities per position
      base_soft = base_soft / (base_soft.sum(dim=-1, keepdim=True) + 1e-8)

      # -----------------------
      # 5) Straight-through in base space: forward=hard, backward=soft
      # -----------------------
      sample_out = base_soft + (base_hard - base_soft).detach()
    # if self.vocab_size > 7 and hasattr(self, "motif2base_stencil"):
    #   B, T, V_logits = logits.shape

    #   stencil = self.motif2base_stencil.to(device=logits.device, dtype=torch.float32)  # [V, k_max, 4]
    #   V_stencil, k_max, C = stencil.shape
    #   assert C == 4

    #   motif_lengths = self.motif_lengths.to(device=logits.device)  # [V], integer dtype preferred (long)
    #   motif_ids = motif_ids_hard.to(device=logits.device)          # [B, T], long

    #   # -----------------------
    #   # 1) Batch compute valid_T, start, base_len, max_base_len
    #   #    (truncate at first EOS/PAD)
    #   # -----------------------
    #   is_stop = (motif_ids == self.eos_token_id) | (motif_ids == self.pad_token_id)  # [B, T] bool

    #   # valid positions are those strictly before the first stop token
    #   valid_mask = (is_stop.cumsum(dim=1) == 0)  # [B, T] bool

    #   # lengths per slot, masked to 0 after stop
    #   L = motif_lengths[motif_ids]                               # [B, T]
    #   L = L * valid_mask.to(L.dtype)                             # [B, T]

    #   # start[b,t] = sum_{j<t} L[b,j]
    #   start = torch.cumsum(L, dim=1) - L                         # [B, T]

    #   base_len = L.sum(dim=1)                                    # [B]
    #   valid_T = valid_mask.sum(dim=1)                            # [B]
    #   max_base_len = int(base_len.max().item()) if B > 0 else 0

    #   # Allocate output
    #   base_soft = torch.zeros(B, max_base_len, 4, device=logits.device, dtype=torch.float32)
    #   base_hard = torch.zeros_like(base_soft)

    #   # -----------------------
    #   # 2) Batch scatter motif slots into base space
    #   #    Your "winner-only" soft: local_soft = p_star * local_hard
    #   # -----------------------
    #   # IMPORTANT:
    #   # If logits are raw logits, use softmax to get probs.
    #   # If logits are already probs, keep probs = logits.
    #   probs = torch.softmax(logits, dim=-1)  # [B, T, V_logits]

    #   v_hard = motif_ids                                         # [B, T]
    #   L_t = motif_lengths[v_hard]                                # [B, T]

    #   # Only process valid slots and nonzero-length motifs
    #   slot_mask = valid_mask & (L_t > 0)                         # [B, T] bool

    #   # local_hard: [B, T, k_max, 4]
    #   local_hard = stencil[v_hard]

    #   # p_star: prob of the chosen motif at each (b,t): [B, T]
    #   p_star = probs.gather(dim=-1, index=v_hard.unsqueeze(-1)).squeeze(-1)

    #   # local_soft: [B, T, k_max, 4]
    #   local_soft = local_hard * p_star.view(B, T, 1, 1)

    #   # Offsets within motif token
    #   offs = torch.arange(k_max, device=logits.device).view(1, 1, k_max)  # [1,1,k_max]
    #   offs_mask = offs < L_t.view(B, T, 1)                                 # [B,T,k_max]

    #   # Combine masks
    #   mask = slot_mask.view(B, T, 1) & offs_mask                           # [B,T,k_max]

    #   # Base index for each (b,t,off): start[b,t] + off
    #   base_idx = start.view(B, T, 1) + offs                                # [B,T,k_max]

    #   # Flatten only the valid entries (this replaces Python loops)
    #   b_ids, t_ids, o_ids = mask.nonzero(as_tuple=True)                    # [N]
    #   pos_ids = base_idx[b_ids, t_ids, o_ids]                              # [N]

    #   soft_vals = local_soft[b_ids, t_ids, o_ids, :]                       # [N,4]
    #   hard_vals = local_hard[b_ids, t_ids, o_ids, :]                       # [N,4]

    #   # Scatter-add into base_soft/base_hard using index_add_ over flattened (b,pos)
    #   lin = b_ids * max_base_len + pos_ids                                  # [N]
    #   base_soft_flat = base_soft.view(B * max_base_len, 4)
    #   base_hard_flat = base_hard.view(B * max_base_len, 4)

    #   base_soft_flat.index_add_(0, lin, soft_vals)
    #   base_hard_flat.index_add_(0, lin, hard_vals)

    #   base_soft = base_soft_flat.view(B, max_base_len, 4)
    #   base_hard = base_hard_flat.view(B, max_base_len, 4)

    #   # do NOT normalize base_soft if you're using p_star * onehot,
    #   # because normalization can wipe out the p_star effect when only one motif contributes.
    #   # base_soft = base_soft / (base_soft.sum(dim=-1, keepdim=True) + 1e-8)

    #   # -----------------------
    #   # 3) Straight-through in base space
    #   # -----------------------
    #   sample_out = base_soft + (base_hard - base_soft).detach()
    else:
      sample_out = logits + (x_argmax - logits).detach()
      if self.pad_token_id is not None and self.pad_token_id < sample_out.shape[-1]:
        pad_idx = self.pad_token_id

        sample_out = torch.cat(

        [sample_out[..., :pad_idx], sample_out[..., pad_idx + 2:]],

        dim=-1)
    return (
      sample_out,
      last_x_list,
      condt_list,
      move_chance_t_list,
      copy_flag_list,
      kl_x_list,
      p_vocab,
      log_p_x0_last,
    )

  # def _motif_tokens_to_base(self, motif_probs, base_len=None):
  #   """Map motif token probabilities to base-level probabilities."""
  #   if not hasattr(self, 'motif2base_stencil'):
  #     raise AttributeError('motif2base_stencil buffer not found on model.')
  #   stencil = self.motif2base_stencil
  #   if stencil.device != motif_probs.device or stencil.dtype != motif_probs.dtype:
  #     stencil = stencil.to(device=motif_probs.device, dtype=motif_probs.dtype)
  #   B, T, V = motif_probs.shape
  #   V_stencil, k_max, C = stencil.shape
  #   if V_stencil != V:
  #     if V_stencil == V + 1:
  #       # Drop an extra vocab row (e.g., mask/pad) to align with logits.
  #       drop_id = getattr(self, 'pad_token_id', None)
  #       if drop_id is None or drop_id >= V_stencil:
  #         drop_id = V_stencil - 1
  #       stencil = torch.cat([stencil[:drop_id], stencil[drop_id + 1:]], dim=0)
  #       V_stencil, k_max, C = stencil.shape
  #     if V_stencil != V:
  #       raise ValueError(f'Stencil vocab {V_stencil} does not match motif probs {V}.')
  #   L = base_len if base_len is not None else T + k_max - 1
  #   Mk = stencil.view(V, -1)
  #   base_scores = motif_probs @ Mk
  #   base_scores = base_scores.view(B, T, k_max, C)
  #   out = torch.zeros(B, L, C, device=motif_probs.device, dtype=motif_probs.dtype)
  #   cover = torch.zeros(B, L, 1, device=motif_probs.device, dtype=motif_probs.dtype)
  #   for r in range(k_max):
  #     n0, n1 = r, min(L, r + T)
  #     out[:, n0:n1, :] += base_scores[:, : (n1 - n0), r, :]
  #     cover[:, n0:n1, :] += 1.0
  #   out = out / (cover + 1e-8)
  #   out = out / (out.sum(-1, keepdim=True) + 1e-8)
  #   return out  # [B, base_len, 4]
  
  @torch.no_grad()
  def _sample_no_gradient(self, num_steps=None, eps=1e-5, eval_sp_size=None, copy_flag_temp=None):
    """Generate samples without keeping the gradient."""
    assert self.parameterization == 'subs' and self.sampler == 'ddpm'
    if eval_sp_size is None:
        batch_size_per_gpu = self.config.loader.eval_batch_size
    else:
        batch_size_per_gpu = eval_sp_size
    if num_steps is None:
        num_steps = self.config.sampling.steps

    # Initialize samples
    x = self._sample_prior(batch_size_per_gpu, self.config.model.length).to(self.device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps

    last_x_list = []
    condt_list = []
    move_chance_t_list = []
    copy_flag_list = []

    for i in range(num_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)

        if i < num_steps - self.config.finetuning.truncate_steps:
            x, last_x, condt, move_chance_t, copy_flag = self._ddpm_update(
                x, t, dt, return_process=True
            )
            x = x.detach()  # Ensure gradients are not retained
            copy_flag = copy_flag.unsqueeze(-1)
            last_x = F.one_hot(last_x, num_classes=self.vocab_size).to(torch.float32).detach()
        else:
            x, last_x, condt, move_chance_t, copy_flag, _ = self._ddpm_update_finetune_gradient(
                x, t, dt, copy_flag_temp, return_process=True
            )

        last_x_list.append(last_x)
        condt_list.append(condt)
        move_chance_t_list.append(move_chance_t)
        copy_flag_list.append(copy_flag)

    x_argmax = x[:, :, :-1].argmax(dim=-1)
    x_argmax = torch.nn.functional.one_hot(x_argmax, num_classes=self.vocab_size - 1).to(torch.float32)

    return x[:, :, :-1] + (x_argmax - x[:, :, :-1]).detach(), last_x_list, condt_list, move_chance_t_list, copy_flag_list

  @torch.no_grad()
  def _ddpm_update_finetune_controlled_SMC(self, x, t, dt, reward_model, alpha = 1.0, target_length=None):

    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    copy_flag = (x != self.mask_index).to(x.dtype)
    sample = copy_flag * x + (1 - copy_flag) * _sample_categorical(q_xs)
    '''
    Calcualte exp(v_{t-1}(x_{t-1})/alpha)
    '''
    reward_num = self._expected_x0_reward_scores(
      sample, sigma_s, reward_model, target_length=target_length).detach()
    '''
    Calcualte exp(v_{t}(x_{t})/alpha)
    '''
    reward_den = self._expected_x0_reward_scores(
      x, sigma_s, reward_model, target_length=target_length).detach()
  
    ratio = torch.exp(1.0/alpha * (reward_num - reward_den)) # Now calculate exp( (v_{t-1}(x_{t-1) -v_{t}(x_{t}) /alpha) 
    ratio = ratio.detach().cpu().numpy()
    final_sample_indices = np.random.choice(reward_num.shape[0], reward_num.shape[0], p =  ratio/ratio.sum() ) 
   
    return sample[final_sample_indices]
  
  def _ddpm_update_finetune_controlled_CG(self, x, t, dt, reward_model,  guidance_scale, target_length=None):

    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    x_onehot = F.one_hot(x, num_classes=self.vocab_size).float()

    x_grad = self.compute_gradient_CG(
      x_onehot, x, reward_model, sigma_s, target_length=target_length)
    guidance = guidance_scale * (x_grad - x_grad[:, :, self.mask_index][:, :, None])
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    q_xs = q_xs * guidance.exp()

    _x = _sample_categorical(q_xs)
    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x 

  def compute_gradient_CG(self, x_onehot, x, reward_model, sigma_s, target_length=None):
    x_onehot = x_onehot.detach().requires_grad_(True)
    expected_x0 = self.forward(x_onehot, sigma_s)

    # This project's base-token order is A,C,G,T in the first four channels.
    base_probs = expected_x0[:, :, :4].exp()
    base_probs = base_probs / base_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    hard = torch.nn.functional.one_hot(
      base_probs.argmax(dim=-1),
      num_classes=base_probs.size(-1)).float()
    st = base_probs + (hard - base_probs).detach()

    reward_input = self._target_length_reward_input(
      st, target_length=target_length, reward_model=reward_model)
    scores = reward_model(reward_input, soft_input=True).reshape(-1).mean()
    x_grad = torch.autograd.grad(scores, x_onehot, retain_graph=False, create_graph=False)[0]
    return x_grad.detach()

  def _target_length_reward_input(self, base_probs, target_length, reward_model):
    if target_length is None:
      return base_probs.transpose(1, 2)

    B, L, C = base_probs.shape
    if isinstance(target_length, int):
      valid_lens = torch.full((B,), target_length, device=base_probs.device, dtype=torch.long)
    elif isinstance(target_length, (list, tuple, np.ndarray)):
      valid_lens = torch.tensor(target_length, device=base_probs.device, dtype=torch.long)
    elif torch.is_tensor(target_length):
      valid_lens = target_length.to(device=base_probs.device, dtype=torch.long).view(-1)
    else:
      raise TypeError(f"Unsupported target_length type: {type(target_length)}")
    valid_lens = valid_lens.clamp(min=0, max=L)

    reward_input_len = getattr(reward_model, "input_length", None)
    if reward_input_len is None:
      seq_idx = torch.arange(L, device=base_probs.device).unsqueeze(0)
      valid_mask = seq_idx < valid_lens.unsqueeze(1)
      return (base_probs * valid_mask.unsqueeze(-1).to(base_probs.dtype)).transpose(1, 2)

    reward_input_len = int(reward_input_len)
    rows = []
    for bi in range(B):
      cur_len = int(valid_lens[bi].item())
      if cur_len <= 0:
        rows.append(torch.zeros(C, reward_input_len, device=base_probs.device, dtype=base_probs.dtype))
        continue
      cur = base_probs[bi, :cur_len, :].transpose(0, 1)
      if cur_len > reward_input_len:
        cur = cur[:, -reward_input_len:]
        cur_len = reward_input_len
      rows.append(F.pad(cur, (reward_input_len - cur_len, 0), mode="constant", value=0.0))
    return torch.stack(rows, dim=0)

  def _expected_x0_reward_scores(self, x, sigma, reward_model, target_length=None):
    expected_x0 = self.forward(x, sigma)
    base_probs = expected_x0[:, :, :4].exp()
    base_probs = base_probs / base_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    hard = torch.nn.functional.one_hot(
      base_probs.argmax(dim=-1),
      num_classes=base_probs.size(-1)).float()
    reward_input = self._target_length_reward_input(
      hard, target_length=target_length, reward_model=reward_model)
    return reward_model(reward_input, soft_input=False).reshape(-1)

  def _ddpm_update_finetune_controlled_TDS(self, x, t, dt, reward_model, alpha = 1.0, guidance_scale=1000, target_length=None):
    # SMC with the twisted proposal

    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    x_onehot = F.one_hot(x, num_classes=self.vocab_size).float()

    x_grad = self.compute_gradient_CG(
      x_onehot, x, reward_model, sigma_s, target_length=target_length)
    guidance = guidance_scale * (x_grad - x_grad[:, :, self.mask_index][:, :, None])
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    # print(q_xs.sum(-1))
    q_xs = q_xs * guidance.exp()

    _x = _sample_categorical(q_xs)
    copy_flag = (x != self.mask_index).to(x.dtype)
    sample = copy_flag * x + (1 - copy_flag) * _x
    prob_multiplier = (1 - copy_flag) * torch.gather(guidance.exp(), 2, _x.unsqueeze(-1)).squeeze(-1) + copy_flag * torch.ones_like(_x)
    '''
    Calcualte exp(v_{t-1}(x_{t-1})/alpha)
    '''
    reward_num = self._expected_x0_reward_scores(
      sample, sigma_s, reward_model, target_length=target_length).detach()
    '''
    Calcualte exp(v_{t}(x_{t})/alpha)
    '''
    reward_den = self._expected_x0_reward_scores(
      x, sigma_s, reward_model, target_length=target_length).detach()
    
    # set the nan values to 1
    prob_multiplier[torch.isnan(prob_multiplier)] = 1
    ratio = torch.exp(1.0/alpha * (reward_num - reward_den)) / prob_multiplier.prod(dim=-1)
    ratio = ratio.detach().cpu().numpy()
    final_sample_indices = np.random.choice(reward_num.shape[0], reward_num.shape[0], p =  ratio/ratio.sum() ) 
   
    return sample[final_sample_indices]
  
  @torch.no_grad()
  def controlled_sample_SMC(self, reward_model, alpha, num_steps=None, eps=1e-5, eval_sp_size=None, target_length=None):
    """Generate samples from the model."""
    if eval_sp_size is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    else:
      batch_size_per_gpu = eval_sp_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    if target_length is not None:
      if isinstance(target_length, int):
        target_length = torch.full((batch_size_per_gpu,), target_length, device=self.device, dtype=torch.long)
      elif isinstance(target_length, (list, tuple, np.ndarray)):
        target_length = torch.tensor(target_length, device=self.device, dtype=torch.long)
      x = self._enforce_target_length(x, target_length)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x  = self._ddpm_update_finetune_controlled_SMC(
          x, t, dt, reward_model, alpha, target_length=target_length)
        if target_length is not None:
          x = self._enforce_target_length(x, target_length)
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        logits = self.forward(x, unet_conditioning)
        x = logits[:, :, :-1].argmax(dim=-1)
    return x

  def controlled_sample_CG(self, reward_model, guidance_scale, num_steps=None, eps=1e-5, eval_sp_size=None, target_length=None):
    """Generate samples from the model."""
    if eval_sp_size is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    else:
      batch_size_per_gpu = eval_sp_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)

    if target_length is not None:
      if isinstance(target_length, int):
        target_length = torch.full((batch_size_per_gpu,), target_length, device=self.device, dtype=torch.long)
      elif isinstance(target_length, (list, tuple, np.ndarray)):
        target_length = torch.tensor(target_length, device=self.device, dtype=torch.long)
      x = self._enforce_target_length(x, target_length)

    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x  = self._ddpm_update_finetune_controlled_CG(
          x, t, dt, reward_model, guidance_scale, target_length=target_length)
        if target_length is not None:
          x = self._enforce_target_length(x, target_length)
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        logits = self.forward(x, unet_conditioning)
        x = logits[:, :, :-1].argmax(dim=-1)
    return x

  def controlled_sample_TDS(self, reward_model, alpha, guidance_scale, num_steps=None, eps=1e-5, eval_sp_size=None, target_length=None):
    """Generate samples from the model."""
    if eval_sp_size is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    else:
      batch_size_per_gpu = eval_sp_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    if target_length is not None:
      if isinstance(target_length, int):
        target_length = torch.full((batch_size_per_gpu,), target_length, device=self.device, dtype=torch.long)
      elif isinstance(target_length, (list, tuple, np.ndarray)):
        target_length = torch.tensor(target_length, device=self.device, dtype=torch.long)
      x = self._enforce_target_length(x, target_length)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x  = self._ddpm_update_finetune_controlled_TDS(
          x, t, dt, reward_model, alpha, guidance_scale, target_length=target_length)
        if target_length is not None:
          x = self._enforce_target_length(x, target_length)
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        logits = self.forward(x, unet_conditioning)
        x = logits[:, :, :-1].argmax(dim=-1)
    return x

  @torch.no_grad()
  def get_likelihood(self, x0, num_steps=None, eps=1e-5, n_samples=1):
    """Compute the likelihood of a sequence under the model.
    x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length)
    """
    if num_steps is None:
      num_steps = self.config.sampling.steps
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device) # t=0 is clean data
    dt = (1 - eps) / num_steps
    log_p_sample_list = []
    for _ in range(n_samples):
      log_p_at_time_list = []
      for i in range(num_steps):
        t = timesteps[i] * torch.ones(
          x0.shape[0], 1, device=self.device)
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
          sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
          sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t) # (1-eps)*t
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None] # [bsz, 1]
        move_chance_s = move_chance_s[:, None]
        unet_conditioning = sigma_t # [bsz]
        multiplier = (move_chance_t - move_chance_s)/move_chance_t # [bsz, 1]
        xt = self.q_xt(x0, move_chance_t) # [bsz, seq_len]
        # log prob, already apply subs parametrization (unmasked token remains unchanged)
        model_output = self.forward(xt, unet_conditioning) # [bsz, seq_len, vocab_size]
        # take the log prob of the token that corresponds to x0
        log_p_x0 = model_output.gather(-1, x0[..., None]).squeeze(-1) # [bsz, seq_len]
        log_p_x0 = log_p_x0 * multiplier
        log_p_at_time_list.append(log_p_x0)
      log_p_x0 = torch.stack(log_p_at_time_list, dim=0).sum(dim=0) # [bsz, seq_len]
      log_p_sample_list.append(log_p_x0.sum(dim=-1))
    log_p_sample = torch.stack(log_p_sample_list, dim=0).mean(dim=0)
    return log_p_sample

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      # for variance reduction
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      raise NotImplementedError('Sub-sampling not implemented')
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0):
    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      # else ts are between 0 and 1
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables: # False
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t) # total noise, rate noise
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    xt = self.q_xt(x0, move_chance) # q(xt|x0)
    model_output = self.forward(xt, unet_conditioning)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(x0)
      elif self.parameterization == 'subs':
        reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]

  def _loss(self, x0, attention_mask):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  # def _suppress_pad_logits(self, log_p_x0):
  #   """Force pad logits to -inf on active positions before sampling."""
  #   if self.pad_token_id is None or self._active_mask is None:
  #     return log_p_x0
  #   log_p_x0 = log_p_x0.clone()
  #   log_p_x0[:, self._active_mask, self.pad_token_id] = self.neg_infinity
  #   return log_p_x0

  # def _enforce_pad_tail(self, x):
  #   if self.pad_token_id is None or self._active_mask is None:
  #     return x
  #   x = x.clone()
  #   x[:, ~self._active_mask] = self.pad_token_id
  #   return x



  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    # seems that it takes y=x0,xt=M case
    # what is the const term for, seems to be y=M,xt=x0 case and x0 is known so score estimation is precise
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy
