import sys
import importlib
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import os
import re

import numpy as np
import torch
from torch import nn
import torch.utils.data

try:
  from esm import Alphabet, FastaBatchedDataset  # type: ignore
  from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS  # type: ignore
  from esm.model.esm2_supervised import ESM2  # type: ignore
  try:
    from esm.model.esm2_only_secondarystructure import ESM2 as ESM2_SS  # type: ignore
  except ImportError:  # pragma: no cover - optional dependency
    ESM2_SS = None
except ImportError:  # pragma: no cover - defer failure until oracle is requested
  Alphabet = None  # type: ignore
  FastaBatchedDataset = None  # type: ignore
  ESM2 = None  # type: ignore
  ESM2_SISS = None  # type: ignore
  ESM2_SS = None  # type: ignore

MTTRANS_ROOT = Path(__file__).resolve().parents[0] / 'MTtrans'
if MTTRANS_ROOT.exists():
  parent_dir = MTTRANS_ROOT.parent
  if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Ensure checkpoints referring to legacy module paths (e.g., `models.Modules`)
# can still be deserialized.
mt_models_pkg = importlib.import_module('MTtrans.models')
sys.modules.setdefault('models', mt_models_pkg)
sys.modules.setdefault('models.Modules', importlib.import_module('MTtrans.models.Modules'))
sys.modules.setdefault('models.ScheduleOptimizer', importlib.import_module('MTtrans.models.ScheduleOptimizer'))
sys.modules.setdefault('utils', importlib.import_module('MTtrans.utils'))

from MTtrans.models.popen import Auto_popen  # type: ignore
from MTtrans.models.reader import one_hot, pad_zeros  # type: ignore
from MTtrans.utils import load_model  # type: ignore
import dataloader_gosai

DEFAULT_MTTRANS_TASK_MAP = {
  "V_293": "RP_293T",
  "V_muscle": "RP_muscle",
  "V_PC3": "RP_PC3",
}


def _infer_mttrans_task_from_name(name: str, task_map: Dict[str, str]) -> str:
  lower = name.lower()
  for key, task in task_map.items():
    if key.lower() in lower:
      return task
  raise ValueError(f'Cannot infer task for checkpoint "{name}".')


def _discover_mttrans_checkpoints(ckpt_dir: Path,
                                  task_map: Dict[str, str],
                                  tasks: Optional[Sequence[str]] = None) -> List[Tuple[Path, str]]:
  if not ckpt_dir.exists():
    raise FileNotFoundError(f'Checkpoint directory not found: {ckpt_dir}')
  pairs: List[Tuple[Path, str]] = []
  for fname in sorted(os.listdir(ckpt_dir)):
    if not fname.endswith('.pth'):
      continue
    try:
      task = _infer_mttrans_task_from_name(fname, task_map)
    except ValueError:
      continue
    if tasks is not None and task not in tasks:
      continue
    pairs.append((ckpt_dir / fname, task))
  if not pairs:
    raise RuntimeError(f'No MTtrans checkpoints found in {ckpt_dir}')
  return pairs


class MTTransOracle(nn.Module):
  """Wrapper that turns diffusion samples into TE predictions via MTtrans."""

  def __init__(self,
               config_path: str,
               checkpoint_path: Optional[str] = None,
               checkpoint_paths: Optional[Sequence[str]] = None,
               checkpoint_dir: Optional[str] = None,
               tasks: Optional[Sequence[str]] = None,
               task_map: Optional[Dict[str, str]] = None,
               task: str = 'MPA_H',
               trim_len: int = 100,
               device: str = 'cuda',
               vocab_json_path: Optional[str] = None,
               pad_token: str = 'N',
               eos_token: str = '<eos>',
               motif_trim_len: Optional[int] = None):
    super().__init__()
    self.config_path = Path(config_path)
    if not self.config_path.exists():
      raise FileNotFoundError(f'Config not found: {self.config_path}')
    self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    self.task = task
    self.trim_len = trim_len

    cfg = Auto_popen(str(self.config_path))
    self.pad_to = getattr(cfg, 'pad_to', 105)

    def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
      state = torch.load(path, map_location='cpu')
      if isinstance(state, dict) and 'state_dict' in state:
        weights = state['state_dict']
        if isinstance(weights, nn.Module):
          weights = weights.state_dict()
      elif isinstance(state, nn.Module):
        weights = state.state_dict()
      else:
        raise TypeError(
          f'Unexpected checkpoint format {type(state)} from {path}; expected dict-like or nn.Module.')
      return weights

    self.models: List[nn.Module] = []
    self.model_tasks: List[str] = []
    checkpoint_task_pairs: List[Tuple[Path, str]] = []
    if checkpoint_dir is not None:
      task_map = task_map or DEFAULT_MTTRANS_TASK_MAP
      checkpoint_task_pairs.extend(
        _discover_mttrans_checkpoints(Path(checkpoint_dir), task_map, tasks))
    if checkpoint_paths is not None:
      if tasks is not None and len(tasks) == len(checkpoint_paths):
        checkpoint_task_pairs.extend(
          (Path(path), task_name) for path, task_name in zip(checkpoint_paths, tasks))
      elif tasks is not None and len(tasks) == 1:
        checkpoint_task_pairs.extend((Path(path), tasks[0]) for path in checkpoint_paths)
      else:
        checkpoint_task_pairs.extend((Path(path), task) for path in checkpoint_paths)

    if checkpoint_task_pairs:
      for ckpt_path, task_name in checkpoint_task_pairs:
        print(task_name)
        print(ckpt_path)
        model = cfg.Model_Class(*cfg.model_args)
        weights = _load_state_dict(ckpt_path)
        model.load_state_dict(weights, strict=False)
        self.models.append(model.to(device))
        self.model_tasks.append(task_name)
    else:
      model = cfg.Model_Class(*cfg.model_args)
      if self.checkpoint_path is not None:
        weights = _load_state_dict(self.checkpoint_path)
        model.load_state_dict(weights, strict=False)
      else:
        model = load_model(cfg, model)
      self.models.append(model.to(device))
      self.model_tasks.append(self.task)

    for model in self.models:
      model.eval()
    self.model = self.models[0]
    self.device = torch.device(device)

    # Detokenizer is optional when callers already operate in base space.
    if vocab_json_path is not None:
      vocab_path = Path(vocab_json_path)
      if not vocab_path.exists():
        raise FileNotFoundError(f'Vocab JSON not found: {vocab_path}')
      resolved_eos = eos_token
      with vocab_path.open() as handle:
        raw_vocab = json.load(handle)
      if resolved_eos not in raw_vocab and "EOS" in raw_vocab:
        resolved_eos = "EOS"
      self.detok = dataloader_gosai.MotifAwareTokenizer(
        vocab_json_path=vocab_path,
        pad_token=pad_token,
        eos_token=resolved_eos,
        base_tokens=("A", "C", "G", "T"),
        max_length=None,
        trim_to=motif_trim_len)
      self.base_tokens = tuple(self.detok.base_tokens)
    else:
      self.detok = None
      self.base_tokens = ("A", "C", "G", "T")

  def forward(self, x: torch.Tensor, denormalize: bool = False, soft_input: bool = False) -> torch.Tensor:
    """
    Args:
      x: Tensor of shape [batch, channels, seq_len] or [batch, seq_len, channels]
         representing soft/hard base probabilities.
      soft_input: If True, `x` already encodes base probabilities and no argmax/
        detokenization is performed.
      denormalize: Whether to map predictions back to original TE scale.
    """
    seq_tensor = self._ensure_channel_first(x)
    if soft_input:
      batch_tensor = self._prepare_soft_batch(seq_tensor)
    else:
      batch_tensor = self._prepare_soft_batch(seq_tensor)
    preds_list = []
    for model, task_name in zip(self.models, self.model_tasks):
      model.task = task_name
      was_training = model.training
      model.train()
      preds = model(batch_tensor).squeeze(-1)
      if not was_training:
        model.eval()
      if denormalize:
        preds = self._reverse_transform(preds, task_name)
      preds_list.append(preds)
    return torch.stack(preds_list, dim=0).mean(dim=0)

  def _ensure_channel_first(self, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 3:
      raise ValueError(f'Expected 3D tensor, got shape {tensor.shape}')
    if tensor.shape[1] == len(self.base_tokens):
      return tensor
    if tensor.shape[2] == len(self.base_tokens):
      return tensor.transpose(1, 2)
    raise ValueError(
      f'Channel dimension must equal {len(self.base_tokens)}, '
      f'got shape {tensor.shape}')

  def _ids_to_sequences(self, ids_tensor: torch.Tensor) -> List[str]:
    if self.detok is None:
      raise ValueError('No detokenizer available; provide vocab_json_path to enable decoding.')
    batch = []
    for row in ids_tensor:
      ints = [int(idx) for idx in row]
      seq = self.detok.decode(ints)
      seq = seq.replace('U', 'T')  # convert to DNA for MTtrans
      batch.append(seq)
    return batch

  def _encode_sequence_for_mttrans(self, seq: str) -> torch.Tensor:
    seq = seq.upper().replace('U', 'T')
    if self.trim_len is not None:
      seq = seq[-self.trim_len:]
    oh = one_hot(seq).astype('float32')
    padded = pad_zeros(oh, self.pad_to)
    return padded.float()

  def _prepare_soft_batch(self, tensor: torch.Tensor) -> torch.Tensor:
    """Convert soft base probabilities to MTtrans input format."""
    seq = tensor.transpose(1, 2)  # [B, L, 4]
    seq = seq / (seq.sum(dim=-1, keepdim=True).clamp_min(1e-8))
    if self.trim_len is not None:
      seq = seq[:, -self.trim_len:, :]
    pad_to = getattr(self, 'pad_to', None)
    batch = []
    for row in seq:
      if pad_to is not None and row.shape[0] > pad_to:
        trimmed = row[-pad_to:, :]
      else:
        trimmed = row
      if pad_to is not None and trimmed.shape[0] < pad_to:
        pad_len = pad_to - trimmed.shape[0]
        pad = torch.zeros(pad_len, trimmed.shape[1], device=trimmed.device, dtype=trimmed.dtype)
        padded = torch.cat([pad, trimmed], dim=0)
      else:
        padded = trimmed
      batch.append(padded)
    batch_tensor = torch.stack(batch, dim=0)
    return batch_tensor.to(self.device)

  def _reverse_transform(self, x: torch.Tensor, task: str) -> torch.Tensor:
    """Map standardized TE predictions back to the original scale."""
    if task not in POSTPROC_MEAN_STD:
      raise ValueError(f'Unknown MTtrans task "{task}" for denormalization.')
    stats = POSTPROC_MEAN_STD[task]
    return x * stats['std'] + stats['mean']


class UTRLMPredictor(nn.Module):
  """CNN + ESM2 predictor used by UTR-LM for TE regression."""

  def __init__(self,
               alphabet: Alphabet,
               predictor_path: Path,
               inp_len: int = 100,
               layers: int = 6,
               heads: int = 16,
               embed_dim: int = 128,
               nodes: int = 40,
               dropout3: float = 0.5,
               cnn_layers: int = 0,
               avg_emb: bool = False,
               bos_emb: bool = True,
               magic: bool = False,
               filter_len: int = 8,
               nbr_filters: int = 120):
    if Alphabet is None or ESM2 is None:
      raise ImportError('UTRLMPredictor requires the `esm` package to be installed.')
    super().__init__()
    self.alphabet = alphabet
    self.predictor_path = predictor_path
    self.inp_len = inp_len
    self.layers = layers
    self.heads = heads
    self.embed_dim = embed_dim
    self.nodes = nodes
    self.dropout3 = dropout3
    self.cnn_layers = cnn_layers
    self.avg_emb = avg_emb
    self.bos_emb = bos_emb
    self.magic = magic
    self.filter_len = filter_len
    self.nbr_filters = nbr_filters
    self.repr_layers = [0, layers]

    ckpt_name = predictor_path.name
    if 'SISS' in ckpt_name:
      self.esm2 = ESM2_SISS(
        num_layers=layers, embed_dim=embed_dim, attention_heads=heads, alphabet=alphabet)
    elif 'SS' in ckpt_name and ESM2_SS is not None:
      self.esm2 = ESM2_SS(
        num_layers=layers, embed_dim=embed_dim, attention_heads=heads, alphabet=alphabet)
    else:
      self.esm2 = ESM2(
        num_layers=layers, embed_dim=embed_dim, attention_heads=heads, alphabet=alphabet)

    self.conv1 = nn.Conv1d(
      in_channels=self.embed_dim,
      out_channels=self.nbr_filters,
      kernel_size=self.filter_len,
      padding='same')
    self.conv2 = nn.Conv1d(
      in_channels=self.nbr_filters,
      out_channels=self.nbr_filters,
      kernel_size=self.filter_len,
      padding='same')

    self.dropout1 = nn.Dropout(0.0)
    self.dropout2 = nn.Dropout(0.0)
    self.dropout3_layer = nn.Dropout(self.dropout3)
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()
    if avg_emb or bos_emb:
      self.fc = nn.Linear(in_features=embed_dim, out_features=self.nodes)
      self.linear = nn.Linear(in_features=self.nbr_filters, out_features=self.nodes)
    else:
      self.fc = nn.Linear(in_features=inp_len * embed_dim, out_features=self.nodes)
      self.linear = nn.Linear(in_features=inp_len * self.nbr_filters, out_features=self.nodes)
    self.output = nn.Linear(in_features=self.nodes, out_features=1)
    if self.cnn_layers == -1:
      self.direct_output = nn.Linear(in_features=embed_dim, out_features=1)
    if self.magic:
      self.magic_output = nn.Linear(in_features=1, out_features=1)

  def forward(self,
              tokens: torch.Tensor = None,
              need_head_weights: bool = False,
              return_contacts: bool = False,
              return_representation: bool = True,
              return_attentions_symm: bool = False,
              return_attentions: bool = False,
              soft_token_probs: torch.Tensor = None):
    if soft_token_probs is not None:
      x_esm2 = self._forward_soft_tokens(
        soft_token_probs,
        need_head_weights=need_head_weights,
        return_representation=return_representation)
    else:
      x_esm2 = self.esm2(
        tokens,
        self.repr_layers,
        need_head_weights,
        return_contacts,
        return_representation,
        return_attentions_symm,
        return_attentions)

    if self.avg_emb:
      x = x_esm2['representations'][self.layers][:, 1:self.inp_len + 1].mean(1)
      x_o = x.unsqueeze(2)
    elif self.bos_emb:
      x = x_esm2['representations'][self.layers][:, 0]
      x_o = x.unsqueeze(2)
    else:
      x_o = x_esm2['representations'][self.layers][:, 1:self.inp_len + 1]
      x_o = x_o.permute(0, 2, 1)

    if self.cnn_layers >= 1:
      x_cnn1 = self.conv1(x_o)
      x_o = self.relu(x_cnn1)
    if self.cnn_layers >= 2:
      x_cnn2 = self.conv2(x_o)
      x_relu2 = self.relu(x_cnn2)
      x_o = self.dropout1(x_relu2)
    if self.cnn_layers >= 3:
      x_cnn3 = self.conv2(x_o)
      x_relu3 = self.relu(x_cnn3)
      x_o = self.dropout2(x_relu3)

    x = self.flatten(x_o)
    if self.cnn_layers != -1:
      if self.cnn_layers != 0:
        o_linear = self.linear(x)
      else:
        o_linear = self.fc(x)
      o_relu = self.relu(o_linear)
      o_dropout = self.dropout3_layer(o_relu)
      o = self.output(o_dropout)
    else:
      o = self.direct_output(x)

    if self.magic:
      o = self.magic_output(o)
    return o, x_esm2, self.esm2

  def _forward_soft_tokens(self,
                           token_probs: torch.Tensor,
                           need_head_weights: bool = False,
                           return_representation: bool = True):
    """
    Run ESM2 transformer on soft token distributions to keep gradients.
    token_probs: [B, T, V] over the ESM vocab.
    """
    if need_head_weights:
      raise NotImplementedError("need_head_weights is not supported for soft tokens.")
    embed_weight = self.esm2.embed_tokens.weight  # [V, E]
    x = torch.einsum('bsv,ve->bse', token_probs, embed_weight)
    x = self.esm2.embed_scale * x

    repr_layers = set(self.repr_layers)
    hidden_representations = {}
    if 0 in repr_layers:
      hidden_representations[0] = x

    # Transformer expects [T, B, E]
    x = x.transpose(0, 1)
    padding_mask = None  # all real tokens (no padding used in soft path)

    for layer_idx, layer in enumerate(self.esm2.layers):
      x, _ = layer(
        x,
        self_attn_padding_mask=padding_mask,
        need_head_weights=False,
      )
      if (layer_idx + 1) in repr_layers:
        hidden_representations[layer_idx + 1] = x.transpose(0, 1)

    x = self.esm2.emb_layer_norm_after(x)
    x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
    if (layer_idx + 1) in repr_layers:
      hidden_representations[layer_idx + 1] = x
    x = self.esm2.lm_head(x)

    if return_representation:
      return {"logits": x, "representations": hidden_representations}
    return {"logits": x}


class UTRLMOracle(nn.Module):
  """Run UTR-LM TE checkpoints and average their predictions."""

  def __init__(self,
               checkpoint_root: str,
               checkpoint_paths: Optional[Sequence[str]] = None,
               device: str = 'cuda',
               dataset_patterns: Sequence[str] = ('HEK', 'Muscle', 'pc3'),
               folds: Sequence[int] = tuple(range(10)),
               use_finetuned: bool = True,
               seq_trim_len: int = 100,
               batch_toks: int = 4096,
               mask_prob: float = 0.0,
               diffusion_base_order: Sequence[str] = ('A', 'C', 'G', 'T'),
               target_base_order: Sequence[str] = ('A', 'G', 'C', 'T')):
    if Alphabet is None:
      raise ImportError('UTRLMOracle requires the `esm` package to be installed.')
    super().__init__()
    self.checkpoint_root = Path(checkpoint_root)
    if not self.checkpoint_root.exists():
      raise FileNotFoundError(f'Checkpoint root not found: {self.checkpoint_root}')
    self.device = torch.device(device)
    self.seq_trim_len = seq_trim_len
    self.batch_toks = batch_toks
    self.mask_prob = mask_prob
    self.diffusion_base_order = tuple(diffusion_base_order)
    self.target_base_order = tuple(target_base_order)
    self.folds = tuple(folds)
    self.dataset_patterns = tuple(dataset_patterns)
    self.use_finetuned = use_finetuned

    self.alphabet = Alphabet(standard_toks='AGCT', mask_prob=self.mask_prob)
    self.batch_converter = self.alphabet.get_batch_converter()
    self.predictor_paths = self._select_checkpoints(checkpoint_paths)
    if not self.predictor_paths:
      raise FileNotFoundError(
        f'No checkpoints found in {self.checkpoint_root} matching requested filters.')

    predictors: List[UTRLMPredictor] = []
    for path in self.predictor_paths:
      predictor = UTRLMPredictor(
        alphabet=self.alphabet,
        predictor_path=path,
        inp_len=self.seq_trim_len,
        avg_emb=False,
        bos_emb=True)
      state_dict = torch.load(str(path), map_location='cpu')
      state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
      predictor.load_state_dict(state_dict, strict=False)
      predictor.to(self.device)
      predictor.eval()
      for param in predictor.parameters():
        param.requires_grad_(False)
      predictors.append(predictor)
    self.predictors = nn.ModuleList(predictors)

  def forward(self, x: torch.Tensor, soft_input: bool = False) -> torch.Tensor:
    if x.dim() != 3:
      raise ValueError(f'Expected 3D input tensor, got shape {x.shape}')
    if soft_input:
      base_probs = self._tensor_to_base_probs(x)
      per_ckpt_preds = []
      for predictor in self.predictors:
        preds = self._predict_with_soft_tokens(predictor, base_probs)
        per_ckpt_preds.append(preds)
      stacked = torch.stack(per_ckpt_preds, dim=0).mean(dim=0)
      return stacked.unsqueeze(-1)

    sequences = self._tensor_to_sequences(x)
    if not sequences:
      return torch.empty((0, 1), device=self.device)

    batches = self._build_batches(sequences)
    if not batches:
      return torch.empty((0, 1), device=self.device)

    per_ckpt_preds = []
    for predictor in self.predictors:
      preds = self._predict_with_model(predictor, batches)
      per_ckpt_preds.append(preds)
    stacked = torch.stack(per_ckpt_preds, dim=0).mean(dim=0)
    return stacked.unsqueeze(-1)

  def _select_checkpoints(self, explicit_paths: Optional[Sequence[str]]) -> List[Path]:
    if explicit_paths:
      paths = []
      for p in explicit_paths:
        path_obj = Path(p)
        if not path_obj.exists():
          raise FileNotFoundError(f'Explicit checkpoint not found: {path_obj}')
        paths.append(path_obj)
      return paths

    all_paths = sorted(self.checkpoint_root.glob('*.pt'))
    filtered = []
    finetune_token = 'finetuneTrue' if self.use_finetuned else 'finetuneFalse'
    for path in all_paths:
      name = path.name
      if '_TE_' not in name:
        continue
      if finetune_token not in name:
        continue
      if not any(pattern.lower() in name.lower() for pattern in self.dataset_patterns):
        continue
      filtered.append(path)

    by_dataset: Dict[str, Dict[int, List[Path]]] = {}
    for path in filtered:
      dataset_key = self._match_dataset(path.name)
      if dataset_key is None:
        continue
      dataset_entry = by_dataset.setdefault(dataset_key, {})
      fold_idx = self._extract_fold(path.name)
      if fold_idx is None or fold_idx not in self.folds:
        continue
      dataset_entry.setdefault(fold_idx, []).append(path)

    selected = []
    for dataset_key in self.dataset_patterns:
      folds_map = by_dataset.get(dataset_key.lower())
      if not folds_map:
        raise FileNotFoundError(f'No checkpoints found for dataset "{dataset_key}".')
      for fold_idx in self.folds:
        candidates = folds_map.get(fold_idx, [])
        if not candidates:
          raise FileNotFoundError(
            f'Checkpoint missing for dataset "{dataset_key}" fold {fold_idx}.')
        candidates = sorted(
          candidates,
          key=lambda p: self._extract_epoch(p.name),
          reverse=True)
        selected.append(candidates[0])
    return selected

  def _match_dataset(self, name: str) -> Optional[str]:
    lower = name.lower()
    for pattern in self.dataset_patterns:
      if pattern.lower() in lower:
        return pattern.lower()
    return None

  def _extract_fold(self, name: str) -> Optional[int]:
    match = re.search(r'_fold(\d+)_', name)
    return int(match.group(1)) if match else None

  def _extract_epoch(self, name: str) -> int:
    match = re.search(r'_epoch(\d+)\.pt$', name)
    return int(match.group(1)) if match else -1

  # def _tensor_to_sequences(self, tensor: torch.Tensor) -> List[str]:
  #   channel_first = self._ensure_channel_first(tensor)
  #   permuted = self._reorder_channels(channel_first)
  #   indices = permuted.detach().cpu().argmax(dim=1)
  #   base_map = {idx: base for idx, base in enumerate(self.target_base_order)}
  #   sequences = []
  #   for row in indices:
  #     chars = ''.join(base_map[int(idx)] for idx in row)
  #     chars = chars.replace('U', 'T')
  #     sequences.append(chars[-self.seq_trim_len:])
  #   return sequences
  def _tensor_to_sequences(self, tensor: torch.Tensor) -> List[str]:
      channel_first = self._ensure_channel_first(tensor)
      permuted = self._reorder_channels(channel_first)            # [B,4,L]

      valid = (permuted.sum(dim=1) > 0.1).detach().cpu()        # [B,L]
      indices = permuted.detach().cpu().argmax(dim=1)             # [B,L]

      base_map = {i: b for i, b in enumerate(self.target_base_order)}
      sequences = []
      for b in range(indices.shape[0]):
          vpos = valid[b].nonzero(as_tuple=True)[0]
          end = int(vpos[-1].item()) + 1 if len(vpos) > 0 else 0  # last valid + 1
          row = indices[b, :end]
          chars = ''.join(base_map[int(i)] for i in row).replace('U', 'T')
          chars = chars[-self.seq_trim_len:] if self.seq_trim_len is not None else chars
          sequences.append(chars)
      return sequences



  def _ensure_channel_first(self, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[1] == len(self.diffusion_base_order):
      return tensor
    if tensor.shape[2] == len(self.diffusion_base_order):
      return tensor.transpose(1, 2)
    raise ValueError(
      f'Channel dimension must equal {len(self.diffusion_base_order)}, got shape {tensor.shape}')

  def _reorder_channels(self, tensor: torch.Tensor) -> torch.Tensor:
    src_idx = {base: i for i, base in enumerate(self.diffusion_base_order)}
    perm = []
    for base in self.target_base_order:
      mapped = base
      if base == 'T' and base not in src_idx and 'U' in src_idx:
        mapped = 'U'
      if mapped not in src_idx:
        raise ValueError(
          f'Base {mapped} missing from source order {self.diffusion_base_order}')
      perm.append(src_idx[mapped])
    return tensor[:, perm, :]

  def _build_batches(self, sequences: List[str]) -> List[torch.Tensor]:
    labels = np.zeros(len(sequences), dtype=float)
    dataset = FastaBatchedDataset(labels, sequences, mask_prob=self.mask_prob)
    batches = dataset.get_batch_indices(toks_per_batch=self.batch_toks, extra_toks_per_seq=2)
    loader = torch.utils.data.DataLoader(
      dataset,
      collate_fn=self.batch_converter,
      batch_sampler=batches,
      shuffle=False)
    stored = []
    for _, _, _, toks, _, _ in loader:
      stored.append(toks.to(self.device, non_blocking=True))
    return stored

  def _predict_with_model(self,
                          predictor: UTRLMPredictor,
                          batches: List[torch.Tensor]) -> torch.Tensor:
    preds = []
    with torch.no_grad():
      for toks in batches:
        outputs, _, _ = predictor(
          toks,
          need_head_weights=False,
          return_contacts=False,
          return_representation=True,
          return_attentions_symm=False,
          return_attentions=False)
        preds.append(outputs.reshape(-1))
    return torch.cat(preds, dim=0)


class ResBlock(nn.Module):
  def __init__(
    self,
    in_planes,
    out_planes,
    stride=1,
    dilation=1,
    conv_layer=nn.Conv2d,
    norm_layer=nn.BatchNorm2d,
  ):
    super().__init__()
    self.bn1 = norm_layer(in_planes)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv1 = conv_layer(
      in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False)
    self.bn2 = norm_layer(out_planes)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv2 = conv_layer(out_planes, out_planes, kernel_size=3, padding=dilation, bias=False)

    if stride > 1 or out_planes != in_planes:
      self.downsample = nn.Sequential(
        conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
        norm_layer(out_planes),
      )
    else:
      self.downsample = None

  def forward(self, x):
    identity = x
    out = self.bn1(x)
    out = self.relu1(out)
    out = self.conv1(out)
    out = self.bn2(out)
    out = self.relu2(out)
    out = self.conv2(out)
    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    return out


class RNAFMUTRPredictor(nn.Module):
  """CNN head used in RNA-FM UTR function prediction tutorial."""

  def __init__(self, alphabet=None, task="rgs", arch="cnn", input_types=("emb-rnafm",)):
    super().__init__()
    self.alphabet = alphabet
    self.task = task
    self.arch = arch
    self.input_types = list(input_types)
    self.padding_mode = "right"
    self.token_len = 100
    self.out_plane = 1
    self.in_channels = 0
    if "seq" in self.input_types:
      self.in_channels += 4
    if "emb-rnafm" in self.input_types:
      self.reductio_module = nn.Linear(640, 32)
      self.in_channels += 32

    if self.arch == "cnn" and self.in_channels != 0:
      self.predictor = self._create_1dcnn_for_emb(in_planes=self.in_channels, out_planes=1)
    else:
      raise ValueError("Wrong Arch Type")

  def forward(self, tokens, inputs):
    ensemble_inputs = []
    if "seq" in self.input_types:
      nest_tokens = (tokens[:, 1:-1] - 4)
      nest_tokens = torch.nn.functional.pad(
        nest_tokens, (0, self.token_len - nest_tokens.shape[1]), value=-2)
      token_padding_mask = nest_tokens.ge(0).long()
      one_hot_tokens = torch.nn.functional.one_hot(
        (nest_tokens * token_padding_mask), num_classes=4)
      one_hot_tokens = one_hot_tokens.float() * token_padding_mask.unsqueeze(-1)
      one_hot_tokens = one_hot_tokens.permute(0, 2, 1)
      ensemble_inputs.append(one_hot_tokens)

    if "emb-rnafm" in self.input_types:
      embeddings = inputs["emb-rnafm"]
      embeddings, _ = self._remove_pend_tokens_1d(tokens, embeddings)
      embeddings = torch.nn.functional.pad(
        embeddings, (0, 0, 0, self.token_len - embeddings.shape[1]))
      embeddings = self.reductio_module(embeddings)
      embeddings = embeddings.permute(0, 2, 1)
      ensemble_inputs.append(embeddings)

    ensemble_inputs = torch.cat(ensemble_inputs, dim=1)
    output = self.predictor(ensemble_inputs).squeeze(-1)
    return output

  def _create_1dcnn_for_emb(self, in_planes, out_planes):
    main_planes = 64
    dropout = 0.2
    emb_cnn = nn.Sequential(
      nn.Conv1d(in_planes, main_planes, kernel_size=3, padding=1),
      ResBlock(main_planes, main_planes, stride=2, dilation=1, conv_layer=nn.Conv1d,
               norm_layer=nn.BatchNorm1d),
      ResBlock(main_planes, main_planes, stride=1, dilation=1, conv_layer=nn.Conv1d,
               norm_layer=nn.BatchNorm1d),
      ResBlock(main_planes, main_planes, stride=2, dilation=1, conv_layer=nn.Conv1d,
               norm_layer=nn.BatchNorm1d),
      ResBlock(main_planes, main_planes, stride=1, dilation=1, conv_layer=nn.Conv1d,
               norm_layer=nn.BatchNorm1d),
      ResBlock(main_planes, main_planes, stride=2, dilation=1, conv_layer=nn.Conv1d,
               norm_layer=nn.BatchNorm1d),
      ResBlock(main_planes, main_planes, stride=1, dilation=1, conv_layer=nn.Conv1d,
               norm_layer=nn.BatchNorm1d),
      nn.AdaptiveAvgPool1d(1),
      nn.Flatten(),
      nn.Dropout(dropout),
      nn.Linear(main_planes, out_planes),
    )
    return emb_cnn

  def _remove_pend_tokens_1d(self, tokens, seqs):
    padding_masks = tokens.ne(self.alphabet.padding_idx)

    if self.alphabet.append_eos:
      eos_masks = tokens.ne(self.alphabet.eos_idx)
      eos_pad_masks = (eos_masks & padding_masks).to(seqs)
      seqs = seqs * eos_pad_masks.unsqueeze(-1)
      seqs = seqs[:, ..., :-1, :]
      padding_masks = padding_masks[:, ..., :-1]

    if self.alphabet.prepend_bos:
      seqs = seqs[:, ..., 1:, :]
      padding_masks = padding_masks[:, ..., 1:]

    if not padding_masks.any():
      padding_masks = None

    return seqs, padding_masks


class RNAFMOracle(nn.Module):
  """Run RNA-FM backbone + UTR predictor for MRL scoring."""

  def __init__(self,
               predictor_checkpoint: str,
               backbone_path: Optional[str] = None,
               fm_root: Optional[str] = None,
               device: str = "cuda",
               seq_trim_len: int = 100,
               diffusion_base_order: Sequence[str] = ("A", "C", "G", "T"),
               target_base_order: Sequence[str] = ("A", "C", "G", "U"),
               max_batch_size: int = 32):
    super().__init__()
    self.device = torch.device(device)
    self.seq_trim_len = seq_trim_len
    self.diffusion_base_order = tuple(diffusion_base_order)
    self.target_base_order = tuple(target_base_order)
    self.max_batch_size = max_batch_size

    self.backbone, self.alphabet = _load_rnafm_backbone(
      backbone_path=backbone_path, fm_root=fm_root)
    self.backbone.to(self.device)
    self.backbone.eval()
    for param in self.backbone.parameters():
      param.requires_grad_(False)

    self.predictor = RNAFMUTRPredictor(
      alphabet=self.alphabet, input_types=("emb-rnafm",)).to(self.device)
    state = torch.load(predictor_checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
      state = state["state_dict"]
    if isinstance(state, nn.Module):
      state = state.state_dict()
    state = {k.replace("module.", ""): v for k, v in state.items()}
    self.predictor.load_state_dict(state, strict=False)
    self.predictor.eval()
    for param in self.predictor.parameters():
      param.requires_grad_(False)

  def forward(self, x: torch.Tensor, soft_input: bool = False) -> torch.Tensor:
    if x.dim() != 3:
      raise ValueError(f"Expected 3D input tensor, got shape {x.shape}")
    if soft_input:
      base_probs = self._tensor_to_base_probs(x)
      token_probs, padding_mask = self._base_probs_to_token_probs(base_probs)
      tokens = self._base_probs_to_tokens(base_probs, padding_mask)
      preds = self._predict_soft(tokens, token_probs, padding_mask)
      return preds.unsqueeze(-1)

    sequences = self._tensor_to_sequences(x)
    if not sequences:
      return torch.empty((0, 1), device=self.device)
    preds = self._predict_hard(sequences)
    return preds.unsqueeze(-1)

  def _ensure_channel_first(self, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[1] == len(self.diffusion_base_order):
      return tensor
    if tensor.shape[2] == len(self.diffusion_base_order):
      return tensor.transpose(1, 2)
    raise ValueError(
      f"Channel dimension must equal {len(self.diffusion_base_order)}, got shape {tensor.shape}")

  def _reorder_channels(self, tensor: torch.Tensor) -> torch.Tensor:
    src_idx = {base: i for i, base in enumerate(self.diffusion_base_order)}
    perm = []
    for base in self.target_base_order:
      mapped = base
      if base == "U" and "U" not in src_idx and "T" in src_idx:
        mapped = "T"
      if base == "T" and "T" not in src_idx and "U" in src_idx:
        mapped = "U"
      if mapped not in src_idx:
        raise ValueError(
          f"Base {mapped} missing from source order {self.diffusion_base_order}")
      perm.append(src_idx[mapped])
    return tensor[:, perm, :]

  def _tensor_to_base_probs(self, tensor: torch.Tensor) -> torch.Tensor:
    channel_first = self._ensure_channel_first(tensor)
    permuted = self._reorder_channels(channel_first)
    base_probs = permuted.transpose(1, 2)  # [B, L, 4]
    if self.seq_trim_len is not None:
      base_probs = base_probs[:, -self.seq_trim_len:, :]
    base_probs = base_probs / (base_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8))
    return base_probs

  # def _tensor_to_sequences(self, tensor: torch.Tensor) -> List[str]:
  #   base_probs = self._tensor_to_base_probs(tensor)
  #   indices = base_probs.detach().cpu().argmax(dim=-1)
  #   base_map = {idx: base for idx, base in enumerate(self.target_base_order)}
  #   sequences = []
  #   for row in indices:
  #     chars = "".join(base_map[int(idx)] for idx in row)
  #     sequences.append(chars)
  #   return sequences
  def _tensor_to_sequences(self, tensor: torch.Tensor) -> List[str]:
    base_probs = self._tensor_to_base_probs(tensor)     # expected [B, L, 4]
    # valid positions are those with some mass
    valid = (base_probs.sum(dim=-1) > 0)                # [B, L] bool

    indices = base_probs.detach().cpu().argmax(dim=-1)  # [B, L]
    valid_cpu = valid.detach().cpu()

    base_map = {idx: base for idx, base in enumerate(self.target_base_order)}
    sequences = []
    for row_idx in range(indices.shape[0]):
        row = indices[row_idx][valid_cpu[row_idx]]      # keep only valid positions
        chars = "".join(base_map[int(idx)] for idx in row)
        sequences.append(chars)
    return sequences

  def _base_to_token_idx(self, base: str) -> int:
    tok_to_idx = self.alphabet.tok_to_idx
    if base == "T" and "U" in tok_to_idx:
      base = "U"
    if base == "U" and "U" not in tok_to_idx and "T" in tok_to_idx:
      base = "T"
    if base not in tok_to_idx:
      raise ValueError(f"Base {base} missing from alphabet tokens {sorted(tok_to_idx.keys())}")
    return tok_to_idx[base]

  def _base_probs_to_tokens(self,
                            base_probs: torch.Tensor,
                            padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    base_token_ids = torch.tensor(
      [self._base_to_token_idx(base) for base in self.target_base_order],
      device=base_probs.device,
      dtype=torch.long,
    )
    hard_ids = base_probs.argmax(dim=-1)
    token_ids = base_token_ids[hard_ids]
    if padding_mask is not None:
      offset = int(self.alphabet.prepend_bos)
      seq_mask = padding_mask[:, offset:offset + base_probs.shape[1]]
      token_ids = token_ids.masked_fill(seq_mask, self.alphabet.padding_idx)
    if self.alphabet.prepend_bos:
      bos = torch.full(
        (token_ids.shape[0], 1),
        self.alphabet.cls_idx,
        device=token_ids.device,
        dtype=token_ids.dtype,
      )
      token_ids = torch.cat([bos, token_ids], dim=1)
    if self.alphabet.append_eos:
      eos = torch.full(
        (token_ids.shape[0], 1),
        self.alphabet.eos_idx,
        device=token_ids.device,
        dtype=token_ids.dtype,
      )
      token_ids = torch.cat([token_ids, eos], dim=1)
    return token_ids

  def _base_probs_to_token_probs(self,
                                 base_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, _ = base_probs.shape
    vocab_size = len(self.alphabet.all_toks)
    extra = int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos)
    token_probs = base_probs.new_zeros((batch_size, seq_len + extra, vocab_size))
    padding_mask = torch.zeros(
      (batch_size, seq_len + extra), device=base_probs.device, dtype=torch.bool)

    offset = 0
    if self.alphabet.prepend_bos:
      token_probs[:, 0, self.alphabet.cls_idx] = 1.0
      offset = 1
    if self.alphabet.append_eos:
      token_probs[:, offset + seq_len, self.alphabet.eos_idx] = 1.0

    valid = base_probs.sum(dim=-1) > 0
    padding_mask[:, offset:offset + seq_len] = ~valid

    base_token_ids = torch.tensor(
      [self._base_to_token_idx(base) for base in self.target_base_order],
      device=base_probs.device,
      dtype=torch.long,
    )
    token_probs[:, offset:offset + seq_len, :].scatter_(
      dim=2,
      index=base_token_ids.view(1, 1, -1).expand(batch_size, seq_len, 4),
      src=base_probs,
    )
    if padding_mask.any().item():
      pad_idx = self.alphabet.padding_idx
      token_probs[:, offset:offset + seq_len, :] = token_probs[
        :, offset:offset + seq_len, :].masked_fill(~valid.unsqueeze(-1), 0)
      token_probs[:, offset:offset + seq_len, pad_idx] = token_probs[
        :, offset:offset + seq_len, pad_idx].masked_fill(~valid, 1.0)
    return token_probs, padding_mask

  def _predict_hard(self, sequences: List[str]) -> torch.Tensor:
    labels = [str(i) for i in range(len(sequences))]
    batch_converter = self.alphabet.get_batch_converter()
    preds = []
    for start in range(0, len(sequences), self.max_batch_size):
      chunk = sequences[start:start + self.max_batch_size]
      _, _, toks = batch_converter(list(zip(labels[start:start + len(chunk)], chunk)))
      toks = toks.to(self.device, non_blocking=True)
      results = self.backbone(
        toks,
        need_head_weights=False,
        repr_layers=[12],
        return_contacts=False,
      )
      inputs = {"emb-rnafm": results["representations"][12]}
      preds.append(self.predictor(toks, inputs).reshape(-1))
    return torch.cat(preds, dim=0)

  def _predict_soft(self,
                    tokens: torch.Tensor,
                    token_probs: torch.Tensor,
                    padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    preds = []
    for start in range(0, token_probs.shape[0], self.max_batch_size):
      end = min(start + self.max_batch_size, token_probs.shape[0])
      slice_probs = token_probs[start:end]
      slice_tokens = tokens[start:end]
      slice_mask = padding_mask[start:end] if padding_mask is not None else None
      reps = self._forward_soft_tokens(slice_probs, repr_layers=[12], padding_mask=slice_mask)
      inputs = {"emb-rnafm": reps["representations"][12]}
      preds.append(self.predictor(slice_tokens.to(self.device), inputs).reshape(-1))
    return torch.cat(preds, dim=0)

  def _forward_soft_tokens(self,
                           token_probs: torch.Tensor,
                           repr_layers: Sequence[int],
                           padding_mask: Optional[torch.Tensor] = None):
    model = self.backbone
    embed_weight = model.embed_tokens.weight
    x = torch.einsum("bsv,ve->bse", token_probs, embed_weight)
    x = model.embed_scale * x

    dummy_tokens = torch.full(
      token_probs.shape[:2],
      model.cls_idx,
      device=token_probs.device,
      dtype=torch.long,
    )
    x = x + model.embed_positions(dummy_tokens)

    if model.model_version == "ESM-1b":
      if getattr(model, "emb_layer_norm_before", None):
        x = model.emb_layer_norm_before(x)

    repr_layers = set(repr_layers)
    hidden_representations = {}
    if 0 in repr_layers:
      hidden_representations[0] = x

    x = x.transpose(0, 1)
    if padding_mask is not None:
      padding_mask = padding_mask.to(device=x.device)
    for layer_idx, layer in enumerate(model.layers):
      x, _ = layer(x, self_attn_padding_mask=padding_mask, need_head_weights=False)
      if (layer_idx + 1) in repr_layers:
        hidden_representations[layer_idx + 1] = x.transpose(0, 1)

    if model.model_version == "ESM-1b":
      x = model.emb_layer_norm_after(x)
      x = x.transpose(0, 1)
      if (layer_idx + 1) in repr_layers:
        hidden_representations[layer_idx + 1] = x
    else:
      x = x.transpose(0, 1)
      if (layer_idx + 1) in repr_layers:
        hidden_representations[layer_idx + 1] = x

    return {"representations": hidden_representations}


def _load_rnafm_backbone(backbone_path: Optional[str], fm_root: Optional[str]):
  try:
    import fm  # type: ignore
    return fm.pretrained.rna_fm_t12(backbone_path)
  except Exception:
    fm_root = Path(fm_root) if fm_root else Path(__file__).resolve().parents[2] / "RNA-FM"
    if not fm_root.exists():
      raise FileNotFoundError(f"RNA-FM root not found: {fm_root}")
    fm_pkg = sys.modules.get("fm")
    if fm_pkg is None:
      fm_pkg = types.ModuleType("fm")
      fm_pkg.__path__ = [str(fm_root / "fm")]
      sys.modules["fm"] = fm_pkg
    data_mod = importlib.import_module("fm.data")
    esm1_mod = importlib.import_module("fm.model.esm1")
    fm_pkg.Alphabet = data_mod.Alphabet
    fm_pkg.BatchConverter = data_mod.BatchConverter
    fm_pkg.FastaBatchedDataset = data_mod.FastaBatchedDataset
    fm_pkg.BioBertModel = esm1_mod.BioBertModel
    pretrained_mod = importlib.import_module("fm.pretrained")
    return pretrained_mod.rna_fm_t12(backbone_path)

  def _tensor_to_base_probs(self, tensor: torch.Tensor) -> torch.Tensor:
    """Reorder to target base order, trim, and keep soft base probabilities."""
    channel_first = self._ensure_channel_first(tensor)
    permuted = self._reorder_channels(channel_first)
    base_probs = permuted.transpose(1, 2)  # [B, L, 4] in A/G/C/T order
    if self.seq_trim_len is not None:
      base_probs = base_probs[:, -self.seq_trim_len:, :]
    return base_probs

  def _predict_with_soft_tokens(self,
                                predictor: UTRLMPredictor,
                                base_probs: torch.Tensor) -> torch.Tensor:
    """Differentiable path: map soft base probs to soft ESM tokens."""
    batch_size, seq_len, _ = base_probs.shape
    vocab_size = len(self.alphabet.all_toks)
    token_probs = base_probs.new_zeros((batch_size, seq_len + 2, vocab_size))

    # Add <cls> at position 0 and <eos> at the end (one-hot, no gradients needed).
    token_probs[:, 0, self.alphabet.cls_idx] = 1.0
    token_probs[:, -1, self.alphabet.eos_idx] = 1.0

    # Map base channels A/G/C/T to alphabet indices.
    base_token_ids = torch.tensor(
      [self.alphabet.tok_to_idx['A'],
       self.alphabet.tok_to_idx['G'],
       self.alphabet.tok_to_idx['C'],
       self.alphabet.tok_to_idx['T']],
      device=base_probs.device,
      dtype=torch.long)
    token_probs[:, 1:-1, :].scatter_(
      dim=2,
      index=base_token_ids.view(1, 1, -1).expand(batch_size, seq_len, 4),
      src=base_probs)

    preds = []
    for start in range(0, batch_size, 16):
      end = min(start + 16, batch_size)
      slice_probs = token_probs[start:end]
      outputs, _, _ = predictor(
        tokens=None,
        need_head_weights=False,
        return_contacts=False,
        return_representation=True,
        return_attentions_symm=False,
        return_attentions=False,
        soft_token_probs=slice_probs)
      preds.append(outputs.reshape(-1))
    return torch.cat(preds, dim=0)


def get_mttrans_oracle(config_path,
                       checkpoint_path=None,
                       checkpoint_paths=None,
                       checkpoint_dir=None,
                       tasks=None,
                       task_map=None,
                       task='MPA_H',
                       trim_len=100,
                       device='cuda',
                       vocab_json_path=None,
                       pad_token='<pad>',
                       eos_token='<eos>',
                       motif_trim_len=None):
    return MTTransOracle(config_path=config_path,
                         checkpoint_path=checkpoint_path,
                         checkpoint_paths=checkpoint_paths,
                         checkpoint_dir=checkpoint_dir,
                         tasks=tasks,
                         task_map=task_map,
                         task=task,
                         trim_len=trim_len,
                         device=device,
                         vocab_json_path=vocab_json_path,
                         pad_token=pad_token,
                         eos_token=eos_token,
                         motif_trim_len=motif_trim_len)


def get_utrlm_oracle(checkpoint_root: str,
                     checkpoint_paths: Optional[Sequence[str]] = None,
                     device: str = 'cuda',
                     dataset_patterns: Sequence[str] = ('HEK', 'Muscle', 'pc3'),
                     folds: Sequence[int] = tuple(range(10)),
                     use_finetuned: bool = True,
                     seq_trim_len: int = 100):
    return UTRLMOracle(
        checkpoint_root=checkpoint_root,
        checkpoint_paths=checkpoint_paths,
        device=device,
        dataset_patterns=dataset_patterns,
        folds=folds,
        use_finetuned=use_finetuned,
        seq_trim_len=seq_trim_len)


def get_rnafm_oracle(predictor_checkpoint: str,
                     backbone_path: Optional[str] = None,
                     fm_root: Optional[str] = None,
                     device: str = "cuda",
                     seq_trim_len: int = 100):
  return RNAFMOracle(
    predictor_checkpoint=predictor_checkpoint,
    backbone_path=backbone_path,
    fm_root=fm_root,
    device=device,
    seq_trim_len=seq_trim_len,
  )


POSTPROC_MEAN_STD = {
    "MPA_U": {"mean": 6.532835230673117, "std": 1.577417132818392},
    "MPA_H": {"mean": 5.786840247447307, "std": 1.5859451175340078},
    "MPA_V": {"mean": 5.269930466671135, "std": 1.355184160931213},
    "RP_293T": {"mean": -0.615189054995973, "std": 1.060197697174912},
    "RP_muscle": {"mean": -0.21022818608263194, "std": 1.4772718609075497},
    "RP_PC3": {"mean": -0.33492097824717426, "std": 0.9745174360808637},
}
