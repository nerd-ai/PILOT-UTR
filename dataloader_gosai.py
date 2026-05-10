import torch
import pandas as pd
import typing
import math
try:
  from DRAKES.drakes_dna import utils as drakes_utils
except ImportError:  # pragma: no cover - fallback when module executed as script
  import utils as drakes_utils
import logging
import numpy as np
import os
import json
import sys
import csv
from collections import defaultdict, Counter
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

GENETIC_BPE_ROOT = Path(__file__).resolve().parents[2] / "GeneticBPE"
if GENETIC_BPE_ROOT.exists() and str(GENETIC_BPE_ROOT) not in sys.path:
  sys.path.insert(0, str(GENETIC_BPE_ROOT))

try:
  from genetic_bpe.tokenizer import GeneticBPETokenizer  # type: ignore
except ImportError:  # Optional; not required for simple-vocab PILOT-UTR workflows.
  GeneticBPETokenizer = None

base_path = '/data/scratch/wangchy/seqft/'
# When oracle_new imports MTtrans, it aliases sys.modules['utils'] to MTtrans.utils,
# which does not provide get_logger. Fall back to stdlib logging in that case.
if hasattr(drakes_utils, 'get_logger'):
  LOGGER = drakes_utils.get_logger(__name__)
else:  # pragma: no cover - defensive fallback
  LOGGER = logging.getLogger(__name__)
DNA_ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3} #, 'M': 4}
INDEX_TO_DNA = {v: k for k, v in DNA_ALPHABET.items()}
lookup_array = np.array([INDEX_TO_DNA[i] for i in range(len(INDEX_TO_DNA))])

def dna_detokenize(seq):
  return ''.join([list(DNA_ALPHABET.keys())[int(i)] for i in seq])

def batch_dna_detokenize(batch_seq):
    """
    batch_seq: numpy array of shape [batch_size, seq_len]
    return: list of strings
    """
    detokenized_batch = lookup_array[batch_seq]
    detokenized_batch = [''.join(seq) for seq in detokenized_batch]
    return detokenized_batch

def dna_tokenize(seq):
  return [DNA_ALPHABET[c] for c in seq]

def batch_dna_tokenize(batch_seq):
    """
    batch_seq: list of strings
    return: numpy array of shape [batch_size, seq_len]
    """
    tokenized_batch = np.array([[DNA_ALPHABET[c] for c in seq] for seq in batch_seq])
    return tokenized_batch

class GosaiDataset(torch.utils.data.Dataset):
    def __init__(self):
        data_df = pd.read_csv(os.path.join(base_path, f'mdlm/gosai_data/processed_data/gosai_all.csv'))
        self.seqs = torch.tensor(data_df['seq'].apply(lambda x: [DNA_ALPHABET[c] for c in x]).tolist())
        self.clss = torch.tensor(data_df[['hepg2', 'k562', 'sknsh']].to_numpy())
        LOGGER.info(f'Loaded data: seqs shape: {self.seqs.shape}, clss shape: {self.clss.shape}')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {'seqs': self.seqs[idx], 'clss': self.clss[idx], 'attention_mask': torch.ones(len(self.seqs[idx]))}


def get_datasets_gosai():
  return GosaiDataset()


class SimpleEncoding:
  def __init__(self, ids, attention_mask):
    self.ids = ids
    self.input_ids = ids  # Alias for HF compatibility
    self.attention_mask = attention_mask


class SimpleVocabTokenizer:
  """Tokenizer that performs a 1:1 character-to-id mapping based on a vocab file."""

  def __init__(self, vocab_dict, pad_token='N', eos_token='EOS', unk_token=None, normalize_case=True):
    if pad_token not in vocab_dict:
      raise ValueError(f'Pad token "{pad_token}" must be present in the vocabulary.')
    if eos_token is not None and eos_token not in vocab_dict:
      raise ValueError(f'EOS token "{eos_token}" must be present in the vocabulary.')
    self.token2id = vocab_dict
    self.id2token = {idx: tok for tok, idx in vocab_dict.items()}
    self.pad_token = pad_token
    self.unk_token = unk_token
    self.eos_token = eos_token
    self.pad_token_id = vocab_dict[pad_token]
    self.unk_token = unk_token if unk_token in vocab_dict else None
    self.unk_token_id = vocab_dict.get(unk_token, None)
    self.eos_token_id = vocab_dict[eos_token] if eos_token is not None else None
    self.normalize_case = normalize_case

  def encode(self, text, max_length: typing.Optional[int] = None) -> SimpleEncoding:
    ids = []
    for ch in text:
      token = ch.upper() if self.normalize_case and isinstance(ch, str) else ch
      if ch in self.token2id:
        ids.append(self.token2id[ch])
      elif token in self.token2id:
        ids.append(self.token2id[token])
      elif self.unk_token_id is not None:
        ids.append(self.unk_token_id)
      else:
        raise ValueError(f'Character "{ch}" not found in vocabulary and no unk token provided.')
    # append EOS once at the end if configured
    if self.eos_token_id is not None:
      ids.append(self.eos_token_id)

    # --- 3. Check Length Guarantee ---
    if max_length is not None and len(ids) > max_length:
      # Raise error to ensure data quality (or you can truncate here if preferred)
      raise ValueError(f"Sequence length ({len(ids)}) exceeds max_length ({max_length}).")

    # --- 4. Create Mask (1 for Content+EOS) ---
    attention_mask = [1] * len(ids)

    # --- 5. Apply Padding ---
    if max_length is not None:
      padding_needed = max_length - len(ids)
      if padding_needed > 0:
        ids.extend([self.pad_token_id] * padding_needed)
        # Mask = 0 for padding
        attention_mask.extend([0] * padding_needed)


    return SimpleEncoding(ids, attention_mask)

  def encode_batch(self, texts):
    return [self.encode(text) for text in texts]

  def decode(self, ids):
    tokens = []
    for idx in ids:
      if self.eos_token_id is not None and idx == self.eos_token_id:
        break
      if idx == self.pad_token_id:
        if self.eos_token_id is None:
          break
        continue
      token = self.id2token.get(idx, '')
      if token == self.unk_token or token is None:
        continue
      tokens.append(token)
    return ''.join(tokens)

  def decode_batch(self, sequences):
    return [self.decode(seq) for seq in sequences]

  def token_to_id(self, token):
    return self.token2id.get(token, None)

  def get_vocab_size(self):
    return len(self.token2id)


# class GeneticBPETorchTokenizer:
#   """Wrap GeneticBPETokenizer to match the minimal interface expected by datasets."""

#   def __init__(self,
#                inner_tokenizer: GeneticBPETokenizer,
#                max_length: int,
#                pad_token: typing.Optional[str] = '<pad>',
#                unk_token: typing.Optional[str] = None):
#     self.inner = inner_tokenizer
#     self.max_length = max_length
#     ordered_vocab = self.inner._ordered_vocab()
#     self.token2id = {token: idx for token, idx in ordered_vocab}
#     self.id2token = {idx: token for token, idx in ordered_vocab}

#     next_id = len(self.token2id)
#     self.pad_token = pad_token
#     self.pad_token_id = None
#     if pad_token is not None:
#       if pad_token not in self.token2id:
#         self.token2id[pad_token] = next_id
#         self.id2token[next_id] = pad_token
#         next_id += 1
#       self.pad_token_id = self.token2id[pad_token]

#     self.unk_token = unk_token if unk_token else None
#     self.unk_token_id = None
#     if self.unk_token is not None:
#       if self.unk_token not in self.token2id:
#         self.token2id[self.unk_token] = next_id
#         self.id2token[next_id] = self.unk_token
#         next_id += 1
#       self.unk_token_id = self.token2id[self.unk_token]

#   def encode(self, text: str) -> SimpleEncoding:
#     tokens = self.inner.tokenize_new(text)
#     ids = []
#     for token in tokens:
#       token_id = self.token2id.get(token)
#       if token_id is None:
#         if self.unk_token_id is not None:
#           token_id = self.unk_token_id
#         else:
#           raise ValueError(f'Token "{token}" not in vocabulary and no unk token defined.')
#       ids.append(token_id)
#     if self.max_length is not None:
#       ids = ids[:self.max_length]
#     return SimpleEncoding(ids)

#   def encode_batch(self, texts: typing.List[str]) -> typing.List[SimpleEncoding]:
#     return [self.encode(text) for text in texts]

#   def decode(self, ids: typing.Iterable[int]) -> str:
#     tokens = []
#     for idx in ids:
#       if self.pad_token_id is not None and idx == self.pad_token_id:
#         continue
#       if self.unk_token_id is not None and idx == self.unk_token_id:
#         continue
#       token = self.id2token.get(idx)
#       if token is not None:
#         tokens.append(token)
#     return ''.join(tokens)

#   def token_to_id(self, token: str) -> typing.Optional[int]:
#     return self.token2id.get(token)

#   def get_vocab_size(self) -> int:
#     return len(self.token2id)



class MotifAwareTokenizer:
  """Tokenizer that matches motifs from a pre-defined vocab JSON."""

  def __init__(self,
               vocab_json_path: typing.Union[str, Path],
               pad_token: str = '<pad>',
               eos_token: typing.Optional[str] = '<eos>',
               base_tokens: typing.Sequence[str] = ("A", "C", "G", "T"),
               max_length: typing.Optional[int] = None,
               trim_to: typing.Optional[int] = None):
    vocab_json_path = Path(vocab_json_path)
    if not vocab_json_path.exists():
      raise FileNotFoundError(f'Vocabulary JSON not found: {vocab_json_path}')
    with vocab_json_path.open() as handle:
      raw_vocab = json.load(handle)
    if not isinstance(raw_vocab, dict):
      raise ValueError(f'Vocabulary JSON must map token -> id: {vocab_json_path}')
    self.token2id = dict(raw_vocab)
    for base in base_tokens:
      if base not in self.token2id:
        raise ValueError(f'Base token "{base}" missing from {vocab_json_path}.')
    self.pad_token = pad_token
    self.eos_token = eos_token
    max_id = max(self.token2id.values()) if self.token2id else -1
    if self.pad_token not in self.token2id:
      self.token2id[self.pad_token] = max_id + 1
    self.pad_token_id = self.token2id[self.pad_token]
    if self.eos_token is not None:
      if self.eos_token not in self.token2id:
        raise ValueError(f'EOS token "{self.eos_token}" must be present in the vocabulary.')
      self.eos_token_id = self.token2id[self.eos_token]
    else:
      self.eos_token_id = None
    self.id2token = {idx: tok for tok, idx in self.token2id.items()}
    self.base_tokens = tuple(base_tokens)
    self.trim_to = trim_to
    self.default_max_length = max_length
    self.motifs_by_length: typing.Dict[int, typing.List[str]] = defaultdict(list)
    for token in self.token2id:
      if token in self.base_tokens or token == self.pad_token or token == self.eos_token:
        continue
      self.motifs_by_length[len(token)].append(token)
    self.max_motif_len = max(self.motifs_by_length) if self.motifs_by_length else 1

  def _canonicalize(self, sequence: str) -> str:
    seq = sequence.upper().replace('U', 'T')
    if self.trim_to is not None:
      seq = seq[-self.trim_to:]
    return seq

  def _tokenize(self, sequence: str) -> typing.List[str]:
    cleaned = self._canonicalize(sequence)
    tokens: typing.List[str] = []
    i = 0
    n = len(cleaned)
    while i < n:
      matched = False
      for length in range(self.max_motif_len, 1, -1):
        motifs = self.motifs_by_length.get(length)
        if not motifs or i + length > n:
          continue
        segment = cleaned[i:i + length]
        if segment in motifs:
          tokens.append(segment)
          i += length
          matched = True
          break
      if not matched:
        tokens.append(cleaned[i])
        i += 1
    return tokens

  def encode(self, sequence: str, max_length: typing.Optional[int] = None) -> SimpleEncoding:
    max_length = max_length or self.default_max_length
    tokens = self._tokenize(sequence)
    ids: typing.List[int] = []
    for token in tokens:
      token_id = self.token2id.get(token)
      if token_id is None:
        raise ValueError(f'Unexpected token {token} encountered during encoding')
      ids.append(token_id)
    # Add the EOS token in the end if configured
    if self.eos_token_id is not None:
      ids.append(self.eos_token_id)
    attention_mask = [1] * len(ids)

        # 4. Padding (Critical for fixed-size Tensor shapes)
    if max_length is not None:
      padding_needed = max_length - len(ids)
      
      # Safety check: Raises error if your guarantee is violated
      if padding_needed < 0:
        raise ValueError(
            f"Sequence length ({len(ids)}) exceeds max_length ({max_length}). "
            "Please check your hyperparameters."
        )
      
      if padding_needed > 0:
        ids.extend([self.pad_token_id] * padding_needed)
        # Padding = 0 (Ignore loss here)
        attention_mask.extend([0] * padding_needed)

    return SimpleEncoding(ids, attention_mask)
  
  def encode_batch(self, sequences: typing.Iterable[str], max_length: typing.Optional[int] = None) -> typing.List[SimpleEncoding]:
    return [self.encode(seq, max_length=max_length) for seq in sequences]

  def decode(self, ids: typing.Iterable[int]) -> str:
    tokens = []
    for idx in ids:
      if self.eos_token_id is not None and idx == self.eos_token_id:
        break
      if idx == self.pad_token_id:
        if self.eos_token_id is None:
          break
        continue
      token = self.id2token.get(idx)
      if token is not None:
        tokens.append(token)
    return ''.join(tokens)

  def decode_batch(self, sequences: typing.Iterable[typing.Iterable[int]]) -> typing.List[str]:
    return [self.decode(seq) for seq in sequences]

  def token_to_id(self, token: str) -> typing.Optional[int]:
    return self.token2id.get(token)

  def get_vocab_size(self) -> int:
    return len(self.token2id)

# class MotifTokenizer:
#   """Tokenizer that emits motif tokens first, then single-base tokens."""

#   def __init__(self,
#                vocab_path: typing.Union[str, Path],
#                fimo_path: typing.Optional[typing.Union[str, Path]] = None,
#                base_tokens: typing.Sequence[str] = ("A", "C", "G", "T"),
#                pad_token: str = "<pad>",
#                max_length: typing.Optional[int] = None):
#     vocab_path = Path(vocab_path)
#     if not vocab_path.exists():
#       raise FileNotFoundError(f'Vocabulary file not found: {vocab_path}')
#     with vocab_path.open() as handle:
#       raw_vocab = json.load(handle)
#     for base in base_tokens:
#       if base not in raw_vocab:
#         raise ValueError(f'Base token "{base}" missing from vocabulary {vocab_path}.')
#     self.token2id = dict(raw_vocab)
#     self.id2token = {idx: tok for tok, idx in self.token2id.items()}
#     self.pad_token = pad_token
#     if pad_token not in self.token2id:
#       pad_id = len(self.token2id)
#       self.token2id[pad_token] = pad_id
#       self.id2token[pad_id] = pad_token
#     self.pad_token_id = self.token2id[pad_token]
#     self.base_tokens = tuple(base_tokens)
#     self.base_token_ids = {base: self.token2id[base] for base in self.base_tokens}
#     self.motif_tokens = {
#       token for token in self.token2id
#       if token not in self.base_tokens and token != pad_token
#     }
#     motif_lengths = [len(token) for token in self.motif_tokens]
#     self.min_motif_len = min(motif_lengths) if motif_lengths else 0
#     self.max_motif_len = max(motif_lengths) if motif_lengths else 0
#     self.motifs_by_length = {
#       length: {token for token in self.motif_tokens if len(token) == length}
#       for length in range(self.min_motif_len, self.max_motif_len + 1)
#       if self.motif_tokens
#     }
#     self.fimo_spans = self._load_fimo_spans(Path(fimo_path)) if fimo_path else {}
#     self.default_max_length = max_length

#   def _load_fimo_spans(self, fimo_path: Path):
#     if not fimo_path.exists():
#       raise FileNotFoundError(f'FIMO results not found: {fimo_path}')
#     spans = defaultdict(list)
#     with fimo_path.open() as handle:
#       reader = csv.DictReader(handle, delimiter='\t')
#       for row in reader:
#         seq_name = row.get("sequence_name")
#         start_str = row.get("start")
#         stop_str = row.get("stop")
#         matched = row.get("matched_sequence")
#         if not seq_name or not start_str or not stop_str or not matched:
#           continue
#         start = int(start_str) - 1  # convert to 0-based inclusive
#         stop = int(stop_str)  # 1-based inclusive -> exclusive after cast
#         score = float(row.get("score", 0.0)) if row.get("score") else 0.0
#         matched = matched.strip().upper().replace("T", "U")
#         spans[seq_name].append((start, stop, matched, score))
#     deduped = {}
#     for seq_name, seq_spans in spans.items():
#       seq_spans.sort(key=lambda item: (item[0], -(item[1] - item[0]), -item[3]))
#       filtered = []
#       current_end = -1
#       for start, stop, token, score in seq_spans:
#         if token not in self.token2id:
#           continue
#         if start < current_end:
#           continue
#         filtered.append((start, stop, token))
#         current_end = stop
#       if filtered:
#         deduped[seq_name] = filtered
#     return deduped

#   def _tokenize_with_spans(self,
#                            sequence: str,
#                            spans: typing.List[typing.Tuple[int, int, str]]) -> typing.List[str]:
#     tokens: typing.List[str] = []
#     pos = 0
#     seq_len = len(sequence)
#     for start, stop, token in spans:
#       if start >= seq_len:
#         break
#       if start < pos:
#         continue
#       if start > pos:
#         for ch in sequence[pos:start]:
#           tokens.append(ch)
#       end = min(stop, seq_len)
#       if token in self.token2id:
#         tokens.append(token)
#       else:
#         for ch in sequence[start:end]:
#           tokens.append(ch)
#       pos = end
#     if pos < seq_len:
#       for ch in sequence[pos:]:
#         tokens.append(ch)
#     return tokens

#   def _tokenize_with_kmers(self, sequence: str) -> typing.List[str]:
#     if not self.motif_tokens:
#       return list(sequence)
#     tokens: typing.List[str] = []
#     i = 0
#     n = len(sequence)
#     while i < n:
#       matched = False
#       for length in range(self.max_motif_len, self.min_motif_len - 1, -1):
#         if length <= 1 or i + length > n:
#           continue
#         motif_set = self.motifs_by_length.get(length)
#         if not motif_set:
#           continue
#         candidate = sequence[i:i + length]
#         if candidate in motif_set:
#           tokens.append(candidate)
#           i += length
#           matched = True
#           break
#       if not matched:
#         tokens.append(sequence[i])
#         i += 1
#     return tokens

#   def _tokenize_sequence(self, sequence_name: str, sequence: str) -> typing.List[str]:
#     spans = self.fimo_spans.get(sequence_name)
#     if spans:
#       return self._tokenize_with_spans(sequence, spans)
#     return self._tokenize_with_kmers(sequence)

#   def encode_sequence(self,
#                       sequence_name: str,
#                       sequence: str,
#                       max_length: typing.Optional[int] = None) -> SimpleEncoding:
#     if max_length is None:
#       max_length = self.default_max_length
#     cleaned = sequence.upper().replace("T", "U")
#     tokens = self._tokenize_sequence(sequence_name, cleaned)
#     ids = []
#     for token in tokens:
#       token_id = self.token2id.get(token)
#       if token_id is None:
#         if len(token) != 1 or token not in self.base_token_ids:
#           raise ValueError(f'Unexpected token "{token}" produced during encoding.')
#         token_id = self.base_token_ids[token]
#       ids.append(token_id)
#     if max_length is not None:
#       ids = ids[:max_length]
#     return SimpleEncoding(ids)

#   def encode_batch(self,
#                    sequences: typing.Iterable[typing.Union[str, typing.Tuple[str, str]]],
#                    max_length: typing.Optional[int] = None) -> typing.List[SimpleEncoding]:
#     if max_length is None:
#       max_length = self.default_max_length
#     prepared: typing.List[typing.Tuple[str, str]] = []
#     for idx, item in enumerate(sequences):
#       if isinstance(item, tuple):
#         prepared.append(item)
#       else:
#         prepared.append((f"seq_{idx}", item))
#     return [self.encode_sequence(name, seq, max_length=max_length) for name, seq in prepared]

#   def decode(self, ids: typing.Iterable[int]) -> str:
#     tokens = []
#     for idx in ids:
#       if idx == self.pad_token_id:
#         continue
#       token = self.id2token.get(idx)
#       if token is not None:
#         tokens.append(token)
#     return ''.join(tokens)

#   def decode_batch(self, sequences: typing.Iterable[typing.Iterable[int]]) -> typing.List[str]:
#     return [self.decode(seq) for seq in sequences]

#   def token_to_id(self, token: str) -> typing.Optional[int]:
#     return self.token2id.get(token)

#   def get_vocab_size(self) -> int:
#     return len(self.token2id)



def build_bpe_tokenizer(vocab_path,
                        merges_path,
                        max_len,
                        pad_token='<pad>',
                        unk_token=None):
  bpe_kwargs = {}
  if unk_token:
    bpe_kwargs['unk_token'] = unk_token
  model = BPE.from_file(vocab_path, merges_path, **bpe_kwargs)
  tokenizer = Tokenizer(model)
  tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

  if pad_token and tokenizer.token_to_id(pad_token) is None:
    tokenizer.add_special_tokens([pad_token])
  pad_id = tokenizer.token_to_id(pad_token) if pad_token else None
  tokenizer.enable_truncation(max_length=max_len)
  return tokenizer, pad_id


# def build_genetic_bpe_tokenizer(state_path: typing.Union[str, Path],
#                                 motif_path: typing.Union[str, Path],
#                                 config_path: typing.Optional[typing.Union[str, Path]],
#                                 max_len: int,
#                                 pad_token: typing.Optional[str] = '<pad>',
#                                 unk_token: typing.Optional[str] = None):
#   state_path = Path(state_path)
#   motif_path = Path(motif_path)
#   if config_path is not None:
#     config_path = Path(config_path)

#   if not state_path.exists():
#     raise FileNotFoundError(f'GeneticBPE state file not found: {state_path}')
#   if not motif_path.exists():
#     raise FileNotFoundError(f'Motif bank file not found: {motif_path}')

#   tokenizer = GeneticBPETokenizer(
#     vocab_size=0,  # Actual vocab size restored from state.
#     min_freq=0,
#     config_path=str(config_path) if config_path else None,
#     motif_file=str(motif_path)
#   )
#   tokenizer.load(str(state_path))
#   wrapper = GeneticBPETorchTokenizer(
#     tokenizer,
#     max_length=max_len,
#     pad_token=pad_token,
#     unk_token=unk_token
#   )
#   pad_id = wrapper.pad_token_id
#   return wrapper, pad_id


# def build_motif_tokenizer(vocab_path: typing.Union[str, Path],
#                           max_len: typing.Optional[int],
#                           pad_token: str = '<pad>',
#                           fimo_path: typing.Optional[typing.Union[str, Path]] = None,
#                           use_fimo: bool = False,
#                           base_tokens: typing.Optional[typing.Sequence[str]] = None):
#   if vocab_path is None:
#     raise ValueError('motif tokenizer requires "motif_vocab_path" to be specified in the config.')
#   resolved_fimo = None
#   if use_fimo and fimo_path:
#     resolved_fimo = Path(str(fimo_path))
#   tokenizer = MotifTokenizer(
#     vocab_path=str(vocab_path),
#     fimo_path=resolved_fimo,
#     base_tokens=tuple(base_tokens) if base_tokens else ("A", "U", "C", "G"),
#     pad_token=pad_token,
#     max_length=max_len)
#   return tokenizer, tokenizer.pad_token_id


# def build_csv_motif_tokenizer(csv_path: typing.Union[str, Path],
#                               max_len: typing.Optional[int],
#                               pad_token: str = '<pad>',
#                               base_tokens: typing.Optional[typing.Sequence[str]] = None):
#   if csv_path is None:
#     raise ValueError('CSV motif tokenizer requires "motif_bank_csv_path" to be provided.')
#   tokenizer = CSVKmerTokenizer(
#     csv_path=str(csv_path),
#     pad_token=pad_token,
#     base_tokens=tuple(base_tokens) if base_tokens else ("A", "U", "C", "G"),
#     max_length=max_len)
#   return tokenizer, tokenizer.pad_token_id


def build_simple_tokenizer(vocab_path,
                           pad_token='N',
                           eos_token='EOS',
                           unk_token=None):
  with open(vocab_path, 'r') as fp:
    vocab_dict = json.load(fp)
  tokenizer = SimpleVocabTokenizer(vocab_dict, pad_token=pad_token, eos_token=eos_token, unk_token=unk_token)
  return tokenizer, tokenizer.pad_token_id, tokenizer.eos_token_id


class UTRDataset(torch.utils.data.Dataset):

  def __init__(self,
               csv_path,
               tokenizer,
               max_length,
               pad_id,
               seq_col="utr",
               label_col="rl"):
    self.df = pd.read_csv(csv_path)
    unnamed_cols = [col for col in self.df.columns if str(col).startswith('Unnamed')]
    if unnamed_cols:
      self.df = self.df.drop(columns=unnamed_cols)
    if seq_col not in self.df.columns:
      raise ValueError(f'Expected a "{seq_col}" column in the UTR CSV.')
    
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.pad_id = pad_id
    self.seq_col = seq_col

    if label_col == "auto":
      self.target_column = next((col for col in self.df.columns if col != self.seq_col), None)
    elif label_col is None:
      self.target_column = None
    else:
      if label_col not in self.df.columns:
        raise ValueError(f'Label column "{label_col}" not found in the UTR CSV.')
      self.target_column = label_col

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    seq = row[self.seq_col]
    
    enc = self.tokenizer.encode(seq)
    ids = enc.ids[:self.max_length]
    pad_length = self.max_length - len(ids)
    if pad_length > 0:
      ids = ids + [self.pad_id] * pad_length
      
    attention_mask = [1 if token_id != self.pad_id else 0 for token_id in ids]
    
    item = {
      'seqs': torch.tensor(ids, dtype=torch.long),
      'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
    }
    
    if self.target_column:
      item['labels'] = torch.tensor(row[self.target_column], dtype=torch.float32)
      
    return item
  def decode_batch(self, batch_ids):
    return [self._decode_single(ids) for ids in batch_ids]

  def _decode_single(self, ids):
    tokens = [token for token in ids if token != self.pad_id]
    return self.tokenizer.decode(tokens)


class UTRDataset_fa(torch.utils.data.Dataset):

  def __init__(self, fasta_path, tokenizer, max_length, pad_id):
    self.records = self._read_fasta(fasta_path)
    if not self.records:
      raise ValueError(f'No sequences found in FASTA file {fasta_path}.')
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.pad_id = pad_id
    self.labels = None

  def _read_fasta(self, fasta_path):
    records = []
    current_name = None
    current_seq = []
    with open(fasta_path, 'r') as handle:
      for raw in handle:
        line = raw.strip()
        if not line:
          continue
        if line.startswith('>'):
          if current_name is not None:
            records.append((current_name, ''.join(current_seq)))
          current_name = line[1:].strip()
          current_seq = []
        else:
          current_seq.append(line)
    if current_name is not None:
      records.append((current_name, ''.join(current_seq)))
    return records

  def __len__(self):
    return len(self.records)

  def __getitem__(self, idx):
    name, seq = self.records[idx]
    encoding = self.tokenizer.encode_sequence(name, seq, max_length=self.max_length)
    ids = encoding.ids
    ids = ids[:self.max_length]
    pad_length = self.max_length - len(ids)
    if pad_length > 0:
      ids = ids + [self.pad_id] * pad_length
    
    attention_mask = [1 if token_id != self.pad_id else 0 for token_id in ids]
    
    return {
      'seqs': torch.tensor(ids, dtype=torch.long),
      'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
    }

  def decode_batch(self, batch_ids):
    return [self._decode_single(ids) for ids in batch_ids]

  def _decode_single(self, ids):
    tokens = [token for token in ids if token != self.pad_id]
    return self.tokenizer.decode(tokens)


def get_datasets_utr(csv_path, tokenizer, max_length):
  return UTRDataset(csv_path, tokenizer, max_length)


def get_dataloaders_gosai(config, skip_valid=False, valid_seed=None):
  num_gpus = torch.cuda.device_count()
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')
  train_set = GosaiDataset()
  # randomly sample a subset of the train_set as valid_set
  valid_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), 40000, replace=False))
  test_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), 40000, replace=False))

  train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.loader.batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=not config.data.streaming,
    persistent_workers=True)
  if skip_valid:
    valid_loader = None
    test_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)
    test_loader = torch.utils.data.DataLoader(
      test_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator)

  return train_loader, valid_loader, test_loader


def get_dataloaders_utr(config,
                        tokenizer,
                        pad_id,
                        csv_path=None,
                        valid_csv_path=None,
                        test_csv_path=None,
                        # fasta_path=None,
                        seq_col="utr",
                        label_col="rl",
                        max_length=None,
                        skip_valid=False,
                        valid_seed=None):
  num_gpus = torch.cuda.device_count()
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.loader.batch_size}'
      f' not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size {config.loader.eval_batch_size} '
      f'not divisible by {num_gpus}.')
  csv_path = csv_path or getattr(config.data, "train_csv_path", None) or getattr(config.data, "utr_csv_path", None)
  valid_csv_path = valid_csv_path or getattr(config.data, "valid_csv_path", None)
  test_csv_path = test_csv_path or getattr(config.data, "test_csv_path", None)
  # fasta_path = fasta_path or getattr(config.data, "utr_fasta_path", None)
  max_length = max_length or config.model.length

  # if fasta_path is not None:
  #   dataset = UTRDataset_fa(fasta_path, tokenizer, max_length, pad_id)
  if csv_path is not None:
    dataset = UTRDataset(csv_path, tokenizer, max_length, pad_id, seq_col=seq_col, label_col=label_col)
  else:
    raise ValueError("Either csv_path or fasta_path must be provided for UTR data.")

  # if len(dataset) == 0:
    # source = fasta_path if fasta_path is not None else csv_path
    # raise ValueError(f'UTR dataset at {source} is empty.')

  if valid_csv_path:
    valid_set = UTRDataset(valid_csv_path, tokenizer, max_length, pad_id, seq_col=seq_col, label_col=label_col)
  else:
    subset_size = min(config.eval.subset_size, len(dataset))
    valid_indices = np.random.choice(len(dataset), subset_size, replace=False)
    valid_set = torch.utils.data.Subset(dataset, valid_indices)

  if test_csv_path:
    test_set = UTRDataset(test_csv_path, tokenizer, max_length, pad_id, seq_col=seq_col, label_col=label_col)
  else:
    test_set = valid_set

  train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.loader.batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=not config.data.streaming,
    persistent_workers=False)

  if skip_valid:
    valid_loader = None
    test_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator,
      persistent_workers=False)
    test_loader = torch.utils.data.DataLoader(
      test_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator,
      persistent_workers=False)

  return train_loader, valid_loader, test_loader


def count_token_frequencies(dataloader,
                            tokenizer,
                            output_path,
                            include_pad: bool = False):
  """
  Count token occurrences in a tokenized dataset and write token->count JSON.

  Args:
    dataloader: DataLoader yielding batches with keys 'seqs' and 'attention_mask'.
    tokenizer: Tokenizer instance with token2id (e.g., MotifAwareTokenizer).
    output_path: Where to write the JSON summary.
    include_pad: Whether to count pad tokens (default: False).

  Returns:
    Dict mapping token string to integer counts.
  """
  counter: Counter = Counter()
  pad_id = getattr(tokenizer, 'pad_token_id', None)

  for batch in dataloader:
    seqs = batch['seqs']
    attn = batch.get('attention_mask')

    if attn is not None:
      for seq, mask in zip(seqs, attn):
        # attention_mask marks real tokens (including EOS) with 1
        active_ids = seq[mask.bool()].tolist()
        if not include_pad and pad_id is not None:
          active_ids = [tid for tid in active_ids if tid != pad_id]
        counter.update(active_ids)
    else:
      for seq in seqs:
        ids = seq.tolist()
        if not include_pad and pad_id is not None:
          ids = [tid for tid in ids if tid != pad_id]
        counter.update(ids)

  freq_by_token = {
    token: int(counter.get(token_id, 0))
    for token, token_id in tokenizer.token2id.items()
  }

  with open(output_path, 'w') as fp:
    json.dump(freq_by_token, fp, indent=2)

  return freq_by_token


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py
class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0
