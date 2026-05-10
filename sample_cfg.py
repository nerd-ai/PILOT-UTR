#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import dataloader_gosai
import diffusion_gosai_cfg as diffusion


CONFIG_DIR = Path("/home/xli263/xli/utr_design/DRAKES/drakes_rna/configs_gosai")
DEFAULT_OUTPUT_CSV = Path(
  "/home/xli263/xli/utr_design/DRAKES/drakes_rna/"
  "baseline_generated_sequences/generated_cfg_high_te.csv"
)
DEFAULT_LENGTH_SOURCE_CSV = Path(
  "/home/xli263/xli/utr_design/DRAKES/drakes_rna/data_te/RP_PC3_te_train.csv"
)


def load_config(config_name: str):
  with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
    return compose(config_name=config_name)


def build_tokenizer(cfg):
  tokenizer_type = str(cfg.data.get("tokenizer_type", "simple_vocab")).lower()
  if tokenizer_type not in ("simple_vocab", "simple"):
    raise ValueError(f"sample_cfg.py expects simple_vocab, got {tokenizer_type}")
  tokenizer, pad_id, eos_id = dataloader_gosai.build_simple_tokenizer(
    cfg.data.tokenizer_vocab_path,
    pad_token=cfg.data.get("pad_token", "N"),
    eos_token=cfg.data.get("eos_token", None),
    unk_token=cfg.data.get("unk_token", None),
  )
  OmegaConf.set_struct(cfg.data, False)
  cfg.data.pad_token_id = pad_id
  cfg.data.eos_token_id = eos_id
  cfg.data.vocab_size = tokenizer.get_vocab_size()
  return tokenizer, pad_id


def decode_sample(tokenizer, token_ids, pad_id):
  ids = [int(token_id) for token_id in token_ids if int(token_id) != int(pad_id)]
  return tokenizer.decode(ids).replace(" ", "")


def clean_sequence(seq: str) -> str:
  seq = str(seq).upper().replace("U", "T")
  return "".join(base for base in seq if base in {"A", "C", "G", "T"})


def load_target_lengths(csv_path: Path, seq_column: str, max_length: int):
  df = pd.read_csv(csv_path)
  if seq_column not in df.columns:
    raise ValueError(f"Missing sequence column '{seq_column}' in {csv_path}")
  lengths = [
    min(len(clean_sequence(seq)), int(max_length))
    for seq in df[seq_column].astype(str).tolist()
  ]
  lengths = [length for length in lengths if length > 0]
  if not lengths:
    raise ValueError(f"No positive target lengths found in {csv_path}")
  return torch.tensor(lengths, dtype=torch.long)


def load_model(checkpoint_path: Path, cfg, device: torch.device):
  ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
  state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
  model = diffusion.Diffusion(config=cfg, eval=False)
  model.load_state_dict(state_dict, strict=False)
  model.to(device).eval()
  return model


def parse_args():
  parser = argparse.ArgumentParser(description="Generate sequences with CFG diffusion checkpoint.")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
  parser.add_argument("--config-name", type=str, default="config_cfg")
  parser.add_argument("--num-samples", type=int, default=1000)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--num-steps", type=int, default=None)
  parser.add_argument("--target-length", type=int, default=None,
                      help="Force one fixed non-padding token length. Overrides --sample-lengths.")
  parser.add_argument("--sample-lengths", action=argparse.BooleanOptionalAction, default=True,
                      help="Sample target lengths from --length-source-csv.")
  parser.add_argument("--length-source-csv", type=Path, default=DEFAULT_LENGTH_SOURCE_CSV)
  parser.add_argument("--length-seq-column", type=str, default="utr")
  parser.add_argument("--cls", type=int, default=1, choices=(0, 1),
                      help="0=low-score class, 1=high-score class.")
  parser.add_argument("--guidance-weight", type=float, default=None,
                      help="CFG weight w. Defaults to model.cls_free_weight from config.")
  parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
  return parser.parse_args()


def main():
  args = parse_args()
  cfg = load_config(args.config_name)
  tokenizer, pad_id = build_tokenizer(cfg)
  device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
  model = load_model(args.checkpoint, cfg, device)
  length_values = None
  if args.target_length is None and args.sample_lengths:
    length_values = load_target_lengths(
      args.length_source_csv,
      args.length_seq_column,
      max_length=cfg.model.length,
    )
    print(
      f"Loaded {len(length_values)} target lengths from {args.length_source_csv}; "
      f"min={int(length_values.min())}, max={int(length_values.max())}, "
      f"mean={float(length_values.float().mean()):.2f}"
    )

  rows = []
  generated = 0
  with torch.no_grad():
    while generated < args.num_samples:
      cur_batch = min(args.batch_size, args.num_samples - generated)
      if args.target_length is not None:
        target_lengths = [int(args.target_length)] * cur_batch
      elif length_values is not None:
        idx = torch.randint(0, len(length_values), (cur_batch,))
        target_lengths = length_values[idx].tolist()
      else:
        target_lengths = None
      samples = model._sample(
        num_steps=args.num_steps,
        eval_sp_size=cur_batch,
        cls=args.cls,
        w=args.guidance_weight,
        target_length=target_lengths,
      ).detach().cpu().tolist()
      for row_i, sample in enumerate(samples):
        rows.append({
          "seq": decode_sample(tokenizer, sample, pad_id),
          "target_length": (
            int(target_lengths[row_i]) if target_lengths is not None else None
          ),
          "cls": args.cls,
          "guidance_weight": (
            float(args.guidance_weight)
            if args.guidance_weight is not None
            else float(cfg.model.cls_free_weight)
          ),
        })
      generated += cur_batch

  args.output_csv.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(rows).to_csv(args.output_csv, index=False)
  print(f"Saved {len(rows)} sequences to {args.output_csv}")


if __name__ == "__main__":
  main()
