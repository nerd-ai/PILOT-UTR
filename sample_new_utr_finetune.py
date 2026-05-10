#!/usr/bin/env python3
import argparse
import collections
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from omegaconf import OmegaConf

import dataloader_gosai
import diffusion_gosai_update as diffusion

BASES = ("A", "C", "G", "T")


def load_config():
    base = OmegaConf.load("/home/xli263/xli/utr_design/DRAKES/drakes_rna/configs_gosai/config_gosai_pretrain.yaml")
    model_cfg = OmegaConf.load("/home/xli263/xli/utr_design/DRAKES/drakes_rna/configs_gosai/model/dnaconv.yaml")
    noise_cfg = OmegaConf.load("/home/xli263/xli/utr_design/DRAKES/drakes_rna/configs_gosai/noise/loglinear.yaml")
    strategy_cfg = OmegaConf.load("/home/xli263/xli/utr_design/DRAKES/drakes_rna/configs_gosai/strategy/ddp.yaml")
    lr_cfg = OmegaConf.load("/home/xli263/xli/utr_design/DRAKES/drakes_rna/configs_gosai/lr_scheduler/constant_warmup.yaml")
    return OmegaConf.merge(
        base,
        OmegaConf.create({"model": model_cfg, "noise": noise_cfg, "strategy": strategy_cfg, "lr_scheduler": lr_cfg}),
    )


def build_tokenizer(cfg):
    tk_cfg = cfg.tokenizer if "tokenizer" in cfg else cfg.data
    tk_type = tk_cfg.get("tokenizer_type", tk_cfg.get("type", "simple_vocab")).lower()

    if tk_type == "genetic":
        tokenizer, pad_id = dataloader_gosai.build_genetic_bpe_tokenizer(
            state_path=tk_cfg.genetic_bpe_state_path,
            motif_path=tk_cfg.motif_bank_path,
            config_path=tk_cfg.get("config_path"),
            max_len=cfg.model.length,
            pad_token=tk_cfg.get("pad_token", "N"),
            unk_token=tk_cfg.get("unk_token", None),
        )
    elif tk_type == "motif":
        vocab_path = tk_cfg.get("motif_vocab_path", None)
        if vocab_path is None and hasattr(cfg, "data"):
            vocab_path = cfg.data.get("motif_vocab_path", None)
        fimo_path = tk_cfg.get("fimo_tsv_path", None)
        if fimo_path is None and hasattr(cfg, "data"):
            fimo_path = cfg.data.get("fimo_tsv_path", None)
        use_fimo = tk_cfg.get("use_fimo", None)
        if use_fimo is None and hasattr(cfg, "data"):
            use_fimo = cfg.data.get("use_fimo", False)
        tokenizer, pad_id = dataloader_gosai.build_motif_tokenizer(
            vocab_path=vocab_path,
            max_len=cfg.model.length,
            pad_token=tk_cfg.get("pad_token", "N"),
            fimo_path=fimo_path,
            use_fimo=bool(use_fimo) if use_fimo is not None else False,
            base_tokens=tk_cfg.get("motif_base_tokens", None),
        )
    elif tk_type == "csv_motif":
        vocab_json = cfg.data.get("motif_vocab_path")
        if vocab_json is None:
            raise ValueError('tokenizer_type="csv_motif" expects data.motif_vocab_path (JSON).')
        vocab_json = Path(vocab_json)
        if not vocab_json.exists():
            raise FileNotFoundError(f"Motif vocabulary JSON not found at {vocab_json}")
        tokenizer = dataloader_gosai.MotifAwareTokenizer(
            vocab_json_path=vocab_json,
            pad_token=cfg.data.get("pad_token", "N"),
            eos_token=cfg.data.get("eos_token", "EOS"),
            base_tokens=cfg.data.get("motif_base_tokens", ("A", "C", "G", "T")),
            max_length=cfg.model.length,
            trim_to=cfg.data.get("motif_trim_len"),
        )
        pad_id = tokenizer.pad_token_id
    elif tk_type == "bpe":
        tokenizer, pad_id = dataloader_gosai.build_bpe_tokenizer(
            cfg.data.tokenizer_vocab_path,
            cfg.data.tokenizer_merges_path,
            max_len=cfg.model.length,
        )
    else:
        tokenizer, pad_id, eos_id = dataloader_gosai.build_simple_tokenizer(
            cfg.data.tokenizer_vocab_path,
            pad_token=cfg.data.get("pad_token", "N"),
            eos_token=cfg.data.get("eos_token", None),
            unk_token=cfg.data.get("unk_token", None),
        )

    if hasattr(cfg, "data") and OmegaConf.is_config(cfg.data):
        OmegaConf.set_struct(cfg.data, False)
    cfg.data.pad_token_id = pad_id
    cfg.data.vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else 0

    eos_id = getattr(tokenizer, "eos_token_id", None)
    return tokenizer, pad_id, eos_id


def _read_fasta(fasta_path: Path) -> List[Tuple[str, str]]:
    records = []
    current_name = None
    current_seq = []
    with fasta_path.open("r") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    records.append((current_name, "".join(current_seq)))
                current_name = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line)
    if current_name is not None:
        records.append((current_name, "".join(current_seq)))
    return records


def compute_token_hist_from_csv(csv_path: Path, tokenizer) -> collections.Counter:
    df = pd.read_csv(csv_path)
    if "utr" not in df.columns:
        raise ValueError(f"Expected column 'utr' in {csv_path}")
    seqs = df["utr"].astype(str).tolist()

    if hasattr(tokenizer, "encode_sequence"):
        names = df["name"].tolist() if "name" in df.columns else [f"seq_{i}" for i in range(len(seqs))]
        encs = tokenizer.encode_batch(list(zip(names, seqs)))
    else:
        encs = tokenizer.encode_batch(seqs)

    return collections.Counter(sum(enc.attention_mask) for enc in encs)


def compute_token_hist_from_fasta(fasta_path: Path, tokenizer) -> collections.Counter:
    records = _read_fasta(fasta_path)
    if not records:
        raise ValueError(f"No sequences found in FASTA file {fasta_path}")

    if hasattr(tokenizer, "encode_sequence"):
        encs = tokenizer.encode_batch([(name, seq) for name, seq in records])
    else:
        encs = tokenizer.encode_batch([seq for name, seq in records])

    return collections.Counter(sum(enc.attention_mask) for enc in encs)


def proportional_round(total: int, weights: Dict[int, float]) -> Dict[int, int]:
    if total <= 0 or not weights:
        return {k: 0 for k in weights}
    wsum = float(sum(weights.values()))
    if wsum == 0:
        return {k: 0 for k in weights}

    exact = {k: (v / wsum) * total for k, v in weights.items()}
    rounded = {k: int(round(v)) for k, v in exact.items()}
    diff = total - sum(rounded.values())
    fracs = {k: exact[k] - rounded[k] for k in weights}
    ordered = sorted(fracs, key=lambda k: fracs[k], reverse=(diff > 0))

    i = 0
    while diff != 0 and ordered:
        k = ordered[i % len(ordered)]
        if diff > 0:
            rounded[k] += 1
            diff -= 1
        else:
            if rounded[k] > 0:
                rounded[k] -= 1
                diff += 1
        i += 1
    return rounded


def _trim_token_ids(token_ids: List[int], pad_id: int, eos_id) -> List[int]:
    out = []
    for t in token_ids:
        ti = int(t)
        if eos_id is not None and ti == int(eos_id):
            break
        if ti == int(pad_id):
            break
        out.append(ti)
    return out


def _decode_from_base_probs(base_probs: torch.Tensor, expected_len: int = None) -> str:
    if base_probs.dim() != 2 or base_probs.shape[-1] != 4:
        raise ValueError(f"Expected [L,4], got {tuple(base_probs.shape)}")
    idx = base_probs.argmax(dim=-1).detach().cpu().tolist()
    if expected_len is not None:
        idx = idx[: max(0, int(expected_len))]
    return "".join(BASES[i] for i in idx)


def save_sequences_csv(output_path: Path, rows: List[Dict[str, object]]):
    if not rows:
        # print("No sequences generated; skipping write.")
        return
    fieldnames = ["seq", "token_length", "nt_length"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_token_ids_csv(output_path: Path, rows: List[Dict[str, object]]):
    if not rows:
        # print("No token-id sequences generated; skipping write.")
        return
    fieldnames = ["token_ids", "token_length"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser("Generate sequences using finetune-time sampling strategy.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.ckpt).")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/home/xli263/xli/utr_design/DRAKES/drakes_rna/data_and_model/train_dataset.csv"),
        help="Training CSV for token-length histogram.",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=None,
        help="Training FASTA for token-length histogram. Overrides --csv.",
    )
    parser.add_argument("--num-samples", type=int, default=32604)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/xli263/xli/utr_design/DRAKES/drakes_rna/generated_sequences_4_base_new/generated_optimized_4_base_sft_finetune_sampler.csv"),
    )
    parser.add_argument("--output-token-ids", type=Path, default=None)
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Denoising steps for _sample_finetune_gradient; defaults to config.sampling.steps.")
    parser.add_argument("--copy-flag-temp", type=float, default=None)
    parser.add_argument("--gradient-type", type=str, choices=["base_soft", "motif_soft"], default="base_soft")
    parser.add_argument("--truncate-steps", type=int, default=50,
                        help="Must match finetuning run: config.finetuning.truncate_steps.")
    parser.add_argument("--gumbel-temp", type=float, default=1.0,
                        help="Must match finetuning run: config.finetuning.gumbel_softmax_temp.")
    parser.add_argument("--min-quota", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config()
    tokenizer, pad_id, eos_id = build_tokenizer(cfg)
    eos_present = eos_id is not None

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = diffusion.Diffusion(config=cfg)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # if missing or unexpected:
    #     print(f"[warn] load_state_dict strict=False missing={len(missing)} unexpected={len(unexpected)}")
    model.config.finetuning.truncate_steps = int(args.truncate_steps)
    model.config.finetuning.gumbel_softmax_temp = float(args.gumbel_temp)
    # print(
    #     f"[finetune-sampler] truncate_steps={model.config.finetuning.truncate_steps} "
    #     f"gumbel_temp={model.config.finetuning.gumbel_softmax_temp}"
    # )
    model.to(device).eval()

    if args.fasta:
        # print(f"Computing token length distribution from FASTA: {args.fasta}")
        tok_hist = compute_token_hist_from_fasta(args.fasta, tokenizer)
    elif args.csv:
        # print(f"Computing token length distribution from CSV: {args.csv}")
        tok_hist = compute_token_hist_from_csv(args.csv, tokenizer)
    else:
        raise ValueError("Either --csv or --fasta must be provided.")

    total_train = sum(tok_hist.values())
    if total_train == 0:
        raise ValueError("Input file for token histogram is empty or could not be processed.")

    p_tok = {L: c / total_train for L, c in tok_hist.items()}
    N = args.num_samples if args.num_samples > 0 else total_train
    quotas = proportional_round(N, p_tok)

    if args.min_quota > 1:
        small = {L: q for L, q in quotas.items() if q < args.min_quota}
        if small:
            kept = {L: q for L, q in quotas.items() if q >= args.min_quota}
            removed = sum(small.values())
            kept_sum = sum(kept.values())
            if kept_sum == 0:
                topL = max(quotas, key=quotas.get)
                quotas = {topL: N}
            else:
                for L in kept:
                    kept[L] += int(round(removed * (kept[L] / kept_sum)))
                drift = N - sum(kept.values())
                order = sorted(kept, key=kept.get, reverse=(drift > 0))
                i = 0
                while drift != 0 and order:
                    L = order[i % len(order)]
                    if drift > 0:
                        kept[L] += 1
                        drift -= 1
                    else:
                        if kept[L] > 0:
                            kept[L] -= 1
                            drift += 1
                    i += 1
                quotas = kept

    # print("Global token-length histogram:", dict(sorted(tok_hist.items())))
    # print(f"Generation quotas (N={N}):", dict(sorted(quotas.items())))

    results: List[Dict[str, object]] = []
    token_id_rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for L, quota in sorted(quotas.items()):
            if quota <= 0:
                continue

            if eos_present:
                target_len_arg = int(L) - 1
                expected_nt_len = int(L) - 1
            else:
                target_len_arg = int(L)
                expected_nt_len = int(L)

            # print(f"Generating {quota} samples with token length {L} (target_len_arg={target_len_arg})")
            remaining = quota
            while remaining > 0:
                bs = min(args.batch_size, remaining)
                sample_out, _, _, _, _, _, p_vocab, _ = model._sample_finetune_gradient(
                    num_steps=args.num_steps,
                    eval_sp_size=bs,
                    copy_flag_temp=args.copy_flag_temp,
                    target_length=target_len_arg,
                    gradient_type=args.gradient_type,
                )

                use_base_decode = (sample_out.dim() == 3 and sample_out.shape[-1] == 4)

                if use_base_decode:
                    for row in sample_out:
                        seq = _decode_from_base_probs(row, expected_len=expected_nt_len)
                        results.append({"seq": seq, "token_length": int(L), "nt_length": len(seq)})
                        token_id_rows.append({"token_ids": "", "token_length": int(L)})
                else:
                    token_batch = p_vocab.argmax(dim=-1).detach().cpu().tolist()
                    for token_ids in token_batch:
                        trimmed = _trim_token_ids(token_ids, pad_id=pad_id, eos_id=eos_id)
                        seq = tokenizer.decode(trimmed).replace(" ", "")
                        token_id_rows.append({"token_ids": " ".join(str(t) for t in trimmed), "token_length": int(L)})
                        results.append({"seq": seq, "token_length": int(L), "nt_length": len(seq)})

                remaining -= bs

    # print(f"Generated {len(results)} sequences.")
    gen_hist = collections.Counter(r["token_length"] for r in results)
    # print("Generated token-length histogram:", dict(sorted(gen_hist.items())))

    if args.output.suffix.lower() != ".csv":
        # print(f"Warning: output extension {args.output.suffix} is not .csv, writing CSV anyway.")
        pass
    save_sequences_csv(args.output, results)
    # print(f"Wrote sequences to {args.output}")

    if args.output_token_ids is not None:
        save_token_ids_csv(args.output_token_ids, token_id_rows)
        # print(f"Wrote token-id sequences to {args.output_token_ids}")


if __name__ == "__main__":
    main()
