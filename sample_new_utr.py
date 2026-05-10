#!/usr/bin/env python3
import argparse
import collections
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import dataloader_gosai
import diffusion_gosai_update as diffusion


# -------------------- Config / Tokenizer --------------------
def load_config():
    base = OmegaConf.load("/home/xli263/xli/utr_design/DRAKES/drakes_rna/configs_gosai/config_gosai_pretrain.yaml")
    model_cfg = OmegaConf.load("/home/xli263/xli/utr_design/DRAKES/drakes_rna/configs_gosai/model/dnaconv.yaml")
    noise_cfg = OmegaConf.load("/home/xli263/xli/utr_design/DRAKES/drakes_rna/configs_gosai/noise/loglinear.yaml")
    strategy_cfg = OmegaConf.load("/home/xli263/xli/utr_design/DRAKES/drakes_rna/configs_gosai/strategy/ddp.yaml")
    lr_cfg = OmegaConf.load("/home/xli263/xli/utr_design/DRAKES/drakes_rna/configs_gosai/lr_scheduler/constant_warmup.yaml")
    cfg = OmegaConf.merge(
        base,
        OmegaConf.create({"model": model_cfg, "noise": noise_cfg, "strategy": strategy_cfg, "lr_scheduler": lr_cfg}),
    )
    return cfg


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
        print(use_fimo)
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
        # csv_path = tk_cfg.get("motif_bank_csv_path", None)
        vocab_json = cfg.data.get('motif_vocab_path')
        if vocab_json is None:
            raise ValueError('tokenizer_type="csv_motif" now expects data.motif_vocab_path (JSON) in the config.')
        vocab_json = Path(vocab_json)
        if not vocab_json.exists():
            raise FileNotFoundError(f"Motif vocabulary JSON not found at {vocab_json}")
        tokenizer = dataloader_gosai.MotifAwareTokenizer(
        vocab_json_path=vocab_json,
        pad_token=cfg.data.get('pad_token', 'N'),
        eos_token=cfg.data.get('eos_token', 'EOS'),
        base_tokens=cfg.data.get('motif_base_tokens', ("A", "C", "G", "T")),
        max_length=cfg.model.length,
        trim_to=cfg.data.get('motif_trim_len'))
        pad_id = tokenizer.pad_token_id
        eos_id = getattr(tokenizer, "eos_token_id", None)
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

    # Store pad_id in config for downstream components (ensure struct unlocked or field exists)
    if hasattr(cfg, "data") and OmegaConf.is_config(cfg.data):
        OmegaConf.set_struct(cfg.data, False)
    cfg.data.pad_token_id = pad_id
    cfg.data.vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else 0

    eos_id = getattr(tokenizer, "eos_token_id", None)
    return tokenizer, pad_id, eos_id



# -------------------- Helpers --------------------
def proportional_round(total: int, weights: Dict[int, float]) -> Dict[int, int]:
    """Round weighted counts to integers that sum to total."""
    if total <= 0 or not weights:
        return {k: 0 for k in weights}
    wsum = float(sum(weights.values()))
    if wsum == 0:
        return {k: 0 for k in weights}

    exact = {k: (v / wsum) * total for k, v in weights.items()}
    rounded = {k: int(round(v)) for k, v in exact.items()}
    diff = total - sum(rounded.values())

    # Distribute rounding drift to the largest fractional parts (or smallest) as needed
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


def decode_tokens(tokenizer, token_ids: List[int], pad_id: int) -> str:
    """Strip PADs and decode to a string."""
    trimmed = [t for t in token_ids if t != pad_id]
    # Some tokenizers insert spaces—normalize by removing them
    return tokenizer.decode(trimmed).replace(" ", "")


# -------------------- Core: Simple token-quota generation --------------------
def _read_fasta(fasta_path: Path) -> List[Tuple[str, str]]:
    """Reads a FASTA file into a list of (name, sequence) tuples."""
    records = []
    current_name = None
    current_seq = []
    with fasta_path.open('r') as handle:
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


def compute_token_hist_from_csv(csv_path: Path, tokenizer) -> collections.Counter:
    """Return a Counter of token lengths from a CSV file."""
    df = pd.read_csv(csv_path)
    if "utr" not in df.columns:
        raise ValueError(f"Expected column 'seq' in {csv_path}")
    seqs = df["utr"].astype(str).tolist()

    # MotifTokenizer can use sequence names, but CSV may not provide them.
    # We pass placeholder names if needed.
    if hasattr(tokenizer, "encode_sequence"):
        names = df["name"].tolist() if "name" in df.columns else [f"seq_{i}" for i in range(len(seqs))]
        encs = tokenizer.encode_batch(list(zip(names, seqs)))
    else:
        encs = tokenizer.encode_batch(seqs)

    hist = collections.Counter(sum(enc.attention_mask) for enc in encs)
    return hist


def compute_token_hist_from_fasta(fasta_path: Path, tokenizer) -> collections.Counter:
    """Return a Counter of token lengths from a FASTA file."""
    records = _read_fasta(fasta_path)
    if not records:
        raise ValueError(f"No sequences found in FASTA file {fasta_path}")

    # The MotifTokenizer uses sequence names, so we pass them.
    if hasattr(tokenizer, "encode_sequence"):
        encs = tokenizer.encode_batch([(name, seq) for name, seq in records])
    else:  # Fallback for simpler tokenizers
        encs = tokenizer.encode_batch([seq for name, seq in records])

    hist = collections.Counter(sum(enc.attention_mask) for enc in encs)
    return hist


def save_sequences_csv(output_path: Path, rows: List[Dict[str, object]]):
    """Saves generated sequences to a CSV file."""
    if not rows:
        print("No sequences generated; skipping write.")
        return
    fieldnames = ["seq", "token_length", "nt_length"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_sequences_fasta(output_path: Path, rows: List[Dict[str, object]]):
    """Saves generated sequences to a FASTA file."""
    if not rows:
        print("No sequences generated; skipping write.")
        return
    with output_path.open("w") as f:
        for i, row in enumerate(rows):
            seq = row["seq"]
            name = f"generated_seq_{i+1}|token_len={row['token_length']}|nt_len={row['nt_length']}"
            f.write(f">{name}\n")
            f.write(f"{seq}\n")


def save_token_ids_csv(output_path: Path, rows: List[Dict[str, object]]):
    """Saves generated token-id sequences to a CSV file."""
    if not rows:
        print("No token-id sequences generated; skipping write.")
        return
    fieldnames = ["token_ids", "token_length"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_token_ids(x_t: torch.Tensor) -> List[List[int]]:
    """
    Convert sampler tensor to token-id lists.
    Accepts either [B, T] ids or [B, T, V] one-hot/probabilities.
    """
    if x_t.dim() == 2:
        return x_t.detach().cpu().tolist()
    if x_t.dim() == 3:
        return x_t.argmax(dim=-1).detach().cpu().tolist()
    raise ValueError(f"Unsupported x_t shape for decoding: {tuple(x_t.shape)}")


def _decode_ids(tokenizer, token_ids: List[int], pad_id: int) -> str:
    trimmed = [int(t) for t in token_ids if int(t) != int(pad_id)]
    return tokenizer.decode(trimmed)


def _format_xt_tokens(
    tokenizer,
    token_ids: List[int],
    mask_id: int,
    pad_id: int,
    eos_id,
) -> str:
    """Format xt token ids with explicit [MASK] markers for trajectory logs."""
    pieces: List[str] = []
    id2token = getattr(tokenizer, "id2token", None)
    for raw in token_ids:
        tid = int(raw)
        if eos_id is not None and tid == int(eos_id):
            break
        if tid == int(pad_id):
            break
        if tid == int(mask_id):
            pieces.append("[MASK]")
            continue
        token = None
        if isinstance(id2token, dict):
            token = id2token.get(tid)
        elif isinstance(id2token, list) and 0 <= tid < len(id2token):
            token = id2token[tid]
        if token is None:
            token = str(tid)
        pieces.append(str(token))
    return " ".join(pieces)


def _resolve_base_token_ids(tokenizer) -> Dict[str, int]:
    """Resolve token ids for A/C/G/T, if present in tokenizer vocabulary."""
    out: Dict[str, int] = {}
    for base in ("A", "C", "G", "T"):
        tid = tokenizer.token_to_id(base) if hasattr(tokenizer, "token_to_id") else None
        if tid is not None:
            out[base] = int(tid)
    return out


def _avg_base_probs_on_masked_positions(
    log_p_vocab_b: torch.Tensor,
    xt_ids_b: List[int],
    mask_id: int,
    base_token_ids: Dict[str, int],
) -> Dict[str, float]:
    """
    Average model token probabilities for A/C/G/T over masked positions only.
    log_p_vocab_b: [T, V] log-probabilities (or logits close to log-probs) for one sample.
    """
    if not base_token_ids:
        return {b: 0.0 for b in "ACGT"}

    probs_b = log_p_vocab_b.exp()
    xt = torch.tensor([int(t) for t in xt_ids_b], device=probs_b.device)
    masked = (xt == int(mask_id))
    if masked.sum().item() == 0:
        return {b: 0.0 for b in "ACGT"}

    masked_probs = probs_b[masked]  # [M, V]
    out = {}
    for base in ("A", "C", "G", "T"):
        tid = base_token_ids.get(base)
        out[base] = float(masked_probs[:, tid].mean().item()) if tid is not None else 0.0
    return out


@torch.no_grad()
def write_trajectory_log(
    model,
    tokenizer,
    pad_id: int,
    eos_id,
    trajectory_log_path: Path,
    num_samples: int,
    num_steps: int,
    target_len_arg,
    token_length_label,
):
    """
    Log denoising trajectory rows for several samples.
    One line per (sample, t).
    """
    sample_out, last_x_list, condt_list, *_ = model._sample_finetune_gradient(
        num_steps=num_steps,
        eval_sp_size=num_samples,
        target_length=target_len_arg,
        gradient_type="base_soft",
    )
    base_token_ids = _resolve_base_token_ids(tokenizer)

    with trajectory_log_path.open("a") as f:
        f.write(f"# trajectory token_length={token_length_label} num_samples={num_samples} num_steps={num_steps}\n")
        for t in range(len(last_x_list)):
            x_t_ids = _to_token_ids(last_x_list[t])
            log_p_x0_t = model.forward(last_x_list[t], condt_list[t])[:, :, :-1]
            x0_ids = log_p_x0_t.argmax(dim=-1).detach().cpu().tolist()
            for b in range(min(num_samples, len(x_t_ids))):
                xt_seq = _format_xt_tokens(
                    tokenizer=tokenizer,
                    token_ids=x_t_ids[b],
                    mask_id=model.mask_index,
                    pad_id=pad_id,
                    eos_id=eos_id,
                ).replace("\n", " ").replace("\t", " ")
                x0_seq = _decode_ids(tokenizer, x0_ids[b], pad_id).replace("\n", " ").replace("\t", " ")
                base_probs = _avg_base_probs_on_masked_positions(
                    log_p_vocab_b=log_p_x0_t[b],
                    xt_ids_b=x_t_ids[b],
                    mask_id=model.mask_index,
                    base_token_ids=base_token_ids,
                )
                xt_len = len([tok for tok in xt_seq.split(" ") if tok]) if xt_seq else 0
                f.write(
                    f"step=0\tt={t}\tsample={b}\tlen={xt_len}\ttoken_length={token_length_label}\t"
                    f"new[A:{base_probs['A']:.3f} C:{base_probs['C']:.3f} "
                    f"G:{base_probs['G']:.3f} T:{base_probs['T']:.3f}]\t"
                    f"xt_seq={xt_seq}\tx0_seq_new={x0_seq}\n"
                )


def main(): 
    parser = argparse.ArgumentParser("Generate sequences to match the GLOBAL token-length distribution.")
    parser.add_argument("--checkpoint", required=True, help="Path to Lightning checkpoint (.ckpt).")
    parser.add_argument(
        "--csv",
        type=Path,
        default="/home/xli263/xli/utr_design/DRAKES/drakes_rna/data_and_model/mrl_human_25_100.csv",
        help="Training CSV used to compute global token-length histogram.",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=None,
        help="Training FASTA used to compute global token-length histogram. Overrides --csv if both are provided.",
    )
    parser.add_argument("--mode", choices=["autonomous", "fixed", "distribution"], default="autonomous", 
                        help="autonomous: Model decides length (Best for 50nt). fixed: Force --fixed-len tokens. distribution: Match training stats.")
    parser.add_argument("--num-samples", type=int, default=5000, help="Total number of sequences to generate.")
    parser.add_argument("--batch-size", type=int, default=512, help="Sampling batch size.")
    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument("--output", type=Path, default=Path("/home/xli263/xli/utr_design/DRAKES/drakes_rna/generated_sequences_4_base_mrl_25_100/generated_sequences_4_base_mrl_25_100_human.csv"), help="Output file (CSV or FASTA).")
    parser.add_argument(
        "--output-token-ids",
        type=Path,
        default=None,
        help="Optional CSV path to save token-id sequences before base decoding.",
    )
    parser.add_argument(
        "--log-trajectory",
        action="store_true",
        help="Write a denoising trajectory log (one row per sample-step).",
    )
    parser.add_argument(
        "--trajectory-log-path",
        type=Path,
        default=None,
        help="Output path for trajectory log. Defaults to <output_stem>_trajectory.log",
    )
    parser.add_argument(
        "--trajectory-num-samples",
        type=int,
        default=3,
        help="How many samples to log in the trajectory file.",
    )
    parser.add_argument(
        "--trajectory-num-steps",
        type=int,
        default=128,
        help="How many denoising steps to log in trajectory.",
    )
    parser.add_argument(
        "--min-quota", type=int, default=1,
        help="Optional: drop token lengths whose quota < min-quota after rounding (mass is redistributed).",
    )
    args = parser.parse_args()

    # Load cfg/tokenizer/model
    cfg = load_config()
    tokenizer, pad_id, eos_id = build_tokenizer(cfg)
    eos_present = eos_id is not None
    print(eos_present)
    device = torch.device(args.device)
    # model = diffusion.Diffusion.load_from_checkpoint(args.checkpoint, config=cfg, eval=False, map_location=device)
    # model.eval().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = diffusion.Diffusion(config=cfg)  # build the module
    # model.load_state_dict(ckpt, strict=True)
    # 1. Extract the actual weights
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt  # Fallback if it's a raw PyTorch save
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()


    # 1) Global token-length histogram & probabilities
    if args.fasta:
        print(f"Computing token length distribution from FASTA: {args.fasta}")
        tok_hist = compute_token_hist_from_fasta(args.fasta, tokenizer)
    elif args.csv:
        print(f"Computing token length distribution from CSV: {args.csv}")
        tok_hist = compute_token_hist_from_csv(args.csv, tokenizer)
    else:
        raise ValueError("Either --csv or --fasta must be provided to compute token length distribution.")

    total_train = sum(tok_hist.values())
    if total_train == 0:
        raise ValueError("Input file for token histogram is empty or could not be processed.")

    p_tok = {L: c / total_train for L, c in tok_hist.items()}

    print("Global token-length histogram from training data:")
    print("  counts:", dict(sorted(tok_hist.items())))
    print("  probs :", {L: round(p, 6) for L, p in sorted(p_tok.items())})

    # 2) Quotas for generation
    N = args.num_samples if args.num_samples > 0 else total_train
    quotas = proportional_round(N, p_tok)
    # quotas = {}
    # if args.mode == "autonomous":
    #     print(f"Mode: AUTONOMOUS. Generating {args.num_samples} samples. Model decides placement of EOS.")
    #     # "AUTO" key tells the loop to pass target_length=None
    #     quotas = {"AUTO": args.num_samples}
    # Optional: prune ultra-small quotas and redistribute
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
                    add = int(round(removed * (kept[L] / kept_sum)))
                    kept[L] += add
                drift = N - sum(kept.values())
                order = sorted(kept, key=kept.get, reverse=(drift > 0))
                i = 0
                while drift != 0 and order:
                    L = order[i % len(order)]
                    if drift > 0:
                        kept[L] += 1; drift -= 1
                    else:
                        if kept[L] > 0:
                            kept[L] -= 1; drift += 1
                    i += 1
                quotas = kept

    print(f"\nGeneration quotas (N={N}):")
    print(" ", dict(sorted(quotas.items())))

    trajectory_logged = False
    trajectory_log_path = args.trajectory_log_path
    if args.log_trajectory:
        if trajectory_log_path is None:
            trajectory_log_path = args.output.with_name(args.output.stem + "_trajectory.log")
        trajectory_log_path.parent.mkdir(parents=True, exist_ok=True)
        with trajectory_log_path.open("w") as f:
            f.write(f"# checkpoint={args.checkpoint}\n")
            f.write(f"# mode={args.mode} num_samples={args.num_samples} batch_size={args.batch_size}\n")
            f.write(f"# trajectory_num_samples={args.trajectory_num_samples} trajectory_num_steps={args.trajectory_num_steps}\n")

    # 3) Generate directly per token length (no bins, no rejection)
    results: List[Dict[str, object]] = []
    token_id_rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for L, quota in sorted(quotas.items()):
            if quota <= 0:
                continue
            if L == "AUTO":
                target_len_arg = None
                print_len = "Auto"
            else:
                # If EOS exists, _sample expects the EOS index (0-based).
                # If EOS does not exist, _sample should get the target length directly.
                if eos_present:
                    target_len_arg = L - 1
                    print_len = f"{L} tokens (EOS at idx {target_len_arg})"
                else:
                    target_len_arg = L
                    print_len = f"{L} tokens (no EOS)"
            print(f"Generating {quota} samples with token length {L}...")

            if args.log_trajectory and not trajectory_logged:
                traj_bs = max(1, int(args.trajectory_num_samples))
                write_trajectory_log(
                    model=model,
                    tokenizer=tokenizer,
                    pad_id=pad_id,
                    eos_id=eos_id,
                    trajectory_log_path=trajectory_log_path,
                    num_samples=traj_bs,
                    num_steps=int(args.trajectory_num_steps),
                    target_len_arg=target_len_arg,
                    token_length_label=L,
                )
                print(f"Wrote trajectory log to {trajectory_log_path}")
                trajectory_logged = True

            remaining = quota
            while remaining > 0:
                bs = min(args.batch_size, remaining)
                samples = model._sample(eval_sp_size=bs, target_length=target_len_arg)
                token_batch = samples.detach().cpu().tolist()
                for token_ids in token_batch:
                    token_id_rows.append(
                        {
                            "token_ids": " ".join(str(t) for t in token_ids),
                            "token_length": L,
                        }
                    )
                    # seq = decode_tokens(tokenizer, token_ids, pad_id)
                    seq = tokenizer.decode(token_ids)
                    results.append({"seq": seq, "token_length": L, "nt_length": len(seq)})
                remaining -= bs

    print(f"\nGenerated {len(results)} sequences.")
    gen_hist = collections.Counter(r["token_length"] for r in results)
    print("Generated token-length histogram:", dict(sorted(gen_hist.items())))

    # 4) Save
    output_suffix = args.output.suffix.lower()
    if output_suffix in [".fasta", ".fa", ".fna"]:
        save_sequences_fasta(args.output, results)
    else:
        if output_suffix != ".csv":
            print(f"Warning: Output file has an unknown extension '{output_suffix}'. Saving as CSV.")
        save_sequences_csv(args.output, results)
    print(f"Wrote sequences to {args.output}")
    if args.output_token_ids is not None:
        if args.output_token_ids.suffix.lower() != ".csv":
            print(
                f"Warning: Token-id output file has non-CSV extension '{args.output_token_ids.suffix}'. "
                "Saving as CSV."
            )
        save_token_ids_csv(args.output_token_ids, token_id_rows)
        print(f"Wrote token-id sequences to {args.output_token_ids}")


if __name__ == "__main__":
    main()
