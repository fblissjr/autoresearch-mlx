"""
Comparative benchmark: our GPT vs an external implementation.
Loads external model dynamically from a given path (must have GPT, GPTConfig, and
optionally AdamW with the same interface as coderef).

Usage:
    uv run bench_compare.py /path/to/other/repo
    uv run bench_compare.py  # ours only (no comparison)
"""

import argparse
import importlib
import os
import platform
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import orjson
from mlx.utils import tree_flatten

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader
from train import (
    ADAM_BETAS,
    ASPECT_RATIO,
    EMBEDDING_LR,
    HEAD_DIM,
    MATRIX_LR,
    SCALAR_LR,
    UNEMBEDDING_LR,
    WEIGHT_DECAY,
    WINDOW_PATTERN,
    X0_BETAS,
)
from train import (
    GPT as OursGPT,
)
from train import (
    GPTConfig as OursGPTConfig,
)
from train import (
    loss_fn as ours_loss_fn,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOTAL_STEPS = 12
DISCARD = 2

CONFIGS = [
    {"label": "D=4 B=16", "depth": 4, "batch": 16},
    {"label": "D=8 B=32", "depth": 8, "batch": 32},
]


def make_config_dict(depth, vocab_size):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return {
        "sequence_len": MAX_SEQ_LEN,
        "vocab_size": vocab_size,
        "n_layer": depth,
        "n_head": num_heads,
        "n_kv_head": num_heads,
        "n_embd": model_dim,
        "window_pattern": WINDOW_PATTERN,
    }


def count_params(model):
    return sum(p.size for _, p in tree_flatten(model.parameters()))


def bench_fwd_bwd(model, loss_fn, batches, total_tokens):
    grad_fn = nn.value_and_grad(model, loss_fn)
    results = []
    for i, (x, y) in enumerate(batches):
        t0 = time.perf_counter()
        loss, grads = grad_fn(model, x, y)
        mx.eval(loss, grads)
        dt = time.perf_counter() - t0
        results.append(
            {"dt": dt, "loss": loss.item(), "tok_sec": int(total_tokens / dt)}
        )
        del grads
    return results


def bench_full_step(model, loss_fn, optimizer, eval_state_fn, batches, total_tokens):
    grad_fn = nn.value_and_grad(model, loss_fn)
    results = []
    for i, (x, y) in enumerate(batches):
        t0 = time.perf_counter()
        loss, grads = grad_fn(model, x, y)
        optimizer.update(model, grads)
        eval_state_fn(optimizer)
        mx.eval(loss, model.parameters())
        dt = time.perf_counter() - t0
        results.append(
            {"dt": dt, "loss": loss.item(), "tok_sec": int(total_tokens / dt)}
        )
        del grads
    return results


def summarize(results):
    steady = results[DISCARD:]
    if not steady:
        return {"avg_ms": 0, "avg_tok_sec": 0}
    avg_dt = sum(r["dt"] for r in steady) / len(steady)
    avg_tok = int(sum(r["tok_sec"] for r in steady) / len(steady))
    return {"avg_ms": avg_dt * 1000, "avg_tok_sec": avg_tok}


def print_results(label, fwd_summary, full_summary, params, peak_mb):
    print(f"\n--- {label} ---")
    print(f"  params:     {params / 1e6:.2f}M")
    print(
        f"  fwd+bwd:    {fwd_summary['avg_ms']:7.1f}ms  {fwd_summary['avg_tok_sec']:>7,} tok/sec"
    )
    print(
        f"  full step:  {full_summary['avg_ms']:7.1f}ms  {full_summary['avg_tok_sec']:>7,} tok/sec"
    )
    print(f"  peak_mem:   {peak_mb:.0f} MB")


def load_external_module(path):
    """Import train.py from an external repo path."""
    sys.path.insert(0, path)
    try:
        mod = importlib.import_module("train")
    finally:
        sys.path.pop(0)
    return mod


def bench_ours(config_dict, batches, total_tokens):
    cfg = OursGPTConfig(**config_dict)
    model = OursGPT(cfg)
    model.init_weights()
    mx.eval(model.parameters())
    params = count_params(model)

    fwd = bench_fwd_bwd(model, ours_loss_fn, batches, total_tokens)
    fwd_s = summarize(fwd)

    dmodel_scale = (cfg.n_embd / 768) ** -0.5

    def is_muon_param(path, weight):
        return "layers" in path and weight.ndim >= 2 and "ve_gate" not in path

    def is_embedding(path, weight):
        return "wte" in path or "value_embeds" in path

    def is_x0_lambdas(path, weight):
        return "x0_lambdas" in path

    def is_resid_lambdas(path, weight):
        return "resid_lambdas" in path

    muon = optim.Muon(learning_rate=MATRIX_LR, momentum=0.95, weight_decay=WEIGHT_DECAY)
    embed = optim.AdamW(
        learning_rate=EMBEDDING_LR * dmodel_scale,
        betas=list(ADAM_BETAS),
        eps=1e-10,
        weight_decay=0.0,
    )
    x0 = optim.AdamW(
        learning_rate=SCALAR_LR * dmodel_scale,
        betas=list(X0_BETAS),
        eps=1e-10,
        weight_decay=0.0,
    )
    resid = optim.AdamW(
        learning_rate=SCALAR_LR * 0.01 * dmodel_scale,
        betas=list(ADAM_BETAS),
        eps=1e-10,
        weight_decay=0.0,
    )
    fallback = optim.AdamW(
        learning_rate=UNEMBEDDING_LR * dmodel_scale,
        betas=list(ADAM_BETAS),
        eps=1e-10,
        weight_decay=0.0,
    )
    opt = optim.MultiOptimizer(
        [muon, embed, x0, resid, fallback],
        [is_muon_param, is_embedding, is_x0_lambdas, is_resid_lambdas],
    )
    full = bench_full_step(
        model, ours_loss_fn, opt, lambda o: mx.eval(o.state), batches, total_tokens
    )  # noqa: mx.eval is MLX graph materialization, not Python eval()
    full_s = summarize(full)

    peak = mx.get_peak_memory() / 1024 / 1024
    print_results(
        "Ours (5-group Muon+AdamW MultiOptimizer)", fwd_s, full_s, params, peak
    )

    del model, opt
    return {"fwd": fwd_s, "full": full_s, "params": params, "peak_mb": peak}


def bench_external(ext_mod, config_dict, batches, total_tokens):
    ExtGPTConfig = ext_mod.GPTConfig
    ExtGPT = ext_mod.GPT

    cfg = ExtGPTConfig(**config_dict)
    model = ExtGPT(cfg)
    model.init_weights()
    mx.eval(model.parameters())
    params = count_params(model)

    ext_loss_fn = lambda model, x, y: model(x, targets=y)

    fwd = bench_fwd_bwd(model, ext_loss_fn, batches, total_tokens)
    fwd_s = summarize(fwd)

    # Use external AdamW if available, otherwise built-in
    if hasattr(ext_mod, "AdamW"):
        ext_opt = ext_mod.AdamW(
            model,
            unembedding_lr=0.004,
            embedding_lr=0.6,
            matrix_lr=MATRIX_LR,
            weight_decay=0.2,
            adam_betas=ADAM_BETAS,
            scalar_lr=0.5,
        )

        def eval_ext_state(o):
            if o.state:
                mx.eval(*o.state)
    else:
        ext_opt = optim.AdamW(
            learning_rate=MATRIX_LR, betas=list(ADAM_BETAS), eps=1e-10, weight_decay=0.0
        )

        def eval_ext_state(o):
            mx.eval(o.state)

    full = bench_full_step(
        model, ext_loss_fn, ext_opt, eval_ext_state, batches, total_tokens
    )
    full_s = summarize(full)

    peak = mx.get_peak_memory() / 1024 / 1024
    print_results("External", fwd_s, full_s, params, peak)

    del model, ext_opt
    return {"fwd": fwd_s, "full": full_s, "params": params, "peak_mb": peak}


def main():
    parser = argparse.ArgumentParser(description="Comparative GPT benchmark")
    parser.add_argument(
        "external_path",
        nargs="?",
        default=None,
        help="Path to external repo with train.py (must have GPT, GPTConfig)",
    )
    args = parser.parse_args()

    mx.random.seed(42)
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    ext_mod = None
    if args.external_path:
        ext_mod = load_external_module(args.external_path)
        print(f"Loaded external module from: {args.external_path}")

    all_results = []

    for cfg in CONFIGS:
        depth, batch, label = cfg["depth"], cfg["batch"], cfg["label"]
        total_tokens = batch * MAX_SEQ_LEN

        print(f"\n{'=' * 70}")
        print(
            f"Config: {label} (depth={depth}, batch={batch}, tokens/step={total_tokens:,})"
        )
        print(f"{'=' * 70}")

        config_dict = make_config_dict(depth, vocab_size)

        loader = make_dataloader(tokenizer, batch, MAX_SEQ_LEN, "train")
        batches = [(x, y) for x, y, _ in (next(loader) for _ in range(TOTAL_STEPS))]

        ours = bench_ours(config_dict, batches, total_tokens)

        ext = None
        if ext_mod:
            ext = bench_external(ext_mod, config_dict, batches, total_tokens)

        all_results.append({"label": label, "ours": ours, "ext": ext})

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}\n")

    if ext_mod:
        header = f"{'Config':<14} {'Metric':<14} {'Ours':>10} {'External':>10} {'Speedup':>8}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            o, e = r["ours"], r["ext"]
            fwd_sp = (
                e["fwd"]["avg_ms"] / o["fwd"]["avg_ms"] if o["fwd"]["avg_ms"] > 0 else 0
            )
            full_sp = (
                e["full"]["avg_ms"] / o["full"]["avg_ms"]
                if o["full"]["avg_ms"] > 0
                else 0
            )
            print(
                f"{r['label']:<14} {'fwd+bwd ms':<14} {o['fwd']['avg_ms']:>10.1f} {e['fwd']['avg_ms']:>10.1f} {fwd_sp:>7.2f}x"
            )
            print(
                f"{'':<14} {'fwd+bwd tok/s':<14} {o['fwd']['avg_tok_sec']:>10,} {e['fwd']['avg_tok_sec']:>10,}"
            )
            print(
                f"{'':<14} {'full step ms':<14} {o['full']['avg_ms']:>10.1f} {e['full']['avg_ms']:>10.1f} {full_sp:>7.2f}x"
            )
            print(
                f"{'':<14} {'full tok/s':<14} {o['full']['avg_tok_sec']:>10,} {e['full']['avg_tok_sec']:>10,}"
            )
            print(
                f"{'':<14} {'params (M)':<14} {o['params'] / 1e6:>10.2f} {e['params'] / 1e6:>10.2f}"
            )
            print(
                f"{'':<14} {'peak MB':<14} {o['peak_mb']:>10.0f} {e['peak_mb']:>10.0f}"
            )
            print()
        print("Speedup > 1.0 means ours is faster.")
    else:
        header = f"{'Config':<14} {'Metric':<14} {'Value':>10}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            o = r["ours"]
            print(f"{r['label']:<14} {'fwd+bwd ms':<14} {o['fwd']['avg_ms']:>10.1f}")
            print(f"{'':<14} {'fwd+bwd tok/s':<14} {o['fwd']['avg_tok_sec']:>10,}")
            print(f"{'':<14} {'full step ms':<14} {o['full']['avg_ms']:>10.1f}")
            print(f"{'':<14} {'full tok/s':<14} {o['full']['avg_tok_sec']:>10,}")
            print(f"{'':<14} {'params (M)':<14} {o['params'] / 1e6:>10.2f}")
            print(f"{'':<14} {'peak MB':<14} {o['peak_mb']:>10.0f}")
            print()

    # Save results to data/
    bench_data = {
        "format_version": "0.1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": {
            "chip": platform.processor() or "Apple Silicon",
            "memory_gb": None,
            "os": platform.system(),
        },
        "configs": all_results,
    }
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("data", f"bench_{timestamp}.json")
    with open(out_path, "wb") as f:
        f.write(orjson.dumps(bench_data, option=orjson.OPT_INDENT_2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
