last updated: 2026-03-15

# AGENTS.md -- Technical Knowledge Base

Distilled findings from session logs (`internal/log/`). For actionable rules and project config, see [CLAUDE.md](CLAUDE.md).

## Model Architecture

Custom GPT with value embeddings, RoPE, sliding window attention, tanh logit capping, relu^2 activation.

- `build_model_config(depth, vocab_size, seq_len)`: model_dim = depth * ASPECT_RATIO, rounded up to HEAD_DIM boundary
- Dict-based layers: `self.layers` uses `{"l0": ..., "l1": ...}` keys (not lists or digit-string keys). MLX's `tree_unflatten` converts string-digit keys back to lists, which breaks MultiOptimizer.
- Value embeddings use `{"v0": ..., "v1": ...}` keys for the same reason.
- Don't store plain Python lists as nn.Module attributes -- they show up in parameter/gradient trees and break optimizers.
- `__main__` guard allows importing model classes (GPT, GPTConfig, loss_fn, hyperparams) without triggering training.

## 5-Group MultiOptimizer Routing

| Group | Optimizer | Params | LR | Betas | Filter |
|-------|-----------|--------|-----|-------|--------|
| 1 | Muon | layers 2D+ weights (excl. ve_gate) | MATRIX_LR (0.04) | momentum 0.85->0.95 (300 steps) | `is_muon_param` |
| 2 | AdamW | wte + value_embeds | EMBEDDING_LR * dmodel_scale | (0.8, 0.95) | `is_embedding` |
| 3 | AdamW | x0_lambdas | SCALAR_LR * dmodel_scale | (0.96, 0.95) | `is_x0_lambdas` |
| 4 | AdamW | resid_lambdas | SCALAR_LR * 0.01 * dmodel_scale | (0.8, 0.95) | `is_resid_lambdas` |
| 5 (fallback) | AdamW | lm_head + ve_gate | UNEMBEDDING_LR * dmodel_scale | (0.8, 0.95) | (default) |

- `dmodel_scale = (n_embd / 768)^-0.5`. At depth=4: 1.732. At depth=8: 1.225.
- 4 filter functions route params to groups 1-4; unmatched fall to group 5. Order matters -- first match wins.
- Phase 2 swaps LR schedules via `opt._schedulers['learning_rate']` on existing instances. Preserves Adam momentum buffers and step counters.

## Agent Feedback Loop

The agent gets 5 metrics via grep: val_bpb (primary), peak_memory_mb, avg_tok_sec, num_steps, eval_seconds. It logs keep/discard decisions to results.tsv (6 columns: commit, val_bpb, memory_gb, avg_tok_sec, status, description). Throughput (avg_tok_sec) is a first-class signal -- the agent uses a quality+throughput decision framework (see program.md) rather than binary val_bpb comparison. Full structured data is archived in data/run_*.json for human analysis via `uv run analysis.py`.

## Known Deviations from PyTorch Reference

- dmodel_scale applied to scalar LRs (reference doesn't)
- Momentum and weight decay schedules were synced with upstream in v0.7.0 (momentum ramps 0.85->0.95 over 300 steps, WD decays linearly to 0)

## Compiled Training

- Works only when `grad_accum_steps == 1`. With `grad_accum > 1`, falls back to uncompiled (~15% slower).
- At DEPTH=4, DEVICE_BATCH_SIZE=16, TOTAL_BATCH_SIZE=65536: grad_accum=2 (uncompiled path).
- To get compiled: either DEVICE_BATCH_SIZE=32 or TOTAL_BATCH_SIZE=32768.

## MLX Implementation Gotchas

- `mx.fast.rms_norm` requires a weight parameter even for "weightless" norm. Cache the ones vector.
- `mx.fast.rope` requires `base` as float (10000.0 not 10000).
- `mx.metal.get_peak_memory()` is deprecated -- use `mx.get_peak_memory()`.
- Training output uses `\r` carriage returns -- pipe through `tr '\r' '\n'` when reading logs.
- First training step is slow (~2.5s) due to graph compilation; subsequent steps much faster.

## Memory and Performance

- Training memory is flat (no leaks, no growth). Fragmentation is in Metal's allocator pool.
- Optimizer cleanup before eval saves ~38s but fragmentation dominates.
- bench.py eval projections assume clean memory (~10GB). After sustained training, actual eval is ~2x projected due to Metal allocator fragmentation.
- Eval at batch=64 causes 14GB memory surge above training peak. Reduced to batch=32 in v0.5.4.

## Data Pipeline

- prepare.py verified line-by-line against PyTorch/CUDA reference. All differences are mechanical API translations. evaluate_bpb is numerically equivalent. See [conformance analysis](internal/analysis/2026-03-08_prepare-py-conformance.md).
- Numpy in prepare.py is intentional: CPU-side data packing in `make_dataloader` uses numpy buffer, converts to `mx.array` once per batch yield. No numpy in training hot path.
- climbmix: 553M rows, single `text` column, no quality scores. No filtering in pipeline.
- tinystories: 2.7M stories, ASCII-only, ~2.19B chars. Downloaded and sharded locally by data_sources.py.

## Training Results

| Version | Depth | val_bpb | Steps | Avg tok/sec | Train peak |
|---------|-------|---------|-------|-------------|------------|
| v0.5.2 | 8 | 1.886 | 193 | 42K | 49GB |
| v0.5.0 | 8 | 1.900 | 154 | 33K | 63GB |
| v0.3.0 | 8 | ~2.28 | ~189 | ~46K | ~53GB |
| v0.6.1 | 4 | 1.859 | 641 | 140K | 10.6GB |

Reference: [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) achieves val_bpb=1.295 with DEPTH=4, flat AdamW.

## Analysis & Investigations

- [internal/data-investigations.md](internal/data-investigations.md) -- data quality levers backlog
- [internal/analysis/2026-03-08_prepare-py-conformance.md](internal/analysis/2026-03-08_prepare-py-conformance.md) -- prepare.py vs upstream verification
- [internal/analysis/2026-03-08_eval-bottleneck.md](internal/analysis/2026-03-08_eval-bottleneck.md) -- eval 2x slower than projected; Metal allocator fragmentation
- [internal/analysis/2026-03-08_throughput-regression.md](internal/analysis/2026-03-08_throughput-regression.md) -- resolved: data cycling with 3 shards

## Future Work

- [internal/ane-integration.md](internal/ane-integration.md) -- ANE (Apple Neural Engine) integration roadmap for M2 Ultra
