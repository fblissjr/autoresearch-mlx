last updated: 2026-03-08

# AGENTS.md

This file provides context for AI agents working in this repository.

## Overview

Autonomous ML research framework that trains GPT models using MLX on Apple Silicon. An AI agent iterates on `train.py` to minimize validation bits-per-byte (val_bpb) within a fixed 5-minute training budget.

## Quick Reference

| Task | Command |
|------|---------|
| Prepare data | `uv run prepare.py` |
| Train (5 min) | `uv run train.py` |
| Benchmark | `uv run bench.py` |
| Run tests | `uv run python -m pytest tests/` or `uv run python tests/test_optimizations.py` |
| Analyze results | `uv run analysis.py` |
| Debug mode | Add `--debug` flag or `AUTORESEARCH_DEBUG=1` to any script |

## Project Structure

```
train.py         -- Model + training loop (the ONLY file the agent edits during experiments)
prepare.py       -- READ ONLY. Data prep, tokenizer, dataloader, evaluate_bpb
bench.py         -- Performance profiling (compiled vs uncompiled, per-phase timing)
analysis.py      -- Experiment results analysis (reads results.tsv, outputs charts)
log_utils.py     -- Project-wide logging framework (--debug flag support)
program.md       -- Detailed experiment loop instructions
tests/           -- Test suite
  test_optimizations.py  -- Tests for mx.compile, eval batch, step-based LR schedule
data/            -- Output files (run results, bench results, analysis). Tracked via .gitkeep, contents gitignored.
  run_*.json     -- Training run results (val_bpb, tok/sec, memory, config)
  bench_*.json   -- Benchmark results (fwd+bwd, full step timing)
internal/        -- Research notes and session logs. Committed to git (open research repo).
  log/           -- Session logs (log_YYYY-MM-DD.md). Document what was done, decisions, and open questions.
  data-investigations.md -- Backlog of data quality improvement ideas
```

## Key Files in Detail

### train.py

The only file modified during experiment runs. Contains:

- **GPTConfig / GPT model** (lines 23-249): Custom GPT with value embeddings, RoPE, sliding window attention, tanh logit capping, relu^2 activation
- **build_model_config(depth, vocab_size)** (line 286): Constructs config from depth/aspect ratio
- **Hyperparameters** (lines 261-280): ASPECT_RATIO, HEAD_DIM, batch sizes, learning rates, schedule ratios
- **loss_fn** (line 319): Simple cross-entropy loss wrapper, importable by other modules
- **Training loop** (inside `__main__`): Two-phase approach:
  1. Warmup (11 uncompiled steps): absorbs graph compilation overhead, measures step time
  2. Compiled training: `mx.compile` with `inputs/outputs` state tracking, step-based LR schedule via `optim.join_schedules`
- **Caches** (lines 33-62): `_norm_weight_cache` and `_sliding_window_mask_cache` -- shared across imports

The `__main__` guard allows importing model classes and constants (GPT, GPTConfig, loss_fn, hyperparams) without triggering training.

### prepare.py (READ ONLY)

- **Constants**: `MAX_SEQ_LEN=2048`, `TIME_BUDGET=300`, `EVAL_TOKENS`
- **Tokenizer**: BPE via rustbpe/tiktoken, cached at `~/.cache/autoresearch/`
- **make_dataloader**: BOS-aligned packing with best-fit, numpy buffer -> mx.array
- **evaluate_bpb**: Fixed evaluation metric (DO NOT CHANGE). Uses `reduction='none'` and token byte counts

Numpy usage in prepare.py is intentional: CPU-side data packing in `make_dataloader` uses a numpy buffer (`row_buffer`) for efficient token packing, converting to `mx.array` once per batch yield. No numpy in the training hot path.

### bench.py

Runs 20 uncompiled + 10 compiled training steps with per-phase instrumentation:
- Data loading, fwd+bwd, optimizer update timing
- Compiled vs uncompiled comparison
- Eval timing at batch=32 and batch=64, compiled vs uncompiled
- Memory and cache verification

### log_utils.py

Logging, diagnostics, and output formatting utilities. Keep `train.py` and `prepare.py` minimal -- measurement helpers, memory sampling, step timing serialization, and similar instrumentation belong here, not inline in core scripts.

```python
from log_utils import logger, is_debug, sample_memory, format_step_timings
from log_utils import hardware_info, save_json, build_bench_data, FORMAT_VERSION

logger.debug("Only shown with --debug")
active_mb, peak_mb = sample_memory(step)  # periodic memory sampling
save_json("run", run_data)                # structured JSON output (dict built inline in train.py)
save_json("bench", build_bench_data(...)) # structured JSON output
```

## Code Organization Convention

**Keep core scripts thin.** `train.py` is the model + training loop; `prepare.py` is data + eval. Diagnostics, profiling helpers, output formatting, and logging utilities go in `log_utils.py` (or dedicated utility modules if they grow). Do not add instrumentation, benchmark logging, or formatting code directly into core scripts.

## Git and Privacy Conventions

- `AGENTS.md` and `CLAUDE.md` are committed (not gitignored)
- `internal/log/` is committed -- this is an open research repo, session logs document our process
- `data/` directory is tracked (via `.gitkeep`) but contents are gitignored
- **Privacy**: Never include absolute filesystem paths, usernames, or system-specific details in committed files. Keep session logs focused on technical decisions and results.
- **Benchmarks**: When comparing against other implementations, report numbers factually. Do not disparage other research efforts.

## Constraints

- Only `train.py` may be modified during experiment runs
- No new dependencies -- only what is in `pyproject.toml`
- Training runs for exactly 5 minutes (wall clock, excluding startup/compilation)
- The metric is `val_bpb` (lower is better)
- The evaluation function in `prepare.py` is the ground truth and must not be changed

## Hardware

- Target: M2 Ultra, 192GB unified memory
- ~40K tok/sec compiled training (flat throughout, no regression with 20 shards)
- DEVICE_BATCH_SIZE=32 uses ~49GB peak memory (training), ~63GB overall (eval at batch=64)
- DEVICE_BATCH_SIZE=64 crashes during training (under investigation)

## MLX Framework Notes

- **Lazy evaluation**: Operations build a graph; `mx.eval()` materializes results
- **Unified memory**: CPU and GPU share memory; no device transfers
- **mx.fast ops**: Always use `mx.fast.rms_norm`, `mx.fast.rope`, `mx.fast.scaled_dot_product_attention`
- **mx.compile**: Fuses element-wise ops. Use `inputs/outputs` for state tracking. Avoid Python scalar constants that change (causes recompilation).
- **Type promotion**: Use Python scalars (not `mx.array`) for constants in bf16 code

## Current Optimization State (v0.5.0)

| Optimization | Status | Impact |
|-------------|--------|--------|
| mx.compile on training | Done | 15% speedup (40.5K -> 46.5K tok/sec) |
| Step-based LR schedule | Done | Enables mx.compile (no recompilation) |
| Eval batch=64 | Done | No speedup (compute-bound, not batch-overhead) |
| Mask caching | Done | Avoids recomputation each forward pass |
| Per-step grad eval | Done | Reduces peak memory in grad accumulation |
| Muon optimizer | Done | Muon for 2D+ matrix params, AdamW for rest |
| 5-group LR routing | Done | Per-param LR/betas matching PyTorch baseline |
| Batch=64 investigation | TODO | Memory headroom exists (53GB/192GB) |
| Async data loading | Skip | Only 1.2% of step time |

## 5-Group MultiOptimizer Routing

| Group | Optimizer | Params | LR | Betas | Filter |
|-------|-----------|--------|-----|-------|--------|
| 1 | Muon | layers 2D+ weights (excl. ve_gate) | MATRIX_LR (0.04) | momentum=0.95 | `is_muon_param` |
| 2 | AdamW | wte + value_embeds | EMBEDDING_LR * dmodel_scale | (0.8, 0.95) | `is_embedding` |
| 3 | AdamW | x0_lambdas | SCALAR_LR * dmodel_scale | (0.96, 0.95) | `is_x0_lambdas` |
| 4 | AdamW | resid_lambdas | SCALAR_LR * 0.01 * dmodel_scale | (0.8, 0.95) | `is_resid_lambdas` |
| 5 (fallback) | AdamW | lm_head + ve_gate | UNEMBEDDING_LR * dmodel_scale | (0.8, 0.95) | (default) |

**Note**: `dmodel_scale = (n_embd / 768)^-0.5`. At depth=8 (n_embd=512), dmodel_scale=1.225. The reference implementation does NOT apply dmodel_scale to scalar LRs (groups 3-4); our code does. This is a deliberate deviation -- empirical comparison will determine if it helps smaller models.

## Implementation Notes

- **Dict-based layers**: `self.layers` uses `{"l0": ..., "l1": ...}` keys (not `[Block(...)]` or `{"0": ...}`). MLX's `tree_unflatten` converts string-digit keys back to lists, which breaks `MultiOptimizer`. The `"l"` prefix prevents this.
- **MultiOptimizer filters**: 4 filter functions route params to groups 1-4; unmatched params fall through to group 5 (fallback). Order matters -- first match wins.
- **Compiled step + grad_accum**: The compiled training step only works when `grad_accum_steps == 1`. With `grad_accum > 1`, the fallback is uncompiled (~15% slower) because intermediate micro-steps need explicit eval for accumulation.
- **Schedule swap**: Phase 2 swaps LR schedules via `opt._schedulers['learning_rate']` on existing optimizer instances. Preserves Adam momentum buffers and step counters naturally (no `_state['step']` hacking).
- **Muon momentum**: Fixed at 0.95 (reference ramps 0.85->0.95 over 300 steps). Deliberate simplification for mx.compile compatibility.
- **Weight decay**: Fixed at 0.2 (reference decays as `WD * (1-progress)`). Same mx.compile limitation.
- **Session logging**: Update `internal/log/log_YYYY-MM-DD.md` every iteration with what was done, decisions made, and open questions.

## Data Investigations

See [internal/data-investigations.md](internal/data-investigations.md) for the backlog of data quality levers under investigation. The current approach is purely algorithm-driven; data quality is an underexplored lever.

## Experiment Workflow

See `program.md` for the full autonomous experiment loop. In short:
1. Edit `train.py` with an idea
2. Commit
3. Run `uv run train.py > run.log 2>&1`
4. Check: `grep "^val_bpb:\|^peak_memory_mb:" run.log` (pipe through `tr '\r' '\n'` for carriage returns)
5. If improved, keep. If not, revert.
6. Log to `results.tsv` (tab-separated: commit, val_bpb, memory_gb, status, description)
