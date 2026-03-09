# Changelog

## 0.6.0

- Multi-dataset support: add `data_sources.py` with dataset registry and `configure_dataset()` function
- Support for TinyStories dataset (`uv run prepare.py --dataset tinystories`)
- Per-dataset cache directories (`~/.cache/autoresearch/{dataset}/`)
- Baseline reset: DEPTH=4 (from 8), DEVICE_BATCH_SIZE=16 (from 32), WARMUP_STEPS=5 (from 11)
- Add DATASET constant to train.py for dataset selection
- Minimal prepare.py changes: `--dataset` CLI flag, deferred default in `Tokenizer.from_directory`
- Add hardware info to CLAUDE.md

## 0.5.4

- Reduce EVAL_BATCH_SIZE from 64 to 32 to avoid 14GB memory surge in fragmented Metal allocator post-training
- Update AGENTS.md: optimization table to v0.5.4, add Known Limitations section, bench.py caveat, prepare.py conformance note
- Mark eval-bottleneck analysis as implemented

## 0.5.3

- Remove `build_run_data` from log_utils.py (zero-value 20-kwarg wrapper); inline dict in train.py
- Add `training_peak_mb` capture before optimizer cleanup to separate training vs eval memory peaks
- Fix data-investigations.md schema examples: replace fake numbers with type placeholders
- Document v0.5.2 training results (val_bpb 1.886, 193 steps, 42K tok/sec, no throughput regression)

## 0.5.2

- Free optimizer state before eval: delete all 5 optimizer groups and compiled_step closure, re-enable GC, clear MLX cache
- Add memory diagnostics: periodic active/peak memory sampling in step_timings via log_utils.sample_memory()
- Reset peak memory at compiled phase start for phase-isolated measurement
- Move structured JSON output (save_json, hardware_info) from train.py to log_utils.py
- Remove inline orjson/platform/os imports from train.py (now handled by log_utils)
- Download 20 data shards (up from 3) for 10x document diversity
- Add code organization convention to AGENTS.md: keep core scripts thin, utilities in log_utils.py

## 0.5.1

- Structured JSON output: restructure run/bench JSON from flat dict to nested format with format_version, hardware, model, training, result, data blocks
- Add per-step timing collection (step_timings array) to run JSON for throughput regression analysis
- Add platform metadata (chip, os) to both run and bench output

## 0.5.0

- Per-param LR groups: expand MultiOptimizer from 2 to 5 groups matching baseline PyTorch config
  - Group 1: Muon (matrix weights), Group 2: AdamW (embeddings), Group 3: AdamW (x0_lambdas, beta1=0.96),
    Group 4: AdamW (resid_lambdas, 0.01x LR), Group 5: AdamW fallback (lm_head + ve_gate)
- Add dmodel_scale = (n_embd / 768)^-0.5 to scale AdamW LRs by model width
- Save training results to data/run_<timestamp>.json (orjson)
- Save benchmark results to data/bench_<timestamp>.json (orjson)

## 0.4.0

- Muon optimizer for 2D+ matrix weights via MultiOptimizer (Muon + AdamW)
- Convert model layers and value_embeds to dict with non-digit keys to fix tree_unflatten list conversion bug
- Move window_sizes from nn.Module attribute to config method (removes list from parameter tree)
- Compiled eval in final evaluation pass (~7% faster eval)
- Fix warmup step time estimation: skip first 4 steps instead of 2
- Document grad_accum > 1 compiled fallback performance implication
- Separate LR schedules per optimizer group (MATRIX_LR for Muon, EMBEDDING_LR for AdamW)

## 0.3.1

- Add bench_compare.py: comparative benchmark against external implementations
- fwd+bwd throughput identical to coderef (MLX compiles to same kernels)
- Built-in AdamW 1.46x faster than coderef's custom per-param AdamW at D=8

## 0.3.0

- mx.compile on training step: 15% speedup (40.5K -> 46.5K tok/sec)
- Two-phase training: uncompiled warmup (11 steps) then compiled with step-based LR schedule
- Convert time-based LR schedule to step-based using optim.join_schedules
- Increase eval batch size to 64 for faster evaluation
- bench.py now compares compiled vs uncompiled training and eval

## 0.2.2

- Add bench.py profiling script (per-phase timing, memory, eval projection)
- Wrap train.py execution in __main__ guard to allow importing model/config without side effects

## 0.2.1

- Cache sliding window attention masks (avoid recomputation every forward pass)
- Evaluate gradients per micro-step to reduce peak memory in gradient accumulation
- Include loss in mx.eval call to avoid potential double-evaluation
- Remove manual GQA head repeat (SDPA handles mismatched head counts natively)
- Use Python scalar in loss denominator (avoid mx.array type promotion)
- Release accumulated_grads before evaluation to reduce peak memory

## 0.2.0

- Port from PyTorch/CUDA to pure MLX for Apple Silicon
- Replace Flash Attention 3 with mx.fast.scaled_dot_product_attention
- Replace manual RoPE with mx.fast.rope
- Replace manual RMS norm with mx.fast.rms_norm
- Replace custom MuonAdamW with MLX AdamW (Muon via MultiOptimizer planned)
- Replace torch data loading with numpy-based packing + mx.array conversion
- Remove CUDA-specific code (pinned memory, device transfers, autocast, torch.compile)
- Switch token_bytes storage from .pt to .npy format
- Reduce default batch size for Apple Silicon memory constraints

## 0.1.0

- Initial PyTorch/CUDA implementation
- Custom GPT with Value Embeddings, RoPE, sliding window attention
- MuonAdamW optimizer (Muon + AdamW hybrid)
- 5-minute fixed time budget training
- BPB evaluation metric
