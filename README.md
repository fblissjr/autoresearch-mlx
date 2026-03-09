# autoresearch-mlx

An MLX port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for Apple Silicon.

The original project gives an AI agent a small LLM training setup and lets it experiment autonomously. It modifies training code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. This fork replaces the PyTorch/CUDA backend with [MLX](https://github.com/ml-explore/mlx) so the same workflow runs natively on Mac.

## Quick start

**Requirements:** Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

For TinyStories (smaller dataset, faster iteration, good for smaller machines):

```bash
uv run prepare.py --dataset tinystories
# Then set DATASET = "tinystories" in train.py before running
```

## MLX optimization approach

MLX-specific optimizations in this port draw on patterns documented in [mlx-skills](https://github.com/fblissjr/mlx-skills/), a collection of MLX optimization knowledge for Apple Silicon ML development. Key techniques: `mx.compile` with state tracking, `mx.fast` ops, cache management for masks/norms, and MultiOptimizer (Muon + AdamW) for mixed-optimizer training.

Development session logs are in `internal/log/` for transparency.

## What changed from the original

- Replaced PyTorch/CUDA with pure MLX for Apple Silicon
- `mx.fast.scaled_dot_product_attention` instead of Flash Attention 3
- `mx.fast.rope` instead of manual rotary position embeddings
- `mx.fast.rms_norm` instead of `F.rms_norm`
- Functional gradients via `nn.value_and_grad` instead of `.backward()`
- Numpy-based data packing with `mx.array` conversion at yield time
- No device management (unified memory on Apple Silicon)
- Batch sizes tuned for Apple Silicon memory constraints
- Multi-dataset support via `data_sources.py` (climbmix default, TinyStories for smaller machines)

## Running the agent

Point your Claude Code (or similar) agent at `program.md`:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

## Project structure

```
train.py        - model, optimizer, training loop (agent modifies this)
prepare.py      - data prep, tokenizer, dataloader, evaluate_bpb (minimize changes)
data_sources.py - dataset registry and configuration (multi-dataset support)
bench.py        - performance profiling (compiled vs uncompiled, per-phase timing)
analysis.py     - experiment results analysis
log_utils.py    - project-wide logging framework (--debug flag support)
program.md      - agent instructions
pyproject.toml  - dependencies
tests/          - test suite (compiled eval, MultiOptimizer, etc.)
data/           - output files (charts, analysis artifacts)
internal/log/   - session-by-session development notes
```

## Current results (v0.5.2)

Tested on M2 Ultra, 192GB unified memory. 5-minute training budget, 50M parameter GPT with value embeddings.

| Metric | Value |
|--------|-------|
| val_bpb | 1.886 |
| Training steps | 193 |
| Avg throughput | 41,931 tok/sec |
| Training peak memory | 49.4 GB |
| Total time (train + eval) | 702.5s |

5-group MultiOptimizer (Muon for matrix weights, AdamW for embeddings/scalars/head), compiled training via `mx.compile`, 20 data shards for diversity.

v0.6.0 resets the baseline to DEPTH=4 (from 8) for more training steps in the 5-minute budget. New baseline results pending.

## Documentation

- [internal/analysis/2026-03-08_prepare-py-conformance.md](internal/analysis/2026-03-08_prepare-py-conformance.md) -- line-by-line verification that our MLX `prepare.py` matches the PyTorch/CUDA reference
- [internal/data-investigations.md](internal/data-investigations.md) -- data quality investigation backlog and structured output schema
- [internal/analysis/](internal/analysis/) -- throughput regression and eval bottleneck analyses
- [internal/log/](internal/log/) -- session-by-session development notes

## Tuning for smaller machines

If you're running on a MacBook or other smaller Apple Silicon machine, here are the main knobs:

1. **Use TinyStories**: `uv run prepare.py --dataset tinystories`, then set `DATASET = "tinystories"` in `train.py`. Lower entropy data gives reasonable results with much smaller models.
2. **Add a new dataset**: Add an entry to the `DATASETS` dict in `data_sources.py` with appropriate `vocab_size`, `max_seq_len`, and `eval_tokens`. Run `uv run prepare.py --dataset <name>`.
3. **Lower DEPTH** in `train.py` (default 4). This is the primary model complexity knob -- model_dim, head count, and parameter count all derive from it.
4. **Adjust DEVICE_BATCH_SIZE** in `train.py`. The number of tokens per fwd/bwd pass is `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`.
5. **Lower TOTAL_BATCH_SIZE** in `train.py`, but keep it a power of 2 (e.g., `2**14` ~16K).
6. **Try WINDOW_PATTERN = "L"** in `train.py`. The default "SSSL" alternating banded attention may be inefficient on smaller machines.

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) -- the OG
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) -- another MLX port whose findings on DEPTH=4 and hyperparameter tuning informed our baseline reset in v0.6.0.

## License

MIT
