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

## Running the agent

Point your Claude Code (or similar) agent at `program.md`:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

## Project structure

```
train.py        - model, optimizer, training loop (agent modifies this)
prepare.py      - constants, data prep + runtime utilities (do not modify)
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

## Documentation

- [internal/analysis/2026-03-08_prepare-py-conformance.md](internal/analysis/2026-03-08_prepare-py-conformance.md) -- line-by-line verification that our MLX `prepare.py` matches the PyTorch/CUDA reference
- [internal/data-investigations.md](internal/data-investigations.md) -- data quality investigation backlog and structured output schema
- [internal/analysis/](internal/analysis/) -- throughput regression and eval bottleneck analyses
- [internal/log/](internal/log/) -- session-by-session development notes

## License

MIT
