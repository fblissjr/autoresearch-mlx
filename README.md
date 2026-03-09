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
- [internal/log/](internal/log/) -- session-by-session development notes\

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Acknowledgements

- [MLX] -- (https://github.com/ml-explore/mlx)
- [https://github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch) -- the OG
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) -- another MLX port whose findings on DEPTH=4 and hyperparameter tuning informed our baseline reset in v0.6.0.

## License

MIT
