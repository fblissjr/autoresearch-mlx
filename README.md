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
prepare.py      - constants, data prep + runtime utilities (do not modify)
train.py        - model, optimizer, training loop (agent modifies this)
program.md      - agent instructions
pyproject.toml  - dependencies
```

## License

MIT
