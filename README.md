# autoresearch-mlx

An MLX port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for Apple Silicon.

An AI agent autonomously experiments with model architecture and data quality to minimize `val_bpb` within a fixed 5-minute training budget. This fork replaces the PyTorch/CUDA backend with [MLX](https://github.com/ml-explore/mlx) so the workflow runs natively on Mac.

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

## What you can do

| Mode | What it does | Instructions |
|------|-------------|--------------|
| **Model experiments** | Edit `train.py` (architecture, optimizer, hyperparams) to minimize val_bpb | [program.md](program.md) |
| **Data experiments** | Edit `prepare.py` + `data_sources.py` (filtering, tokenizer, curriculum, mixing) to minimize val_bpb | [program_data.md](program_data.md) |
| **Engineering** | Throughput and hardware work (ANE integration, compilation) | [internal/ane-integration.md](internal/ane-integration.md) |
| **Manual runs** | Training, benchmarking, analysis | [docs/guide.md](docs/guide.md) |

See [docs/guide.md](docs/guide.md) for step-by-step instructions for each mode.

## Running the agent

Point your Claude Code (or similar) agent at the appropriate program file:

```
# Model experiments
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.

# Data experiments
Hi have a look at program_data.md and let's kick off a new data experiment! let's do the setup first.
```

## What changed from the original

- PyTorch/CUDA replaced with pure MLX for Apple Silicon
- `mx.fast.*` ops (SDPA, RoPE, RMS norm) instead of Flash Attention 3 and manual implementations
- Multi-dataset support via `data_sources.py`
- Autonomous data experiment loop alongside the model experiment loop

## Project structure

```
train.py          - model, optimizer, training loop (model program modifies this)
prepare.py        - data prep, tokenizer, dataloader, evaluate_bpb (data program modifies this)
data_sources.py   - dataset registry and configuration (data program modifies this)
program.md        - model experiment agent instructions
program_data.md   - data experiment agent instructions
bench.py          - performance profiling
analysis.py       - experiment results analysis
log_utils.py      - project-wide logging framework
docs/guide.md     - detailed usage guide for all modes
pyproject.toml    - dependencies
tests/            - test suite
data/             - output files (charts, analysis artifacts)
internal/log/     - session-by-session development notes
```

## Current results (v0.7.0)

Tested on M2 Ultra, 192GB unified memory. 5-minute training budget, 11.5M parameter GPT with value embeddings.

| Metric | Value |
|--------|-------|
| val_bpb | 1.859 |
| Training steps | 641 |
| Avg throughput | 139,999 tok/sec |
| Peak memory | 10.6 GB |
| Total time (train + eval) | 350.7s |

DEPTH=4, 5-group MultiOptimizer (Muon + AdamW), Muon momentum ramp 0.85->0.95, weight decay linear decay to 0.

## Documentation

- [docs/guide.md](docs/guide.md) -- step-by-step for all modes (model experiments, data experiments, engineering, manual ops)
- [AGENTS.md](AGENTS.md) -- accumulated technical knowledge (optimizer routing, MLX gotchas, training results)
- [internal/data-investigations.md](internal/data-investigations.md) -- data quality investigation backlog
- [internal/ane-integration.md](internal/ane-integration.md) -- ANE integration roadmap
- [internal/analysis/](internal/analysis/) -- throughput regression and eval bottleneck analyses
- [internal/log/](internal/log/) -- session-by-session development notes

Data experiment backlog draws on the NVIDIA Nemotron 3 Super technical report (Section 2.3) for techniques: two-phase curriculum, quality-tiered mixing, synthetic data augmentation, deduplication.

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) -- the OG
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) -- another MLX port whose findings on DEPTH=4 and hyperparameter tuning informed our baseline reset in v0.6.0.

## License

MIT
