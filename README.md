# autoresearch-mlx

An MLX port of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for Apple Silicon.

An AI agent autonomously experiments with model architecture and data quality to minimize `val_bpb` within a fixed 5-minute training budget. This fork replaces the PyTorch/CUDA backend with [MLX](https://github.com/ml-explore/mlx) so the workflow runs natively on Mac.

## Quick start

**Requirements:** Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 3. Run a single training experiment (~5 min)
uv run train.py
```

For TinyStories (smaller dataset, good for smaller machines):

```bash
uv run prepare.py --dataset tinystories
# Then set DATASET = "tinystories" in train.py before running
```

## Architecture

The project is organized around two autonomous experiment programs that share a common training/evaluation infrastructure but operate on different parts of the codebase:

```
                  +-------------------+
                  |   evaluate_bpb    |  <-- ground truth metric (locked)
                  +-------------------+
                          |
            +-------------+-------------+
            |                           |
    +-------v-------+          +--------v--------+
    | Model Program |          | Data Program    |
    | (program.md)  |          | (program_data.md)|
    +-------+-------+          +--------+--------+
            |                           |
     edits train.py            edits prepare.py
     (architecture,            + data_sources.py
      optimizer,               (filtering, tokenizer,
      hyperparams)              curriculum, mixing)
            |                           |
            +-------------+-------------+
                          |
                  +-------v-------+
                  |  train.py     |  <-- 5-minute training run
                  +-------+-------+
                          |
              +-----------+-----------+
              |                       |
      +-------v-------+      +-------v-------+
      | run.log       |      | data/         |
      | (raw output   |      | last_run.json |
      |  for crashes) |      | run_*.json    |
      +---------------+      +---------------+
```

### Separation of concerns

The two programs have **non-overlapping edit scopes** by design. This prevents an agent from making coupled changes across model and data simultaneously, which would be hard to attribute and hard to revert.

| | Model program | Data program |
|---|---|---|
| **Edits** | `train.py` only | `prepare.py` + `data_sources.py` |
| **Read-only** | `prepare.py`, `data_sources.py` | `train.py` |
| **Locked** | `evaluate_bpb` | `evaluate_bpb` |
| **Branch prefix** | `autoresearch/<tag>` | `autoresearch-data/<tag>` |
| **Typical cycle** | ~6 min (5 train + 1 eval) | ~8 min (2 prepare + 5 train + 1 eval) |

Changes that span both (e.g., token-level loss weighting) are flagged for human-directed work.

### Dual-channel output

`train.py` produces two output channels:

1. **`run.log`** -- raw stdout+stderr capture (`> run.log 2>&1`). Contains progress lines with `\r` carriage returns and the human-readable summary block. The agent uses this for crash diagnostics (`tail -50 run.log`). Overwritten each run, gitignored.

2. **`data/last_run.json`** -- structured JSON written by `log_utils.save_json`. Contains all metrics in a machine-readable format. The agent reads this for results instead of grepping stdout. A timestamped copy (`data/run_YYYYMMDD_HHMMSS.json`) is archived alongside it.

This separates the human interface (stdout text) from the machine interface (structured JSON). The upstream [karpathy/autoresearch](https://github.com/karpathy/autoresearch) uses only the grep-based approach; the structured JSON output is our addition.

### Data flow

```
train.py finishes
    |
    +-- stdout/stderr --> run.log (overwritten, gitignored)
    |                       \-- agent reads on crash: tail -50 run.log
    |
    +-- save_json() --> data/last_run.json (stable path, always latest)
    |               \-> data/run_20260315_142301.json (timestamped archive)
    |                       \-- agent reads for metrics
    |
    +-- agent logs --> results.tsv (append-only, gitignored)
    |                   \-- 6 columns: commit, val_bpb, memory_gb, avg_tok_sec, status, description
    |
    +-- analysis.py reads --> data/run_*.json + results.tsv
```

### Experiment loop

Both programs follow the same loop (model program shown):

1. Edit `train.py` with an idea
2. `git commit`
3. `uv run train.py > run.log 2>&1`
4. Read `data/last_run.json` for metrics
5. Keep (advance branch) or discard (`git reset`) based on quality+throughput framework
6. Log to `results.tsv`
7. Repeat indefinitely

The agent runs autonomously on a dedicated branch. It never pushes to remote. The human can walk away and come back to a `results.tsv` full of experiments.

## Running the agent

### Interactive mode

```bash
claude
```

Then paste:
```
Read program.md and follow the instructions. Let's use run tag "mar15".
```

### Autonomous mode (unattended)

The project ships `.claude/settings.json` with a scoped allowlist. `git push` is explicitly denied.

```bash
# Model experiments
claude -p "Read program.md and follow the experiment loop instructions. Run tag: mar15. Dataset: climbmix. Do not ask for confirmation -- start the loop immediately."

# Data experiments
claude -p "Read program_data.md and follow the experiment loop instructions. Run tag: mar15-data. Dataset: climbmix. Do not ask for confirmation -- start the loop immediately."
```

See [docs/guide.md](docs/guide.md) for full details on permissions, safety, and tuning for smaller machines.

## Project structure

```
train.py          - model, optimizer, training loop (model program edits)
prepare.py        - data prep, tokenizer, dataloader, evaluate_bpb (data program edits)
data_sources.py   - dataset registry and configuration (data program edits)
log_utils.py      - structured output, diagnostics, logging
program.md        - model experiment agent instructions
program_data.md   - data experiment agent instructions
bench.py          - performance profiling (compiled vs uncompiled)
analysis.py       - experiment results analysis (reads run_*.json + results.tsv)
docs/guide.md     - usage guide for all modes
tests/            - test suite
data/             - run archives (last_run.json, run_*.json, bench_*.json)
internal/log/     - session-by-session development notes
```

## Current results (v0.7.1)

Tested on M2 Ultra, 192GB unified memory. 5-minute training budget, 11.5M parameter GPT with value embeddings.

| Metric | Value |
|--------|-------|
| val_bpb | 1.859 |
| Training steps | 641 |
| Avg throughput | 139,999 tok/sec |
| Peak memory | 10.6 GB |
| Total time (train + eval) | 350.7s |

DEPTH=4, 5-group MultiOptimizer (Muon + AdamW), Muon momentum ramp 0.85->0.95, weight decay linear decay to 0.

## What changed from the original

- PyTorch/CUDA replaced with pure MLX for Apple Silicon
- `mx.fast.*` ops (SDPA, RoPE, RMS norm) instead of Flash Attention 3 and manual implementations
- Multi-dataset support via `data_sources.py`
- Autonomous data experiment loop alongside the model experiment loop
- Structured JSON output (`data/last_run.json`) for machine-readable results -- upstream uses grep-from-stdout only
- Scoped Claude Code permissions for safe unattended operation
- File path enforcement hook for sandboxed agent execution

## Documentation

- [docs/guide.md](docs/guide.md) -- step-by-step for all modes (model, data, engineering, manual, autonomous)
- [AGENTS.md](AGENTS.md) -- accumulated technical knowledge (optimizer routing, MLX gotchas, training results)
- [internal/data-investigations.md](internal/data-investigations.md) -- data quality investigation backlog
- [internal/ane-integration.md](internal/ane-integration.md) -- ANE integration roadmap
- [internal/analysis/](internal/analysis/) -- throughput regression and eval bottleneck analyses
- [internal/log/](internal/log/) -- session-by-session development notes

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) -- the OG
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) -- another MLX port whose findings on DEPTH=4 and hyperparameter tuning informed our baseline reset in v0.6.0

## License

MIT
