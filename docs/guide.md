# Usage Guide

This repo supports three activities, all measured against the same metric: `val_bpb` (lower is better).

## Model experiments

Edit `train.py` to improve model architecture, optimizer, hyperparameters, or training loop. The data pipeline is fixed.

**Full instructions**: [program.md](../program.md)

**Quick summary**:
1. Create branch `autoresearch/<tag>`
2. Edit `train.py` with an idea
3. Commit, run `uv run train.py > run.log 2>&1`
4. Read results from `data/last_run.json`
5. Keep or discard based on val_bpb + throughput

**Technical reference**: [AGENTS.md](../AGENTS.md) -- optimizer routing, MLX gotchas, compiled training, accumulated findings.

## Data experiments

Edit `prepare.py` and `data_sources.py` to improve data quality, filtering, ordering, tokenization, or mixing. The model is fixed.

**Full instructions**: [program_data.md](../program_data.md)

**Quick summary**:
1. Create branch `autoresearch-data/<tag>`
2. Edit `prepare.py` / `data_sources.py` with a data idea
3. If pipeline changed structurally: run `uv run prepare.py --dataset <name>`
4. Commit, run `uv run train.py > run.log 2>&1`
5. Read results from `data/last_run.json`, keep or discard based on val_bpb + throughput

**Backlog**: [internal/data-investigations.md](../internal/data-investigations.md) -- quality analysis, experiment ideas.

**Key reference**: NVIDIA Nemotron 3 Super technical report (Section 2.3) -- two-phase curriculum, quality-tiered mixing, synthetic data augmentation, deduplication. These techniques informed the data experiment backlog.

### What requires re-running prepare.py

| Change type | Needs prepare.py? | Notes |
|---|---|---|
| Dataloader packing/ordering | No | Changes take effect on next `train.py` run |
| Document filtering in `text_iterator` | Yes | Affects shard contents |
| New dataset in `data_sources.py` | Yes | Downloads and shards data |
| Tokenizer vocab size change | Yes | Retrains tokenizer; `train.py` auto-adapts |
| Shard count change | Yes | Re-downloads shards |

## Engineering work

Infrastructure improvements to throughput, hardware utilization, or developer tooling. Not experiment-loop work -- validated against val_bpb to ensure no regressions.

**ANE integration**: [internal/ane-integration.md](../internal/ane-integration.md) -- phased roadmap for Apple Neural Engine acceleration. Gated on Phase 0 benchmark.

**Throughput work**: Handled within the model experiment loop via the quality+throughput decision framework in [program.md](../program.md).

## Running with Claude Code

Both experiment programs (model and data) are designed to run as autonomous loops inside Claude Code. The `experiment-plugin/` directory provides namespaced skills for launching experiments.

### Loading the plugin

The experiment skills require the plugin to be loaded:

```bash
claude --plugin-dir ./experiment-plugin
```

Recommended shell alias (add to your `.zshrc` or `.bashrc`):

```bash
alias ar='claude --plugin-dir ./experiment-plugin'
```

Then just use `ar` to start Claude Code with the experiment plugin.

### Run tags

A **run tag** is just a label that becomes the git branch name for an experiment session. It can be anything -- the agent creates a branch like `autoresearch/<tag>` or `autoresearch-data/<tag>` and runs all experiments there.

Recommended convention: `<date>-<focus>` or just `<date>`. Examples:

| Program | Tag | Branch created |
|---------|-----|----------------|
| Model | `mar15` | `autoresearch/mar15` |
| Model | `mar15-muon-lr` | `autoresearch/mar15-muon-lr` |
| Data | `mar15-data` | `autoresearch-data/mar15-data` |
| Data | `mar15-dedup` | `autoresearch-data/mar15-dedup` |

Tags must be unique -- the agent will refuse to reuse an existing branch. This lets you run multiple sessions and tell them apart in `git log`.

### Available skills

| Skill | Description |
|-------|-------------|
| `/experiment:model <tag> [dataset]` | Launch autonomous model experiment loop (edits train.py) |
| `/experiment:data <tag> [dataset]` | Launch autonomous data experiment loop (edits prepare.py + data_sources.py) |
| `/experiment:run [description]` | Run a single experiment cycle (commit, train, extract, log) |
| `/experiment:compare [count]` | Compare recent training runs |

### Interactive mode (human-in-the-loop)

Start Claude Code with the plugin (or use the `ar` alias) and invoke a skill:

```
/experiment:model mar15
/experiment:data mar15-data
```

### Autonomous mode (unattended)

For overnight or hands-free runs where no human is present to approve tool calls.

**Step 1: Permissions are pre-configured.** The project ships `.claude/settings.json` with a scoped allowlist covering all experiment operations (file reads/edits, git commits, training runs, result extraction). `git push` is explicitly denied -- the agent works locally only.

**Step 2: Launch with a non-interactive prompt.** Use `-p` to provide the full prompt upfront:

```bash
# Model experiments
claude --plugin-dir ./experiment-plugin -p "/experiment:model mar15"

# Data experiments
claude --plugin-dir ./experiment-plugin -p "/experiment:data mar15-data"
```

Or with the alias:
```bash
ar -p "/experiment:model mar15"
ar -p "/experiment:data mar15-data"
```

**Step 3: Walk away.** The agent will run experiments until interrupted with Ctrl+C.

### What the allowlist permits

The committed `.claude/settings.json` allows:
- File operations: Read, Edit, Write (for editing experiment files and logging)
- Specific scripts only: `uv run train.py`, `uv run prepare.py`, `uv run bench.py`, `uv run analysis.py`, `uv run python tests/*`
- Git operations: add, commit, checkout, branch, reset, log, diff, status (for the experiment loop)
- Shell utilities: grep, tail, head, tr, ls, cat, pgrep, printf (for crash diagnostics)
- Process management: `pkill -f train.py`, `pkill -f prepare.py` (for killing hung runs only)

**Explicitly denied**:
- `git push` -- the agent never pushes to remote
- `uv run python -c` -- prevents arbitrary Python execution
- `uv run --script` -- prevents running arbitrary scripts
- General `pkill` -- only training processes can be killed

### Safety notes

- The agent works on a dedicated branch (`autoresearch/<tag>` or `autoresearch-data/<tag>`). Master is never modified.
- `git reset --hard` is allowed because the experiment loop uses it to discard failed experiments on the experiment branch. This is expected and safe.
- The agent commits frequently (one commit per experiment). If something goes wrong, `git log` and `git reset` can recover any state.
- `uv run` is scoped to known scripts -- the agent cannot run `uv run python -c "..."` or arbitrary Python files. This prevents accidental code execution outside the experiment loop.
- `pkill` is scoped to training processes only -- the agent cannot kill unrelated processes.
- To add personal overrides without affecting the project, use `.claude/settings.local.json` (gitignored).

## Manual operations

### Single training run

```bash
uv run train.py
```

Runs for exactly 5 minutes (wall clock, excluding startup/compilation). Prints summary metrics at the end.

### Prepare data

```bash
# climbmix (default, ~2 min first time)
uv run prepare.py

# tinystories (smaller, faster)
uv run prepare.py --dataset tinystories
```

Downloads data, trains tokenizer, creates shards. Only needed once per dataset, or when data pipeline changes.

### Benchmark

```bash
uv run bench.py
```

Profiles compiled vs uncompiled training and eval. Useful before starting experiments and when investigating throughput regressions.

### Analyze results

```bash
uv run analysis.py
```

Reads `results.tsv` and `data/run_*.json`. Shows experiment progression, throughput trends, and kept-experiment summary.

### Tuning for smaller machines

If running on a MacBook or other smaller Apple Silicon machine:

1. **Use TinyStories**: `uv run prepare.py --dataset tinystories`, then set `DATASET = "tinystories"` in `train.py`. Lower entropy data, reasonable results with smaller models.
2. **Add a new dataset**: Add an entry to `DATASETS` in `data_sources.py` with appropriate `vocab_size`, `max_seq_len`, and `eval_tokens`. Run `uv run prepare.py --dataset <name>`.
3. **Lower DEPTH** in `train.py` (default 4). Primary complexity knob -- model_dim, head count, and parameter count all derive from it.
4. **Adjust DEVICE_BATCH_SIZE** in `train.py`. Tokens per fwd/bwd = `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`.
5. **Lower TOTAL_BATCH_SIZE** in `train.py`, keep it a power of 2 (e.g., `2**14` ~16K).
