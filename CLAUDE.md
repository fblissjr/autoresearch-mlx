last updated: 2026-03-15

# CLAUDE.md

Autonomous ML research framework: trains GPT models using MLX on Apple Silicon. AI agents iterate on the model (`train.py`) and data pipeline (`prepare.py` + `data_sources.py`) to minimize val_bpb within a fixed 5-minute training budget.

## Hardware

- Mac Studio M2 Ultra, 192GB unified memory

## Quick Reference

| Task | Command |
|------|---------|
| Prepare data (climbmix) | `uv run prepare.py` |
| Prepare data (tinystories) | `uv run prepare.py --dataset tinystories` |
| Train (5 min) | `uv run train.py` |
| Run tests | `uv run python tests/test_optimizations.py` |
| Benchmark | `uv run bench.py` |
| Analyze results | `uv run analysis.py` |
| Debug mode | `AUTORESEARCH_DEBUG=1` or `--debug` flag |

## Project Structure

```
train.py              -- Model + training loop (model program edits this)
prepare.py            -- Data prep, tokenizer, dataloader, evaluate_bpb (data program edits this)
data_sources.py       -- Dataset registry and configuration (data program edits this)
log_utils.py          -- Structured JSON output, diagnostics, logging
program.md            -- Model experiment autonomous loop
program_data.md       -- Data experiment autonomous loop
bench.py              -- Performance profiling
analysis.py           -- Experiment results analysis
docs/guide.md         -- Detailed usage guide for all modes
experiment-plugin/    -- Claude Code plugin: experiment skills and agents
  skills/model/       -- /experiment:model <tag> [dataset]
  skills/data/        -- /experiment:data <tag> [dataset]
  skills/run/         -- /experiment:run [description]
  skills/compare/     -- /experiment:compare [count]
  agents/             -- experiment-reviewer pre-flight agent
tests/                -- Test suite
data/                 -- Run archives and output (gitignored contents, tracked via .gitkeep)
  last_run.json       -- Stable path: most recent run's structured results
  run_*.json          -- Timestamped archive of every run
  bench_*.json        -- Benchmark archives
internal/             -- Research notes and session logs (committed)
  log/                -- Session logs (log_YYYY-MM-DD.md)
```

## Experiment Plugin

The `experiment-plugin/` directory is a Claude Code plugin providing namespaced skills and agents. Load it with:

```bash
claude --plugin-dir ./experiment-plugin
```

Skills and agents live **only** in `experiment-plugin/` -- do not create `.claude/skills/` or `.claude/agents/`.

| Skill | Usage | Description |
|-------|-------|-------------|
| `/experiment:model` | `/experiment:model mar15 [climbmix]` | Launch model experiment loop (reads program.md) |
| `/experiment:data` | `/experiment:data mar15-data [climbmix]` | Launch data experiment loop (reads program_data.md) |
| `/experiment:run` | `/experiment:run` | Single experiment cycle (commit, train, extract, log) |
| `/experiment:compare` | `/experiment:compare [5]` | Compare recent training runs |

The plugin also includes the `experiment-reviewer` agent for pre-flight checks on train.py changes.

## Program Architecture

Two autonomous programs with non-overlapping edit scopes:

- **Model experiments** (program.md): only `train.py` may be modified. `prepare.py` and `data_sources.py` are read-only.
- **Data experiments** (program_data.md): `prepare.py` and `data_sources.py` may be modified. `train.py` is read-only.
- Both: `evaluate_bpb` in `prepare.py` is the ground truth and must not be changed.

Cross-cutting changes (e.g., token loss weighting) require human direction.

## Constraints

- No new dependencies -- only what is in `pyproject.toml`
- Training runs for exactly 5 minutes (wall clock, excluding startup/compilation)
- The metric is `val_bpb` (lower is better)
- Dataset configuration lives in `data_sources.py`, not in prepare.py
- `program.md` must match actual train.py output format and field names -- verify after changing train.py output

## Experiment Workflow

Both programs follow the same loop. In short (model program):
1. Edit `train.py` with an idea
2. Commit
3. Run `uv run train.py > run.log 2>&1`
4. Read results from `data/last_run.json`
5. If improved, keep. If not, revert.
6. Log to `results.tsv` (tab-separated: commit, val_bpb, memory_gb, avg_tok_sec, status, description)

## Output Channels

`train.py` produces two output channels with different consumers:

| Channel | Path | Format | Consumer | Lifetime |
|---------|------|--------|----------|----------|
| Raw log | `run.log` | stdout+stderr text | Agent (crash diagnostics), human (progress) | Overwritten each run, gitignored |
| Structured results | `data/last_run.json` | JSON | Agent (metric extraction) | Overwritten each run, gitignored |
| Archived results | `data/run_YYYYMMDD_HHMMSS.json` | JSON | `analysis.py`, human review | Permanent, gitignored |
| Experiment log | `results.tsv` | TSV (6 columns) | Agent (session tracking), `analysis.py` | Append-only, gitignored |

The agent reads `data/last_run.json` for metrics (not grep). On crash, `data/last_run.json` won't be created; the agent reads `tail -50 run.log` for the stack trace.

Key fields in `last_run.json`: `result.val_bpb`, `training.peak_memory_mb`, `training.avg_tok_sec`, `training.total_steps`, `training.eval_seconds`.

## Code Conventions

- Keep core scripts thin. Diagnostics, profiling, formatting go in `log_utils.py`.
- New modules: avoid names that conflict with popular packages (e.g., don't use "datasets")
- When using module globals as function default params, use `None` + runtime resolve (defaults bind at definition time)
- `data_sources.configure_dataset()` updates prepare.py module globals via `prepare.DATA_DIR = ...`; functions that read globals in their body pick up changes correctly
- `save_json(prefix, data, write_latest=False)`: only training runs pass `write_latest=True` to update `data/last_run.json`

## Claude Code Hooks

- Hook input comes via **stdin as JSON** (not env vars). Fields: `tool_input`, `tool_name`, `hook_event_name`, etc.
- Project dir env var is `CLAUDE_PROJECT_DIR` (not `PROJECT_DIR`)
- PreToolUse hooks that exit non-zero block the tool call; stderr is shown as the error message
- Prefer pure shell hooks over python3 subprocesses (~50ms startup overhead per call)

## Git and Privacy

- `CLAUDE.md` and `AGENTS.md` are committed
- `internal/log/` is committed -- open research repo
- `data/` tracked via `.gitkeep`, contents gitignored
- `run.log` and `results.tsv` are gitignored
- Never include absolute paths, usernames, or system-specific details in committed files
- Use neutral language when referencing other implementations (not "competing", "rival", etc.)

## Session Logging (Required)

After every commit, update `internal/log/log_YYYY-MM-DD.md` (today's date) with what was done. This is a blocking requirement -- do not skip it. Each entry should include:
- What changed and why
- Key technical findings or decisions
- Any new open questions

If the log file for today doesn't exist, create it with the standard header format (see existing logs for reference).

## MLX Notes

- **Lazy evaluation**: operations build a graph; `mx.eval()` materializes results
- **Unified memory**: CPU and GPU share memory; no device transfers
- **mx.fast ops**: Always use `mx.fast.rms_norm`, `mx.fast.rope`, `mx.fast.scaled_dot_product_attention`
- **mx.compile**: Fuses ops. Use `inputs/outputs` for state tracking. Avoid changing Python scalar constants (causes recompilation).
- **Type promotion**: Use Python scalars (not `mx.array`) for constants in bf16 code

## Current State (v0.7.2)

- DEPTH=4, DEVICE_BATCH_SIZE=16 (baseline reset for performance catch-up)
- 5-group MultiOptimizer: Muon (matrix), AdamW (embeds, x0_lambdas, resid_lambdas, fallback)
- Muon momentum ramp 0.85->0.95 over 300 steps, weight decay linear decay to 0
- Multi-dataset: climbmix (default) and tinystories via `data_sources.py`
- Compiled training when grad_accum=1; uncompiled fallback otherwise
- Throughput is a first-class signal in the experiment loop (see program.md decision framework)
- Structured JSON output via `data/last_run.json` for agent metric extraction

See [AGENTS.md](AGENTS.md) for accumulated technical knowledge: optimizer routing, implementation notes, known limitations, analysis links.
See [internal/ane-integration.md](internal/ane-integration.md) for ANE (Apple Neural Engine) integration roadmap.
