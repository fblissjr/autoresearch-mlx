last updated: 2026-03-09

# CLAUDE.md

Autonomous ML research framework: trains GPT models using MLX on Apple Silicon. An AI agent iterates on `train.py` to minimize val_bpb within a fixed 5-minute training budget.

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
| Debug mode | `AUTORESEARCH_DEBUG=1` or `--debug` flag |

## Project Structure

```
train.py         -- Model + training loop (the ONLY file the agent edits during experiments)
prepare.py       -- Data prep, tokenizer, dataloader, evaluate_bpb (minimize changes, keep aligned with upstream)
data_sources.py  -- Dataset registry and configuration (multi-dataset support)
bench.py         -- Performance profiling
analysis.py      -- Experiment results analysis
log_utils.py     -- Logging, diagnostics, output formatting utilities
program.md       -- Autonomous experiment loop instructions
tests/           -- Test suite
data/            -- Output files (gitignored contents, tracked via .gitkeep)
internal/        -- Research notes and session logs (committed)
  log/           -- Session logs (log_YYYY-MM-DD.md)
```

## Constraints

- Only `train.py` may be modified during experiment runs
- No new dependencies -- only what is in `pyproject.toml`
- Training runs for exactly 5 minutes (wall clock, excluding startup/compilation)
- The metric is `val_bpb` (lower is better)
- `evaluate_bpb` in `prepare.py` is the ground truth and must not be changed
- `prepare.py` should be minimally modified to stay aligned with upstream
- Dataset configuration lives in `data_sources.py`, not in prepare.py
- `program.md` must match actual train.py output format and field names -- verify after changing train.py output

## Code Conventions

- Keep core scripts thin. Diagnostics, profiling, formatting go in `log_utils.py`.
- New modules: avoid names that conflict with popular packages (e.g., don't use "datasets")
- When using module globals as function default params, use `None` + runtime resolve (defaults bind at definition time)
- `data_sources.configure_dataset()` updates prepare.py module globals via `prepare.DATA_DIR = ...`; functions that read globals in their body pick up changes correctly

## Git and Privacy

- `CLAUDE.md` and `AGENTS.md` are committed
- `internal/log/` is committed -- open research repo
- `data/` tracked via `.gitkeep`, contents gitignored
- Never include absolute paths, usernames, or system-specific details in committed files
- Use neutral language when referencing other implementations (not "competing", "rival", etc.)

## Session Logging (Required)

After every commit, update `internal/log/log_YYYY-MM-DD.md` (today's date) with what was done. This is a blocking requirement -- do not skip it. Each entry should include:
- What changed and why
- Key technical findings or decisions
- Any new open questions

If the log file for today doesn't exist, create it with the standard header format (see existing logs for reference).

## Experiment Workflow

See `program.md` for the full autonomous loop. In short:
1. Edit `train.py` with an idea
2. Commit
3. Run `uv run train.py > run.log 2>&1`
4. Check: `grep "^val_bpb:\|^peak_memory_mb:" run.log` (pipe through `tr '\r' '\n'` for carriage returns)
5. If improved, keep. If not, revert.
6. Log to `results.tsv` (tab-separated: commit, val_bpb, memory_gb, status, description)

## MLX Notes

- **Lazy evaluation**: operations build a graph; `mx.eval()` materializes results
- **Unified memory**: CPU and GPU share memory; no device transfers
- **mx.fast ops**: Always use `mx.fast.rms_norm`, `mx.fast.rope`, `mx.fast.scaled_dot_product_attention`
- **mx.compile**: Fuses ops. Use `inputs/outputs` for state tracking. Avoid changing Python scalar constants (causes recompilation).
- **Type promotion**: Use Python scalars (not `mx.array`) for constants in bf16 code

## Current State (v0.6.0)

- DEPTH=4, DEVICE_BATCH_SIZE=16 (baseline reset for performance catch-up)
- 5-group MultiOptimizer: Muon (matrix), AdamW (embeds, x0_lambdas, resid_lambdas, fallback)
- Multi-dataset: climbmix (default) and tinystories via `data_sources.py`
- Compiled training when grad_accum=1; uncompiled fallback otherwise

See [AGENTS.md](AGENTS.md) for accumulated technical knowledge: optimizer routing, implementation notes, known limitations, analysis links.
