---
name: run-experiment
description: Run a 5-minute training experiment, capture results, and log outcomes
disable-model-invocation: true
---

# Run Experiment

Execute a full experiment cycle: commit, train, capture results, log.

## Prerequisites

Before running, verify:
1. `train.py` has uncommitted changes (the experiment)
2. Data shards exist in `~/.cache/autoresearch/`
3. No other training process is running: `pgrep -f "python.*train.py"`

## Steps

### 1. Commit the experiment

Create a short commit describing what this experiment tries. Use conventional format:
```
experiment: <short description of what changed>
```

### 2. Run training

```bash
uv run train.py > run.log 2>&1
```

This takes ~5 minutes (training) + eval time.

**Timeout**: If the run exceeds 10 minutes, kill it (`pkill -f "python.*train.py"`) and treat as failure.

### 3. Extract results

Read `data/last_run.json`. Key fields: `result.val_bpb`, `training.peak_memory_mb`, `training.avg_tok_sec`, `training.total_steps`, `training.eval_seconds`.

If `data/last_run.json` was not created (or its timestamp is stale), the run crashed. Read the traceback:
```bash
tail -50 run.log
```

### 4. Compare against previous best

Read the most recent `data/run_*.json` file (sorted by timestamp) and compare `result.val_bpb`. The new run also writes its own JSON to `data/`.

### 5. Log to results.tsv

Append a row to `results.tsv` (tab-separated):
```
<commit-hash-7char>\t<val_bpb>\t<peak_memory_gb>\t<keep|discard|crash>\t<description>
```

- Use `0.000000` / `0.0` / `crash` for crashed runs
- Round peak_memory to 0.1f GB (divide MB by 1024)

### 6. Keep or revert

- If val_bpb **improved** (lower): keep the commit, report the improvement
- If val_bpb **equal or worse**: `git reset --hard HEAD~1` to revert
- If **crashed** and fixable: fix, re-commit, re-run
- If **crashed** and unfixable: revert and log as crash

### 7. Update session log

Add entry to `internal/log/log_YYYY-MM-DD.md` with:
- What was tried
- val_bpb result and comparison to previous best
- Decision (keep/discard/crash) and reasoning

### 8. Report summary

Print a one-line summary:
```
[keep|discard|crash] val_bpb=X.XXXXXX (prev=X.XXXXXX, delta=+/-X.XXX%) | peak=XX.XGB | steps=XXX
```
