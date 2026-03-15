# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist -- this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` -- repository context.
   - `CLAUDE.md` -- project constraints and conventions.
   - `prepare.py` -- data prep, tokenizer, dataloader, evaluation. Minimize modifications.
   - `data_sources.py` -- dataset registry and configuration (multi-dataset support).
   - `train.py` -- the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that the cache directory for the selected dataset exists and contains data shards and a tokenizer. For climbmix: `~/.cache/autoresearch/`. For tinystories: `~/.cache/autoresearch/tinystories/`. If not, tell the human to run `uv run prepare.py` (or `uv run prepare.py --dataset tinystories`).
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Run bench.py** (optional): `uv run bench.py` to establish a baseline throughput profile. Not required every session, but useful before starting and when investigating a throughput regression.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

This runs on Apple Silicon (MLX framework, unified memory). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` -- this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Change `DATASET` in `train.py` to switch between prepared datasets (e.g., `"climbmix"`, `"tinystories"`). Run `uv run prepare.py --dataset <name>` first to prepare data.

**What you CANNOT do:**
- Modify `prepare.py` or `data_sources.py`. They contain the fixed evaluation, data loading, tokenizer, and dataset configuration.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time -- it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically. This runs on Apple Silicon with unified memory (no separate VRAM). Monitor peak_memory_mb in the output.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome -- that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Apple Silicon notes

- **Throughput matters**: On Apple Silicon, more training steps in the 5-minute budget is critical. A smaller/faster model that runs more steps often beats a larger model. DEPTH=4 typically outperforms DEPTH=8 for this reason.
- **MLX specifics**: Use `mx.fast.*` ops (rms_norm, rope, scaled_dot_product_attention). Use `mx.compile` when grad_accum=1. Avoid changing Python scalar constants in compiled functions (causes recompilation).
- **Unified memory**: No CPU/GPU transfers needed. Memory pressure from optimizer state is real -- 5-group MultiOptimizer holds ~10GB of momentum/variance buffers. The code frees optimizer state before eval to manage this.
- **Batch size**: DEVICE_BATCH_SIZE is typically 16-32 (limited by memory, not compute). TOTAL_BATCH_SIZE controls gradient accumulation. With TOTAL_BATCH_SIZE=2^16 and DEVICE_BATCH_SIZE=16, grad_accum=2.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          1.859000
training_seconds: 300.1
total_seconds:    371.4
train_peak_mb:    10600.5
peak_memory_mb:   14000.2
eval_seconds:     47.8
total_tokens_M:   41.0
avg_tok_sec:      140,000
num_steps:        641
num_params_M:     12.5
depth:            4
dmodel_scale:     1.7321
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. The script also writes structured results to `data/last_run.json` (stable path, always the most recent run) and a timestamped archive `data/run_YYYYMMDD_HHMMSS.json`.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated -- commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	val_bpb	memory_gb	avg_tok_sec	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) -- use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 13.7 -- divide peak_memory_mb by 1024) -- use 0.0 for crashes
4. avg_tok_sec from last_run.json (e.g. 140000) -- use 0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	avg_tok_sec	status	description
a1b2c3d	1.859000	13.7	140000	keep	baseline (depth=4)
b2c3d4e	1.840000	13.9	138000	keep	increase MATRIX_LR to 0.05
c3d4e5f	1.870000	13.7	142000	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	0	crash	double model width (OOM)
```

## Session logging

After every commit, update `internal/log/log_YYYY-MM-DD.md` (today's date) with what was done. This is a blocking requirement from CLAUDE.md. Each entry should include what changed, key findings, and open questions.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (discard stdout -- do NOT use tee or let output flood your context)
5. Read the results from `data/last_run.json`. Key fields: `result.val_bpb`, `training.peak_memory_mb`, `training.avg_tok_sec`, `training.total_steps`, `training.eval_seconds`.
   - val_bpb is the primary metric. But also consider: if avg_tok_sec dropped significantly, you may be getting fewer training steps in the budget, which hurts convergence. If total_steps is much lower than the baseline, the change may have made the model too large or slow. If eval_seconds is unusually high, something may be wrong with the model's forward pass.
6. If `data/last_run.json` was not created (or its timestamp is stale), the run crashed. Run `tail -50 run.log` for the stack trace and training progress up to the crash. Attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. Decide whether to keep or discard the change using the **quality+throughput framework** below.
9. If keeping, you "advance" the branch. If discarding, `git reset` back to where you started.

### Keep/discard decision framework

val_bpb is the primary metric, but throughput (avg_tok_sec) is a first-class signal because faster training = more steps in the 5-minute budget = better eventual val_bpb. Use both metrics together:

**Keep if any of:**
- val_bpb improved and throughput didn't regress significantly (>15%)
- val_bpb unchanged (within ~0.001) but throughput improved meaningfully (>10%)
- val_bpb regressed slightly (<0.003) but throughput improved massively (>25%)

**Discard if:**
- val_bpb regressed and throughput didn't improve enough to compensate
- val_bpb improved marginally (<0.005) but throughput cratered (>15% drop)

These are **guidance thresholds, not hard rules.** Use judgment -- same spirit as the simplicity criterion. Note that avg_tok_sec varies ~3-5% between identical runs due to Metal scheduling and thermals. Re-run marginal cases before deciding.

The **simplicity criterion** still applies on top of this: a tiny improvement that adds ugly complexity is not worth it regardless of the metrics.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ startup and eval overhead, typically ~60-90s on Apple Silicon). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder -- read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
