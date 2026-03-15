# autoresearch -- data experiments

This is an experiment to have the LLM improve training data quality and pipeline.

## Setup

To set up a new data experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar15-data`). The branch `autoresearch-data/<tag>` must not already exist -- this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch-data/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` -- data prep, tokenizer, dataloader. You modify this.
   - `data_sources.py` -- dataset registry and configuration. You modify this.
   - `internal/data-investigations.md` -- backlog of data experiments and findings.
   - `AGENTS.md` -- accumulated technical knowledge. The "Data Pipeline" section covers how prepare.py works (numpy packing, text_iterator semantics).
   - `CLAUDE.md` -- project constraints and conventions.
   - `train.py` -- the training script. Read-only context; you run it but don't edit it.
4. **Verify data exists**: Check that the cache directory for the selected dataset exists and contains data shards and a tokenizer. For climbmix: `~/.cache/autoresearch/`. For tinystories: `~/.cache/autoresearch/tinystories/`. If not, tell the human to run `uv run prepare.py` (or `uv run prepare.py --dataset tinystories`).
5. **Establish baseline**: Run `uv run train.py > run.log 2>&1` with the current data pipeline. Read `data/last_run.json` for the baseline val_bpb.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. Record the baseline after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

This runs on Apple Silicon (MLX framework, unified memory). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN edit:**
- `data_sources.py` -- DATASETS registry, `configure_dataset()`, download helpers. Add new datasets here.
- `prepare.py` -- `text_iterator` (filtering, ordering), `make_dataloader` (packing, batching), `train_tokenizer` (vocabulary), constants (`VOCAB_SIZE`, `MAX_SEQ_LEN`, `NUM_SHARDS`, etc.)
- New pre-processing scripts -- for data filtering, synthetic data mixing, deduplication, or any other data pipeline work. Keep them in the project root.

**What you CANNOT edit:**
- `evaluate_bpb` in `prepare.py` -- the ground truth metric. This function is locked.
- `train.py` -- that's the model program's domain. The data program runs it as-is.
- `pyproject.toml` -- no new dependencies. Only use what's already available.
- **Eval data compatibility**: The validation shards must remain loadable by `evaluate_bpb`. If you change sharding format, shard naming, or tokenization, verify that the eval path still works before committing. A broken eval loader produces confusing crashes, not a clear error.

**The goal is simple: get the lowest val_bpb** by improving the data that the model trains on. The model architecture and training loop are fixed -- you're optimizing what the model sees, not how it learns.

**Important differences from the model experiment loop (program.md):**
- Some experiments require re-running `uv run prepare.py` (adds ~2 min). Changes to data filtering, sharding, or the tokenizer all need a prepare step. Dataloader-only changes (e.g., packing strategy, batch ordering) don't.
- Changing `VOCAB_SIZE` triggers tokenizer retraining. `train.py` auto-adapts because it reads vocab_size from the loaded tokenizer via `tokenizer.get_vocab_size()`.
- Data changes affect all future training runs (unlike model experiments which are self-contained). Be careful with destructive changes to cached data. When in doubt, use a separate cache directory.

**Memory** is a soft constraint. Data pipeline changes can affect memory profile (e.g., changing MAX_SEQ_LEN, VOCAB_SIZE, or packing strategy). Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically. This runs on Apple Silicon with unified memory (no separate VRAM). Monitor peak_memory_mb in the output.

**Simplicity criterion**: All else being equal, simpler is better. A complex filtering scheme that gains 0.001 bpb isn't worth it. Conversely, removing something and getting equal or better results is a great outcome -- that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

## Output format

Same as the model program. Once the script finishes it prints:

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

The script also writes structured results to `data/last_run.json` (stable path, always the most recent run) and a timestamped archive `data/run_YYYYMMDD_HHMMSS.json`.

## Logging results

Same format as the model program. Log to `results.tsv` (tab-separated, NOT comma-separated).

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

## Session logging

After every commit, update `internal/log/log_YYYY-MM-DD.md` (today's date) with what was done. This is a blocking requirement from CLAUDE.md. Each entry should include what changed, key findings, and open questions.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch-data/mar15-data`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Edit `prepare.py` and/or `data_sources.py` with a data idea.
3. If the data pipeline changed structurally (tokenizer retrained, sharding changed, new dataset added): run `uv run prepare.py --dataset <name>` before training.
4. git commit
5. Run the experiment: `uv run train.py > run.log 2>&1` (discard stdout -- do NOT use tee or let output flood your context)
6. Read the results from `data/last_run.json`. Key fields: `result.val_bpb`, `training.peak_memory_mb`, `training.avg_tok_sec`, `training.total_steps`, `training.eval_seconds`.
   - val_bpb is the primary metric. Also watch avg_tok_sec -- data pipeline changes shouldn't tank throughput. If they do, the dataloader or tokenizer change may be introducing overhead.
7. If `data/last_run.json` was not created (or its timestamp is stale), the run crashed. Run `tail -50 run.log` for the stack trace and training progress up to the crash. Attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. Decide whether to keep or discard the change using the **quality+throughput framework** below.
10. If keeping, you "advance" the branch. If discarding, `git reset` back to where you started.

### Keep/discard decision framework

Same as the model program. val_bpb is the primary metric, throughput is a first-class signal:

**Keep if any of:**
- val_bpb improved and throughput didn't regress significantly (>15%)
- val_bpb unchanged (within ~0.001) but throughput improved meaningfully (>10%)
- val_bpb regressed slightly (<0.003) but throughput improved massively (>25%)

**Discard if:**
- val_bpb regressed and throughput didn't improve enough to compensate
- val_bpb improved marginally (<0.005) but throughput cratered (>15% drop)

These are **guidance thresholds, not hard rules.** Use judgment. Note that avg_tok_sec varies ~3-5% between identical runs due to Metal scheduling and thermals. Re-run marginal cases before deciding.

**Timeout**: Each experiment should take ~5 minutes total (+ startup and eval overhead, typically ~60-90s). If `prepare.py` was needed, add ~2 min. If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: Same as model program. Fix dumb bugs and re-run, skip fundamentally broken ideas.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder -- re-read `internal/data-investigations.md` for new angles, try combining previous near-misses, try more radical data pipeline changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~7 minutes (5 min training + 2 min prepare overhead) then you can run approx 8/hour, for a total of about 60-70 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Starting backlog

Prioritized list of data experiments to try, ordered by implementation difficulty (easy wins first). See `internal/data-investigations.md` for detailed findings.

### Quick wins

1. **More shards for diversity**: Currently using 20 of 6,542 available shards. More shards = more document diversity per training run. Zero code changes, just adjust `NUM_SHARDS`.

2. **Document length filtering**: Skip documents <500 chars to reduce BOS overhead and context fragmentation. ~8% of climbmix docs affected. Simple change to `text_iterator`.

3. **Tokenizer optimization**: Different vocab sizes (4K, 16K, 32K) may improve encoding efficiency and downstream val_bpb. Requires tokenizer retraining and a fresh `prepare.py` run, but the code change is straightforward.

### Data quality techniques

4. **Deduplication**: MinHash or exact-match dedup at document or paragraph level. climbmix likely has some dedup already, but additional passes may help.

5. **Quality-tiered data mixing**: climbmix is likely FineWeb-Edu derived. Partition documents by quality signals (length, perplexity, repetition ratio) and weight higher-quality tiers more heavily in the training mix.

6. **Two-phase curriculum**: Phase 1 = broad diversity (all shards, shuffled). Phase 2 = high-quality focused (select best documents by quality signals). Implement via data ordering in `make_dataloader` or a two-phase shard selection. (From NVIDIA Nemotron 3 Super technical report, Section 2.3.)

7. **Synthetic data augmentation**: Mix in open datasets with specialized content. Register as new datasets in `data_sources.py`. Candidates: code/algorithm concepts, formal logic and reasoning, high-quality educational content.

### Cross-program (human-directed)

8. **Token weighting via loss modification**: Weight loss by token informativeness. This requires changes to both `train.py` and `prepare.py`, which means it crosses program boundaries. Cross-cutting changes are manual human-directed work, not autonomous loop work -- the programs are deliberately scoped to prevent agents from touching too much at once. Flag this to the human if you think it's worth pursuing.
