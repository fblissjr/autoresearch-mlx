last updated: 2026-03-08

# Data Investigations Backlog

The baseline autoresearch approach is algorithm-driven: optimize the model architecture, optimizer, and training procedure. Data quality is an underexplored lever that could yield significant val_bpb improvements, especially under a fixed 5-minute time budget where every token must count.

## Active

### 1. Data quality filtering on climbmix
**Status**: Investigated -- low priority
**Finding**: climbmix-400b-shuffle is already curated educational content (likely FineWeb-Edu derived). Quality is high:
- Only 0.5% of docs < 100 chars, 8.3% < 500 chars
- <0.2% low alpha ratio, <1.3% repetitive, <0.2% all-caps heavy
- No URL-heavy spam detected in 5K sample

**Data volume finding**: 12.5x surplus. We have ~126M tokens in 2 training shards but only consume ~10M per 5-min run (epoch=1). This means:
- Filtering is free (no data starvation risk)
- But the data is already clean, so simple heuristic filtering won't yield much
- The val shard is from the same distribution, so training/eval distribution mismatch isn't an issue

**Remaining lever**: Minimum document length filter (skip <500 char docs to reduce BOS overhead and context fragmentation). ~8% of docs affected. Would need either a pre-filtering script or modifying make_dataloader -- but prepare.py is READ ONLY. Options:
  1. Pre-filter parquet files and replace in cache dir (hacky but works)
  2. Propose a filter parameter to make_dataloader (requires prepare.py change)
  3. Skip -- 8% is probably not enough to move val_bpb meaningfully

**Verdict**: climbmix quality is not the bottleneck. Deprioritize simple filtering in favor of more impactful levers (curriculum learning, data mixing).

## Backlog

### 2. Download more shards (data diversity)
**Hypothesis**: We only have 2 training shards of 6,542 available. More shards = more document diversity per training run. Even though we only use ~8% of data per run, the 8% we see is drawn from the same 2 shards every time.
**Approach**: `uv run prepare.py --num-shards 20` (or more). Measure if val_bpb improves with same training config but more diverse data.
**Considerations**: Download time, cache disk space. Each shard is ~60MB. 20 shards = ~1.2GB.
**Priority**: High -- cheapest experiment, zero code changes.

### 3. Curriculum learning (data ordering)
**Hypothesis**: Ordering training data by difficulty (shorter/simpler first, harder later) could improve convergence speed. The fixed 5-minute budget makes this especially interesting -- early convergence means more useful training at the end.
**Approach**: Sort documents by perplexity (from a reference model), length, or vocabulary complexity. Train with ordered vs shuffled data.
**Considerations**: The dataloader currently does best-fit packing with random order. Changing order means modifying prepare.py (READ ONLY constraint -- would need to discuss).

### 4. Token weighting (non-uniform loss)
**Hypothesis**: Not all tokens are equally informative. Weighting loss by token informativeness could improve BPB on the evaluation set.
**Approach**: Weight loss by token byte count (already available), inverse frequency, or a learned quality signal. The `token_bytes` mask already skips some tokens; extend this to soft weighting.
**Considerations**: Modifies the loss function in train.py. Must ensure the evaluation metric (val_bpb) is unchanged.

### 5. Data mixing (domain ratios)
**Hypothesis**: FineWeb is web crawl data with uncontrolled domain distribution. Intentionally mixing domains (code, wiki, books, web) at tuned ratios could improve generalization.
**Approach**: Categorize FineWeb documents by domain, experiment with mixing ratios.
**Considerations**: Requires domain classification of training data. May need multiple data shards.

### 6. Tokenizer optimization
**Hypothesis**: The 8K BPE vocabulary is small by design, but vocabulary composition affects BPB. A vocabulary optimized for the training distribution could improve encoding efficiency.
**Approach**: Compare current tokenizer against alternatives (different vocab sizes, different BPE training data). Measure both encoding efficiency and downstream val_bpb.
**Considerations**: Tokenizer is in prepare.py (READ ONLY). Would need careful coordination.

### 7. Deduplication
**Hypothesis**: Near-duplicate documents waste training tokens. Dedup at document or paragraph level could improve effective data diversity per training step.
**Approach**: MinHash or exact-match dedup on the training data. Compare val_bpb with deduped vs original.
**Considerations**: FineWeb already has some dedup applied. Additional dedup may have diminishing returns.

## Implemented: Structured Output Format (v0.1)

Implemented in `train.py` and `bench_compare.py`. Both write structured JSON to `data/` with the following schema:

### Run output (`data/run_*.json`)

```json
{
  "format_version": "0.1",
  "timestamp": "<ISO 8601 string>",
  "hardware": {
    "chip": "<string>",
    "memory_gb": null,
    "os": "<string>"
  },
  "model": {
    "depth": "<int>",
    "n_embd": "<int>",
    "params": "<int>",
    "vocab_size": "<int>",
    "config": { "...full GPTConfig as dict..." },
    "param_counts": { "wte": "<int>", "...": "..." }
  },
  "training": {
    "budget_seconds": "<int>",
    "actual_seconds": "<float>",
    "total_seconds": "<float>",
    "total_steps": "<int>",
    "total_tokens": "<int>",
    "avg_tok_sec": "<int>",
    "training_peak_mb": "<float>  (peak during compiled training phase only)",
    "peak_memory_mb": "<float>  (overall peak including eval)",
    "optimizer_groups": "<int>",
    "compiled": "<bool>",
    "batch_size": "<int>",
    "total_batch_size": "<int>",
    "dmodel_scale": "<float>"
  },
  "result": {
    "val_bpb": "<float>"
  },
  "data": {
    "source": "climbmix-400b-shuffle",
    "filtering": "none",
    "tokenizer": "bpe-<vocab_size>"
  },
  "step_timings": [
    {"step": "<int>", "dt": "<float>", "tok_sec": "<int>", "loss": "<float>"},
    {"step": "<int>", "dt": "<float>", "tok_sec": "<int>", "loss": "<float>", "active_mb": "<float>", "peak_mb": "<float>"},
    "... every step has (step, dt, tok_sec, loss); every 10th step adds (active_mb, peak_mb) ..."
  ]
}
```

The `step_timings` array enables post-hoc throughput regression analysis without re-running training. Only compiled-phase steps are recorded (warmup steps excluded). `mx.reset_peak_memory()` is called at compiled-phase start so peak reflects only that phase. Memory fields (`active_mb`, `peak_mb`) are omitted from entries where they weren't sampled (every 10th step only) to keep the JSON compact.

### Bench output (`data/bench_*.json`)

```json
{
  "format_version": "0.1",
  "timestamp": "2026-03-08T14:39:00",
  "hardware": { "chip": "arm", "memory_gb": null, "os": "Darwin" },
  "configs": [
    {
      "label": "D=4 B=16",
      "ours": { "fwd": { "avg_ms": 205.9, "avg_tok_sec": 159154 }, "full": { "..." }, "params": 12345678, "peak_mb": 10884 },
      "ext": null
    }
  ]
}
```

Other repos may track different fields or at different granularity. The idea is to make experiment comparison tractable for those who want a data-driven approach, not to impose a standard.
