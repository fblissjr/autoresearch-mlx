last updated: 2026-03-08

# Data Investigations Backlog

The baseline autoresearch approach is algorithm-driven: optimize the model architecture, optimizer, and training procedure. Data quality is an underexplored lever that could yield significant val_bpb improvements, especially under a fixed 5-minute time budget where every token must count.

## Active

### 1. FineWeb quality filtering
**Status**: Investigating
**Hypothesis**: FineWeb-Edu provides quality scores per document. Filtering to higher-quality documents should improve val_bpb at same compute budget, since low-quality web text wastes training tokens.
**Approach**: Check if FineWeb-Edu scores are available in our data pipeline, add a quality threshold parameter to prepare.py, compare val_bpb with/without filtering.
**Risk**: Filtering reduces data volume. With a 5-minute budget we only see each token once anyway, so reducing volume may not matter. But if the val set is unfiltered, training on filtered data could hurt.

## Backlog

### 2. Curriculum learning (data ordering)
**Hypothesis**: Ordering training data by difficulty (shorter/simpler first, harder later) could improve convergence speed. The fixed 5-minute budget makes this especially interesting -- early convergence means more useful training at the end.
**Approach**: Sort documents by perplexity (from a reference model), length, or vocabulary complexity. Train with ordered vs shuffled data.
**Considerations**: The dataloader currently does best-fit packing with random order. Changing order means modifying prepare.py (READ ONLY constraint -- would need to discuss).

### 3. Token weighting (non-uniform loss)
**Hypothesis**: Not all tokens are equally informative. Weighting loss by token informativeness could improve BPB on the evaluation set.
**Approach**: Weight loss by token byte count (already available), inverse frequency, or a learned quality signal. The `token_bytes` mask already skips some tokens; extend this to soft weighting.
**Considerations**: Modifies the loss function in train.py. Must ensure the evaluation metric (val_bpb) is unchanged.

### 4. Data mixing (domain ratios)
**Hypothesis**: FineWeb is web crawl data with uncontrolled domain distribution. Intentionally mixing domains (code, wiki, books, web) at tuned ratios could improve generalization.
**Approach**: Categorize FineWeb documents by domain, experiment with mixing ratios.
**Considerations**: Requires domain classification of training data. May need multiple data shards.

### 5. Tokenizer optimization
**Hypothesis**: The 8K BPE vocabulary is small by design, but vocabulary composition affects BPB. A vocabulary optimized for the training distribution could improve encoding efficiency.
**Approach**: Compare current tokenizer against alternatives (different vocab sizes, different BPE training data). Measure both encoding efficiency and downstream val_bpb.
**Considerations**: Tokenizer is in prepare.py (READ ONLY). Would need careful coordination.

### 6. Deduplication
**Hypothesis**: Near-duplicate documents waste training tokens. Dedup at document or paragraph level could improve effective data diversity per training step.
**Approach**: MinHash or exact-match dedup on the training data. Compare val_bpb with deduped vs original.
**Considerations**: FineWeb already has some dedup applied. Additional dedup may have diminishing returns.

## Proposed: Common Benchmark Output Format

For cross-repo/cross-hardware comparison, we write structured JSON to `data/`. This is one possible format -- not a standard, just a starting point for data-driven comparison across experiments and forks.

```json
{
  "format_version": "0.1",
  "timestamp": "2026-03-08T14:39:00",
  "hardware": {
    "chip": "M2 Ultra",
    "memory_gb": 192,
    "os": "macOS"
  },
  "model": {
    "depth": 8,
    "n_embd": 512,
    "params": 50332176,
    "vocab_size": 8192
  },
  "training": {
    "budget_seconds": 300,
    "actual_seconds": 300.0,
    "total_steps": 189,
    "total_tokens": 12386304,
    "avg_tok_sec": 41287,
    "optimizer_groups": 5,
    "compiled": true,
    "batch_size": 32
  },
  "result": {
    "val_bpb": 2.28
  },
  "data": {
    "source": "fineweb",
    "filtering": "none",
    "tokenizer": "bpe-8k"
  }
}
```

Other repos may track different fields or at different granularity. The idea is to make experiment comparison tractable for those who want a data-driven approach, not to impose a standard.
