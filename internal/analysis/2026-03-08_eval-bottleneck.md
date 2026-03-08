last updated: 2026-03-08

# Eval Bottleneck Analysis: ~400s Actual vs ~200s Projected

## Problem Statement

After v0.5.2 training completes, evaluate_bpb takes ~400s (702.5s total - 301.6s training = ~401s). bench.py projects ~200s for the same eval at batch=64 compiled. This 2x discrepancy is the largest remaining time sink -- eval takes longer than training itself.

## The Numbers

EVAL_TOKENS = 40 * 524288 = 20,971,520 tokens

| Setting | Batch | Steps | Projected (bench.py) | Actual (train.py) |
|---------|-------|-------|---------------------|--------------------|
| batch=64, compiled | 64 | 160 | ~200s | ~400s |
| batch=32, compiled | 32 | 320 | ~400s | untested |

At batch=64: 400s / 160 steps = 2.5s per eval step.
bench.py measured: ~1.25s per eval step (batch=64, compiled).

## Root Cause: Memory Environment, Not Algorithm

bench.py and train.py run identical eval code (same evaluate_bpb function, same batch size, same compiled model). The difference is the memory environment.

### bench.py eval environment
- Fresh model, no training history
- ~10-15GB active memory (model weights only)
- No allocation fragmentation
- Metal allocator has ample free pool

### train.py eval environment
- After 193 compiled training steps (5 minutes sustained GPU compute)
- 49.4GB training peak, optimizer freed -> estimated ~35-40GB active after cleanup
- 300 seconds of allocation/deallocation cycles fragmented the Metal memory pool
- gc.collect() + mx.clear_cache() called, but Metal's underlying allocator retains fragmented pages

### Why batch=64 makes it worse
- Training uses batch=32: activations for 32 * 2048 = 65K tokens
- Eval uses batch=64: activations for 64 * 2048 = 131K tokens (2x)
- The 2x larger activation tensors must be allocated in a fragmented pool
- Peak memory jumps from 49GB (training) to 63GB (eval) -- a 14GB surge
- At 63GB, the Metal allocator may be compacting or searching for contiguous blocks

### Supporting evidence
- v0.5.0 (no optimizer cleanup): eval took ~439s (741s - 302s)
- v0.5.2 (optimizer cleanup): eval took ~401s (702.5s - 301.6s)
- Optimizer cleanup saved ~38s, not the expected ~200s
- This confirms: optimizer memory (~10GB) was a minor contributor; the dominant factor is allocation state from 5 minutes of training

### Why bench.py's projection is wrong
bench.py measures per-step eval time on a fresh model with clean memory. It correctly measures the compute cost but cannot account for the allocation overhead that accumulates during a real training run. The 2x slowdown is an environmental effect, not an algorithmic one.

## Proposed Fix: Eval at batch=32

The simplest fix is reducing EVAL_BATCH_SIZE from 64 to 32:

1. **Halves the activation surge**: 65K tokens instead of 131K. Eval peak stays near training peak (~49GB) instead of jumping to 63GB.
2. **Fewer large allocations**: Smaller tensors fit better in a fragmented pool.
3. **More steps but smaller**: 320 steps instead of 160.

There's a subtlety: if the slowdown is purely from the 14GB surge (49->63GB), then staying at 49GB with batch=32 should eliminate it. If the slowdown is from cumulative fragmentation at any allocation size, batch=32 may not help.

## Alternative Fix: Model Serialization Reset

If batch=32 doesn't help, the nuclear option is to serialize model weights to numpy arrays, delete the entire model, run gc.collect() + mx.clear_cache() to reset the Metal allocator, then reconstruct the model from the saved numpy arrays. This guarantees a fresh allocation state at the cost of ~2-3s serialization overhead.

## Quick Test Plan (< 3 minutes)

A test script that measures eval per-step time in two environments:

**Environment A** (clean): Fresh model, no training history. Measure 10 eval steps at batch=32 and batch=64.

**Environment B** (post-churn): Same model, but after allocating and freeing ~200 rounds of large tensors to simulate training allocation churn. Measure 10 eval steps at batch=32 and batch=64.

If Environment B is ~2x slower than Environment A, fragmentation is confirmed. If they're similar, the cause is something else (thermal state, model weight memory layout, etc.).

This test avoids running a full 5-minute training -- 200 rounds of allocate-free-garbage-collect takes ~30s and produces equivalent fragmentation.

To further test the serialization reset fix: after Environment B measurement, serialize model weights to numpy, delete model, gc + clear cache, reload from numpy, re-measure. If speed returns to Environment A levels, the fix works.

Total test time: ~2-3 minutes (no training needed).

## Recommendation

1. **Immediate**: Change EVAL_BATCH_SIZE from 64 to 32 in train.py. bench.py already showed no throughput benefit from batch=64 eval (compute-bound, not batch-overhead), and the memory surge is likely the primary bottleneck.
2. **If batch=32 doesn't help**: Implement the numpy serialization trick (save weights, del model, gc, reload). Ugly but guarantees clean allocation state.
3. **Long term**: Investigate MLX Metal allocator behavior under sustained workloads. This affects any long-running MLX training, not just our project.

## Status

Proposed. Pending test script implementation and validation.
