last updated: 2026-03-08

# Throughput Regression Analysis: 39K -> 21K tok/sec at Step 104

## Problem Statement

During v0.5.0 training (5-group MultiOptimizer), steady-state throughput of ~39K tok/sec drops to ~21K tok/sec at approximately step 104. This 1.82x slowdown persists for the remainder of training, reducing effective training steps from an estimated 189 to 154 (18% fewer steps in the same 5-minute budget).

The regression was not observed in v0.3.0 (single AdamW, ~46K tok/sec sustained) or in bench_compare.py runs (12 steps at steady 39K tok/sec with 5-group optimizer).

## Timeline

| Phase | Steps | tok/sec | Notes |
|-------|-------|---------|-------|
| Warmup (uncompiled) | 0-10 | ~15K | Expected: graph compilation overhead |
| Compiled, steady | 11-103 | ~39K | Normal compiled throughput |
| Compiled, degraded | 104-154 | ~21K | 1.82x slowdown, sustained |

Key timestamps:
- Warmdown schedule starts at step ~94 (10 steps before regression)
- Regression onset at step ~104 (sustained, not transient)
- Brief partial recovery observed around steps 142-144, then regression returns

## Per-Step Data

No per-step timing was collected in the v0.5.0 run (this analysis motivated adding `step_timings` to the JSON output format). The timeline above is reconstructed from training log output.

Future runs will include `step_timings` array in `data/run_*.json` for precise diagnosis.

## Correlation Analysis

### Schedule transition
- The warmdown phase begins at `warmdown_start = int((1.0 - 0.5) * 189) = 94`
- LR schedules were swapped onto existing optimizer instances via `opt._schedulers['learning_rate']` before the compiled graph was defined
- The compiled graph (`compiled_step`) includes the `join_schedules` logic, so the schedule boundary at step 94 should be handled by data-dependent branching (`mx.where`), not graph recompilation
- The 10-step gap between warmdown start (94) and regression onset (104) weakens a direct causal link but doesn't eliminate it

### Memory profile
- v0.3.0 (2-group): 53GB peak, no regression
- v0.5.0 (5-group): 63GB peak, regression at step 104
- Delta: +10GB from 3 additional AdamW optimizer states (momentum + variance per parameter)
- System: M2 Ultra with 192GB unified memory -- no swap should occur at 63GB

### Eval performance
- Eval took 440s (vs ~205s expected from bench.py profiling)
- This 2x slowdown during eval (which doesn't use the optimizer) suggests memory pressure is the root cause, not optimizer computation overhead

## Hypotheses (Ranked by Likelihood)

### H1: Memory pressure / MLX allocation fragmentation (HIGH)

**Theory**: At 63GB, MLX's memory pool may fragment over ~100 steps of allocation/deallocation cycles. The 5-group optimizer creates more intermediate tensors per step (5 separate update passes with momentum/variance buffers). After ~100 steps, allocation patterns cause the Metal memory allocator to slow down or trigger compaction.

**Evidence for**:
- 10GB higher peak memory than non-regressing 2-group config
- Eval is also 2x slower (440s vs 205s), despite not using the optimizer -- points to system-wide memory pressure
- Bench runs (12 steps) never hit the regression -- too short to trigger fragmentation

**Evidence against**:
- 63GB is only 33% of 192GB available -- should have ample headroom
- Brief recovery at steps 142-144 would be unusual for monotonic fragmentation

**Diagnostic**: Add `mx.get_peak_memory()` sampling every 10 steps. If memory grows beyond 63GB during training, fragmentation or leak is confirmed.

### H2: Thermal throttling (MEDIUM)

**Theory**: M2 Ultra sustained GPU load for ~2.5 minutes (steps 11-103 at ~1.7s/step) before regression. Apple Silicon throttles under sustained thermal load, reducing GPU clock speeds.

**Evidence for**:
- Sustained slowdown (not spiky) matches thermal throttling profile
- Onset at ~2.5 minutes of sustained load is plausible for Apple Silicon thermal envelope
- Brief recovery at steps 142-144 could be thermal oscillation

**Evidence against**:
- bench_compare.py D=8 B=32 runs 12 full steps at steady 39K without regression (but total runtime is only ~20s, well within thermal budget)
- v0.3.0 runs the full 5 minutes without regression (but at lower memory pressure -- thermal and memory may compound)

**Diagnostic**: Monitor CPU/GPU temperature during training via `sudo powermetrics --samplers gpu_power -i 1000`. If GPU temp exceeds ~100C at step 104, thermal throttling is confirmed.

### H3: MLX compiled graph recompilation (LOW)

**Theory**: When `join_schedules` crosses the warmdown boundary, MLX might recompile the graph.

**Evidence against**:
- `join_schedules` uses `mx.where` for data-dependent branching -- this doesn't change graph shape
- MLX's `mx.compile` traces the computation graph structure, not data values
- The 10-step delay between boundary crossing (step 94) and regression onset (step 104) argues against a one-time recompilation spike
- A recompilation would cause a brief spike, not sustained 2x slowdown

**Diagnostic**: Run with fixed LR (no schedule) to isolate schedule as cause.

### H4: Lazy optimizer state initialization (LOW)

**Theory**: Some optimizer groups only allocate momentum/variance buffers when parameters are first updated. If a group's parameters don't receive gradients until a specific step, the sudden allocation could increase memory pressure.

**Evidence against**:
- All 5 optimizer groups receive gradients from step 0 (the filter functions are evaluated at construction time, and all parameter groups have non-zero gradients from the start)
- Warmup phase (steps 0-10) uses the same optimizer instance, so all states are initialized before compiled phase

## Proposed Diagnostic Experiments

1. **2-group baseline**: Run with only Muon + AdamW (merge groups 3-5 into fallback). If regression disappears, the extra optimizer states are the cause.

2. **Memory sampling**: Add `mx.get_peak_memory()` every 10 steps to the training loop. Plot memory growth over time.

3. **Free optimizer before eval**: Add `del optimizer; gc.collect()` before `evaluate_bpb()`. If eval time drops from 440s to ~205s, memory pressure is confirmed.

4. **Fixed LR run**: Remove all schedule logic. If regression still occurs, schedules are eliminated as a cause.

5. **Temperature monitoring**: Run with `powermetrics` in parallel to capture GPU temperature at regression onset.

## Proposed Fixes (After Diagnosis)

### If memory pressure (H1):
- Free optimizer state before eval: `del optimizer; gc.collect()`
- Reduce optimizer groups from 5 to 3: merge resid_lambdas (16 params) into fallback group
- Explicitly call `mx.metal.clear_cache()` periodically during training

### If thermal throttling (H2):
- Insert `time.sleep(0.05)` every 20 steps (trade 50ms latency for thermal recovery)
- Reduce batch size to lower sustained power draw (but this reduces tokens/step)
- Accept the regression and budget for it in step estimation

### If both (compound effect):
- Reduce to 3 optimizer groups AND add periodic thermal cooldown
- This is the most likely scenario: higher memory pressure raises power draw, which triggers thermal throttling earlier
