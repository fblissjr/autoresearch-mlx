# ANE (Apple Neural Engine) Integration Roadmap

last updated: 2026-03-15

Multi-session engineering effort to evaluate and potentially integrate the M2 Ultra's ANE for training acceleration. This is separate from the experiment loop -- it's infrastructure work.

## Motivation

The M2 Ultra has a 32-core Neural Engine rated at ~31 TFLOPS (FP16). Currently all training runs on the GPU cores only. If the ANE can handle even a subset of operations (e.g., forward pass, attention), it could free GPU cycles for backward pass + optimizer, improving overall throughput.

However, the ANE has significant constraints (FP16 only, limited programmability, high compile cost) that may make it impractical. This plan gates on a go/no-go benchmark.

## Phase 0: M2 Ultra ANE Benchmarking (go/no-go gate)

**Goal**: Determine whether ANE throughput justifies the engineering cost.

- Build and run `coderef/ANE/inmem_bench.m` on our M2 Ultra hardware
- Measure: peak TFLOPS, dispatch latency overhead, SRAM behavior
- Compare against current GPU throughput: ~140K tok/sec at 12.5M params (DEPTH=4)
- **Decision gate**: if effective ANE throughput for matrix operations is lower than what we get from the GPU, stop here. The engineering cost of the bridge, MIL compilation, and FP16 constraints isn't justified.

**Expected outcome**: The ANE's 31 TFLOPS (FP16) is competitive with the M2 Ultra GPU's ~27 TFLOPS (FP32), but dispatch latency and limited op support may reduce effective throughput significantly. The benchmark will tell us.

## Phase 1: Bridge Validation

**Goal**: Prove Python can talk to the ANE at all.

- Write a Python ctypes wrapper around `coderef/ANE/bridge/libane_bridge.dylib`
- Minimal test: init ANE context, compile a trivial MIL kernel (e.g., matmul), execute, read output back
- Verify numeric correctness against MLX for the same operation
- Measure round-trip latency (Python -> ctypes -> ANE -> back)

**Key risk**: The bridge library may not expose enough functionality, or the compile step may be too slow for iterative use.

## Phase 2: Single-Layer Forward Pass

**Goal**: Run one transformer layer on the ANE and compare against MLX.

- Port one transformer layer to MIL (Model Intermediate Language)
- Include: linear projections, RoPE (may need CPU fallback), attention, FFN
- Numeric comparison: forward pass output should match MLX to within FP16 precision
- Latency comparison: ANE layer time vs GPU layer time

**Constraints to handle**:
- FP16 only -- our training uses bf16. Need to verify numeric stability.
- Channel-first layout [1, C, 1, S] -- different from MLX's row-major. Transpose overhead matters.
- SDPA causal masking may not be available on ANE -- may need to fall back to explicit mask or CPU.

## Phase 3: Forward + Backward

**Goal**: Full differentiable forward+backward through ANE for one layer.

- Backward pass through ANE-compiled operations
- This may require custom gradient implementations since ANE MIL may not support automatic differentiation
- Alternative: forward on ANE, backward on GPU (split execution)
- Memory layout: unified memory means no copies, but IOSurface interaction with MLX arrays needs investigation

## Phase 4: Training Loop Integration

**Goal**: Hybrid ANE+GPU training loop.

- Decide which operations go to ANE vs GPU based on Phase 2-3 results
- Likely candidates for ANE: forward pass linear projections (bulk of FLOPs)
- Likely stays on GPU: attention (causal masking), backward pass, optimizer
- Integration point: replace specific MLX ops with ANE calls while keeping the rest of the training loop unchanged

## Open Questions

- **M2 Ultra dual-engine exposure**: The M2 Ultra has two ANE engines (one per die). Does the ANE API expose both, or does it abstract them as one? Can we dispatch to both in parallel?
- **Compile limit**: References mention ~119 operation compile limit per ANE program. What's the scope -- per layer? Per model? Can we work around it with multiple programs?
- **Unified memory + IOSurface**: Can we share memory between MLX arrays and ANE buffers without copies? IOSurface is the Apple mechanism for zero-copy GPU/ANE sharing, but MLX may not use it.
- **mx.compile interaction**: If parts of the graph run on ANE, does this break mx.compile's fusion? We may need to treat ANE calls as opaque operations in the compile graph.
- **Thermal throttling**: Running both GPU and ANE at full throughput may hit thermal limits on sustained workloads. The 5-minute training budget is long enough for this to matter.

## References

- `coderef/ANE/` -- local ANE benchmarking and bridge code
- `coderef/ANE/inmem_bench.m` -- Objective-C ANE benchmark
- `coderef/ANE/bridge/` -- C bridge library for Python access
