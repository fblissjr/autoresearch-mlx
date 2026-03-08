last updated: 2026-03-08

# prepare.py Conformance Analysis: PyTorch/CUDA Reference vs MLX Port

## Purpose

Verify that our MLX port of prepare.py preserves the evaluation semantics of the original PyTorch/CUDA reference (coderef/autoresearch/prepare.py). This matters because evaluate_bpb is the ground truth metric -- any deviation in its behavior would invalidate all our training results.

## Methodology

Line-by-line comparison of every function and constant in both files. Categorized each difference as:
- **Semantic**: Changes the behavior or output of the function
- **Mechanical**: Same behavior, different API (torch -> mlx/numpy)
- **Identical**: No change at all

## Constants (Identical)

All fixed constants match exactly:

| Constant | Reference | Ours | Status |
|----------|-----------|------|--------|
| MAX_SEQ_LEN | 2048 | 2048 | Identical |
| TIME_BUDGET | 300 | 300 | Identical |
| EVAL_TOKENS | 40 * 524288 = 20,971,520 | 40 * 524288 = 20,971,520 | Identical |
| CACHE_DIR | ~/.cache/autoresearch/ | ~/.cache/autoresearch/ | Identical |
| MAX_SHARD | 6542 | 6542 | Identical |
| VAL_SHARD | 6542 (pinned) | 6542 (pinned) | Identical |
| VOCAB_SIZE | 8192 | 8192 | Identical |
| SPLIT_PATTERN | GPT-4 style BPE | GPT-4 style BPE | Identical |
| SPECIAL_TOKENS | 4 reserved tokens | 4 reserved tokens | Identical |
| BOS_TOKEN | reserved_0 | reserved_0 | Identical |

## Data Download (Identical)

download_single_shard() and download_data() are character-for-character identical. No torch dependency in these functions.

## Tokenizer Training (Mechanical)

| Component | Reference | Ours | Impact |
|-----------|-----------|------|--------|
| BPE training | rustbpe + tiktoken | rustbpe + tiktoken | Identical |
| Tokenizer save | serialized to tokenizer.pkl | serialized to tokenizer.pkl | Identical |
| token_bytes save | torch.save to token_bytes.pt (int32) | np.save to token_bytes.npy (int32) | Mechanical |
| token_bytes values | len(token_str.encode("utf-8")) per token, 0 for special | Same | Identical |

The token_bytes array contains the same integer values -- only the serialization format differs (.pt vs .npy). Both are int32. The values are deterministic (UTF-8 byte lengths of each vocabulary token), so the arrays are bit-identical regardless of framework.

**Note**: Because the file extension differs, the two prepare.py files are not interchangeable at runtime -- our MLX port loads .npy, the reference loads .pt. This is expected and correct.

## Runtime Utilities

### Tokenizer class (Identical)

The Tokenizer class is framework-agnostic (uses only serialization + tiktoken). Character-for-character identical.

### get_token_bytes() (Mechanical)

| | Reference | Ours |
|-|-----------|------|
| Load | torch.load("token_bytes.pt", map_location=device) | np.load("token_bytes.npy") then mx.array(..., dtype=mx.int32) |
| Return type | torch.Tensor (int32, on specified device) | mx.array (int32) |
| Values | Same integer array | Same integer array |

The reference accepts a device parameter (defaults to "cpu", called with "cuda" from evaluate_bpb). Our version has no device parameter -- MLX uses unified memory, so there's no device transfer.

### _document_batches() (Identical)

Framework-agnostic Python generator. Character-for-character identical.

### make_dataloader() (Mechanical)

This is the largest divergence, but entirely mechanical:

| Component | Reference | Ours |
|-----------|-----------|------|
| Row buffer | torch.empty((B, T+1), dtype=torch.long) | np.empty((B, T+1), dtype=np.int32) |
| Token insertion | torch.tensor(doc, dtype=torch.long) | Direct numpy slice assignment |
| Output pipeline | row_buffer -> cpu_buffer (pinned) -> gpu_buffer (CUDA) -> yield views | row_buffer -> mx.array(..., dtype=mx.int32) -> yield |
| Pre-allocation | 3 buffers: row, cpu (pinned), gpu | 1 buffer: row (numpy) |

**Packing algorithm**: Identical best-fit BOS-aligned packing. Same buffer_size=1000, same best-fit search, same shortest-crop fallback.

**Output equivalence**: Both yield (inputs, targets, epoch) where inputs = row_buffer[:, :-1] and targets = row_buffer[:, 1:]. The reference yields CUDA tensors (int64), ours yields MLX arrays (int32). Token IDs are small integers (<8192), so int32 is sufficient.

**dtype difference**: Reference uses torch.long (int64) throughout. Ours uses int32. This is safe because:
- Vocabulary size is 8192 (fits in int16, let alone int32)
- Token IDs are used only for indexing (embedding lookup, cross-entropy targets)
- MLX's cross_entropy and embedding ops work correctly with int32

### evaluate_bpb() (Mechanical -- Critical Path)

This is the most important function. Granular line-by-line comparison:

| Line | Reference | Ours | Why |
|------|-----------|------|-----|
| Decorator | @torch.no_grad() | None | MLX doesn't track gradients by default. Gradients are only computed when explicitly requested via nn.value_and_grad. No-op equivalent. |
| token_bytes device | get_token_bytes("cuda") | get_token_bytes() | MLX unified memory -- no device parameter needed. |
| reshape | .view(-1) | .reshape(-1) | MLX equivalent of torch's .view(). Both produce a contiguous 1D view. |
| mask type | nbytes > 0 (bool tensor) | (nbytes > 0).astype(mx.float32) | PyTorch auto-promotes bool * float32 -> float32. MLX requires explicit cast. Numerically identical: True->1.0, False->0.0. |
| sum pattern | (loss_flat * mask).sum().item() | mx.sum(loss_flat * mask).item() | Functional vs method syntax. Same operation. |
| accumulation | Inline in += | Via intermediate variable | Identical result. |

**Numerical equivalence**: The computation is:
1. Per-token cross-entropy in nats (float32)
2. Masked sum (exclude special tokens where byte_length=0)
3. Total byte count
4. nats / (ln(2) * bytes) = bits per byte

Both implementations compute the same masked sums in float32, accumulate into Python float64 (total_nats), and divide by the same denominator. The only possible numerical difference is float32 reduction order, which is determined by the framework's sum implementation -- but both sum the same elements, so any difference would be at the float32 epsilon level (~1e-7 relative), far below the precision we report (6 decimal places of BPB).

## Additional Deviations (Non-Eval)

### Default num_shards

Both files default to --num-shards 10. However, we manually ran prepare.py --num-shards 20 to download 20 shards for data diversity. This affects training data variety but NOT evaluation, since eval always uses the single pinned VAL_SHARD (shard_06542).

### Import differences

| Reference | Ours |
|-----------|------|
| import torch | import numpy as np + import mlx.core as mx |

No overlapping functionality -- torch was used for tensor ops and serialization, numpy + mlx serve the same roles.

## Conclusion

Our prepare.py is a faithful mechanical port from PyTorch/CUDA to MLX. Every behavioral difference is a direct API translation (torch.Tensor -> mx.array, .view -> .reshape, CUDA device management -> unified memory). The evaluation function evaluate_bpb computes the same metric with the same precision. No semantic deviations exist.

The evaluation results produced by our port are directly comparable to the reference implementation's results.
