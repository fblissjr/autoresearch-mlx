"""
Autoresearch pretraining script. Single-machine, single-file, MLX on Apple Silicon.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import gc
import time
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


_norm_weight_cache = {}

def norm(x):
    key = (x.shape[-1], x.dtype)
    if key not in _norm_weight_cache:
        w = mx.ones(key[0], dtype=key[1])
        mx.eval(w)
        _norm_weight_cache[key] = w
    return mx.fast.rms_norm(x, _norm_weight_cache[key], eps=1e-5)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def make_sliding_window_mask(T, window_size):
    """Build additive causal sliding window mask."""
    row = mx.arange(T)
    col = mx.arange(T)
    diff = mx.expand_dims(row, axis=1) - mx.expand_dims(col, axis=0)
    valid = (diff >= 0) & (diff < window_size)
    return mx.where(valid, mx.array(0.0, dtype=mx.float32), mx.array(float("-inf"), dtype=mx.float32))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self._has_ve = has_ve(layer_idx, config.n_layer)
        if self._has_ve:
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)

    def __call__(self, x, ve, window_size):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2.0 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + mx.expand_dims(gate, axis=-1) * ve

        # Apply RoPE via mx.fast.rope
        q = mx.fast.rope(q, dims=self.head_dim, traditional=False, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, dims=self.head_dim, traditional=False, base=10000.0, scale=1.0, offset=0)

        # Normalize Q and K
        q = norm(q)
        k = norm(k)

        # Transpose to (B, n_heads, T, head_dim) for scaled_dot_product_attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Handle GQA: repeat KV heads if needed
        if self.n_kv_head < self.n_head:
            repeats = self.n_head // self.n_kv_head
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        # Build sliding window causal mask
        if window_size is not None and window_size < T:
            mask = make_sliding_window_mask(T, window_size)
        else:
            mask = None

        # Q and K are already normalized, so scale=1.0
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)

        # Transpose back to (B, T, n_heads, head_dim) and reshape
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = nn.relu(x) ** 2
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, window_size):
        x = x + self.attn(norm(x), ve, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones(config.n_layer)
        self.x0_lambdas = mx.zeros(config.n_layer)
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {}
        for i in range(config.n_layer):
            if has_ve(i, config.n_layer):
                self.value_embeds[str(i)] = nn.Embedding(config.vocab_size, kv_dim)

    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        # Embedding and unembedding
        self.wte.weight = mx.random.normal(shape=self.wte.weight.shape)
        self.lm_head.weight = mx.random.normal(shape=self.lm_head.weight.shape) * 0.001
        # Transformer blocks
        for block in self.layers:
            block.attn.c_q.weight = mx.random.uniform(-s, s, shape=block.attn.c_q.weight.shape)
            block.attn.c_k.weight = mx.random.uniform(-s, s, shape=block.attn.c_k.weight.shape)
            block.attn.c_v.weight = mx.random.uniform(-s, s, shape=block.attn.c_v.weight.shape)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)
            block.mlp.c_fc.weight = mx.random.uniform(-s, s, shape=block.mlp.c_fc.weight.shape)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas = mx.ones(self.config.n_layer)
        self.x0_lambdas = mx.full((self.config.n_layer,), 0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-s, s, shape=ve.weight.shape)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.layers:
            if block.attn._has_ve:
                block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight)
        # Cast embeddings to bf16
        self.wte.weight = self.wte.weight.astype(mx.bfloat16)
        for ve in self.value_embeds.values():
            ve.weight = ve.weight.astype(mx.bfloat16)

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = long_window
        return window_sizes

    def num_scaling_params(self):
        wte_n = self.wte.weight.size
        ve_n = sum(ve.weight.size for ve in self.value_embeds.values())
        lm_n = self.lm_head.weight.size
        layers_n = sum(p.size for _, p in tree_flatten(
            [block.parameters() for block in self.layers]
        ))
        scalars_n = self.resid_lambdas.size + self.x0_lambdas.size
        total = wte_n + ve_n + lm_n + layers_n + scalars_n
        return {
            'wte': wte_n, 'value_embeds': ve_n, 'lm_head': lm_n,
            'transformer_matrices': layers_n, 'scalars': scalars_n, 'total': total,
        }

    def __call__(self, idx, targets=None, reduction='mean'):
        B, T = idx.shape

        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, self.window_sizes[i])
        x = norm(x)

        logits = self.lm_head(x)
        logits = logits.astype(mx.float32)
        logits = 15.0 * mx.tanh(logits / 15.0)

        if targets is not None:
            # Cross-entropy with ignore_index=-1
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            per_token_loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
            mask = (targets_flat != -1).astype(mx.float32)
            if reduction == 'mean':
                loss = mx.sum(per_token_loss * mask) / mx.maximum(mx.sum(mask), mx.array(1.0))
                return loss
            else:
                return per_token_loss * mask
        return logits

# ---------------------------------------------------------------------------
# Optimizer: MLX built-in Muon + AdamW via MultiOptimizer
# ---------------------------------------------------------------------------
# (No custom optimizer needed -- using mlx.optimizers.Muon and AdamW)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**16  # ~65K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 32   # per-device batch size (tuned for Apple Silicon)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = 6 * num_params  # simplified estimate
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# ---------------------------------------------------------------------------
# Optimizer setup using MultiOptimizer with Muon + AdamW
# ---------------------------------------------------------------------------

# Start simple: single AdamW for all parameters to get working first
# TODO: switch to MultiOptimizer with Muon for matrix params once working
optimizer = optim.AdamW(
    learning_rate=MATRIX_LR,
    betas=list(ADAM_BETAS),
    eps=1e-10,
    weight_decay=0.0,
)
initial_lr = MATRIX_LR

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def loss_fn(model, x, y):
    return model(x, y, reduction='mean')

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    t0 = time.time()

    accumulated_grads = None
    train_loss_val = None

    for micro_step in range(grad_accum_steps):
        loss, grads = loss_and_grad_fn(model, x, y)
        train_loss_val = loss

        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = tree_map(lambda a, g: a + g, accumulated_grads, grads)

        x, y, epoch = next(train_loader)

    # Average gradients
    accumulated_grads = tree_map(lambda g: g * (1.0 / grad_accum_steps), accumulated_grads)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)

    # Update learning rate
    optimizer.learning_rate = initial_lr * lrm

    optimizer.update(model, accumulated_grads)
    mx.eval(model.parameters(), optimizer.state)

    train_loss_f = train_loss_val.item()

    # Fast fail: abort if loss is exploding
    if train_loss_f > 100:
        print("FAIL")
        exit(1)

    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up -- but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final evaluation
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
peak_mem_bytes = mx.get_peak_memory()
peak_mem_mb = peak_mem_bytes / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_memory_mb:   {peak_mem_mb:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
