"""amu: a miniature GPT-style language model, written from scratch.

No pretrained weights, no external API. The character tokenizer lives in
train.py / generate.py; this file is only the neural network.

The stack is deliberately the modern one, minus everything that needs scale:
pre-norm RMSNorm, rotary positions (RoPE), grouped-query attention, a SwiGLU
feed-forward, tied input/output embeddings, and a KV cache for decoding. Each
piece is a few lines; none of it is imported from a model zoo. See the README
for which ideas came from Ling-3.1-flash and which ones do not transfer to a
model this small.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

DEFAULT_ROPE_BASE = 10_000.0


class RMSNorm(nn.Module):
    """Layer norm without mean subtraction or bias.

    Used by most current LLMs: one weight vector, no shift, and the scaling is
    invariant to a change of sign, which is all a pre-norm block needs.

    `mean` here averages the per-token residual stream over its channels, so it
    is O(variance), not the usual per-feature batch statistic.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


def rope_tables(head_dim, block_size, base=DEFAULT_ROPE_BASE):
    """cos/sin angles for every position, shaped (block_size, head_dim).

    Position is a rotation, not a learned vector: no position parameters, and
    the model can in principle run past the length it was trained on.
    """
    if head_dim % 2:
        raise ValueError("head_dim must be even for rotary positions")

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(block_size).float()
    angles = torch.outer(positions, inv_freq)          # (block_size, head_dim / 2)
    doubled = torch.cat([angles, angles], dim=-1)      # (block_size, head_dim)
    return torch.cos(doubled), torch.sin(doubled)


def apply_rope(x, cos, sin):
    """Rotate each pair of a (B, heads, T, head_dim) tensor by its position."""
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    rotated = torch.cat([-second, first], dim=-1)
    return x * cos + rotated * sin


class SelfAttention(nn.Module):
    """Causal attention with grouped key/value heads and rotary positions.

    With `kv_heads < heads`, several query heads share one key/value pair, so
    the cache and the key/value projections shrink without losing query
    capacity - the same trade a large model makes with grouped-query attention.
    """

    def __init__(self, embed_dim, heads, block_size, kv_heads=None,
                 rope_base=DEFAULT_ROPE_BASE):
        super().__init__()

        assert embed_dim % heads == 0

        kv_heads = kv_heads or heads
        assert heads % kv_heads == 0, "heads must be a multiple of kv_heads"

        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = embed_dim // heads
        self.group = heads // kv_heads

        self.query = nn.Linear(embed_dim, heads * self.head_dim, bias=False)
        self.key = nn.Linear(embed_dim, kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(embed_dim, kv_heads * self.head_dim, bias=False)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

        cos, sin = rope_tables(self.head_dim, block_size, rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _split_heads(self, x, count):
        B, T, _ = x.shape
        return x.view(B, T, count, self.head_dim).transpose(1, 2)

    def forward(self, x, past=None, pos_offset=0):
        B, T, _ = x.shape

        query = self._split_heads(self.query(x), self.heads)
        key = self._split_heads(self.key(x), self.kv_heads)
        value = self._split_heads(self.value(x), self.kv_heads)

        cos = self.cos[pos_offset:pos_offset + T]
        sin = self.sin[pos_offset:pos_offset + T]
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)

        if past is not None:
            key = torch.cat([past[0], key], dim=2)
            value = torch.cat([past[1], value], dim=2)

        cache = (key, value)

        if self.group > 1:
            key = key.repeat_interleave(self.group, dim=1)
            value = value.repeat_interleave(self.group, dim=1)

        # A cached step has one query and only past keys, so no mask is needed.
        output = F.scaled_dot_product_attention(
            query, key, value, is_causal=past is None
        )

        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.projection(output), cache


class SwiGLU(nn.Module):
    """Gated feed-forward: two projections, one gated by SiLU.

    The hidden width is ~8/3 of the embedding so the parameters stay close to
    the classic 4x GELU network it replaces.
    """

    def __init__(self, embed_dim, hidden=None):
        super().__init__()

        hidden = hidden or ((int(8 * embed_dim / 3) + 7) // 8) * 8

        self.w1 = nn.Linear(embed_dim, hidden, bias=False)
        self.gate = nn.Linear(embed_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, embed_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.gate(x)) * self.w1(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, block_size, kv_heads=None,
                 ffn_hidden=None, rope_base=DEFAULT_ROPE_BASE):
        super().__init__()

        self.norm1 = RMSNorm(embed_dim)
        self.attention = SelfAttention(
            embed_dim, heads, block_size, kv_heads=kv_heads, rope_base=rope_base
        )
        self.norm2 = RMSNorm(embed_dim)
        self.feed_forward = SwiGLU(embed_dim, hidden=ffn_hidden)

    def forward(self, x, past=None, pos_offset=0):
        attention, cache = self.attention(self.norm1(x), past, pos_offset)
        x = x + attention
        x = x + self.feed_forward(self.norm2(x))
        return x, cache


class Amu(nn.Module):
    """The whole network: embeddings -> N blocks -> norm -> vocabulary head."""

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        heads=4,
        layers=4,
        block_size=128,
        kv_heads=None,
        ffn_hidden=None,
        rope_base=DEFAULT_ROPE_BASE,
        tie_weights=True,
    ):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                heads,
                block_size,
                kv_heads=kv_heads,
                ffn_hidden=ffn_hidden,
                rope_base=rope_base,
            )
            for _ in range(layers)
        ])

        self.final_norm = RMSNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)

        if tie_weights:
            # One matrix read as input embeddings and as the output head: fewer
            # parameters, and a shared notion of what each token means.
            self.output_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # Scale residual projections down so deep-ish stacks start quiet.
        residual_std = 0.02 / math.sqrt(2 * layers)
        for name, param in self.named_parameters():
            if name.endswith("projection.weight") or name.endswith("w2.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _run(self, idx, past=None):
        """Logits for `idx`, plus the key/value cache to continue from."""
        B, T = idx.shape
        offset = 0 if past is None else past[0][0].shape[2]

        if offset + T > self.block_size:
            raise ValueError(
                "context of %d tokens exceeds block_size %d"
                % (offset + T, self.block_size)
            )

        x = self.token_embedding(idx)

        new_past = []
        for index, block in enumerate(self.blocks):
            layer_past = None if past is None else past[index]
            x, cache = block(x, layer_past, offset)
            new_past.append(cache)

        return self.output_head(self.final_norm(x)), new_past

    def forward(self, idx, targets=None):
        logits, _ = self._run(idx)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C), targets.reshape(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.8, top_k=None):
        """Sample tokens one at a time, reusing the key/value cache."""
        self.eval()

        idx = idx[:, -self.block_size:]
        logits, past = self._run(idx)

        for _ in range(max_new_tokens):
            step = logits[:, -1, :] / max(temperature, 0.01)

            if top_k:
                kth = torch.topk(step, min(top_k, step.size(-1))).values[:, -1:]
                step = step.masked_fill(step < kth, float("-inf"))

            probabilities = F.softmax(step, dim=-1)
            token = torch.multinomial(probabilities, num_samples=1)

            idx = torch.cat((idx, token), dim=1)

            if past[0][0].shape[2] >= self.block_size:
                # The cache is full. Keep only the newest half so the next
                # steps are cached again, instead of re-running a full window
                # on every single character.
                window = idx[:, -(self.block_size // 2):]
                logits, past = self._run(window)
            else:
                logits, past = self._run(token, past)

        return idx
