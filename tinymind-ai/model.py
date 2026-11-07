"""amu: a miniature GPT-style language model, written from scratch.

No pretrained weights, no external API. Character-level tokenizer lives in
train.py / generate.py; this file is only the neural network.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, block_size):
        super().__init__()

        assert embed_dim % heads == 0

        self.heads = heads
        self.head_dim = embed_dim // heads

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.projection = nn.Linear(embed_dim, embed_dim)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * (
            self.head_dim ** -0.5
        )

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)

        output = weights @ v

        output = output.transpose(1, 2).contiguous()
        output = output.view(B, T, C)

        return self.projection(output)


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, block_size):
        super().__init__()

        self.attention = SelfAttention(
            embed_dim, heads, block_size
        )

        self.feed_forward = FeedForward(embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class Amu(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        heads=4,
        layers=4,
        block_size=128
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size, embed_dim
        )

        self.position_embedding = nn.Embedding(
            block_size, embed_dim
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, heads, block_size)
                for _ in range(layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

        self.output_head = nn.Linear(
            embed_dim, vocab_size
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding(idx)

        positions = torch.arange(
            T, device=idx.device
        )

        position_emb = self.position_embedding(positions)

        x = token_emb + position_emb

        x = self.blocks(x)

        x = self.final_norm(x)

        logits = self.output_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits.view(B, T, -1), loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.8):
        self.eval()

        for _ in range(max_new_tokens):
            idx_crop = idx[:, -self.block_size:]

            logits, _ = self(idx_crop)

            logits = logits[:, -1, :] / max(temperature, 0.01)

            probabilities = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probabilities, num_samples=1
            )

            idx = torch.cat((idx, next_token), dim=1)

        return idx
