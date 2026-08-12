# Shared mini-GPT used by both the DDP and FSDP labs.
# Same architecture as the Optimization Sprint, sized up so multi-GPU actually matters.
import math
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as cp


@dataclass
class GPTConfig:
    vocab: int = 50304
    seq: int = 512
    dim: int = 2048
    n_layer: int = 16
    n_head: int = 16
    use_ckpt: bool = False        # activation checkpointing (trade compute for memory)


class CausalSelfAttention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        assert c.dim % c.n_head == 0
        self.nh, self.hd = c.n_head, c.dim // c.n_head
        self.qkv = nn.Linear(c.dim, 3 * c.dim)
        self.proj = nn.Linear(c.dim, c.dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # FlashAttention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.dim), nn.LayerNorm(c.dim)
        self.attn = CausalSelfAttention(c)
        self.mlp = nn.Sequential(nn.Linear(c.dim, 4 * c.dim), nn.GELU(), nn.Linear(4 * c.dim, c.dim))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.dim)
        self.pos = nn.Embedding(c.seq, c.dim)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.lnf = nn.LayerNorm(c.dim)
        self.head = nn.Linear(c.dim, c.vocab, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        for b in self.blocks:
            if self.c.use_ckpt and self.training:
                x = cp.checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.c.vocab), targets.reshape(-1))
        return logits, loss


def count_params(c: GPTConfig) -> int:
    per_layer = 12 * c.dim * c.dim                       # attn(4·dim²) + mlp(8·dim²)
    embeds = 2 * c.vocab * c.dim + c.seq * c.dim         # token + head + positions
    return per_layer * c.n_layer + embeds
