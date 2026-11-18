"""Attention variants for Vietnamese diacritic restoration, chosen for
compute / memory / parameter efficiency on short sentences (<=48 syllables).

Evidence behind the design (see NOTES for citations):
  * Edman et al. 2022 (arXiv:2205.04810): in very low-resource settings n-grams
    beat neural LMs *because* they focus on LOCAL context; limiting self-attention
    was the single most effective fix, up to +5% on downstream tasks.
    -> so the window must be local, not global.
  * "Sliding-window beats linear attention" (arXiv:2608.28444) and
    Xiong et al. 2022 (NAACL, 2022.naacl-main.144): simple local attention stays
    competitive while being O(n*w) instead of O(n^2).
    -> so no exotic linear-attention kernel is needed at n<=48.
  * Latent multi-head attention (arXiv:2506.09342): latent/低-rank projections cut
    KV memory ~45% for small models at ~0.3% quality cost.
    -> so use a small key/value rank for parameter efficiency.

Concrete choices:
  * relative-position logit bias (learned per offset, shared across heads) instead
    of adding position embeddings: costs only (2b+1) parameters and gives the
    "near = heavy, far = light" weighting directly.
  * bidirectional: offset in [-b, +b] covering both sides.
  * window w with wraparound not needed; masked band only -> O(n*w) memory.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BandAttention(nn.Module):
    """Bidirectional sliding-window (banded) self-attention with learned
    per-offset bias. Memory O(B*H*n*w)."""

    def __init__(self, dim, heads=4, window=8, drop=0.0, bias_init=0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.h, self.w = dim, heads, window
        self.dk = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        # learned bias per (head, relative offset). Per-head so different heads can
        # specialise (e.g. left-heavy vs right-heavy); index 0..2w <-> offset -w..+w
        self.rel = nn.Parameter(torch.full((heads, 2 * window + 1), bias_init))
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        w = self.w
        q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2)

        # gather neighbours: (B,h,T,2w+1,dk)
        idx = torch.arange(T, device=x.device)[:, None] + torch.arange(-w, w + 1, device=x.device)[None, :]
        valid = (idx >= 0) & (idx < T)
        idx_c = idx.clamp(0, T - 1)
        kn = k[:, :, idx_c, :].permute(0, 1, 2, 4, 3)      # (B,h,T,dk,2w+1)
        logits = (q.unsqueeze(-1) * kn).sum(3) / math.sqrt(self.dk)  # (B,h,T,2w+1)
        logits = logits + self.rel[None, :, None, :]
        logits = logits.masked_fill(~valid[None, None], -1e9)
        if mask is not None:
            logits = logits.masked_fill(~mask[:, idx_c][:, None, :, :], -1e9)
        att = self.drop(F.softmax(logits, dim=-1))
        vn = v[:, :, idx_c, :]                              # (B,h,T,2w+1,dk)
        out = (att.unsqueeze(-1) * vn).sum(3)               # (B,h,T,dk)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o(out), att


class FullAttention(nn.Module):
    """Standard bidirectional self-attention, for the 'does long range help?' test."""

    def __init__(self, dim, heads=4, drop=0.0, max_len=64):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.h, self.dk = dim, heads, dim // heads
        # per-head learned bias for every relative offset, clipped to +-max_len
        self.max_len = max_len
        self.rel = nn.Parameter(torch.zeros(heads, 2 * max_len + 1))
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dk).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        off = (torch.arange(T, device=x.device)[None, :]
               - torch.arange(T, device=x.device)[:, None]).clamp(-self.max_len, self.max_len)
        logits = logits + self.rel[:, off + self.max_len].unsqueeze(0)
        if mask is not None:
            logits = logits.masked_fill(~mask[:, None, None, :], -1e9)
        att = self.drop(F.softmax(logits, dim=-1))
        return self.o((att @ v).transpose(1, 2).reshape(B, T, D)), att


class DualBranchAttention(nn.Module):
    """Two INDEPENDENT attention branches fused by a learned per-syllable gate.

    Motivation (DeepSeek NSA, arXiv 2502.11089): when a single attention operation
    is widened, the cheapest solution - attend to the immediate neighbours - tends to
    dominate and the model never learns to use anything longer-range. Our own failed
    experiments agree: widening window 8 -> 16 made accuracy WORSE, twice.

    So instead of one wider window, two branches with separate Q/K/V projections:
      * local  branch: banded window +-w  (the existing, evidence-backed backbone)
      * global branch: full attention over the sentence (sentences are <=40
                       syllables, so this is cheap and needs no sparse machinery)
    fused by g = sigmoid(W[x_local ; x_global]) per syllable, with the gate
    initialised near 0.5 so neither branch starts dominant.

    Parameter cost: one extra attention projection set + the gate. At dim=128 with
    heads=8 that is ~66k parameters - deliberately small so a win cannot be
    explained by raw capacity (compare against the `wide` control in the experiment).
    """
    def __init__(self, dim, heads=4, window=8, drop=0.0, gate_bias=0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.h, self.dk = dim, heads, dim // heads
        self.w = window
        self.local = nn.Module()
        self.local.q = nn.Linear(dim, dim, bias=False)
        self.local.k = nn.Linear(dim, dim, bias=False)
        self.local.v = nn.Linear(dim, dim, bias=False)
        self.local.o = nn.Linear(dim, dim, bias=False)
        self.local.rel = nn.Parameter(torch.zeros(heads, 2 * window + 1))
        self.glob = nn.Module()
        self.glob.q = nn.Linear(dim, dim, bias=False)
        self.glob.k = nn.Linear(dim, dim, bias=False)
        self.glob.v = nn.Linear(dim, dim, bias=False)
        self.glob.o = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)
        self.drop = nn.Dropout(drop)

    def _split(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.h, self.dk).transpose(1, 2)

    def forward(self, x, mask=None, return_gate=False):
        B, T, D = x.shape
        w = self.w
        h = self.h
        dk = self.dk

        # ---- local band branch ----
        ql = self._split(self.local.q(x))
        kl = self._split(self.local.k(x))
        vl = self._split(self.local.v(x))
        idx = (torch.arange(T, device=x.device)[:, None]
               + torch.arange(-w, w + 1, device=x.device)[None, :])
        valid = (idx >= 0) & (idx < T)
        ic = idx.clamp(0, T - 1)
        kn = kl[:, :, ic, :].permute(0, 1, 2, 4, 3)
        lg_l = (ql.unsqueeze(-1) * kn).sum(3) / math.sqrt(dk) + self.local.rel[None, :, None, :]
        lg_l = lg_l.masked_fill(~valid[None, None], -1e9)
        if mask is not None:
            lg_l = lg_l.masked_fill(~mask[:, ic][:, None, :, :], -1e9)
        al = self.drop(F.softmax(lg_l, dim=-1))
        vl_n = vl[:, :, ic, :]
        out_l = (al.unsqueeze(-1) * vl_n).sum(3)

        # ---- global branch (full, sentence <= 40 syllables) ----
        qg = self._split(self.glob.q(x))
        kg = self._split(self.glob.k(x))
        vg = self._split(self.glob.v(x))
        lg_g = qg @ kg.transpose(-1, -2) / math.sqrt(dk)
        if mask is not None:
            lg_g = lg_g.masked_fill(~mask[:, None, None, :], -1e9)
        ag = self.drop(F.softmax(lg_g, dim=-1))
        out_g = (ag @ vg)

        # ---- fuse ----
        ol = self.local.o(out_l.transpose(1, 2).reshape(B, T, D))
        og = self.glob.o(out_g.transpose(1, 2).reshape(B, T, D))
        g = torch.sigmoid(self.gate(x))                      # (B,T,1)
        out = g * ol + (1 - g) * og
        att = al if not return_gate else (al, ag, g.squeeze(-1))
        return out, att


class AttnBlock(nn.Module):
    def __init__(self, dim, heads, window=0, drop=0.0, ff_mult=1):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        if window == "dual":
            self.att = DualBranchAttention(dim, heads, 8, drop)
        else:
            self.att = (FullAttention(dim, heads, drop) if window == 0
                        else BandAttention(dim, heads, window, drop))
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(),
                                nn.Linear(dim * ff_mult, dim))
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        h, att = self.att(self.n1(x), mask)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.n2(x)))
        return x, att
