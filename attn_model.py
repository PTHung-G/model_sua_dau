"""Attention-based tiny diacritic restorer.

Architecture: char-BiGRU over the stripped syllable -> 1 attention block over the
syllable sequence (banded, bidirectional) -> candidate slot scores + unigram prior.

Variants are selected by `window`:
    window=0   full bidirectional attention   (does long range help?)
    window=8   local band, bidirectional      (evidence-backed default)
    window=2   local band, narrow             (near-context only)

`direction='forward'` masks the right side, giving a one-sided ablation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attn import AttnBlock

NEG = -1e9


class AttnDiac(nn.Module):
    def __init__(self, nchar, max_c, ch_dim=16, syl=32, dim=64, heads=8,
                 window=8, blocks=1, drop=0.0, ff_mult=1,
                 acc_feedback=False, out_std=1e-2, factorized=False, packed_chars=False):
        super().__init__()
        # acc_feedback: allow a second decoding pass to be conditioned on the
        # accents predicted by the first pass. Slot 0 of the embedding means
        # "no accent information", so a single forward() call still works and the
        # model can learn to fall back when feedback is absent.
        self.acc_feedback = acc_feedback
        self.acc_emb = nn.Embedding(max_c + 1, dim) if acc_feedback else None
        self.packed_chars = packed_chars
        # Legacy fixed-size factorization is retained only for compatibility.
        # New pilot builds collision-checked, position-aware modifier signatures.
        self.factorized = factorized
        self.cemb = nn.Embedding(nchar, ch_dim, padding_idx=0)
        self.cgru = nn.GRU(ch_dim, syl // 2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(syl, dim)
        self.blocks = nn.ModuleList(
            [AttnBlock(dim, heads, window, drop, ff_mult) for _ in range(blocks)])
        self.norm = nn.LayerNorm(dim)
        if factorized:
            from vn import tone_of, decompose
            self.n_tone = 6
            self.n_mod = 7
            self.tone_head = nn.Linear(dim, self.n_tone)
            self.mod_head = nn.Linear(dim, self.n_mod)
            # start with a small, non-zero output scale for the same reason out_std
            # exists: a large init collapses onto the majority class immediately.
            nn.init.normal_(self.tone_head.weight, std=out_std)
            nn.init.zeros_(self.tone_head.bias)
            nn.init.normal_(self.mod_head.weight, std=out_std)
            nn.init.zeros_(self.mod_head.bias)
        else:
            self.out = nn.Linear(dim, max_c)
            nn.init.normal_(self.out.weight, std=out_std)
            nn.init.zeros_(self.out.bias)
        if acc_feedback:
            nn.init.zeros_(self.acc_emb.weight)

    def encode_syllables(self, xc, cl):
        B, T, L = xc.shape
        emb = self.cemb(xc.reshape(B * T, L))
        if self.packed_chars:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, cl.reshape(-1).clamp(min=1).cpu(), batch_first=True,
                enforce_sorted=False)
            _, hn = self.cgru(packed)
            return torch.cat((hn[-2], hn[-1]), dim=-1).reshape(B, T, -1)
        h, _ = self.cgru(emb)
        i = (cl.reshape(-1) - 1).clamp(min=0)
        return h[torch.arange(h.size(0)), i].reshape(B, T, -1)

    def forward(self, xc, cl, mask=None, return_att=False, acc_ids=None):
        B, T, L = xc.shape
        h = self.proj(self.encode_syllables(xc, cl))
        if self.acc_emb is not None and acc_ids is not None:
            h = h + self.acc_emb(acc_ids)
        atts = []
        for b in self.blocks:
            h, a = b(h, mask)
            atts.append(a)
        h = self.norm(h)
        if self.factorized:
            return self.tone_head(h), self.mod_head(h)
        out = self.out(h)
        if return_att:
            return out, atts
        return out


def logits_of(model, xc, cl, f, ln, pri):
    lg = model(xc, cl) + pri[f]
    ar = torch.arange(lg.size(-1), device=lg.device)[None, None, :]
    return lg.masked_fill(ar >= ln[..., None], NEG)


def evaluate(model, S, pri, bs=128):
    from tiny2 import batches
    model.eval()
    tot = ok = ta = oa = tu = ou = 0
    with torch.no_grad():
        for xc, cl, f, y, ln, m in batches(S, bs, shuffle=False):
            pred = logits_of(model, xc, cl, f, ln, pri).argmax(-1)
            hit = (pred == y) & m
            tot += m.sum().item(); ok += hit.sum().item()
            a = (ln > 1) & m; u = (ln == 1) & m
            ta += a.sum().item(); oa += (hit & a).sum().item()
            tu += u.sum().item(); ou += (hit & u).sum().item()
    return ok / tot, oa / max(ta, 1), ou / max(tu, 1), tot


@torch.no_grad()
def rel_bias_report(model):
    """Return the learned distance bias per head. This is the direct evidence for
    'near context matters more': if the curve is flat, the model ignores distance."""
    import json
    out = {}
    for i, b in enumerate(model.blocks):
        rel = b.att.rel.detach().cpu()
        w = getattr(b.att, "w", 64)
        out[f"block{i}"] = {
            "offsets": list(range(-w, w + 1)) if hasattr(b.att, "w")
                       else list(range(-b.att.max_len, b.att.max_len + 1)),
            "per_head": rel.tolist(),
            "mean_over_heads": rel.mean(0).tolist(),
        }
    return out
