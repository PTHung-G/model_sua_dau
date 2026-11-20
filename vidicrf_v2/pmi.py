"""Train-only compound-word PMI sketch for ViDiCRF v3 (Kernel-B ablation).

Builds an adjacent-syllable PMI table from the TRAIN partition only (no val/test
contact), stores it as sorted int64 keys + float32 values for O(log P)
`torch.searchsorted` lookup, and injects log-PMI into the CRF bigram potentials
between pruned Top-M candidates.

PMI(x, y) = log( c(x,y) * N / (c(x) * c(y)) ), clipped to [clip_min, clip_max].

Injection is a *frozen additive feature* scaled by a single learnable scalar
`lambda_pmi` (init 0.5), so the ablation attributes exactly the PMI contribution.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

TOK = re.compile(r"[^\W_]+", re.UNICODE)


def norm_form(tok: str, style: str = "classic") -> str:
    from .orthography import normalize_tone_placement

    return normalize_tone_placement(unicodedata.normalize("NFC", tok).lower(), style=style)


class PMITable:
    """Sorted-key sparse bigram PMI table over normalized syllable forms."""

    def __init__(
        self,
        train_targets: Iterable[str],
        min_pair_count: int = 3,
        max_pairs: int = 400_000,
        clip_min: float = -4.0,
        clip_max: float = 8.0,
        style: str = "classic",
    ):
        uni: Counter = Counter()
        bi: Counter = Counter()
        n_rows = 0
        for tgt in train_targets:
            toks = [norm_form(t, style) for t in TOK.findall(unicodedata.normalize("NFC", tgt))]
            if not toks:
                continue
            n_rows += 1
            uni.update(toks)
            bi.update(zip(toks, toks[1:]))

        total_bi = sum(bi.values())
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.stats = {
            "train_rows": n_rows,
            "unigram_types": len(uni),
            "bigram_types": len(bi),
            "bigram_tokens": total_bi,
            "min_pair_count": min_pair_count,
            "max_pairs": max_pairs,
        }

        # Rank candidate pairs by count; compute PMI only for survivors.
        kept = [(p, c) for p, c in bi.items() if c >= min_pair_count]
        kept.sort(key=lambda pc: (-pc[1], pc[0]))
        kept = kept[:max_pairs]

        words = sorted({w for pair, _ in kept for w in pair})
        wid = {w: i for i, w in enumerate(words)}
        self.words = words
        self.wid = wid
        self.word_id_of_vocab: Dict[str, int] = {}

        keys: List[int] = []
        vals: List[float] = []
        n = max(1, total_bi)
        for (x, y), c in kept:
            cx = uni.get(x, 1)
            cy = uni.get(y, 1)
            pmi = math.log((c * n) / (cx * cy))
            keys.append(wid[x] * (len(words) + 1) + wid[y])
            vals.append(max(clip_min, min(clip_max, pmi)))

        order = sorted(range(len(keys)), key=lambda i: keys[i])
        self.keys = torch.tensor([keys[i] for i in order], dtype=torch.int64)
        self.values = torch.tensor([vals[i] for i in order], dtype=torch.float32)
        self.stats["kept_pairs"] = len(keys)
        self.stats["clip"] = [clip_min, clip_max]

    def attach_vocab(self, vocab: Sequence[str]) -> None:
        """Map inventory candidate vocab ids -> PMI word ids (0 = not in table)."""
        wid = self.wid
        n_words = len(self.words)
        mapping = torch.zeros(len(vocab), dtype=torch.int64)
        hits = 0
        for vi, w in enumerate(vocab):
            wi = wid.get(w, 0)
            if wi:
                hits += 1
            mapping[vi] = wi if wi else 0
        self.vocab_to_pmi = mapping
        self.stats["vocab_covered"] = hits
        self.stats["vocab_size"] = len(vocab)

    def to(self, device: str | torch.device) -> "PMITable":
        self.keys = self.keys.to(device)
        self.values = self.values.to(device)
        if hasattr(self, "vocab_to_pmi"):
            self.vocab_to_pmi = self.vocab_to_pmi.to(device)
        return self

    def lookup_pairs(self, wids_x: torch.Tensor, wids_y: torch.Tensor) -> torch.Tensor:
        """PMI values for pair word-ids; 0.0 for pairs absent from the table."""
        n = len(self.words) + 1
        keys = wids_x * n + wids_y
        return self.lookup_keys(keys)

    def lookup_keys(self, keys: torch.Tensor) -> torch.Tensor:
        if self.keys.numel() == 0:
            return torch.zeros_like(keys, dtype=torch.float32)
        kmax_ = self.keys[-1]
        kmin_ = self.keys[0]
        in_range = (keys >= kmin_) & (keys <= kmax_)
        kc = keys.clamp(min=kmin_, max=kmax_)
        idx = torch.searchsorted(self.keys, kc)
        idx_c = idx.clamp(max=self.keys.numel() - 1)
        found = (self.keys[idx_c] == keys) & in_range
        vals = self.values[idx_c] * found.to(torch.float32)
        return vals


def pmi_bigram_addend(
    table: PMITable,
    bigram: torch.Tensor,
    base_ids: torch.Tensor,
    top_idx: torch.Tensor,
    cand_ids: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """Return lam * PMI over (B, L-1, M, M) candidate pairs (add to bigram potentials).

    top_idx: (B, L, M) pruned slot ids; cand_ids: (n_bases+1, K) slot->vid.
    """
    if not hasattr(table, "vocab_to_pmi"):
        raise RuntimeError("PMITable.attach_vocab() must be called before injection")
    rows = base_ids.clamp(min=0).unsqueeze(-1).expand(-1, -1, top_idx.shape[-1])
    vids = cand_ids[rows, top_idx]  # (B, L, M)
    pmid = table.vocab_to_pmi[vids]  # (B, L, M)
    px = pmid[:, :-1, :].unsqueeze(-1)  # (B, L-1, M, 1)
    py = pmid[:, 1:, :].unsqueeze(-2)  # (B, L-1, 1, M)
    n = len(table.words) + 1
    keys = (px * n + py).expand(bigram.shape[0], bigram.shape[1], bigram.shape[2], bigram.shape[3])
    vals = table.lookup_keys(keys.reshape(-1)).view(bigram.shape)
    return lam * vals
