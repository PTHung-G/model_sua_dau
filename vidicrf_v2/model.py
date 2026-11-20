"""ViDiCRF v2: Phonology-aware Conformer-RoPE + Anchor-Propagating Top-M Trigram CRF.

Key architectural advances over v1 (PunctSylCRF):
1. Compositional Word Encoder: Multi-scale character Conv1D (base + mod + tone)
   combined with 6-part Vietnamese phonological decomposition, eliminating
   per-character sequential RNN loops and CPU synchronization barriers.
2. Lexical Syllable Embedding + Explicit Train Log-Priors: Separates lexical
   semantics and empirical frequency priors from character spelling encoding.
3. Zero-Copy Dynamic OOV & Precomputed Inference Cache: Eliminates the ~266 MB
   tensor copy on unseen OOV tokens and caches static candidate embeddings in eval.
4. Surface Diacritic Awareness: Reads existing diacritics on partially accented
   inputs instead of stripping them away before encoding.
5. Conformer-Lite Backbone (Multi-scale Gated Depthwise Conv1D + RoPE Attention
   with SwiGLU): Fast SIMD execution on CPU and length-extrapolative attention.
6. Iterative Anchor Propagation: Mid-network expected accented embeddings and
   uncertainty feed forward into Stage 2 so ambiguous tokens attend to resolved anchors.
7. Top-M Lattice Pruning + Exact Second-Order (Trigram) Signed-Biaffine CRF:
   Prunes K=32 -> M=4 candidates and runs exact second-order CRF (4^3 = 64 states/step)
   with signed contextual bigram and low-rank trigram potentials.
8. Softmax-Margin (Hamming Cost-Augmented) CRF + Auxiliary Emission CE Loss.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .orthography import PHONO_CARDINALITIES

NEG = -1e9


def _apply_rope(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings to q, k of shape (B, H, L, Dh)."""
    _, _, L, Dh = q.shape
    half = Dh // 2
    if half == 0:
        return q, k
    inv_freq = torch.exp(
        torch.arange(0, half, device=q.device, dtype=torch.float32)
        * (-math.log(10000.0) / half)
    )
    pos = torch.arange(L, device=q.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # (L, half)
    cos = freqs.cos().to(q.dtype)[None, None, :, :]
    sin = freqs.sin().to(q.dtype)[None, None, :, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :half]
        x2 = x[..., half : 2 * half]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        if 2 * half < Dh:
            return torch.cat([out1, out2, x[..., 2 * half :]], dim=-1)
        return torch.cat([out1, out2], dim=-1)

    return rotate(q), rotate(k)


class CompositionalWordEncoder(nn.Module):
    """Encodes a word from compositional character Conv1D + 6-component VN phonology."""

    def __init__(
        self,
        char_base_map: torch.Tensor,
        char_mod_map: torch.Tensor,
        char_tone_map: torch.Tensor,
        n_base_chars: int,
        n_char_mods: int,
        n_char_tones: int,
        d: int = 128,
        char_d: int = 64,
        phono_d: int = 16,
    ):
        super().__init__()
        self.d = d
        self.register_buffer("char_base_map", char_base_map.long())
        self.register_buffer("char_mod_map", char_mod_map.long())
        self.register_buffer("char_tone_map", char_tone_map.long())

        self.c_base_emb = nn.Embedding(n_base_chars, char_d, padding_idx=0)
        self.c_mod_emb = nn.Embedding(n_char_mods, char_d, padding_idx=0)
        self.c_tone_emb = nn.Embedding(n_char_tones, char_d, padding_idx=0)

        self.conv2 = nn.Conv1d(char_d, char_d // 2, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(char_d, char_d // 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(char_d, char_d // 2, kernel_size=4, padding=2)
        # mean + max over (3 * char_d // 2) + raw char mean (char_d)
        conv_out_dim = 2 * (3 * (char_d // 2)) + char_d
        self.char_proj = nn.Sequential(nn.Linear(conv_out_dim, d), nn.GELU())

        n_on, n_gl, n_nuc, n_cod, n_mp, n_tn = PHONO_CARDINALITIES
        self.p_onset = nn.Embedding(n_on, phono_d, padding_idx=0)
        self.p_glide = nn.Embedding(n_gl, phono_d // 2, padding_idx=0)
        self.p_nuc = nn.Embedding(n_nuc, phono_d, padding_idx=0)
        self.p_coda = nn.Embedding(n_cod, phono_d, padding_idx=0)
        self.p_mod = nn.Embedding(n_mp, phono_d // 2, padding_idx=0)
        self.p_tone = nn.Embedding(n_tn, phono_d, padding_idx=0)
        p_in = 4 * phono_d + 2 * (phono_d // 2)
        self.phono_mlp = nn.Sequential(
            nn.Linear(p_in, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.out_norm = nn.LayerNorm(d)

    def forward(self, char_ids: torch.Tensor, phono_ids: torch.Tensor) -> torch.Tensor:
        """Encode flattened words: char_ids (N, W), phono_ids (N, 6) -> (N, d)."""
        if char_ids.numel() == 0:
            return char_ids.new_empty((0, self.d), dtype=torch.float32)

        N, W = char_ids.shape
        ids_l = char_ids.long().clamp(min=0, max=self.char_base_map.shape[0] - 1)
        slots = torch.arange(1, W + 1, device=char_ids.device).unsqueeze(0)
        lengths = (ids_l.ne(0) * slots).amax(dim=-1)  # (N,)
        valid_pos = torch.arange(W, device=char_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        has_any = lengths > 0

        b_id = self.char_base_map[ids_l]
        m_id = self.char_mod_map[ids_l]
        t_id = self.char_tone_map[ids_l]
        x = self.c_base_emb(b_id) + self.c_mod_emb(m_id) + self.c_tone_emb(t_id)
        x = x * valid_pos.unsqueeze(-1).to(x.dtype)  # (N, W, char_d)

        # Raw masked mean
        denom = lengths.clamp(min=1).unsqueeze(-1).to(x.dtype)
        raw_mean = x.sum(dim=1) / denom  # (N, char_d)

        xt = x.transpose(1, 2)  # (N, char_d, W)
        # Slice to width W cleanly (conv2 and conv4 with even kernels output W+1)
        c2 = F.gelu(self.conv2(xt)[:, :, :W])
        c3 = F.gelu(self.conv3(xt)[:, :, :W])
        c4 = F.gelu(self.conv4(xt)[:, :, :W])
        feats = torch.cat([c2, c3, c4], dim=1)  # (N, 3*char_d//2, W)

        vp = valid_pos.unsqueeze(1)
        feats_zero = feats.masked_fill(~vp, 0.0)
        f_mean = feats_zero.sum(dim=-1) / denom
        feats_neg = feats.masked_fill(~vp, -1e9)
        f_max = torch.where(has_any.unsqueeze(-1), feats_neg.amax(dim=-1), torch.zeros_like(f_mean))

        char_vec = self.char_proj(torch.cat([f_mean, f_max, raw_mean], dim=-1))
        char_vec = char_vec * has_any.unsqueeze(-1).to(char_vec.dtype)

        p = phono_ids.long()
        p_cat = torch.cat(
            [
                self.p_onset(p[:, 0]),
                self.p_glide(p[:, 1]),
                self.p_nuc(p[:, 2]),
                self.p_coda(p[:, 3]),
                self.p_mod(p[:, 4]),
                self.p_tone(p[:, 5]),
            ],
            dim=-1,
        )
        p_vec = self.phono_mlp(p_cat)
        out = self.out_norm(char_vec + p_vec)
        return out * has_any.unsqueeze(-1).to(out.dtype)


class ConformerLiteBlock(nn.Module):
    """Multi-scale Gated Depthwise Conv1D + RoPE Multi-Head Attention + SwiGLU FFN."""

    def __init__(self, d: int = 128, n_heads: int = 4, ffn_dim: int = 256, drop: float = 0.15):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        assert self.head_dim * n_heads == d

        # 1. Multi-scale Gated Depthwise Conv1D
        self.conv_norm = nn.LayerNorm(d)
        self.conv_in = nn.Linear(d, 2 * d)
        self.dw_conv3 = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d)
        self.dw_conv5 = nn.Conv1d(d, d, kernel_size=5, padding=2, groups=d)
        self.conv_out = nn.Linear(2 * d, d)
        self.conv_drop = nn.Dropout(drop)

        # 2. RoPE Self-Attention
        self.attn_norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.attn_out = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(drop)

        # 3. SwiGLU FFN
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn_in = nn.Linear(d, 2 * ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d)
        self.ffn_drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d), valid: bool (B, L)."""
        vmask = valid.unsqueeze(-1).to(x.dtype)

        # Gated Depthwise Conv1D
        c_in = self.conv_in(self.conv_norm(x))
        a, g = c_in.chunk(2, dim=-1)
        u = (a * torch.sigmoid(g)) * vmask  # zero padding before conv!
        ut = u.transpose(1, 2)
        c3 = self.dw_conv3(ut).transpose(1, 2)
        c5 = self.dw_conv5(ut).transpose(1, 2)
        c_mix = self.conv_out(torch.cat([c3, c5], dim=-1))
        x = (x + self.conv_drop(c_mix)) * vmask

        # RoPE Attention
        B, L, _ = x.shape
        qkv = self.qkv(self.attn_norm(x)).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, L, Dh)
        q, k = _apply_rope(q, k)
        # Ensure key position 0 is never masked out to prevent NaN on padded query rows
        safe_valid = valid.clone()
        safe_valid[:, 0] = True
        z = F.scaled_dot_product_attention(
            q, k, v, attn_mask=safe_valid[:, None, None, :], dropout_p=0.0
        )
        z = z.transpose(1, 2).reshape(B, L, self.d)
        x = (x + self.attn_drop(self.attn_out(z))) * vmask

        # SwiGLU FFN
        f1, f2 = self.ffn_in(self.ffn_norm(x)).chunk(2, dim=-1)
        f_out = self.ffn_out(F.silu(f1) * f2)
        x = (x + self.ffn_drop(f_out)) * vmask
        return x


class ViDiCRFv2(nn.Module):
    """Full ViDiCRF v2 architecture with Anchor Propagation and Top-M Trigram CRF."""

    def __init__(
        self,
        char_base_map: torch.Tensor,
        char_mod_map: torch.Tensor,
        char_tone_map: torch.Tensor,
        n_base_chars: int,
        n_char_mods: int,
        n_char_tones: int,
        base_ch: torch.Tensor,
        base_phono: torch.Tensor,
        cand_ch: torch.Tensor,
        cand_phono: torch.Tensor,
        cand_lex: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_priors: torch.Tensor,
        npunct: int,
        n_lex_vocab: int,
        d: int = 128,
        n_heads: int = 4,
        stage1_layers: int = 2,
        stage2_layers: int = 2,
        ffn_dim: int = 256,
        lex_dim: int = 64,
        r_big: int = 32,
        r_tri: int = 16,
        top_m: int = 8,
        drop: float = 0.10,
    ):
        super().__init__()
        self.d = d
        self.r_big = r_big
        self.r_tri = r_tri
        self.top_m = top_m
        self.n_lex_vocab = n_lex_vocab

        # Inventory buffers (never mutated during OOV inference)
        self.register_buffer("base_ch", base_ch.to(torch.int16))
        self.register_buffer("base_phono", base_phono.to(torch.uint8))
        self.register_buffer("cand_ch", cand_ch.to(torch.int16))
        self.register_buffer("cand_phono", cand_phono.to(torch.uint8))
        self.register_buffer("cand_lex", cand_lex.to(torch.int32))
        self.register_buffer("cand_ids", cand_ids.to(torch.int32))
        self.register_buffer("cand_priors", cand_priors.to(torch.float16))

        # Tầng 1: Word & Candidate Representation
        self.word_enc = CompositionalWordEncoder(
            char_base_map,
            char_mod_map,
            char_tone_map,
            n_base_chars,
            n_char_mods,
            n_char_tones,
            d=d,
        )
        self.input_proj = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.cand_proj = nn.Sequential(nn.Linear(d, d), nn.GELU())

        # Direct zero-initialized lexical syllable embedding (full update magnitude from epoch 1)
        self.lex_emb = nn.Embedding(max(1, n_lex_vocab), d, padding_idx=0)
        nn.init.zeros_(self.lex_emb.weight)

        # Side features: left_gap, right_gap, case, has_input_diacritic
        self.left_gap_emb = nn.Embedding(npunct, 24, padding_idx=0)
        self.right_gap_emb = nn.Embedding(npunct, 24, padding_idx=0)
        self.case_emb = nn.Embedding(5, 8, padding_idx=0)
        self.diac_emb = nn.Embedding(2, 8)
        self.side_proj = nn.Linear(24 + 24 + 8 + 8, d)

        self.in_norm = nn.LayerNorm(d)
        self.in_drop = nn.Dropout(drop)

        # Tầng 2: Hybrid BiGRU + Conformer-Lite Stage 1 -> Anchor Propagation -> Stage 2
        self.seq_gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.seq_gru_norm = nn.LayerNorm(d)
        self.stage1 = nn.ModuleList(
            [ConformerLiteBlock(d, n_heads, ffn_dim, drop) for _ in range(stage1_layers)]
        )

        # Prior + case interaction head
        # priors (3): log P(c|b), log(count+1)/15, is_identity
        self.prior_head1 = nn.Sequential(nn.Linear(3 + 8, 16), nn.GELU(), nn.Linear(16, 1))
        self.prior_head2 = nn.Sequential(nn.Linear(3 + 8, 16), nn.GELU(), nn.Linear(16, 1))
        self.h1_norm = nn.LayerNorm(d)
        self.h2_norm = nn.LayerNorm(d)
        self.emit1_w = nn.Linear(d, d, bias=True)
        self.emit2_w = nn.Linear(d, d, bias=True)

        # Iterative Anchor Propagation gate
        self.anchor_proj = nn.Linear(d, d)
        self.anchor_gate = nn.Linear(2 * d + 1, d)
        nn.init.zeros_(self.anchor_gate.bias)

        self.stage2 = nn.ModuleList(
            [ConformerLiteBlock(d, n_heads, ffn_dim, drop) for _ in range(stage2_layers)]
        )

        # Tầng 3: Signed Contextual Biaffine Bigram + Low-Rank Trigram CRF
        self.trans_u_stat = nn.Linear(d, r_big, bias=False)
        self.trans_v_stat = nn.Linear(d, r_big, bias=False)
        self.trans_u_ctx = nn.Linear(d, r_big, bias=False)
        self.trans_v_ctx = nn.Linear(d, r_big, bias=False)
        # Zero-initialized signed context modulation (NOT clamped to 1+tanh > 0)
        self.edge_signed = nn.Linear(2 * d + 24, r_big)
        nn.init.zeros_(self.edge_signed.weight)
        nn.init.zeros_(self.edge_signed.bias)

        self.tri_p = nn.Linear(d, r_tri, bias=False)
        self.tri_q = nn.Linear(d, r_tri, bias=False)
        self.tri_w = nn.Linear(d, r_tri, bias=False)
        self.tri_gate = nn.Linear(d + 24 + 24, r_tri)
        nn.init.zeros_(self.tri_gate.weight)
        nn.init.zeros_(self.tri_gate.bias)

        # Optional eval-time precomputed candidate embedding cache
        self._cached_cand_vecs: Optional[torch.Tensor] = None

    def invalidate_cache(self) -> None:
        self._cached_cand_vecs = None

    @torch.no_grad()
    def build_inference_cache(self, batch_size: int = 8192) -> None:
        """Precompute candidate embeddings for the entire vocab during eval()."""
        self.eval()
        nvocab = self.cand_ch.shape[0]
        if nvocab == 0:
            self._cached_cand_vecs = self.cand_ch.new_empty((0, self.d), dtype=torch.float32)
            return
        chunks = []
        for start in range(0, nvocab, batch_size):
            end = min(nvocab, start + batch_size)
            comp = self.cand_proj(
                self.word_enc(self.cand_ch[start:end], self.cand_phono[start:end])
            )
            lex = self.lex_emb(self.cand_lex[start:end].long())
            chunks.append(comp + lex)
        self._cached_cand_vecs = torch.cat(chunks, dim=0)

    def _encode_vocab_ids(self, flat_vids: torch.Tensor) -> torch.Tensor:
        if self._cached_cand_vecs is not None and not self.training:
            return self._cached_cand_vecs[flat_vids]
        uniq, inverse = torch.unique(flat_vids, return_inverse=True)
        comp = self.cand_proj(self.word_enc(self.cand_ch[uniq], self.cand_phono[uniq]))
        lex = self.lex_emb(self.cand_lex[uniq].long())
        return (comp + lex)[inverse]

    def _encode_surface_words(
        self,
        surface_chars: torch.Tensor,
        surface_phono: torch.Tensor,
        surface_lex: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, W = surface_chars.shape
        comp_raw = self.word_enc(
            surface_chars.reshape(B * L, W),
            surface_phono.reshape(B * L, 6),
        ).view(B, L, self.d)
        if surface_lex is not None:
            lex_vec = self.lex_emb(surface_lex.long())
        else:
            lex_vec = torch.zeros_like(comp_raw)
        in_vec = self.input_proj(comp_raw) + lex_vec
        oov_cand_vec = self.cand_proj(comp_raw) + lex_vec
        return in_vec, oov_cand_vec

    def forward(
        self,
        base_ids: torch.Tensor,
        sent_len: torch.Tensor,
        case_ids: torch.Tensor,
        left_gaps: torch.Tensor,
        right_gaps: torch.Tensor,
        surface_chars: torch.Tensor,
        surface_phono: torch.Tensor,
        has_input_diac: Optional[torch.Tensor] = None,
        surface_lex: Optional[torch.Tensor] = None,
        gold: Optional[torch.Tensor] = None,
        forced_slots: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, L = base_ids.shape
        valid_tok = torch.arange(L, device=base_ids.device).unsqueeze(0) < sent_len.unsqueeze(1)
        if has_input_diac is None:
            has_input_diac = torch.zeros_like(base_ids)

        in_vec, oov_cand_vec = self._encode_surface_words(
            surface_chars, surface_phono, surface_lex
        )
        lg_vec = self.left_gap_emb(left_gaps.long())
        rg_vec = self.right_gap_emb(right_gaps.long())
        c_vec = self.case_emb(case_ids.long())
        d_vec = self.diac_emb(has_input_diac.long())
        side_vec = self.side_proj(torch.cat([lg_vec, rg_vec, c_vec, d_vec], dim=-1))

        vmask = valid_tok.unsqueeze(-1).to(in_vec.dtype)
        h = self.in_drop(self.in_norm(in_vec + side_vec)) * vmask

        # Fast-path BiGRU (skip pack_padded_sequence when B==1 or unpadded batch)
        if B == 1 or bool((sent_len == L).all()):
            gru_out, _ = self.seq_gru(h)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                h, sent_len.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.seq_gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=L
            )
        h = self.seq_gru_norm(h + gru_out) * vmask

        # Stage 1 Conformer-Lite
        for blk in self.stage1:
            h = blk(h, valid_tok)
        h1 = h

        # Build candidate tensor (B, L, K, d) with Zero-Copy OOV support at base_id == 0
        is_known = (base_ids > 0) & valid_tok
        safe_base = base_ids.clamp(min=0, max=self.cand_ids.shape[0] - 1).long()
        cid = self.cand_ids[safe_base].long()  # (B, L, K)
        K = cid.shape[-1]

        valid_known = (cid >= 0) & is_known.unsqueeze(-1)
        if self.cand_ch.shape[0] > 0:
            ce_known = self._encode_vocab_ids(cid.clamp(min=0).reshape(-1)).view(B, L, K, self.d)
        else:
            ce_known = h1.new_zeros((B, L, K, self.d))

        priors_known = self.cand_priors[safe_base].to(h1.dtype)  # (B, L, K, 3)

        # For OOV / identity-copy tokens (base_id == 0), slot 0 is the surface form itself
        slot0 = torch.zeros((B, L, K), dtype=torch.bool, device=base_ids.device)
        slot0[:, :, 0] = (~is_known) & valid_tok
        valid_cand = valid_known | slot0

        ce = ce_known.clone()
        ce[:, :, 0] = torch.where(is_known.unsqueeze(-1), ce_known[:, :, 0], oov_cand_vec)

        oov_prior = priors_known.new_tensor([0.0, 0.0, 1.0]).view(1, 1, 3).expand(B, L, 3)
        priors = priors_known.clone()
        priors[:, :, 0] = torch.where(is_known.unsqueeze(-1), priors_known[:, :, 0], oov_prior)

        # Prior dropout forces contextual backbone to disambiguate instead of relying solely on unigram priors
        priors_in = F.dropout(priors, p=0.25, training=self.training)
        case_exp = c_vec.unsqueeze(2).expand(B, L, K, 8)
        prior_in = torch.cat([priors_in, case_exp], dim=-1)
        bias1 = self.prior_head1(prior_in).squeeze(-1)
        bias2 = self.prior_head2(prior_in).squeeze(-1)

        # Stage-1 Emission (LayerNorm on h1 + natural Linear initialization scale)
        q1 = self.emit1_w(self.h1_norm(h1))
        emit1 = torch.einsum("bld,blkd->blk", q1, ce) + bias1
        emit1 = emit1.masked_fill(~valid_cand, NEG)

        # Iterative Anchor Propagation
        p1 = F.softmax(emit1, dim=-1) * valid_cand.to(emit1.dtype)
        anchor_emb = torch.einsum("blk,blkd->bld", p1, ce)
        ent = -(p1 * p1.clamp(min=1e-9).log()).sum(dim=-1, keepdim=True) / math.log(max(2, K))
        gate_a = torch.sigmoid(self.anchor_gate(torch.cat([h1, anchor_emb, ent], dim=-1)))
        h_mid = (h1 + gate_a * self.anchor_proj(anchor_emb)) * vmask

        # Stage 2 Conformer-Lite
        h2 = h_mid
        for blk in self.stage2:
            h2 = blk(h2, valid_tok)

        # Stage-2 Final Emission
        q2 = self.emit2_w(self.h2_norm(h2))
        emit2 = torch.einsum("bld,blkd->blk", q2, ce) + bias2
        emit2 = emit2.masked_fill(~valid_cand, NEG)

        # Top-M Lattice Pruning
        M = min(self.top_m, K)
        _, top_idx = emit2.topk(M, dim=-1)  # (B, L, M)
        gold_p: Optional[torch.Tensor] = None

        keep_slot = gold if gold is not None else forced_slots
        if keep_slot is not None:
            has_slot = valid_tok & (keep_slot >= 0)
            s_clamped = keep_slot.long().clamp(min=0, max=K - 1)
            in_top = (top_idx == s_clamped.unsqueeze(-1)).any(dim=-1)
            need_fix = has_slot & (~in_top)
            if need_fix.any():
                top_idx = top_idx.clone()
                top_idx[:, :, -1] = torch.where(need_fix, s_clamped, top_idx[:, :, -1])
            if gold is not None:
                gold_match = top_idx == s_clamped.unsqueeze(-1)
                gold_p = gold_match.long().argmax(dim=-1)  # (B, L)

        emit_p = emit2.gather(-1, top_idx)
        mask_p = valid_cand.gather(-1, top_idx)
        emit_p = emit_p.masked_fill(~mask_p, NEG)

        ce_p = ce.gather(2, top_idx.unsqueeze(-1).expand(B, L, M, self.d))

        # Bigram Signed Contextual Biaffine Transition: (B, L-1, M, M)
        if L >= 2:
            u_s = self.trans_u_stat(ce_p[:, :-1])
            v_s = self.trans_v_stat(ce_p[:, 1:])
            big_stat = torch.einsum("blmr,blnr->blmn", u_s, v_s)

            u_c = self.trans_u_ctx(ce_p[:, :-1])
            v_c = self.trans_v_ctx(ce_p[:, 1:])
            edge_ctx = torch.cat([h2[:, :-1], h2[:, 1:], rg_vec[:, :-1]], dim=-1)
            edge_w = self.edge_signed(edge_ctx)
            big_ctx = torch.einsum("blmr,blr,blnr->blmn", u_c, edge_w, v_c)
            bigram = big_stat + big_ctx
        else:
            bigram = emit_p.new_zeros((B, 0, M, M))

        # Trigram Low-Rank Contextual Transition: (B, L-2, M, M, M)
        if L >= 3:
            a3 = self.tri_p(ce_p[:, :-2])
            b3 = self.tri_q(ce_p[:, 1:-1])
            c3 = self.tri_w(ce_p[:, 2:])
            tri_ctx = torch.cat(
                [h2[:, 1:-1], lg_vec[:, 1:-1], rg_vec[:, 1:-1]], dim=-1
            )
            tri_g = self.tri_gate(tri_ctx)
            trigram = torch.einsum("blir,bljr,blkr,blr->blijk", a3, b3, c3, tri_g)
        else:
            trigram = emit_p.new_zeros((B, 0, M, M, M))

        out: Dict[str, torch.Tensor] = {
            "emit1": emit1,
            "emit2": emit2,
            "emit_p": emit_p,
            "mask_p": mask_p,
            "top_idx": top_idx,
            "bigram": bigram,
            "trigram": trigram,
            "valid_tok": valid_tok,
            "valid_cand": valid_cand,
        }
        if gold_p is not None:
            out["gold_p"] = gold_p
        return out


# =============================================================================
# Exact Second-Order (Trigram) CRF Algorithms: LogZ, Gold, Viterbi, Marginals
# =============================================================================

def trigram_log_partition(
    emit_p: torch.Tensor,
    bigram: torch.Tensor,
    trigram: torch.Tensor,
    mask_p: torch.Tensor,
) -> torch.Tensor:
    """Exact log partition function over (y_0, ..., y_{T-1}) for 2nd-order CRF."""
    B, L, M = emit_p.shape
    emit_m = emit_p.masked_fill(~mask_p, NEG)
    real = mask_p.any(dim=-1)  # (B, L)

    lp_eq1 = torch.logsumexp(emit_m[:, 0], dim=-1)
    if L == 1:
        return lp_eq1

    # alpha2[b, i, j]: log sum up to step 1 ending with (y_0=i, y_1=j)
    alpha2 = emit_m[:, 0].unsqueeze(2) + bigram[:, 0] + emit_m[:, 1].unsqueeze(1)
    for t in range(2, L):
        # cand[b, i, j, k] = alpha2[b, i, j] + trigram[t-2, i, j, k] + bigram[t-1, j, k]
        cand = alpha2.unsqueeze(3) + trigram[:, t - 2] + bigram[:, t - 1].unsqueeze(1)
        nxt = torch.logsumexp(cand, dim=1) + emit_m[:, t].unsqueeze(1)
        alpha2 = torch.where(real[:, t].view(B, 1, 1), nxt, alpha2)

    lp_ge2 = torch.logsumexp(alpha2.reshape(B, M * M), dim=-1)
    return torch.where(real[:, 1], lp_ge2, lp_eq1)


def trigram_gold_score(
    emit_p: torch.Tensor,
    bigram: torch.Tensor,
    trigram: torch.Tensor,
    mask_p: torch.Tensor,
    gold_p: torch.Tensor,
) -> torch.Tensor:
    """Exact score of gold path `gold_p` (B, L) inside pruned lattice."""
    B, L, _ = emit_p.shape
    real = mask_p.any(dim=-1)
    b_idx = torch.arange(B, device=emit_p.device)

    s = emit_p[b_idx, 0, gold_p[:, 0]]
    if L >= 2:
        step1 = emit_p[b_idx, 1, gold_p[:, 1]] + bigram[b_idx, 0, gold_p[:, 0], gold_p[:, 1]]
        s = s + torch.where(real[:, 1], step1, torch.zeros_like(step1))
    for t in range(2, L):
        step_t = (
            emit_p[b_idx, t, gold_p[:, t]]
            + bigram[b_idx, t - 1, gold_p[:, t - 1], gold_p[:, t]]
            + trigram[b_idx, t - 2, gold_p[:, t - 2], gold_p[:, t - 1], gold_p[:, t]]
        )
        s = s + torch.where(real[:, t], step_t, torch.zeros_like(step_t))
    return s


@torch.no_grad()
def trigram_viterbi(
    emit_p: torch.Tensor,
    bigram: torch.Tensor,
    trigram: torch.Tensor,
    mask_p: torch.Tensor,
    locked_mask_p: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact second-order Viterbi decoding returning pruned slot indices (B, L)."""
    eff_mask = mask_p
    if locked_mask_p is not None:
        has_lock = locked_mask_p.any(dim=-1, keepdim=True)
        comb = mask_p & locked_mask_p
        eff_mask = torch.where(has_lock & comb.any(dim=-1, keepdim=True), comb, mask_p)

    emit_m = emit_p.masked_fill(~eff_mask, NEG)
    B, L, M = emit_m.shape
    real = eff_mask.any(dim=-1)

    pred_eq1 = emit_m[:, 0].argmax(dim=-1)
    if L == 1:
        return pred_eq1.unsqueeze(1)

    score2 = emit_m[:, 0].unsqueeze(2) + bigram[:, 0] + emit_m[:, 1].unsqueeze(1)
    ptr = torch.zeros((B, L, M, M), dtype=torch.long, device=emit_m.device)

    for t in range(2, L):
        cand = score2.unsqueeze(3) + trigram[:, t - 2] + bigram[:, t - 1].unsqueeze(1)
        best, idx = cand.max(dim=1)  # (B, M, M)
        nxt = best + emit_m[:, t].unsqueeze(1)
        score2 = torch.where(real[:, t].view(B, 1, 1), nxt, score2)
        ptr[:, t] = idx

    best_flat = score2.reshape(B, M * M).argmax(dim=-1)
    j_cur = best_flat // M
    k_cur = best_flat % M

    out = torch.zeros((B, L), dtype=torch.long, device=emit_m.device)
    b_idx = torch.arange(B, device=emit_m.device)

    for t in range(L - 1, 1, -1):
        w = real[:, t]
        out[:, t] = torch.where(w, k_cur, out[:, t])
        prev_i = ptr[b_idx, t, j_cur, k_cur]
        k_cur = torch.where(w, j_cur, k_cur)
        j_cur = torch.where(w, prev_i, j_cur)

    w1 = real[:, 1]
    out[:, 1] = torch.where(w1, k_cur, out[:, 1])
    out[:, 0] = torch.where(w1, j_cur, pred_eq1)
    return out


@torch.no_grad()
def trigram_marginals(
    emit_p: torch.Tensor,
    bigram: torch.Tensor,
    trigram: torch.Tensor,
    mask_p: torch.Tensor,
    locked_mask_p: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact Forward-Backward marginal log-probabilities on second-order CRF."""
    eff_mask = mask_p
    if locked_mask_p is not None:
        has_lock = locked_mask_p.any(dim=-1, keepdim=True)
        comb = mask_p & locked_mask_p
        eff_mask = torch.where(has_lock & comb.any(dim=-1, keepdim=True), comb, mask_p)

    emit_m = emit_p.masked_fill(~eff_mask, NEG)
    B, L, M = emit_m.shape
    real = eff_mask.any(dim=-1)

    if L == 1:
        log_z = torch.logsumexp(emit_m[:, 0], dim=-1)
        log_marg = (emit_m[:, 0] - log_z.unsqueeze(1)).masked_fill(~eff_mask[:, 0], NEG).unsqueeze(1)
        return log_marg, log_z

    alpha_hist = [None] * L
    alpha2 = emit_m[:, 0].unsqueeze(2) + bigram[:, 0] + emit_m[:, 1].unsqueeze(1)
    alpha_hist[1] = alpha2
    for t in range(2, L):
        cand = alpha2.unsqueeze(3) + trigram[:, t - 2] + bigram[:, t - 1].unsqueeze(1)
        nxt = torch.logsumexp(cand, dim=1) + emit_m[:, t].unsqueeze(1)
        alpha2 = torch.where(real[:, t].view(B, 1, 1), nxt, alpha2)
        alpha_hist[t] = alpha2

    lp_ge2 = torch.logsumexp(alpha2.reshape(B, M * M), dim=-1)
    lp_eq1 = torch.logsumexp(emit_m[:, 0], dim=-1)
    log_z = torch.where(real[:, 1], lp_ge2, lp_eq1)

    beta_hist = [None] * L
    beta = torch.zeros((B, M, M), dtype=emit_m.dtype, device=emit_m.device)
    beta_hist[L - 1] = beta
    for t in range(L - 2, 0, -1):
        # Transition from (i, j) at step t to (j, k) at step t+1
        step_msg = (
            trigram[:, t - 1]
            + (bigram[:, t] + emit_m[:, t + 1].unsqueeze(1) + beta).unsqueeze(1)
        )
        nxt_beta = torch.logsumexp(step_msg, dim=3)  # (B, M, M) over k
        beta = torch.where(real[:, t + 1].view(B, 1, 1), nxt_beta, beta)
        beta_hist[t] = beta

    log_marg = emit_m.new_full((B, L, M), NEG)
    # Position 0 and 1 from step t=1 pair marginal
    pair1 = alpha_hist[1] + beta_hist[1] - log_z.view(B, 1, 1)  # (B, M, M)
    marg0_ge2 = torch.logsumexp(pair1, dim=2)
    marg0_eq1 = emit_m[:, 0] - lp_eq1.unsqueeze(1)
    log_marg[:, 0] = torch.where(real[:, 1].unsqueeze(1), marg0_ge2, marg0_eq1)
    log_marg[:, 1] = torch.where(real[:, 1].unsqueeze(1), torch.logsumexp(pair1, dim=1), NEG)

    for t in range(2, L):
        pair_t = alpha_hist[t] + beta_hist[t] - log_z.view(B, 1, 1)
        log_marg[:, t] = torch.where(real[:, t].unsqueeze(1), torch.logsumexp(pair_t, dim=1), NEG)

    log_marg = log_marg.masked_fill(~eff_mask, NEG)
    return log_marg, log_z


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    gold: torch.Tensor,
    sent_len: torch.Tensor,
    margin: float = 0.15,
    lambda_emit2: float = 0.35,
    lambda_emit1: float = 0.25,
    core_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Softmax-Margin Trigram CRF Loss + Auxiliary Stage-1 & Stage-2 Emission Cross-Entropy."""
    emit1 = outputs["emit1"]
    emit2 = outputs["emit2"]
    emit_p = outputs["emit_p"]
    mask_p = outputs["mask_p"]
    top_idx = outputs["top_idx"]
    bigram = outputs["bigram"]
    trigram = outputs["trigram"]
    gold_p = outputs["gold_p"]
    valid_tok = outputs["valid_tok"]

    # Hamming cost augmentation inside log-partition (Softmax-Margin)
    if margin > 0.0:
        wrong_slots = (top_idx != gold.long().unsqueeze(-1)) & mask_p
        emit_aug = emit_p + wrong_slots.to(emit_p.dtype) * margin
    else:
        emit_aug = emit_p

    lp = trigram_log_partition(emit_aug, bigram, trigram, mask_p)
    gs = trigram_gold_score(emit_p, bigram, trigram, mask_p, gold_p)
    # Normalize by total real tokens in the batch (not per-row average)
    total_toks = sent_len.float().sum().clamp(min=1.0)
    crf_loss = (lp - gs).sum() / total_toks

    ce_mask = valid_tok if core_mask is None else (valid_tok & core_mask)
    flat_mask = ce_mask.reshape(-1)
    flat_gold = gold.long().reshape(-1)[flat_mask]

    if flat_gold.numel() > 0:
        ce2 = F.cross_entropy(emit2.reshape(-1, emit2.shape[-1])[flat_mask], flat_gold)
        ce1 = F.cross_entropy(emit1.reshape(-1, emit1.shape[-1])[flat_mask], flat_gold)
    else:
        ce2 = emit2.sum() * 0.0
        ce1 = emit1.sum() * 0.0

    total = crf_loss + lambda_emit2 * ce2 + lambda_emit1 * ce1
    stats = {
        "crf_loss": float(crf_loss.detach()),
        "ce2_loss": float(ce2.detach()),
        "ce1_loss": float(ce1.detach()),
        "total_loss": float(total.detach()),
    }
    return total, stats
