"""Candidate-pair CRF with an explicit punctuation-sequence encoder.

Only change from sylmodel_attention: the model reads the whole gap sequence
(punctuation and spacing before every word, plus the trailing gap) with a small
bidirectional GRU. Gap context enters the word representations and the CRF
transition gate. Punctuation is never predicted; it is copied from the input.
"""
import torch
import torch.nn as nn

NEG = -1e9


class PunctSylCRF(nn.Module):
    def __init__(self, nchar, base_ch, cand_ch, cand_ids, npunct,
                 d=128, r=32, hidden=136, layers=2, drop=0.2,
                 gap_dim=16, gap_hidden=32):
        super().__init__()
        self.d, self.r = d, r
        self.emb = nn.Embedding(nchar, d, padding_idx=0)
        self.register_buffer('base_ch', base_ch)
        self.register_buffer('cand_ch', cand_ch)
        self.register_buffer('cand_ids', cand_ids)

        self.char_encoder = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.base_proj = nn.Sequential(nn.Linear(d, d), nn.GELU())

        # Punctuation/gap stream: one gap before each word plus one trailing gap.
        self.gap_dim = gap_dim
        self.gap_emb = nn.Embedding(npunct, gap_dim, padding_idx=0)
        self.gap_encoder = nn.GRU(gap_dim, gap_hidden, batch_first=True,
                                  bidirectional=True)
        # Zero-initialized so training starts identical to the previous model.
        self.gap_proj = nn.Linear(2 * gap_hidden, d, bias=False)
        nn.init.zeros_(self.gap_proj.weight)

        self.case_emb = nn.Embedding(5, 8, padding_idx=0)
        self.case_proj = nn.Linear(8, d, bias=False)
        nn.init.zeros_(self.case_proj.weight)
        self.cand_proj = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.lstm = nn.LSTM(d, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=drop if layers > 1 else 0.0)
        self.ctx = nn.Sequential(nn.Linear(2 * hidden, d), nn.GELU(), nn.Dropout(drop))
        self.attn_norm = nn.LayerNorm(d)
        self.attn_qkv = nn.Linear(d, 3 * 32)
        self.attn_out = nn.Linear(32, d)
        nn.init.zeros_(self.attn_out.weight)
        nn.init.zeros_(self.attn_out.bias)
        self.attn_drop = nn.Dropout(drop)
        self.emit_w = nn.Bilinear(d, d, 1)
        self.trans_p = nn.Linear(d, r)
        self.trans_q = nn.Linear(d, r)
        self.edge_gate = nn.Linear(2 * d + 2 * gap_hidden, r)
        nn.init.zeros_(self.edge_gate.weight)
        nn.init.zeros_(self.edge_gate.bias)

    def attend(self, ctx, sent_len):
        B, L, _ = ctx.shape
        valid = torch.arange(L, device=ctx.device)[None, :] < sent_len[:, None]
        qkv = self.attn_qkv(self.attn_norm(ctx)).reshape(B, L, 3, 2, 16)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        z = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=valid[:, None, None, :], dropout_p=0.0)
        z = z.transpose(1, 2).reshape(B, L, 32)
        return (ctx + self.attn_drop(self.attn_out(z))) * valid.unsqueeze(-1)

    def encode_chars(self, ids):
        shape = ids.shape[:-1]
        flat = ids.reshape(-1, ids.shape[-1])
        # Zero can occur inside a real spelling because the current inventory
        # encoder also uses it for unknown characters. Counting nonzero entries
        # would truncate known characters after such a hole. Retain every slot
        # through the final known character (trailing unknowns remain ambiguous
        # until the inventory encoder assigns UNK a separate nonzero id).
        slots = torch.arange(1, flat.shape[-1] + 1, device=flat.device)
        lengths = (flat.ne(0) * slots).amax(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(flat.long()), lengths.clamp(min=1).cpu(), batch_first=True,
            enforce_sorted=False)
        _, hidden = self.char_encoder(packed)
        encoded = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return (encoded * lengths.ne(0).unsqueeze(-1)).reshape(*shape, self.d)

    def cand_embed(self, flat_ids):
        uniq, inverse = torch.unique(flat_ids, return_inverse=True)
        ue = self.cand_proj(self.encode_chars(self.cand_ch[uniq].long()))
        return ue[inverse]

    def gap_context(self, gaps, sent_len, width):
        """gaps (B,L+1) -> (B,L+1,2*gap_hidden); padded positions are zeroed."""
        if gaps.ndim != 2 or gaps.shape != (sent_len.numel(), width):
            raise ValueError('gaps must have shape (batch, word_width + 1)')
        if torch.any(sent_len < 1) or torch.any(sent_len >= width):
            raise ValueError('word lengths must be in [1, gap_width - 1]')
        if torch.any(gaps < 0) or torch.any(gaps >= self.gap_emb.num_embeddings):
            raise ValueError('gap id outside embedding vocabulary; sparse IDs require max_id + 1 entries')
        valid = torch.arange(width, device=gaps.device)[None, :] <= sent_len[:, None]
        # Masking the OUTPUT is insufficient for a bidirectional GRU: its reverse
        # state would already have consumed padding and contaminated real gaps.
        packed = nn.utils.rnn.pack_padded_sequence(
            self.gap_emb(gaps.long()), (sent_len + 1).cpu(), batch_first=True,
            enforce_sorted=False)
        packed_h, _ = self.gap_encoder(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_h, batch_first=True, total_length=width)
        return h * valid.unsqueeze(-1)

    def forward(self, base_ids, sent_len, input_features=None, gaps=None, *, return_unary_features=False):
        """base_ids (B,L); gaps (B,L+1) punctuation-gap IDs from the input only."""
        B, L = base_ids.shape
        unique_base, inverse = torch.unique(base_ids, return_inverse=True)
        be = self.encode_chars(self.base_ch[unique_base].long())[inverse].reshape(B, L, self.d)
        encoded = self.base_proj(be)

        if input_features is not None:
            encoded = encoded + self.case_proj(self.case_emb(input_features))
        if gaps is None:
            raise ValueError('Punctuation gaps are required by PunctSylCRF')
        gc = self.gap_context(gaps, sent_len, L + 1)          # (B,L+1,2gh)
        encoded = encoded + self.gap_proj(gc[:, :L])

        packed = nn.utils.rnn.pack_padded_sequence(
            encoded, sent_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(packed_h, batch_first=True, total_length=L)
        ctx = self.attend(self.ctx(h), sent_len)

        cid = self.cand_ids[base_ids].long()
        valid = cid >= 0
        K = cid.shape[-1]
        ce = self.cand_embed(cid.clamp(min=0).reshape(-1)).view(B, L, K, self.d)
        projected = torch.nn.functional.linear(ctx, self.emit_w.weight[0].T)
        emit = torch.einsum('bld,blkd->blk', projected, ce)
        if self.emit_w.bias is not None:
            emit = emit + self.emit_w.bias[0]
        emit = emit.masked_fill(~valid, NEG)

        u = self.trans_p(ce[:, :-1])
        v = self.trans_q(ce[:, 1:])
        # The gap between word t and t+1 is gap index t+1.
        edge_context = torch.cat((ctx[:, :-1], ctx[:, 1:], gc[:, 1:L]), dim=-1)
        gate = 1.0 + torch.tanh(self.edge_gate(edge_context))
        trans = torch.einsum('blir,blkr->blik', u * gate.unsqueeze(2), v)

        m = torch.arange(L, device=base_ids.device).unsqueeze(0) < sent_len.unsqueeze(1)
        mask = valid & m.unsqueeze(-1)
        if return_unary_features:
            return emit, trans, mask, ctx, ce, cid
        return emit, trans, mask


def log_partition(emit, trans, mask):
    B, L, K = emit.shape
    alpha = emit[:, 0]
    for t in range(1, L):
        nxt = torch.logsumexp(alpha.unsqueeze(2) + trans[:, t - 1], dim=1) + emit[:, t]
        real = mask[:, t].any(1)
        alpha = torch.where(real.unsqueeze(1), nxt, alpha)
    return torch.logsumexp(alpha, dim=1)


def gold_score(emit, trans, mask, gold):
    B, L, K = emit.shape
    g0 = gold[:, :1]
    s = emit[:, 0].gather(1, g0).squeeze(1)
    for t in range(1, L):
        e = emit[:, t].gather(1, gold[:, t:t + 1]).squeeze(1)
        row = trans[:, t - 1].gather(
            1, gold[:, t - 1].view(B, 1, 1).expand(B, 1, K)).squeeze(1)
        tr = row.gather(1, gold[:, t:t + 1]).squeeze(1)
        real = mask[:, t].any(1)
        s = s + torch.where(real, e + tr, torch.zeros_like(e))
    return s


def nll(emit, trans, mask, gold, sent_len):
    lp = log_partition(emit, trans, mask)
    gs = gold_score(emit, trans, mask, gold)
    return ((lp - gs) / sent_len.float()).mean()


@torch.no_grad()
def viterbi(emit, trans, mask):
    B, L, K = emit.shape
    score = emit[:, 0].clone()
    ptr = torch.zeros(B, L, K, dtype=torch.long, device=emit.device)
    for t in range(1, L):
        best, idx = (score.unsqueeze(2) + trans[:, t - 1]).max(dim=1)
        real = mask[:, t].any(1)
        score = torch.where(real.unsqueeze(1), best + emit[:, t], score)
        ptr[:, t] = idx
    cur = score.argmax(1)
    out = torch.zeros(B, L, dtype=torch.long, device=emit.device)
    for t in range(L - 1, -1, -1):
        w = mask[:, t].any(1)
        out[:, t] = torch.where(w, cur, out[:, t])
        cur = torch.where(w, ptr[:, t].gather(1, cur.unsqueeze(1)).squeeze(1), cur)
    return out
