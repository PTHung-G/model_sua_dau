"""CRF muc am tiet: moi vi tri co mot tap ung vien rieng, khong dung chung nhan.

Khac CRF muc ky tu: nhan khong phai 180 ky tu ma la chi so trong tap ung vien
cua tung am tiet (trung binh 7, toi da 24). Khac mo hinh factorized: co diem
chuyen tiep hoc duoc giua hai am tiet lien tiep, nen khong cham tung am tiet
doc lap nua.

Diem chuyen tiep dung phan ra hang thap: u(dang_trai) . v(dang_phai), voi u,v la
vector r chieu. Nho vay so tham so khong no theo binh phuong so dang am tiet.
"""
import torch
import torch.nn as nn

NEG = -1e9


def seg_pool(emb, ids, mask):
    """Trung binh embedding theo do dai that. ids (..., C), mask (..., C)."""
    w = mask.unsqueeze(-1).float()
    return (emb(ids) * w).sum(-2) / w.sum(-2).clamp(min=1.0)


class SylCRF(nn.Module):
    def __init__(self, nchar, base_ch, cand_ch, cand_ids, d=192, r=32,
                 hidden=192, layers=2, drop=0.2):
        super().__init__()
        self.d, self.r = d, r
        self.emb = nn.Embedding(nchar, d, padding_idx=0)
        self.register_buffer('base_ch', base_ch)
        self.register_buffer('cand_ch', cand_ch)
        self.register_buffer('cand_ids', cand_ids)          # (nbase, K) int32, -1 = khong co
        self.register_buffer('base_mask', (base_ch != 0).float())
        self.register_buffer('cand_mask', (cand_ch != 0).float())

        self.char_encoder = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.base_proj = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.cand_proj = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.lstm = nn.LSTM(d, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=drop if layers > 1 else 0.0)
        self.ctx = nn.Sequential(nn.Linear(2 * hidden, d), nn.GELU(), nn.Dropout(drop))
        self.emit_w = nn.Bilinear(d, d, 1)
        self.trans_p = nn.Linear(d, r)
        self.trans_q = nn.Linear(d, r)

    def encode_chars(self, ids):
        shape = ids.shape[:-1]
        flat = ids.reshape(-1, ids.shape[-1])
        lengths = flat.ne(0).sum(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(flat), lengths.clamp(min=1).cpu(), batch_first=True,
            enforce_sorted=False)
        _, hidden = self.char_encoder(packed)
        encoded = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return (encoded * lengths.ne(0).unsqueeze(-1)).reshape(*shape, self.d)

    def cand_embed(self, flat_ids):
        uniq, inverse = torch.unique(flat_ids, return_inverse=True)
        ue = self.cand_proj(self.encode_chars(self.cand_ch[uniq]))
        return ue[inverse]

    def forward(self, base_ids, sent_len):
        """base_ids (B,L) -> emit (B,L,K), trans (B,L-1,K,K), mask (B,L,K)."""
        B, L = base_ids.shape
        unique_base, inverse = torch.unique(base_ids, return_inverse=True)
        be = self.encode_chars(self.base_ch[unique_base])[inverse].reshape(B, L, self.d)
        packed = nn.utils.rnn.pack_padded_sequence(
            self.base_proj(be), sent_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(packed_h, batch_first=True, total_length=L)
        ctx = self.ctx(h)

        cid = self.cand_ids[base_ids].long()                 # (B,L,K)
        valid = cid >= 0
        K = cid.shape[-1]
        ce = self.cand_embed(cid.clamp(min=0).reshape(-1)).view(B, L, K, self.d)

        # x^T W y: project context once, not a K-fold expanded bilinear.
        # The expanded Bilinear backward can allocate (B*L*K, d, d).
        projected = torch.nn.functional.linear(ctx, self.emit_w.weight[0].T)
        emit = torch.einsum('bld,blkd->blk', projected, ce)
        if self.emit_w.bias is not None:
            emit = emit + self.emit_w.bias[0]
        emit = emit.masked_fill(~valid, NEG)

        u = self.trans_p(ce[:, :-1])                         # (B,L-1,K,r)
        v = self.trans_q(ce[:, 1:])
        trans = torch.einsum('blir,blkr->blik', u, v)        # (B,L-1,K,K)

        m = torch.arange(L, device=base_ids.device).unsqueeze(0) < sent_len.unsqueeze(1)
        mask = valid & m.unsqueeze(-1)
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
