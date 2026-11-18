import torch, torch.nn as nn

class CharTransformer(nn.Module):
    """Transformer encoder mức ký tự, dự đoán 1 nhãn cho mỗi vị trí."""
    def __init__(self, V, d=128, layers=6, heads=8, ff=4, maxlen=256):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.pos = nn.Embedding(maxlen, d)
        el = nn.TransformerEncoderLayer(d, heads, ff*d, batch_first=True,
                                        norm_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(el, layers)
        self.out = nn.Linear(d, V)

    def forward(self, x, **kw):
        B, L = x.shape
        p = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.emb(x) + self.pos(p)
        pad = x.eq(0)
        h = self.enc(h, src_key_padding_mask=pad)
        return self.out(h)


class CRF(nn.Module):
    """CRF chuỗi tuyến tính, log-space, có xử lý độ dài thật."""
    def __init__(self, V):
        super().__init__()
        self.V = V
        self.trans = nn.Parameter(torch.zeros(V, V))
        self.start = nn.Parameter(torch.zeros(V))
        self.end = nn.Parameter(torch.zeros(V))

    def _logz(self, e, lens):
        B, L, V = e.shape
        dev = e.device
        alpha = self.start.unsqueeze(0) + e[:, 0]
        ts = torch.arange(L, device=dev).unsqueeze(0)
        real = ts < lens.unsqueeze(1)
        for t in range(1, L):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans.unsqueeze(0), dim=1) + e[:, t]
            alpha = torch.where(real[:, t].unsqueeze(1), nxt, alpha)
        return torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)

    def _gold(self, e, y, lens):
        B, L, V = e.shape
        dev = e.device
        ts = torch.arange(L, device=dev).unsqueeze(0)
        real = ts < lens.unsqueeze(1)
        s = self.start[y[:, 0]] + e[:, 0].gather(1, y[:, :1]).squeeze(1)
        for t in range(1, L):
            prev, cur = y[:, t-1], y[:, t]
            step = self.trans[prev, cur] + e[:, t].gather(1, cur.unsqueeze(1)).squeeze(1)
            s = s + torch.where(real[:, t], step, torch.zeros_like(step))
        last = y.gather(1, (lens-1).unsqueeze(1)).squeeze(1)
        return s + self.end[last]

    def nll(self, e, y, lens, mask):
        logz = self._logz(e, lens)
        gold = self._gold(e, y, lens)
        return ((logz - gold) / lens.float()).mean()

    @torch.no_grad()
    def viterbi(self, e, lens):
        B, L, V = e.shape
        dev = e.device
        score = self.start.unsqueeze(0) + e[:, 0]
        ptr = torch.zeros(B, L, V, dtype=torch.long, device=dev)
        ts = torch.arange(L, device=dev).unsqueeze(0)
        real = ts < lens.unsqueeze(1)
        for t in range(1, L):
            cand = score.unsqueeze(2) + self.trans.unsqueeze(0)
            best, idx = cand.max(dim=1)
            nxt = best + e[:, t]
            score = torch.where(real[:, t].unsqueeze(1), nxt, score)
            ptr[:, t] = idx
        score = score + self.end.unsqueeze(0)
        cur = score.argmax(1)
        out = torch.zeros(B, L, dtype=torch.long, device=dev)
        for t in range(L - 1, -1, -1):
            w = real[:, t]
            out[:, t] = torch.where(w, cur, out[:, t])
            nxt = ptr[:, t].gather(1, cur.unsqueeze(1)).squeeze(1)
            cur = torch.where(w, nxt, cur)
        return out


class BiLSTMCRF(nn.Module):
    def __init__(self, V, emb=256, hidden=128, layers=2):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=0.2 if layers > 1 else 0.0)
        self.out = nn.Linear(2*hidden, V)
        self.crf = CRF(V)

    def emissions(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.out(h)

    def forward(self, x, y=None, lens=None, mask=None):
        e = self.emissions(x)
        if y is None:
            return self.crf.viterbi(e, lens)
        return e, self.crf.nll(e, y, lens, mask)
