"""Fast replacement for pilot_balance.make_batches.

The original builds every batch with a doubly nested Python loop (batch x syllable),
calling strip_all()/dict lookups ~2,200 times per batch. Profiling showed that cost
~50 ms/batch, i.e. 30% of every training step for the small model, all single-threaded
while the GPU idles.

Changes here, none of which alter what the model sees:
  1. Every sentence is encoded ONCE into flat arrays (sentence -> stride block), so a
     batch is a single numpy gather instead of a Python traversal.
  2. Length bucketing: sentences are grouped into fixed-size shuffling "chunks" and
     sorted by length inside a chunk only. A batch is drawn from one chunk, so its
     rows are similar in length and far less padding is computed and masked out.
     Within a window of `chunk` sentences the sampling distribution is unchanged.
  3. Windows of a sentence are padded per batch by slicing a preallocated buffer
     rather than calling torch.tensor() once per syllable.

Output tuple and dtypes are identical to make_batches, so training code is unchanged.
"""
import numpy as np
import torch

from vn import strip_all

PAD_ID_OFFSET = 1  # pad char id is nchar - 1; stored as nchar via +1, fixed at emit


class FastCorpus:
    def __init__(self, sents, cand, char2id, max_c, nchar, forms, chunk=8192):
        self.nchar = nchar
        self.max_c = max_c
        pad_id = nchar - 1
        # ---- encode every sentence once into flat per-sentence blocks ----
        keep = []
        for i, ws in enumerate(sents):
            has = False
            for w in ws:
                if cand.get(strip_all(w.lower())):
                    has = True
                    break
            if has:
                keep.append(i)

        lens, starts, stride = [], [], 1
        enc = []  # per sentence: (T, Lmax) char ids, (T,) cl, (T,) form, (T,) gold, (T,) ln, (T,) ok, T
        for i in keep:
            ws = sents[i]
            cs_, cs_k, cs_ci, cs_char = [], [], [], []
            for w in ws:
                wl = w.lower()
                s = strip_all(wl)
                cs = cand.get(s)
                if not cs:
                    continue
                ci = cs.index(wl) if wl in cs else -1
                cs_.append(s); cs_k.append(len(cs)); cs_ci.append(ci)
            T = len(cs_)
            L = max(len(s) for s in cs_)
            # Fill with the pad id, not 0: id 0 means "unknown character" and the
            # original generator padded with nchar-1. Using 0 here silently feeds
            # the embedding table a different token and changes what the GRU sees.
            ids = np.full((T, L), pad_id, dtype=np.int16)
            cl = np.ones(T, dtype=np.int16)
            f = np.zeros(T, dtype=np.int32)
            y = np.zeros(T, dtype=np.int16)
            ln = np.ones(T, dtype=np.int16)
            ok = np.zeros(T, dtype=bool)
            for t, (s, k, ci) in enumerate(zip(cs_, cs_k, cs_ci)):
                u = s[:64]
                n = len(u)
                ids[t, :n] = [char2id.get(c, 0) for c in u]
                cl[t] = max(n, 1)
                f[t] = forms[s]
                y[t] = max(ci, 0)
                ln[t] = k
                ok[t] = ci >= 0
            enc.append((ids, cl, f, y, ln, ok, T))
            stride = max(stride, L)
            lens.append(T)
        # pad every sentence block to a common stride so a batch is one gather
        for j, (ids, cl, f, y, ln, ok, T) in enumerate(enc):
            blk = np.full((T, stride), pad_id, dtype=np.int16)
            blk[:, :ids.shape[1]] = ids
            enc[j] = (blk, cl, f, y, ln, ok, T)
        self.enc = enc
        self.lens = np.asarray(lens, dtype=np.int32)
        # per-sentence true character width, so a batch can crop to what it needs
        self.clen = np.asarray([e[0].shape[1] for e in enc], dtype=np.int32)
        self.stride = stride
        self.Tmax = int(self.lens.max()) if lens else 0
        self.pad_id = pad_id

        # ---- length bucketing: sort inside chunks, keep a shuffled chunk order ----
        order = np.argsort(self.lens, kind="stable")
        self.chunks = [order[i:i + chunk] for i in range(0, len(order), chunk)]
        self.small = int((self.lens < 2).sum())

    def batch(self, idx, device="cpu"):
        """Build one batch from sentence indices. Padding only up to this batch's needs."""
        T = int(max(self.lens[i] for i in idx))
        # Crop the character axis too. Using the global stride here wastes ~57% of the
        # GRU/attention work on character padding, which more than cancels the data win.
        L = int(max(self.clen[i] for i in idx))
        B = len(idx)
        nchar = self.nchar
        xc = np.full((B, T, L), self.pad_id, dtype=np.int16)
        cl = np.ones((B, T), dtype=np.int16)
        f = np.zeros((B, T), dtype=np.int32)
        y = np.zeros((B, T), dtype=np.int16)
        ln = np.ones((B, T), dtype=np.int16)
        ok = np.zeros((B, T), dtype=bool)
        for b, i in enumerate(idx):
            blk, c_, f_, y_, l_, o_, sT = self.enc[i]
            w = min(L, self.clen[i])
            xc[b, :sT, :w] = blk[:sT, :w]
            cl[b, :sT] = c_
            f[b, :sT] = f_
            y[b, :sT] = y_
            ln[b, :sT] = l_
            ok[b, :sT] = o_
        # Rows shorter than T are pure padding: the original generator never emitted
        # them because it sized T per batch. Mark them invalid (mk=False) or the
        # model treats empty slots as real syllables and the loss is diluted.
        mk = np.zeros((B, T), dtype=bool)
        for b, i in enumerate(idx):
            mk[b, :self.enc[i][6]] = True
        t = lambda a: torch.from_numpy(a).to(device=device, non_blocking=True).long()
        return (t(xc), t(cl), t(f), t(y), t(ln),
                torch.from_numpy(mk).to(device), torch.from_numpy(ok).to(device))


def make_batches_fast(corpus, bs, rng, chunks_per_batch=8, chunk_order=None):
    """Endless batch stream. Yields the same 7-tensor tuple as make_batches.

    A batch is drawn from `chunks_per_batch` different length-buckets, not just one.
    Drawing from a single bucket gave every batch the same sentence length (~11
    syllables, ~728 tokens) instead of the corpus average (~940), so each optimizer
    step saw 23% less data than plain sampling. Mixing buckets keeps padding low
    while restoring the natural length spread.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc = len(corpus.chunks)
    queue = []
    while True:
        if len(queue) < bs:
            # top up the buffer from fresh, shuffled buckets
            order = rng.permutation(nc) if chunk_order is None else chunk_order
            for ci in order:
                ch = corpus.chunks[ci].copy()
                rng.shuffle(ch)
                queue.extend(ch.tolist())
        idx = queue[:bs]
        del queue[:bs]
        xc, cl, f, y, ln, mk, ok = corpus.batch(idx, device)
        yield xc, cl, f, y, ln, mk, ok


def enumerate_batches(corpus, bs, device="cpu"):
    """Fixed pass in bucketed order (used for validation if needed)."""
    for idx in np.concatenate(corpus.chunks):
        yield corpus.batch([idx], device)
