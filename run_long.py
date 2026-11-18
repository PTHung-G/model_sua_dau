import torch, torch.nn.functional as F
from data import load_sents
from tiny import build, batches, CharTiny, evaluate
torch.set_num_threads(2)
tr = load_sents("train.txt", 20000); te = load_sents("test.txt", 2000)
P = build(tr, te, min_count=2)
model = CharTiny(P["nchar"], P["max_c"], hidden=96, syl=48)
print("params", sum(p.numel() for p in model.parameters()), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)
for e in range(8):
    model.train(); tot = 0; nb = 0
    for xc, cl, y, ln, mk in batches(P["tr"], 64, seed=e):
        lg = model(xc, cl).masked_fill(
            torch.arange(model.out.out_features)[None, None, :] >= ln[..., None], -1e9)
        loss = F.cross_entropy(lg[mk], y[mk]); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += loss.item(); nb += 1
    acc, aa, au, n = evaluate(model, P["te"])
    print(f"ep{e+1} loss={tot/nb:.3f} acc={acc:.4f} amb={aa:.4f}", flush=True)
