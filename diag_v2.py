"""Where does the v2 model actually lose accuracy? Break errors down by cause."""
import collections
import json
import unicodedata

import torch

import pilot_balance as pb
from attn_model import AttnDiac
from vn import strip_all

CAND = json.load(open("data/cand.json"))
d = torch.load("ckpt_v2_m1m.pt", map_location="cpu")
m = AttnDiac(d["nchar"], d["max_c"], window=8, out_std=0.1,
             syl=192, dim=384, blocks=1).to(pb.DEV)
m.load_state_dict(d["model"])
m.eval()
char2id = d["char2id"]
forms = sorted(CAND)
forms_by_strip = {s: i for i, s in enumerate(forms)}


@torch.inference_mode()
def predict(sents):
    out = []
    for s in sents:
        toks = s.split()
        rows = []
        for w in toks:
            st = strip_all(w.lower())
            cs = CAND.get(st)
            if not cs:
                continue
            rows.append((st, len(cs)))
        if not rows:
            out.append(s.split())
            continue
        T = len(rows)
        L = max(len(r[0]) for r in rows)
        xc = torch.full((1, T, L), d["nchar"] - 1, dtype=torch.long)
        cl = torch.ones((1, T), dtype=torch.long)
        f = torch.zeros((1, T), dtype=torch.long)
        ln = torch.ones((1, T), dtype=torch.long)
        for t, (st, k) in enumerate(rows):
            ids = [char2id.get(c, 0) for c in st[:64]]
            xc[0, t, :len(ids)] = torch.tensor(ids)
            cl[0, t] = max(len(ids), 1)
            f[0, t] = forms_by_strip[st]
            ln[0, t] = k
        lg = pb.logits_of(m, xc.to(pb.DEV), cl.to(pb.DEV), ln.to(pb.DEV))
        pred = lg.argmax(-1)[0].tolist()
        words, j = [], 0
        for w in toks:
            st = strip_all(w.lower())
            cs = CAND.get(st)
            if not cs:
                words.append(w)
                continue
            ch = cs[pred[j]] if pred[j] < len(cs) else cs[0]
            j += 1
            words.append(ch.capitalize() if w[:1].isupper() else ch)
        out.append(words)
    return out


def analyze(name, path):
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
    preds = predict(lines)
    by_ncand = collections.defaultdict(lambda: [0, 0])   # n_candidates -> [ok, tot]
    by_pos = collections.defaultdict(lambda: [0, 0])     # sentence position bucket
    tone_conf = collections.Counter()                    # toneless-confusion pairs
    mod_conf = collections.Counter()
    tot = ok = tot_amb = ok_amb = 0
    for gold, pred in zip(lines, preds):
        n = len(gold)
        for i, (w_t, w_p) in enumerate(zip(gold.split(), pred)):
            st = strip_all(w_t.lower())
            cs = CAND.get(st)
            if not cs:
                continue
            tot += 1
            good = w_t.lower() == w_p.lower()
            ok += good
            k = min(len(cs), 9)
            by_ncand[k][0] += good
            by_ncand[k][1] += 1
            frac = i / max(n - 1, 1)
            b = 0 if frac < 0.2 else (4 if frac > 0.8 else 2) if frac > 0.5 else 1
            by_pos[b][0] += good
            by_pos[b][1] += 1
            if len(cs) > 1:
                tot_amb += 1
                ok_amb += good
                if not good:
                    # is the error a tone error, a modifier error, or both?
                    gt, pt = gt_tone(w_t), gt_tone(w_p)
                    gm, pm = gt_mod(w_t), gt_mod(w_p)
                    if gt != pt and gm == pm:
                        tone_conf[(gt, pt)] += 1
                    elif gm != pm and gt == pt:
                        mod_conf[(gm, pm)] += 1
                    else:
                        tone_conf[("MIXED", "MIXED")] += 1
    print(f"\n=== {name} ===")
    print(f"ALL {ok/tot:.4f} on {tot:,}   AMB {ok_amb/max(tot_amb,1):.4f} on {tot_amb:,}")
    print(" by number of candidates:")
    for k in sorted(by_ncand):
        a, t = by_ncand[k]
        print(f"   {k if k<9 else '9+':>3} cand: {a/t:.4f}  (n={t:,}, {t/tot:.1%} of tokens)")
    print(" by sentence position (0=start, 4=end):")
    for k in sorted(by_pos):
        a, t = by_pos[k]
        print(f"   pos {k}: {a/t:.4f}  (n={t:,})")
    print(" top tone confusions on ambiguous tokens:")
    for (a, b), c in tone_conf.most_common(8):
        print(f"   {a}->{b}: {c}")
    print(" top modifier confusions:")
    for (a, b), c in mod_conf.most_common(6):
        print(f"   {a}->{b}: {c}")


def gt_tone(w):
    from vn import tone_of
    return tone_of(w.lower()) or "ngang"


def gt_mod(w):
    from vn import decompose
    _, _, mods = decompose(w.lower())
    return "".join(mods) or "-"


if __name__ == "__main__":
    analyze("WIKI test", "data/te.txt_sample.txt")
    analyze("NEWS test", "data/te_news_sample.txt")
