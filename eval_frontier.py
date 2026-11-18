"""Public GRID diagnostics using public inputs, not an official leaderboard.

Case-insensitive NFC punctuation-core accuracy; stem and exact candidate coverage
are separate. Legacy checkpoints use unpacked chars and singleton inference.
New pilot training and evaluation are independent of this compatibility script.
"""
import argparse
import json

import torch

import pilot_balance as pb
import score_all as sa

REGISTERS = ["business_55", "formal_72", "conversational_300", "literary_800"]
LEVEL = {"business_55": "business", "formal_72": "formal",
         "conversational_300": "conv", "literary_800": "literary"}
CAND = {}


def load_grid():
    return {cfg: sa.load_pairs("GRID:" + cfg) for cfg in REGISTERS}


def predict_sentences(model, sents, cand, char2id, nchar, forms_by_strip, max_c):
    sa.CAND.clear()
    sa.CAND.update(cand)
    sa.FORMS_BY_STRIP.clear()
    sa.FORMS_BY_STRIP.update(forms_by_strip)
    return [" ".join(row) for row in sa.predict(model, sents, char2id, nchar, max_c)]


is_scoring_token = sa.is_scoring_token
core_of = sa.core


def word_scores(pairs, preds):
    """Compatibility interface: case-insensitive hits and stem coverage flags."""
    if len(pairs) != len(preds):
        raise ValueError("Prediction sentence count differs from gold")
    hits, covered = [], []
    for (_, target), pred in zip(pairs, preds):
        pp = pred.split()
        for i, token in enumerate(target.split()):
            if not is_scoring_token(token):
                continue
            gold = sa.normalized_core(token)
            hits.append(int(i < len(pp) and gold == sa.normalized_core(pp[i])))
            covered.append(sa.strip_all(gold) in CAND)
    return hits, covered


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt")
    ap.add_argument("--variant", default="m1m", choices=["small", "m1m"])
    ap.add_argument("--coverage-only", action="store_true")
    ap.add_argument("--out", default="frontier_eval.json")
    args = ap.parse_args()
    cand = json.load(open("data/cand.json"))
    CAND.clear()
    CAND.update(cand)
    forms = {s: i for i, s in enumerate(sorted(cand))}
    chars = sorted({c for s in cand for c in s})
    char2id = {c: i for i, c in enumerate(chars, 1)}
    nchar, max_c = len(chars) + 2, max(map(len, cand.values()))
    model = None
    if not args.coverage_only:
        if not args.ckpt:
            ap.error("--ckpt is required unless --coverage-only is used")
        import big_model
        from attn_model import AttnDiac
        d = torch.load(args.ckpt, map_location=pb.DEV)
        char2id = d.get("char2id", char2id)
        nchar, max_c = d.get("nchar", nchar), d.get("max_c", max_c)
        model = AttnDiac(nchar, max_c, window=d.get("window", 8),
                         out_std=d.get("out_std", 0.1), packed_chars=False,
                         **big_model.VARIANTS[args.variant]).to(pb.DEV)
        model.load_state_dict(d.get("model", d))
        model.eval()
    sa.CAND.clear()
    sa.CAND.update(cand)
    rows = []
    for cfg, pairs in load_grid().items():
        preds = (predict_sentences(model, [inp for inp, _ in pairs], cand,
                                  char2id, nchar, forms, max_c)
                 if model is not None else ["" for _ in pairs])
        r = sa.score([gold for _, gold in pairs], [p.split() for p in preds])
        row = dict(register=LEVEL[cfg], n_sent=len(pairs), n_tok=r["n_scoring"],
                   coverage=r["stem_cov"], stem_cov=r["stem_cov"],
                   exact_cov=r["exact_cov"], n_covered=r["n_covered"],
                   n_exact_covered=r["n_exact_covered"],
                   acc_cov=r["syll_acc"] if model is not None else None,
                   acc_exact=r["exact_acc"] if model is not None else None,
                   acc_all=r["all_syll_acc"] if model is not None else None,
                   inference_protocol="legacy unpacked_chars; sentence batch_size=1")
        rows.append(row)
        print(row, flush=True)
    json.dump(rows, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
