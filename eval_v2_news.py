import json
import torch
import pilot_balance as pb
import evaluate as ev
from attn_model import AttnDiac

d = torch.load('ckpt_v2_m1m.pt', map_location='cpu')
m = AttnDiac(d['nchar'], d['max_c'], window=d['window'], out_std=d['out_std'],
             syl=192, dim=384, blocks=1).to(pb.DEV)
m.load_state_dict(d['model'])
m.eval()
cand = json.load(open('data/cand.json'))
char2id = d['char2id']


def pred(xc, cl, f, ln):
    return pb.logits_of(m, xc, cl, ln).argmax(-1)


for tag, path in [('NEWS test', 'data/te_news.txt')]:
    toks = ev.Tokens(pb.load_sents(path), cand, char2id)
    r = ev.evaluate(pred, toks, device=pb.DEV, bs=128)
    print(ev.fmt(tag, r), flush=True)
