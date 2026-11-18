"""Verify overfit checkpoint full reload and immutable original generator."""
import argparse
import json
from pathlib import Path
import torch
import hybrid_rerank_model as hm


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--base',required=True)
    a=p.parse_args();out=Path(a.output);original=torch.load(a.base,map_location='cpu',weights_only=False)
    report={}
    for kind in ('best','latest'):
        model,payload=hm.load_reranker_checkpoint(out/f'overfit_hybrid_{kind}.pt')
        assert all(torch.equal(t,model.base.state_dict()[name]) for name,t in original['model'].items())
        assert all(not q.requires_grad for q in model.base.parameters())
        assert not model.base.training
        assert all(torch.isfinite(t).all() for t in payload['model'].values() if t.is_floating_point())
        report[kind]=dict(step=payload['step'],strict_load=True,frozen_base_equal=True,finite=True)
    (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
