"""Strict-reload best/latest checkpoints and verify frozen original tensors."""
import argparse
import json
from pathlib import Path
import torch
import hybrid_rerank_model as hm
import context_pilot as cp
import sentence_loss_epoch as se


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--base',default=str(Path(__file__).resolve().parent/'kaggle_factorized_full_output/factorized_full_results/factorized_best.pt'))
    args=p.parse_args()
    torch.set_num_threads(4)
    base=torch.load(args.base,map_location='cpu',weights_only=False)
    out=Path(args.output)
    report={}
    for kind in ('best','latest'):
        model,payload=hm.load_reranker_checkpoint(out/f'hybrid_rerank_{kind}.pt',device='cpu')
        for name,value in base['model'].items():
            if not torch.equal(value,model.base.state_dict()[name]):
                raise AssertionError(f'Frozen base changed: {name}')
        batch,_=se.collate([['tôi','yêu','việt','nam'],['học','sinh']],payload,'cpu')
        with torch.inference_mode():
            a=model.logits(batch[0],batch[1],batch[2],batch[4],batch[5])
        assert torch.isfinite(a).all()
        assert not model.base.training
        assert all(not q.requires_grad for q in model.base.parameters())
        report[kind]=dict(strict_load=True,frozen_base_equal=True,finite=True,
                          step=payload['step'],counts=model.parameter_counts())
        del model,payload
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    main()
