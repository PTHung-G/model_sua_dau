"""Strict artifact checks for the two-pass pilot."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
import two_pass_pilot as tp

def main(directory,smoke=False):
 root=Path(directory);r=json.loads((root/'results.json').read_text());assert r['status']=='complete'
 v=r['arms']['two_pass'];torch.set_num_threads(4)
 order=np.load(root/'epoch_order.npy');m=r['provenance']['train']
 assert len(np.unique(order))==len(order)
 assert hashlib.sha256(order.tobytes()).hexdigest()==m['permutation_sha256']
 visited=v['visited_source_rows'];assert hashlib.sha256(order[:visited].tobytes()).hexdigest()==m['planned_prefix_sha256']
 assert v['budget_complete'] and v['optimizer_steps']+v['skipped_batches']==v['planned_batches']
 if not smoke:assert v['optimizer_steps']==10000 and visited==640000 and v['skipped_batches']==0
 original_path=Path('kaggle_factorized_full_output/factorized_full_results/factorized_best.pt')
 original=torch.load(original_path,map_location='cpu',weights_only=False)
 assert tp.se.sha256_file(original_path)==r['provenance']['checkpoint']['sha256']
 proofs=[]
 for kind,step in [('best',v['best']['step']),('latest',v['consumed_batches'])]:
  path=root/f'two_pass_{kind}.pt';model,p=tp.load_two_pass_checkpoint(path,device='cpu')
  assert p['step']==step and p['epoch_cursor']==min(step*64,len(order))
  assert p['provenance']==r['provenance']
  assert all(torch.equal(model.base.state_dict()[key],value) for key,value in original['model'].items())
  assert all(not q.requires_grad for q in model.base.parameters())
  assert sum(q.numel() for q in model.parameters())==p['params']
  assert sum(q.numel() for q in model.parameters() if q.requires_grad)==p['trainable_params']
  model.train();assert not model.base.training;model.eval()
  raw=[['toi','di','hoc'],['toi']*17]
  one=list(tp.cp.raw_batches(raw[:1],p,1))[0][-1];batch=list(tp.cp.raw_batches(raw,p,2))[0][-1]
  with torch.inference_mode():
   a=model.logits(*one)[0];b=model.logits(*batch)[0,:len(a)]
  delta=float((a-b).abs().max());assert torch.allclose(a,b,atol=1e-4,rtol=1e-5),delta
  assert torch.equal(a.argmax(-1),b.argmax(-1))
  for key in ('optimizer','scheduler','rng','corrector_config','frozen_base_metadata'):assert key in p
  proofs.append(dict(kind=kind,step=step,params=p['params'],trainable_params=p['trainable_params'],frozen_generator_identical=True,padding_max_delta=delta,sha256=tp.se.sha256_file(path)))
 proof=dict(verified=True,unique_permutation=True,visited_unique_rows=visited,optimizer_steps=v['optimizer_steps'],checkpoints=proofs)
 (root/'verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--smoke',action='store_true');args=p.parse_args();main(args.directory,args.smoke)
