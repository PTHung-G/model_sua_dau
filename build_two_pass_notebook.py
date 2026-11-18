"""Freeze two-pass model experiment exact sources."""
import json
from pathlib import Path
files=['attn.py','attn_model.py','pilot_arch.py','pilot_balance.py','evaluate.py','vn.py','score_all.py','context_pilot.py','sentence_loss_pilot.py','sentence_loss_epoch.py','two_pass_pilot.py','test_two_pass.py']
sources={f:Path(f).read_text() for f in files}
setup="import os,pathlib,torch\nassert torch.cuda.is_available()\nos.chdir('/kaggle/working')\n"
setup+='sources='+repr(sources)+'\n'
setup+='''for name,text in sources.items(): pathlib.Path(name).write_text(text)
data=[p.parent for p in pathlib.Path('/kaggle/input').rglob('cand.json') if (p.parent/'tr.txt').exists() and (p.parent/'va.txt').exists()]
weights=list(pathlib.Path('/kaggle/input').rglob('factorized_best.pt'))
assert len(data)==1 and len(weights)==1,(data,weights)
print('GPU',torch.cuda.get_device_name(0),flush=True)
'''
run="""import subprocess,sys
subprocess.run([sys.executable,'-u','two_pass_pilot.py','--data-dir',str(data[0]),'--checkpoint',str(weights[0]),'--out-dir','two_pass_results','--steps','10000','--val-sents','3000','--lr','0.0003'],check=True)
"""
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)} for s in (setup,run)]}
Path('kp_two_pass/two_pass.ipynb').write_text(json.dumps(nb));print('Notebook frozen')
