"""Build self-contained full-training notebook from exact local source snapshots."""
import json
from pathlib import Path

files = ['attn.py', 'attn_model.py', 'pilot_balance.py', 'evaluate.py', 'vn.py',
         'pilot_arch.py', 'train_factorized_full.py']
sources = {f: Path(f).read_text() for f in files}
setup = '''import os, pathlib, torch
assert torch.cuda.is_available(), 'A Kaggle GPU is required'
os.chdir('/kaggle/working')
'''
setup += 'sources=' + repr(sources) + '\n'
setup += '''for name, text in sources.items(): pathlib.Path(name).write_text(text)
roots=[p.parent for p in pathlib.Path('/kaggle/input').rglob('cand.json') if (p.parent/'tr.txt').exists() and (p.parent/'va.txt').exists()]
assert len(roots)==1,roots
DATA=str(roots[0])
print('GPU',torch.cuda.get_device_name(0),'DATA',DATA,flush=True)
'''
run = '''import subprocess, sys
subprocess.run([sys.executable,'-u','train_factorized_full.py','--data-dir',DATA,'--out-dir','factorized_full_results','--steps','120000','--val-sents','10000','--eval-every','10000','--save-every','5000'],check=True)
'''
nb = {'nbformat':4,'nbformat_minor':5,
      'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
      'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
                'source':s.splitlines(True)} for s in (setup,run)]}
Path('kp_factorized_full/train.ipynb').write_text(json.dumps(nb))
print('Notebook exact source snapshot generated')
