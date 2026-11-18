"""Freeze capacity-control/hybrid experiments for Kaggle."""
import json
from pathlib import Path
files = ['attn.py','attn_model.py','pilot_arch.py','pilot_balance.py','evaluate.py',
         'vn.py','score_all.py','context_pilot.py','sentence_loss_pilot.py',
         'sentence_loss_epoch.py','span_rerank_pilot.py','hybrid_rerank_model.py',
         'hybrid_rerank_train.py','verify_hybrid_rerank.py','test_span_rerank.py','test_hybrid_rerank.py']
sources = {f: Path(f).read_text() for f in files}
setup = "import os,pathlib,torch\nassert torch.cuda.is_available()\nos.chdir('/kaggle/working')\n"
setup += 'sources=' + repr(sources) + '\n'
setup += '''for name,text in sources.items(): pathlib.Path(name).write_text(text)
data=[p.parent for p in pathlib.Path('/kaggle/input').rglob('cand.json') if (p.parent/'tr.txt').exists() and (p.parent/'va.txt').exists()]
weights=list(pathlib.Path('/kaggle/input').rglob('factorized_best.pt'))
assert len(data)==1 and len(weights)==1,(data,weights)
print('GPU',torch.cuda.get_device_name(0),flush=True)
'''
test = "import subprocess,sys\nsubprocess.run([sys.executable,'-m','unittest','test_hybrid_rerank','-v'],check=True)\n"
run = '''import json,subprocess,sys
for config in ('CONTROL','HYBRID'):
    subprocess.run([sys.executable,'-u','hybrid_rerank_train.py','--config',config,
                    '--data-dir',str(data[0]),'--checkpoint',str(weights[0]),
                    '--out-dir',f'hybrid_rerank_{config.lower()}_results',
                    '--steps','10000','--val-sents','3000','--lr','0.0001'],check=True)
    subprocess.run([sys.executable,'verify_hybrid_rerank.py','--output',f'hybrid_rerank_{config.lower()}_results','--base',str(weights[0])],check=True)
'''
cells=[dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=s.splitlines(True)) for s in (setup,test,run)]
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=cells)
dest=Path('kp_hybrid_rerank');dest.mkdir(exist_ok=True)
(dest/'hybrid_rerank.ipynb').write_text(json.dumps(nb))
(dest/'kernel-metadata.json').write_text(json.dumps(dict(id='tunhngphm/vi-diac-hybrid-rerank',title='vi-diac-hybrid-rerank',code_file='hybrid_rerank.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=False,dataset_sources=['tunhngphm/vi-diac-split-v2','tunhngphm/vi-diac-factorized-context-base'],competition_sources=[],kernel_sources=[])))
print('Notebook frozen')
