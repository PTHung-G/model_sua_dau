"""Build a separate Kaggle capacity/one-pass learning diagnostic."""
import json
from pathlib import Path
names=['attn.py','attn_model.py','pilot_arch.py','pilot_balance.py','evaluate.py','vn.py','score_all.py',
       'context_pilot.py','sentence_loss_pilot.py','sentence_loss_epoch.py','hybrid_rerank_model.py',
       'hybrid_rerank_train.py','hybrid_full_epoch_train.py','fixed_learning_metrics.py',
       'overfit_hybrid.py','verify_overfit_hybrid.py','verify_hybrid_rerank.py','test_hybrid_rerank.py','test_overfit_hybrid.py']
sources={n:Path(n).read_text() for n in names}
setup="import os,pathlib,torch\nassert torch.cuda.is_available()\nos.chdir('/kaggle/working')\nsources="+repr(sources)+"\n"
setup+='''for name,text in sources.items():pathlib.Path(name).write_text(text)
data=[p.parent for p in pathlib.Path('/kaggle/input').rglob('cand.json') if (p.parent/'tr.txt').exists() and (p.parent/'va.txt').exists()]
weights=list(pathlib.Path('/kaggle/input').rglob('factorized_best.pt'))
assert len(data)==len(weights)==1
print('GPU',torch.cuda.get_device_name(0),flush=True)
'''
run='''import subprocess,sys,json
subprocess.run([sys.executable,'-m','unittest','test_hybrid_rerank','test_overfit_hybrid','-v'],check=True)
subprocess.run([sys.executable,'-u','overfit_hybrid.py','--data-dir',str(data[0]),'--checkpoint',str(weights[0]),'--out-dir','overfit_results'],check=True)
subprocess.run([sys.executable,'verify_overfit_hybrid.py','--output','overfit_results','--base',str(weights[0])],check=True)
diagnostic=json.loads(pathlib.Path('overfit_results/results.json').read_text())
passed=diagnostic['final']['supervised_ce'] <= .5*diagnostic['initial']['supervised_ce']
pathlib.Path('overfit_gate.json').write_text(json.dumps(dict(passed=passed,initial_ce=diagnostic['initial']['supervised_ce'],final_ce=diagnostic['final']['supervised_ce'])))
if passed:
    subprocess.run([sys.executable,'-u','hybrid_full_epoch_train.py','--config','HYBRID','--data-dir',str(data[0]),'--checkpoint',str(weights[0]),'--out-dir','hybrid_full_epoch_results','--steps','0','--lr','0.0001','--val-sents','3000'],check=True)
    subprocess.run([sys.executable,'verify_hybrid_rerank.py','--output','hybrid_full_epoch_results','--base',str(weights[0])],check=True)
else:
    print('OVERFIT_DIAGNOSTIC_FAILED_THRESHOLD: full epoch skipped; investigate saved artifacts.',flush=True)
'''
nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=[dict(cell_type='code',id=f'learning-{i}',execution_count=None,metadata={},outputs=[],source=s.splitlines(True)) for i,s in enumerate((setup,run))])
out=Path('kp_hybrid_learning');out.mkdir(exist_ok=True)
(out/'hybrid_learning.ipynb').write_text(json.dumps(nb))
(out/'kernel-metadata.json').write_text(json.dumps(dict(id='tunhngphm/vi-diac-hybrid-learning',title='vi-diac-hybrid-learning',code_file='hybrid_learning.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=False,dataset_sources=['tunhngphm/vi-diac-split-v2','tunhngphm/vi-diac-factorized-context-base'],competition_sources=[],kernel_sources=[])))
print('Frozen learning diagnostic notebook')
