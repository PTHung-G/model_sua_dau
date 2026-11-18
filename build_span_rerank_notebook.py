"""Freeze the sequential A/B/C span-reranker Kaggle notebook."""
import json
from pathlib import Path

files = ['attn.py', 'attn_model.py', 'pilot_arch.py', 'pilot_balance.py', 'evaluate.py',
         'vn.py', 'score_all.py', 'context_pilot.py', 'sentence_loss_pilot.py',
         'sentence_loss_epoch.py', 'span_rerank_pilot.py', 'test_span_rerank.py']
sources = {f: Path(f).read_text() for f in files}

setup = "import os,pathlib,torch\nassert torch.cuda.is_available()\nos.chdir('/kaggle/working')\n"
setup += 'sources=' + repr(sources) + '\n'
setup += '''for name,text in sources.items(): pathlib.Path(name).write_text(text)
data=[p.parent for p in pathlib.Path('/kaggle/input').rglob('cand.json') if (p.parent/'tr.txt').exists() and (p.parent/'va.txt').exists()]
weights=list(pathlib.Path('/kaggle/input').rglob('factorized_best.pt'))
assert len(data)==1 and len(weights)==1,(data,weights)
print('GPU',torch.cuda.get_device_name(0),flush=True)
'''

test = """import subprocess,sys
subprocess.run([sys.executable,'-m','unittest','test_span_rerank','-v'],check=True)
"""

run = """import json,subprocess,sys
beats=None
for config in ('A','B','C'):
    print('=== configuration',config,flush=True)
    subprocess.run([sys.executable,'-u','span_rerank_pilot.py','--config',config,
                    '--data-dir',str(data[0]),'--checkpoint',str(weights[0]),
                    '--out-dir',f'span_rerank_{config.lower()}_results',
                    '--steps','2000','--val-sents','3000','--lr','0.0003'],check=True)
    result=json.loads(pathlib.Path(f'span_rerank_{config.lower()}_results/results.json').read_text())
    arm=list(result['arms'].values())[0]
    best=arm['best']['step']; exact=arm['final_best']['sentence_exact']
    base=sum(e==0 for e in result['base_init']['sentence_errors'])
    correct=sum(e==0 for e in arm['final_best']['sentence_errors'])
    print(json.dumps(dict(config=config,best_step=best,sentences_correct=correct,base_correct=base,delta=correct-base)),flush=True)
    if correct<=base and best==0:
        print('stopping: configuration',config,'did not beat the frozen generator',flush=True)
        break
"""

cells = [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
          'source': s.splitlines(True)} for s in (setup, test, run)]
notebook = {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                        'name': 'python3'}},
            'cells': cells}
dest = Path('kp_span_rerank')
dest.mkdir(exist_ok=True)
(dest / 'span_rerank.ipynb').write_text(json.dumps(notebook))
(dest / 'kernel-metadata.json').write_text(json.dumps({
    'id': 'tunhngphm/vi-diac-span-rerank-pilot',
    'title': 'vi-diac-span-rerank-pilot',
    'code_file': 'span_rerank.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': True,
    'enable_internet': False,
    'dataset_sources': ['tunhngphm/vi-diac-split-v2',
                        'tunhngphm/vi-diac-factorized-context-base'],
    'competition_sources': [],
    'kernel_sources': [],
}))
print('Notebook frozen')
