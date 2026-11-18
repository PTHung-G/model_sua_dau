"""Generate a self-contained Kaggle kernel for punctuation-aware training.

Embeds sylmodel_punct.py and syltrain_punct.py verbatim so the kernel does not
depend on the project filesystem. The tensors dataset must be attached.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'syl_kernel_punct_audit_v2'
OUT.mkdir(exist_ok=True)

model_src = (ROOT / 'charexp/sylmodel_punct.py').read_text()
train_src = (ROOT / 'charexp/syltrain_punct.py').read_text()

main = f'''"""Kaggle: punctuation-aware Vietnamese diacritic restoration."""
import json, sys, shutil
from pathlib import Path

for path in [Path('/kaggle/input')] and list(Path('/kaggle/input').rglob('train_ctab.npy')):
    root = path.parent
    print('tensor root', root, flush=True)
    break
else:
    raise SystemExit('Punctuation tensor dataset not attached')

import base64
Path('/kaggle/working/sylmodel_punct.py').write_text(base64.b64decode(MODEL_SOURCE).decode())
Path('/kaggle/working/syltrain.py').write_text(base64.b64decode(TRAIN_SOURCE).decode())
sys.path.insert(0, '/kaggle/working')
sys.modules['sylmodel_punct'] = __import__('sylmodel_punct')

assert (root / 'train_gaps.npy').exists(), 'gaps sidecar missing'
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest.get('pipeline_schema') == 2, 'Legacy tensor dataset is invalid; rebuild schema2 first'
print('manifest', json.dumps(manifest), flush=True)

train = __import__('syltrain')
sys.argv = ['x', '--prefix', str(root / 'train'), '--val-prefix', str(root / 'validation'),
            '--inventory-prefix', str(root / 'train'), '--epochs', '8', '--seed', '0',
            '--bs', '32', '--d', '128', '--hidden', '136', '--r', '32', '--layers', '2',
            '--eval-every', '20000', '--max-minutes', '690',
            '--out', '/kaggle/working/punct_latest.pt']
train.main()
'''

import base64
main = main.replace('MODEL_SOURCE', repr(base64.b64encode(model_src.encode()).decode()))
main = main.replace('TRAIN_SOURCE', repr(base64.b64encode(train_src.encode()).decode()))
(OUT / 'main.py').write_text(main)

(OUT / 'kernel-metadata.json').write_text(json.dumps({
    'id': 'tunhngphm/vi-diac-punctuation-aware-attention',
    'title': 'VI diacritic punctuation-aware attention',
    'code_file': 'main.py',
    'language': 'python',
    'kernel_type': 'script',
    'is_private': True,
    'enable_gpu': False,
    'enable_internet': False,
    'dataset_sources': ['tunhngphm/vi-diac-punctuation-tensors-v2-audit-required'],
    'competition_sources': [],
    'kernel_sources': [],
}, indent=2))
print('wrote', OUT)
