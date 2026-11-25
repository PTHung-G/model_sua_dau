"""Package the matched fine-tune as a private, offline Kaggle script."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = 'vi-diac-unary-finetune-pilot'
DATA = 'tunhngphm/vi-diac-verified-full-training-tensors-v2'
BASE = 'tunhngphm/vi-diac-corrected-four-epochs-v2'
SOURCES = ('punct_unary_finetune.py', 'punct_unary_pilot.py', 'eval_punct_vs_best.py',
           'charexp/syltrain_punct.py', 'charexp/sylmodel_punct.py',
           'charexp/input_features.py', 'build_punct_tensors_v1.py')
RUNNER = '''
from pathlib import Path
import subprocess
import sys

root = Path('/kaggle/input')
work = Path('/kaggle/working')
src = work / 'unary_sources'
for name, content in SOURCES.items():
    path = src / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')

def unique(pattern, validate=lambda p: True):
    paths = [p for p in root.rglob(pattern) if validate(p)]
    if len(paths) != 1:
        raise RuntimeError(f'Expected one mounted {pattern}; found {paths}')
    return paths[0]

inventory = unique('train.npz', lambda p: (p.parent/'train_ctab.npy').exists() and (p.parent/'validation.npz').exists()).parent
base = unique('punct_four_epochs_latest.pt.best.pt')
print(f'Using {inventory} and {base}',flush=True)
subprocess.run([sys.executable,str(src/'punct_unary_finetune.py'), '--inventory-dir',str(inventory),
                '--checkpoint',str(base), '--out',str(work/'unary_finetune_results'), *EXTRA_ARGS],check=True)
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--fivefold', action='store_true')
    args = p.parse_args()
    extra = ['--steps','2400','--extend-rows','38000','--milestones','480','1200','2400'] if args.fivefold else []
    slug = 'vi-diac-unary-finetune-5x' if args.fivefold else SLUG
    title = 'Vi diac unary finetune 5x' if args.fivefold else 'Vi diac unary finetune pilot'
    contents = {name:(HERE/name).read_text(encoding='utf-8') for name in SOURCES}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'main.py').write_text('SOURCES = '+repr(contents)+'\nEXTRA_ARGS = '+repr(extra)+'\n'+RUNNER,encoding='utf-8')
    meta = dict(id=f'tunhngphm/{slug}', title=title,code_file='main.py',
                language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=False,
                dataset_sources=[DATA],kernel_sources=[BASE],competition_sources=[])
    (args.out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
