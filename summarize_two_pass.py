"""Summarize completed two-pass validation pilot, without claiming significance."""
import argparse,json
from pathlib import Path
import numpy as np

def main(directory):
 p=Path(directory);r=json.loads((p/'results.json').read_text());assert r['status']=='complete'
 print('Top-level keys:',list(r))
 print('Base:',{k:v for k,v in r.get('base_init',{}).items() if k in ('sentence_exact','token_accuracy','n_sentences','n_tokens','topk_oracle')})
 arms=r.get('arms',{})
 for arm,v in arms.items():
  print('Arm',arm,'beststep',v['best']['step'])
  for step,m in v['marks'].items():
   q=m['validation'];print(step,{k:v for k,v in q.items() if k not in ('sentence_errors','sentence_lengths','known_tokens','by_length','paired_vs_base_init','oracle_dictionary_ceiling')})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory');main(p.parse_args().directory)
