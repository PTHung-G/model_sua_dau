"""Reuse historical exclusion audit and additionally exclude correction-study groups."""
import json
from pathlib import Path
import numpy as np
import correction_samples as cs

ROOT=Path(__file__).resolve().parent

def main():
    old=json.loads((ROOT/'correction_calibration/samples.json').read_text())
    # Reconstruct excluded historical line set from the audited JSON manifests.
    excluded=set()
    def walk(x):
        if isinstance(x,dict):
            if str(x.get('source','')).endswith('va.txt') and x.get('source_sha256')==old['source_sha256']:
                excluded.update(int(e['line_1based']) for e in x.get('entries',[]) if 'line_1based' in e)
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    for p in ROOT.rglob('*.json'):
        if p.stat().st_size>100000000:continue
        try:walk(json.loads(p.read_text()))
        except (UnicodeError,json.JSONDecodeError):continue
    for n,seed in [(10000,1730),(3000,1730),(512,1732),(2000,2718)]:
        _,m=cs.pa.reservoir(ROOT/'data/va.txt',n,seed)
        excluded.update(e['line_1based'] for e in m['entries'])
    for split in ('calibration','evaluation'):excluded.update(e['line_1based'] for e in old[split])
    words=[s.split() for s in (ROOT/'data/va.txt').read_text().splitlines()]
    seen={cs.group(words[i-1]) for i in excluded};chosen=[]
    for i in np.random.default_rng(2403).permutation(len(words)):
        key=cs.group(words[i])
        if key in seen or not key.strip():continue
        seen.add(key);chosen.append(dict(line_1based=int(i)+1,words=words[i]))
        if len(chosen)==1500:break
    assert len(chosen)==1500
    out=ROOT/'correction_gate';out.mkdir(exist_ok=True)
    path=out/'samples.json';assert not path.exists()
    path.write_text(json.dumps(dict(source=old['source'],source_sha256=old['source_sha256'],
                                   seed=2403,excluded_lines=len(excluded),evaluation=chosen),ensure_ascii=False,indent=2))
    print('Locked 1500 evaluation sentences; excluded lines',len(excluded))

if __name__=='__main__':main()
