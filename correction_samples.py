"""Prepare auditable, exposure-excluded sentence groups before inference."""
import hashlib
import json
from pathlib import Path
import unicodedata
import numpy as np
import pilot_arch as pa
import context_pilot as cp

ROOT=Path(__file__).resolve().parent

def group(words):
    # Same model input must not leak across calibration/evaluation, even with different gold accents.
    return ' '.join(cp.strip_all(cp.sa.normalized_core(w)) for w in words)

def main():
    out=ROOT/'correction_calibration';out.mkdir(exist_ok=True)
    if (out/'samples.json').exists():raise FileExistsError('Samples already locked')
    source=ROOT/'data/va.txt'
    digest=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
    excluded=set();audit=[]
    def walk(obj,path):
        if isinstance(obj,dict):
            if str(obj.get('source','')).endswith('va.txt') and isinstance(obj.get('entries'),list):
                sha=obj.get('source_sha256')
                if sha is not None and sha!=digest:
                    audit.append(dict(path=str(path.relative_to(ROOT)),status='other_source_hash'));return
                lines={int(e['line_1based']) for e in obj['entries'] if 'line_1based' in e}
                excluded.update(lines)
                audit.append(dict(path=str(path.relative_to(ROOT)),status='excluded',rows=len(lines)))
            for value in obj.values():walk(value,path)
        elif isinstance(obj,list):
            for value in obj:walk(value,path)
    for path in ROOT.rglob('*.json'):
        if out in path.parents or path.stat().st_size>100_000_000:continue
        try:walk(json.loads(path.read_text()),path)
        except (UnicodeError,json.JSONDecodeError):continue
    # Explicit known exposures, including comparison scripts without saved manifests.
    for count,seed in [(10000,1730),(3000,1730),(512,1732),(2000,2718)]:
        _,manifest=pa.reservoir(source,count,seed)
        lines={int(e['line_1based']) for e in manifest['entries']};excluded.update(lines)
        audit.append(dict(reconstructed_count=count,seed=seed,status='excluded',rows=len(lines)))
    rows=[unicodedata.normalize('NFC',line).split() for line in source.read_text().splitlines()]
    exposed_groups={group(rows[i-1]) for i in excluded if 0<i<=len(rows)}
    seen=set(exposed_groups);pool=[]
    # Shuffle before group deduplication, so representative choice is deterministic but not first-row biased.
    for i in np.random.default_rng(2401).permutation(len(rows)):
        words=rows[i];key=group(words)
        if int(i)+1 in excluded or key in seen or not key.strip():continue
        seen.add(key);pool.append(dict(line_1based=int(i)+1,words=words,
                                      input_group_sha256=hashlib.sha256(key.encode()).hexdigest()))
    assert len(pool)>=2012
    result=dict(source=str(source),source_sha256=digest,seed=2401,
                excluded_line_count=len(excluded),excluded_group_count=len(exposed_groups),
                audit=audit,limitations='Exclusion covers recorded JSON manifests and explicitly reconstructed known samples; unrecorded historical exposure and train overlap are not ruled out. Same-corpus feasibility only.',
                calibration=pool[:512],evaluation=pool[512:2012])
    assert not ({e['input_group_sha256'] for e in result['calibration']}&{e['input_group_sha256'] for e in result['evaluation']})
    (out/'samples.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('audit','calibration','evaluation')},indent=2))

if __name__=='__main__':main()
