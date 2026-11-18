"""Pre-inference leakage audit and fixed-seed sample; no model inference/training.
A clean exact audit is not proof of absence of near duplicates or historical exposure.
"""
import collections, hashlib, json, random, re, unicodedata
from pathlib import Path
ROOT=Path('/home/dsh/ML/diac')
OUT=Path(__file__).resolve().parent
TOK=re.compile(r'[^\W_]+')
def key(s):
    return hashlib.sha256(' '.join(TOK.findall(unicodedata.normalize('NFC',s).lower())).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
# Lock sample once, without reading predictions or selecting on scores.
test=ROOT/'punctuation_corpus_v1/test.jsonl'
with test.open() as f:
    n=sum(1 for _ in f)
seed=20261115
indices=sorted(random.Random(seed).sample(range(n),500))
wanted=set(indices); rows=[]
with test.open() as f:
    for i,line in enumerate(f):
        if i in wanted:
            row=json.loads(line); rows.append((i,row))
assert len(rows)==500
docs={str(r['document_id']) for _,r in rows}
keys={key(r['target']) for _,r in rows}
hits={}; population={}
for split in ['train','validation']:
    p=ROOT/f'punctuation_corpus_v1/{split}.jsonl'
    matches=[]; count=0
    with p.open() as f:
        for i,line in enumerate(f):
            r=json.loads(line); count+=1
            same_doc=str(r['document_id']) in docs
            same_text=key(r['target']) in keys
            if same_doc or same_text:
                matches.append({'row':i,'document_id':r['document_id'],'same_article':same_doc,'exact_lexical_match':same_text})
    hits[split]=matches; population[split]=count
    print(split,count,'matches',len(matches),flush=True)
manifest={'seed':seed,'sampling':'uniform without replacement over all stored test paragraph rows; no gold or prediction filtering','test_rows':n,'sample_rows':500,'sample_article_count':len(docs),'source_indices_zero_based':indices,'corpus_sha256':sha(test),'input_target_pairs_sha256':hashlib.sha256(json.dumps([(r['input'],r['target']) for _,r in rows],ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),'tokens':sum(len(TOK.findall(r['target'])) for _,r in rows),'audit_population':population,'article_or_exact_lexical_matches':hits,'limitations':['No semantic or near-duplicate audit yet.','Source-code and checkpoint evidence is not a complete historical data-access log.','Prior test inspection means this is not a pristine confirmatory benchmark.','Paragraph-uniform sampling weights long articles more than short articles.']}
with (OUT/'sampling_audit.json').open('x') as f: json.dump(manifest,f,indent=2)
assert not any(hits.values()), 'Stop: overlap found; do not infer or silently resample'
with (OUT/'sample.jsonl').open('x') as f:
    for _,r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps({k:manifest[k] for k in ['seed','test_rows','sample_rows','sample_article_count','tokens','input_target_pairs_sha256']},indent=2),flush=True)
