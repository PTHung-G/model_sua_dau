"""Download pinned Vietnamese Wikipedia and build punctuation-preserving pairs.
No model training. All previous corpora/checkpoints are untouched.
"""
from pathlib import Path
import collections, hashlib, json, re, time, unicodedata
import requests
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'punctuation_corpus_v1'
TOK = re.compile(r'[^\W_]+', re.UNICODE)

def strip(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c)).replace('đ', 'd').replace('Đ', 'D')

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def key(text):
    # Conservative exact lexical duplicate removal, insensitive to punctuation/case.
    return hashlib.blake2b(' '.join(TOK.findall(text.lower())).encode(), digest_size=16).digest()

def split_for(doc_id):
    n = int(digest('punctuation-v1:' + doc_id)[:8], 16) % 100
    return 'test' if n < 2 else 'validation' if n < 4 else 'train'

def make_pair(text):
    target = unicodedata.normalize('NFC', text)
    source = strip(target)
    assert strip(target) == source
    assert len(source) == len(target), 'Only NFC 1:1 restoration is supported'
    # Every non-letter/digit character, including punctuation and spaces, is intact.
    assert [(i, c) for i, c in enumerate(source) if not c.isalnum()] == [(i, c) for i, c in enumerate(target) if not c.isalnum()]
    return source, target

def main():
    assert not OUT.exists(), 'Use a new version rather than overwrite an existing corpus'
    OUT.mkdir(); raw = OUT / 'raw'; raw.mkdir()
    session = requests.Session()
    def get(url):
        response = session.get(url, timeout=120); response.raise_for_status(); return response
    meta = get('https://huggingface.co/api/datasets/wikimedia/wikipedia').json()
    revision = meta['sha']
    listing = get(f'https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/{revision}/20231101.vi').json()
    card = get(f'https://huggingface.co/datasets/wikimedia/wikipedia/raw/{revision}/README.md').text
    (OUT / 'SOURCE_README.md').write_text(card)
    downloads = []
    for entry in listing:
        if not entry['path'].endswith('.parquet'): continue
        url = f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{revision}/{entry['path']}"
        dest = raw / Path(entry['path']).name
        print('DOWNLOADING', dest.name, entry['size'], flush=True)
        h = hashlib.sha256()
        with session.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with dest.open('wb') as f:
                for block in response.iter_content(2**20):
                    if block: f.write(block); h.update(block)
        assert dest.stat().st_size == entry['size']
        expected = entry.get('lfs', {}).get('oid')
        if expected: assert h.hexdigest() == expected
        downloads.append({'path': entry['path'], 'bytes': dest.stat().st_size, 'sha256': h.hexdigest(), 'url': url})
    (OUT / 'download_manifest.json').write_text(json.dumps({'dataset': 'wikimedia/wikipedia', 'config': '20231101.vi', 'revision': revision, 'files': downloads}, indent=2))
    files = sorted(raw.glob('*.parquet'))
    def docs():
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=256, columns=['id', 'url', 'title', 'text']):
                yield from batch.to_pylist()
    stats = {s: collections.Counter() for s in ('train', 'validation', 'test')}
    punct = {s: collections.Counter() for s in stats}
    reject = collections.Counter(); heldout = set(); seen_eval = set(); seen_train = set()
    handles = {s: (OUT / (s + '.jsonl')).open('w', encoding='utf8') for s in stats}
    samples = []
    try:
        # Process evaluation before training; lexical exact matches never enter train.
        for phase in ('evaluation', 'training'):
            for doc in docs():
                split = split_for(str(doc['id']))
                if (split == 'train') != (phase == 'training'): continue
                stats[split]['documents_scanned'] += 1
                for paragraph_id, paragraph in enumerate(doc['text'].splitlines()):
                    target = unicodedata.normalize('NFC', paragraph.strip())
                    n = len(TOK.findall(target))
                    # Keep natural paragraphs intact, never hallucinate punctuation or
                    # split at arbitrary periods (abbreviations / decimals are preserved).
                    if not 5 <= n <= 256:
                        reject[split + ':length_outside_5_256'] += 1; continue
                    if '\ufffd' in target or any(unicodedata.category(c) == 'Cc' and c != '\t' for c in target):
                        reject[split + ':invalid_control'] += 1; continue
                    if not any(c in 'ăâđêôơưĂÂĐÊÔƠƯ' or unicodedata.combining(d) for c in target for d in unicodedata.normalize('NFD', c)):
                        reject[split + ':no_diacritics'] += 1; continue
                    k = key(target)
                    if split == 'train':
                        if k in heldout: reject['train:overlap_with_new_eval'] += 1; continue
                        if k in seen_train: reject['train:duplicate'] += 1; continue
                        seen_train.add(k)
                    else:
                        if k in seen_eval: reject[split + ':duplicate_or_cross_eval'] += 1; continue
                        seen_eval.add(k); heldout.add(k)
                    try: source, target = make_pair(target)
                    except AssertionError: reject[split + ':non_1to1'] += 1; continue
                    row = {'document_id': str(doc['id']), 'source_url': doc['url'], 'title': doc['title'], 'paragraph_id': paragraph_id, 'input': source, 'target': target}
                    handles[split].write(json.dumps(row, ensure_ascii=False) + '\n')
                    stats[split]['paragraphs'] += 1; stats[split]['tokens'] += n
                    ps = collections.Counter(c for c in target if unicodedata.category(c).startswith('P'))
                    punct[split].update(ps); stats[split]['with_punctuation'] += bool(ps)
                    if split == 'train' and len(samples) < 12 and ps: samples.append(row)
            for h in handles.values(): h.flush()
            print('PHASE_DONE', phase, {s: dict(v) for s, v in stats.items()}, flush=True)
    finally:
        for h in handles.values(): h.close()
    report = {'dataset': 'wikimedia/wikipedia 20231101.vi', 'revision': revision, 'source': 'https://huggingface.co/datasets/wikimedia/wikipedia', 'license_as_declared': ['CC-BY-SA-3.0', 'GFDL'], 'attribution': 'Wikipedia contributors; per-document source_url and title retained in every record. Dataset card archived alongside corpus.', 'split': 'SHA256(punctuation-v1:article_id) modulo100: test0..1, validation2..3, train4..99. Document-disjoint; exact lexical paragraphs deduplicated with evaluation priority.', 'input': 'Strip Unicode combining marks and map đ/Đ only; punctuation, case, numbers and internal whitespace unchanged.', 'unit': 'Natural nonempty paragraphs with 5..256 alphanumeric runs, NFC; long paragraphs excluded, not cut.', 'statistics': {s: dict(v) for s,v in stats.items()}, 'punctuation_counts': {s: dict(v) for s,v in punct.items()}, 'rejections': dict(reject), 'limitations': ['Wikipedia only: no claim of dialogue or news balance.', 'Existing old project models may have seen these articles: new splits are for retraining, not a fresh evaluation of old models.', 'Exact lexical dedup, not semantic near-duplicate removal.', 'Wikipedia extraction may already simplify source typography; no inferred or synthetic punctuation is added.', 'Rare-context rebalancing and model-array encoding are not applied in this raw corpus version.']}
    for split in stats:
        path = OUT / (split + '.jsonl'); h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(2**20), b''): h.update(chunk)
        report.setdefault('output_hashes', {})[path.name] = {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (OUT / 'examples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    print('COMPLETE', json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
