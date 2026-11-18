"""RAW frozen-checkpoint comparison; no gold filtering, no training.

Historical tensor-filtered scores are NOT comparable to this protocol. All
models see the same provided JSONL input rows. Targets are scoring-only. Missing
output tokens count as errors. Unknown input bases receive singleton COPY.
Punctuation checkpoints without pipeline_schema >= 2 are rejected, not migrated.
"""
import argparse
from collections import ChainMap, Counter
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'charexp'))
TOK = re.compile(r'[^\W_]+', re.UNICODE)


def strip(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if not unicodedata.combining(c)).replace('đ', 'd').replace('Đ', 'D')


def restore_case(candidate, source):
    """Transfer case character by character; never title-case an entire word."""
    if len(candidate) != len(source):
        raise ValueError('Candidate and source lengths differ')
    return ''.join(c.upper() if s.isupper() else c.lower()
                   for c, s in zip(candidate, source))


def replace_spans(source, words):
    matches = list(TOK.finditer(source))
    if len(matches) != len(words):
        raise ValueError('Prediction/source token count mismatch')
    chunks, end = [], 0
    for match, word in zip(matches, words):
        chunks.extend((source[end:match.start()], word))
        end = match.end()
    return ''.join(chunks) + source[end:]


def score_pairs(pairs, predictions):
    """Case-sensitive NFC alphanumeric-run exact; full gold denominator.

    Additional output tokens are separately counted (and affect sentence exact).
    Punctuation preservation is measured against INPUT, not against gold.
    """
    if len(pairs) != len(predictions):
        raise ValueError('Every raw row must have exactly one prediction')
    result = Counter(rows=len(pairs))
    for (source, target), prediction in zip(pairs, predictions):
        gold = TOK.findall(unicodedata.normalize('NFC', target))
        pred = TOK.findall(unicodedata.normalize('NFC', prediction))
        result['tokens'] += len(gold)
        result['correct'] += sum(a == b for a, b in zip(gold, pred))
        result['missing_tokens'] += max(0, len(gold) - len(pred))
        result['extra_tokens'] += max(0, len(pred) - len(gold))
        result['sentence_exact_count'] += unicodedata.normalize('NFC', target) == unicodedata.normalize('NFC', prediction)
        result['input_punctuation_preserved_rows'] += TOK.sub('', source) == TOK.sub('', prediction)
    result = dict(result)
    for name in ('tokens', 'correct', 'missing_tokens', 'extra_tokens', 'sentence_exact_count', 'input_punctuation_preserved_rows'):
        result.setdefault(name, 0)
    result['token_accuracy'] = result['correct'] / result['tokens'] if result['tokens'] else None
    result['sentence_exact'] = result['sentence_exact_count'] / len(pairs) if pairs else None
    return result


def require_punct_schema(checkpoint):
    schema = checkpoint.get('pipeline_schema', checkpoint.get('config', {}).get('pipeline_schema', 0))
    if not isinstance(schema, int) or isinstance(schema, bool) or schema < 2:
        raise ValueError('Rejected legacy/invalid punctuation checkpoint: pipeline_schema >= 2 required; historical gap-misaligned artifacts cannot be evaluated as corrected models')


def load_gap_mapping(path):
    raw = json.loads(Path(path).read_text(encoding='utf8'))
    if isinstance(raw, dict):
        raw = raw.get('gap_to_id', raw.get('tokens', raw.get('vocabulary', raw)))
    mapping = {s: i for i, s in enumerate(raw)} if isinstance(raw, list) else raw
    if not isinstance(mapping, dict) or not all(isinstance(s, str) and isinstance(i, int) for s, i in mapping.items()):
        raise ValueError('Invalid punctuation vocabulary')
    if mapping.get('') != 2 or len(set(mapping.values())) != len(mapping) or min(mapping.values()) < 0:
        raise ValueError('Corrected gap vocabulary requires distinct IDs with empty gap=2, padding=0, unknown=1')
    if 0 not in mapping.values() or 1 not in mapping.values():
        raise ValueError('Missing reserved PAD=0 / UNK=1 entries')
    return mapping


def read_pairs(path, limit=0):
    pairs = []
    with Path(path).open(encoding='utf8') as stream:
        for row_id, line in enumerate(stream):
            row = json.loads(line)
            if not isinstance(row.get('input'), str) or not isinstance(row.get('target'), str):
                raise ValueError(f'Invalid raw pair at row {row_id}')
            pairs.append((row['input'], row['target']))
            if limit and len(pairs) >= limit:
                break
    return pairs


class FrozenRestorer:
    def __init__(self, kind, checkpoint, prefix):
        self.kind = kind
        self.prefix = str(prefix)
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if kind == 'punct':
            require_punct_schema(ck)
        self.vocab = json.loads(Path(self.prefix + '_vocab.json').read_text(encoding='utf8'))
        bases = json.loads(Path(self.prefix + '_bases.json').read_text(encoding='utf8'))
        self.bid = {b: i + 1 for i, b in enumerate(bases)}
        trainer = importlib.import_module('syltrain_punct' if kind == 'punct' else 'syltrain')
        # Old branch MUST use native unpruned character IDs.
        nchar, bc, cc, cid, _ = trainer.build_tensors(self.prefix)
        state, cfg = ck['model'], ck['config']
        for name, constructed in [('base_ch', bc), ('cand_ch', cc), ('cand_ids', cid)]:
            if name not in state or not torch.equal(state[name].cpu().long(), constructed.long()):
                raise ValueError(f'Checkpoint/inventory encoding mismatch: {name}')
        if state['emb.weight'].shape[0] != nchar:
            raise ValueError('Checkpoint character vocabulary mismatch')
        self.sm = importlib.import_module('sylmodel_punct' if kind == 'punct' else 'sylmodel_attention')
        kwargs = {name: cfg[name] for name in ('d', 'hidden', 'r', 'layers')}
        if kind == 'punct':
            self.gap_map = load_gap_mapping(self.prefix + '_punct.json')
            size = max(self.gap_map.values()) + 1
            if state['gap_emb.weight'].shape[0] != size:
                raise ValueError('Checkpoint punctuation vocabulary mismatch')
            self.model = self.sm.PunctSylCRF(nchar, state['base_ch'], state['cand_ch'], state['cand_ids'], size, **kwargs)
        else:
            self.model = self.sm.SylCRF(nchar, state['base_ch'], state['cand_ch'], state['cand_ids'], **kwargs)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        # Native character maps are reconstructed exactly as their trainers do.
        chars = Counter()
        for word in bases + self.vocab:
            chars.update(word)
        if kind == 'punct':
            threshold = cfg.get('min_char_count', 100)
            self.c2i = {c: i + 2 for i, c in enumerate(sorted(c for c, n in chars.items() if n >= threshold))}
            self.unk_char = 1
        else:
            self.c2i = {c: i + 1 for i, c in enumerate(sorted(chars))}
            self.unk_char = 0
        self.native_features = importlib.import_module('input_features')
        self.builder = importlib.import_module('build_punct_tensors_v1') if kind == 'punct' else None

    def encode(self, strings, reference):
        width = reference.shape[-1]
        return torch.tensor([[self.c2i.get(c, self.unk_char) for c in s[:width]] + [0] * max(0, width - len(s))
                             for s in strings], dtype=reference.dtype)

    @torch.inference_mode()
    def predict(self, source):
        # Preserve original punctuation/spacing; strip only inside word spans.
        matches = list(TOK.finditer(source))
        if not matches:
            return source, {'input_tokens': 0, 'known_base_tokens': 0, 'oov_copy_tokens': 0}
        forms = [strip(m.group()).lower() for m in matches]
        m = self.model
        saved = {name: getattr(m, name) for name in ('base_ch', 'cand_ch', 'cand_ids', 'base_mask', 'cand_mask') if hasattr(m, name)}
        bid, vocab = ChainMap({}, self.bid), self.vocab
        new_forms, new_rows = [], []
        original_vocab_size = len(vocab)
        coverage = {'input_tokens': len(forms), 'known_base_tokens': sum(f in self.bid for f in forms),
                    'oov_copy_tokens': sum(f not in self.bid for f in forms)}
        try:
            for form in forms:
                if form not in bid:
                    bid[form] = len(self.bid) + len(new_forms) + 1
                    row = torch.full((m.cand_ids.shape[1],), -1, dtype=m.cand_ids.dtype)
                    row[0] = len(vocab)
                    vocab.append(form); new_forms.append(form); new_rows.append(row)
            if new_forms:
                m.base_ch = torch.cat((m.base_ch, self.encode(new_forms, m.base_ch)))
                m.cand_ch = torch.cat((m.cand_ch, self.encode(new_forms, m.cand_ch)))
                m.cand_ids = torch.cat((m.cand_ids, torch.stack(new_rows)))
                for name, ref in [('base_mask', m.base_ch), ('cand_mask', m.cand_ch)]:
                    if name in saved:
                        setattr(m, name, ref.ne(0).float())
            x = torch.tensor([[bid[f] for f in forms]])
            ln = torch.tensor([len(forms)])
            # Feature extraction uses the provided input, never target.
            if self.kind == 'punct':
                _, gaps = self.builder.gaps(source)
                sequence = [p for p, _ in gaps] + [gaps[-1][1]]
                gg = torch.tensor([[self.gap_map.get(g, 1) for g in sequence]])
                ff = torch.tensor([[self.builder.case_id(mm.group()) for mm in matches]])
                emit, trans, mask = m(x, ln, ff, gg)
            else:
                ff = torch.tensor([self.native_features.features(source)])
                emit, trans, mask = m(x, ln, ff)
            pred = self.sm.viterbi(emit, trans, mask)[0]
            words = []
            for i, (form, match) in enumerate(zip(forms, matches)):
                if form not in self.bid:
                    words.append(match.group())  # exact input-only COPY, including existing accents
                else:
                    candidate = vocab[int(m.cand_ids[bid[form], int(pred[i])])]
                    words.append(restore_case(candidate, match.group()))
            return replace_spans(source, words), coverage
        finally:
            del vocab[original_vocab_size:]
            for name, tensor in saved.items():
                setattr(m, name, tensor)


def evaluate(restorer, pairs):
    predictions, coverage = [], Counter()
    for source, _target in pairs:
        prediction, counts = restorer.predict(source)
        predictions.append(prediction)
        coverage.update(counts)
        # Post-hoc diagnostic ONLY: never controls input, candidates or row selection.
        gold_words = TOK.findall(unicodedata.normalize('NFC', _target))
        input_words = TOK.findall(source)
        if hasattr(restorer, 'bid'):
            for token, gold in zip(input_words, gold_words):
                bi = restorer.bid.get(strip(token).lower())
                if bi is not None:
                    candidates = restorer.model.cand_ids[bi].tolist()
                    coverage['gold_in_native_candidates_tokens'] += any(
                        ci >= 0 and restore_case(restorer.vocab[ci], token) == gold
                        for ci in candidates)
        coverage['gold_tokens_for_coverage'] += len(gold_words)
    result = score_pairs(pairs, predictions)
    result['coverage'] = dict(coverage)
    result['coverage']['known_base_fraction'] = coverage['known_base_tokens'] / coverage['input_tokens'] if coverage['input_tokens'] else None
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--limit', type=int, default=0, help='First N raw rows; 0 = all')
    ap.add_argument('--kind', choices=('both', 'attention', 'punct'), default='both')
    ap.add_argument('--attention-ckpt', default=str(ROOT / 'charexp/syl_rebalance_output_v1/rebalance_latest.pt.best.pt'))
    ap.add_argument('--attention-inventory', default=str(ROOT / 'charexp/rebalance_v1/train'))
    ap.add_argument('--punct-ckpt')
    ap.add_argument('--punct-inventory')
    ap.add_argument('--out', required=True, help='New output file; existing files are never overwritten')
    args = ap.parse_args()
    if args.limit < 0:
        ap.error('--limit must be >= 0')
    if args.kind in ('both', 'punct') and not (args.punct_ckpt and args.punct_inventory):
        ap.error('Corrected --punct-ckpt and --punct-inventory are required')
    if Path(args.out).exists():
        ap.error('Refusing to overwrite existing results')
    torch.set_num_threads(2)
    pairs = read_pairs(args.jsonl, args.limit)
    identity = hashlib.sha256(json.dumps(pairs, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    report = {'protocol': 'RAW input-target; all selected rows, no gold filtering; input-only OOV COPY; native features; case-sensitive alphanumeric exact with full gold denominator',
              'raw_pairs_sha256': identity, 'rows': len(pairs),
              'historical_results': 'Tensor-filtered scores and schema<2 punctuation artifact numbers are invalid for corrected comparisons; preserved unchanged, non-comparable.',
              'models': {}}
    for kind in ('attention', 'punct'):
        if args.kind not in ('both', kind):
            continue
        checkpoint = getattr(args, kind + '_ckpt')
        inventory = getattr(args, kind + '_inventory')
        restorer = FrozenRestorer(kind, checkpoint, inventory)
        report['models'][kind] = {'checkpoint': checkpoint, 'inventory': inventory, **evaluate(restorer, pairs)}
    with Path(args.out).open('x', encoding='utf8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
