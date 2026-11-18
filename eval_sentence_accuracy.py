"""Sentence exact by explicit terminal punctuation; inference retains paragraph context.
Strict requested rule: every . ! ? … run ends a unit, including abbreviation dots.
Also report a decimal-aware variant, never silently change the headline rule.
Unterminated tails are reported separately, not counted as terminated sentences.
"""
import json,re,hashlib,unittest
from pathlib import Path
import torch
from eval_punct_vs_best import FrozenRestorer,read_pairs,TOK,score_pairs
ROOT=Path(__file__).resolve().parent
END=re.compile(r'[.!?…]+[\"\'”’»）)\]}]*')

def sentence_spans(text,decimal_aware=False):
    spans=[];start=0
    for m in END.finditer(text):
        if decimal_aware and m.group()=='.' and m.start()>0 and m.end()<len(text) and text[m.start()-1].isdigit() and text[m.end()].isdigit():continue
        if TOK.search(text[start:m.end()]):spans.append((start,m.end(),True))
        start=m.end()
    if TOK.search(text[start:]):spans.append((start,len(text),False))
    return spans

def score_sentence_units(pairs,predictions,decimal_aware=False):
    correct=total=tail_correct=tails=0;buckets={}
    for (source,target),prediction in zip(pairs,predictions):
        assert len(source)==len(target)==len(prediction),'Character alignment required'
        assert TOK.sub('',source)==TOK.sub('',target)==TOK.sub('',prediction)
        for start,end,terminated in sentence_spans(source,decimal_aware):
            exact=target[start:end].strip()==prediction[start:end].strip()
            if not terminated:tails+=1;tail_correct+=exact;continue
            total+=1;correct+=exact
            n=len(TOK.findall(target[start:end]));key='1-10' if n<=10 else '11-20' if n<=20 else '21-40' if n<=40 else '41-80' if n<=80 else '81+'
            b=buckets.setdefault(key,{'sentences':0,'correct':0});b['sentences']+=1;b['correct']+=exact
    for b in buckets.values():b['accuracy']=b['correct']/b['sentences']
    return {'sentences':total,'correct_sentences':correct,'sentence_accuracy':correct/total if total else None,'unterminated_tail_fragments':tails,'correct_tail_fragments':tail_correct,'length_buckets':buckets}

class Tests(unittest.TestCase):
    def test_boundaries(self):
        s='“Toi di?!” Ban o lai. Phan cuoi'
        spans=sentence_spans(s);self.assertEqual([s[a:b].strip() for a,b,_ in spans],['“Toi di?!”','Ban o lai.','Phan cuoi']);self.assertEqual([x[2] for x in spans],[True,True,False])
    def test_decimal_and_abbreviation_explicit(self):
        self.assertEqual(len(sentence_spans('Gia 3.14. TS. A di!')),4)
        self.assertEqual(len(sentence_spans('Gia 3.14. TS. A di!',True)),3)
    def test_denominator_and_error(self):
        src='Toi di. Ban o lai? Cuoi';gold='Tôi đi. Bạn ở lại? Cuối';pred='Tôi đi. Ban o lai? Cuoi'
        r=score_sentence_units([(src,gold)],[pred]);self.assertEqual(r['sentences'],2);self.assertEqual(r['correct_sentences'],1);self.assertEqual(r['unterminated_tail_fragments'],1)

if __name__=='__main__':
    import sys
    if '--test' in sys.argv:unittest.main(argv=[sys.argv[0]])
    else:
        torch.set_num_threads(2);pairs=read_pairs(ROOT/'punctuation_corpus_v1/test.jsonl',500)
        report={'raw_rows':len(pairs),'raw_pairs_sha256':hashlib.sha256(json.dumps(pairs,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),'protocol':'Same first 500 raw test paragraphs as previous benchmark; predict full paragraph once, score punctuation-delimited sentence spans. Strict .!?… terminal runs; closing quotes attached. Abbreviation dots counted as requested; unterminated tails separate. No gold filtering.','models':{}}
        for kind,ck,inv in [('attention',ROOT/'charexp/syl_rebalance_output_v1/rebalance_latest.pt.best.pt',ROOT/'charexp/rebalance_v1/train'),('punct',ROOT/'punct_four_epochs_output_v2/punct_four_epochs_latest.pt.best.pt',ROOT/'punct_tensors_audit_v2/train')]:
            restorer=FrozenRestorer(kind,ck,inv);predictions=[restorer.predict(s)[0] for s,_ in pairs]
            report['models'][kind]={'strict_terminal_rule':score_sentence_units(pairs,predictions),'decimal_aware_sensitivity':score_sentence_units(pairs,predictions,True),'paragraph_and_token_metrics':score_pairs(pairs,predictions)}
            print(kind,json.dumps(report['models'][kind],ensure_ascii=False),flush=True)
            del restorer
        out=ROOT/'sentence_accuracy_raw500_v1.json'
        with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
