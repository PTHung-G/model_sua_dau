"""Report arithmetic/integrity tests independent of long GPU training."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import report_hybrid_rerank as report


class ReportTests(unittest.TestCase):
    def fixture(self,root):
        for arm,errors in [('CONTROL',[0,0,1]),('HYBRID',[0,0,0])]:
            folder=root/f'hybrid_rerank_{arm.lower()}_results';folder.mkdir()
            metric=dict(sentence_errors=errors,token_accuracy=.95)
            base=dict(sentence_errors=[0,1,1],token_accuracy=.9)
            record=dict(budget_complete=True,best={'step':500},final_best=metric,final=metric,
                        consumed_batches=500,visited_source_rows=32000,train_seconds=20,
                        skipped_batches=0,evaluation_log=[{'step':0,'validation':base},
                                                       {'step':500,'validation':metric}])
            result=dict(status='complete',base_init=base,arms={f'hybrid_rerank_{arm}':record},
                        provenance={'validation':{'sample_sha256':'same'},
                                    'train':{'planned_prefix_sha256':'same'}})
            verification={k:dict(strict_load=True,frozen_base_equal=True,finite=True,
                                 step=500,counts={'trainable':10,'frozen':5,'total':15})
                          for k in ('best','latest')}
            (folder/'results.json').write_text(json.dumps(result))
            (folder/'verification.json').write_text(json.dumps(verification))
            for k in ('best','latest'):(folder/f'hybrid_rerank_{k}.pt').touch()

    def run_report(self,root):
        with patch.object(sys,'argv',['report','--root',str(root)]),contextlib.redirect_stdout(io.StringIO()):
            report.main()

    def test_arithmetic(self):
        x=report.compare([0,1,0,1],[0,0,1,0])
        self.assertEqual((x['rescued'],x['broken'],x['delta_pp']),(2,1,25))

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root);self.run_report(root)
            out=json.loads((root/'hybrid_summary.json').read_text())
            self.assertEqual(out['HYBRID']['best_correct'],3)
            self.assertEqual(out['hybrid_vs_control_best']['rescued'],1)
            self.assertEqual(len(out['equal_step_comparisons']),2)

    def test_rejects_partial_run(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root)
            path=root/'hybrid_rerank_hybrid_results/results.json'
            d=json.loads(path.read_text());d['status']='training';path.write_text(json.dumps(d))
            with self.assertRaisesRegex(AssertionError,'Incomplete'):self.run_report(root)

    def test_rejects_different_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root)
            path=root/'hybrid_rerank_hybrid_results/results.json'
            d=json.loads(path.read_text());d['provenance']['validation']['sample_sha256']='different'
            path.write_text(json.dumps(d))
            with self.assertRaisesRegex(AssertionError,'Validation manifests'):self.run_report(root)

if __name__=='__main__':unittest.main()
