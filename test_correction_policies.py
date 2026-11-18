import unittest
import numpy as np
from correction_policies import predict,select,metrics

class PolicyTests(unittest.TestCase):
    def data(self):
        return dict(base=np.array([[2.,1.,-1e9],[2.,1.,-1e9],[2.,1.,-1e9]]),
                    rerank=np.array([[1.,2.,-1e9],[2.,1.,-1e9],[1.,2.,-1e9]]),
                    gold=np.array([1,0,-1]),sentence=np.array([0,0,1]))
    def test_endpoints_and_invalid(self):
        d=self.data()
        np.testing.assert_array_equal(predict(d,'alpha',0),d['base'].argmax(-1))
        np.testing.assert_array_equal(predict(d,'alpha',1),d['rerank'].argmax(-1))
        self.assertNotIn(2,predict(d,'alpha',.5))
    def test_threshold_zero(self):
        d=self.data();np.testing.assert_array_equal(predict(d,'threshold',0),d['rerank'].argmax(-1))
    def test_oov_denominator_and_changes(self):
        d=self.data();m=metrics(d,predict(d,'alpha',1),2)
        self.assertAlmostEqual(m['token_accuracy'],2/3)
        self.assertEqual(m['fixed_tokens'],1);self.assertEqual(m['wrong_to_wrong'],1)
        self.assertEqual(m['sentence_exact'],.5)
    def test_conservative_tie(self):
        d=self.data();d['gold']=np.array([0,0,0]);d['rerank']=d['base'].copy()
        self.assertEqual(select(d,'alpha'),0)
        self.assertEqual(select(d,'threshold'),.4)

if __name__=='__main__':unittest.main()
