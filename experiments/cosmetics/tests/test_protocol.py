"""Synthetic temporal, candidate and score-tie checks."""
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import run_validation as v

class CosmeticsProtocolTests(unittest.TestCase):
    def test_strict_history_and_maturity(self):
        rows=[(0,'cart',1,10),(0,'view',1,11),(5,'purchase',1,10),
              (10,'cart',1,12),(86400,'cart',1,13),(86401,'cart',1,14),
              (86402,'purchase',1,13),(0,'view',2,10),(10,'cart',2,11),(20,'purchase',2,12)]
        d=pd.DataFrame(rows,columns=['t','event_type','user_id','product_id']).sort_values(['t','user_id','product_id'])
        d['event_type']=pd.Categorical(d.event_type,categories=['view','cart','purchase','remove_from_cart'])
        original=v.OUT
        try:
            with tempfile.TemporaryDirectory() as folder:
                v.OUT=Path(folder);c=v.feature_frame(d,'synthetic')
                first=c[(c.user_id==1)&(c.t==0)].iloc[0]
                self.assertEqual(first.log_user_events,0);self.assertEqual(first.log_item_events,0)
                self.assertEqual(first.y,0)
                equal=c[(c.user_id==1)&(c.t==86400)].iloc[0]
                later=c[(c.user_id==1)&(c.t==86401)].iloc[0]
                self.assertEqual(equal.log_mature_user_success,0)
                np.testing.assert_allclose(later.log_mature_user_success,np.log(2),rtol=1e-6)
                self.assertEqual(later.log_mature_user_nonprogression,0)
                self.assertTrue(np.isfinite(c[v.FEATURES].to_numpy()).all())
        finally:v.OUT=original
    def test_exact_rank_ties(self):
        ids=np.array([[20,10,30],[30,20,10]])
        scores=np.array([[1,1,0],[0,0,0]],float)
        np.testing.assert_array_equal(v.ranks(ids,scores),[2,3])
    def test_candidate_exclusion_and_reproducibility(self):
        c=v.sampled_candidates(np.arange(1200),[1,2,3],9,999,7,100)
        self.assertEqual(len(np.unique(c)),1000);self.assertEqual(c[0],9)
        self.assertFalse(np.isin(c,[1,2,3]).any())
        np.testing.assert_array_equal(c,v.sampled_candidates(np.arange(1200),[1,2,3],9,999,7,100))

if __name__=='__main__':unittest.main()
