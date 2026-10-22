import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ranking_driver_core import PreparedFusion,load_training,p3_graph,P3_D,P3_R,relation_scores,gather_scores
from protocol_primitives import exact_sparse_fusion


class RankingDriverTests(unittest.TestCase):
    def test_prepared_fusion_matches_reference(self):
        rng=np.random.default_rng(51)
        for _ in range(100):
            n=40;catalog=np.sort(rng.choice(n,30,replace=False));seen=rng.choice(n,9,replace=False)
            ids=rng.choice(n,20,replace=False);values=rng.integers(0,4,20).astype(float)
            aux=rng.integers(0,4,n)/3
            for gamma in [0,.1,1,2]:
                got=PreparedFusion(catalog,aux,gamma).rank(ids,values,seen)
                expected=exact_sparse_fusion(ids,values,aux,catalog,seen,gamma)
                np.testing.assert_array_equal(got,expected)

    def fixture(self,future):
        df=pd.DataFrame({'user_idx':[0,0,1,1,2,2,0,3],
            'item_idx':[0,1,1,2,0,2,3,3],
            'event':['click','contact_chat','click','contact_phone_click_2','click','click','contact_chat','click'],
            'timestamp':[1,2,3,4,5,6,future,future]})
        c=duckdb.connect();c.register('events',df);return c

    def test_training_prefix_priors_and_p3_dense_oracle(self):
        with patch('ranking_driver_core.connection',side_effect=lambda:self.fixture(11)):
            a=load_training(10,np.arange(3))
        with patch('ranking_driver_core.connection',side_effect=lambda:self.fixture(10000)):
            b=load_training(10,np.arange(3))
        for name in a.priors:np.testing.assert_array_equal(a.priors[name],b.priors[name])
        self.assertEqual(a.seen.nnz,6);self.assertLess(a.pairs.last_ts.max(),10)
        edge,sim=p3_graph(a,block_size=1)
        x=a.seen.toarray();ud=x.sum(1);ideg=x.sum(0)
        matrices=[np.zeros_like(x) for _ in range(3)]
        for row in a.pairs.itertuples():
            age=(row.user_last-row.last_ts)/86400
            for k,degree in enumerate([ideg[row.item_idx],ud[row.u],ideg[row.item_idx]]):
                matrices[k][row.u,row.item_idx]=degree**-P3_D[k]*np.exp(-age*P3_R[k])
        expected=matrices[1].T@matrices[2];np.fill_diagonal(expected,0)
        np.testing.assert_allclose(edge.toarray(),matrices[0]);np.testing.assert_allclose(sim.toarray(),expected)

    def test_profile_formula_and_score_alignment(self):
        candidates=np.array([[2,0,1],[1,2,0]])
        profiles={'chat_click':np.array([[0.,3.,8.],[2.,0.,4.]]),
                  'contact_phone_click_1':np.array([[1.,2.,3.],[1.,4.,2.]])}
        events=np.array(['chat_click','contact_phone_click_1']);risk=np.array([.3,.7])
        got=relation_scores(candidates,events,risk,profiles)
        for r in range(2):
            raw=np.array([np.log1p(profiles[events[r]][0,i])-risk[r]*np.log1p(profiles[events[r]][1,i]) for i in candidates[r]])
            np.testing.assert_allclose(got[r],(raw-raw.mean())/(raw.std()+1e-8))
        scores=sparse.csr_matrix([[4,0,3],[0,2,8]])
        np.testing.assert_array_equal(gather_scores(scores,candidates),[[3,4,0],[2,8,0]])
        with self.assertRaises(ValueError):relation_scores(candidates,events,np.array([2.,.7]),profiles)


if __name__=='__main__':unittest.main()
