import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from protocol_primitives import (stable_topk,stable_topk_rows,sampled_candidates,
    rank_target,exact_sparse_fusion,full_catalog_user_metrics,
    require_component_available,require_disjoint_outcome_intervals)
from evaluate_full_catalog_published_protocol_olx import evaluate


class ProtocolTests(unittest.TestCase):
    def test_component_selection_boundary(self):
        require_component_available(1582493743,1582493743)
        with self.assertRaises(ValueError):
            require_component_available(1582493743,1582401299)
        with self.assertRaises(ValueError):
            require_component_available(1582312306,1582239167)

    def test_disjoint_half_open_periods(self):
        require_disjoint_outcome_intervals((1,5),(5,10))
        with self.assertRaises(ValueError):
            require_disjoint_outcome_intervals((1,5),(1,5))
        with self.assertRaises(ValueError):
            require_disjoint_outcome_intervals((1,7),(5,10))

    def test_topk_boundary_ties_and_input_order(self):
        rng=np.random.default_rng(91)
        for _ in range(200):
            ids=rng.permutation(50)
            scores=rng.integers(0,4,len(ids))
            for k in [0,1,10,49,50,60]:
                expected=ids[np.lexsort((ids,-scores))[:k]]
                np.testing.assert_array_equal(stable_topk(ids,scores,k),expected)

    def test_sparse_topk_rows(self):
        x=sparse.csr_matrix([[1,2,2,2],[0,0,3,3]],dtype=float)
        got=stable_topk_rows(x,2)
        np.testing.assert_array_equal(got.toarray(),[[0,2,2,0],[0,0,3,3]])

    def test_candidates_distinct_available_unseen(self):
        candidates=sampled_candidates(np.arange(1100),[0,1,5],10,999,20261020,123)
        self.assertEqual(candidates[0],10)
        self.assertEqual(len(np.unique(candidates)),1000)
        self.assertFalse(np.isin(candidates,[0,1,5]).any())

    def test_candidates_invariant_to_order_and_batching(self):
        ids=np.arange(30)
        first={i:sampled_candidates(ids,[0],i,10,90,i) for i in [1,2,3,4]}
        for i in [4,2,1,3]:
            np.testing.assert_array_equal(first[i],sampled_candidates(ids,[0],i,10,90,i))

    def test_candidate_failures(self):
        for args in [(np.arange(3),[],1,3,1,1),(np.arange(3),[1],1,1,1,1),
                     (np.arange(3),[],4,1,1,1)]:
            with self.assertRaises(ValueError):
                sampled_candidates(*args)

    def test_target_tie_ranking(self):
        self.assertEqual(rank_target([9,2,4,1],[1.,1.,2.,1.],9),4)
        self.assertEqual(rank_target([9,2,4,1],[1.,1.,2.,1.],2),3)

    def test_sparse_fusion_matches_exhaustive_catalog(self):
        rng=np.random.default_rng(72)
        for _ in range(500):
            n=int(rng.integers(2,70))
            eligible=np.flatnonzero(rng.random(n)>.2)
            seen=np.flatnonzero(rng.random(n)>.8)
            ids=np.flatnonzero(rng.random(n)>.4)
            values=rng.integers(0,5,len(ids)).astype(float)/rng.choice([1,10])
            auxiliary=rng.integers(-2,5,n).astype(float)
            available=np.setdiff1d(eligible,seen)
            for gamma in [0,.1,1,5]:
                base=np.zeros(n)
                base[ids]=values
                scale=float(base[available].max()) if len(available) else 0
                scale=scale if scale>0 else 1
                full=base/scale+gamma*auxiliary
                expected=stable_topk(available,full[available],10)
                got=exact_sparse_fusion(ids,values,auxiliary,eligible,seen,gamma,10)
                np.testing.assert_array_equal(got,expected)

    def test_negative_base_fails(self):
        with self.assertRaises(ValueError):
            exact_sparse_fusion([0],[-1],[1,2],[0,1],[],.1)

    def test_metric_formula_and_existing_code(self):
        recs=np.array([[1,3,2],[4,2,1]])
        relevance=sparse.csr_matrix([[0,1,1,0,0],[0,1,0,0,1]])
        a=full_catalog_user_metrics(recs[0],[1,2],3)
        self.assertAlmostEqual(a['recall'],1)
        self.assertAlmostEqual(a['precision'],2/3)
        self.assertAlmostEqual(a['HR'],1)
        existing=evaluate(recs,relevance,k=3)
        b=full_catalog_user_metrics(recs[1],[1,4],3)
        for metric in a:
            self.assertAlmostEqual(existing[metric],(a[metric]+b[metric])/2)

    def test_empty_relevance_rejected(self):
        with self.assertRaises(ValueError):
            full_catalog_user_metrics([1,2],[],3)


if __name__=='__main__':
    unittest.main()
