import sys
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from comparison_adapter import build_rp3beta, rank_full_rows, sampled_target_ranks, select_operating_point


class ComparisonAdapterTests(unittest.TestCase):
    def test_graph_dense_oracle_and_blocks(self):
        x = np.array([[1,1,0,0],[0,1,1,0],[1,0,1,1]], dtype=float)
        p = x / x.sum(1)[:,None]
        reverse = x.T / x.sum(0)[:,None]
        expected = (reverse @ p) * (x.sum(0) ** -.3)[None,:]
        np.fill_diagonal(expected, 0)
        a, b = build_rp3beta(sparse.csr_matrix(x), topk=4, block_size=1)
        c, d = build_rp3beta(sparse.csr_matrix(x), topk=4, block_size=3)
        np.testing.assert_allclose(a.toarray(),p)
        np.testing.assert_allclose(b.toarray(),expected)
        np.testing.assert_allclose(b.toarray(),d.toarray())

    def test_global_similarity_boundary(self):
        _, sim = build_rp3beta(sparse.csr_matrix(np.ones((2,6))), topk=2)
        self.assertEqual(sim[5].indices.tolist(), [0,1])

    def test_full_and_sampled_shared_order(self):
        scores = sparse.csr_matrix([[0,.5,.5,0]])
        seen = sparse.csr_matrix([[1,0,0,0]])
        output = rank_full_rows(scores,seen,np.arange(4),np.zeros(4),0,k=3)
        self.assertEqual(output.tolist(), [[1,2,3]])
        self.assertEqual(sampled_target_ranks([[3,2,1]],[[0,.5,.5]],[2]).tolist(),[2])

    def test_selection_rejects_test_and_enforces_constraint(self):
        rows = [dict(split='development',ndcg=.4,recall=.5,gamma=0,coverage=.8),
                dict(split='development',ndcg=.9,recall=.9,gamma=1,coverage=.5),
                dict(split='development',ndcg=.4,recall=.5,gamma=.1,coverage=.8)]
        self.assertEqual(select_operating_point(rows,.8)['gamma'],0)
        with self.assertRaises(ValueError):
            select_operating_point([dict(rows[0],split='test')],.8)


if __name__ == '__main__': unittest.main()
