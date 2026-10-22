"""Algebra, tie, and controlled-scale checks independent of OLX outcomes."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_normalization_experiment import normalized_signal, target_ranks
from ranking_driver_core import zscore_rows


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(14)
        self.x = rng.uniform(0, 6, (7, 31))
        self.y = rng.uniform(0, 6, (7, 31))
        self.r = rng.uniform(0, 1, 7)
        self.r0 = np.full(7, .51)

    def test_existing_formula_exact(self):
        np.testing.assert_array_equal(
            normalized_signal(self.x, self.y, self.r, self.r0, "current"),
            zscore_rows(self.x - self.r[:, None] * self.y))

    def test_reference_control_exact(self):
        np.testing.assert_array_equal(
            normalized_signal(self.x, self.y, self.r0, self.r0, "current"),
            normalized_signal(self.x, self.y, self.r0, self.r0, "reference"))

    def test_shared_scale_preserves_linear_risk_change(self):
        a = normalized_signal(self.x, self.y, self.r, self.r0, "reference")
        b = normalized_signal(self.x, self.y, self.r0, self.r0, "reference")
        sd = (self.x - self.r0[:, None] * self.y).std(axis=1, keepdims=True) + 1e-8
        expected = -(self.r - self.r0)[:, None] * (self.y - self.y.mean(1, keepdims=True)) / sd
        np.testing.assert_allclose(a - b, expected, atol=1e-14)

    def test_collinear_example(self):
        x = np.arange(20, dtype=float)[None, :]
        y = .8 * x + 2
        r0, r = np.array([.2]), np.array([.8])
        c0 = normalized_signal(x, y, r0, r0, "current")
        c1 = normalized_signal(x, y, r, r0, "current")
        np.testing.assert_allclose(c0, c1, atol=1e-8)
        r1 = normalized_signal(x, y, r, r0, "reference")
        self.assertGreater(float(np.abs(r1 - c0).max()), .5)

    def test_zero_profiles_finite(self):
        z = np.zeros((2, 10))
        for mode in ("reference", "current"):
            np.testing.assert_array_equal(
                normalized_signal(z, z, np.array([.1, .9]), np.array([.5, .5]), mode), z)

    def test_target_ranks_random_and_ties(self):
        rng = np.random.default_rng(17)
        ids = np.array([rng.choice(100, 20, replace=False) for _ in range(40)])
        score = rng.integers(0, 4, size=ids.shape).astype(float)
        got = target_ranks(ids, score)
        expected = [int(np.flatnonzero(np.lexsort((i, -s)) == 0)[0]) + 1
                    for i, s in zip(ids, score)]
        np.testing.assert_array_equal(got, expected)

    def test_no_target_column_dependence_of_normalization(self):
        perm = np.arange(31)[::-1]
        for mode in ("reference", "current"):
            original = normalized_signal(self.x, self.y, self.r, self.r0, mode)
            changed = normalized_signal(self.x[:, perm], self.y[:, perm], self.r, self.r0, mode)
            np.testing.assert_allclose(original[:, perm], changed, atol=1e-14)

    def test_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            target_ranks(np.array([[0, 1]]), np.array([[0., np.nan]]))


if __name__ == "__main__":
    unittest.main()
