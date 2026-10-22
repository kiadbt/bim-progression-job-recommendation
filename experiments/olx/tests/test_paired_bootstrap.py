import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from paired_bootstrap import bootstrap_means, paired_summary


class BootstrapTests(unittest.TestCase):
    def test_constant(self):
        np.testing.assert_array_equal(bootstrap_means(np.full(12, .2), 200), np.full(200, .2))

    def test_exact_small_distribution(self):
        values = np.array([-1., 0., 2.])
        exact = np.array([np.mean(values[list(i)]) for i in itertools.product(range(3), repeat=3)])
        sampled = bootstrap_means(values, 100000)
        self.assertAlmostEqual(sampled.mean(), exact.mean(), delta=.012)
        self.assertAlmostEqual(sampled.var(), exact.var(), delta=.015)
        self.assertAlmostEqual(np.mean(sampled==2), 1/27, delta=.004)

    def test_permutation_and_reproducibility(self):
        values = np.array([0., .1, -.1, 0., .5])
        np.testing.assert_array_equal(bootstrap_means(values, 1000), bootstrap_means(values[::-1], 1000))

    def test_pairing_and_family_interval(self):
        a = np.array([.4, .5, .6]); b = np.array([.2, .4, .5])
        result, draws = paired_summary(a, b, key='test', primary=True)
        self.assertAlmostEqual(result['difference'], (a-b).mean())
        self.assertLessEqual(result['ci99_bonferroni_five'][0], result['ci95'][0])
        self.assertGreaterEqual(result['ci99_bonferroni_five'][1], result['ci95'][1])
        self.assertEqual(len(draws), 5000)

    def test_invalid(self):
        for values in [[], [np.nan], [[1, 2]]]:
            with self.assertRaises(ValueError):
                bootstrap_means(values)
        with self.assertRaises(ValueError):
            paired_summary([1], [1, 2], key='bad')


if __name__ == '__main__':
    unittest.main()
