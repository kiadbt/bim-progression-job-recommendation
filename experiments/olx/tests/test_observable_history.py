import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from observable_history import build_histories, label_admissible_before


def frame(times, labels=None, horizons=None, groups=None, pathways=None):
    n = len(times)
    return pd.DataFrame({'sequence_id': groups if groups is not None else [1]*n,
                         'source_event': pathways if pathways is not None else ['chat']*n,
                         'source_timestamp': times,
                         'q90_seconds': horizons if horizons is not None else [10]*n,
                         'progressed_q90': labels if labels is not None else [1]*n})


class ObservableHistoryTests(unittest.TestCase):
    def test_first_attempt(self):
        r = build_histories(frame([100]))
        self.assertEqual(r.prior_attempt_count.tolist(), [0])
        self.assertTrue(np.isnan(r.previous_source_timestamp.iloc[0]))

    def test_unfinished_positive_is_not_counted(self):
        r = build_histories(frame([100, 105]))
        self.assertEqual(r.prior_progression_count_q90.tolist(), [0, 0])
        self.assertEqual(r.prior_unresolved_attempt_count_q90.tolist(), [0, 1])

    def test_unfinished_negative_is_not_counted(self):
        r = build_histories(frame([100, 105], [0, 1]))
        self.assertEqual(r.prior_nonprogression_count_q90.tolist(), [0, 0])

    def test_strict_horizon_boundary(self):
        r = build_histories(frame([100, 110, 111], [0, 1, 1]))
        self.assertEqual(r.prior_nonprogression_count_q90.tolist(), [0, 0, 1])

    def test_equal_timestamps_are_not_prior(self):
        r = build_histories(frame([100, 100, 101], horizons=[0, 0, 0]))
        self.assertEqual(r.prior_attempt_count.tolist(), [0, 0, 2])
        self.assertEqual(r.prior_progression_count_q90.tolist(), [0, 0, 2])

    def test_pair_isolation(self):
        r = build_histories(frame([100, 200], groups=[1, 2]))
        self.assertEqual(r.prior_attempt_count.tolist(), [0, 0])

    def test_pathway_isolation(self):
        r = build_histories(frame([100, 200], pathways=['chat', 'phone']))
        self.assertEqual(r.prior_attempt_count.tolist(), [0, 0])

    def test_future_label_mutation_invariance(self):
        d = frame([100, 105, 200])
        before = build_histories(d)
        d.loc[0, 'progressed_q90'] = 0
        after = build_histories(d)
        pd.testing.assert_frame_equal(before.iloc[:2], after.iloc[:2])
        self.assertNotEqual(before.iloc[2].prior_progression_count_q90,
                            after.iloc[2].prior_progression_count_q90)

    def test_future_rows_do_not_change_past(self):
        d = frame([100, 120])
        pd.testing.assert_frame_equal(build_histories(d),
                                      build_histories(pd.concat([d, frame([300])], ignore_index=True)).iloc[:2])

    def test_unsorted_variable_horizons(self):
        r = build_histories(frame([150, 100, 130], [1, 0, 1], [10, 100, 2]))
        self.assertEqual(r.prior_progression_count_q90.tolist(), [1, 0, 0])
        self.assertEqual(r.prior_nonprogression_count_q90.tolist(), [0, 0, 0])

    def test_empty(self):
        self.assertTrue(build_histories(frame([])).empty)

    def test_invalid_labels(self):
        with self.assertRaises(ValueError): build_histories(frame([100], [2]))

    def test_negative_horizon(self):
        with self.assertRaises(ValueError): build_histories(frame([100], horizons=[-1]))

    def test_missing_values(self):
        with self.assertRaises(ValueError): build_histories(frame([np.nan]))

    def test_fit_cutoff_purges_boundary(self):
        d = frame([100, 101, 102]); d['q95_seconds'] = 10
        self.assertEqual(label_admissible_before(d, 111).tolist(), [True, False, False])

    def test_matches_brute_force(self):
        rng = np.random.default_rng(271)
        d = frame(rng.integers(100, 200, 250), rng.integers(0, 2, 250),
                  rng.integers(0, 20, 250), rng.integers(0, 8, 250),
                  rng.choice(['chat', 'phone'], 250))
        actual = build_histories(d)
        for i, a in d.iterrows():
            same = (d.sequence_id == a.sequence_id) & (d.source_event == a.source_event)
            raw = same & (d.source_timestamp < a.source_timestamp)
            mature = same & (d.source_timestamp + d.q90_seconds < a.source_timestamp)
            self.assertEqual(actual.loc[i, 'prior_attempt_count'], int(raw.sum()))
            self.assertEqual(actual.loc[i, 'prior_progression_count_q90'], int((mature & (d.progressed_q90 == 1)).sum()))
            self.assertEqual(actual.loc[i, 'prior_nonprogression_count_q90'], int((mature & (d.progressed_q90 == 0)).sum()))


if __name__ == '__main__': unittest.main()
