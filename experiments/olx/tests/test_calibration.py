import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from calibration_core import fit_validation_sigmoid,apply_sigmoid


class CalibrationTests(unittest.TestCase):
    def frame(self):
        rng=np.random.default_rng(24)
        x=rng.normal(size=5000)
        return pd.DataFrame({'split':'validation','source_timestamp':100,'q95_seconds':20,
            'risk_raw':expit(x-1),'nonprogression_q90':rng.binomial(1,expit(x+1))})

    def test_bias_correction_and_monotonicity(self):
        frame=self.frame()
        fit=fit_validation_sigmoid(frame,121)
        raw=frame.risk_raw.to_numpy()
        p=apply_sigmoid(raw,fit)
        y=frame.nonprogression_q90.to_numpy()
        self.assertGreater(fit['slope'],0)
        self.assertLess(np.mean((p-y)**2),np.mean((raw-y)**2))
        self.assertTrue((np.diff(p[np.argsort(raw)])>=0).all())

    def test_reject_test_or_train_fitting(self):
        for split in ['test','train']:
            frame=self.frame();frame.loc[0,'split']=split
            with self.assertRaises(ValueError):fit_validation_sigmoid(frame,121)

    def test_boundary_strictness(self):
        with self.assertRaises(ValueError):fit_validation_sigmoid(self.frame(),120)

    def test_degenerate_labels(self):
        frame=self.frame();frame.nonprogression_q90=1
        with self.assertRaises(ValueError):fit_validation_sigmoid(frame,121)

    def test_test_scores_cannot_change_fitted_parameters(self):
        frame=self.frame()
        before=fit_validation_sigmoid(frame,121)
        apply_sigmoid(np.array([.01,.4,.99]),before)
        after=fit_validation_sigmoid(frame,121)
        self.assertEqual(before,after)


if __name__=='__main__':unittest.main()
