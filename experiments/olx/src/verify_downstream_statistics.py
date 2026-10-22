"""Verify saved bootstrap results and numerical consistency."""
import io
import unittest

import numpy as np

from artifact_hash import sha256
from ranking_driver_core import REV, read_json, write_json


def main():
    output=REV/'statistical_analysis'
    manifest=read_json(output/'analysis_manifest.json')
    for name,digest in manifest['files'].items():
        assert sha256(output/name)==digest,name
    for name,digest in manifest['source_code'].items():
        assert sha256(REV/'src'/name)==digest,name
    results=read_json(output/'results.json')
    draws=np.load(output/'bootstrap_replicates.npz')
    assert len(results['contrasts'])==62
    assert sum(c['primary_family'] for c in results['contrasts'])==5
    errors=[]
    for row in results['contrasts']:
        x=draws[row['contrast']]
        assert len(x)==5000 and np.isfinite(x).all()
        np.testing.assert_allclose(np.quantile(x,[.025,.975],method='linear'),row['ci95'],rtol=0,atol=0)
        np.testing.assert_allclose(row['left_mean']-row['right_mean'],row['difference'],rtol=0,atol=1e-13)
        if row['primary_family']:
            np.testing.assert_allclose(np.quantile(x,[.005,.995]),row['ci99_bonferroni_five'],rtol=0,atol=0)
        expected=row['empirical_bootstrap_standard_error']
        if expected>0:
            errors.append(abs(row['bootstrap_standard_error']/expected-1))
        else:
            assert np.all(x==row['difference'])
    assert max(errors)<.06
    suite=unittest.defaultTestLoader.discover(str(REV/'tests'))
    log=io.StringIO(); tested=unittest.TextTestRunner(stream=log,verbosity=1).run(suite)
    assert tested.wasSuccessful(),log.getvalue()
    write_json(output/'verification.json',{
        'status':'passed','contrasts':62,'primary_family_size':5,'replicates_each':5000,
        'bootstrap_draws_verified':310000,'regression_tests':tested.testsRun,
        'max_relative_bootstrap_se_difference_from_exact_empirical_se':max(errors),
        'results_sha256':sha256(output/'results.json'),
        'test_log':log.getvalue()})
    print(f'Verified 62 contrasts, 310000 stored replicates and {tested.testsRun} regression tests')


if __name__=='__main__':main()
