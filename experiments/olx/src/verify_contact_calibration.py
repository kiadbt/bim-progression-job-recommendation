"""Verify score lineage, calibration exports and negative-control inputs."""
import json

import numpy as np
import pandas as pd

from rebuild_features import ROOT,REV,VALIDATION_END
from artifact_hash import sha256
from calibration_core import apply_sigmoid
from analyse_and_calibrate_contact_models import cluster_loss_comparison


def check_controls(frame,actual,shuffled,global_col,path_col):
    assert frame.attempt_id.is_unique
    assert np.isfinite(frame[[actual,shuffled,global_col,path_col]]).all().all()
    for _,group in frame.groupby(['split','source_event']):
        np.testing.assert_array_equal(np.sort(group[actual]),np.sort(group[shuffled]))
    assert frame[global_col].nunique()==1
    for _,group in frame.groupby('source_event'):assert group[path_col].nunique()==1


def main():
    base=REV/'contact_analysis'
    summary=json.loads((base/'summary.json').read_text(encoding='utf-8'))
    assert summary['status']=='completed' and not summary['downstream_ranking_comparisons_started']
    for relative,digest in summary['input_sha256'].items():assert sha256(REV/relative)==digest
    count=0
    for name,record in summary['score_exports'].items():
        path=base/'scores'/name
        assert sha256(path)==record['sha256']
        frame=pd.read_parquet(path)
        key=name.rsplit('__',1)[0]
        params=summary['calibrators'][key]
        np.testing.assert_allclose(frame.bim_probability,apply_sigmoid(frame.risk_raw,params),rtol=0,atol=1e-12)
        assert frame.attempt_id.is_unique and len(frame)==record['rows']
        assert params['available_from']==VALIDATION_END and params['slope']>0
        if (frame.split=='test').all():assert (frame.source_timestamp>=VALIDATION_END).all()
        else:assert (frame.source_timestamp+frame.q95_seconds<VALIDATION_END).all()
        count+=len(frame)
    control=base/summary['control_inputs']['file']
    assert sha256(control)==summary['control_inputs']['sha256']
    cases=pd.read_parquet(REV/'protocol_checks/evaluation_inputs/sampled_cases.parquet')
    frame=pd.read_parquet(control)
    np.testing.assert_array_equal(frame.attempt_id,cases.attempt_id)
    check_controls(frame,'bim_probability','shuffled_within_pathway_probability',
                   'fixed_global_probability','fixed_pathway_probability')
    conditional=base/'conditional_calibration'
    cs=json.loads((conditional/'summary.json').read_text(encoding='utf-8'))
    assert cs['available_from']==VALIDATION_END and not cs['downstream_ranking_comparisons_started']
    for name,digest in cs['source_sha256'].items():assert sha256(base/'scores'/name)==digest
    for name,record in cs['exports'].items():
        path=conditional/name
        assert sha256(path)==record['sha256']
        f=pd.read_parquet(path)
        variant=name.split('__')[0]
        np.testing.assert_allclose(f.conditional_bim_probability,apply_sigmoid(f.risk_raw,cs['calibrators'][variant]),rtol=0,atol=1e-12)
    f=pd.read_parquet(conditional/cs['control_file'])
    assert sha256(conditional/cs['control_file'])==cs['control_sha256']
    np.testing.assert_array_equal(f.attempt_id,cases.attempt_id)
    check_controls(f,'conditional_bim_probability','shuffled_within_pathway_conditional_probability',
                   'fixed_global_conditional_probability','fixed_pathway_conditional_probability')
    a=pd.read_parquet(conditional/'primitive_evidence__test.parquet')
    b=pd.read_parquet(conditional/'context_control__test.parquet')
    a['bim_probability']=a.conditional_bim_probability
    b['bim_probability']=b.conditional_bim_probability
    paired=cluster_loss_comparison(a,b)
    (conditional/'paired_prediction_loss.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    risk_plan=json.loads((base/'risk_comparison_preconditions.json').read_text(encoding='utf-8'))
    assert not risk_plan['downstream_ranking_comparisons_started']
    assert sha256(base/risk_plan['input_file'])==risk_plan['input_sha256']
    controls=pd.read_parquet(base/risk_plan['input_file'])
    np.testing.assert_array_equal(controls.attempt_id,cases.attempt_id)
    check_controls(controls,'risk_raw','shuffled_within_pathway_raw_score',
                   'fixed_global_raw_score','fixed_pathway_raw_score')
    check_controls(controls,'conditional_bim_probability','shuffled_within_pathway_conditional_probability',
                   'fixed_global_conditional_probability','fixed_pathway_conditional_probability')
    record={'status':'passed_with_population_and_retrospective_restrictions',
        'all_attempt_score_files':len(summary['score_exports']),'all_attempt_score_rows':count,
        'conditional_cases':len(f),'ranking_control_rows_verified':len(controls),
        'calibration_code_sha256':sha256(REV/'src/calibration_core.py'),
        'risk_plan_sha256':sha256(base/'risk_comparison_preconditions.json'),
        'checks':['input checkpoint and prediction hashes','exact calibration replay',
            'split maturity and availability','score and target alignment','within-pathway shuffle distribution',
            'validation-only fixed controls','conditional population calibration exports'],
        'summary_sha256':sha256(base/'summary.json'),'conditional_summary_sha256':sha256(conditional/'summary.json'),
        'ranking_policy':'No ranking comparisons started; both calibration versions and raw scores retained',
        'restrictions':['All-attempt calibration is not approved as conditional-cohort calibration',
            'Conditional calibration is retrospective and is not a deployment eligibility rule',
            'Current checkpoints remain unavailable at the earlier full-catalog evaluation start'],
        'ranking_representations':['Conditional primary representation and raw-score sensitivity; no test-based representation selection'],
        'scope':'Conditional benchmark calibration only; deployment calibration is not established'}
    (base/'verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({'verification':record,'conditional_paired_prediction_loss':paired},indent=2))


if __name__=='__main__':main()
