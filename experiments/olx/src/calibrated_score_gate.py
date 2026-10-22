"""Read verified, population-matched BIM inputs without running rankings."""
import json

import numpy as np
import pandas as pd

from rebuild_features import REV,VALIDATION_END
from artifact_hash import sha256
from approved_evaluation_inputs import load_approved_inputs


def load_bim_controls(task,representation,acknowledge_retrospective_population=False):
    if not acknowledge_retrospective_population:
        raise ValueError('Conditional benchmark inputs require their retrospective population restriction')
    if task not in ['sampled_test','sampled_validation_development']:
        raise ValueError('Current BIM models are not approved for full-catalog or production use')
    if representation not in ['conditional_primary','raw_sensitivity']:
        raise ValueError('All-attempt calibration is not approved for the conditional benchmark')
    base=REV/'contact_analysis'
    verification=json.loads((base/'verification.json').read_text(encoding='utf-8'))
    assert verification['status']=='passed_with_population_and_retrospective_restrictions'
    assert sha256(base/'summary.json')==verification['summary_sha256']
    assert sha256(base/'conditional_calibration/summary.json')==verification['conditional_summary_sha256']
    assert sha256(REV/'src/calibration_core.py')==verification['calibration_code_sha256']
    plan=json.loads((base/'risk_comparison_preconditions.json').read_text(encoding='utf-8'))
    assert sha256(base/'risk_comparison_preconditions.json')==verification['risk_plan_sha256']
    assert sha256(base/plan['input_file'])==plan['input_sha256']
    manifest=load_approved_inputs('sampled_test',VALIDATION_END,is_learned_bim=True)
    frame=pd.read_parquet(base/plan['input_file'])
    cases=pd.read_parquet(REV/'protocol_checks/evaluation_inputs/sampled_cases.parquet')
    np.testing.assert_array_equal(frame.attempt_id,cases.attempt_id)
    split='test' if task=='sampled_test' else 'validation'
    selected=frame[frame.split==split].copy()
    if split=='test':assert (selected.source_timestamp>=VALIDATION_END).all()
    columns=({'actual':'conditional_bim_probability','fixed_global':'fixed_global_conditional_probability',
              'fixed_pathway':'fixed_pathway_conditional_probability',
              'shuffled_pathway':'shuffled_within_pathway_conditional_probability',
              'context':'context_conditional_probability'} if representation=='conditional_primary' else
             {'actual':'risk_raw','fixed_global':'fixed_global_raw_score','fixed_pathway':'fixed_pathway_raw_score',
              'shuffled_pathway':'shuffled_within_pathway_raw_score'})
    return selected[['candidate_row','attempt_id','split',*columns.values()]],columns,manifest


if __name__=='__main__':
    for representation in ['conditional_primary','raw_sensitivity']:
        data,columns,_=load_bim_controls('sampled_test',representation,True)
        assert len(data)==12783
    for task,representation in [('full_catalog_later','conditional_primary'),
                                 ('sampled_test','all_attempt_probability')]:
        try:load_bim_controls(task,representation,True)
        except ValueError:pass
        else:raise AssertionError('Unsupported score use was not rejected')
    print('Verified sampled score inputs load correctly; unsupported populations rejected; no rankings executed')
