"""Freeze risk representations and matched controls before ranking comparisons."""
import json
from datetime import datetime,timezone

import numpy as np
import pandas as pd

from rebuild_features import REV
from artifact_hash import sha256
from calibration_core import apply_sigmoid


def main():
    base=REV/'contact_analysis'
    cs=json.loads((base/'conditional_calibration/summary.json').read_text(encoding='utf-8'))
    source=base/'conditional_calibration'/cs['control_file']
    frame=pd.read_parquet(source)
    val=frame[frame.split=='validation']
    frame['fixed_global_raw_score']=float(val.risk_raw.mean())
    means=val.groupby('source_event').risk_raw.mean().to_dict()
    frame['fixed_pathway_raw_score']=frame.source_event.map(means)
    frame['shuffled_within_pathway_raw_score']=np.nan
    for split,split_frame in frame.groupby('split'):
        rng=np.random.default_rng(20261020 if split=='validation' else 20261021)
        for event,indices in split_frame.groupby('source_event').groups.items():
            frame.loc[indices,'shuffled_within_pathway_raw_score']=rng.permutation(frame.loc[indices,'risk_raw'].to_numpy())
    np.testing.assert_allclose(apply_sigmoid(frame.shuffled_within_pathway_raw_score,cs['calibrators']['primitive_evidence']),
        frame.shuffled_within_pathway_conditional_probability,rtol=0,atol=1e-12)
    context=pd.concat([pd.read_parquet(base/'conditional_calibration'/f'context_control__{s}.parquet')
                       [['attempt_id','conditional_bim_probability']] for s in ['validation','test']])
    context=context.rename(columns={'conditional_bim_probability':'context_conditional_probability'})
    frame=frame.merge(context,on='attempt_id',validate='one_to_one').sort_values('candidate_row').reset_index(drop=True)
    path=base/'ranking_control_inputs.parquet'
    if path.exists():raise FileExistsError(path)
    frame.to_parquet(path,index=False,compression='zstd')
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Retrospective conditional sampled benchmark only; no deployment probability or fresh validation claim',
        'primary_risk_representation':'conditional_bim_probability',
        'primary_controls':['fixed_global_conditional_probability','fixed_pathway_conditional_probability',
                            'shuffled_within_pathway_conditional_probability','context_conditional_probability','zero_risk'],
        'sensitivity_risk_representation':'risk_raw',
        'sensitivity_controls':['fixed_global_raw_score','fixed_pathway_raw_score','shuffled_within_pathway_raw_score','zero_risk'],
        'prohibited_primary_representation':'bim_probability from all-attempt calibration because population mismatch was demonstrated',
        'gamma_grid':[0,.05,.1,.25,.5,1,2],
        'selection':'Same full eligible validation cohort and gamma grid for every risk variant; NDCG@10 then Recall@10 then smaller gamma',
        'test_rule':'Freeze each development-selected gamma before scoring test; do not choose risk representation using test ranking results',
        'reporting':'Report the predeclared primary conditional representation and raw-score sensitivity, even if results disagree',
        'availability':'All learned scoring and calibration selected by 1582493743; current BIM prohibited for full-catalog start 1582401299',
        'input_file':path.name,'input_sha256':sha256(path),'cases':len(frame),
        'conditional_calibration_summary_sha256':sha256(base/'conditional_calibration/summary.json'),
        'control_source_sha256':sha256(source),'downstream_ranking_comparisons_started':False}
    (base/'risk_comparison_preconditions.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    claims={'main_operational_path':{'implementation':'Pre-cutoff global successful-contact-count prior fused with RP3Beta',
            'source':'src/evaluate_deployable_progression_full_catalog_olx.py load_split',
            'uses_mlp':False,'uses_maturity_horizons':False,'uses_negative_gnn':False},
        'auxiliary_bim_path':{'implementation':'Maturity-aware attempt histories and MLP predicting Q90 nonprogression pseudo-labels',
            'status':'Prediction and calibration reruns complete; recommendation effect not yet tested under repaired protocol'},
        'architecture_revision_rule':'Draw separate operational count-fusion and auxiliary BIM paths; no unsupported maturity-to-main-score or dual-GNN connection',
        'supported_so_far':['Corrected primitive evidence has better contact-prediction discrimination and probability loss than the context control in this retrospective cohort',
            'Calibration is population-specific; an overall improvement did not transfer to the conditional ranking cohort'],
        'not_supported_yet':['BIM improves recommendations','A unique progression mechanism explains the full-catalog gain',
            'State-of-the-art accuracy','Independent cross-domain transfer','Deployment or employment benefit'],
        'scope':'Source-outcome prediction and candidate ranking are distinct evaluation tasks'}
    (base/'implementation_claims_map.json').write_text(json.dumps(claims,indent=2),encoding='utf-8')
    print(json.dumps({'status':'control_inputs_prepared_not_ranked','cases':len(frame),'sha256':plan['input_sha256']},indent=2))


if __name__=='__main__':main()
