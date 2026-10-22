"""Retrospective population-matched calibration for the conditional benchmark."""
import json
from datetime import datetime,timezone

import numpy as np
import pandas as pd

from rebuild_features import REV,VALIDATION_END
from artifact_hash import sha256
from calibration_core import fit_validation_sigmoid,apply_sigmoid
from analyse_and_calibrate_contact_models import probability_metrics


def main():
    base=REV/'contact_analysis'
    output=base/'conditional_calibration'
    output.mkdir(exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('Conditional calibration outputs already exist')
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'motivation':'The all-attempt calibrator showed calibration mismatch on conditional sampled cases',
        'status':'retrospective corrective analysis after inspecting prior test diagnostics, not an untouched test',
        'method':'same fixed monotonic sigmoid; fit eligible validation cases only; no test optimization',
        'target_population':'conditional alternative-contact benchmark, not all job seekers or deployment traffic',
        'deployment_limit':'Eligibility depends on a later alternative contact; this conditional calibrator cannot be selected for production users using future eligibility',
        'comparison_policy':'Report raw, all-attempt-calibrated and conditional-calibrated results; retain unfavorable findings',
        'downstream_rankings_started':False}
    (output/'analysis_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    cases=pd.read_parquet(REV/'protocol_checks/evaluation_inputs/sampled_cases.parquet')
    metrics=[];calibrators={};exports={};source_hashes={};frames={}
    for variant in ['primitive_evidence','context_control']:
        source=base/'scores'/f'successful_contact__{variant}__validation.parquet'
        val=pd.read_parquet(source)
        source_hashes[source.name]=sha256(source)
        selected=cases[cases.split=='validation'][['attempt_id','candidate_row']].merge(val,on='attempt_id',validate='one_to_one')
        assert len(selected)==21046
        params=fit_validation_sigmoid(selected,VALIDATION_END)
        calibrators[variant]=params
        (output/f'calibrator_{variant}.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
        for split in ['validation','test']:
            source=base/'scores'/f'successful_contact__{variant}__{split}.parquet'
            source_hashes[source.name]=sha256(source)
            scores=pd.read_parquet(source)
            frame=cases[cases.split==split][['attempt_id','candidate_row']].merge(scores,on='attempt_id',validate='one_to_one')
            frame['conditional_bim_probability']=apply_sigmoid(frame.risk_raw,params)
            for version,col in [('raw','risk_raw'),('all_attempt_calibration','bim_probability'),
                                ('conditional_calibration','conditional_bim_probability')]:
                metrics.append({'variant':variant,'split':split,'version':version,
                    **probability_metrics(frame.nonprogression_q90,frame[col])})
            path=output/f'{variant}__{split}.parquet'
            frame.to_parquet(path,index=False,compression='zstd')
            exports[path.name]={'sha256':sha256(path),'rows':len(frame)}
            if variant=='primitive_evidence':frames[split]=frame
    controls=[]
    val=frames['validation']
    mean=float(val.conditional_bim_probability.mean())
    pathway_means=val.groupby('source_event').conditional_bim_probability.mean().to_dict()
    for split,frame in frames.items():
        f=frame[['attempt_id','candidate_row','split','source_event','source_timestamp',
            'risk_raw','bim_probability','conditional_bim_probability']].copy()
        f['fixed_global_conditional_probability']=mean
        f['fixed_pathway_conditional_probability']=f.source_event.map(pathway_means)
        f['shuffled_within_pathway_conditional_probability']=np.nan
        rng=np.random.default_rng(20261020 if split=='validation' else 20261021)
        for event,indices in f.groupby('source_event').groups.items():
            f.loc[indices,'shuffled_within_pathway_conditional_probability']=rng.permutation(f.loc[indices,'conditional_bim_probability'].to_numpy())
        controls.append(f)
    control=pd.concat(controls).sort_values('candidate_row').reset_index(drop=True)
    np.testing.assert_array_equal(control.attempt_id,cases.attempt_id)
    path=output/'sampled_conditional_control_inputs.parquet'
    control.to_parquet(path,index=False,compression='zstd')
    pd.DataFrame(metrics).to_csv(output/'probability_metrics.csv',index=False)
    summary={'status':'completed_retrospective_conditional_analysis','metrics':metrics,'calibrators':calibrators,
        'source_sha256':source_hashes,'exports':exports,'control_file':path.name,
        'control_sha256':sha256(path),'fixed_validation_mean':mean,'fixed_validation_pathway_means':pathway_means,
        'available_from':VALIDATION_END,'eligible_validation_cases':21046,'eligible_test_cases':12783,
        'purpose':'Conditional benchmark scores only; not a validated production probability model',
        'selection_status':'Corrective reanalysis after prior diagnostics; no confirmatory interpretation',
        'downstream_ranking_comparisons_started':False}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'status':summary['status'],'test_metrics':[r for r in metrics if r['split']=='test']},indent=2))


if __name__=='__main__':main()
