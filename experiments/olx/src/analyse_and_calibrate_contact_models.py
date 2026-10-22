"""Consolidate corrected contact models and export validation-calibrated scores."""
from __future__ import annotations

import json
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss

from calibration_core import fit_validation_sigmoid,apply_sigmoid
from artifact_hash import sha256
from rebuild_features import ROOT,REV,VALIDATION_END
from approved_evaluation_inputs import load_approved_inputs

SEEDS=[20260827,20260828,20260829]


def probability_metrics(y,p):
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    bins=np.minimum((p*15).astype(int),14)
    ece=sum(np.mean(bins==b)*abs(float(y[bins==b].mean())-float(p[bins==b].mean()))
            for b in range(15) if np.any(bins==b))
    return {'rows':len(y),'label_prevalence':float(y.mean()),'mean_probability':float(p.mean()),
        'roc_auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),
        'log_loss':float(log_loss(y,p)),'brier_score':float(brier_score_loss(y,p)),
        'ece_15_bins':float(ece)}


def cluster_loss_comparison(a,b,replicates=2000):
    joined=a[['attempt_id','user_idx','nonprogression_q90','bim_probability']].merge(
        b[['attempt_id','bim_probability']],on='attempt_id',validate='one_to_one',suffixes=('_a','_b'))
    assert len(joined)==len(a)==len(b)
    y=joined.nonprogression_q90.to_numpy()
    pa=joined.bim_probability_a.to_numpy();pb=joined.bim_probability_b.to_numpy()
    rows=[]
    for name,la,lb in [('brier',(pa-y)**2,(pb-y)**2),
        ('log_loss',-(y*np.log(np.clip(pa,1e-12,1))+(1-y)*np.log(np.clip(1-pa,1e-12,1))),
                    -(y*np.log(np.clip(pb,1e-12,1))+(1-y)*np.log(np.clip(1-pb,1e-12,1))))]:
        grouped=pd.DataFrame({'user':joined.user_idx,'delta':la-lb}).groupby('user').delta.agg(['sum','count'])
        totals=grouped['sum'].to_numpy();counts=grouped['count'].to_numpy()
        rng=np.random.default_rng(20261020)
        samples=np.empty(replicates)
        for start in range(0,replicates,25):
            n=min(25,replicates-start)
            ix=rng.integers(0,len(grouped),size=(n,len(grouped)))
            samples[start:start+n]=totals[ix].sum(1)/counts[ix].sum(1)
        lo,hi=np.quantile(samples,[.025,.975])
        rows.append({'comparison':'primitive_evidence_minus_context_control','metric':name,
            'difference':float((la-lb).mean()),'ci95_lower':float(lo),'ci95_upper':float(hi),
            'attempts':len(joined),'user_clusters':len(grouped),'replicates':replicates,
            'interpretation':'Negative means lower prediction loss; exploratory pointwise interval; not recommendation improvement'})
    return rows


def main():
    started=time.time()
    manifest=load_approved_inputs('sampled_test',VALIDATION_END,is_learned_bim=True)
    output=REV/'contact_analysis'
    output.mkdir(exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('Analysis folder must be empty')
    (output/'scores').mkdir()
    # Fix the analysis before reading test prediction values.
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'calibration':'one monotonic sigmoid of ensemble mean logit, fit on all admissible validation attempts',
        'positive_class':'Q90 nonprogression pseudo-label, not verified dislike or psychological risk',
        'weighting':'unweighted calibration to observed attempt distribution; no Q or transition weighting',
        'method_selection':'fixed sigmoid, no test-based choice between calibration methods',
        'validation_role':'development and apparent calibration fit, not held-out calibration evidence',
        'test_role':'retrospective temporal assessment; previously inspected period',
        'ensemble':'fixed equal mean of three seeds; engineered-Q reference has only one seed',
        'paired_inference':'2000 paired user-cluster bootstrap replicates for pooled test Brier and log loss; exploratory',
        'downstream_rankings':'not run',
        'operational_limit':'calibration on all attempts need not transfer to conditional ranking cases or real-world harm'}
    (output/'analysis_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    original_results=json.loads((REV/'model_reruns/summary.json').read_text(encoding='utf-8'))
    assert original_results['status']=='completed' and len(original_results['results'])==13
    aggregate=pd.DataFrame(original_results['results']).groupby(['outcome','variant']).agg(
        seeds=('seed','count'),roc_auc_mean=('roc_auc','mean'),roc_auc_sd=('roc_auc','std'),
        average_precision_mean=('average_precision','mean'),average_precision_sd=('average_precision','std'),
        raw_brier_mean=('brier_score','mean'),raw_log_loss_mean=('log_loss','mean')).reset_index()
    aggregate.to_csv(output/'seed_summary.csv',index=False)
    raw_calibrated=[];reliability=[];calibrators={};score_manifest={};test_frames={};input_hashes={}
    sampled=pd.read_parquet(REV/'protocol_checks/evaluation_inputs/sampled_cases.parquet')
    groups=[('successful_contact','primitive_evidence'),('successful_contact','context_control'),
            ('successful_contact','engineered_Q_weighted'),('contact_chat','primitive_evidence'),
            ('phone_contact','primitive_evidence')]
    for outcome,variant in groups:
        print(f'Analysing and calibrating {outcome} / {variant}',flush=True)
        key=f'{outcome}__{variant}'
        feature_path=REV/'features'/f'bim_features_{outcome}.parquet'
        features=pd.read_parquet(feature_path,columns=['attempt_id','split','user_idx','source_event',
            'source_timestamp','q95_seconds','nonprogression_q90'])
        feature_hash=sha256(feature_path)
        frames={};params=None
        seeds=SEEDS if variant!='engineered_Q_weighted' else SEEDS[:1]
        for split in ['validation','test']:
            frame=features[features.split==split].copy()
            probabilities=[]
            for seed in seeds:
                directory=REV/'model_reruns'/outcome/variant/f'seed_{seed}'
                cfg=json.loads((directory/'config.json').read_text(encoding='utf-8'))
                assert cfg['feature_sha256']==feature_hash
                for name,digest in cfg['dependency_sha256'].items():assert sha256(ROOT/'src'/name)==digest
                path=directory/f'{split}_predictions.parquet'
                p=pd.read_parquet(path)
                assert p.attempt_id.is_unique and len(p)==len(frame)
                aligned=frame[['attempt_id','nonprogression_q90']].merge(p,on='attempt_id',validate='one_to_one',suffixes=('_feature','_pred'))
                assert (aligned.nonprogression_q90_feature==aligned.nonprogression_q90_pred).all()
                probabilities.append(aligned.inc_score.to_numpy(dtype=float))
                for source in [path,directory/'config.json',directory/'model.pt']:
                    input_hashes[str(source.relative_to(REV))]=sha256(source)
            matrix=np.stack(probabilities,axis=1)
            frame['risk_raw']=matrix.mean(axis=1)
            frame['seed_score_sd']=matrix.std(axis=1,ddof=1) if len(seeds)>1 else np.nan
            if split=='validation':
                params=fit_validation_sigmoid(frame,VALIDATION_END)
                calibrators[key]=params
                (output/f'calibrator_{key}.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
            frame['bim_probability']=apply_sigmoid(frame.risk_raw,params)
            if split=='test':assert (frame.source_timestamp>=VALIDATION_END).all()
            frames[split]=frame
            for version,column in [('raw','risk_raw'),('calibrated','bim_probability')]:
                y=frame.nonprogression_q90.to_numpy();p=frame[column].to_numpy()
                raw_calibrated.append({'outcome':outcome,'variant':variant,'split':split,'version':version,
                    'validation_status':'apparent_fit' if split=='validation' else 'retrospective_assessment',**probability_metrics(y,p)})
                bins=np.minimum((p*15).astype(int),14)
                for b in range(15):
                    mask=bins==b
                    if mask.any():reliability.append({'outcome':outcome,'variant':variant,'split':split,'version':version,
                        'bin':b,'count':int(mask.sum()),'mean_score':float(p[mask].mean()),'observed_rate':float(y[mask].mean())})
            score_path=output/'scores'/f'{key}__{split}.parquet'
            frame.to_parquet(score_path,index=False,compression='zstd')
            score_manifest[score_path.name]={'sha256':sha256(score_path),'rows':len(frame),
                'use':'development_only' if split=='validation' else 'sampled_test_queries_after_calibration_availability'}
        test_frames[key]=frames['test']
        if outcome=='successful_contact' and variant=='primitive_evidence':
            for split,frame in frames.items():
                for event,g in frame.groupby('source_event'):
                    for version,col in [('raw','risk_raw'),('calibrated','bim_probability')]:
                        raw_calibrated.append({'outcome':outcome,'variant':variant,'split':split,'version':version,
                            'subgroup':event,**probability_metrics(g.nonprogression_q90,g[col])})
                selected=sampled[sampled.split==split][['attempt_id','candidate_row']].merge(frame,on='attempt_id',validate='one_to_one')
                assert len(selected)==sum(sampled.split==split)
                for version,col in [('raw','risk_raw'),('calibrated','bim_probability')]:
                    raw_calibrated.append({'outcome':outcome,'variant':variant,'split':split,'version':version,
                        'subgroup':'eligible_sampled_cases',**probability_metrics(selected.nonprogression_q90,selected[col])})
    metrics=pd.DataFrame(raw_calibrated)
    metrics.to_csv(output/'probability_metrics.csv',index=False)
    pd.DataFrame(reliability).to_csv(output/'reliability_bins.csv',index=False)
    print('Computing paired user-cluster loss intervals...',flush=True)
    paired=cluster_loss_comparison(test_frames['successful_contact__primitive_evidence'],
                                   test_frames['successful_contact__context_control'])
    pd.DataFrame(paired).to_csv(output/'paired_prediction_loss.csv',index=False)
    # Control inputs only: no recommendation scores, ranks or ranking metrics are calculated.
    pooled_val=pd.read_parquet(output/'scores/successful_contact__primitive_evidence__validation.parquet')
    val_means=pooled_val.groupby('source_event').bim_probability.mean().to_dict()
    global_mean=float(pooled_val.bim_probability.mean())
    control_frames=[]
    for split in ['validation','test']:
        scores=pd.read_parquet(output/'scores'/f'successful_contact__primitive_evidence__{split}.parquet')
        cases=sampled[sampled.split==split].merge(scores[['attempt_id','bim_probability']],on='attempt_id',validate='one_to_one')
        cases=cases[['candidate_row','attempt_id','split','source_event','source_timestamp','bim_probability']].sort_values('candidate_row')
        cases['fixed_global_probability']=global_mean
        cases['fixed_pathway_probability']=cases.source_event.map(val_means)
        cases['shuffled_within_pathway_probability']=np.nan
        rng=np.random.default_rng(20261020 if split=='validation' else 20261021)
        for event,indices in cases.groupby('source_event').groups.items():
            cases.loc[indices,'shuffled_within_pathway_probability']=rng.permutation(cases.loc[indices,'bim_probability'].to_numpy())
            np.testing.assert_array_equal(np.sort(cases.loc[indices,'shuffled_within_pathway_probability']),
                                           np.sort(cases.loc[indices,'bim_probability']))
        control_frames.append(cases)
    controls=pd.concat(control_frames).sort_values('candidate_row').reset_index(drop=True)
    np.testing.assert_array_equal(controls.attempt_id,sampled.attempt_id)
    control_path=output/'sampled_bim_control_inputs.parquet'
    controls.to_parquet(control_path,index=False,compression='zstd')
    old=json.loads((ROOT/'results/E8D_evidence_no_Q_cross_outcome/cross_outcome_summary.json').read_text(encoding='utf-8'))
    historical=[{'outcome':r['outcome'],'old_auc_mean':r['roc_auc_mean'],
        'old_ap_mean':r['average_precision_mean'],'interpretation':'Historical context only; horizons, contexts, label eligibility and histories changed together'} for r in old['results']]
    summary={'status':'completed','completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'seed_summary':json.loads(aggregate.to_json(orient='records')),'probability_metrics':json.loads(metrics.to_json(orient='records')),
        'calibrators':calibrators,'paired_prediction_loss':paired,'historical_references':historical,
        'input_sha256':input_hashes,'score_exports':score_manifest,
        'control_inputs':{'file':control_path.name,'sha256':sha256(control_path),'rows':len(controls),
            'fixed_values_fit_on':'all admissible validation attempts','global_fixed_probability':global_mean,
            'pathway_fixed_probabilities':val_means,'shuffle':'within pathway and split; seed 20261020 validation / 20261021 test'},
        'availability_boundary':VALIDATION_END,'downstream_ranking_comparisons_started':False,
        'limits':['Nonprogression is an observed-window pseudo-label, not psychological risk or proven dislike',
            'Calibration assesses probabilities, not recommendation utility or causal effects',
            'Seed standard deviations are optimization variability, not population confidence intervals',
            'Engineered-Q reference changes both features and loss weighting',
            'Context control retains correlated behavioral counts',
            'Current calibrated scores must not be used in full-catalog evaluation starting 1582401299',
            'Validation calibration performance is apparent fit; test period was previously inspected'],
        'duration_seconds':time.time()-started}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':'completed','paired_prediction_loss':paired,'duration_seconds':summary['duration_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
