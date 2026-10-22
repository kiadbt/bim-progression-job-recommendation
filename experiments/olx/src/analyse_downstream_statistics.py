"""Analyse the completed retrospective ranking experiments without refitting."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from artifact_hash import sha256
from paired_bootstrap import paired_summary
from ranking_driver_core import REV, INPUTS, read_json, write_json
from run_downstream_batch import verify_output
from run_repaired_ranking import source_hashes

RUN = REV/'downstream_comparisons/run_2026-10-20'
OUT = REV/'statistical_analysis'


def rank_metric(ranks, metric):
    ranks = np.asarray(ranks, dtype=float); hit = ranks<=10
    if metric in ['recall', 'HR']:
        return hit.astype(float)
    if metric == 'ndcg':
        return np.where(hit, 1/np.log2(ranks+1), 0)
    if metric == 'MRR':
        return np.where(hit, 1/ranks, 0)
    raise ValueError(metric)


def main():
    if OUT.exists():
        raise FileExistsError('Analysis output already exists')
    started = time.perf_counter(); OUT.mkdir()
    status = read_json(RUN/'batch_status.json')
    assert status['stage']=='completed'
    verification = {task:verify_output(RUN/folder, task, source_hashes())
                    for task,folder in [('sampled','sampled_run'),('full','full_catalog_run')]}
    contract = read_json(REV/'protocol_checks/comparison_contract.json')
    write_json(OUT/'analysis_specification.json', {
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'contract_sha256':sha256(REV/'protocol_checks/comparison_contract.json'),
        'completed_comparison_verification':verification, 'statistics':contract['statistics'],
        'resampling':'Ordinary paired user bootstrap; exact multinomial compression of identical unrounded differences',
        'replicates':5000, 'master_seed':20261020,
        'random_streams':'Deterministic contrast-key SHA-256 substreams of the master seed',
        'primary_family_size':5, 'percentile_method':'linear',
        'secondary_scope':'Exploratory pointwise intervals; not confirmatory significance claims',
        'selection_uncertainty':'Settings held fixed; neither retuning nor seed-level model uncertainty is resampled',
        'source_hashes':{p.name:sha256(p) for p in [Path(__file__),REV/'src/paired_bootstrap.py']},
        'model_refitting':False})
    records=[]; draws={}

    def compare(group, metric, left_name, right_name, left, right, primary=False):
        key=f'{group}__{metric}__{left_name}_minus_{right_name}'
        record, values = paired_summary(left, right, key=key, primary=primary)
        record.update(group=group, metric=metric, left=left_name, right=right_name)
        records.append(record); draws[key]=values
        print(f'{key}: {record["difference"]:+.9f}; CI {record["ci95"]}', flush=True)

    base=RUN/'full_catalog_run/full_catalog/later_retrospective'
    users=np.load(base/'user_ids.npy'); models=['rp3beta','popularity','contact_count','contact_rate','recency','p3ltr_degree_recency']
    frames={}; full_metrics=[]
    for name in models:
        folder=base/('p3ltr' if name=='p3ltr_degree_recency' else 'rp3beta')
        frame=pd.read_parquet(folder/f'{name}_per_user.parquet')
        np.testing.assert_array_equal(frame.user_idx, users)
        assert frame.user_idx.is_unique
        reported=next(r for r in read_json(folder/'metrics.json')['metrics'] if r['model']==name)
        for metric in ['ndcg','recall','MRR','HR']:
            np.testing.assert_allclose(frame[metric].mean(), reported[metric], atol=1e-14, rtol=0)
        full_metrics.append(reported); frames[name]=frame
    for name in ['rp3beta','popularity','contact_rate','recency','p3ltr_degree_recency']:
        compare('full_primary','ndcg','contact_count',name,frames['contact_count'].ndcg,frames[name].ndcg,True)
    for name in ['popularity','contact_rate','recency','p3ltr_degree_recency']:
        compare('full_exploratory','ndcg',name,'rp3beta',frames[name].ndcg,frames['rp3beta'].ndcg)
    for metric in ['recall','MRR','HR']:
        compare('full_exploratory',metric,'contact_count','rp3beta',frames['contact_count'][metric],frames['rp3beta'][metric])
    tail=frames['rp3beta'].tail_target.to_numpy(); tail_counts={}
    for name in models:
        np.testing.assert_array_equal(frames[name].tail_target,tail)
        tail_counts[name]=int(np.count_nonzero(frames[name].loc[tail,'tail_ndcg']>0))
        if name!='rp3beta':
            compare('tail_exploratory','tail_ndcg',name,'rp3beta',frames[name].loc[tail,'tail_ndcg'],frames['rp3beta'].loc[tail,'tail_ndcg'])
    compare('tail_exploratory','tail_ndcg','contact_count','p3ltr_degree_recency',
            frames['contact_count'].loc[tail,'tail_ndcg'],frames['p3ltr_degree_recency'].loc[tail,'tail_ndcg'])
    bands=np.load(base/'item_bands.npy'); exposures={}
    for name in models:
        folder=base/('p3ltr' if name=='p3ltr_degree_recency' else 'rp3beta')
        rec=np.load(folder/f'{name}_recommendations.npy',mmap_mode='r')
        valid=rec>=0; rec_bands=np.full(rec.shape,-1,dtype=np.int8); rec_bands[valid]=bands[rec[valid]]
        exposures[name]=np.mean(rec_bands==0,axis=1)
        assert valid.all()
    for name in ['contact_count','popularity','recency','p3ltr_degree_recency']:
        compare('exposure_exploratory','tail_exposure',name,'rp3beta',exposures[name],exposures['rp3beta'])

    sampled=RUN/'sampled_run/sampled'
    cases=pd.read_parquet(INPUTS/'sampled_cases.parquet')
    cases=cases[cases.split=='test'].set_index('attempt_id')
    p3=pd.read_parquet(sampled/'p3ltr_paired_ranks.parquet')
    p3=p3[p3.split=='test'].set_index('attempt_id')
    sampled_metrics={}; sampled_frames={}
    for representation in ['conditional_primary','raw_sensitivity']:
        frame=pd.read_parquet(sampled/representation/'test/paired_ranks.parquet')
        assert frame.user_idx.is_unique and len(frame)==12783
        np.testing.assert_array_equal(frame.candidate_row,cases.loc[frame.attempt_id].candidate_row)
        np.testing.assert_array_equal(frame.candidate_row,p3.loc[frame.attempt_id].candidate_row)
        frame['p3ltr_degree_recency_rank']=p3.loc[frame.attempt_id].p3ltr_degree_recency_rank.to_numpy()
        sampled_frames[representation]=frame
        metrics=read_json(sampled/representation/'test/metrics.json')['metrics']
        metrics.append({'model':'p3ltr_degree_recency','gamma':None,'cases':len(frame),
                        **{m:float(rank_metric(frame.p3ltr_degree_recency_rank,m).mean()) for m in ['ndcg','recall','MRR','HR']}})
        for row in metrics:
            for metric in ['ndcg','recall','MRR','HR']:
                np.testing.assert_allclose(rank_metric(frame[row['model']+'_rank'],metric).mean(),row[metric],atol=1e-14,rtol=0)
        sampled_metrics[representation]=metrics
        others=['rp3beta','fixed_global','fixed_pathway','shuffled_pathway','zero','p3ltr_degree_recency']
        if representation=='conditional_primary':
            others.insert(-2,'context')
        for name in others:
            for metric in (['ndcg','recall','MRR'] if representation=='conditional_primary' else ['ndcg']):
                compare(representation,metric,'actual',name,rank_metric(frame.actual_rank,metric),rank_metric(frame[name+'_rank'],metric))
        for name in (['fixed_global','fixed_pathway','shuffled_pathway','context','zero','p3ltr_degree_recency']
                     if representation=='conditional_primary' else []):
            compare('sampled_controls_exploratory','ndcg',name,'rp3beta',rank_metric(frame[name+'_rank'],'ndcg'),rank_metric(frame.rp3beta_rank,'ndcg'))
    primary=sampled_frames['conditional_primary']; raw=sampled_frames['raw_sensitivity']
    np.testing.assert_array_equal(primary.attempt_id,raw.attempt_id)
    compare('calibration_ranking_exploratory','ndcg','calibrated_actual','raw_actual',
            rank_metric(primary.actual_rank,'ndcg'),rank_metric(raw.actual_rank,'ndcg'))
    pathways=cases.loc[primary.attempt_id].source_event.to_numpy()
    for pathway in np.unique(pathways):
        chosen=pathways==pathway
        for other in ['rp3beta','fixed_global','zero']:
            compare('pathway_'+pathway,'ndcg','actual',other,
                    rank_metric(primary.loc[chosen,'actual_rank'],'ndcg'),rank_metric(primary.loc[chosen,other+'_rank'],'ndcg'))

    development=read_json(RUN/'full_catalog_run/full_catalog/development/rp3beta/metrics.json')['metrics']
    selected=read_json(RUN/'full_catalog_run/full_catalog/selected_settings.json')['weights']
    dev_base=next(r for r in development if r['model']=='rp3beta')['coverage']
    dev_points=[next(r for r in development if r['prior']==name and r['gamma']==weight)
                for name,weight in selected.items()]
    controls_different={name:int(np.count_nonzero(primary.actual_rank!=primary[name+'_rank']))
                        for name in ['fixed_global','fixed_pathway','shuffled_pathway','context','zero']}
    write_json(OUT/'results.json',{
        'status':'completed_statistical_analysis', 'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds':time.perf_counter()-started,
        'full_catalog':full_metrics,'sampled':sampled_metrics,'contrasts':records,
        'development_selected_points':dev_points,'development_baseline_coverage':dev_base,
        'tail_target_users':int(tail.sum()),'tail_target_positive_ndcg_users':tail_counts,
        'sampled_users_with_different_target_ranks':controls_different,
        'full_catalog_execution':read_json(base/'execution.json'),
        'sampled_execution':read_json(sampled/'execution.json'),
        'full_catalog_combined_scoring_seconds':{
            name:read_json(base/name/'metrics.json')['combined_scoring_seconds'] for name in ['rp3beta','p3ltr']},
        'prediction_loss_comparison':read_json(REV/'contact_analysis/conditional_calibration/paired_prediction_loss.json'),
        'interpretation_limits':contract['statistics'],
        'public_repository_updated':False})
    np.savez_compressed(OUT/'bootstrap_replicates.npz',**draws)
    pd.DataFrame(records).to_parquet(OUT/'paired_contrasts.parquet',index=False)
    write_json(OUT/'analysis_manifest.json',{
        'files':{p.name:sha256(p) for p in OUT.iterdir() if p.is_file()},
        'comparison_readiness':{k:v['readiness_sha256'] for k,v in verification.items()},
        'model_runs_reused':True,'source_code':{p.name:sha256(p) for p in [Path(__file__),REV/'src/paired_bootstrap.py']}})
    print(f'Completed {len(records)} contrasts in {time.perf_counter()-started:.1f} seconds',flush=True)


if __name__=='__main__':
    main()
