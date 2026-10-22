"""Run shared-protocol comparisons or a bounded integration smoke test."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
import importlib.metadata
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd

from approved_evaluation_inputs import load_approved_inputs
from calibrated_score_gate import load_bim_controls
from comparison_adapter import select_operating_point, sampled_target_ranks
from protocol_primitives import exact_sparse_fusion, stable_topk
from ranking_driver_core import (REV,INPUTS,SOURCE,verify_sources,read_json,write_json,
    load_training,build_graph,PreparedFusion,evaluate_lists,gather_scores,zscore_rows,
    sampled_metrics,build_mature_profiles,relation_scores,connection,peak_working_set_bytes)
from artifact_hash import sha256


def source_hashes():
    names=['run_repaired_ranking.py','ranking_driver_core.py','comparison_adapter.py',
        'protocol_primitives.py','approved_evaluation_inputs.py','calibrated_score_gate.py']
    return {name:sha256(REV/'src'/name) for name in names}


def selected_smoke_users(users,n):
    # Selection is deterministic and independent of relevance or model scores.
    return np.sort(np.asarray(users,dtype=np.int64))[:n]


def full_pass(data,edge,similarity,users,relevant,settings,folder,batch_size,smoke):
    folder.mkdir();local=data.rows(users);rec_arrays={};rankers={}
    for label,prior,gamma in settings:
        rec_arrays[label]=np.lib.format.open_memmap(folder/f'{label}_recommendations.npy',mode='w+',
                dtype=np.int32,shape=(len(users),10))
        rankers[label]=PreparedFusion(data.catalog,data.priors.get(prior,np.zeros(data.seen.shape[1])),gamma)
    started=time.perf_counter();oracle_checks=0
    for start in range(0,len(users),batch_size):
        batch=local[start:start+batch_size];scores=(edge[batch]@similarity).tocsr();scores.sort_indices()
        for row,user_row in enumerate(batch):
            lo,hi=scores.indptr[row:row+2];a,b=data.seen.indptr[user_row:user_row+2]
            ids=scores.indices[lo:hi];values=scores.data[lo:hi];seen=data.seen.indices[a:b]
            for label,_,_ in settings:
                ranker=rankers[label];ranked=ranker.rank(ids,values,seen)
                rec_arrays[label][start+row]=-1;rec_arrays[label][start+row,:len(ranked)]=ranked
                if smoke and start+row<2:
                    expected=exact_sparse_fusion(ids,values,ranker.aux,data.catalog,seen,ranker.gamma)
                    np.testing.assert_array_equal(ranked,expected)
                    # Independent dense enumeration across the entire eligible catalog.
                    full=np.zeros(data.seen.shape[1]);full[ids]=values
                    eligible=np.setdiff1d(data.catalog,seen)
                    scale=full[eligible].max(initial=0)
                    if scale>0:full/=scale
                    exact=stable_topk(eligible,full[eligible]+ranker.gamma*ranker.aux[eligible],10)
                    np.testing.assert_array_equal(ranked,exact);oracle_checks+=1
        if start==0 or (start//batch_size)%200==0:
            print(f'{folder.name}: ranked {min(start+batch_size,len(users))}/{len(users)} users',flush=True)
    serving=time.perf_counter()-started;rows=[]
    for label,prior,gamma in settings:
        rec_arrays[label].flush()
        metrics,per_user,freq=evaluate_lists(rec_arrays[label],relevant,users,data.catalog,data.bands)
        per_user.to_parquet(folder/f'{label}_per_user.parquet',index=False)
        np.save(folder/f'{label}_exposure.npy',freq)
        rows.append(dict(model=label,prior=prior,gamma=float(gamma),**metrics))
    write_json(folder/'metrics.json',{'purpose':'smoke_only' if smoke else 'retrospective_comparison',
        'metrics':rows,'combined_scoring_seconds':serving,'oracle_checks':oracle_checks,
        'serving_measurement_scope':'Batched graph multiplication and all requested settings together; not isolated production latency'})
    del rec_arrays
    return rows,oracle_checks


def run_full(manifest,args,folder):
    folder.mkdir();contract=read_json(REV/'protocol_checks/comparison_contract.json')
    grid=contract['selection_and_comparators']['weight_grid_per_family']
    families=['popularity','contact_count','contact_rate','recency'];selections={};checks=0
    for phase in ['development','later_retrospective']:
        record=next(r for r in manifest['full_catalog'] if r['name']==phase)
        catalog=np.load(INPUTS/record['catalog_file']);rel_frame=pd.read_parquet(INPUTS/record['relevance_file'])
        users=np.load(INPUTS/manifest['development_selection']['file']) if phase=='development' else np.sort(rel_frame.user_idx.unique())
        if args.mode=='smoke':users=selected_smoke_users(users,args.smoke_users)
        relevant={int(u):g.item_idx.to_numpy(np.int64) for u,g in rel_frame[rel_frame.user_idx.isin(users)].groupby('user_idx')}
        if len(relevant)!=len(users):raise AssertionError('Relevance and selected user set disagree')
        data=load_training(record['training_end_exclusive'],catalog,users if args.mode=='smoke' else None)
        phase_dir=folder/phase;phase_dir.mkdir();np.save(phase_dir/'user_ids.npy',users)
        np.save(phase_dir/'item_bands.npy',data.bands)
        stats={'history_cutoff':data.cutoff,'training_pairs':int(data.seen.nnz),'training_users':len(data.users),
               'eligible_items':len(catalog),'evaluated_users':len(users),'miniature_training_graph':data.miniature_graph,
               'preparation_seconds':data.preparation_seconds,'bands':'Equal item-count quartiles by training event count; ascending ID ties'}
        edge,sim,build=build_graph(data,'rp3beta',args.block_size);stats['rp3beta']=build
        if phase=='development':
            settings=[('rp3beta','contact_count',0)]+[(f'{family}_g{gamma:g}',family,gamma) for family in families for gamma in grid if gamma>0]
        else:
            # Selection was serialized before later graph construction and scoring.
            saved=read_json(folder/'selected_settings.json')
            if saved['selection_split']!='development' or saved['source_code']!=source_hashes():
                raise ValueError('Invalid frozen selection record')
            settings=[('rp3beta','contact_count',0)]+[(family,family,saved['weights'][family]) for family in families]
        rows,checked=full_pass(data,edge,sim,users,relevant,settings,phase_dir/'rp3beta',args.batch_size,args.mode=='smoke')
        checks+=checked
        if phase=='development':
            base=next(r for r in rows if r['model']=='rp3beta')
            for family in families:
                points=[dict(base,split='development')]+[dict(r,split='development') for r in rows if r['prior']==family and r['gamma']>0]
                selections[family]=float(select_operating_point(points,base['coverage'])['gamma'])
            write_json(folder/'selected_settings.json',{'selection_split':'development','weights':selections,
                'selection_outcomes_end':record['outcome_interval'][1],'source_code':source_hashes(),
                'purpose':'smoke_only' if args.mode=='smoke' else 'retrospective_comparison',
                'selection_rule':contract['selection_and_comparators']['primary_selection']})
        del edge,sim;gc.collect()
        edge,sim,build=build_graph(data,'p3ltr_degree_recency',args.block_size);stats['p3ltr_degree_recency']=build
        p3,checked=full_pass(data,edge,sim,users,relevant,[('p3ltr_degree_recency','contact_count',0)],
                           phase_dir/'p3ltr',args.batch_size,args.mode=='smoke');checks+=checked
        stats['peak_process_working_set_bytes']=peak_working_set_bytes()
        write_json(phase_dir/'execution.json',stats)
        del edge,sim,data,rel_frame;gc.collect()
    return {'oracle_checks':checks,'development_and_later_completed':True,'matched_family_count':4,
            'p3ltr_degree_recency_included':True,'purpose':'smoke_only' if args.mode=='smoke' else 'retrospective_comparison'}


def load_controls_for_cases(cases,representation):
    all_frames=[];columns=None
    for task in ['sampled_validation_development','sampled_test']:
        frame,columns,_=load_bim_controls(task,representation,True);all_frames.append(frame)
    scores=pd.concat(all_frames).set_index('attempt_id')
    if not scores.index.is_unique:raise ValueError('Duplicate BIM score identities')
    aligned=scores.loc[cases.attempt_id].reset_index()
    np.testing.assert_array_equal(aligned.candidate_row,cases.candidate_row)
    np.testing.assert_array_equal(aligned.split,cases.split)
    risks={name:aligned[column].to_numpy(float) for name,column in columns.items()}
    risks['zero']=np.zeros(len(cases));return risks


def sampled_pass(edge,sim,data,cases,candidates,risks,profiles,grid,selections,output,batch_size,phase,smoke):
    output.mkdir();all_ranks={};ids=cases.attempt_id.to_numpy();users=cases.user_idx.to_numpy()
    settings=[('rp3beta',0)]+[(name,g) for name in risks for g in (grid if selections is None else [selections[name]])]
    buffers={key:[] for key in settings};checks=0
    for start in range(0,len(cases),batch_size):
        batch=cases.iloc[start:start+batch_size];cand=np.asarray(candidates[batch.candidate_row],dtype=np.int64)
        if not np.array_equal(cand[:,0],batch.target_item_idx):raise AssertionError('Candidate target mismatch')
        score_matrix=(edge[data.rows(batch.user_idx)]@sim).tocsr()
        base=zscore_rows(gather_scores(score_matrix,cand))
        if not np.isfinite(base).all():raise AssertionError('Nonfinite base')
        events=batch.source_event.to_numpy();targets=batch.target_item_idx.to_numpy()
        relations={name:relation_scores(cand,events,risk[start:start+len(batch)],profiles) for name,risk in risks.items()}
        for name,gamma in settings:
            combined=base if name=='rp3beta' else base+gamma*relations[name]
            ranks=sampled_target_ranks(cand,combined,targets);buffers[(name,gamma)].append(ranks)
            if smoke:
                for row in range(min(2,len(batch))):
                    ordered=cand[row,np.lexsort((cand[row],-combined[row]))]
                    assert int(np.flatnonzero(ordered==targets[row])[0])+1==ranks[row];checks+=1
        if start==0 or (start//batch_size)%50==0:print(f'{output.name}: sampled cases {min(start+batch_size,len(cases))}/{len(cases)}',flush=True)
    rows=[]
    for (name,gamma),parts in buffers.items():
        ranks=np.concatenate(parts);all_ranks[(name,gamma)]=ranks
        rows.append({'model':name,'gamma':float(gamma),'split':phase,**sampled_metrics(ranks)})
    if selections is None:
        selections={}
        for name in risks:
            chosen=max([r for r in rows if r['model']==name],key=lambda r:(r['ndcg'],r['recall'],-r['gamma']))
            selections[name]=chosen['gamma']
    paired=pd.DataFrame({'attempt_id':ids,'candidate_row':cases.candidate_row.to_numpy(),'user_idx':users,
                        'split':phase,'rp3beta_rank':all_ranks[('rp3beta',0)]})
    for name,gamma in selections.items():paired[name+'_rank']=all_ranks[(name,gamma)]
    paired.to_parquet(output/'paired_ranks.parquet',index=False)
    write_json(output/'metrics.json',{'purpose':'smoke_only' if smoke else 'retrospective_comparison','metrics':rows})
    return selections,checks


def verify_query_masks(cases,candidates):
    con=connection();con.register('queries',cases[['attempt_id','user_idx','source_timestamp']])
    seen=con.execute('''SELECT q.attempt_id,e.item_idx FROM queries q JOIN events e USING(user_idx)
        WHERE e.timestamp<=q.source_timestamp GROUP BY q.attempt_id,e.item_idx''').fetchdf();con.close()
    grouped={int(a):g.item_idx.to_numpy() for a,g in seen.groupby('attempt_id')}
    for row in cases.itertuples():
        c=np.asarray(candidates[row.candidate_row]);assert len(np.unique(c))==1000
        if np.isin(c,grouped[int(row.attempt_id)]).any():raise AssertionError('Query-seen candidate')
    return len(cases)


def run_sampled(manifest,args,folder):
    folder.mkdir();cases=pd.read_parquet(INPUTS/'sampled_cases.parquet')
    if args.mode=='smoke':
        cases=cases.groupby(['split','source_event'],sort=True,group_keys=False).head(args.smoke_cases_per_pathway)
        cases=cases.sort_values('candidate_row').reset_index(drop=True)
    catalog=np.load(INPUTS/'sampled_catalog_items.npy')
    candidates=np.load(INPUTS/'sampled_candidates.npy',mmap_mode='r')
    for row in cases.itertuples():
        if not np.isin(candidates[row.candidate_row],catalog).all():raise AssertionError('Ineligible sampled job')
    mask_checks=verify_query_masks(cases,candidates) if args.mode=='smoke' else 0
    cutoff=manifest['sampled']['training_end_exclusive']
    profiles=build_mature_profiles(cutoff,catalog,folder)
    data=load_training(cutoff,catalog,np.unique(cases.user_idx) if args.mode=='smoke' else None)
    edge,sim,build=build_graph(data,'rp3beta',args.block_size)
    plan=read_json(REV/'contact_analysis/risk_comparison_preconditions.json');checks=0
    for representation in ['conditional_primary','raw_sensitivity']:
        risks=load_controls_for_cases(cases,representation);rep_dir=folder/representation;rep_dir.mkdir()
        val=cases.split.to_numpy()=='validation';test=~val
        selected,checked=sampled_pass(edge,sim,data,cases[val],candidates,
            {n:v[val] for n,v in risks.items()},profiles,plan['gamma_grid'],None,
            rep_dir/'development',args.batch_size,'development',args.mode=='smoke');checks+=checked
        selection={'selection_split':'validation_development','weights':selected,'selection_outcomes_end':manifest['sampled']['validation_end_exclusive'],
                   'representation':representation,'source_code':source_hashes(),
                   'purpose':'smoke_only' if args.mode=='smoke' else 'retrospective_comparison'}
        write_json(rep_dir/'selected_settings.json',selection)
        loaded=read_json(rep_dir/'selected_settings.json')
        if not (cases[test].source_timestamp>=loaded['selection_outcomes_end']).all():
            raise AssertionError('Test query predates component selection')
        _,checked=sampled_pass(edge,sim,data,cases[test],candidates,
            {n:v[test] for n,v in risks.items()},profiles,plan['gamma_grid'],loaded['weights'],
            rep_dir/'test',args.batch_size,'test',args.mode=='smoke');checks+=checked
    del edge,sim;gc.collect()
    # Same sampled candidates and training history for the external P3LTR component.
    edge,sim,p3build=build_graph(data,'p3ltr_degree_recency',args.block_size)
    p3ranks=[]
    for start in range(0,len(cases),args.batch_size):
        batch=cases.iloc[start:start+args.batch_size];cand=np.asarray(candidates[batch.candidate_row])
        scores=gather_scores((edge[data.rows(batch.user_idx)]@sim).tocsr(),cand)
        p3ranks.extend(sampled_target_ranks(cand,scores,batch.target_item_idx))
    pd.DataFrame({'attempt_id':cases.attempt_id,'candidate_row':cases.candidate_row,'split':cases.split,
                  'p3ltr_degree_recency_rank':p3ranks}).to_parquet(folder/'p3ltr_paired_ranks.parquet',index=False)
    write_json(folder/'execution.json',{'rp3beta':build,'p3ltr_degree_recency':p3build,
        'preparation_seconds':data.preparation_seconds,'training_pairs':int(data.seen.nnz),
        'miniature_training_graph':data.miniature_graph,'cases':len(cases),
        'risk_formula':'zscore(base candidate scores) + gamma * zscore(log1p(pathway mature successes) - risk * log1p(pathway mature nonprogressions))',
        'profile_rule':'Q95-eligible corrected training attempts; profiles use Q90 labels',
        'normalization':'Sampled score z-scores are within the fixed 1000-item candidate set; full-catalog score normalization differs',
        'peak_process_working_set_bytes':peak_working_set_bytes()})
    return {'cases':len(cases),'query_seen_checks':mask_checks,'oracle_checks':checks,
        'primary_and_raw_sensitivity_completed':True,'p3ltr_included':True,
        'purpose':'smoke_only' if args.mode=='smoke' else 'retrospective_comparison'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','compare'],default='smoke')
    p.add_argument('--task',choices=['all','full','sampled'],default='all');p.add_argument('--output',required=True,type=Path)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--block-size',type=int,default=200)
    p.add_argument('--smoke-users',type=int,default=16);p.add_argument('--smoke-cases-per-pathway',type=int,default=8)
    p.add_argument('--readiness-record',type=Path);args=p.parse_args()
    if min(args.batch_size,args.block_size,args.smoke_users,args.smoke_cases_per_pathway)<1:raise ValueError('Positive sizes required')
    if args.output.exists():raise FileExistsError('Output must be a new directory')
    if args.mode=='compare':
        if args.readiness_record is None:raise ValueError('Comparisons require a passed smoke readiness record')
        ready=read_json(args.readiness_record)
        if ready.get('status')!='passed_smoke' or ready.get('source_code')!=source_hashes():
            raise ValueError('Readiness record does not match current driver code')
        if ready.get('tasks')!=['full','sampled']:raise ValueError('Both tasks must pass smoke testing')
        if ready.get('input_manifest_sha256')!=sha256(INPUTS/'manifest.json'):
            raise ValueError('Evaluation inputs changed after smoke verification')
        if ready.get('risk_plan_sha256')!=sha256(REV/'contact_analysis/risk_comparison_preconditions.json'):
            raise ValueError('BIM comparison plan changed after smoke verification')
    started=time.perf_counter();sources=verify_sources()
    manifest=load_approved_inputs('sampled_test',1582493743,is_learned_bim=True)
    load_approved_inputs('full_catalog_later')
    args.output.mkdir(parents=True)
    write_json(args.output/'run_specification.json',{'mode':args.mode,'task':args.task,'sources':sources,
        'source_code':source_hashes(),'input_manifest_sha256':sha256(INPUTS/'manifest.json'),
        'comparison_contract_sha256':sha256(REV/'protocol_checks/comparison_contract.json'),
        'risk_plan_sha256':sha256(REV/'contact_analysis/risk_comparison_preconditions.json'),
        'batch_size':args.batch_size,'block_size':args.block_size,
        'smoke_definition':'Small deterministic user subset with induced training graph; full approved catalogs, global priors, profiles and original candidate arrays',
        'selection_policy':'Development scores only; selected weights saved before later scoring',
        'python':platform.python_version(),'platform':platform.platform(),'logical_cpus':os.cpu_count(),
        'packages':{name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','duckdb','pyarrow']},
        'started_at_utc':datetime.now(timezone.utc).isoformat()})
    outputs={}
    if args.task in ['all','full']:outputs['full']=run_full(manifest,args,args.output/'full_catalog')
    if args.task in ['all','sampled']:outputs['sampled']=run_sampled(manifest,args,args.output/'sampled')
    record={'status':'passed_smoke' if args.mode=='smoke' else 'completed_comparisons','mode':args.mode,
        'tasks':list(outputs),'checks':outputs,'source_code':source_hashes(),'source_data':sources,
        'input_manifest_sha256':sha256(INPUTS/'manifest.json'),
        'risk_plan_sha256':sha256(REV/'contact_analysis/risk_comparison_preconditions.json'),
        'duration_seconds':time.perf_counter()-started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'downstream_ranking_comparisons_started':args.mode=='compare',
        'smoke_metrics_are_scientific_results':False,'evaluation_scope':'retrospective; no fresh holdout',
        'artifact_hashes':{str(f.relative_to(args.output)):sha256(f) for f in args.output.rglob('*') if f.is_file()}}
    write_json(args.output/'readiness.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ['artifact_hashes','source_code']},indent=2),flush=True)


if __name__=='__main__':main()
