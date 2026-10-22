"""Validate materialized evaluation inputs before model comparisons are allowed."""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from rebuild_features import REV,SOURCE,TRAIN_END,VALIDATION_END
from artifact_hash import sha256
from protocol_primitives import sampled_candidates,require_disjoint_outcome_intervals,require_component_available


def main():
    started=time.time()
    output=REV/'protocol_checks'
    inputs=output/'evaluation_inputs'
    manifest=json.loads((inputs/'manifest.json').read_text(encoding='utf-8'))
    audit=json.loads((output/'legacy_protocol_audit.json').read_text(encoding='utf-8'))
    tests=json.loads((output/'unit_test_results.json').read_text(encoding='utf-8'))
    assert tests['successful'] and tests['protocol_primitives_sha256']==sha256(REV/'src/protocol_primitives.py')
    require_disjoint_outcome_intervals((1582239167,1582401299),(1582401299,1582675200))
    require_component_available(VALIDATION_END,VALIDATION_END)
    for name,record in manifest['files'].items():
        assert sha256(inputs/name)==record['sha256'],name
    development_users=np.load(inputs/manifest['development_selection']['file'])
    assert len(development_users)==len(np.unique(development_users))==manifest['development_selection']['users']==30000
    assert sha256(SOURCE)==audit['source_sha256']
    con=duckdb.connect()
    con.execute("SET memory_limit='5GB'")
    con.execute('SET threads=4')
    con.execute(f"CREATE VIEW events AS SELECT * FROM read_parquet('{SOURCE.as_posix()}')")
    end=int(con.execute('SELECT max(timestamp)+1 FROM events').fetchone()[0])
    con.execute('CREATE TEMP TABLE pairs AS SELECT user_idx,item_idx,min(timestamp) first_ts FROM events GROUP BY user_idx,item_idx')
    con.execute('CREATE TEMP TABLE items AS SELECT item_idx,min(first_ts) first_ts FROM pairs GROUP BY item_idx')
    con.execute('CREATE TEMP TABLE users AS SELECT user_idx,min(first_ts) first_ts FROM pairs GROUP BY user_idx')
    full=[]
    for record in manifest['full_catalog']:
        cutoff=record['training_end_exclusive']
        stop=record['outcome_interval'][1]
        catalog=np.load(inputs/record['catalog_file'])
        expected=con.execute('SELECT item_idx FROM items WHERE first_ts<? ORDER BY item_idx',[cutoff]).fetchnumpy()['item_idx']
        np.testing.assert_array_equal(catalog,expected)
        path=(inputs/record['relevance_file']).as_posix()
        violations=con.execute(f"""SELECT count(*) FROM read_parquet('{path}') r
            LEFT JOIN pairs p USING(user_idx,item_idx) LEFT JOIN users u USING(user_idx)
            LEFT JOIN items i USING(item_idx)
            WHERE p.first_ts IS NULL OR p.first_ts<{cutoff} OR p.first_ts>={stop}
               OR u.first_ts>={cutoff} OR i.first_ts>={cutoff}""").fetchone()[0]
        assert violations==0,record['name']
        counts=con.execute(f"SELECT count(*),count(DISTINCT (user_idx,item_idx)),count(DISTINCT user_idx) FROM read_parquet('{path}')").fetchone()
        assert counts[0]==counts[1]==record['relevance_pairs']
        assert counts[2]==record['users']
        if record['name']=='development':
            users=con.execute(f"SELECT DISTINCT user_idx FROM read_parquet('{path}') ORDER BY user_idx").fetchnumpy()['user_idx']
            expected_users=np.sort(np.random.default_rng(10).choice(users,30000,replace=False))
            np.testing.assert_array_equal(development_users,expected_users)
        full.append({'name':record['name'],'pairs_verified':int(counts[0]),'users':int(counts[2]),'violations':0})
    cases=pd.read_parquet(inputs/'sampled_cases.parquet')
    candidates=np.load(inputs/'sampled_candidates.npy',mmap_mode='r')
    catalog=np.load(inputs/'sampled_catalog_items.npy')
    assert candidates.shape==(len(cases),1000)
    assert cases.attempt_id.is_unique
    assert not cases.duplicated(['split','user_idx']).any()
    np.testing.assert_array_equal(cases.candidate_row,np.arange(len(cases)))
    np.testing.assert_array_equal(candidates[:,0],cases.target_item_idx)
    assert np.isin(candidates,catalog,kind='table').all()
    ordered=np.sort(candidates,axis=1)
    assert (np.diff(ordered,axis=1)>0).all()
    del ordered
    con.register('queries',cases[['attempt_id','user_idx','source_timestamp','target_item_idx','split']])
    violations=con.execute(f"""SELECT count(*) FROM queries q JOIN users u USING(user_idx)
        JOIN items i ON i.item_idx=q.target_item_idx JOIN pairs p ON p.user_idx=q.user_idx AND p.item_idx=q.target_item_idx
        WHERE u.first_ts>={TRAIN_END} OR i.first_ts>={TRAIN_END} OR p.first_ts<=q.source_timestamp
          OR (q.split='validation' AND (q.source_timestamp<{TRAIN_END} OR q.source_timestamp+86400>={VALIDATION_END}))
          OR (q.split='test' AND (q.source_timestamp<{VALIDATION_END} OR q.source_timestamp+86400>={end}))""").fetchone()[0]
    assert violations==0
    histories=con.execute('''SELECT q.attempt_id,p.item_idx FROM queries q JOIN pairs p
        ON q.user_idx=p.user_idx AND p.first_ts<=q.source_timestamp
        ORDER BY q.attempt_id,p.item_idx''').fetchdf()
    by_case={int(key):g.item_idx.to_numpy(np.int64) for key,g in histories.groupby('attempt_id')}
    selected=set(np.random.default_rng(20261020).choice(len(cases),min(100,len(cases)),replace=False).tolist())
    for row,case in enumerate(cases.itertuples(index=False)):
        seen=by_case.get(int(case.attempt_id),np.empty(0,dtype=np.int64))
        assert not np.isin(candidates[row],seen).any(),int(case.attempt_id)
        if row in selected:
            replay=sampled_candidates(catalog,seen,int(case.target_item_idx),999,20261020,int(case.attempt_id))
            np.testing.assert_array_equal(candidates[row],replay)
    con.close()
    for split,g in cases.groupby('split'):
        for seed in [20260827,20260828,20260829]:
            directory=REV/'model_reruns/successful_contact/primitive_evidence'/f'seed_{seed}'
            config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
            predictions=pd.read_parquet(directory/f'{split}_predictions.parquet')
            assert predictions.attempt_id.is_unique
            assert np.isfinite(predictions.inc_score).all()
            assert predictions.inc_score.between(0,1).all()
            assert g.attempt_id.isin(predictions.attempt_id).all()
            assert config['feature_sha256']==sha256(REV/'features/bim_features_successful_contact.parquet')
    record={'status':'passed_for_retrospective_reanalysis_only',
        'full_catalog_checks':full,'common_development_users_verified':len(development_users),'sampled_rows_verified':len(cases),
        'sampled_candidate_slots_verified':int(candidates.size),'candidate_replays_verified':len(selected),
        'checks':['all file hashes','eligible known catalog','unique full-catalog relevance',
            'eligible relevance dates and warm users','complete sampled followup',
            'distinct candidates','target alignment','no query-seen candidates','per-case reproducibility',
            'pooled MLP prediction coverage and feature lineage'],
        'unit_tests_passed':tests['tests_run'],'test_execution_record':'unit_test_results.json',
        'manifest_sha256':sha256(inputs/'manifest.json'),
        'comparison_contract_sha256':sha256(output/'comparison_contract.json'),
        'remaining_limits':['No untouched new data period','No verified listing publication or expiry metadata',
            'Drivers must use repaired primitives and shared inputs; archived code is not approved for reuse unchanged',
            'Fixed shrinkage and recency settings do not establish performance against optimally tuned model families'],
        'downstream_comparisons_started':False,'duration_seconds':time.time()-started}
    (output/'verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
