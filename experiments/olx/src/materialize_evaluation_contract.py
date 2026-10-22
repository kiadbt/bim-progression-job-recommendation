"""Materialize model-independent repaired targets and sampled candidate arrays."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from rebuild_features import ROOT,REV,SOURCE,TRAIN_END,VALIDATION_END
from artifact_hash import sha256
from protocol_primitives import sampled_candidates


def main():
    started=time.time()
    output=REV/'protocol_checks/evaluation_inputs'
    output.mkdir(exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Evaluation input directory must be empty')
    con=duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute('SET threads=4')
    con.execute(f"CREATE VIEW events AS SELECT * FROM read_parquet('{SOURCE.as_posix()}')")
    end=int(con.execute('SELECT max(timestamp)+1 FROM events').fetchone()[0])
    con.execute('CREATE TEMP TABLE pairs AS SELECT user_idx,item_idx,min(timestamp) first_ts FROM events GROUP BY user_idx,item_idx')
    con.execute('CREATE TEMP TABLE items AS SELECT item_idx,min(first_ts) first_ts FROM pairs GROUP BY item_idx')
    con.execute('CREATE TEMP TABLE users AS SELECT user_idx,min(first_ts) first_ts FROM pairs GROUP BY user_idx')
    full=[]
    for name,cutoff,stop in [('development',1582239167,1582401299),('later_retrospective',1582401299,end)]:
        catalog=con.execute('SELECT item_idx FROM items WHERE first_ts<? ORDER BY item_idx',[cutoff]).fetchnumpy()['item_idx'].astype(np.int32)
        catalog_path=output/f'full_catalog_{name}_items.npy'
        np.save(catalog_path,catalog)
        relevance_path=output/f'full_catalog_{name}_relevance.parquet'
        con.execute(f"""COPY (SELECT p.user_idx,p.item_idx FROM pairs p
            JOIN users u USING(user_idx) JOIN items i USING(item_idx)
            WHERE u.first_ts<{cutoff} AND i.first_ts<{cutoff}
              AND p.first_ts>={cutoff} AND p.first_ts<{stop}
            ORDER BY p.user_idx,p.item_idx)
            TO '{relevance_path.as_posix()}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
        counts=con.execute(f"SELECT count(*),count(DISTINCT user_idx) FROM read_parquet('{relevance_path.as_posix()}')").fetchone()
        full.append({'name':name,'training_end_exclusive':cutoff,'outcome_interval':[cutoff,stop],
            'catalog_items':len(catalog),'relevance_pairs':int(counts[0]),'users':int(counts[1]),
            'catalog_file':catalog_path.name,'relevance_file':relevance_path.name})
    target=ROOT/'data/olx/09_graphs/recommendation_pilot/recommendation_eval_targets.parquet'
    features=REV/'features/bim_features_successful_contact.parquet'
    con.execute(f"CREATE VIEW targets AS SELECT * FROM read_parquet('{target.as_posix()}')")
    con.execute(f"CREATE VIEW repaired_features AS SELECT attempt_id,split FROM read_parquet('{features.as_posix()}')")
    # Complete 24-hour follow-up is enforced before choosing each user's first eligible case.
    cases=con.execute(f"""WITH eligible AS (
      SELECT t.* FROM targets t JOIN repaired_features f USING(attempt_id,split)
      JOIN users u USING(user_idx) JOIN items i ON i.item_idx=t.target_item_idx
      JOIN pairs p ON p.user_idx=t.user_idx AND p.item_idx=t.target_item_idx
      WHERE u.first_ts<{TRAIN_END} AND i.first_ts<{TRAIN_END}
        AND p.first_ts>t.source_timestamp
        AND ((t.split='validation' AND t.source_timestamp>={TRAIN_END}
              AND t.source_timestamp+86400<{VALIDATION_END})
          OR (t.split='test' AND t.source_timestamp>={VALIDATION_END}
              AND t.source_timestamp+86400<{end}))
    ), chosen AS (SELECT *,row_number() OVER(PARTITION BY split,user_idx
        ORDER BY source_timestamp,attempt_id) rn FROM eligible)
      SELECT * EXCLUDE(rn) FROM chosen WHERE rn=1 ORDER BY split,user_idx""").fetchdf()
    if cases.empty or set(cases.split)!=set(['validation','test']):
        raise RuntimeError('Repaired sampled cohort is empty or lacks a partition')
    cases['candidate_row']=np.arange(len(cases))
    case_path=output/'sampled_cases.parquet'
    cases.to_parquet(case_path,index=False,compression='zstd')
    catalog=con.execute('SELECT item_idx FROM items WHERE first_ts<? ORDER BY item_idx',[TRAIN_END]).fetchnumpy()['item_idx'].astype(np.int64)
    np.save(output/'sampled_catalog_items.npy',catalog.astype(np.int32))
    con.register('chosen_queries',cases[['attempt_id','user_idx','source_timestamp']])
    histories=con.execute('''SELECT q.attempt_id,p.item_idx FROM chosen_queries q JOIN pairs p
        ON q.user_idx=p.user_idx AND p.first_ts<=q.source_timestamp
        ORDER BY q.attempt_id,p.item_idx''').fetchdf()
    by_case={int(key):g.item_idx.to_numpy(np.int64) for key,g in histories.groupby('attempt_id')}
    candidate_path=output/'sampled_candidates.npy'
    candidate_array=open_memmap(candidate_path,mode='w+',dtype=np.int32,shape=(len(cases),1000))
    for row,case in enumerate(cases.itertuples(index=False)):
        candidate_array[row]=sampled_candidates(catalog,by_case.get(int(case.attempt_id),[]),
            int(case.target_item_idx),999,20261020,int(case.attempt_id))
        if row%5000==0:
            print(f'Materialized candidates {row:,}/{len(cases):,}',flush=True)
    candidate_array.flush()
    del candidate_array
    con.close()
    files={p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in output.iterdir() if p.is_file()}
    record={'status':'materialized','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'full_catalog':full,'sampled':{'training_end_exclusive':TRAIN_END,
            'validation_end_exclusive':VALIDATION_END,'data_end_exclusive':end,
            'cases_by_split':{k:int(v) for k,v in cases.groupby('split').size().items()},
            'catalog_items':len(catalog),'candidate_count':1000,'seed':20261020,
            'task':'First successful contact on a different job within 24 hours, conditional on such an outcome; not generic next-action prediction',
            'seen_policy':'Target and negatives exclude any user-item interaction at or before source timestamp',
            'catalog_policy':'Items first observed strictly before the training cutoff',
            'followup_policy':'Complete 24-hour horizon strictly before each partition endpoint',
            'case_policy':'First eligible source attempt per user and split after model-independent eligibility filters',
            'sampling_policy':'999 distinct unobserved items without replacement; per-attempt SeedSequence',
            'negative_interpretation':'Unobserved comparison items, not verified dislikes; other future positives are not excluded using future information'},
        'evaluation_status':'Retrospective reanalysis; all old periods have been inspected; no independent generalization claim',
        'files':files,'duration_seconds':time.time()-started}
    (output/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
