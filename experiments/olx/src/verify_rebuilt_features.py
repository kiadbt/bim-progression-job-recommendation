"""Compare rebuilt real-data histories with independent SQL counts."""
import json
from pathlib import Path

import duckdb
import pandas as pd

from artifact_hash import sha256
from rebuild_features import ROOT, REV, TRAIN_END, VALIDATION_END, SOURCE


def main():
    summary=json.loads((REV/'features/feature_build_summary.json').read_text(encoding='utf-8'))
    intermediate=REV/'features/label_construction_intermediate/successful_contact_summary.json'
    intermediate_record=json.loads(intermediate.read_text(encoding='utf-8'))
    intermediate_record.pop('leakage_validation_passed',None)
    intermediate_record['status']='label_construction_only_not_validated_predictive_features'
    intermediate.write_text(json.dumps(intermediate_record,indent=2),encoding='utf-8')
    path=REV/'features/bim_features_successful_contact.parquet'
    frame=pd.read_parquet(path)
    sample=frame.sample(n=min(1000,len(frame)),random_state=20261020)
    con=duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.register('queries',sample[['attempt_id','sequence_id','source_event','source_timestamp']])
    labels=(REV/'features/label_construction_intermediate/bim_features_successful_contact.parquet').as_posix()
    oracle=con.execute(f"""SELECT q.attempt_id,
        count(a.source_timestamp) FILTER (WHERE a.source_timestamp<q.source_timestamp) raw_count,
        count(a.source_timestamp) FILTER (WHERE a.source_timestamp+a.q90_seconds<q.source_timestamp AND a.progressed_q90=1) positive_count,
        count(a.source_timestamp) FILTER (WHERE a.source_timestamp+a.q90_seconds<q.source_timestamp AND a.progressed_q90=0) negative_count
      FROM queries q LEFT JOIN read_parquet('{labels}') a
      ON q.sequence_id=a.sequence_id AND q.source_event=a.source_event
      GROUP BY q.attempt_id""").fetchdf().set_index('attempt_id')
    observed=sample.set_index('attempt_id').loc[oracle.index]
    for actual,expected in [('prior_attempt_count','raw_count'),
                            ('prior_progression_count_q90','positive_count'),
                            ('prior_nonprogression_count_q90','negative_count')]:
        assert (observed[actual]==oracle[expected]).all(),actual
    events=con.execute(f"""SELECT q.attempt_id,count(e.timestamp) prior_events,
        count(e.timestamp) FILTER (WHERE e.event='click') prior_clicks,
        max(e.timestamp) previous_time
      FROM queries q LEFT JOIN read_parquet('{SOURCE.as_posix()}') e
      ON q.sequence_id=e.sequence_id AND e.timestamp<q.source_timestamp
      GROUP BY q.attempt_id""").fetchdf().set_index('attempt_id')
    observed=sample.set_index('attempt_id').loc[events.index]
    assert (observed.position==events.prior_events).all()
    assert (observed.prior_click_count==events.prior_clicks).all()
    assert (observed.previous_time.fillna(-1)==events.previous_time.fillna(-1)).all()
    assert ((frame.loc[frame.split=='train','source_timestamp']+
             frame.loc[frame.split=='train','q95_seconds'])<TRAIN_END).all()
    assert ((frame.loc[frame.split=='validation','source_timestamp']+
             frame.loc[frame.split=='validation','q95_seconds'])<VALIDATION_END).all()
    for outcome,record in summary['outputs'].items():
        assert sha256(Path(record['path']))==record['sha256'],outcome
    con.close()
    record={'status':'passed','real_data_oracle_queries':len(sample),
        'checks':['strict prior event context','maturity-aware prior labels','train label boundary',
                  'validation label boundary','all output hashes'],
        'feature_manifest_sha256':sha256(REV/'features/feature_build_summary.json')}
    (REV/'features/verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
