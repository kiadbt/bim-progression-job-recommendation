"""Rebuild OLX contact features from a training-prefix parameter fit."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from generate_bim_features_successful_contact_olx import run as generate, pathway_parameters
from observable_history import build_histories, label_admissible_before
from artifact_hash import sha256

TRAIN_END = 1582312306
VALIDATION_END = 1582493743
SOURCE = ROOT / 'data/olx/04_sequences/ui_sequences.parquet'


def fit_parameters(con, source: str, cutoff: int):
    """Only transitions whose target has arrived before the fit boundary."""
    stats = con.execute(f"""
      SELECT previous_event source_event, event target_event, count(*) transition_count,
        quantile_disc(delta_seconds, [0.5,0.75,0.9,0.95]) horizons
      FROM read_parquet('{source}')
      WHERE valid_temporal_transition AND timestamp < {int(cutoff)}
      GROUP BY previous_event,event ORDER BY previous_event,event
    """).fetchdf()
    total = int(stats.transition_count.sum())
    outgoing = stats.groupby('source_event').transition_count.sum()
    incoming = stats.groupby('target_event').transition_count.sum()
    stats['conditional_probability'] = stats.transition_count / stats.source_event.map(outgoing)
    stats['lift'] = stats.conditional_probability / (stats.target_event.map(incoming) / total)
    for i, name in enumerate(['median_seconds','q75_seconds','q90_seconds','q95_seconds']):
        stats[name] = stats.horizons.map(lambda x: int(x[i]))
    return stats.drop(columns='horizons')


def context_sql(source: str, restrict_to_attempts: bool = False):
    counts = {
        'click': 'prior_click_count', 'bookmark': 'prior_bookmark_count',
        'chat_click': 'prior_chat_click_count', 'contact_chat': 'prior_contact_chat_count',
    }
    terms = [f"coalesce(sum(event='{event}') OVER w,0) {name}" for event,name in counts.items()]
    terms += ["coalesce(sum(event LIKE 'contact_phone_click_%') OVER w,0) prior_phone_event_count",
              "coalesce(sum(event='contact_partner_click') OVER w,0) prior_partner_event_count",
              "count(DISTINCT event) OVER w prior_distinct_behavior_count",
              "count(*) OVER w context_prior_event_count",
              "max(timestamp) OVER w previous_time",
              "arg_max(event, struct_pack(t := timestamp, e := event)) OVER w previous_event"]
    restriction = 'WHERE sequence_id IN (SELECT DISTINCT sequence_id FROM attempt_keys)' if restrict_to_attempts else ''
    return f"""SELECT sequence_id,position,{','.join(terms)}
      FROM read_parquet('{source}')
      {restriction}
      WINDOW w AS (PARTITION BY sequence_id ORDER BY timestamp
        RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)"""


def main():
    started = time.time()
    output = REV / 'features'
    output.mkdir(exist_ok=True)
    if (output / 'feature_build_summary.json').exists():
        raise FileExistsError('Feature rebuild already completed')
    source = str(SOURCE.resolve()).replace("'", "''")
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute('SET threads=4')
    spill = output / 'duckdb_spill'
    spill.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{spill.as_posix()}'")
    print('Fitting transition probabilities and exact empirical horizons on training prefix...', flush=True)
    stats = fit_parameters(con, source, TRAIN_END)
    transition_file = output / 'training_transition_statistics.csv'
    temporal_file = output / 'training_temporal_windows.csv'
    if transition_file.exists() or temporal_file.exists():
        raise FileExistsError('Parameter outputs already exist')
    stats.to_csv(transition_file,index=False)
    stats.to_csv(temporal_file,index=False)
    params = pathway_parameters(transition_file, temporal_file)
    if any(p['q90'] <= 0 or p['q95'] < p['q90'] for p in params.values()):
        raise ValueError('Invalid fitted maturity horizon')
    print(json.dumps(params,indent=2),flush=True)
    intermediate = output / 'label_construction_intermediate'
    generate(SOURCE, transition_file, temporal_file, intermediate, '6GB', 4)
    intermediate_summary = intermediate / 'successful_contact_summary.json'
    intermediate_record = json.loads(intermediate_summary.read_text(encoding='utf-8'))
    intermediate_record.pop('leakage_validation_passed',None)
    intermediate_record['status'] = 'label_construction_only_not_validated_predictive_features'
    intermediate_summary.write_text(json.dumps(intermediate_record,indent=2),encoding='utf-8')
    frame = pd.read_parquet(intermediate / 'bim_features_successful_contact.parquet')
    original = pd.read_parquet(ROOT / 'data/olx/07_bim_features/bim_features_successful_contact.parquet',
                               columns=['attempt_id','sequence_id','position'])
    # Retain old identifiers for attempts present in both builds, despite changed horizons.
    frame = frame.drop(columns='attempt_id').merge(original,on=['sequence_id','position'],how='left',validate='one_to_one')
    new = frame.attempt_id.isna()
    frame.loc[new,'attempt_id'] = np.arange(int(original.attempt_id.max())+1,int(original.attempt_id.max())+1+int(new.sum()))
    frame.attempt_id = frame.attempt_id.astype('int64')
    del original
    print('Rebuilding ordinary event contexts with strict timestamp precedence...',flush=True)
    con.register('attempt_keys',frame[['sequence_id','position']])
    contexts = con.execute(f"""SELECT c.* FROM ({context_sql(source, restrict_to_attempts=True)}) c
        INNER JOIN attempt_keys a USING(sequence_id,position)""").fetchdf()
    context_names = [c for c in contexts if c not in ['sequence_id','position']]
    frame = frame.drop(columns=[c for c in context_names if c in frame])
    frame = frame.merge(contexts,on=['sequence_id','position'],how='left',validate='one_to_one')
    frame['immediately_previous_event'] = frame.previous_event
    frame['seconds_since_previous_event'] = frame.source_timestamp-frame.previous_time
    # Position is an identity/order key only; the predictor receives the strict prior-event count.
    frame['original_position'] = frame.position
    frame['position'] = frame.context_prior_event_count
    histories = build_histories(frame)
    for column in histories:
        frame[column] = histories[column]
    n = frame.prior_attempt_count.to_numpy()
    neg = frame.prior_nonprogression_count_q90.to_numpy()
    frame['D_candidate'] = frame.expected_transition_probability.to_numpy()*neg/np.maximum(n,1)
    frame['R_candidate'] = np.log1p(neg)
    frame['A_candidate'] = (1-np.exp(-frame.seconds_since_previous_source_attempt/frame.q90_seconds)).fillna(0)
    frame['Q_candidate'] = 1-np.exp(-(n+1)/2)
    frame['split'] = np.where(frame.source_timestamp < TRAIN_END,'train',
                             np.where(frame.source_timestamp < VALIDATION_END,'validation','test'))
    admissible = ((frame.split=='train') & label_admissible_before(frame,TRAIN_END)
                  | (frame.split=='validation') & label_admissible_before(frame,VALIDATION_END)
                  | (frame.split=='test'))
    purged = frame.loc[~admissible].groupby('split').size().to_dict()
    frame = frame.loc[admissible].sort_values(['split','source_timestamp','attempt_id']).reset_index(drop=True)
    assert frame.attempt_id.is_unique
    assert (frame.previous_time.dropna() < frame.loc[frame.previous_time.notna(),'source_timestamp']).all()
    assert ((frame.prior_progression_count_q90+frame.prior_nonprogression_count_q90)
            <=frame.prior_attempt_count).all()
    outputs = {}
    for outcome, mask in [('successful_contact',np.ones(len(frame),dtype=bool)),
                          ('contact_chat',frame.source_event=='chat_click'),
                          ('phone_contact',frame.source_event=='contact_phone_click_1')]:
        subset = frame.loc[mask]
        path = output / f'bim_features_{outcome}.parquet'
        if path.exists():
            raise FileExistsError(path)
        subset.to_parquet(path,index=False,compression='zstd')
        outputs[outcome] = {'path':str(path),'sha256':sha256(path),
            'split_counts':subset.groupby('split').size().to_dict(),
            'nonprogression_rates':subset.groupby('split').nonprogression_q90.mean().to_dict()}
    summary = {'status':'completed','training_fit_cutoff':TRAIN_END,'validation_end_cutoff':VALIDATION_END,
        'source_path':str(SOURCE),'source_sha256':sha256(SOURCE),
        'pathway_parameters':params,'parameter_quantiles':'exact empirical quantile_disc',
        'parameter_fit_policy':'positive-time adjacent transitions with target timestamp strictly before training cutoff',
        'context_policy':'events strictly before query time; latest-time event ties resolved lexicographically using observed events only',
        'position_policy':'predictive position replaced by strict prior-event count; original_position retained for identity',
        'history_policy':'Q90 outcome counts admitted only after complete maturity, strictly before query time',
        'label_policy':'Q95 complete before train/validation boundary; test Q95 within observed data endpoint',
        'evaluation_status':'retrospective validity repair, not a new untouched holdout',
        'purged_rows':{str(k):int(v) for k,v in purged.items()},'outputs':outputs,
        'duration_seconds':time.time()-started}
    (output/'feature_build_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    con.close()
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
