"""Matched-budget retrospective contact-model reruns on repaired features."""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
REV=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from train_bim_e8a_olx import MLP, prepare_data, train_model, predict, save_predictions, seed_everything
from ablate_bim_mlp_olx import COMPONENTS, subset_data
from confirm_bim_evidence_cross_outcome_olx import set_seed, transition_only_weights
from artifact_hash import sha256

SEEDS=[20260827,20260828,20260829]
EPOCHS=8
BATCH_SIZE=8192
EVIDENCE={'prior_attempt_count','prior_nonprogression_count_q90','prior_progression_count_q90',
          'seconds_since_previous_source_attempt','seconds_since_previous_source_attempt_missing'}


class TrainingLog:
    def __init__(self,path,original):
        self.stream=path.open('a',encoding='utf-8',buffering=1)
        self.original=original

    def write(self,text):
        self.stream.write(text)
        return self.original.write(text)

    def flush(self):
        self.stream.flush()
        self.original.flush()


def main():
    build=json.loads((REV/'features/feature_build_summary.json').read_text(encoding='utf-8'))
    verification=json.loads((REV/'features/verification.json').read_text(encoding='utf-8'))
    if verification['status']!='passed' or verification['feature_manifest_sha256']!=sha256(REV/'features/feature_build_summary.json'):
        raise RuntimeError('Feature verification missing or stale')
    output=REV/'model_reruns'
    output.mkdir(exist_ok=True)
    sys.stdout=TrainingLog(output/'training.log',sys.stdout)
    rows=[]
    dependencies={name:sha256(ROOT/'src'/name) for name in
        ['train_bim_e8a_olx.py','ablate_bim_mlp_olx.py','confirm_bim_evidence_cross_outcome_olx.py']}
    seed_everything()
    for outcome in ['successful_contact','contact_chat','phone_contact']:
        path=REV/'features'/f'bim_features_{outcome}.parquet'
        feature_hash=sha256(path)
        if feature_hash!=build['outputs'][outcome]['sha256']:
            raise RuntimeError('Feature hash mismatch')
        full=prepare_data(path)
        keep=[i for i,name in enumerate(full.feature_names) if name not in COMPONENTS]
        evidence=transition_only_weights(subset_data(full,keep),path)
        variants={'primitive_evidence':evidence}
        if outcome=='successful_contact':
            context_keep=[i for i,name in enumerate(evidence.feature_names) if name not in EVIDENCE]
            variants['context_control']=subset_data(evidence,context_keep)
            variants['engineered_Q_weighted']=full
        for variant,data in variants.items():
            seeds=SEEDS if variant!='engineered_Q_weighted' else SEEDS[:1]
            for seed in seeds:
                directory=output/outcome/variant/f'seed_{seed}'
                config={'outcome':outcome,'variant':variant,'seed':seed,'feature_sha256':feature_hash,
                    'dependency_sha256':dependencies,
                    'feature_names':data.feature_names,'scaler':data.scaler,
                    'epochs_requested':EPOCHS,'batch_size':BATCH_SIZE,'learning_rate':0.001,
                    'model':'MLP 64-ReLU-Dropout(.20)-32-ReLU-Dropout(.10)-1',
                    'weighting':'transition only' if variant!='engineered_Q_weighted' else 'transition times Q',
                    'selection':'minimum weighted validation loss; patience 2; threshold maximizes validation F1',
                    'interpretation':'retrospective repair; previously inspected periods are not an untouched holdout'}
                metric_path=directory/'metrics.json'
                if metric_path.exists():
                    old=json.loads((directory/'config.json').read_text(encoding='utf-8'))
                    if old!=config:
                        raise RuntimeError(f'Resume configuration mismatch: {directory}')
                    result=json.loads(metric_path.read_text(encoding='utf-8'))
                else:
                    if directory.exists() and any(directory.iterdir()):
                        raise FileExistsError(f'Incomplete run requires inspection: {directory}')
                    directory.mkdir(parents=True,exist_ok=True)
                    (directory/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
                    set_seed(seed)
                    print(f'Training {outcome} / {variant} / {seed}',flush=True)
                    model=MLP(len(data.feature_names))
                    result,test_p=train_model(model,data,directory,EPOCHS,BATCH_SIZE,learning_rate=.001)
                    save_predictions(directory/'test_predictions.parquet',data.attempt_ids['test'],data.y['test'],test_p)
                    validation_p=predict(model,data.x['validation'],BATCH_SIZE)
                    save_predictions(directory/'validation_predictions.parquet',data.attempt_ids['validation'],data.y['validation'],validation_p)
                    (directory/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
                    del model
                rows.append({'outcome':outcome,'variant':variant,'seed':seed,
                    'epochs_completed':result['epochs_completed'],**result['test']})
                pd.DataFrame(rows).to_csv(output/'completed_runs.csv',index=False)
                (output/'progress.json').write_text(json.dumps({'status':'running',
                    'completed_runs':len(rows),'planned_runs':13,'last_completed':rows[-1]},indent=2),encoding='utf-8')
                print(json.dumps(rows[-1]),flush=True)
        del full,evidence,variants,data
        gc.collect()
    summary={'status':'completed','seeds':SEEDS,'epochs_requested':EPOCHS,
        'results':rows,'feature_manifest_sha256':sha256(REV/'features/feature_build_summary.json'),
        'scope':'contact prediction and pooled evidence/context ablations; ranking controls require separate rerun'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (output/'progress.json').write_text(json.dumps({'status':'completed','completed_runs':len(rows)},indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
