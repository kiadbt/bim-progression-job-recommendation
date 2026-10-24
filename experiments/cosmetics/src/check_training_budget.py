"""Fixed-budget training diagnostic using October and November records only."""
import argparse
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '4'
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from copy import deepcopy
import json
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work-dir', type=Path, default=ROOT / 'local_work/cosmetics')
parser.add_argument('--output-dir', type=Path, default=ROOT / 'local_work/cosmetics_training_budget')
args = parser.parse_args()
STUDY = args.work_dir.resolve()
OUT = args.output_dir.resolve()
if OUT.exists():
    raise FileExistsError('Use a fresh output directory; existing diagnostics are preserved.')
OUT.mkdir(parents=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation as original

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def main():
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'question': 'Does a fixed extension from eight to 32 training epochs materially change development prediction diagnostics?',
        'budget': 'Exactly 24 partial_fit epochs continuing each saved eight-epoch MLP; no checkpoint or seed selection.',
        'fixed': ['training cases', 'features', 'training scaler', 'architecture', 'optimizer state', 'learning rate', 'seed'],
        'assessment': 'October conditional calibration and November development prediction only; report original and extended fits.',
        'calibration': 'Refit the same positive-slope sigmoid on original October calibration cases for each extended model.',
        'stopping': 'Fixed 32 total epochs; no stopping or model selection based on calibration or November metrics.',
        'exclusions': 'No December data, risks, ranks, metrics, graph construction or model selection. No replacement of published eight-epoch results.',
        'status': 'Post hoc training diagnostic, not independent confirmation or a ranking experiment.',
        'inputs': {name: digest(STUDY / name) for name in [
            'models.pkl', 'cart_features_development.parquet', 'calibration_cases.parquet', 'development/paired_cases.parquet']},
        'script_sha256': digest(Path(__file__)),
    }
    lock = OUT / 'protocol.json'
    assert not lock.exists(), 'Do not overwrite a completed diagnostic or its protocol.'
    write(lock, protocol)
    with (STUDY / 'models.pkl').open('rb') as stream:
        saved = pickle.load(stream)
    bundle = deepcopy(saved)
    features = pd.read_parquet(STUDY / 'cart_features_development.parquet')
    assert features.t.max() < original.ts('2019-12-01')
    train = features[(features.t >= original.ts('2019-10-01')) &
        (features.t < original.ts('2019-10-20')) &
        (features.t + original.DAY < original.ts('2019-10-21'))].drop_duplicates('user_id', keep='last')
    train = original.cap_users(train, 50000)
    assert len(train) == 50000
    assess = {}
    for phase, filename in [('calibration', 'calibration_cases.parquet'), ('development', 'development/paired_cases.parquet')]:
        ids = pd.read_parquet(STUDY / filename)
        merged = ids[['case_id', 'user_id', 't', 'y']].merge(features, on=['case_id', 'user_id', 't', 'y'], validate='one_to_one')
        assert len(merged) == len(ids)
        assess[phase] = merged
    results = {'models': {}, 'training_users': len(train), 'december_accessed': False, 'ranking_rerun': False}
    for name in ['actual', 'context']:
        scaler, model, old_calibration, n = bundle[name]
        x = scaler.transform(train[original.FEATURES[:n]].to_numpy(np.float32))
        assert np.allclose(scaler.mean_, train[original.FEATURES[:n]].to_numpy(np.float32).mean(0), atol=1e-5)
        before = len(model.loss_curve_)
        assert before == 8
        for _ in range(24):
            model.partial_fit(x, train.y)
            print(name, 'total_epochs', len(model.loss_curve_), 'loss', model.loss_curve_[-1], flush=True)
        assert len(model.loss_curve_) == 32
        calibration_cases = assess['calibration']
        raw = model.predict_proba(scaler.transform(calibration_cases[original.FEATURES[:n]].to_numpy(np.float32)))[:, 1]
        new_calibration = original.fit_calibration(raw, calibration_cases.y)
        bundle[name] = (scaler, model, new_calibration, n)
        row = {'loss_curve': model.loss_curve_, 'final_loss_change': model.loss_curve_[-1] - model.loss_curve_[-2],
            'no_improvement_count': int(model._no_improvement_count), 'patience': model.n_iter_no_change,
            'tolerance': model.tol, 'calibration_parameters': new_calibration, 'assessment': {}}
        for phase, c in assess.items():
            row['assessment'][phase] = {'users': len(c), 'nonprogression_rate': float(c.y.mean())}
            for label, packed in [('original_eight_epochs', saved[name]), ('extended_32_epochs', bundle[name])]:
                sc, clf, cal, count = packed
                raw = clf.predict_proba(sc.transform(c[original.FEATURES[:count]].to_numpy(np.float32)))[:, 1]
                risk = original.calibrated(raw, cal)
                row['assessment'][phase][label] = {
                    'auc': float(roc_auc_score(c.y, risk)), 'average_precision': float(average_precision_score(c.y, risk)),
                    'brier': float(brier_score_loss(c.y, risk)), 'log_loss': float(log_loss(c.y, risk)),
                    'raw_brier': float(brier_score_loss(c.y, raw)), 'raw_log_loss': float(log_loss(c.y, raw))}
        results['models'][name] = row
    with (OUT / 'extended_models.pkl').open('wb') as stream:
        pickle.dump(bundle, stream)
    for name, expected in protocol['inputs'].items():
        assert digest(STUDY / name) == expected
    results['completed_utc'] = datetime.now(timezone.utc).isoformat()
    results['protocol_sha256'] = digest(lock)
    write(OUT / 'results.json', results)
    print(json.dumps(results, indent=2), flush=True)

if __name__ == '__main__':
    main()
