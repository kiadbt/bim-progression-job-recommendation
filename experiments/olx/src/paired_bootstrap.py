"""Paired user bootstrap for differences in bounded ranking metrics."""
import hashlib

import numpy as np


def bootstrap_means(differences, replicates=5000, seed=20261020, key='default'):
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('A nonempty finite one-dimensional paired difference is required')
    if replicates < 2:
        raise ValueError('At least two bootstrap replicates are required')
    unique, counts = np.unique(values, return_counts=True)
    entropy = [seed, *np.frombuffer(hashlib.sha256(key.encode()).digest()[:16], dtype='<u4').tolist()]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    output = np.empty(replicates)
    if len(unique) == 1:
        output.fill(unique[0])
        return output
    # Multinomial frequencies over identical differences are distributionally
    # equivalent to sampling N paired users with replacement. No rounding or
    # normal approximation is used. This avoids materializing N indices per draw.
    probabilities = counts / len(values)
    for start in range(0, replicates, 64):
        weights = rng.multinomial(len(values), probabilities, size=min(64, replicates-start))
        output[start:start+len(weights)] = (weights @ unique) / len(values)
    return output


def paired_summary(left, right, *, key, primary=False, replicates=5000, seed=20261020):
    left = np.asarray(left, dtype=float); right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError('Paired arrays must have identical shapes')
    delta = left-right
    draws = bootstrap_means(delta, replicates, seed, key)
    q = np.quantile(draws, [.005, .025, .975, .995], method='linear')
    baseline = float(right.mean())
    result = {
        'contrast': key, 'n_users': len(left), 'left_mean': float(left.mean()),
        'right_mean': baseline, 'difference': float(delta.mean()),
        'relative_change_percent': float(100*delta.mean()/baseline) if baseline else None,
        'ci95': [float(q[1]), float(q[2])],
        'ci99_bonferroni_five': [float(q[0]), float(q[3])] if primary else None,
        'primary_family': primary, 'replicates': replicates, 'master_seed': seed,
        'positive_users': int(np.count_nonzero(delta>0)),
        'negative_users': int(np.count_nonzero(delta<0)),
        'unchanged_users': int(np.count_nonzero(delta==0)),
        'unique_differences': len(np.unique(delta)),
        'bootstrap_standard_error': float(draws.std(ddof=1)),
        'empirical_bootstrap_standard_error': float(delta.std(ddof=0)/np.sqrt(len(delta))),
        'interval_scope': '99% marginal intervals give approximate 95% Bonferroni family coverage for five contrasts'
                          if primary else 'Exploratory pointwise 95% interval without multiplicity adjustment',
    }
    interval = result['ci99_bonferroni_five'] if primary else result['ci95']
    result['direction_supported'] = 'positive' if interval[0]>0 else 'negative' if interval[1]<0 else 'not_resolved'
    return result, draws
