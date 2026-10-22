"""Construct strict prediction-time histories for source attempts."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_DTYPE = np.dtype([('group', np.int64), ('time', np.int64)])


def _keys(groups: np.ndarray, times: np.ndarray) -> np.ndarray:
    keys = np.empty(len(groups), dtype=KEY_DTYPE)
    keys['group'], keys['time'] = groups, times
    return keys


def _prior_counts(groups: np.ndarray, times: np.ndarray,
                  event_groups: np.ndarray, available_at: np.ndarray) -> np.ndarray:
    ordered = np.sort(_keys(event_groups, available_at))
    lower = np.searchsorted(
        ordered, _keys(groups, np.full(len(groups), np.iinfo(np.int64).min)), side='left'
    )
    upper = np.searchsorted(ordered, _keys(groups, times), side='left')
    return (upper - lower).astype(np.int64)


def build_histories(attempts: pd.DataFrame) -> pd.DataFrame:
    """Require full Q90 maturity strictly before a query for either outcome count.

    Raw prior-attempt counts require only an earlier source timestamp. Grouping
    is by user-item sequence and pathway. Equal-time attempts are not ordered
    using an arbitrary position field. Input labels must come from a complete
    outcome window; this function does not estimate windows or create labels.
    """
    required = ['sequence_id', 'source_event', 'source_timestamp',
                'q90_seconds', 'progressed_q90']
    missing = set(required).difference(attempts.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if attempts[required].isna().any().any():
        raise ValueError('History inputs contain missing values')
    if not attempts.progressed_q90.isin([0, 1]).all():
        raise ValueError('Outcome labels must be binary')
    for name in ['source_timestamp', 'q90_seconds']:
        values = attempts[name].to_numpy()
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f'{name} must contain finite integer seconds')
    if (attempts.q90_seconds < 0).any():
        raise ValueError('Maturity horizons cannot be negative')
    groups, _ = pd.factorize(pd.MultiIndex.from_frame(attempts[['sequence_id', 'source_event']]))
    groups = groups.astype(np.int64)
    times = attempts.source_timestamp.to_numpy(np.int64)
    horizon = attempts.q90_seconds.to_numpy(np.int64)
    if np.any(times > np.iinfo(np.int64).max - horizon):
        raise ValueError('Timestamp plus horizon exceeds int64')
    matured_at = times + horizon
    positive = attempts.progressed_q90.to_numpy(np.int8) == 1
    raw = _prior_counts(groups, times, groups, times)
    pos = _prior_counts(groups, times, groups[positive], matured_at[positive])
    neg = _prior_counts(groups, times, groups[~positive], matured_at[~positive])
    ordered = np.sort(_keys(groups, times))
    preceding_index = np.searchsorted(ordered, _keys(groups, times), side='left') - 1
    valid = preceding_index >= 0
    safe_index = np.maximum(preceding_index, 0)
    if len(ordered):
        valid &= ordered['group'][safe_index] == groups
    previous = np.full(len(times), np.nan)
    if valid.any():
        previous[valid] = ordered['time'][preceding_index[valid]]
    result = pd.DataFrame({
        'prior_attempt_count': raw,
        'prior_progression_count_q90': pos,
        'prior_nonprogression_count_q90': neg,
        'prior_unresolved_attempt_count_q90': raw - pos - neg,
        'previous_source_timestamp': previous,
        'seconds_since_previous_source_attempt': times - previous,
    }, index=attempts.index)
    if (result.prior_unresolved_attempt_count_q90 < 0).any():
        raise AssertionError('Resolved histories exceed prior attempts')
    return result


def label_admissible_before(attempts: pd.DataFrame, fit_cutoff: int) -> np.ndarray:
    """Conservatively purge labels whose complete Q95 window crosses fit time."""
    return (attempts.source_timestamp.to_numpy(np.int64)
            + attempts.q95_seconds.to_numpy(np.int64)) < fit_cutoff
