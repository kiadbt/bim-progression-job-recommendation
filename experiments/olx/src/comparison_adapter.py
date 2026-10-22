"""Common graph and ranking adapter for repaired comparison drivers."""
import numpy as np
from scipy import sparse

from protocol_primitives import stable_topk_rows, exact_sparse_fusion, rank_target


def build_rp3beta(binary_history, beta=.3, topk=200, block_size=200):
    """Alpha one, zero diagonal, destination degree penalty, exact boundary ties."""
    if beta < 0 or topk < 1 or block_size < 1:
        raise ValueError('Invalid graph parameters')
    history = binary_history.tocsr(copy=True).astype(np.float64)
    history.sum_duplicates(); history.eliminate_zeros()
    if not np.isfinite(history.data).all() or np.any(history.data < 0):
        raise ValueError('History must be finite and nonnegative')
    history.data[:] = 1
    user_degree = np.asarray(history.sum(1)).ravel()
    item_degree = np.asarray(history.sum(0)).ravel()
    p_ui = sparse.diags(1 / np.maximum(user_degree, 1)) @ history
    p_iu = sparse.diags(1 / np.maximum(item_degree, 1)) @ history.T
    penalty = np.maximum(item_degree, 1) ** -beta
    blocks = []
    for start in range(0, history.shape[1], block_size):
        block = (p_iu[start:start + block_size] @ p_ui).tocsr()
        block.data *= penalty[block.indices]
        for row in range(block.shape[0]):
            lo, hi = block.indptr[row:row + 2]
            block.data[lo:hi][block.indices[lo:hi] == start + row] = 0
        block.eliminate_zeros()
        blocks.append(stable_topk_rows(block, topk))
    similarity = sparse.vstack(blocks, format='csr') if blocks else sparse.csr_matrix((0, 0))
    return p_ui.tocsr(), similarity


def rank_full_rows(scores, seen, eligible_ids, auxiliary, gamma, k=10):
    """Rows of scores and seen must represent exactly the same ordered users."""
    scores = scores.tocsr(); seen = seen.tocsr()
    if scores.shape != seen.shape:
        raise ValueError('Score and seen rows must align')
    output = np.full((scores.shape[0], k), -1, dtype=np.int64)
    for row in range(scores.shape[0]):
        lo, hi = scores.indptr[row:row + 2]
        a, b = seen.indptr[row:row + 2]
        ids = exact_sparse_fusion(scores.indices[lo:hi], scores.data[lo:hi],
                                  auxiliary, eligible_ids, seen.indices[a:b], gamma, k)
        output[row, :len(ids)] = ids
    return output


def sampled_target_ranks(candidates, scores, targets):
    candidates = np.asarray(candidates); scores = np.asarray(scores)
    if candidates.shape != scores.shape or len(targets) != len(candidates):
        raise ValueError('Sampled scores must align to shared candidates')
    return np.array([rank_target(ids, values, target)
                     for ids, values, target in zip(candidates, scores, targets)])


def select_operating_point(rows, baseline_coverage):
    """Development metrics only; caller must not pass evaluation-period rows."""
    if baseline_coverage <= 0:
        raise ValueError('A positive development baseline coverage is required')
    eligible = []
    for row in rows:
        if row.get('split') != 'development':
            raise ValueError('Operating point selection is development only')
        if not all(np.isfinite(row[key]) for key in ['ndcg', 'recall', 'gamma', 'coverage']):
            raise ValueError('Nonfinite selection metric')
        if row['gamma'] < 0:
            raise ValueError('Negative weight')
        if row['coverage'] >= .95 * baseline_coverage:
            eligible.append(row)
    if not eligible:
        raise ValueError('No coverage-admissible operating point')
    return max(eligible, key=lambda r: (r['ndcg'], r['recall'], -r['gamma']))
