"""Deterministic candidate selection and exact sparse additive ranking."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def require_component_available(selection_outcomes_end,evaluation_start):
    if selection_outcomes_end>evaluation_start:
        raise ValueError('Component selection uses outcomes after evaluation starts')


def require_disjoint_outcome_intervals(development,evaluation):
    if development[0]>=development[1] or evaluation[0]>=evaluation[1]:
        raise ValueError('Empty or reversed outcome interval')
    if max(development[0],evaluation[0])<min(development[1],evaluation[1]):
        raise ValueError('Development and evaluation outcome intervals overlap')


def stable_topk(ids, scores, k):
    ids=np.asarray(ids,dtype=np.int64)
    scores=np.asarray(scores,dtype=np.float64)
    if k<0 or len(ids)!=len(scores) or len(np.unique(ids))!=len(ids):
        raise ValueError('Invalid k, mismatched arrays, or duplicate IDs')
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite scores')
    n=min(k,len(ids))
    if n==0:
        return ids[:0]
    if len(ids)>n:
        threshold=np.partition(scores,-n)[-n]
        above=np.flatnonzero(scores>threshold)
        ties=np.flatnonzero(scores==threshold)
        ties=ties[np.argsort(ids[ties],kind='stable')[:n-len(above)]]
        chosen=np.concatenate([above,ties])
    else:
        chosen=np.arange(len(ids))
    return ids[chosen[np.lexsort((ids[chosen],-scores[chosen]))]]


def stable_topk_rows(matrix,k):
    matrix=matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    rows=[];cols=[];values=[]
    for row in range(matrix.shape[0]):
        lo,hi=matrix.indptr[row:row+2]
        ids=matrix.indices[lo:hi]
        scores=matrix.data[lo:hi]
        keep=stable_topk(ids,scores,k)
        pos=np.searchsorted(ids,keep)
        rows.extend([row]*len(keep));cols.extend(keep);values.extend(scores[pos])
    return sparse.csr_matrix((np.asarray(values,dtype=matrix.dtype),(rows,cols)),shape=matrix.shape)


def sampled_candidates(eligible_ids, seen_ids, positive, negatives, seed, case_id):
    """Per-case RNG makes candidates invariant to batching and case order."""
    eligible=np.unique(np.asarray(eligible_ids,dtype=np.int64))
    seen=np.unique(np.asarray(seen_ids,dtype=np.int64))
    if positive not in eligible or positive in seen or negatives<0 or case_id<0:
        raise ValueError('Ineligible target or invalid sampling configuration')
    pool=np.setdiff1d(eligible,np.union1d(seen,[positive]),assume_unique=True)
    if len(pool)<negatives:
        raise ValueError('Insufficient distinct unseen candidates')
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(case_id)]))
    chosen=rng.choice(pool,size=negatives,replace=False)
    return np.concatenate([[positive],chosen]).astype(np.int64)


def rank_target(candidate_ids,scores,positive):
    ids=np.asarray(candidate_ids)
    scores=np.asarray(scores)
    if len(np.unique(ids))!=len(ids) or len(ids)!=len(scores) or not np.isfinite(scores).all():
        raise ValueError('Candidates must be distinct with finite aligned scores')
    hit=np.flatnonzero(ids==positive)
    if len(hit)!=1:
        raise ValueError('Exactly one target is required')
    target_score=scores[hit[0]]
    return int(1+np.count_nonzero((scores>target_score)|((scores==target_score)&(ids<positive))))


def exact_sparse_fusion(base_ids,base_scores,auxiliary,eligible_ids,seen_ids,gamma,k=10):
    """Score nonnegative sparse support plus best K eligible outside-support items.

    The auxiliary ordering is reusable across users. Here it is constructed
    inside the reference implementation for clarity. Base normalization uses
    the maximum eligible unseen base score and defaults to one if it is zero.
    """
    ids=np.asarray(base_ids,dtype=np.int64)
    scores=np.asarray(base_scores,dtype=np.float64)
    auxiliary=np.asarray(auxiliary,dtype=np.float64)
    eligible=np.unique(np.asarray(eligible_ids,dtype=np.int64))
    seen=np.asarray(seen_ids,dtype=np.int64)
    if gamma<0 or len(ids)!=len(scores) or len(np.unique(ids))!=len(ids):
        raise ValueError('Invalid sparse additive inputs')
    if not np.isfinite(scores).all() or np.any(scores<0) or not np.isfinite(auxiliary).all():
        raise ValueError('Nonnegative finite base and finite auxiliary required')
    if np.any(eligible<0) or np.any(eligible>=len(auxiliary)) or k<0:
        raise ValueError('Invalid eligible catalog or k')
    if np.any(ids<0) or np.any(ids>=len(auxiliary)):
        raise ValueError('Sparse IDs outside catalog')
    available=np.setdiff1d(eligible,seen,assume_unique=False)
    keep=np.isin(ids,available)
    ids,scores=ids[keep],scores[keep]
    if len(scores) and scores.max()>0:
        scores=scores/scores.max()
    outside=np.setdiff1d(available,ids,assume_unique=False)
    extra=stable_topk(outside,gamma*auxiliary[outside],k)
    candidate_ids=np.concatenate([ids,extra])
    combined=np.concatenate([scores,np.zeros(len(extra))])+gamma*auxiliary[candidate_ids]
    return stable_topk(candidate_ids,combined,k)


def full_catalog_user_metrics(recommendations,relevant,k):
    if k<=0 or len(recommendations)>k or len(set(recommendations))!=len(recommendations):
        raise ValueError('Invalid recommendation list')
    relevant=set(relevant)
    if not relevant:
        raise ValueError('User must have eligible relevance')
    hits=np.zeros(k,dtype=bool)
    hits[:len(recommendations)]=[item in relevant for item in recommendations]
    ranks=np.arange(1,k+1)
    where=np.flatnonzero(hits)
    return {'precision':float(hits.sum()/k),'recall':float(hits.sum()/len(relevant)),
        'ndcg':float(np.sum(hits/np.log2(ranks+1))/np.sum(1/np.log2(np.arange(2,min(k,len(relevant))+2)))),
        'mAP':float(np.sum(np.cumsum(hits)/ranks*hits)/min(k,len(relevant))),
        'MRR':float(1/(where[0]+1) if len(where) else 0),'HR':float(bool(hits.any()))}
