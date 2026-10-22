"""Shared data, scoring and output contracts for retrospective OLX experiments."""
from __future__ import annotations

import ctypes
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

from artifact_hash import sha256
from protocol_primitives import stable_topk, stable_topk_rows, full_catalog_user_metrics
from comparison_adapter import build_rp3beta

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parents[1]
INPUTS = REV / 'protocol_checks/evaluation_inputs'
SOURCE = ROOT / 'data/olx/04_sequences/ui_sequences.parquet'
FEATURES = REV / 'features/bim_features_successful_contact.parquet'
CONTACTS = ('contact_chat', 'contact_phone_click_2', 'contact_phone_click_3')
P3_D = (.631, .163, .433)
P3_R = (.081, .008, .028)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def verify_sources():
    build = read_json(REV / 'features/feature_build_summary.json')
    if sha256(SOURCE) != build['source_sha256']:
        raise ValueError('Interaction source changed since feature repair')
    expected = build['outputs']['successful_contact']['sha256']
    if sha256(FEATURES) != expected:
        raise ValueError('Corrected feature source changed')
    return {'interactions_sha256': build['source_sha256'], 'features_sha256': expected}


def connection():
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute('SET threads=4')
    con.execute(f"CREATE VIEW events AS SELECT user_idx,item_idx,event,timestamp FROM read_parquet('{SOURCE.as_posix()}')")
    return con


def peak_working_set_bytes():
    if platform.system() != 'Windows':
        return None
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
    counter = Counters(); counter.cb = ctypes.sizeof(counter)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    fn = ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if fn(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
        return int(counter.PeakWorkingSetSize)
    return None


@dataclass
class TrainingData:
    cutoff: int
    catalog: np.ndarray
    users: np.ndarray
    pairs: pd.DataFrame
    seen: sparse.csr_matrix
    priors: dict
    counts: np.ndarray
    bands: np.ndarray
    preparation_seconds: float
    miniature_graph: bool

    def rows(self, users):
        users = np.asarray(users, dtype=np.int64)
        rows = np.searchsorted(self.users, users)
        if np.any(rows >= len(self.users)) or not np.array_equal(self.users[rows], users):
            raise ValueError('Evaluation user is absent from the training graph')
        return rows


def load_training(cutoff, catalog, graph_users=None):
    started = time.perf_counter(); con = connection()
    catalog = np.asarray(catalog, dtype=np.int64)
    if len(catalog) == 0 or not np.array_equal(catalog, np.unique(catalog)):
        raise ValueError('Catalog must be nonempty, unique and sorted')
    # Global priors always use the full training prefix, including in the smoke run.
    stats = con.execute('''SELECT item_idx,count(*) n,
        count(*) FILTER(WHERE event IN ('contact_chat','contact_phone_click_2','contact_phone_click_3')) s,
        max(timestamp) last_ts FROM events WHERE timestamp<? GROUP BY item_idx ORDER BY item_idx''', [int(cutoff)]).fetchdf()
    if not np.array_equal(stats.item_idx.to_numpy(), catalog):
        raise ValueError('Training item universe differs from approved catalog')
    clause = ''
    if graph_users is not None:
        con.register('graph_users', pd.DataFrame({'user_idx': np.unique(graph_users)}))
        clause = 'JOIN graph_users USING(user_idx)'
    pairs = con.execute(f'''SELECT user_idx,item_idx,max(timestamp) last_ts
        FROM events {clause} WHERE timestamp<? GROUP BY user_idx,item_idx ORDER BY user_idx,item_idx''', [int(cutoff)]).fetchdf()
    con.close()
    if pairs.empty or (pairs.last_ts >= cutoff).any():
        raise ValueError('Empty or temporally invalid training graph')
    users = np.sort(pairs.user_idx.unique()); pairs['u'] = np.searchsorted(users, pairs.user_idx)
    pairs['user_last'] = pairs.groupby('user_idx').last_ts.transform('max')
    size = int(catalog.max()) + 1
    seen = sparse.csr_matrix((np.ones(len(pairs)), (pairs.u.to_numpy(), pairs.item_idx.to_numpy())), shape=(len(users), size))
    seen.sort_indices()
    n = stats.n.to_numpy(float); s = stats.s.to_numpy(float)
    global_rate = s.sum() / n.sum()
    definitions = {'popularity':np.log1p(n), 'contact_count':np.log1p(s),
        'contact_rate':(s + 20 * global_rate) / (n + 20),
        'recency':np.exp(-np.log(2) * (cutoff-stats.last_ts.to_numpy()) / 86400)}
    priors = {}
    for name, values in definitions.items():
        out = np.zeros(size); out[catalog] = values / max(float(values.max()), np.finfo(float).tiny)
        priors[name] = out
    counts = np.zeros(size, dtype=np.int64); counts[catalog] = stats.n.to_numpy()
    # Four equally sized item groups, low to high training event count; IDs break ties.
    ordered = catalog[np.lexsort((catalog, counts[catalog]))]
    bands = np.full(size, -1, dtype=np.int8)
    for band, ids in enumerate(np.array_split(ordered, 4)): bands[ids] = band
    return TrainingData(int(cutoff),catalog,users,pairs,seen,priors,counts,bands,
                        time.perf_counter()-started,graph_users is not None)


def p3_graph(data, block_size=200):
    pairs = data.pairs; u = pairs.u.to_numpy(); i = pairs.item_idx.to_numpy()
    ud = np.asarray(data.seen.sum(1)).ravel(); degree = np.asarray(data.seen.sum(0)).ravel()
    age = (pairs.user_last.to_numpy()-pairs.last_ts.to_numpy())/86400
    w0 = np.maximum(degree[i],1)**-P3_D[0] * np.exp(-age*P3_R[0])
    w1 = np.maximum(ud[u],1)**-P3_D[1] * np.exp(-age*P3_R[1])
    w2 = np.maximum(degree[i],1)**-P3_D[2] * np.exp(-age*P3_R[2])
    ui0 = sparse.csr_matrix((w0,(u,i)),shape=data.seen.shape)
    iu1 = sparse.csr_matrix((w1,(i,u)),shape=data.seen.T.shape)
    ui2 = sparse.csr_matrix((w2,(u,i)),shape=data.seen.shape)
    parts=[]
    for start in range(0,data.seen.shape[1],block_size):
        z=(iu1[start:start+block_size]@ui2).tocsr()
        for row in range(z.shape[0]):
            lo,hi=z.indptr[row:row+2]
            z.data[lo:hi][z.indices[lo:hi]==start+row]=0
        z.eliminate_zeros();parts.append(stable_topk_rows(z,205))
    return ui0,sparse.vstack(parts,format='csr')


def build_graph(data, name, block_size):
    started=time.perf_counter()
    if name=='rp3beta': edge,similarity=build_rp3beta(data.seen,block_size=block_size)
    elif name=='p3ltr_degree_recency':edge,similarity=p3_graph(data,block_size)
    else:raise ValueError(name)
    return edge,similarity,{'build_seconds':time.perf_counter()-started,
        'peak_process_working_set_bytes':peak_working_set_bytes(),'similarity_nnz':int(similarity.nnz)}


class PreparedFusion:
    """Reuse catalog membership and outside-support ordering across users."""
    def __init__(self,catalog,auxiliary,gamma,k=10):
        self.catalog=np.asarray(catalog,dtype=np.int64);self.aux=np.asarray(auxiliary,dtype=float)
        self.gamma=float(gamma);self.k=k
        if gamma<0 or not np.isfinite(self.aux).all():raise ValueError('Invalid prior')
        self.mask=np.zeros(len(self.aux),bool);self.mask[self.catalog]=True
        self.order=self.catalog[np.lexsort((self.catalog,-gamma*self.aux[self.catalog]))]

    def rank(self,ids,values,seen):
        ids=np.asarray(ids,dtype=np.int64);values=np.asarray(values,dtype=float)
        if len(ids)!=len(values) or len(np.unique(ids))!=len(ids) or np.any(values<0) or not np.isfinite(values).all():
            raise ValueError('Invalid sparse base score')
        keep=self.mask[ids]&~np.isin(ids,seen);ids=ids[keep];values=values[keep]
        if len(values) and values.max()>0:values=values/values.max()
        excluded=set(ids.tolist());excluded.update(np.asarray(seen).tolist());extra=[]
        for item in self.order:
            if int(item) not in excluded:
                extra.append(int(item))
                if len(extra)==self.k:break
        items=np.concatenate([ids,np.asarray(extra,dtype=np.int64)])
        scores=np.concatenate([values,np.zeros(len(extra))])+self.gamma*self.aux[items]
        result=stable_topk(items,scores,self.k)
        if np.isin(result,seen).any() or not self.mask[result].all():raise AssertionError('Eligibility violation')
        return result


def gather_scores(scores,candidates):
    scores=scores.tocsr();scores.sort_indices();out=np.zeros(candidates.shape)
    for row,c in enumerate(candidates):
        lo,hi=scores.indptr[row:row+2];ids=scores.indices[lo:hi]
        pos=np.searchsorted(ids,c);valid=pos<len(ids);matched=np.zeros(len(c),bool)
        matched[valid]=ids[pos[valid]]==c[valid]
        out[row,matched]=scores.data[lo:hi][pos[matched]]
    return out


def zscore_rows(x):
    return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-8)


def build_mature_profiles(cutoff,catalog,output):
    if cutoff != 1582312306:raise ValueError('Current maturity estimates require sampled training cutoff')
    con=duckdb.connect()
    frame=con.execute('''SELECT item_idx,source_event,count(*) attempts,
        sum(progressed_q90) successes,sum(nonprogression_q90) nonprogressions,
        max(source_timestamp+q95_seconds) latest_admissibility
        FROM read_parquet(?) WHERE split='train' AND source_timestamp+q95_seconds<?
        GROUP BY item_idx,source_event ORDER BY item_idx,source_event''',[str(FEATURES),int(cutoff)]).fetchdf()
    total=con.execute("SELECT count(*) FROM read_parquet(?) WHERE split='train'",[str(FEATURES)]).fetchone()[0]
    con.close()
    if int(frame.attempts.sum())!=total or (frame.latest_admissibility>=cutoff).any():
        raise AssertionError('Training labels fail maturity boundary')
    if not np.isin(frame.item_idx,catalog).all() or not np.array_equal(frame.successes+frame.nonprogressions,frame.attempts):
        raise AssertionError('Invalid profile counts or catalog')
    frame.to_parquet(output/'mature_item_profiles.parquet',index=False)
    arrays={}
    for event in ['chat_click','contact_phone_click_1']:
        group=frame[frame.source_event==event];a=np.zeros((2,int(max(catalog))+1))
        a[0,group.item_idx.to_numpy()]=group.successes.to_numpy()
        a[1,group.item_idx.to_numpy()]=group.nonprogressions.to_numpy();arrays[event]=a
    write_json(output/'mature_item_profiles_provenance.json',{'training_cutoff':cutoff,
       'admitted_attempts':int(total),'maximum_label_admissibility':int(frame.latest_admissibility.max()),
       'source_sha256':sha256(FEATURES),'profile_sha256':sha256(output/'mature_item_profiles.parquet'),
       'definition':'Q90 outcomes from corrected training attempts with complete Q95 before cutoff; counts are attempts, not distinct users',
       'no_validation_or_test_profiles':True})
    return arrays


def relation_scores(candidates,events,risk,profiles):
    risk=np.asarray(risk,dtype=float)
    if not np.isfinite(risk).all() or np.any((risk<0)|(risk>1)):raise ValueError('Risk outside [0,1]')
    relation=np.zeros(candidates.shape)
    for event in np.unique(events):
        if event not in profiles:raise ValueError('Unknown source pathway')
        selected=events==event;c=candidates[selected];a=profiles[event]
        relation[selected]=np.log1p(a[0,c])-risk[selected,None]*np.log1p(a[1,c])
    return zscore_rows(relation)


def sampled_metrics(ranks,k=10):
    r=np.asarray(ranks);hit=r<=k
    return {'recall':float(hit.mean()),'ndcg':float(np.where(hit,1/np.log2(r+1),0).mean()),
            'MRR':float(np.where(hit,1/r,0).mean()),'HR':float(hit.mean()),'cases':len(r)}


def evaluate_lists(recs,relevant,users,catalog,bands):
    records=[];frequency=np.zeros(len(bands),np.int64)
    for user,items in zip(users,recs):
        valid=items[items>=0];rel=relevant[int(user)]
        metrics=full_catalog_user_metrics(valid,rel,10)
        tail=[item for item in rel if bands[item]==0]
        metrics.update(user_idx=int(user),tail_target=bool(tail),
                       tail_ndcg=full_catalog_user_metrics(valid,tail,10)['ndcg'] if tail else np.nan)
        records.append(metrics);np.add.at(frequency,valid,1)
    frame=pd.DataFrame(records);summary={name:float(frame[name].mean()) for name in ['precision','recall','ndcg','mAP','MRR','HR']}
    summary.update(users=len(users),coverage=float(np.count_nonzero(frequency[catalog])/len(catalog)),
       tail_exposure=float(frequency[bands==0].sum()/max(frequency.sum(),1)),
       head_exposure=float(frequency[bands==3].sum()/max(frequency.sum(),1)),
       tail_target_users=int(frame.tail_target.sum()),
       tail_target_ndcg=float(frame.loc[frame.tail_target,'tail_ndcg'].mean()) if frame.tail_target.any() else None)
    return summary,frame,frequency
