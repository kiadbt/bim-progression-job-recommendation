"""Independent full-catalog metric oracle used by regression tests."""
import numpy as np
from scipy import sparse

def evaluate(recs: np.ndarray,relevance: sparse.csr_matrix,k=10):
 sums={x:0.0 for x in ['precision','recall','ndcg','mAP','MRR','HR']};freq=np.zeros(relevance.shape[1],np.int64)
 discount=1/np.log2(np.arange(2,k+2));ideal=np.cumsum(discount)
 for row,items in enumerate(recs):
  lo,hi=relevance.indptr[row],relevance.indptr[row+1];rel=relevance.indices[lo:hi];valid=items>=0;hits=np.zeros(k,bool);hits[valid]=np.isin(items[valid],rel,assume_unique=False);nh=hits.sum();nrel=len(rel)
  sums['precision']+=nh/k;sums['recall']+=nh/nrel;sums['ndcg']+=np.dot(hits,discount)/ideal[min(k,nrel)-1]
  sums['mAP']+=np.dot(np.cumsum(hits)/np.arange(1,k+1),hits)/min(k,nrel);where=np.flatnonzero(hits);sums['MRR']+=0 if not len(where) else 1/(where[0]+1);sums['HR']+=nh>0
  np.add.at(freq,items[valid],1)
 n=len(recs);result={x:v/n for x,v in sums.items()};used=freq>0;p=freq[freq>0]/freq.sum();result.update({'catalog_coverage':used.mean(),'shannon':float(-np.dot(p,np.log(p))),'users':n});return result
