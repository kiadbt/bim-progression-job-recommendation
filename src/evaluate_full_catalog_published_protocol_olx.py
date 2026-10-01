"""Full-catalog evaluation using the authors' published OLX 80/20 protocol."""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import duckdb,numpy as np,pandas as pd
from scipy import sparse
from evaluate_rp3beta_fixed_candidates_olx import build_similarity as build_rp_similarity,topk_rows
from evaluate_p3ltr_degree_recency_olx import PUBLISHED_D,PUBLISHED_R,build_similarity as build_p3_similarity

def top10(scores: sparse.csr_matrix, seen: sparse.csr_matrix, users: np.ndarray, k=10):
 scores.sort_indices();out=np.full((len(users),k),-1,np.int32)
 for row,u in enumerate(users):
  lo,hi=scores.indptr[row],scores.indptr[row+1];idx=scores.indices[lo:hi];val=scores.data[lo:hi]
  slo,shi=seen.indptr[u],seen.indptr[u+1];seen_idx=seen.indices[slo:shi]
  keep=~np.isin(idx,seen_idx,assume_unique=True);idx=idx[keep];val=val[keep]
  if len(val):
   n=min(k,len(val));chosen=np.argpartition(val,-n)[-n:];chosen=chosen[np.lexsort((idx[chosen],-val[chosen]))];out[row,:n]=idx[chosen]
 return out

def evaluate(recs: np.ndarray,relevance: sparse.csr_matrix,k=10):
 sums={x:0.0 for x in ['precision','recall','ndcg','mAP','MRR','HR']};freq=np.zeros(relevance.shape[1],np.int64)
 discount=1/np.log2(np.arange(2,k+2));ideal=np.cumsum(discount)
 for row,items in enumerate(recs):
  lo,hi=relevance.indptr[row],relevance.indptr[row+1];rel=relevance.indices[lo:hi];valid=items>=0;hits=np.zeros(k,bool);hits[valid]=np.isin(items[valid],rel,assume_unique=False);nh=hits.sum();nrel=len(rel)
  sums['precision']+=nh/k;sums['recall']+=nh/nrel;sums['ndcg']+=np.dot(hits,discount)/ideal[min(k,nrel)-1]
  sums['mAP']+=np.dot(np.cumsum(hits)/np.arange(1,k+1),hits)/min(k,nrel);where=np.flatnonzero(hits);sums['MRR']+=0 if not len(where) else 1/(where[0]+1);sums['HR']+=nh>0
  np.add.at(freq,items[valid],1)
 n=len(recs);result={x:v/n for x,v in sums.items()};used=freq>0;p=freq[freq>0]/freq.sum();result.update({'catalog_coverage':used.mean(),'shannon':float(-np.dot(p,np.log(p))),'users':n});return result

def recommend_all(edge_ui,similarity,target_users,seen,relevance,batch,output_path):
 rec_parts=[]
 for st in range(0,len(target_users),batch):
  u=target_users[st:st+batch];rec_parts.append(top10((edge_ui[u]@similarity).tocsr(),seen,u))
  if st==0 or st//batch%200==0:print(f'recommend {min(st+batch,len(target_users)):,}/{len(target_users):,}',flush=True)
 recs=np.vstack(rec_parts);np.save(output_path,recs);return evaluate(recs,relevance)

def main():
 p=argparse.ArgumentParser();p.add_argument('--interactions',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--cutoff',type=int,default=1582401299);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--block-size',type=int,default=200);a=p.parse_args();started=time.time();a.output.mkdir(parents=True,exist_ok=True);out=a.output/'summary.json'
 if out.exists():raise FileExistsError(out)
 path=str(a.interactions.resolve()).replace("'","''");c=duckdb.connect();print('Loading published-protocol training features...',flush=True)
 pairs=c.execute(f"""with e as(select user_idx,item_idx,max(timestamp)last_ts from read_parquet('{path}')where timestamp<{a.cutoff} group by user_idx,item_idx),u as(select user_idx,max(last_ts)user_last from e group by user_idx)select e.*,u.user_last from e join u using(user_idx)order by e.user_idx,e.item_idx""").fetchdf()
 test=c.execute(f"""with tr as(select distinct user_idx,item_idx from read_parquet('{path}')where timestamp<{a.cutoff}),tu as(select distinct user_idx from tr),te as(select distinct x.user_idx,x.item_idx from read_parquet('{path}')x join tu using(user_idx)left join tr using(user_idx,item_idx)where x.timestamp>={a.cutoff}and tr.item_idx is null)select * from te order by user_idx,item_idx""").fetchdf()
 item_count=int(c.execute(f"select max(item_idx)+1 from read_parquet('{path}')").fetchone()[0]);c.close()
 users=np.sort(pairs.user_idx.unique());umap=pd.Series(np.arange(len(users),dtype=np.int64),index=users);pairs['u']=pairs.user_idx.map(umap).astype(np.int64);test['u']=test.user_idx.map(umap).astype(np.int64);target_global=np.sort(test.user_idx.unique());target_local=umap.loc[target_global].to_numpy(np.int64);target_row=pd.Series(np.arange(len(target_global),dtype=np.int64),index=target_global);test['row']=test.user_idx.map(target_row).astype(np.int64)
 u=pairs.u.to_numpy(np.int64);i=pairs.item_idx.to_numpy(np.int64);ones=np.ones(len(pairs),np.float32);seen=sparse.csr_matrix((ones,(u,i)),shape=(len(users),item_count));relevance=sparse.csr_matrix((np.ones(len(test),np.int8),(test.row.to_numpy(np.int64),test.item_idx.to_numpy(np.int64))),shape=(len(target_global),item_count));ud=np.asarray(seen.sum(1)).ravel();ideg=np.asarray(seen.sum(0)).ravel();rec=((pairs.user_last-pairs.last_ts).to_numpy(np.float32)/86400)
 results=[]
 print('Building frozen RP3Beta alpha=1 beta=.3 topK=200...',flush=True);pui=sparse.diags(1/np.maximum(ud,1).astype(np.float32))@seen;piu=sparse.diags(1/np.maximum(ideg,1).astype(np.float32))@seen.T.tocsr();rp=build_rp_similarity(piu,pui,ideg,.3,200,a.block_size)
 m=recommend_all(pui,rp,target_local,seen,relevance,a.batch_size,a.output/'rp3beta_recommendations.npy');m['model']='RP3Beta_frozen';results.append(m);print(json.dumps(m),flush=True);del rp,piu,pui
 print('Building published P3LTR degree-recency topK=205...',flush=True);w0=np.power(np.maximum(ideg[i],1),-PUBLISHED_D[0])*np.exp(-rec*PUBLISHED_R[0]);w1=np.power(np.maximum(ud[u],1),-PUBLISHED_D[1])*np.exp(-rec*PUBLISHED_R[1]);w2=np.power(np.maximum(ideg[i],1),-PUBLISHED_D[2])*np.exp(-rec*PUBLISHED_R[2]);ui0=sparse.csr_matrix((w0.astype(np.float32),(u,i)),shape=seen.shape);iu1=sparse.csr_matrix((w1.astype(np.float32),(i,u)),shape=(item_count,len(users)));ui2=sparse.csr_matrix((w2.astype(np.float32),(u,i)),shape=seen.shape);sim=build_p3_similarity(iu1,ui2,205,a.block_size)
 m=recommend_all(ui0,sim,target_local,seen,relevance,a.batch_size,a.output/'p3ltr_dr_recommendations.npy');m['model']='P3LTR_degree_recency';results.append(m);print(json.dumps(m),flush=True)
 pd.DataFrame(results).to_csv(a.output/'full_catalog_metrics.csv',index=False);pd.DataFrame({'row':np.arange(len(target_global)),'user_idx':target_global}).to_parquet(a.output/'target_users.parquet',index=False)
 summary={'status':'completed','protocol':'authors official 80th timestamp percentile; train users only; seen test pairs removed; unique test pairs; full catalog top10','cutoff':a.cutoff,'training_pairs':int(seen.nnz),'test_pairs':int(relevance.nnz),'test_users':len(target_global),'item_count':item_count,'models':results,'p3ltr_scope':'published stable degree-recency component; event parameters unavailable','duration_seconds':time.time()-started,'completed_at_utc':datetime.now(timezone.utc).isoformat()};out.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
