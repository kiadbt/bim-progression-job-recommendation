"""Evaluate the published stable degree/recency component of P3LTR."""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import duckdb,numpy as np,pandas as pd
from scipy import sparse
from evaluate_baselines_fixed_candidates_olx import metrics,sample_negatives
from evaluate_rp3beta_fixed_candidates_olx import topk_rows

PUBLISHED_D=(0.631,0.163,0.433)
PUBLISHED_R=(0.081,0.008,0.028)

def build_similarity(w_iu,w_ui,topk,block):
 parts=[];n=w_iu.shape[0]
 for st in range(0,n,block):
  en=min(st+block,n);z=(w_iu[st:en]@w_ui).tocsr()
  for lr,gr in enumerate(range(st,en)):
   lo,hi=z.indptr[lr],z.indptr[lr+1];hit=np.flatnonzero(z.indices[lo:hi]==gr)
   if len(hit):z.data[lo+hit[0]]=0
  z.eliminate_zeros();parts.append(topk_rows(z,topk))
  if st==0 or en==n or (st//block)%100==0:print(f'P3LTR-DR similarity {en:,}/{n:,}',flush=True)
 return sparse.vstack(parts,format='csr')

def gather(x,cand):
 x.sort_indices();out=np.zeros(cand.shape,np.float32)
 for row,c in enumerate(cand):
  lo,hi=x.indptr[row],x.indptr[row+1];idx=x.indices[lo:hi];p=np.searchsorted(idx,c);valid=p<len(idx);m=np.zeros(len(c),bool);m[valid]=idx[p[valid]]==c[valid];out[row,m]=x.data[lo:hi][p[m]]
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--sequences',required=True,type=Path);p.add_argument('--targets',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
 p.add_argument('--topk',type=int,default=205);p.add_argument('--block-size',type=int,default=200);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--evaluation-seed',type=int,default=20260827);p.add_argument('--evaluation-negatives',type=int,default=999);a=p.parse_args()
 started=time.time();a.output.mkdir(parents=True,exist_ok=True);out=a.output/'summary.json'
 if out.exists():raise FileExistsError(out)
 esc=lambda x:str(x.resolve()).replace("'","''");seq,tgt=map(esc,[a.sequences,a.targets]);c=duckdb.connect();cutoff=int(c.execute(f"select min(source_timestamp)from read_parquet('{tgt}')where split='validation'").fetchone()[0])
 ev=c.execute(f"""with r as(select *,row_number()over(partition by split,user_idx order by source_timestamp,attempt_id)rn from read_parquet('{tgt}'))select * from r where rn=1 order by split,user_idx""").fetchdf()
 print('Aggregating pre-cutoff P3LTR edge features...',flush=True)
 pairs=c.execute(f"""with e as(select user_idx,item_idx,max(timestamp)last_ts from read_parquet('{seq}')where timestamp<{cutoff} group by user_idx,item_idx),
 u as(select user_idx,max(last_ts)user_last from e group by user_idx)
 select e.user_idx,e.item_idx,e.last_ts,u.user_last from e join u using(user_idx) order by e.user_idx,e.item_idx""").fetchdf()
 item_count=int(c.execute(f"select max(item_idx)+1 from read_parquet('{seq}')").fetchone()[0]);c.close()
 users=np.sort(pairs.user_idx.unique());umap=pd.Series(np.arange(len(users),dtype=np.int64),index=users);pairs['u']=pairs.user_idx.map(umap).astype(np.int64);ev['u']=ev.user_idx.map(umap);cold=int(ev.u.isna().sum());ev=ev.dropna(subset=['u']).copy();ev['u']=ev.u.astype(np.int64)
 u=pairs.u.to_numpy(np.int64);i=pairs.item_idx.to_numpy(np.int64);ud=np.bincount(u,minlength=len(users)).astype(np.float32);ideg=np.bincount(i,minlength=item_count).astype(np.float32);rec=((pairs.user_last-pairs.last_ts).to_numpy(np.float32)/86400)
 # Published feature encoder with unreported/nonconverged event terms omitted; their constant sigmoid factors cancel ranking.
 w0=np.power(np.maximum(ideg[i],1),-PUBLISHED_D[0])*np.exp(-rec*PUBLISHED_R[0]);w1=np.power(np.maximum(ud[u],1),-PUBLISHED_D[1])*np.exp(-rec*PUBLISHED_R[1]);w2=np.power(np.maximum(ideg[i],1),-PUBLISHED_D[2])*np.exp(-rec*PUBLISHED_R[2])
 ui0=sparse.csr_matrix((w0.astype(np.float32),(u,i)),shape=(len(users),item_count));iu1=sparse.csr_matrix((w1.astype(np.float32),(i,u)),shape=(item_count,len(users)));ui2=sparse.csr_matrix((w2.astype(np.float32),(u,i)),shape=(len(users),item_count));seen=set((u*item_count+i).tolist());del pairs,u,i,w0,w1,w2,rec
 sim=build_similarity(iu1,ui2,a.topk,a.block_size);rng=np.random.default_rng(a.evaluation_seed);rows=[];paired=[]
 for split,g in ev.groupby('split',sort=True):
  ua=g.u.to_numpy(np.int64);pa=g.target_item_idx.to_numpy(np.int64);parts=[]
  for st in range(0,len(g),a.batch_size):
   uu=ua[st:st+a.batch_size];pos=pa[st:st+a.batch_size];cand=np.empty((len(uu),a.evaluation_negatives+1),np.int64);cand[:,0]=pos
   for j in range(1,cand.shape[1]):cand[:,j]=sample_negatives(uu,pos,item_count,seen,rng)
   score=gather((ui0[uu]@sim).tocsr(),cand);parts.append(1+(score[:,1:]>=score[:,[0]]).sum(1))
  rank=np.concatenate(parts);row={'split':str(split),'cases':len(rank)}
  for k in (5,10,20,100):row.update(metrics(rank,k))
  rows.append(row);paired.append(pd.DataFrame({'attempt_id':g.attempt_id.to_numpy(),'split':str(split),'p3ltr_dr_rank':rank}));print(json.dumps(row),flush=True)
 pd.DataFrame(rows).to_csv(a.output/'metrics.csv',index=False);pd.concat(paired,ignore_index=True).to_parquet(a.output/'paired_ranks.parquet',index=False)
 summary={'experiment':'P3LTR degree-recency published-parameter reproduction','status':'completed','scope':'stable published degree and recency encoders only; event parameters unavailable and reported nonconverged','published_parameters':{'d':PUBLISHED_D,'r':PUBLISHED_R,'topk':a.topk},'training_cutoff':cutoff,'training_pairs':int(ui0.nnz),'cold_cases_excluded':cold,'evaluation':rows,'test_selection':'external published parameters; no local test tuning','duration_seconds':time.time()-started,'completed_at_utc':datetime.now(timezone.utc).isoformat()};out.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
