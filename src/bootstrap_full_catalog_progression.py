"""Paired user bootstrap for deployable progression fusion versus RP3Beta."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import duckdb,numpy as np,pandas as pd
from scipy import sparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--interactions',required=True,type=Path);p.add_argument('--targets',required=True,type=Path);p.add_argument('--base',required=True,type=Path);p.add_argument('--model',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--cutoff',type=int,default=1582401299);p.add_argument('--replicates',type=int,default=1000);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 users=pd.read_parquet(a.targets).sort_values('row');umap=pd.Series(users.row.to_numpy(np.int64),index=users.user_idx);path=str(a.interactions.resolve()).replace("'","''");c=duckdb.connect();te=c.execute(f"""with tr as(select distinct user_idx,item_idx from read_parquet('{path}')where timestamp<{a.cutoff}),tu as(select distinct user_idx from tr)select distinct x.user_idx,x.item_idx from read_parquet('{path}')x join tu using(user_idx)left join tr using(user_idx,item_idx)where x.timestamp>={a.cutoff}and tr.item_idx is null""").fetchdf();c.close();te['row']=te.user_idx.map(umap).astype(np.int64);rel=sparse.csr_matrix((np.ones(len(te),np.int8),(te.row.to_numpy(),te.item_idx.to_numpy())),shape=(len(users),185395));base=np.load(a.base);model=np.load(a.model);discount=1/np.log2(np.arange(2,12));ideal=np.cumsum(discount);contrib={}
 for name,recs in [('base',base),('model',model)]:
  vals={m:np.zeros(len(users),float) for m in ['recall','ndcg','mrr','hit_rate']}
  for row,items in enumerate(recs):
   lo,hi=rel.indptr[row],rel.indptr[row+1];hits=np.isin(items,rel.indices[lo:hi]);n=hi-lo;where=np.flatnonzero(hits);vals['recall'][row]=hits.sum()/n;vals['ndcg'][row]=np.dot(hits,discount)/ideal[min(10,n)-1];vals['mrr'][row]=0 if not len(where)else 1/(where[0]+1);vals['hit_rate'][row]=len(where)>0
  contrib[name]=vals
 rng=np.random.default_rng(20260901);rows=[]
 for metric in contrib['base']:
  d=contrib['model'][metric]-contrib['base'][metric];boot=np.empty(a.replicates)
  for st in range(0,a.replicates,10):n=min(10,a.replicates-st);ix=rng.integers(0,len(d),(n,len(d)));boot[st:st+n]=d[ix].mean(1)
  lo,hi=np.quantile(boot,[.025,.975]);rows.append({'metric':metric+'_at_10','progression_minus_rp3beta':d.mean(),'ci95_lower':lo,'ci95_upper':hi,'p_improvement_le_zero':(np.count_nonzero(boot<=0)+1)/(a.replicates+1),'users':len(d),'replicates':a.replicates})
 frame=pd.DataFrame(rows);frame.to_csv(a.output/'paired_bootstrap.csv',index=False);(a.output/'summary.json').write_text(json.dumps({'results':frame.to_dict('records')},indent=2),encoding='utf-8');print(frame.to_string(index=False))
if __name__=='__main__':main()
