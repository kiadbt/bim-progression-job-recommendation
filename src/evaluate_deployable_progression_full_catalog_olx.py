"""Leakage-free full-catalog progression fusion over frozen RP3Beta."""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import duckdb,numpy as np,pandas as pd
from scipy import sparse
from evaluate_rp3beta_fixed_candidates_olx import build_similarity
from evaluate_full_catalog_published_protocol_olx import evaluate

def load_split(path,cutoff,end=None):
 c=duckdb.connect();end_clause='' if end is None else f' and x.timestamp<{end}'
 pairs=c.execute(f"select distinct user_idx,item_idx from read_parquet('{path}')where timestamp<{cutoff} order by user_idx,item_idx").fetchdf()
 test=c.execute(f"""with tr as(select distinct user_idx,item_idx from read_parquet('{path}')where timestamp<{cutoff}),tu as(select distinct user_idx from tr),te as(select distinct x.user_idx,x.item_idx from read_parquet('{path}')x join tu using(user_idx)left join tr using(user_idx,item_idx)where x.timestamp>={cutoff}{end_clause}and tr.item_idx is null)select * from te order by user_idx,item_idx""").fetchdf()
 prog=c.execute(f"select item_idx,count(*) n from read_parquet('{path}')where timestamp<{cutoff}and event in('contact_chat','contact_phone_click_2','contact_phone_click_3')group by item_idx").fetchdf();item_count=int(c.execute(f"select max(item_idx)+1 from read_parquet('{path}')").fetchone()[0]);c.close();return pairs,test,prog,item_count

def prepare(pairs,test,item_count,target_globals=None):
 users=np.sort(pairs.user_idx.unique());umap=pd.Series(np.arange(len(users),dtype=np.int64),index=users);pairs['u']=pairs.user_idx.map(umap).astype(np.int64);test=test[test.user_idx.isin(umap.index)].copy()
 globals_=np.sort(test.user_idx.unique()) if target_globals is None else np.asarray(target_globals);test=test[test.user_idx.isin(globals_)].copy();local=umap.loc[globals_].to_numpy(np.int64);rowmap=pd.Series(np.arange(len(globals_),dtype=np.int64),index=globals_);test['row']=test.user_idx.map(rowmap).astype(np.int64)
 u=pairs.u.to_numpy(np.int64);i=pairs.item_idx.to_numpy(np.int64);ui=sparse.csr_matrix((np.ones(len(pairs),np.float32),(u,i)),shape=(len(users),item_count));rel=sparse.csr_matrix((np.ones(len(test),np.int8),(test.row.to_numpy(np.int64),test.item_idx.to_numpy(np.int64))),shape=(len(globals_),item_count));return ui,rel,local,globals_

def rank_batch(scores,seen,users,prog,gammas):
 scores.sort_indices();order=np.lexsort((np.arange(len(prog)),-prog));outputs={g:np.full((len(users),10),-1,np.int32) for g in gammas}
 for row,u in enumerate(users):
  lo,hi=scores.indptr[row],scores.indptr[row+1];idx=scores.indices[lo:hi];base=scores.data[lo:hi].astype(np.float64);slo,shi=seen.indptr[u],seen.indptr[u+1];seenidx=seen.indices[slo:shi];keep=~np.isin(idx,seenidx,assume_unique=True);idx=idx[keep];base=base[keep]
  if len(base):base/=base.max()
  existing=set(idx.tolist());extra=[]
  for item in order:
   if item not in existing and np.searchsorted(seenidx,item)>=len(seenidx) or (item not in existing and seenidx[np.searchsorted(seenidx,item)]!=item):
    extra.append(int(item))
    if len(extra)==10:break
  allidx=np.concatenate([idx,np.asarray(extra,np.int32)]);allbase=np.concatenate([base,np.zeros(len(extra))])
  for g in gammas:
   val=allbase+g*prog[allidx];n=min(10,len(val));ch=np.argpartition(val,-n)[-n:];ch=ch[np.lexsort((allidx[ch],-val[ch]))];outputs[g][row,:n]=allidx[ch]
 return outputs

def recommend(ui,sim,targets,prog,gammas,batch):
 parts={g:[] for g in gammas}
 for st in range(0,len(targets),batch):
  u=targets[st:st+batch];z=rank_batch((ui[u]@sim).tocsr(),ui,u,prog,gammas)
  for g in gammas:parts[g].append(z[g])
  if st==0 or st//batch%200==0:print(f'fusion recommend {min(st+batch,len(targets)):,}/{len(targets):,}',flush=True)
 return {g:np.vstack(v) for g,v in parts.items()}

def build(ui,item_count,block):
 ud=np.asarray(ui.sum(1)).ravel();ideg=np.asarray(ui.sum(0)).ravel();pui=sparse.diags(1/np.maximum(ud,1).astype(np.float32))@ui;piu=sparse.diags(1/np.maximum(ideg,1).astype(np.float32))@ui.T.tocsr();return pui,build_similarity(piu,pui,ideg,.3,200,block)

def main():
 p=argparse.ArgumentParser();p.add_argument('--interactions',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--validation-cutoff',type=int,default=1582239167);p.add_argument('--test-cutoff',type=int,default=1582401299);p.add_argument('--gammas',nargs='+',type=float,default=[0,.01,.025,.05,.1,.25,.5,1]);p.add_argument('--validation-users',type=int,default=30000);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--block-size',type=int,default=200);a=p.parse_args();started=time.time();a.output.mkdir(parents=True,exist_ok=True);out=a.output/'summary.json'
 if out.exists():raise FileExistsError(out)
 path=str(a.interactions.resolve()).replace("'","''");print('Preparing internal chronological validation...',flush=True);pairs,val,prog,item_count=load_split(path,a.validation_cutoff,a.test_cutoff);allusers=np.sort(val.user_idx.unique());rng=np.random.default_rng(10);selected=np.sort(rng.choice(allusers,min(a.validation_users,len(allusers)),replace=False));ui,rel,targets,_=prepare(pairs,val,item_count,selected);pv=np.zeros(item_count,np.float32);pv[prog.item_idx.to_numpy(np.int64)]=np.log1p(prog.n.to_numpy(np.float32));pv/=max(float(pv.max()),1);pui,sim=build(ui,item_count,a.block_size);recs=recommend(pui,sim,targets,pv,a.gammas,a.batch_size);rows=[]
 for g,r in recs.items():m=evaluate(r,rel);m['gamma']=g;rows.append(m)
 sweep=pd.DataFrame(rows).sort_values(['ndcg','recall','gamma'],ascending=[False,False,True]);sweep.to_csv(a.output/'validation_gamma_sweep.csv',index=False);gamma=float(sweep.iloc[0].gamma);print('selected gamma',gamma,flush=True);del pairs,val,prog,ui,rel,pui,sim,recs
 print('Rebuilding complete published-protocol graph...',flush=True);pairs,test,prog,item_count=load_split(path,a.test_cutoff,None);ui,rel,targets,globals_=prepare(pairs,test,item_count);pv=np.zeros(item_count,np.float32);pv[prog.item_idx.to_numpy(np.int64)]=np.log1p(prog.n.to_numpy(np.float32));pv/=max(float(pv.max()),1);pui,sim=build(ui,item_count,a.block_size);final=recommend(pui,sim,targets,pv,[0,gamma],a.batch_size);testrows=[]
 for g,r in final.items():m=evaluate(r,rel);m['gamma']=g;m['model']='RP3Beta' if g==0 else 'deployable_progression_RP3Beta';testrows.append(m);np.save(a.output/f'recommendations_gamma_{g:g}.npy',r)
 pd.DataFrame(testrows).to_csv(a.output/'full_catalog_test_metrics.csv',index=False);summary={'status':'completed','context_policy':'global item successful-contact count strictly before cutoff; no target pathway','validation_cutoff':a.validation_cutoff,'test_cutoff':a.test_cutoff,'validation_users':len(selected),'gamma_grid':a.gammas,'selected_gamma':gamma,'validation_selected':sweep[sweep.gamma==gamma].to_dict('records'),'test_users':len(targets),'test_pairs':int(rel.nnz),'test':testrows,'duration_seconds':time.time()-started,'completed_at_utc':datetime.now(timezone.utc).isoformat()};out.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
