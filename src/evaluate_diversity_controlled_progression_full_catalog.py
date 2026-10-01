"""Diversity-constrained progression fusion over frozen full-catalog RP3Beta."""
from __future__ import annotations
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from evaluate_deployable_progression_full_catalog_olx import load_split,prepare,build
from evaluate_full_catalog_published_protocol_olx import evaluate

def rank_batch(scores,seen,users,progression,popularity,configs):
 scores.sort_indices();aux={cfg:cfg[0]*progression-cfg[1]*popularity for cfg in configs};orders={cfg:np.lexsort((np.arange(len(progression)),-v)) for cfg,v in aux.items()};out={cfg:np.full((len(users),10),-1,np.int32) for cfg in configs}
 for row,u in enumerate(users):
  lo,hi=scores.indptr[row],scores.indptr[row+1];idx=scores.indices[lo:hi];base=scores.data[lo:hi].astype(float);slo,shi=seen.indptr[u],seen.indptr[u+1];seenidx=seen.indices[slo:shi];keep=~np.isin(idx,seenidx,assume_unique=True);idx=idx[keep];base=base[keep]
  if len(base):base/=base.max()
  existing=set(idx.tolist())
  for cfg in configs:
   extra=[]
   for item in orders[cfg]:
    pos=np.searchsorted(seenidx,item)
    if item not in existing and (pos==len(seenidx) or seenidx[pos]!=item):extra.append(int(item))
    if len(extra)==10:break
   ids=np.concatenate([idx,np.asarray(extra,np.int32)]);vals=np.concatenate([base,np.zeros(len(extra))])+aux[cfg][ids];n=min(10,len(vals));ch=np.argpartition(vals,-n)[-n:];ch=ch[np.lexsort((ids[ch],-vals[ch]))];out[cfg][row,:n]=ids[ch]
 return out

def recommend(ui,sim,targets,progression,popularity,configs,batch):
 parts={cfg:[] for cfg in configs}
 for st in range(0,len(targets),batch):
  u=targets[st:st+batch];z=rank_batch((ui[u]@sim).tocsr(),ui,u,progression,popularity,configs)
  for cfg in configs:parts[cfg].append(z[cfg])
  if st==0 or st//batch%200==0:print(f'diversity fusion {min(st+batch,len(targets)):,}/{len(targets):,}',flush=True)
 return {cfg:np.vstack(v) for cfg,v in parts.items()}

def vectors(prog,ui,item_count):
 p=np.zeros(item_count,np.float32);p[prog.item_idx.to_numpy(np.int64)]=np.log1p(prog.n.to_numpy(np.float32));p/=max(float(p.max()),1);degree=np.asarray(ui.sum(0)).ravel().astype(np.float32);degree=np.log1p(degree);degree/=max(float(degree.max()),1);return p,degree

def main():
 p=argparse.ArgumentParser();p.add_argument('--interactions',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--validation-cutoff',type=int,default=1582239167);p.add_argument('--test-cutoff',type=int,default=1582401299);p.add_argument('--gammas',nargs='+',type=float,default=[0,.1,.25,.5]);p.add_argument('--lambdas',nargs='+',type=float,default=[0,.025,.05,.1,.2]);p.add_argument('--minimum-coverage-ratio',type=float,default=.95);p.add_argument('--validation-users',type=int,default=30000);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--block-size',type=int,default=200);a=p.parse_args();started=time.time();a.output.mkdir(parents=True,exist_ok=True);out=a.output/'summary.json'
 if out.exists():raise FileExistsError(out)
 path=str(a.interactions.resolve()).replace("'","''");configs=[(g,l) for g in a.gammas for l in a.lambdas];print('Preparing diversity-constrained validation...',flush=True);pairs,val,prog,item_count=load_split(path,a.validation_cutoff,a.test_cutoff);allusers=np.sort(val.user_idx.unique());rng=np.random.default_rng(10);selected=np.sort(rng.choice(allusers,min(a.validation_users,len(allusers)),replace=False));ui,rel,targets,_=prepare(pairs,val,item_count,selected);pv,pop=vectors(prog,ui,item_count);pui,sim=build(ui,item_count,a.block_size);recs=recommend(pui,sim,targets,pv,pop,configs,a.batch_size);rows=[]
 for (g,l),r in recs.items():m=evaluate(r,rel);m.update({'gamma':g,'lambda_popularity':l});rows.append(m)
 frame=pd.DataFrame(rows);baseline=float(frame[(frame.gamma==0)&(frame.lambda_popularity==0)].catalog_coverage.iloc[0]);threshold=a.minimum_coverage_ratio*baseline;frame['coverage_feasible']=frame.catalog_coverage>=threshold;feasible=frame[frame.coverage_feasible].sort_values(['ndcg','recall','gamma','lambda_popularity'],ascending=[False,False,True,True]);
 if feasible.empty:raise RuntimeError('no coverage-feasible configuration')
 selected_gamma=float(feasible.iloc[0].gamma);selected_lambda=float(feasible.iloc[0].lambda_popularity);frame.to_csv(a.output/'validation_grid.csv',index=False);print(f'selected gamma={selected_gamma} lambda={selected_lambda} coverage threshold={threshold}',flush=True);del pairs,val,prog,ui,rel,pui,sim,recs
 pairs,test,prog,item_count=load_split(path,a.test_cutoff,None);ui,rel,targets,_=prepare(pairs,test,item_count);pv,pop=vectors(prog,ui,item_count);pui,sim=build(ui,item_count,a.block_size);test_configs=[(0.,0.),(selected_gamma,selected_lambda)];final=recommend(pui,sim,targets,pv,pop,test_configs,a.batch_size);testrows=[]
 for cfg,r in final.items():m=evaluate(r,rel);m.update({'gamma':cfg[0],'lambda_popularity':cfg[1],'model':'RP3Beta' if cfg==(0.,0.) else 'diversity_controlled_progression_RP3Beta'});testrows.append(m);np.save(a.output/f'recommendations_g{cfg[0]:g}_l{cfg[1]:g}.npy',r)
 pd.DataFrame(testrows).to_csv(a.output/'test_metrics.csv',index=False);summary={'status':'completed','selection':'maximum validation NDCG subject to catalog coverage >= 95% of validation RP3Beta','validation_cutoff':a.validation_cutoff,'test_cutoff':a.test_cutoff,'validation_users':len(selected),'baseline_validation_coverage':baseline,'coverage_threshold':threshold,'selected_gamma':selected_gamma,'selected_lambda_popularity':selected_lambda,'selected_validation':frame[(frame.gamma==selected_gamma)&(frame.lambda_popularity==selected_lambda)].to_dict('records'),'test':testrows,'duration_seconds':time.time()-started,'completed_at_utc':datetime.now(timezone.utc).isoformat()};out.write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
