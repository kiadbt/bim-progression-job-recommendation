"""Bounded cart-to-purchase validation with a gated held-out phase."""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
from datetime import datetime, timezone
import sys, json, time, warnings, argparse, pickle
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from sklearn.exceptions import ConvergenceWarning

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'local_work/cosmetics'
DATA=ROOT/'local_data/cosmetics'
SRC=ROOT/'experiments/olx/src'
sys.path.insert(0,str(SRC))
from comparison_adapter import build_rp3beta
from protocol_primitives import sampled_candidates,stable_topk_rows
from paired_bootstrap import paired_summary
DAY=86400; SEED=20261024
FEATURES=['log_user_events','log_item_events','log_pair_events','log_user_views','log_user_carts',
 'log_user_purchases','log_user_removals','log_user_recency','recency_missing','hour_sin','hour_cos',
 'log_mature_user_success','log_mature_user_nonprogression']
P3_D=(.631,.163,.433); P3_R=(.081,.008,.028)
def ts(s): return int(pd.Timestamp(s,tz='UTC').timestamp())
def digest(p):
    h=sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def log(s):
    line=datetime.now(timezone.utc).isoformat()+' '+str(s)
    print(line,flush=True)
    with (OUT/'execution.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
def signature(): return {'protocol':digest(OUT/'protocol.json'),'driver':digest(__file__),
 'rp3_adapter':digest(SRC/'comparison_adapter.py'),'primitives':digest(SRC/'protocol_primitives.py')}
def load_events(phase):
    frames=[]; hashes={}
    for month in ['Oct','Nov']+(['Dec'] if phase=='test' else []):
        name=f'2019-{month}.csv'; cache=OUT/f'events_{month}.parquet'
        archive=DATA/(name+'.zip')
        hashes[name]=digest(archive)
        if cache.exists(): d=pd.read_parquet(cache)
        else:
            with ZipFile(archive) as z:
                assert z.namelist()==[name] and z.testzip() is None
                with z.open(name) as f:
                    d=pd.read_csv(f,usecols=['event_time','event_type','product_id','user_id'],dtype={'event_type':'category','product_id':'int64','user_id':'int64'})
            assert not d.isna().any().any()
            d['t']=pd.to_datetime(d.pop('event_time'),utc=True,format='%Y-%m-%d %H:%M:%S UTC').dt.as_unit('ns').astype('int64')//10**9
            m={'Oct':10,'Nov':11,'Dec':12}[month]
            assert d.t.min()>=ts(f'2019-{m:02d}-01') and d.t.max()<ts('2020-01-01' if m==12 else f'2019-{m+1:02d}-01')
            d=d.drop_duplicates(['t','event_type','product_id','user_id'])
            d.to_parquet(cache,index=False)
        frames.append(d); log(f'Loaded {name}: {len(d):,} event keys')
    d=pd.concat(frames,ignore_index=True).sort_values(['t','user_id','product_id','event_type'],kind='stable').reset_index(drop=True)
    assert d.t.is_monotonic_increasing
    return d,hashes
def feature_frame(d,phase):
    cache=OUT/f'cart_features_{phase}.parquet'
    if cache.exists(): return pd.read_parquet(cache)
    c=d.loc[d.event_type=='cart',['t','user_id','product_id']].copy()
    p=d.loc[d.event_type=='purchase',['t','user_id','product_id']].rename(columns={'t':'purchase_t'})
    c=pd.merge_asof(c,p,left_on='t',right_on='purchase_t',by=['user_id','product_id'],direction='forward',allow_exact_matches=False)
    c['y']=(~(c.purchase_t.notna() & (c.purchase_t<=c.t+DAY))).astype('int8')
    c['case_id']=np.arange(len(c),dtype='int64')
    for key,name in [(['user_id'],'log_user_events'),(['product_id'],'log_item_events'),(['user_id','product_id'],'log_pair_events')]:
        g=d.groupby(key+['t'],observed=True).size().rename('n').reset_index()
        g[name]=np.log1p(g.groupby(key,observed=True).n.cumsum()-g.n).astype('float32')
        c=c.merge(g[key+['t',name]],on=key+['t'],validate='many_to_one')
    g=d.groupby(['user_id','t','event_type'],observed=True).size().unstack(fill_value=0)
    g=g.reindex(columns=['view','cart','purchase','remove_from_cart'],fill_value=0)
    g=g.groupby(level=0).cumsum()-g
    g=np.log1p(g).astype('float32').rename(columns={'view':'log_user_views','cart':'log_user_carts',
       'purchase':'log_user_purchases','remove_from_cart':'log_user_removals'}).reset_index()
    c=c.merge(g,on=['user_id','t'],validate='many_to_one')
    times=d[['user_id','t']].drop_duplicates().sort_values(['user_id','t'])
    times['prev']=times.groupby('user_id').t.shift()
    times['recency_missing']=times.prev.isna().astype('float32')
    times['log_user_recency']=np.log1p((times.t-times.prev).fillna(30*DAY).clip(upper=30*DAY)).astype('float32')
    c=c.merge(times[['user_id','t','recency_missing','log_user_recency']],on=['user_id','t'],validate='many_to_one')
    hour=(c.t%DAY)/3600
    c['hour_sin']=np.sin(2*np.pi*hour/24).astype('float32');c['hour_cos']=np.cos(2*np.pi*hour/24).astype('float32')
    mat=c[['user_id','t','y']].copy();mat['available']=mat.t+DAY
    mat['success']=1-mat.y;mat['nonprogress']=mat.y
    mat=mat.groupby(['user_id','available'],observed=True)[['success','nonprogress']].sum().reset_index().sort_values(['user_id','available'])
    mat[['success','nonprogress']]=mat.groupby('user_id')[['success','nonprogress']].cumsum()
    c=pd.merge_asof(c.sort_values('t'),mat.sort_values('available'),left_on='t',right_on='available',by='user_id',direction='backward',allow_exact_matches=False)
    assert (c.available.dropna()<c.loc[c.available.notna(),'t']).all()
    c['log_mature_user_success']=np.log1p(c.success.fillna(0)).astype('float32')
    c['log_mature_user_nonprogression']=np.log1p(c.nonprogress.fillna(0)).astype('float32')
    c=c[['case_id','user_id','product_id','t','purchase_t','y']+FEATURES].sort_values(['t','user_id','product_id'],kind='stable').reset_index(drop=True)
    assert np.isfinite(c[FEATURES].to_numpy()).all()
    c.to_parquet(cache,index=False); log(f'Constructed {len(c):,} strictly pre-source feature rows')
    return c
def cap_users(c,maximum):
    c=c.copy(); c['hash_order']=[sha256(f'{SEED}:{int(u)}'.encode()).hexdigest() for u in c.user_id]
    return c.sort_values(['hash_order','user_id']).head(maximum).drop(columns='hash_order').sort_values('user_id').reset_index(drop=True)
def cases(d,c,start,end,cutoff,maximum):
    positive=d[d.event_type.isin(['view','cart','purchase']) & (d.t<cutoff)]
    warm=set(positive.user_id.unique());catalog=np.sort(positive.product_id.unique())
    first=d.groupby(['user_id','product_id'],observed=True).t.min().rename('first_seen').reset_index()
    buys=d[d.event_type=='purchase'][['user_id','product_id','t']].rename(columns={'t':'tt'}).merge(first,on=['user_id','product_id'],validate='many_to_one')
    buys=buys[(buys.tt>start)&(buys.tt<end)&buys.user_id.isin(warm)&buys.product_id.isin(catalog)].sort_values(['user_id','tt','product_id'])
    sources=c[(c.t>=start)&(c.t+DAY<end)&c.user_id.isin(warm)]
    target_users={int(u):g[['tt','product_id','first_seen']].to_numpy(dtype='int64') for u,g in buys.groupby('user_id')}
    records=[]
    for u,g in sources.groupby('user_id',sort=True):
        b=target_users.get(int(u))
        if b is None: continue
        found=False
        for row in g.itertuples():
            a=np.searchsorted(b[:,0],row.t,side='right');z=np.searchsorted(b[:,0],row.t+DAY,side='right')
            for tt,j,fs in b[a:z]:
                if j!=row.product_id and fs>row.t:
                    records.append((row.case_id,int(j),int(tt)));found=True;break
            if found:break
    records=pd.DataFrame(records,columns=['case_id','target','target_t'])
    result=sources.merge(records,on='case_id',validate='one_to_one')
    count=len(result);result=cap_users(result,maximum)
    assert result.user_id.is_unique
    assert ((result.target_t>result.t)&(result.target_t<=result.t+DAY)&(result.t+DAY<end)).all()
    assert (result.product_id!=result.target).all()
    return result,catalog,{'eligible_distinct_users_before_cap':count,'selected_users':len(result),'cutoff':cutoff,'query_start':start,'query_end':end}
def fit_calibration(raw,y):
    x=logit(np.clip(raw,1e-7,1-1e-7));y=np.asarray(y,float)
    def obj(v):
        z=v[0]*x+v[1];r=expit(z)-y
        return np.mean(np.logaddexp(0,z)-y*z),np.array([np.mean(r*x),np.mean(r)])
    fit=minimize(obj,[1.,float(logit(y.mean())-logit(raw.mean()))],jac=True,method='L-BFGS-B',bounds=[(1e-8,None),(None,None)])
    assert fit.success and np.isfinite(fit.x).all()
    return {'slope':float(fit.x[0]),'intercept':float(fit.x[1])}
def calibrated(raw,p): return expit(p['slope']*logit(np.clip(raw,1e-7,1-1e-7))+p['intercept'])
def fit_models(d,c):
    train=c[(c.t>=ts('2019-10-01'))&(c.t<ts('2019-10-20'))&(c.t+DAY<ts('2019-10-21'))].drop_duplicates('user_id',keep='last')
    train=cap_users(train,50000)
    calibration,_,meta=cases(d,c,ts('2019-10-21'),ts('2019-10-27'),ts('2019-10-21'),10000)
    assert len(calibration)>=200 and calibration.y.value_counts().min()>=10
    bundle={};record={'training_users':len(train),'calibration':meta,'calibration_class_counts':calibration.y.value_counts().to_dict(),
      'training_users_with_mature_history':int(((train.log_mature_user_success+train.log_mature_user_nonprogression)>0).sum()),'models':{}}
    for name,n in [('actual',len(FEATURES)),('context',len(FEATURES)-2)]:
        scaler=StandardScaler();x=scaler.fit_transform(train[FEATURES[:n]].to_numpy(np.float32))
        clf=MLPClassifier(hidden_layer_sizes=(32,16),activation='relu',solver='adam',learning_rate_init=.001,
          alpha=.0001,batch_size=512,max_iter=8,early_stopping=False,random_state=SEED)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always',ConvergenceWarning);clf.fit(x,train.y)
        raw=clf.predict_proba(scaler.transform(calibration[FEATURES[:n]].to_numpy(np.float32)))[:,1]
        p=fit_calibration(raw,calibration.y)
        bundle[name]=(scaler,clf,p,n)
        record['models'][name]={'iterations':int(clf.n_iter_),'training_loss':float(clf.loss_),
          'warnings':[str(w.message) for w in ws],'calibration_parameters':p,'feature_names':FEATURES[:n]}
    bundle['global_risk']=float(calibration.y.mean())
    with (OUT/'models.pkl').open('wb') as f:pickle.dump(bundle,f)
    write(OUT/'model_fit.json',record); calibration[['case_id','user_id','t','y','target','target_t']].to_parquet(OUT/'calibration_cases.parquet',index=False)
    log(f'Fit two fixed-budget MLPs using {len(train):,} users; calibrated on {len(calibration):,} conditional cases')
    return bundle
def model_risks(bundle,c):
    r={}
    for name in ['actual','context']:
        scaler,clf,p,n=bundle[name]
        r[name]=calibrated(clf.predict_proba(scaler.transform(c[FEATURES[:n]].to_numpy(np.float32)))[:,1],p)
    r['fixed_global']=np.full(len(c),bundle['global_risk'])
    r['shuffled']=np.random.default_rng(SEED).permutation(r['actual'])
    r['success_only']=np.zeros(len(c))
    return r
def graph_inputs(d,c,cutoff,catalog):
    pre=d[(d.t<cutoff)&d.event_type.isin(['view','cart','purchase'])]
    pairs=pre.groupby(['user_id','product_id'],observed=True).t.max().reset_index()
    original=len(pairs)
    pairs=pairs.sort_values(['user_id','t','product_id'],ascending=[True,False,True]).groupby('user_id',sort=False).head(100)
    users=np.sort(pairs.user_id.unique());pairs['u']=np.searchsorted(users,pairs.user_id);pairs['i']=np.searchsorted(catalog,pairs.product_id)
    history=sparse.csr_matrix((np.ones(len(pairs)),(pairs.u,pairs.i)),shape=(len(users),len(catalog)))
    counts=pre.groupby('product_id').size().reindex(catalog,fill_value=0).to_numpy()
    mature=c[c.t+DAY<cutoff]
    prof=mature.groupby('product_id').agg(n=('y','size'),fail=('y','sum')).reindex(catalog,fill_value=0)
    successes=(prof.n-prof.fail).to_numpy();failures=prof.fail.to_numpy()
    return users,pairs,history,np.log1p(counts),np.log1p(successes),np.log1p(failures),{
       'uncapped_distinct_pairs':original,'capped_distinct_pairs':len(pairs),'graph_users':len(users),
       'catalog_items':len(catalog),'mature_cart_sources':len(mature),'latest_mature_source_plus_24h':int(mature.t.max()+DAY)}
def make_candidates(d,c,catalog):
    relevant=d[d.user_id.isin(c.user_id)].sort_values('t')
    histories={int(u):g[['t','product_id']].to_numpy(dtype='int64') for u,g in relevant.groupby('user_id')}
    out=[]
    for row in c.itertuples():
        a=histories[int(row.user_id)];seen=a[:np.searchsorted(a[:,0],row.t,side='right'),1]
        cand=sampled_candidates(catalog,seen,row.target,999,SEED,int(row.user_id))
        assert len(np.unique(cand))==1000 and not np.isin(cand,seen).any()
        out.append(cand)
    return np.asarray(out,dtype='int64')
def p3(history,pairs,block=64):
    ud=np.asarray(history.sum(1)).ravel();degree=np.asarray(history.sum(0)).ravel()
    age=(pairs.groupby('user_id').t.transform('max').to_numpy()-pairs.t.to_numpy())/DAY
    u=pairs.u.to_numpy();i=pairs.i.to_numpy()
    a=sparse.csr_matrix((np.maximum(degree[i],1)**-P3_D[0]*np.exp(-age*P3_R[0]),(u,i)),shape=history.shape)
    b=sparse.csr_matrix((np.maximum(ud[u],1)**-P3_D[1]*np.exp(-age*P3_R[1]),(i,u)),shape=history.T.shape)
    z=sparse.csr_matrix((np.maximum(degree[i],1)**-P3_D[2]*np.exp(-age*P3_R[2]),(u,i)),shape=history.shape)
    parts=[]
    for start in range(0,history.shape[1],block):
        g=(b[start:start+block]@z).tocsr()
        for r in range(g.shape[0]):
            lo,hi=g.indptr[r:r+2];g.data[lo:hi][g.indices[lo:hi]==start+r]=0
        g.eliminate_zeros();parts.append(stable_topk_rows(g,205))
    return a,sparse.vstack(parts,format='csr')
def gather(edge,sim,rows,ci):
    result=np.zeros(ci.shape)
    for start in range(0,len(rows),64):
        m=(edge[rows[start:start+64]]@sim).tocsr();m.sort_indices()
        for r,items in enumerate(ci[start:start+64]):
            lo,hi=m.indptr[r:r+2];ids=m.indices[lo:hi];p=np.searchsorted(ids,items)
            ok=p<len(ids);matched=np.zeros(len(items),bool);matched[ok]=ids[p[ok]]==items[ok]
            result[start+r,matched]=m.data[lo:hi][p[matched]]
    return result
def zscore(x):return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-8)
def ranks(cand,scores):
    assert cand.shape==scores.shape and np.isfinite(scores).all()
    out=1+((scores>scores[:,:1])|((scores==scores[:,:1])&(cand<cand[:,:1]))).sum(1)
    for r in range(min(10,len(cand))):
        order=np.lexsort((cand[r],-scores[r]));assert np.flatnonzero(order==0)[0]+1==out[r]
    return out
def metrics(r):
    return {'ndcg':float(np.where(r<=10,1/np.log2(r+1),0).mean()),'recall':float((r<=10).mean()),
      'mrr':float(np.where(r<=10,1/r,0).mean()),'users':len(r)}
def ndcg(r):return np.where(r<=10,1/np.log2(r+1),0)
def score(phase,d,c,bundle):
    start=ts('2019-11-01' if phase=='development' else '2019-12-01')
    end=ts('2019-12-01' if phase=='development' else '2020-01-01')
    caseset,catalog,meta=cases(d,c,start,end,start,2000 if phase=='development' else 3000)
    assert len(caseset)>=200,'Insufficient evaluable users for bounded validation'
    log(f'{phase}: {len(caseset):,} cases selected from {meta["eligible_distinct_users_before_cap"]:,}')
    cand=make_candidates(d,caseset,catalog);ci=np.searchsorted(catalog,cand)
    assert np.array_equal(catalog[ci],cand)
    users,pairs,history,pop,succ,fail,info=graph_inputs(d,c,start,catalog)
    assert info['latest_mature_source_plus_24h']<start
    rows=np.searchsorted(users,caseset.user_id);assert np.array_equal(users[rows],caseset.user_id)
    log(f'{phase}: constructing RP3Beta on {history.nnz:,} capped pairs and {len(catalog):,} products')
    edge,sim=build_rp3beta(history,block_size=64)
    base_raw=gather(edge,sim,rows,ci);base=zscore(base_raw)
    risk=model_risks(bundle,caseset)
    ref=succ[ci]-bundle['global_risk']*fail[ci];scale=ref.std(1,keepdims=True)+1e-8
    signals={}
    for name,r in risk.items():
        x=succ[ci]-r[:,None]*fail[ci]; signals[name]=(x-x.mean(1,keepdims=True))/scale
    signals['popularity']=zscore(pop[ci])
    grid=[0,.01,.03,.1]; allr={};table=[]
    for name,x in signals.items():
        for gamma in grid:
            rr=ranks(cand,base+gamma*x);allr[(name,gamma)]=rr
            table.append({'variant':name,'gamma':gamma,**metrics(rr)})
    base_r=ranks(cand,base); selected={}
    if phase=='development':
        for name in signals:
            best=max([r for r in table if r['variant']==name],key=lambda r:(r['ndcg'],r['recall'],-r['gamma']))
            selected[name]=best['gamma']
    else:selected=json.loads((OUT/'selected_settings.json').read_text())['weights']
    frame=caseset[['case_id','user_id','t','product_id','target','target_t','y']].copy()
    frame['rp3beta_rank']=base_r
    summaries=[{'variant':'rp3beta','gamma':0,**metrics(base_r)}]
    for name,gamma in selected.items():
        frame[name+'_rank']=allr[(name,gamma)];frame[name+'_risk']=risk.get(name,np.full(len(frame),np.nan))
        summaries.append({'variant':name,'gamma':gamma,**metrics(allr[(name,gamma)])})
        frame[name+'_matched_weight_rank']=allr[(name,selected['actual'])]
    del sim,edge
    log(f'{phase}: constructing fixed P3LTR degree-recency component')
    edge,sim=p3(history,pairs)
    p3r=ranks(cand,gather(edge,sim,rows,ci));frame['p3ltr_component_rank']=p3r
    summaries.append({'variant':'p3ltr_component','gamma':0,**metrics(p3r)})
    prediction={}
    for name in ['actual','context','fixed_global']:
        r=risk[name];y=caseset.y.to_numpy()
        prediction[name]={'auc':float(roc_auc_score(y,r)) if len(np.unique(y))==2 else None,
          'average_precision':float(average_precision_score(y,r)),'brier':float(brier_score_loss(y,r)),
          'log_loss':float(log_loss(y,r,labels=[0,1])),'nonprogression_rate':float(y.mean())}
    folder=OUT/phase;folder.mkdir(exist_ok=True)
    frame.to_parquet(folder/'paired_cases.parquet',index=False);np.save(folder/'candidates.npy',cand)
    write(folder/'summary.json',{'case_selection':meta,'graph':info,'selected_metrics':summaries,'weight_grid_metrics':table if phase=='development' else None,
       'prediction':prediction,'target_zero_rp3_score_fraction':float((base_raw[:,0]==0).mean()),
       'candidate_checks_passed':True,'exact_rank_oracle_cases_per_setting':min(10,len(frame))})
    if phase=='development':
        write(OUT/'selected_settings.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'selection_period':'November 2019 only',
          'selection_outcomes_end_exclusive':end,'weights':selected,'signature':signature(),
          'model_sha256':digest(OUT/'models.pkl'),'development_cases_sha256':digest(folder/'paired_cases.parquet'),
          'december_download_authorized_after_this_record':True})
    else:
        contrasts=[]
        for other in ['fixed_global','rp3beta','shuffled','context','success_only','popularity','p3ltr_component']:
            left=ndcg(frame.actual_rank.to_numpy());right=ndcg(frame[other+'_rank'].to_numpy())
            row,_=paired_summary(left,right,key='actual_minus_'+other,primary=False,replicates=5000,seed=SEED)
            row['prespecified_primary']=other=='fixed_global'
            row['interpretation']='Single prespecified primary 95% interval' if other=='fixed_global' else 'Exploratory pointwise 95% interval; no multiplicity adjustment'
            contrasts.append(row)
        write(folder/'statistical_comparisons.json',contrasts)
    log(json.dumps({'phase':phase,'selected_metrics':summaries,'prediction':prediction}))
def main():
    global OUT,DATA
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['development','test'])
    parser.add_argument('--work-dir',type=Path,default=OUT)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    args=parser.parse_args()
    OUT=args.work_dir.resolve();DATA=args.data_dir.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/(args.phase+'_completion.json')).exists():
        raise FileExistsError('Completed phase exists; use a new work directory')
    config=ROOT/'configs/cosmetics_protocol.json'
    if not (OUT/'protocol.json').exists():(OUT/'protocol.json').write_bytes(config.read_bytes())
    assert digest(config)==digest(OUT/'protocol.json'),'Protocol copy differs from repository configuration'
    protocol=json.loads((OUT/'protocol.json').read_text())
    lock=OUT/'protocol_lock.json'
    if not lock.exists():write(lock,{'created_utc':datetime.now(timezone.utc).isoformat(),'signature':signature()})
    assert json.loads(lock.read_text())['signature']==signature(),'Code/protocol changed after lock; record amendment before continuing'
    if args.phase=='test':
        selection=json.loads((OUT/'selected_settings.json').read_text())
        assert selection['signature']==signature() and selection['model_sha256']==digest(OUT/'models.pkl')
        assert selection['selection_outcomes_end_exclusive']<=ts('2019-12-01')
        gate_path=OUT/'december_download_record.json'
        if not gate_path.exists():
            write(gate_path,{'selected_settings_sha256':digest(OUT/'selected_settings.json'),
                'archive_sha256':digest(DATA/'2019-Dec.csv.zip'),
                'scope':'Reproduction with locally supplied data, not a new independent evaluation'})
        gate=json.loads(gate_path.read_text())
        assert gate['selected_settings_sha256']==digest(OUT/'selected_settings.json')
    d,inputs=load_events(args.phase);c=feature_frame(d,args.phase)
    if args.phase=='development':bundle=fit_models(d,c)
    else:
        with (OUT/'models.pkl').open('rb') as f:bundle=pickle.load(f)
    score(args.phase,d,c,bundle)
    write(OUT/(args.phase+'_completion.json'),{'created_utc':datetime.now(timezone.utc).isoformat(),
       'signature':signature(),'inputs':inputs,'complete':True})
if __name__=='__main__':main()
