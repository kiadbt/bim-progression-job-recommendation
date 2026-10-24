"""Check aggregate cosmetics records without source data or model fitting."""
from pathlib import Path
import json, math

ROOT=Path(__file__).resolve().parents[1]
read=lambda name:json.loads((ROOT/'results/cosmetics'/name).read_text(encoding='utf-8'))

def main():
    evaluation=read('evaluation.json')
    rows={r['variant']:r for r in evaluation['selected_metrics']}
    assert evaluation['case_selection']['selected_users']==3000
    assert evaluation['case_selection']['eligible_distinct_users_before_cap']==8782
    for r in rows.values():
        assert r['users']==3000 and 0<=r['mrr']<=r['ndcg']<=r['recall']<=1
    comparisons=read('paired_comparisons.json')
    primary=[r for r in comparisons if r['prespecified_primary']]
    assert len(primary)==1 and primary[0]['contrast']=='actual_minus_fixed_global'
    for r in comparisons:
        other=r['contrast'].removeprefix('actual_minus_')
        delta=rows['actual']['ndcg']-rows[other]['ndcg']
        assert math.isclose(delta,r['difference'],abs_tol=1e-14)
        assert r['positive_users']+r['negative_users']+r['unchanged_users']==3000
        assert r['ci95'][0]<=r['ci95'][1] and r['replicates']==5000
    assert primary[0]['positive_users']==0 and primary[0]['negative_users']==2
    assert primary[0]['ci95'][1]==0
    selected=read('selected_settings.json')
    development=read('development.json')
    for name,gamma in selected['weights'].items():
        best=max((r for r in development['weight_grid_metrics'] if r['variant']==name),
                 key=lambda r:(r['ndcg'],r['recall'],-r['gamma']))
        assert best['gamma']==gamma==0.1
    fit=read('prediction_fit.json')
    assert fit['training_users']==50000 and fit['calibration']['selected_users']==1739
    assert all(r['iterations']==8 and r['warnings'] for r in fit['models'].values())
    assert evaluation['prediction']['actual']['auc']<evaluation['prediction']['context']['auc']
    print('Verified cosmetics aggregate means, seven contrasts, primary endpoint, development selection and prediction qualifications.')

if __name__=='__main__':main()
