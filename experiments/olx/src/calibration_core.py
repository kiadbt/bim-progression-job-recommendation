"""Validation-only monotonic calibration of nonprogression pseudo-label scores."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit


def fit_validation_sigmoid(frame,available_from):
    required=['split','source_timestamp','q95_seconds','risk_raw','nonprogression_q90']
    if frame[required].isna().any().any() or not (frame.split=='validation').all():
        raise ValueError('Calibration fitting requires complete validation rows only')
    if not ((frame.source_timestamp+frame.q95_seconds)<available_from).all():
        raise ValueError('Calibration outcomes cross the availability boundary')
    p=frame.risk_raw.to_numpy(dtype=float)
    y=frame.nonprogression_q90.to_numpy(dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:
        raise ValueError('Scores must be finite probabilities and labels must include both classes')
    x=logit(np.clip(p,1e-7,1-1e-7))
    def objective(theta):
        z=theta[0]*x+theta[1]
        residual=expit(z)-y
        return float(np.mean(np.logaddexp(0,z)-y*z)),np.array([np.mean(residual*x),np.mean(residual)])
    initial=[1.,float(logit(y.mean())-logit(p.mean()))]
    fit=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(1e-8,None),(None,None)],
                 options={'maxiter':500,'ftol':1e-12,'gtol':1e-9})
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f'Calibration failed: {fit.message}')
    return {'method':'sigmoid_of_logit','slope':float(fit.x[0]),'intercept':float(fit.x[1]),
        'input_clip':1e-7,'fit_split':'validation','available_from':int(available_from),
        'fit_rows':len(frame),'validation_nonprogression_rate':float(y.mean()),
        'loss':'unweighted binary cross entropy','iterations':int(fit.nit),
        'validation_loss_apparent':float(fit.fun),'selection_policy':'fixed method, no test-based choice'}


def apply_sigmoid(p,parameters):
    p=np.asarray(p,dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid raw scores')
    return expit(parameters['slope']*logit(np.clip(p,parameters['input_clip'],1-parameters['input_clip']))+parameters['intercept'])
