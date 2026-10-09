import sys, json, time
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT = Path('/root/autodl-tmp/u-det')
sys.path.insert(0, str(ROOT/'scripts'))
import cc_common as cc
design = cc.load_design(); b = design['bundle']; rows = design['rows']
nT = len(design['tasks_all'])
r0_rows = b['gmodel_idx']*nT*2 + b['task_idx']*2 + 1
emb_base = np.load(cc.R0/'emb_base.npz')['emb'][r0_rows]
variants = {'small': b['hy_small'], 'base': emb_base, 'concat': np.hstack([b['hy_small'], emb_base])}
res={}
for name, sem in variants.items():
    aucs=[]
    for fold in design['folds']:
        fit_mask, ev_mask, pos_mask, _ = cc.fold_setup(design, fold, split='dev', inner=False)
        y_fit = cc.fold_series_target(design, fold)[fit_mask]
        ev = np.where(ev_mask)[0]; fit = np.where(fit_mask)[0]
        sc = StandardScaler().fit(sem[fit])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(sem[fit]), y_fit)
        aucs.append(cc.auroc(pos_mask[ev].astype(int), clf.decision_function(sc.transform(sem[ev]))))
    res[name]={'mean': float(np.mean(aucs)), 'per_fold': aucs}
    print(name, round(res[name]['mean'],4), flush=True)
json.dump(res, open('/tmp/cc_sem_test.json','w'))
