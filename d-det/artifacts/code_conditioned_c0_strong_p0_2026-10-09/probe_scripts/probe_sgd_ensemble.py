import sys, json, time
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
ROOT = Path('/root/autodl-tmp/u-det')
sys.path.insert(0, str(ROOT/'scripts'))
import cc_common as cc
design = cc.load_design(); b = design['bundle']; rows = design['rows']
texts = [r['code'] for r in rows]
Xc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), min_df=1, sublinear_tf=True, lowercase=False).fit_transform(texts)
Xw = TfidfVectorizer(analyzer='word', token_pattern=r'[A-Za-z_][A-Za-z0-9_]*', ngram_range=(1,3), min_df=1, sublinear_tf=True, lowercase=False).fit_transform(texts)
sm = np.hstack([b['style'], b['meta'], b['sizelen'][:, :3]])
def sgd_ens(Xtr, ytr, Xev, seeds=(0,1,2), epochs=5, bs=4096, alpha=1e-6):
    out = np.zeros(Xev.shape[0])
    for s in seeds:
        clf = SGDClassifier(loss='log_loss', alpha=alpha, random_state=s)
        rng = np.random.default_rng(s)
        for ep in range(epochs):
            perm = rng.permutation(Xtr.shape[0])
            for i in range(0, len(perm), bs):
                sel = perm[i:i+bs]
                clf.partial_fit(Xtr[sel], ytr[sel], classes=np.array([0,1]))
        out += clf.decision_function(Xev)
    return out / len(seeds)
per=[]
for fold in design['folds']:
    fit_mask, ev_mask, pos_mask, _ = cc.fold_setup(design, fold, split='dev', inner=False)
    y_fit = cc.fold_series_target(design, fold)[fit_mask]
    ev = np.where(ev_mask)[0]; fit = np.where(fit_mask)[0]
    sc = StandardScaler().fit(b['hy_small'][fit]); sc2 = StandardScaler().fit(sm[fit])
    s_tr, s_ev = {}, {}
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(b['hy_small'][fit]), y_fit)
    s_tr['sem']=clf.decision_function(sc.transform(b['hy_small'][fit])); s_ev['sem']=clf.decision_function(sc.transform(b['hy_small'][ev]))
    clf = LogisticRegression(max_iter=2000).fit(sc2.transform(sm[fit]), y_fit)
    s_tr['sty']=clf.decision_function(sc2.transform(sm[fit])); s_ev['sty']=clf.decision_function(sc2.transform(sm[ev]))
    s_tr['char']=sgd_ens(Xc[fit], y_fit, Xc[fit]); s_ev['char']=sgd_ens(Xc[fit], y_fit, Xc[ev])
    s_tr['word']=sgd_ens(Xw[fit], y_fit, Xw[fit]); s_ev['word']=sgd_ens(Xw[fit], y_fit, Xw[ev])
    f_ev,_,_ = cc.zfit_fuse(s_tr, s_ev)
    per.append(cc.auroc(pos_mask[ev].astype(int), f_ev))
print('W3_sgd_ens', round(float(np.mean(per)),4), flush=True)
json.dump({'mean': float(np.mean(per)), 'per': per}, open('/tmp/cc_sgd_test.json','w'))
