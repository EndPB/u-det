import sys, json, time
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT = Path('/root/autodl-tmp/u-det')
sys.path.insert(0, str(ROOT/'scripts'))
import cc_common as cc
design = cc.load_design(); b = design['bundle']; rows = design['rows']
texts = [r['code'] for r in rows]
t0=time.time()
Xc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), min_df=1, sublinear_tf=True, lowercase=False).fit_transform(texts)
Xw = TfidfVectorizer(analyzer='word', token_pattern=r'[A-Za-z_][A-Za-z0-9_]*', ngram_range=(1,3), min_df=1, sublinear_tf=True, lowercase=False).fit_transform(texts)
print('tfidf', Xc.shape, Xw.shape, f'{time.time()-t0:.0f}s', flush=True)
sm = np.hstack([b['style'], b['meta'], b['sizelen'][:, :3]])
def run(tag, balanced, C_lex):
    per=[]
    for fold in design['folds']:
        fit_mask, ev_mask, pos_mask, _ = cc.fold_setup(design, fold, split='dev', inner=False)
        y_fit = cc.fold_series_target(design, fold)[fit_mask]
        ev = np.where(ev_mask)[0]; fit = np.where(fit_mask)[0]
        sc = StandardScaler().fit(b['hy_small'][fit]); sc2 = StandardScaler().fit(sm[fit])
        mats_tr = {'sem': sc.transform(b['hy_small'][fit]), 'char': Xc[fit], 'word': Xw[fit], 'sty': sc2.transform(sm[fit])}
        mats_ev = {'sem': sc.transform(b['hy_small'][ev]), 'char': Xc[ev], 'word': Xw[ev], 'sty': sc2.transform(sm[ev])}
        s_tr, s_ev = {}, {}
        for k in mats_tr:
            cw = 'balanced' if balanced else None
            clf = LogisticRegression(max_iter=3000, C=(C_lex if k in ('char','word') else 1.0), class_weight=cw)
            clf.fit(mats_tr[k], y_fit)
            s_tr[k]=clf.decision_function(mats_tr[k]); s_ev[k]=clf.decision_function(mats_ev[k])
        f_ev,_,_ = cc.zfit_fuse(s_tr, s_ev)
        per.append(cc.auroc(pos_mask[ev].astype(int), f_ev))
    m=float(np.mean(per)); print(tag, round(m,4), flush=True); return {'mean':m,'per':per}
res={}
res['W_wide'] = run('W_wide', False, 4.0)
res['W2_wide_bal'] = run('W2_wide_bal', True, 4.0)
json.dump(res, open('/tmp/cc_wide_test.json','w'))
