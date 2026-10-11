#!/usr/bin/env python3
"""Independent saved-score checks and train-only task CI; no GPU reruns."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import shutil
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from h3_local_revision_v1 import code_lengths
from h3_ammc_mix_v5 import ROOT, DATA, OUT as BUILD, read, write, TRANSFORMS
import h3_v4_local_experiment as prior

OUT=ROOT/'d-det/artifacts/h3_ammc_v5_local_batch_2026-10-11'

def main():
    failures=[];checks=0
    for section in ['neural_metrics.json','source_metrics.json']:
        runs=read_json= json.loads((OUT/section).read_bytes())
        for run in runs:
            for name,m in run['eval'].items():
                x=np.load(OUT/m['score_file']['path'],allow_pickle=False)
                assert len(set(x['row_ids'].tolist()))==len(x['row_ids'])
                if section.startswith('neural'):
                    rows=[{'task_id':t,'label':int(y)} for t,y in zip(x['task_ids'],x['labels'])]
                    got=prior.summary(rows,x['scores'],.5)
                    for k in ['row_auroc','task_macro_auroc','balanced_accuracy']:
                        checks+=1
                        if abs(got[k]-m[k])>1e-12:failures.append([section,run['arm'],name,k])
                else:
                    got=float((x['scores'].argmax(1)==x['source_labels']).mean());checks+=1
                    if abs(got-m['accuracy'])>1e-12:failures.append([section,run['arm'],name,'accuracy'])
    controls=json.loads((OUT/'control_metrics.json').read_bytes())
    for fold,arms in controls.items():
        for arm,views in arms.items():
            for view,m in views.items():
                for name in ['clean','hardneg']:
                    v=m[name];x=np.load(OUT/v['score_file']['path'],allow_pickle=False)
                    rows=[{'task_id':t,'label':int(y)} for t,y in zip(x['task_ids'],x['labels'])]
                    got=prior.summary(rows,x['scores'],0)
                    for k in ['row_auroc','task_macro_auroc','balanced_accuracy']:
                        checks+=1
                        if abs(got[k]-v[k])>1e-12:failures.append([fold,arm,view,name,k])
    write(OUT/'saved_score_verification.json',{'checks':checks,'failures':failures,'passed':not failures,
        'scope':'saved logits -> row/task metrics; no training or encoder rerun'})
    assert not failures
    parents=sorted(read(ROOT/'d-det/data/h3_stacad_revision_v4/hardneg_train_v4.jsonl'),key=lambda r:r['row_id'])
    schedules=read(DATA/'train_schedule.jsonl');views=read(DATA/'unique_train_views.jsonl')
    codes={r['code_sha256']:r['code'] for r in parents+views}
    fit=np.asarray([int(hashlib.sha256(r['task_id'].encode()).hexdigest()[:8],16)%5!=0 for r in parents])
    y=np.asarray([r['label'] for r in parents]);tasks=np.asarray([r['task_id'] for r in parents])
    ci={}
    for arm in ['single','composed']:
        byp=defaultdict(list)
        for r in schedules:
            if r['arm']==arm:byp[r['parent_row_id']].append(r)
        matrices=defaultdict(list)
        for p in parents:
            rec=byp[p['row_id']]
            for kind in ['requested','accepted']:
                matrices[kind].append(np.mean([[int(t in r[kind+'_mask']) for t,_ in TRANSFORMS] for r in rec],axis=0))
            original=np.log1p(code_lengths(p['code']))
            matrices['view_delta'].append(np.mean([np.log1p(code_lengths(codes[r['code_sha256']]))-original for r in rec],axis=0))
        ci[arm]={}
        for kind,values in matrices.items():
            x=np.asarray(values);clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=300,random_state=2026101105))
            clf.fit(x[fit],y[fit]);scores=clf.decision_function(x[~fit]);yt=y[~fit];tt=tasks[~fit]
            tauc=[]
            for t in sorted(set(tt)):
                ix=tt==t;tauc.append(roc_auc_score(yt[ix],scores[ix]))
            tauc=np.asarray(tauc);rng=np.random.default_rng(2026101105)
            boot=tauc[rng.integers(len(tauc),size=(2000,len(tauc)))].mean(1)
            ci[arm][kind]={'row_auroc':float(roc_auc_score(yt,scores)),'task_macro_auroc':float(tauc.mean()),
                'task_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'tasks':len(tauc),'bootstrap':2000,
                'interpretation':'train-only heldout tasks; CI conditional on fitted probe; no model/tuning selection'}
            np.savez_compressed(OUT/f'probe_{arm}_{kind}.npz',scores=scores,labels=yt,task_ids=tt,row_ids=np.asarray([r['row_id'] for r in parents])[~fit])
    write(OUT/'train_only_probe_ci.json',ci)
    write(OUT/'closeout_config.json',{'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'score_checks':checks,'probe_bootstrap':2000,'probe_seed':2026101105,'test_read':False})
    log=ROOT/'d-det/artifacts/h3_ammc_v5_local_batch_2026-10-11_run.log.txt'
    shutil.copyfile(log,OUT/'run.log.txt')
    for directory in [OUT,BUILD]:
        lines=[]
        for p in sorted(directory.rglob('*')):
            if p.is_file() and p.name!='SHA256SUMS.txt':lines.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(directory).as_posix()}\n')
        (directory/'SHA256SUMS.txt').write_bytes(''.join(lines).encode())
    print(json.dumps({'saved_score_checks':checks,'failures':len(failures),'probe_ci':ci},ensure_ascii=False))

if __name__=='__main__':main()
