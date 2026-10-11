#!/usr/bin/env python3
"""Train-only AMMC-like mix with fixed per-parent exposures and paired arms.

All syntax/token checks are safety proxies, never execution correctness claims.
No test data is loaded. Existing v4 dev-informed selection stays exploratory.
"""
from __future__ import annotations
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from h3_local_revision_v1 import parser_for, transformed_code, code_lengths, LANGS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'd-det/data/h3_ammc_mix_v5'
OUT = ROOT / 'd-det/artifacts/h3_ammc_mix_v5_2026-10-11'
SEED = 2026101105
TRANSFORMS = [('comment', .2), ('whitespace', .3), ('rename', .4)]

def sha(v):
    return hashlib.sha256(v).hexdigest()

def read(p):
    return [json.loads(l) for l in p.open(encoding='utf-8') if l.strip()]

def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((json.dumps(v, ensure_ascii=False, indent=2)+'\n').encode())

def jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('wb') as f:
        for r in rows:
            f.write((json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n').encode())

def rng_for(key):
    return np.random.default_rng(int(sha(f'{SEED}:{key}'.encode())[:16], 16))

def probes(parents, schedules, variants):
    """Split TRAIN tasks first; average repeated exposures per parent."""
    lookup = {r['code_sha256']: r['code'] for r in parents+variants}
    train_tasks = sorted({r['task_id'] for r in parents})
    test_tasks = {t for t in train_tasks if int(sha(t.encode())[:8], 16)%5 == 0}
    fit = np.asarray([r['task_id'] not in test_tasks for r in parents])
    y = np.asarray([r['label'] for r in parents])
    output = {'scope': 'train-only task heldout; no dev/test used', 'fit_tasks': len(train_tasks)-len(test_tasks),
              'eval_tasks': len(test_tasks), 'arms': {}}
    for arm in ['single', 'composed']:
        by_parent = defaultdict(list)
        for r in schedules:
            if r['arm']==arm: by_parent[r['parent_row_id']].append(r)
        matrices = defaultdict(list)
        for p in parents:
            records = by_parent[p['row_id']]
            matrices['requested_mask'].append(np.mean([[int(t in r['requested_mask']) for t,_ in TRANSFORMS] for r in records], axis=0))
            matrices['effective_mask'].append(np.mean([[int(t in r['accepted_mask']) for t,_ in TRANSFORMS] for r in records], axis=0))
            original = np.log1p(code_lengths(p['code']))
            matrices['view_delta'].append(np.mean([np.log1p(code_lengths(lookup[r['code_sha256']]))-original for r in records], axis=0))
        output['arms'][arm] = {}
        for name, values in matrices.items():
            x = np.asarray(values)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=300, random_state=SEED))
            clf.fit(x[fit], y[fit]); s = clf.decision_function(x[~fit])
            output['arms'][arm][name] = {'row_auroc': float(roc_auc_score(y[~fit], s)), 'rows': int((~fit).sum()),
                'label_mean_features': {str(lab):x[y==lab].mean(0).tolist() for lab in [0,1]}}
    return output

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT/'d-det/data/h3_stacad_revision_v4/hardneg_train_v4.jsonl'
    parents = sorted(read(source), key=lambda r:r['row_id'])
    assert all(r['task_split']=='train' for r in parents)
    config = {'seed':SEED, 'epochs':3, 'probabilities':dict(TRANSFORMS), 'arms':['original','single','composed'],
              'selection':'independent Bernoulli per parent/epoch; single chooses one triggered transform; composed applies all in fixed order',
              'exposures':'exactly one row per parent per epoch in each arm; rejection falls back to current/original view',
              'original_sha256':sha(source.read_bytes()), 'script_sha256':sha(Path(__file__).read_bytes()),
              'transform_helper_sha256':sha((ROOT/'scripts/h3_local_revision_v1.py').read_bytes()),
              'test_read':False, 'generation':False, 'code_execution':False, 'gate':'revise_data',
              'scope':'exploratory; v4 selected with dev information; not project heldout',
              'safety':'parser/tree/token alignment and Python AST where applicable; proxy only'}
    write(OUT/'config.json',config)
    parsers = {lang:parser_for(lang) for lang in LANGS}
    schedules=[]; unique={}; counts=Counter(); proofs=[]
    cache={}
    for i,p in enumerate(parents):
        for epoch in range(3):
            rng=rng_for(f"{p['row_id']}:{epoch}")
            requested=[t for t,prob in TRANSFORMS if rng.random()<prob]
            chosen=[] if not requested else [requested[int(rng.integers(len(requested)))]]
            for arm, ts in [('original',[]),('single',chosen),('composed',requested)]:
                current=dict(p); accepted=[]; rejected=[]
                for t in ts:
                    key=(current['code_sha256'], p['language'], t)
                    if key not in cache:
                        try: cache[key]=transformed_code(current,t,parsers[p['language']])
                        except (RecursionError,ValueError,UnicodeError,SyntaxError) as exc: cache[key]=(None,type(exc).__name__,{})
                    new,reason,proof=cache[key]
                    counts[f'{arm}:{t}:{reason}']+=1
                    before=current['code_sha256']
                    if new is not None:
                        accepted.append(t); current['code']=new; current['code_sha256']=sha(new.encode())
                    else: rejected.append({'transform':t,'reason':reason})
                    proofs.append({'parent_row_id':p['row_id'],'epoch':epoch,'arm':arm,'transform':t,
                        'status':reason,'before_sha256':before,'after_sha256':current['code_sha256'],'proof':proof})
                sch={'parent_row_id':p['row_id'],'task_id':p['task_id'],'language':p['language'],'label':p['label'],
                     'generator':p['generator'],'arm':arm,'epoch':epoch,'requested_mask':requested,
                     'selected_mask':ts,'accepted_mask':accepted,'rejected':rejected,'code_sha256':current['code_sha256']}
                schedules.append(sch)
                if current['code_sha256']!=p['code_sha256']:
                    unique[current['code_sha256']]={'code_sha256':current['code_sha256'],'code':current['code']}
        if i%1000==0: print('BUILD',i,'/',len(parents),flush=True)
    variants=list(unique.values())
    jsonl(DATA/'unique_train_views.jsonl',variants)
    jsonl(DATA/'train_schedule.jsonl',schedules)
    jsonl(OUT/'transform_proofs.jsonl',proofs)
    audit={'parents':len(parents),'epochs':3,'exposures_per_arm':len(parents)*3,'unique_changed_codes':len(variants),
           'attempt_status':dict(counts),'test_read':False,'no_eval_views_generated':True}
    for arm in config['arms']:
        rows=[r for r in schedules if r['arm']==arm]
        assert len(rows)==len(parents)*3
        assert all(n==3 for n in Counter(r['parent_row_id'] for r in rows).values())
        audit[arm]={'changed_exposures':sum(bool(r['accepted_mask']) for r in rows),
                    'composed_ge2':sum(len(r['accepted_mask'])>=2 for r in rows),
                    'effective_mask_counts':dict(Counter('+'.join(r['accepted_mask']) or 'identity' for r in rows))}
    write(OUT/'audit.json',audit)
    write(OUT/'train_only_probes.json',probes(parents,schedules,variants))
    write(OUT/'data_manifest.json',{p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(DATA.glob('*.jsonl'))})
    print(json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
