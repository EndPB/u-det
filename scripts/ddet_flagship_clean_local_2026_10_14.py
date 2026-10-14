"""Remove docstring-placeholder blank lines fairly, then rerun the actual task models."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,ast,gzip,io,itertools,json,shutil,tokenize
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import ddet_flagship_local_2026_10_14 as d
import ddet_two_tasks_local_2026_10_14 as b

OLD_DATA=d.BASE_DATA;OLD_OUT=d.BASE_OUT
DATA=b.ROOT/'d-det/data/flagship_two_tasks_clean_2026-10-14'
OUT=b.ROOT/'d-det/artifacts/flagship_two_tasks_clean_2026-10-14'


def fast_encode(model,rows,cache,field,device,head=None):
    """Length-bucket inference, restoring original row order; no change to training."""
    model.eval()
    if head is not None:head.eval()
    order=sorted(range(len(rows)),key=lambda i:len(cache[b.sh(rows[i][field])]))
    hidden=np.empty((len(rows),int(model.config.d_model)),dtype='float32')
    logits=None
    with torch.inference_mode():
        for start in range(0,len(order),32):
            ix=order[start:start+32]
            ids,mask=b.batch_tokens([rows[i] for i in ix],cache,field,device)
            with torch.autocast('cuda',dtype=torch.float16):
                h=b.pooled(model,ids,mask);pred=head(h) if head is not None else None
            hidden[ix]=h.float().cpu().numpy()
            if pred is not None:
                if logits is None:logits=np.empty((len(rows),pred.shape[1]),dtype='float32')
                logits[ix]=pred.float().cpu().numpy()
    assert np.isfinite(hidden).all()
    return hidden,logits


def no_blank_placeholders(code):
    protected=set()
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type==tokenize.STRING:protected.update(range(token.start[0],token.end[0]+1))
    result=''.join(line for i,line in enumerate(code.splitlines(keepends=True),1) if line.strip() or i in protected)
    assert ast.dump(ast.parse(code),include_attributes=False)==ast.dump(ast.parse(result),include_attributes=False)
    return result


def prepare(name):
    target=DATA/name;output=OUT/name;target.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    if (target/'pilot.jsonl.gz').exists() and (output/'protocol.json').exists():return
    rows=list(b.readgz(OLD_DATA/name/'pilot.jsonl.gz'));proof=Counter();statuses=Counter()
    for i,row in enumerate(rows):
        old=row['code'];old_stress=row['stress_view']
        row['code']=no_blank_placeholders(old)
        assert b.sh(__import__('re').sub(r'\s+','',old))==b.sh(__import__('re').sub(r'\s+','',row['code']))
        content=b.content_view(row['code']);assert content==row['content_view']
        row['train_view'],ts=b.augment(row['code'],row['record_id']) if row['split']=='train' else (row['code'],[])
        row['stress_view'],ss=b.augment(row['code'],row['record_id']+':stress',force=True)
        # Semantics-proxy text identity permits reuse of the frozen semantic encoder cache only.
        assert b.content_view(row['stress_view'])==b.content_view(old_stress)
        row['train_transform_status']=ts;row['stress_transform_status']=ss;statuses.update(ts)
        row['parent_diagnostic_view_sha256']=row['pilot_code_sha256'];row['pilot_code_sha256']=b.sh(row['code'])
        proof['rows']+=1;proof['changed']+=old!=row['code'];proof['ast_preserved']+=1;proof['semantic_proxy_text_identical']+=1
        if i%5000==0:b.log('CLEAN_VIEW',name,i,'/',len(rows))
    with gzip.open(target/'pilot.jsonl.gz','wt',encoding='utf-8',newline='\n') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    p=json.loads((OLD_OUT/name/'protocol.json').read_text(encoding='utf-8'))
    p['parent_diagnostic_data_sha256']=p['data_sha256'];p['data_sha256']=b.digest(target/'pilot.jsonl.gz')
    p['input_view']='remove non-string blank lines consistently for all classes; preserve AST; no artificial docstring placeholders'
    p['blank_view_proof']=dict(proof);p['train_arms']=['B1'];p['retrain_encoder_from_pretrained']=True
    p['semantic_cache_reused_only_after_exact_proxy_text_identity']=True
    p['ammc']['train_attempt_status']=dict(statuses);p['clean_script_sha256']=b.digest(Path(__file__))
    p['execution_source_sha256']={f:b.digest(b.ROOT/'scripts'/f) for f in ['ddet_flagship_local_2026_10_14.py','ddet_two_tasks_local_2026_10_14.py']}
    p['no_new_split_or_sample_selection']=True
    b.save(output/'protocol.json',p)
    shutil.copy2(OLD_OUT/name/'semantic_features.npz',output/'semantic_features.npz')
    # This old diagnostic depends only on source identities and labels, both unchanged.
    if (OLD_OUT/name/'source_identity_diagnostic.json').exists():shutil.copy2(OLD_OUT/name/'source_identity_diagnostic.json',output/'source_identity_diagnostic.json')
    def blank_run(code):return max([sum(1 for _ in group) for key,group in itertools.groupby(not line.strip() for line in code.splitlines()) if key] or [0])
    x=np.log1p([[blank_run(r['code'])] for r in rows]);y=np.array([r['label_D'] for r in rows]);tr=np.array([r['split']=='train' for r in rows]);dv=np.array([r['split']=='dev' for r in rows])
    scaler=StandardScaler().fit(x[tr]);clf=LogisticRegression(C=1,class_weight='balanced',max_iter=500).fit(scaler.transform(x[tr]),y[tr])
    b.save(output/'blank_run_diagnostic.json',{'feature':'max non-string blank run after fair cleanup','dev_auroc':float(roc_auc_score(y[dv],clf.decision_function(scaler.transform(x[dv])))),'proof':dict(proof)})
    b.log('CLEAN_READY',name,len(rows),dict(proof))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--slice',choices=['controlled','broad','both'],default='both');args=parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    (DATA/'.gitignore').write_text('*\n!.gitignore\n!README.md\n',encoding='utf-8')
    (OUT/'.gitignore').write_text('**/*_features.npz\n**/semantic_features.npz\n',encoding='utf-8')
    shutil.copy2(OLD_OUT/'flagship_registry_and_pool.json',OUT/'flagship_registry_and_pool.json')
    names=['controlled','broad'] if args.slice=='both' else [args.slice]
    for name in names:prepare(name)
    if args.prepare_only:return
    d.BASE_DATA=DATA;d.BASE_OUT=OUT;b.references=d.references;b.weights=d.task_weights;b.token_cache=d.token_cache
    b.encode=fast_encode
    for name in names:d.run(name,baseline_arms=['B1'])
    b.log('CLEAN_FLAGSHIP_BATCH_DONE')


if __name__=='__main__':main()
