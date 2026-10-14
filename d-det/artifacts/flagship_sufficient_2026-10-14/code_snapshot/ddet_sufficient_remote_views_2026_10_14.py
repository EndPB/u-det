"""Finish GPU encodings in the same remote backend, then publish transfer inventory."""
from pathlib import Path
import functools,hashlib,json,sys,time
import numpy as np
import torch
from transformers import T5EncoderModel
import ddet_sufficient_local_2026_10_14 as t
import ddet_two_tasks_local_2026_10_14 as b
from ddet_sufficient_analysis_2026_10_14 import cohort_indices
from ddet_t5_local_fast_2026_10_14 import install


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    t.setup();deadline=time.monotonic()+43200;log=t.OUT/'remote_train.log.txt'
    while time.monotonic()<deadline:
        if log.exists() and 'SUFFICIENT_ENCODERS_AND_FEATURES_DONE' in log.read_text(encoding='utf-8',errors='replace')[-10000:]:break
        time.sleep(30)
    else:raise TimeoutError('encoder batch did not finish')
    rows=list(b.readgz(t.DATA/'pilot.jsonl.gz'));tok=b.tokenizer();cache={}
    # If the initial cap was reached while still improving, continue the same optimizer.
    t.MICRO=32;t.encode=functools.partial(t.encode,batch=64);events=[]
    for cap in [24,36]:
        pending=[(task,arm) for task in ['F','D'] for arm in ['B0','B1'] if not json.loads((t.OUT/f'{task}_{arm}_training.json').read_text())['validation_saturated']]
        if not pending:break
        if not cache:cache,_=__import__('ddet_flagship_local_2026_10_14').token_cache(rows,tok)
        t.MAX_EPOCH=cap
        for task,arm in pending:
            path=t.OUT/f'{task}_{arm}_training.json';fit=json.loads(path.read_text());events.append({'task':task,'arm':arm,'from_epochs':len(fit['history']),'new_cap':cap,'reason':'calibration_not_saturated_at_initial_cap','eval_scores_used':False})
            b.save(t.OUT/'training_extensions.json',events);b.save(t.OUT/f'{task}_{arm}_before_extension_{cap}_training.json',fit);fit['completed']=False;b.save(path,fit)
            t.train(rows,cache,tok,task,arm)
            old=t.OUT/f'{task}_{arm}_features.npz';backup=t.OUT/f'{task}_{arm}_before_extension_{cap}_features.npz'
            if old.exists():old.replace(backup)
            t.features(rows,cache,task,arm)
    del cache;cache={};cohorts=cohort_indices(rows);tr=[i for i,r in enumerate(rows) if r['split']=='train'];anchor=sorted(tr,key=lambda i:b.sh(rows[i]['record_id']))[:2048]
    selected=np.array(sorted(set(anchor).union(*(ix.tolist() for ix in cohorts.values()))));path=t.OUT/'pretraining_visual_features.npz'
    if not path.exists():
        for i in selected:
            ids=tok.encode(rows[i]['code'],add_special_tokens=False);cache[b.sh(rows[i]['code'])]=ids if len(ids)<=512 else ids[:384]+ids[-128:]
        model=install(T5EncoderModel.from_pretrained(b.MODEL,local_files_only=True)).cuda();h,_=t.encode(model,None,rows,cache,indices=selected);del model;torch.cuda.empty_cache()
        np.savez_compressed(path,h=h,indices=selected,record_ids=np.array([rows[i]['record_id'] for i in selected]))
    extra=list(b.readgz(t.DATA/'unseen_families.jsonl.gz'));cache={}
    for r in extra:r['stress_content']=b.content_view(r['stress_view'])
    for r in extra:
        for field in ['code','stress_view','content_view','stress_content']:
            key=b.sh(r[field])
            if key not in cache:
                ids=tok.encode(r[field],add_special_tokens=False);cache[key]=ids if len(ids)<=512 else ids[:384]+ids[-128:]
    for arm in ['B0','B1']:
        path=t.OUT/f'unseen_families_D_{arm}_features.npz'
        if path.exists():continue
        model,head=t.load_model('D',arm);h,base=t.encode(model,head,extra,cache);hs,stress=t.encode(model,head,extra,cache,'stress_view');del model,head;torch.cuda.empty_cache()
        np.savez_compressed(path,h=h,base=base,hs=hs,stress=stress,record_ids=np.array([r['record_id'] for r in extra]))
    path=t.OUT/'unseen_families_semantic_features.npz'
    if not path.exists():
        model=install(T5EncoderModel.from_pretrained(b.MODEL,local_files_only=True)).cuda();sem,_=t.encode(model,None,extra,cache,'content_view');ss,_=t.encode(model,None,extra,cache,'stress_content');del model;torch.cuda.empty_cache()
        np.savez_compressed(path,sem=sem,stress_sem=ss,record_ids=np.array([r['record_id'] for r in extra]))
    files=[]
    wanted=['protocol.json','remote_execution.json','remote_train.log.txt','attention_equivalence_and_benchmark.json','semantic_features.npz','pretraining_visual_features.npz','unseen_families_D_B0_features.npz','unseen_families_D_B1_features.npz','unseen_families_semantic_features.npz']
    wanted += [f'{task}_{arm}_{suffix}' for task in ['F','D'] for arm in ['B0','B1'] for suffix in ['features.npz','training.json']]
    if (t.OUT/'training_extensions.json').exists():wanted.append('training_extensions.json')
    for name in wanted:
        path=t.OUT/name;files.append({'kind':'artifact','name':name,'remote_path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)})
    for task in ['F','D']:
        for arm in ['B0','B1']:
            path=t.DATA/f'{task}_{arm}_best.pt';files.append({'kind':'best_checkpoint','name':path.name,'remote_path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)})
    metadata={'files':files,'all_visual_and_extra_encodings_same_remote_backend':True,'raw_source_main_remains_local':True,'runtime_source_sha256':{p.name:sha(p) for p in (t.ROOT/'scripts').glob('ddet_*.py')}}
    b.save(t.OUT/'remote_views_done.json',metadata);b.log('REMOTE_GPU_VIEWS_DONE',len(files))


def status():
    out=t.OUT;fits={}
    for task in ['F','D']:
        for arm in ['B0','B1']:
            path=out/f'{task}_{arm}_training.json'
            if path.exists():
                try:fits[f'{task}_{arm}']=json.loads(path.read_text())
                except json.JSONDecodeError:pass
    log=out/'remote_train.log.txt';last=log.read_text(encoding='utf-8',errors='replace')[-3000:] if log.exists() else ''
    viewslog=out/'remote_views.log.txt';views=viewslog.read_text(encoding='utf-8',errors='replace')[-3000:] if viewslog.exists() else ''
    print(json.dumps({'fits':fits,'last_encoder_log':last,'last_views_log':views,'inventory_ready':(out/'remote_views_done.json').exists()},ensure_ascii=False),flush=True)


if __name__=='__main__':
    if '--status' in sys.argv:status()
    else:main()
