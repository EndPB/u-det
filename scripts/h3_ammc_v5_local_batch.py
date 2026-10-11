#!/usr/bin/env python3
"""Paired local original/single/composed AMMC experiment, fixed parent budget."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import h3_v4_local_experiment as prior
from h3_gpu_diagnostic_batch_2026_10_11 import SharedHeads, encode_unique, make_tokenizer, set_seed

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'d-det/artifacts/h3_ammc_v5_local_batch_2026-10-11'
DATA=ROOT/'d-det/data/h3_ammc_mix_v5'
ARMS=['original','single','composed']
JOBS=['detection_only','joint','joint_invariance']
SEEDS=[20261011,20261012,20261013]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prior.OUT=OUT
    write=prior.write
    paths={'parents':ROOT/'d-det/data/h3_stacad_revision_v4/hardneg_train_v4.jsonl',
           'hardneg':ROOT/'d-det/data/h3_stacad_revision_v4/hardneg_dev_v4.jsonl',
           'clean':ROOT/'d-det/data/h3_stacad_revision_v1/dev_clean.jsonl',
           'schedule':DATA/'train_schedule.jsonl','views':DATA/'unique_train_views.jsonl'}
    parents,hardneg,clean,schedule,views=[prior.read(paths[k]) for k in ['parents','hardneg','clean','schedule','views']]
    byid={r['row_id']:r for r in parents}
    assert all(p['task_split']=='train' for p in parents)
    assert not ({r['task_id'] for r in parents}&{r['task_id'] for r in clean})
    assert {r['row_id'] for r in hardneg}<={r['row_id'] for r in clean}
    codes={r['code_sha256']:r['code'] for r in parents+views+clean}
    for h,code in codes.items(): assert prior.hashlib.sha256(code.encode()).hexdigest()==h
    lookup={}
    for r in schedule:
        assert r['parent_row_id'] in byid and r['code_sha256'] in codes
        key=(r['arm'],r['epoch'],r['parent_row_id']); assert key not in lookup
        lookup[key]=r['code_sha256']
    assert len(lookup)==len(parents)*3*3
    gens=sorted({p['generator'] for p in parents if p['label']==1})
    folds={'fold_0':gens[:3],'fold_1':gens[3:]}
    config={'schema':'h3_ammc_v5_fixed_parent_budget','arms':ARMS,'jobs':JOBS,'folds':folds,'seeds':SEEDS,
        'epochs':3,'learning_rate':.002,'batch_size':256,'lambda_source':.2,'lambda_invariance':.1,
        'train_budget':'exactly one scheduled view per parent per epoch in every arm; fixed parent order/optimizer steps',
        'lexical_budget':'three scheduled views per parent; original repeats three times to equalize document frequency/exposure',
        'frozen_encoder':'CodeT5-small; 512 head/tail tokens; mean pooling; source char cap 24000',
        'criterion':'exploratory only; improvement >=.01 with paired task CI lower>0 in both folds and both eval views against raw lexical and original same job',
        'no_causal_claim':'augmentation arms differ in accepted transforms; safety proxy does not establish correctness; no augmentation stress evaluation yet',
        'gate':'revise_data','formal_gpu_ready':False,'post_dev_selection':True,'project_provenance':'unavailable',
        'test_read':False,'generation':False,'weights_downloaded':False,'code_execution':False,
        'input_sha256':{k:prior.digest(p) for k,p in paths.items()},'script_sha256':prior.digest(Path(__file__)),
        'helpers':{n:prior.digest(ROOT/'scripts'/n) for n in ['h3_v4_local_experiment.py','h3_ammc_mix_v5.py','h3_local_revision_v1.py','h3_gpu_diagnostic_batch_2026_10_11.py']},
        'model_sha256':prior.digest(prior.MODEL/'pytorch_model.bin')}
    write(OUT/'config.json',config)
    write(OUT/'data_role.json',{k:'train-only' if k in ['parents','schedule','views'] else 'dev-only, unchanged, exploratory' for k in paths})
    (OUT/'hypothesis.md').write_text('# AMMC v5 hypothesis\n\nFixed parent exposure: does composed AMMC-like augmentation improve over original and single transforms while reducing surface-feature dependence? Raw lexical remains a control. No test, no formal claim, no change to closed historical batches.\n',encoding='utf-8')
    (OUT/'git_head.txt').write_bytes(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT))
    write(OUT/'env.json',{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
        'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})
    splits={};controls={};control_scores={}
    for fold,held in folds.items():
        fit=[p for p in parents if p['label']==0 or p['generator'] not in held]
        ev={name:[r for r in rows if r['label']==0 or r['generator'] in held] for name,rows in [('clean',clean),('hardneg',hardneg)]}
        splits[fold]=(fit,ev);controls[fold]={};control_scores[fold]={}
        for arm in ARMS:
            train=[]
            for epoch in range(3):
                for p in fit:
                    q=dict(p); h=lookup[(arm,epoch,p['row_id'])]; q.update(code=codes[h],code_sha256=h); train.append(q)
            controls[fold][arm]={};control_scores[fold][arm]={}
            for view in (['raw','masked'] if arm=='composed' else ['raw']):
                print('CONTROL',fold,arm,view,len(train),flush=True)
                m,s=prior.lexical(train,ev,f'{fold}_{arm}',view)
                controls[fold][arm][view]=m;control_scores[fold][arm][view]=s
            write(OUT/'control_metrics.json',controls)
    assert torch.cuda.is_available(),'Local CUDA required'
    torch.set_num_threads(4);device=torch.device('cuda');torch.cuda.reset_peak_memory_stats()
    from transformers import T5EncoderModel
    cache=DATA/'encoder_cache.npz';cache_keys=sorted(codes)
    if cache.exists():
        x=np.load(cache,allow_pickle=False)
        assert x['keys'].tolist()==cache_keys and x['model_sha256'].item()==config['model_sha256']
        vectors={k:v for k,v in zip(cache_keys,x['vectors'])};info={'cache_reused':True,'unique_codes':len(cache_keys)}
    else:
        model=T5EncoderModel.from_pretrained(str(prior.MODEL),local_files_only=True,dtype=torch.float16).to(device).eval()
        for p in model.parameters():p.requires_grad_(False)
        vectors,info=encode_unique([{'code_sha256':h,'code':codes[h]} for h in cache_keys],make_tokenizer(),model,device,8,512)
        np.savez_compressed(cache,keys=np.asarray(cache_keys),vectors=np.stack([vectors[h] for h in cache_keys]),model_sha256=config['model_sha256'])
        del model;torch.cuda.empty_cache()
    write(OUT/'encoding.json',info)
    source_map={g:i for i,g in enumerate(gens)}
    def matrix(hashes):return torch.tensor(np.stack([vectors[h] for h in hashes]),device=device)
    def fit_head(base,arm,job,seed):
        set_seed(seed);head=SharedHeads(len(next(iter(vectors.values()))),len(gens)).to(device)
        opt=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=1e-4)
        y=torch.tensor([r['label'] for r in base],device=device,dtype=torch.float32)
        s=torch.tensor([source_map.get(r['generator'],-1) for r in base],device=device)
        orig=matrix([r['code_sha256'] for r in base]); history=[]
        for epoch in range(3):
            hashes=[lookup[(arm,epoch,r['row_id'])] for r in base]
            x=matrix(hashes); changed=torch.tensor([h!=r['code_sha256'] for h,r in zip(hashes,base)],device=device)
            order=np.random.permutation(len(base));head.train();losses=[]
            for start in range(0,len(base),256):
                idx=torch.as_tensor(order[start:start+256],device=device);det,src,_=head(x[idx]);keep=s[idx]>=0
                if job=='source_only':loss=F.cross_entropy(src,s[idx])
                else:
                    loss=F.binary_cross_entropy_with_logits(det,y[idx])
                    if job!='detection_only' and bool(keep.any()):loss=loss+.2*F.cross_entropy(src[keep],s[idx][keep])
                    ci=idx[changed[idx]]
                    if job=='joint_invariance' and len(ci):
                        # Evaluation mode removes dropout noise from paired consistency.
                        head.eval();left=head(x[ci])[0].sigmoid();right=head(orig[ci])[0].sigmoid();head.train()
                        loss=loss+.1*F.mse_loss(left,right)
                opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
            history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'parent_exposures':len(base),'changed_exposures':int(changed.sum())})
        head.eval();return head,history
    results=[];task_scores={}
    for fold,(fit,ev) in splits.items():
        for arm in ARMS:
            for job in JOBS:
                for seed in SEEDS:
                    print('RUN',fold,arm,job,seed,flush=True)
                    head,hist=fit_head(fit,arm,job,seed);one={'fold':fold,'arm':arm,'job':job,'seed':seed,'history':hist,'eval':{}}
                    for name,rows in ev.items():
                        x=matrix([r['code_sha256'] for r in rows])
                        with torch.inference_mode(): scores=torch.cat([head(c)[0].sigmoid() for c in x.split(256)]).cpu().numpy()
                        one['eval'][name]={**prior.summary(rows,scores,.5),'score_file':prior.save_scores(f'{fold}_{arm}_{job}_{seed}_{name}',rows,scores)}
                        task_scores[(fold,arm,job,seed,name)]=prior.task_aucs(rows,scores)
                    results.append(one);write(OUT/'neural_metrics.json',results);del head
    source_runs=[];base_ai=[r for r in parents if r['label']==1]
    for arm in ARMS:
        for seed in SEEDS:
            print('SOURCE',arm,seed,flush=True)
            head,hist=fit_head(base_ai,arm,'source_only',seed);one={'arm':arm,'seed':seed,'history':hist,'eval':{}}
            for name,rows in [('clean',clean),('hardneg',hardneg)]:
                rows=[r for r in rows if r['label']==1];x=matrix([r['code_sha256'] for r in rows])
                with torch.inference_mode():logits=torch.cat([head(c)[1] for c in x.split(256)]).cpu().numpy()
                labels=np.asarray([source_map[r['generator']] for r in rows])
                one['eval'][name]={'accuracy':float((logits.argmax(1)==labels).mean()),'chance':1/7,'rows':len(rows),
                    'score_file':prior.save_scores(f'source_{arm}_{seed}_{name}',rows,logits,source_labels=labels)}
            source_runs.append(one);write(OUT/'source_metrics.json',source_runs);del head
    comparisons={}
    for arm in ARMS:
        comparisons[arm]={}
        for job in JOBS:
            entries=[]
            for fold,(_,ev) in splits.items():
                for name,rows in ev.items():
                    ts=[task_scores[(fold,arm,job,seed,name)] for seed in SEEDS]
                    raw=prior.task_aucs(rows,control_scores[fold][arm]['raw'][name])
                    original=[task_scores[(fold,'original',job,seed,name)] for seed in SEEDS]
                    avg={k:float(np.mean([r[k] for r in original])) for k in original[0]}
                    entries.append({'fold':fold,'view':name,'vs_raw_lexical':prior.paired(ts,raw,SEEDS[0]),'vs_original':prior.paired(ts,avg,SEEDS[0])})
            comparisons[arm][job]=entries
    passed={}
    for arm in ['single','composed']:
        passed[arm]={job:all(e[b]['delta']>=.01 and e[b]['task_bootstrap_ci95'][0]>0 for e in comparisons[arm][job] for b in ['vs_raw_lexical','vs_original']) for job in JOBS}
    decision='augmentation_candidate_requires_independent_validation' if any(v for a in passed.values() for v in a.values()) else 'augmentation_increment_not_established'
    metrics={'status':'exploratory_completed','gate':'revise_data','decision':decision,'detection_runs':len(results),
        'unique_source_runs':len(source_runs),'comparisons':comparisons,'success':passed,
        'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2}
    write(OUT/'metrics.json',metrics)
    lines=['# H3 v5 固定 parent 预算的 AMMC 类增强对照','','探索性 train/dev；v4 曾使用 dev 选样；不读取 test，不声称 project-heldout。',
           '所有 arm 每个 parent 每个 epoch 恰好曝光一次；失败回退原视图；dev 未增强。','','| fold | eval | arm | raw lexical tm | detection tm | joint tm | invariance tm |','|---|---|---|---:|---:|---:|---:|']
    for fold,(_,ev) in splits.items():
        for name in ev:
            for arm in ARMS:
                vals=[next(e['vs_raw_lexical']['mean_seed_task_macro_auroc'] for e in comparisons[arm][job] if e['fold']==fold and e['view']==name) for job in JOBS]
                ctrl=controls[fold][arm]['raw'][name]['task_macro_auroc']
                lines.append(f'| {fold} | {name} | {arm} | {ctrl:.4f} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} |')
    lines+=['',f'结论：`{decision}`。完整 task 成对 bootstrap CI 见 metrics.json。',
        '此实验只检验冻结表示上的增强效果；编码器未训练，不能据此排除 AMMC 在端到端微调上的作用。',
        '未做 dev 变换压力测试，因此不能把原视图指标称为词法不变性证据。',
        '词法控制使用全文，CodeT5 使用长度上限，结果只解释当前实例化；source-only 是已见 generator/task-heldout，不是未知 generator 归因。']
    (OUT/'report.md').write_bytes(('\n'.join(lines)+'\n').encode())
    (OUT/'commands.txt').write_bytes(b'python -u scripts/h3_ammc_v5_local_batch.py\n')
    (OUT/'SHA256SUMS.txt').write_bytes(''.join(f'{prior.digest(p)}  {p.relative_to(OUT).as_posix()}\n' for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['SHA256SUMS.txt','run.log.txt']).encode())
    print(json.dumps({'decision':decision,'runs':len(results)+len(source_runs),'peak_cuda_mib':metrics['peak_cuda_mib']}),flush=True)

if __name__=='__main__':main()
