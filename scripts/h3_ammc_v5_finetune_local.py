#!/usr/bin/env python3
"""Local low-VRAM end-to-end AMMC feasibility run.

Only the final two CodeT5 encoder blocks and a detection head are trainable.
The parent exposure schedule is shared with h3_ammc_v5_local_batch.py; no test
payloads or generated code are read. This is exploratory, not a formal gate.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, subprocess, sys, time
os.environ.setdefault('OMP_NUM_THREADS','4'); os.environ.setdefault('MKL_NUM_THREADS','4'); os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from h3_gpu_diagnostic_batch_2026_10_11 import make_tokenizer, head_tail, set_seed
import h3_v4_local_experiment as prior

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'d-det/data/h3_ammc_mix_v5'; OUT=ROOT/'d-det/artifacts/h3_ammc_v5_local_finetune_2026-10-11'
SEEDS=[20261011]; ARMS=['original','composed']; MAX_LEN=512; BATCH=2; ACCUM=8

def read(p): return [json.loads(x) for x in p.open(encoding='utf8') if x.strip()]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def metric(rows,s):
    if not np.isfinite(s).all(): raise ValueError('non-finite evaluation score')
    y=np.asarray([r['label'] for r in rows]); groups=defaultdict(list)
    for i,r in enumerate(rows): groups[r['task_id']].append(i)
    task={t:float(roc_auc_score(y[ix],np.asarray(s)[ix])) for t,ix in groups.items() if len(set(y[ix]))==2}
    return {'rows':len(rows),'row_auroc':float(roc_auc_score(y,s)),'task_macro_auroc':float(np.mean(list(task.values()))),'balanced_accuracy':float(balanced_accuracy_score(y,np.asarray(s)>=.5))}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('local CUDA required')
    from transformers import T5EncoderModel
    OUT.mkdir(parents=True,exist_ok=True); set_seed(SEEDS[0]); torch.set_num_threads(4); device=torch.device('cuda')
    ptrain=ROOT/'d-det/data/h3_stacad_revision_v4/hardneg_train_v4.jsonl'; phard=ROOT/'d-det/data/h3_stacad_revision_v4/hardneg_dev_v4.jsonl'; pclean=ROOT/'d-det/data/h3_stacad_revision_v1/dev_clean.jsonl'
    parents=sorted(read(ptrain),key=lambda r:r['row_id']); hard=read(phard); clean=read(pclean); sched=read(DATA/'train_schedule.jsonl'); views=read(DATA/'unique_train_views.jsonl')
    codes={r['code_sha256']:r['code'] for r in parents+views+clean}; lookup={(r['arm'],r['epoch'],r['parent_row_id']):r['code_sha256'] for r in sched}
    gens=sorted({r['generator'] for r in parents if r['label']==1}); folds={'fold_0':gens[:3],'fold_1':gens[3:]}
    if args.smoke: folds={'fold_0':folds['fold_0']}; ARMS_RUN=['composed']; epochs=1; max_steps=20
    else: ARMS_RUN=ARMS; epochs=2; max_steps=None
    tokenizer=make_tokenizer(); all_metrics=[]; all_scores={}; torch.cuda.reset_peak_memory_stats()
    def batch_encode(rows, hashes):
        toks=[head_tail(tokenizer.encode(codes[h],add_special_tokens=False),MAX_LEN) for h in hashes]; width=max(map(len,toks),default=1)
        ids=torch.full((len(toks),width),int(tokenizer.pad_token_id),dtype=torch.long,device=device); mask=torch.zeros_like(ids)
        for i,t in enumerate(toks):
            if t: ids[i,:len(t)]=torch.tensor(t,device=device); mask[i,:len(t)]=1
        return ids,mask
    for fold,held in folds.items():
        base=[r for r in parents if r['label']==0 or r['generator'] not in held]
        evals={'clean':[r for r in clean if r['label']==0 or r['generator'] in held],
               'hardneg':[r for r in hard if r['label']==0 or r['generator'] in held]}
        for arm in ARMS_RUN:
            for seed in SEEDS:
                set_seed(seed); model=T5EncoderModel.from_pretrained(str(ROOT/'d-det/models/codet5-small'),local_files_only=True).to(device)
                for p in model.parameters(): p.requires_grad_(False)
                blocks=list(model.encoder.block)
                for block in blocks[-2:]:
                    for p in block.parameters(): p.requires_grad_(True)
                if hasattr(model.encoder,'final_layer_norm'):
                    for p in model.encoder.final_layer_norm.parameters(): p.requires_grad_(True)
                head=nn.Sequential(nn.LayerNorm(model.config.d_model),nn.Linear(model.config.d_model,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,1)).to(device)
                params=[p for p in list(model.parameters())+list(head.parameters()) if p.requires_grad]
                opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-4)
                scaler=torch.cuda.amp.GradScaler(enabled=True)
                losses=[]; steps=0
                for epoch in range(epochs):
                    order=np.arange(len(base)); np.random.default_rng(seed+epoch).shuffle(order); opt.zero_grad(set_to_none=True)
                    for start in range(0,len(base),BATCH):
                        idx=order[start:start+BATCH]; rows=[base[int(i)] for i in idx]; hashes=[lookup[(arm,epoch,r['row_id'])] for r in rows]
                        ids,mask=batch_encode(rows,hashes)
                        with torch.autocast(device_type='cuda',dtype=torch.float16):
                            hidden=model(input_ids=ids,attention_mask=mask,return_dict=True).last_hidden_state
                            z=(hidden.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1).float(); logits=head(z).squeeze(-1)
                            loss=F.binary_cross_entropy_with_logits(logits,torch.tensor([r['label'] for r in rows],device=device,dtype=torch.float32))/ACCUM
                        scaler.scale(loss).backward(); losses.append(float(loss.detach())*ACCUM)
                        if ((start//BATCH)+1)%ACCUM==0 or start+BATCH>=len(base):
                            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params,1.0)
                            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); steps+=1
                        if args.smoke and steps>=max_steps: break
                    if args.smoke and steps>=max_steps: break
                run={'fold':fold,'arm':arm,'seed':seed,'train_rows':len(base),'epochs':epochs,'optimizer_steps':steps,'loss_mean':float(np.mean(losses)),'eval':{}}
                model.eval();head.eval()
                for name,rows in evals.items():
                    scores=[]
                    for start in range(0,len(rows),BATCH):
                        chunk=rows[start:start+BATCH]; ids,mask=batch_encode(chunk,[r['code_sha256'] for r in chunk])
                        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16):
                            hidden=model(input_ids=ids,attention_mask=mask,return_dict=True).last_hidden_state; z=(hidden.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1).float(); scores.extend(head(z).squeeze(-1).sigmoid().float().cpu().tolist())
                    run['eval'][name]=metric(rows,scores); np.savez_compressed(OUT/f'{fold}_{arm}_{seed}_{name}.npz',scores=np.asarray(scores),labels=np.asarray([r['label'] for r in rows]),row_ids=np.asarray([r['row_id'] for r in rows]),task_ids=np.asarray([r['task_id'] for r in rows]))
                all_metrics.append(run); write(OUT/'metrics.json',{'status':'running','runs':all_metrics,'smoke':args.smoke}); del model,head;torch.cuda.empty_cache()
                if args.smoke: break
    result={'status':'smoke_completed' if args.smoke else 'completed','runs':len(all_metrics),'runs_detail':all_metrics,'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2,'formal_gpu_ready':False,'gate':'revise_data','test_read':False,'generation':False,'code_execution':False}
    write(OUT/'metrics.json',result); write(OUT/'config.json',{'script_sha256':digest(Path(__file__)),'inputs':{k:digest(v) for k,v in {'train':ptrain,'hardneg':phard,'clean':pclean,'schedule':DATA/'train_schedule.jsonl'}.items()},'arms':ARMS_RUN,'folds':folds,'epochs':epochs,'batch':BATCH,'accum':ACCUM,'unfrozen':'last two CodeT5 encoder blocks + final layer norm','test_read':False,'smoke':args.smoke})
    (OUT/'env.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.cuda.get_device_name(0)},indent=2)+'\n')
    lines=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='SHA256SUMS.txt': lines.append(f'{digest(p)}  {p.relative_to(OUT).as_posix()}\n')
    (OUT/'SHA256SUMS.txt').write_text(''.join(lines),encoding='utf8'); print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
