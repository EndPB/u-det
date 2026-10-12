"""Historical AICD numeric external-control screening.

This script is retained for reproducibility only and is blocked from the d-det
ABC mainline. It evaluates AICD T1/T2/T3 numeric labels, which are not the
project's A/B/C task semantics and contain no task/family provenance. The real
mainline must use task-centred semantic residuals, AMMC train-only views, and
Human-inclusive family labels.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from pathlib import Path
if __name__ == '__main__' and (Path(__file__).resolve().parents[1]/'d-det/data/mainline_abc_200k_2026-10-12/ROLE_EXTERNAL_CONTROL.md').exists():
    raise SystemExit('WITHDRAWN_EXTERNAL_CONTROL: no training started; see d-det ABC route correction document.')
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from format_channel_local_audit_2026_10_11 import format_features

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'d-det/data/mainline_abc_200k_2026-10-12'
MODEL=ROOT/'d-det/models/codet5-small'
OUT=ROOT/'d-det/artifacts/mainline_abc_gpu_batch_2026-10-12'
TASKS={'A':2,'B':12,'C':4}; SEED=20261012; MAX_LEN=512

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_rows(task, split):
    compressed=DATA/task/f'{split}.jsonl.gz'
    if compressed.exists():
        import gzip
        with gzip.open(compressed,'rt',encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    raise RuntimeError('missing standard-library bundle: '+str(compressed))
def write(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def metric(y,p):
    pred=p.argmax(1); out={'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'per_id_recall':{str(k):float((pred[y==k]==k).mean()) for k in sorted(set(y))},'confusion_matrix_numeric':confusion_matrix(y,pred).tolist()}
    if p.shape[1]==2: out['auroc_numeric_label_1']=float(roc_auc_score(y,p[:,1]))
    return out
class Head(nn.Module):
    def __init__(self,d,n):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,n))
    def forward(self,x): return self.net(x)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=2); ap.add_argument('--batch',type=int,default=32); ap.add_argument('--max-rows',type=int,default=0); ap.add_argument('--stratified',action='store_true'); args=ap.parse_args()
    if (DATA/'ROLE_EXTERNAL_CONTROL.md').exists():
        raise RuntimeError('AICD external-control batch withdrawn from d-det ABC mainline; see route correction document. No training started.')
    if not torch.cuda.is_available(): raise RuntimeError('GPU required for this batch')
    torch.manual_seed(SEED); device=torch.device('cuda'); OUT.mkdir(parents=True,exist_ok=True)
    from transformers import T5EncoderModel,RobertaTokenizer
    tok=RobertaTokenizer(vocab=str(MODEL/'vocab.json'),merges=str(MODEL/'merges.txt'),unk_token='<unk>',bos_token='<s>',eos_token='</s>',sep_token='</s>',cls_token='<s>',pad_token='<pad>',mask_token='<mask>',add_prefix_space=False)
    enc=T5EncoderModel.from_pretrained(str(MODEL),local_files_only=True).to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8')); summary={'schema':'mainline_abc_gpu_batch_v1','status':'running','manifest_sha256':sha(DATA/'manifest.json'),'device':torch.cuda.get_device_name(0),'tasks':{}}
    write(OUT/'metrics.json',summary)
    for task,nclasses in TASKS.items():
        train=read_rows(task,'train'); val=read_rows(task,'validation')
        if args.max_rows:
            def take(rows, limit):
                if not args.stratified: return rows[:limit]
                by={k:[] for k in sorted({r['label'] for r in rows})}
                for r in rows: by[r['label']].append(r)
                per=max(1,limit//max(1,len(by)))
                out=[]
                for k in by: out.extend(by[k][:per])
                return out[:limit]
            train=take(train,args.max_rows); val=take(val,min(len(val),max(1000,args.max_rows//10)))
        def encode(rows):
            feats=[]
            with torch.inference_mode():
                for st in range(0,len(rows),args.batch):
                    text=[r['code'][:24000] for r in rows[st:st+args.batch]]
                    q=tok(text,padding=True,truncation=True,max_length=MAX_LEN,return_tensors='pt').to(device)
                    with torch.autocast(device_type='cuda',dtype=torch.float16):
                        h=enc(**q,return_dict=True).last_hidden_state
                    z=(h.float()*q['attention_mask'].unsqueeze(-1)).sum(1)/q['attention_mask'].sum(1,keepdim=True).clamp_min(1)
                    feats.append(z.float().cpu())
                    if st==0 or st+args.batch>=len(rows): print(f'ENCODE {task} {st+len(text)}/{len(rows)}',flush=True)
            return torch.cat(feats)
        xt,xv=encode(train),encode(val)
        ft=np.asarray([list(format_features(r['code']).values()) for r in train],dtype=np.float32); fv=np.asarray([list(format_features(r['code']).values()) for r in val],dtype=np.float32)
        ft=torch.tensor(np.log1p(ft),dtype=torch.float32); fv=torch.tensor(np.log1p(fv),dtype=torch.float32)
        mu,sd=ft.mean(0),ft.std(0).clamp_min(1e-6); ft=(ft-mu)/sd; fv=(fv-mu)/sd
        y=torch.tensor([r['label'] for r in train],device=device); yv=np.asarray([r['label'] for r in val])
        arms={'codet5_frozen':(xt,xv),'codet5_layout':(torch.cat([xt,.05*ft],1),torch.cat([xv,.05*fv],1))}
        summary['tasks'][task]={'train_rows':len(train),'validation_rows':len(val),'arms':{}}
        for name,(a,b) in arms.items():
            head=Head(a.shape[1],nclasses).to(device); opt=torch.optim.AdamW(head.parameters(),lr=2e-4,weight_decay=1e-4); start=time.time(); a=a.to(device); b=b.to(device)
            for epoch in range(args.epochs):
                order=torch.randperm(len(y),device=device)
                for st in range(0,len(order),args.batch):
                    ix=order[st:st+args.batch]; loss=nn.functional.cross_entropy(head(a[ix]),y[ix]); opt.zero_grad(); loss.backward(); opt.step()
            with torch.inference_mode(): p=torch.softmax(head(b),1).cpu().numpy()
            summary['tasks'][task]['arms'][name]=dict(metric(yv,p),seconds=time.time()-start,epochs=args.epochs)
            np.savez_compressed(OUT/f'{task}_{name}_scores.npz',p=p,y=yv,sha=np.array([r['code_sha256'] for r in val]))
            print(f'RESULT {task} {name} macro-F1={summary["tasks"][task]["arms"][name]["macro_f1"]:.4f}',flush=True)
        write(OUT/'metrics.json',summary)
    summary['status']='completed'; summary['script_sha256']=sha(Path(__file__)); summary['test_opened']=False; summary['generation']=False; summary['code_execution']=False; write(OUT/'metrics.json',summary)
if __name__=='__main__': main()
