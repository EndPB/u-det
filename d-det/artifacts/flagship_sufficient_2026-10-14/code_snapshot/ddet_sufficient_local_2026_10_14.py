"""Full eligible flagship pool, monitored training, and deployment-safe references."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,copy,gzip,json,math,re,time
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,f1_score
import ddet_two_tasks_local_2026_10_14 as b
import ddet_flagship_local_2026_10_14 as d
from ddet_flagship_clean_local_2026_10_14 import no_blank_placeholders
from ddet_t5_local_fast_2026_10_14 import install

ROOT=b.ROOT;OUT=ROOT/'d-det/artifacts/flagship_sufficient_2026-10-14';DATA=ROOT/'d-det/data/flagship_sufficient_2026-10-14'
PARENT=ROOT/'d-det/artifacts/flagship_two_tasks_clean_2026-10-14/broad'
HELD=set(json.loads((PARENT/'protocol.json').read_text())['heldout_model_ids'])
MAX_EPOCH=12;MIN_EPOCH=4;PATIENCE=3;MICRO=16;EFFECTIVE=32


def setup():
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    (DATA/'.gitignore').write_text('*\n!.gitignore\n!README.md\n',encoding='utf-8')
    (OUT/'.gitignore').write_text('*_features.npz\n*_reference.npz\n*.receiving\nWORK_STATE.md\n',encoding='utf-8')
    b.OUT=OUT;b.DATA=DATA;b.MAXLEN=512;b.weights=d.task_weights
    torch.set_num_threads(4)


def prepare():
    if (OUT/'protocol.json').exists():
        p=json.loads((OUT/'protocol.json').read_text())
        if 'task_count' not in p:
            p['objectives']=p.pop('tasks');p['task_count']=len({r['task_cluster_id'] for r in b.readgz(DATA/'pilot.jsonl.gz')})
            p['source_code_sha256']=b.digest(Path(__file__));b.save(OUT/'protocol.json',p)
        return
    pool=list(b.readgz(ROOT/'d-det/data/flagship_two_tasks_local_2026-10-14/eligible_pool.jsonl.gz'))
    parent={r['task_cluster_id']:r['task_cluster_id'] for r in pool}
    def find(t):
        while parent[t]!=t:parent[t]=parent[parent[t]];t=parent[t]
        return t
    seen={}
    for r in pool:
        for key in [r['pilot_code_sha256'],b.sh(re.sub(r'\s+','',r['code']))]:
            if key in seen:
                x,y=find(r['task_cluster_id']),find(seen[key]);parent[max(x,y)]=min(x,y)
            seen[key]=r['task_cluster_id']
    anchor={}
    for r in b.readgz(ROOT/'d-det/data/flagship_two_tasks_clean_2026-10-14/broad/pilot.jsonl.gz'):
        g=find(r['task_cluster_id']);split='dev' if r['split']=='unseen_generator' else r['split']
        assert g not in anchor or anchor[g]==split;anchor[g]=split
    external={find(r['task_cluster_id']) for r in pool if r['source_dataset']==d.NEW_SOURCE or str(r['task_cluster_id']).startswith('public:HumanEval/')}
    for g in external:assert anchor.get(g,'external')=='external';anchor[g]='external'
    strata=defaultdict(set)
    for r in pool:
        g=find(r['task_cluster_id'])
        if g not in anchor:strata['human' if r['family']=='Human' else 'ai'].add(g)
    # New groups are assigned once, independently of scores; inherited groups keep their roles.
    remaining=sorted(set().union(*strata.values()),key=lambda x:b.sh(str(b.SEED)+x))
    for i,g in enumerate(remaining):anchor[g]='train' if i<.7*len(remaining) else 'calibration' if i<.8*len(remaining) else 'dev'
    clean_hash_families=defaultdict(set)
    for i,r in enumerate(pool):
        r['code']=no_blank_placeholders(r['code']);r['pilot_code_sha256']=b.sh(r['code'])
        clean_hash_families[r['pilot_code_sha256']].add(r['family'])
        if i%10000==0:b.log('FULL_PREP_CLEAN',i,'/',len(pool))
    ambiguous={h for h,families in clean_hash_families.items() if len(families)>1}
    rows=[];reserved=[];excluded=Counter()
    for i,r in enumerate(pool):
        if r['pilot_code_sha256'] in ambiguous:excluded['identical_clean_code_multiple_families']+=1;continue
        g=find(r['task_cluster_id']);split=anchor[g]
        if r['model_id'] in HELD and split in {'train','calibration'}:
            reserved.append({'record_id':r['record_id'],'task_cluster_id':r['task_cluster_id'],'group':g,'reason':'heldout_version_on_learning_task'});continue
        if r['model_id'] in HELD and split=='dev':split='unseen_generator'
        r.update(split=split,leakage_group=g,label_F=b.CLASSES.index(r['family']),label_D=int(r['family']!='Human'))
        r['content_view']=b.content_view(r['code']);r['train_view']=r['code']
        r['stress_view']=b.augment(r['code'],r['record_id']+':stress',force=True)[0] if split!='train' else r['code']
        rows.append(r)
        if i%10000==0:b.log('FULL_PREP_VIEWS',i,'/',len(pool))
    rows.sort(key=lambda r:(r['split'],r['task_cluster_id'],r['family'],r['record_id']))
    groups={s:{r['leakage_group'] for r in rows if r['split']==s} for s in ['train','calibration','dev','unseen_generator','external']}
    for first in ['train','calibration']:
        for second in groups:
            if first!=second:assert not groups[first]&groups[second]
    with gzip.open(DATA/'pilot.jsonl.gz','wt',encoding='utf-8',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    b.save(OUT/'reserved_and_excluded.json',{'excluded':dict(excluded),'reserved_count':len(reserved),'reserved':reserved})
    protocol={'status':'frozen_before_training','eligible_pool':len(pool),'rows':len(rows),'task_count':len({r['task_cluster_id'] for r in rows}),
              'counts':{s:dict(Counter(r['family'] for r in rows if r['split']==s)) for s in groups},'heldout_model_ids':sorted(HELD),
              'parent_protocol_sha256':b.digest(PARENT/'protocol.json'),'source_main_sha256':b.digest(b.MAIN/'samples.jsonl.gz'),
              'data_sha256':b.digest(DATA/'pilot.jsonl.gz'),'max_epochs':MAX_EPOCH,'min_epochs':MIN_EPOCH,'patience':PATIENCE,
              'meaningful_calibration_delta':{'F':.002,'D':.0005},'encoder_lr':3e-5,'head_lr':1e-3,'effective_batch':EFFECTIVE,
              'micro_batch':MICRO,'length':512,'truncation':'head384+tail128','augmentation':'B1 original .5, otherwise comment .2/whitespace .3/safe rename .4, refreshed per epoch',
              'selection':'calibration only; maximum score checkpoint; plateau tracked separately; no evaluation-set model selection',
              'early_stop_scope':'validation saturation, not proof of global optimum','seed':b.SEED,'encoder':'CodeT5-small full encoder',
              'objectives':['Human_AI_detection','Human_inclusive_family_attribution'],'historical_test_read':False,'api_generation':False,
              'reference_teacher':'train tasks with Human and >=2 AI families; all other train codes still used for classification',
              'base_similarity':'working hypothesis; no causal base-to-post labels claimed','source_code_sha256':b.digest(Path(__file__))}
    b.save(OUT/'protocol.json',protocol)
    (DATA/'README.md').write_text('旗舰来源全可用池的训练与解释实验副本；原主集不改动。训练/校准/评估按题目及重复内容簇隔离。保留模型版本的学习题目输出单独封存，不计入独立泛化评估。\n',encoding='utf-8')
    b.log('FULL_READY',protocol['rows'],protocol['task_count'],protocol['counts'])


def encode(model,head,rows,cache,field='code',indices=None,batch=32):
    indices=np.arange(len(rows)) if indices is None else np.array(indices)
    order=sorted(range(len(indices)),key=lambda j:len(cache[b.sh(rows[indices[j]][field])]))
    h=np.empty((len(indices),512),dtype='float32');z=np.empty((len(indices),7 if head is not None and head[-1].out_features==7 else 1),dtype='float32') if head is not None else None
    model.eval()
    if head is not None:head.eval()
    with torch.inference_mode():
        for start in range(0,len(order),batch):
            js=order[start:start+batch];chunk=[rows[indices[j]] for j in js];ids,mask=b.batch_tokens(chunk,cache,field,'cuda')
            with torch.autocast('cuda',dtype=torch.float16):hh=b.pooled(model,ids,mask);zz=head(hh) if head is not None else None
            h[js]=hh.float().cpu().numpy()
            if z is not None:z[js]=zz.float().cpu().numpy()
    assert np.isfinite(h).all()
    return h,z


def quick_metric(rows,logits,task):
    y=np.array([r['label_'+task] for r in rows]);return float(roc_auc_score(y,logits.ravel()) if task=='D' else f1_score(y,logits.argmax(1),labels=range(7),average='macro',zero_division=0))


def epoch_view(rows,cache,tok,epoch,arm):
    tr=[i for i,r in enumerate(rows) if r['split']=='train'];texts={};statuses=Counter()
    for i in tr:
        code,status=b.augment(rows[i]['code'],rows[i]['record_id']+f':epoch{epoch}') if arm=='B1' else (rows[i]['code'],[])
        texts[i]=code;statuses.update(status);key=b.sh(code)
        if key not in cache:
            ids=tok.encode(code,add_special_tokens=False);cache[key]=ids if len(ids)<=512 else ids[:384]+ids[-128:]
    return tr,texts,statuses


def train(rows,cache,tok,task,arm):
    from transformers import T5EncoderModel
    path=DATA/f'{task}_{arm}_best.pt';summary=OUT/f'{task}_{arm}_training.json'
    if summary.exists() and json.loads(summary.read_text()).get('completed'):return
    b.seed_all(b.SEED);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    model=install(T5EncoderModel.from_pretrained(b.MODEL,local_files_only=True)).cuda();model.gradient_checkpointing_enable()
    head=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,7 if task=='F' else 1)).cuda()
    opt=torch.optim.AdamW([{'params':model.parameters(),'lr':3e-5},{'params':head.parameters(),'lr':1e-3}],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');tr=np.array([i for i,r in enumerate(rows) if r['split']=='train']);cal=np.array([i for i,r in enumerate(rows) if r['split']=='calibration'])
    calrows=[rows[i] for i in cal];weights=d.task_weights([rows[i] for i in tr],task);wm=dict(zip(tr,weights))
    history=[];best=-np.inf;anchor=-np.inf;stale=0;selected=0;start_epoch=1;started=time.time()
    resume=DATA/f'{task}_{arm}_resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False);model.load_state_dict(state['encoder']);head.load_state_dict(state['head']);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        history=state['history'];best=state['best'];anchor=state['anchor'];stale=state['stale'];selected=state['selected'];start_epoch=state['epoch']+1
        for value in opt.state.values():
            for key,item in value.items():
                if torch.is_tensor(item):value[key]=item.cuda()
        b.log('RESUME',task,arm,start_epoch)
    for epoch in range(start_epoch,MAX_EPOCH+1):
        epoch_start=time.time();b.seed_all(b.SEED+epoch)
        train_ids,texts,status=epoch_view(rows,cache,tok,epoch,arm)
        order=np.random.default_rng(b.SEED+epoch).permutation(train_ids).tolist()
        # Shuffle blocks, then bucket by length inside each block; all samples appear once.
        order=[i for start in range(0,len(order),256) for i in sorted(order[start:start+256],key=lambda j:len(cache[b.sh(texts[j])]))]
        model.train();head.train();opt.zero_grad(set_to_none=True);loss_sum=0.;seen=0;updates=0
        for start in range(0,len(order),MICRO):
            ix=order[start:start+MICRO];chunk=[dict(rows[i],epoch_view=texts[i]) for i in ix];ids,mask=b.batch_tokens(chunk,cache,'epoch_view','cuda')
            y=torch.tensor([r['label_'+task] for r in chunk],dtype=torch.long,device='cuda');w=torch.tensor([wm[i] for i in ix],device='cuda')
            group_start=(start//EFFECTIVE)*EFFECTIVE;denom=min(EFFECTIVE,len(order)-group_start)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=head(b.pooled(model,ids,mask))
                individual=torch.nn.functional.cross_entropy(logits,y,reduction='none') if task=='F' else torch.nn.functional.binary_cross_entropy_with_logits(logits.ravel(),y.float(),reduction='none')
                weighted=(individual*w).sum();loss=weighted/denom
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            scaler.scale(loss).backward();loss_sum+=float(weighted.detach());seen+=len(ix)
            if (start+MICRO)%EFFECTIVE==0 or start+MICRO>=len(order):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1.);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);updates+=1
            if start%(MICRO*200)==0:b.log('TRAIN',task,arm,'epoch',epoch,start,'/',len(order),'sec',round(time.time()-epoch_start))
        _,logits=encode(model,head,rows,cache,indices=cal)
        score=quick_metric(calrows,logits,task)
        if score>best:
            best=score;selected=epoch;torch.save({'encoder':model.state_dict(),'head':head.state_dict(),'epoch':epoch},path)
        delta=.002 if task=='F' else .0005
        if score>anchor+delta:anchor=score;stale=0
        else:stale+=1
        history.append({'epoch':epoch,'weighted_train_loss':loss_sum/seen,'calibration_metric':score,'seconds':time.time()-epoch_start,'optimizer_steps':updates,'augmentation':dict(status),'selected_epoch_so_far':selected})
        torch.save({'encoder':model.state_dict(),'head':head.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'history':history,'best':best,'anchor':anchor,'stale':stale,'selected':selected},resume)
        b.save(summary,{'completed':False,'task':task,'arm':arm,'history':history,'selected_epoch':selected,'best_calibration_metric':best})
        b.log('EPOCH_DONE',task,arm,epoch,'cal',round(score,5),'best_epoch',selected,'stale',stale)
        if epoch>=MIN_EPOCH and stale>=PATIENCE:break
    saturated=stale>=PATIENCE
    b.save(summary,{'completed':True,'task':task,'arm':arm,'history':history,'selected_epoch':selected,'best_calibration_metric':best,
                   'validation_saturated':saturated,'stop_reason':'calibration_plateau' if saturated else 'epoch_cap_not_confirmed_saturated',
                   'seconds_this_invocation':time.time()-started,'peak_cuda_mib':torch.cuda.max_memory_allocated()/1048576})
    del model,head,opt,scaler;torch.cuda.empty_cache()


def load_model(task,arm):
    from transformers import T5EncoderModel
    model=install(T5EncoderModel.from_pretrained(b.MODEL,local_files_only=True));head=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,7 if task=='F' else 1))
    state=torch.load(DATA/f'{task}_{arm}_best.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['encoder']);head.load_state_dict(state['head'])
    return model.cuda(),head.cuda()


def features(rows,cache,task,arm):
    path=OUT/f'{task}_{arm}_features.npz'
    if path.exists():return
    model,head=load_model(task,arm);h,base=encode(model,head,rows,cache)
    ix=np.array([i for i,r in enumerate(rows) if r['split']!='train']);hs=h.copy();stress=base.copy()
    hs[ix],stress[ix]=encode(model,head,rows,cache,'stress_view',indices=ix)
    np.savez_compressed(path,h=h,base=base,hs=hs,stress=stress,row_ids=np.array([r['record_id'] for r in rows]))
    del model,head;torch.cuda.empty_cache();b.log('FEATURES_DONE',task,arm,len(rows))


def semantic(rows,cache):
    path=OUT/'semantic_features.npz'
    if path.exists():return
    from transformers import T5EncoderModel
    model=install(T5EncoderModel.from_pretrained(b.MODEL,local_files_only=True)).cuda();sem,_=encode(model,None,rows,cache,'content_view')
    ix=np.array([i for i,r in enumerate(rows) if r['split']!='train']);rr=[dict(r,stress_content=b.content_view(r['stress_view'])) for r in rows]
    tok=b.tokenizer()
    for i in ix:
        key=b.sh(rr[i]['stress_content'])
        if key not in cache:
            ids=tok.encode(rr[i]['stress_content'],add_special_tokens=False);cache[key]=ids if len(ids)<=512 else ids[:384]+ids[-128:]
    ss=sem.copy();ss[ix],_=encode(model,None,rr,cache,'stress_content',indices=ix)
    np.savez_compressed(path,sem=sem,stress_sem=ss,row_ids=np.array([r['record_id'] for r in rows]));del model;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--task',choices=['F','D','both'],default='both');args=parser.parse_args()
    setup();prepare()
    if args.prepare_only:return
    rows=list(b.readgz(DATA/'pilot.jsonl.gz'));tok=b.tokenizer();cache,_=d.token_cache(rows,tok)
    tasks=['F','D'] if args.task=='both' else [args.task]
    for task in tasks:
        for arm in ['B0','B1']:train(rows,cache,tok,task,arm)
    # Evaluation happens only after checkpoint selection is finished.
    semantic(rows,cache)
    for task in tasks:
        for arm in ['B0','B1']:features(rows,cache,task,arm)
    b.log('SUFFICIENT_ENCODERS_AND_FEATURES_DONE')


if __name__=='__main__':main()
