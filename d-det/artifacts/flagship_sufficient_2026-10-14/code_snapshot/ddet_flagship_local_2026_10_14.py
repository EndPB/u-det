"""Flagship-only continuation: controlled and broader local two-task pilots."""
from __future__ import annotations
import argparse, ast, copy, gzip, hashlib, json, re, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
import ddet_two_tasks_local_2026_10_14 as b

ROOT=b.ROOT
BASE_DATA=ROOT/'d-det/data/flagship_two_tasks_local_2026-10-14'
BASE_OUT=ROOT/'d-det/artifacts/flagship_two_tasks_local_2026-10-14'
CLASSES=b.CLASSES
REGISTRY={
 'OpenAI': {'gpt-4','gpt-4-0613','gpt-4-turbo-2024-04-09','gpt-4o','gpt-4o-2024-08-06',
            'gpt-4o-2024-11-20--main','chatgpt-4o-latest--main','gpt-4.5-preview--main',
            'o1-2024-12-17--main','o1-2024-12-17--low--main','o1-2024-12-17--high--main','o1-preview-2024-09-12','gpt-5.4'},
 'Claude': {'claude-3-opus-20240229','claude-3-5-sonnet-20240620','claude-3-5-sonnet-20241022',
            'claude-3-7-sonnet-20250219--main','claude-3-7-sonnet-20250219--3200-output-128k-2025-02-19--main','claude-opus-4-6'},
 'Gemini': {'gemini-1.5-pro','gemini-1.5-pro-002','gemini-1.5-pro-exp-0801','gemini-1.5-pro-exp-0827',
            'gemini-2.0-pro-exp-02-05--main','Gemini-Pro-1.5','gemini-3.1-pro-preview'},
 'DeepSeek': {'deepseek-ai--DeepSeek-V3','deepseek-ai--DeepSeek-R1','deepseek-v3-2-251201'},
 'Qwen': {'qwen-max-latest--main'},
 'Grok': {'grok-beta--main'},
}
HELDOUT={'gpt-4o-2024-11-20--main','claude-3-5-sonnet-20241022','deepseek-ai--DeepSeek-R1'}
BCC={'BigCodeBench','public_same_task_full_2026-10-07'}
NEW_SOURCE='Zenodo:20823175'


def configure(name):
    b.DATA=BASE_DATA/name;b.OUT=BASE_OUT/name;b.MAXLEN=512
    b.DATA.mkdir(parents=True,exist_ok=True);b.OUT.mkdir(parents=True,exist_ok=True)


def task_alias(r):
    tid=r.get('task_id') or r.get('source_task_id') or ''
    if str(tid).startswith('HumanEval/') and r['source_dataset'] in {'GitHub:jamesmurdza/humaneval-results',NEW_SOURCE}:
        return 'public:HumanEval/'+str(tid).split('/')[-1]
    return r['task_cluster_id']


def get_pool():
    BASE_DATA.mkdir(parents=True,exist_ok=True);BASE_OUT.mkdir(parents=True,exist_ok=True)
    cached=BASE_DATA/'eligible_pool.jsonl.gz'
    if cached.exists():return list(b.readgz(cached))
    metadata={};heprompts={}
    with (ROOT/'d-det/data/public_same_task_full_2026-10-07/bigcodebench_task_metadata.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);metadata[r['task_id']]=r
    for r in b.readgz(b.MAIN/'samples.jsonl.gz'):
        if r['source_dataset']=='GitHub:jamesmurdza/humaneval-results' and r.get('prompt_text'):
            heprompts[r.get('source_task_id')]=r['prompt_text']
    pool=[];reject=Counter();family_total=Counter();bycode=defaultdict(list)
    for r in b.readgz(b.MAIN/'samples.jsonl.gz'):
        family=r['family_label'];source=r['source_dataset']
        if family not in CLASSES:reject['family_outside_seven_class_pilot']+=1;continue
        if r['representation_type']!='code':reject['patch_not_complete_code']+=1;continue
        if r['label']=='Human':
            if 'realhumaneval' in source:reject['human_assistance_possible']+=1;continue
        elif r.get('model_id') not in REGISTRY[family]:reject['not_explicit_flagship_registry']+=1;continue
        if r.get('identical_code_from_multiple_families'):reject['raw_content_ambiguity']+=1;continue
        code=r['code'];reconstruction=None;tid=r.get('task_id') or r.get('source_task_id')
        if family=='Human' and source in BCC:
            info=metadata.get(tid)
            if not info or info['canonical_solution'].strip()!=code.strip():reject['bcc_reference_not_verified']+=1;continue
            code=info['code_prompt']+info['canonical_solution'];reconstruction='BCC_official_code_prompt_plus_matching_body'
        elif family=='Human' and source==NEW_SOURCE and str(tid).startswith('HumanEval/'):
            prompt=heprompts.get(tid)
            if not prompt:reject['HE_reference_missing_header']+=1;continue
            code=prompt+code;reconstruction='public_HumanEval_task_prompt_plus_source_reference_body'
        try:code=b.remove_docs(code)
        except (SyntaxError,ValueError,RecursionError):reject['not_parseable_python_implementation']+=1;continue
        # Avoid artificially overweighting duplicate protocol/publication records.
        row={k:r.get(k) for k in ['record_id','model_id','source_dataset','code_sha256','task_center_id','prompt_protocol_status']}
        row.update(code=code,family=family,task_id=tid,task_cluster_id=task_alias(r),pilot_code_sha256=b.sh(code),
                   human_reconstructed=reconstruction,model_registry_status='Human_reference' if family=='Human' else 'explicit_flagship_product_or_version')
        pool.append(row);family_total[family]+=1;bycode[row['pilot_code_sha256']].append(row)
    ambiguous={h for h,rs in bycode.items() if len({r['family'] for r in rs})>1}
    pool=[r for r in pool if r['pilot_code_sha256'] not in ambiguous]
    # Within identical source/family/model/task/input keep only one observation.
    dedup={}
    for r in pool:
        key=(r['task_cluster_id'],r['family'],r['model_id'],r['pilot_code_sha256'])
        if key not in dedup or r['record_id']<dedup[key]['record_id']:dedup[key]=r
    pool=list(dedup.values())
    with gzip.open(cached,'wt',encoding='utf-8',newline='\n') as f:
        for r in pool:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    registry={'allowed':{k:sorted(v) for k,v in REGISTRY.items()},'heldout_model_ids':sorted(HELDOUT),
              'excluded':'Qwen dense 72B/110B, CodeLlama, Coder small models, Mini/Haiku/Flash, ambiguous aliases',
              'historical_flagship_not_claimed_latest_2026':True,'qwen_max_exact_snapshot_unknown':True,
              'MoE_rule':'do not confuse active parameter count with total model size',
              'source_alias_audit':{'CoderEval gpt':'source gpt_model.py uses gpt-4o-mini: excluded',
                                    'CoderEval deepseek':'deepseek-chat exact snapshot not pinned: excluded',
                                    'CoderEval claude':'generic identifier excluded despite source script evidence, avoid silent row re-labeling'},
              'primary_sources':['https://qwenlm.github.io/blog/qwen2.5-max/','https://huggingface.co/deepseek-ai/DeepSeek-V3',
                                 'https://huggingface.co/deepseek-ai/DeepSeek-R1','https://www.anthropic.com/news/3-5-models-and-computer-use',
                                 'https://ai.google.dev/gemini-api/docs/changelog'],
              'eligible_rows':len(pool),'family_counts':dict(Counter(r['family'] for r in pool)),
              'eligible_tasks':len({r['task_cluster_id'] for r in pool}),'rejections':dict(reject),
              'source_main_sha256':b.digest(b.MAIN/'samples.jsonl.gz'),'pool_sha256':b.digest(cached)}
    b.save(BASE_OUT/'flagship_registry_and_pool.json',registry)
    b.log('FLAGSHIP_POOL',len(pool),registry['family_counts'])
    return pool


def choose_core(pool):
    candidates=defaultdict(list)
    for r in pool:
        if r['source_dataset'] in BCC and r['model_id'] not in HELDOUT:
            candidates[(r['task_cluster_id'],r['family'])].append(r)
    rows=[]
    for task in sorted({t for t,f in candidates if f=='Human'}):
        if any((task,f) not in candidates for f in ['Human','OpenAI','Claude','Gemini','DeepSeek']):continue
        for family in CLASSES:
            group=candidates.get((task,family),[])
            if not group:continue
            models=sorted({r['model_id'] or 'Human' for r in group})
            m=models[int(b.sh(task+family)[:12],16)%len(models)]
            rows.append(copy.deepcopy(min([r for r in group if (r['model_id'] or 'Human')==m],key=lambda r:b.sh(r['record_id']))))
    return rows


def select_broad(pool,core):
    selected={r['record_id']:copy.deepcopy(r) for r in core}
    for family in CLASSES:
        existing=sum(r['family']==family for r in selected.values())
        choices=[r for r in pool if r['family']==family and r['source_dataset']!=NEW_SOURCE and not str(r['task_cluster_id']).startswith('public:HumanEval/') and r['model_id'] not in HELDOUT]
        choices.sort(key=lambda r:b.sh(str(b.SEED)+r['record_id']))
        tasks=set(r['task_cluster_id'] for r in selected.values() if r['family']==family)
        for r in choices:
            if existing>=4000:break
            if r['record_id'] in selected or (family=='Human' and r['task_cluster_id'] in tasks):continue
            selected[r['record_id']]=copy.deepcopy(r);existing+=1;tasks.add(r['task_cluster_id'])
    # Add full heldout-version and external-source evaluation pools, never training them.
    for r in pool:
        if r['model_id'] in HELDOUT or r['source_dataset']==NEW_SOURCE:
            selected[r['record_id']]=copy.deepcopy(r)
    return list(selected.values())


def finish_data(rows,pool,name):
    configure(name)
    # Build leakage components over the eligible pool, including rows not selected for training.
    parent={r['task_cluster_id']:r['task_cluster_id'] for r in pool}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={}
    for r in pool:
        for key in [r['pilot_code_sha256'],b.sh(re.sub(r'\s+','',r['code']))]:
            if key in seen:
                a,c=find(r['task_cluster_id']),find(seen[key]);parent[max(a,c)]=min(a,c)
            seen[key]=r['task_cluster_id']
    external={find(r['task_cluster_id']) for r in pool if r['source_dataset']==NEW_SOURCE or str(r['task_cluster_id']).startswith('public:HumanEval/')}
    groups=sorted({find(r['task_cluster_id']) for r in rows}-external,key=lambda g:b.sh(str(b.SEED)+g))
    # Stratify components on availability of the two sparse flagship families.
    sparse=defaultdict(set)
    for r in rows:
        if r['family'] in {'Qwen','Grok'} and r['model_id'] not in HELDOUT:sparse[find(r['task_cluster_id'])].add(r['family'])
    strata=defaultdict(list)
    for group in groups:strata[tuple(sorted(sparse[group]))].append(group)
    split={}
    for group_list in strata.values():
        for i,g in enumerate(group_list):split[g]='train' if i<.7*len(group_list) else 'calibration' if i<.8*len(group_list) else 'dev'
    final=[];statuses=Counter()
    for r in rows:
        group=find(r['task_cluster_id']);partition='external' if group in external else split[group]
        if r['model_id'] in HELDOUT:
            if partition!='dev':continue
            partition='unseen_generator'
        r['leakage_group']=group;r['split']=partition
        r['label_F']=CLASSES.index(r['family']);r['label_D']=int(r['family']!='Human')
        r['train_view'],train_status=b.augment(r['code'],r['record_id']) if partition=='train' else (r['code'],[])
        r['stress_view'],stress_status=b.augment(r['code'],r['record_id']+':stress',force=True)
        r['train_transform_status']=train_status;r['stress_transform_status']=stress_status
        r['content_view']=b.content_view(r['code']);statuses.update(train_status);final.append(r)
    final.sort(key=lambda r:(r['split'],r['task_cluster_id'],r['family'],r['record_id']))
    # Dev and heldout-generator samples may share tasks, but neither shares tasks with learning/calibration.
    sets={s:{r['leakage_group'] for r in final if r['split']==s} for s in ['train','calibration','dev','unseen_generator','external']}
    assert not sets['train'] & (sets['calibration']|sets['dev']|sets['unseen_generator']|sets['external'])
    assert not sets['calibration'] & (sets['dev']|sets['unseen_generator']|sets['external'])
    assert all(r['model_id'] not in HELDOUT for r in final if r['split'] in {'train','calibration'})
    for partition in ['train','calibration','dev']:
        assert {r['family'] for r in final if r['split']==partition}==set(CLASSES),(partition,Counter(r['family'] for r in final if r['split']==partition))
    with gzip.open(b.DATA/'pilot.jsonl.gz','wt',encoding='utf-8',newline='\n') as f:
        for r in final:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    protocol={'status':'frozen_before_training','slice':name,'classes':CLASSES,'source_main_sha256':b.digest(b.MAIN/'samples.jsonl.gz'),
              'rows':len(final),'tasks':len({r['task_cluster_id'] for r in final}),
              'counts':{s:dict(Counter(r['family'] for r in final if r['split']==s)) for s in sets},
              'family_task_counts':{s:{family:len({r['task_cluster_id'] for r in final if r['split']==s and r['family']==family}) for family in CLASSES} for s in sets},
              'training_models':dict(Counter(r['model_id'] or 'Human_reference' for r in final if r['split']=='train')),
              'heldout_model_ids':sorted(HELDOUT),'registry_sha256':b.digest(BASE_OUT/'flagship_registry_and_pool.json'),
              'script_sha256':b.digest(Path(__file__)),'helper_sha256':b.digest(Path(b.__file__)),
              'data_sha256':b.digest(b.DATA/'pilot.jsonl.gz'),'encoder':'CodeT5-small full encoder fine-tuning, 1 epoch',
              'encoder_lr':3e-5,'classifier_lr':1e-3,'effective_batch':16,'maximum_tokens':512,'coverage':'head384+tail128',
              'main_length_bins':['<=512','513-1024','1025-2048','>2048'],'encoder_seed':b.SEED,'head_seeds':[0,1,2],
              'reference_folds':5,'ridge_alpha':100,'reference_teacher':'train tasks containing Human plus >=2 AI families only; inference predicts reference from code alone',
              'broad_sampling':'up to 4000 per class plus all eligible external and dev heldout-model rows; paired core retained; Human-only allowed',
              'ammc':{'original':.5,'comment':.2,'whitespace':.3,'safe_local_rename':.4,'train_attempt_status':dict(statuses)},
              'formal_test':False,'historical_test_read':False,'api_generation':False,'domain_used':False,'independent_test_claim':False,
              'limitations':['historically exposed public benchmarks','legacy prompt modes not fully aligned','reference labels not independently certified unaided Human',
                             'Qwen/Grok have only one flagship model ID; their member-heldout is unsupported','source and length shortcuts can remain, especially broad pool',
                             '512 token pilot does not claim full long-code coverage','heldout exact API/model identifiers do not prove distinct base weights']}
    b.save(b.OUT/'protocol.json',protocol);b.log('READY',name,protocol['rows'],protocol['tasks'],protocol['counts'])
    return final


def token_cache(rows,tok):
    cache={};lengths={}
    for r in rows:
        for field in ['code','train_view','stress_view','content_view']:
            text=r[field];key=b.sh(text)
            if key not in cache:
                ids=tok.encode(text,add_special_tokens=False);lengths[key]=len(ids)
                cache[key]=ids if len(ids)<=512 else ids[:384]+ids[-128:]
    b.save(b.OUT/'token_lengths.json',{r['record_id']:lengths[b.sh(r['code'])] for r in rows})
    return cache,lengths


def task_weights(rows,task):
    if task=='F':
        counts=Counter(r['family'] for r in rows)
        return np.array([len(rows)/(len(CLASSES)*counts[r['family']]) for r in rows],dtype='float32')
    counts=Counter(r['family'] for r in rows);k=len(counts)-1
    return np.array([len(rows)*(.5 if r['family']=='Human' else .5/k)/counts[r['family']] for r in rows],dtype='float32')


def references(rows,h,sem,stress_sem,task):
    tr=np.array([i for i,r in enumerate(rows) if r['split']=='train']);groups=np.array([r['leakage_group'] for r in rows])
    bytask=defaultdict(list)
    for i in tr:bytask[rows[i]['task_cluster_id']].append(i)
    targets=np.zeros_like(h);teacher=[];sample_weights=np.zeros(len(rows),dtype='float32')
    for ix in bytask.values():
        fam=defaultdict(list)
        for i in ix:fam[rows[i]['family']].append(i)
        if 'Human' not in fam or len(fam)<3:continue
        centers={}
        for family,ids in fam.items():
            members=defaultdict(list)
            for i in ids:members[rows[i]['model_id'] or 'Human_reference'].append(i)
            centers[family]=np.mean([h[js].mean(0) for js in members.values()],axis=0)
            for js in members.values():sample_weights[js]=1/(len(fam)*len(members)*len(js))
        center=np.mean(list(centers.values()),axis=0) if task=='F' else .5*centers['Human']+.5*np.mean([v for f,v in centers.items() if f!='Human'],axis=0)
        targets[ix]=center;teacher.extend(ix)
    teacher=np.array(teacher);global_center=targets[teacher].mean(0)
    nonteacher=np.setdiff1d(tr,teacher);targets[nonteacher]=global_center
    oof=np.zeros_like(h);full=[];stress=[]
    for _,(fit,hold) in enumerate(GroupKFold(5).split(tr,groups=groups[tr])):
        fit=np.intersect1d(tr[fit],teacher);hold=tr[hold]
        assert not set(groups[fit]) & set(groups[hold])
        scaler=StandardScaler().fit(sem[fit]);reg=Ridge(alpha=100).fit(scaler.transform(sem[fit]),targets[fit],sample_weight=sample_weights[fit]*len(fit)/sample_weights[fit].sum())
        oof[hold]=reg.predict(scaler.transform(sem[hold]));full.append(reg.predict(scaler.transform(sem)));stress.append(reg.predict(scaler.transform(stress_sem)))
    b.save(b.OUT/f'{task}_reference_support.json',{'teacher_rows':len(teacher),'teacher_tasks':len({rows[i]['task_cluster_id'] for i in teacher}),
           'singleton_or_unpaired_train_rows':len(nonteacher),'no_pseudo_same_task_groups':True})
    return oof,full,stress,targets


def metric(rows,logits,task):
    result=b.evaluate(rows,logits,task)
    if task=='F':
        y=np.array([r['label_F'] for r in rows]);present=sorted(set(y.tolist()))
        result['supported_macro_f1']=float(f1_score(y,np.asarray(logits).argmax(1),labels=present,average='macro',zero_division=0))
        result['supported_classes']=[CLASSES[i] for i in present]
    result['support']=dict(Counter(r['family'] for r in rows))
    return result


def run(name,baseline_arms=None):
    configure(name);rows=list(b.readgz(b.DATA/'pilot.jsonl.gz'));torch.set_num_threads(4);b.seed_all(b.SEED)
    device=torch.device('cuda');assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats()
    tok=b.tokenizer();cache,lengths=token_cache(rows,tok)
    path=b.OUT/'semantic_features.npz'
    if path.exists():
        z=np.load(path);sem=z['sem'];stress_sem=z['stress_sem']
    else:
        from transformers import T5EncoderModel
        model=T5EncoderModel.from_pretrained(b.MODEL,local_files_only=True).to(device)
        sem,_=b.encode(model,rows,cache,'content_view',device)
        stresses=[dict(r,content_view=b.content_view(r['stress_view'])) for r in rows]
        for r in stresses:
            key=b.sh(r['content_view'])
            if key not in cache:
                ids=tok.encode(r['content_view'],add_special_tokens=False);cache[key]=ids if len(ids)<=512 else ids[:384]+ids[-128:]
        stress_sem,_=b.encode(model,stresses,cache,'content_view',device)
        np.savez_compressed(path,sem=sem,stress_sem=stress_sem);del model;torch.cuda.empty_cache();b.log('SEMANTIC_DONE',name,len(rows))
    allmetrics={};predictions={}
    for task in ['F','D']:
        arms=b.lexical(rows,task)
        for arm in (baseline_arms or (['B0','B1'] if name=='controlled' else ['B1'])):
            path=b.OUT/f'{task}_{arm}_features.npz'
            if path.exists():
                z=np.load(path);h,base,hs,stress=(z[k] for k in ['h','base','hs','stress'])
            else:h,base,hs,stress=b.fine_tune(rows,cache,task,arm,device)
            arms[arm]=(base,stress)
            if arm=='B1':arms.update(b.correction_fit(rows,h,base,hs,stress,sem,stress_sem,task,device))
        for arm in ['B2','B3','Ours']:
            arms[arm+'_mean']=(np.mean([arms[f'{arm}_seed{s}'][0] for s in [0,1,2]],axis=0),np.mean([arms[f'{arm}_seed{s}'][1] for s in [0,1,2]],axis=0))
        results={};comparisons={}
        partitions=['dev','external','unseen_generator']
        for partition in partitions:
            idx=np.array([i for i,r in enumerate(rows) if r['split']==partition],dtype=int)
            if not len(idx):continue
            if partition=='unseen_generator':
                # Include the dev Human reference rows for genuine two-class detection.
                held_tasks={rows[i]['task_cluster_id'] for i in idx}
                idx=np.concatenate([idx,np.array([i for i,r in enumerate(rows) if r['split']=='dev' and r['family']=='Human' and r['task_cluster_id'] in held_tasks],dtype=int)])
            subset=[rows[i] for i in idx];results[partition]={}
            for arm,(clean,stress) in arms.items():
                entry={'clean':metric(subset,clean[idx],task),'stress':metric(subset,stress[idx],task),'lengths':{}}
                for label,lo,hi in [('<=512',0,512),('513-1024',512,1024),('1025-2048',1024,2048),('>2048',2048,10**9)]:
                    ix=np.array([i for i,r in enumerate(subset) if lo<lengths[b.sh(r['code'])]<=hi],dtype=int)
                    support=dict(Counter(subset[i]['family'] for i in ix))
                    if len(ix) and (task=='F' or len({subset[i]['label_D'] for i in ix})==2):
                        entry['lengths'][label]=metric([subset[i] for i in ix],clean[idx][ix],task)
                    else:entry['lengths'][label]={'n':len(ix),'support':support,'auroc':None,'reason':'empty_or_single_class_bucket'}
                if task=='D':entry['prediction_flip_rate']=float(np.mean((clean[idx].flatten()>=0)!=(stress[idx].flatten()>=0)))
                else:entry['prediction_flip_rate']=float(np.mean(clean[idx].argmax(1)!=stress[idx].argmax(1)))
                results[partition][arm]=entry
                np.savez_compressed(b.OUT/f'{task}_{arm}_{partition}_predictions.npz',clean=clean[idx],stress=stress[idx],
                                    labels=np.array([r['label_'+task] for r in subset]),record_ids=np.array([r['record_id'] for r in subset]),
                                    task_ids=np.array([r['task_cluster_id'] for r in subset]))
            if partition=='dev':
                comparisons={arm:b.paired_ci(subset,arms['Ours_mean'][0][idx],arms[arm][0][idx],task) for arm in ['B1','B2_mean','B3_mean','lexical_char','lexical_word']}
        allmetrics[task]={'evaluations':results,'paired_dev_comparisons':comparisons}
        b.save(b.OUT/'metrics.json',{'status':'running','slice':name,'tasks':allmetrics})
        b.log('TASK_DONE',name,task,{arm:round(x['clean']['auroc' if task=='D' else 'macro_f1'],4) for arm,x in results['dev'].items()})
    b.save(b.OUT/'metrics.json',{'status':'completed','slice':name,'tasks':allmetrics,'formal_test':False,'api_generation':False})
    write_report(name,allmetrics)


def write_report(name,metrics):
    configure(name);protocol=json.loads((b.OUT/'protocol.json').read_text(encoding='utf-8'))
    lines=[f'# 旗舰限定本机双任务验证：{name}','','仅二分类检测和含 Human 的家族归因；领域不参与。本轮不是正式独立测试。','',
           f'当前切片 {protocol["rows"]:,} 份代码，{protocol["tasks"]:,} 个来源任务。具体训练、校准、开发和外部支持数见 protocol.json。','',
           '| 方法 | 开发归因 Macro-F1 | 归因增强压力 | 开发检测 AUROC | 检测增强压力 |','|---|---:|---:|---:|---:|']
    arms=['length_layout','lexical_char','lexical_word']+(['B0'] if 'B0' in metrics['F']['evaluations']['dev'] else [])+['B1','B2_mean','B3_mean','Ours_mean']
    for arm in arms:
        f=metrics['F']['evaluations']['dev'][arm];d=metrics['D']['evaluations']['dev'][arm]
        lines.append(f'| {arm} | {f["clean"]["macro_f1"]:.4f} | {f["stress"]["macro_f1"]:.4f} | {d["clean"]["auroc"]:.4f} | {d["stress"]["auroc"]:.4f} |')
    for partition,title in [('unseen_generator','训练中未出现的公开模型标识'),('external','全来源留出的新一代模型代码')]:
        if partition not in metrics['F']['evaluations']:continue
        lines+=['',f'## {title}','','| 方法 | 有支持类别归因 Macro-F1 | 检测 AUROC |','|---|---:|---:|']
        for arm in ['B1','Ours_mean','lexical_char']:
            f=metrics['F']['evaluations'][partition][arm];d=metrics['D']['evaluations'][partition][arm]
            lines.append(f'| {arm} | {f["clean"]["supported_macro_f1"]:.4f} | {d["clean"]["auroc"]:.4f} |')
        lines+=['', '只统计该评测中有真值支持的类别，不能与七类开发 Macro-F1 当作同一难度比较。未见模型组加入开发 Human 参考用于真实二分类；外部组同时变化了来源、题目和模型代际，不能把损失只归因于一个因素。']
    lines+=['','## 实验口径','',
            'B0：直接微调；B1：同预算 AMMC 类增强微调；B2：等容量普通校正；B3：全局训练参考；Ours：训练侧同题多来源教师、五折 OOF 内容参考预测、低秩条件残差校正。三个小头 seed 的均值是优化重复，编码器只训练一个 seed。', '',
            '同题组使用有 Human 的 BigCodeBench 题目，但不要求 Qwen/Grok 覆盖所有题目。扩大组纳入不完全配对的 Human 与 AI，主要验证规模/覆盖扩展后的可用性；仍存在来源、格式和长度混杂，不能把其高分直接当作方法突破。参考器只在训练中的真实多来源题目拟合，未配对代码参加分类训练，不伪造同题教师。', '',
            'Human 的 BigCodeBench 函数体依据官方代码头补全，HumanEval 函数体依据已有同题公开任务头补全。所有类别统一移除题面文档字符串，保留其它实现内容；原始主集未改。来源不明、Mini/Haiku/Flash、小型 Coder 和 Qwen 72B/110B 不进入本次旗舰组。旗舰版本包含历史公开旗舰，并不宣称全为 2026 年最新模型；动态 Max 标识不推定精确快照。', '',
            '模型用本机 CodeT5-small 全编码器微调 1 epoch，512 token 首384+尾128覆盖；不能冒称 CodeT5-base 正式主表。长度按完整原文 token 数分 <=512、513–1024、1025–2048、>2048；超512的分桶实际只看规定覆盖，单类桶 AUROC=NA，详见 metrics.json。', '',
            'AMMC 类训练：50% 保留原文，否则独立按 .2/.3/.4 尝试去注释、轻量空白与保守局部变量改名；失败回退并记原因。压力副本在开发/外部各自生成，永不加入训练；语法/AST 保真不等于运行正确性或绝对语义等价。', '',
            '任务、跨来源 HumanEval 题号别名、相同/空白归一化代码共同隔离。模型版本留出在编码器、分类器、参考教师和校准中全部排除。Qwen/Grok 当前只有一个旗舰成员，不能支持其家族内未见成员测试。旧历史测试集未读取；公共基准已有暴露，不声明 fresh test。', '',
            '强词法 char/word TF-IDF 与长度布局控制单列，尚未实施完整融合 C-P0。paired_dev_comparisons 给出 Ours 相对基线/等容量/词法的任务簇配对区间；点估计最高不等于显著优势。', '',
            '主模型公共语义参考是学习的内容代理，不是旗舰 base 或真实后训练位移，不证明纯因果后训练差异。']
    (b.OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (b.OUT/'SHA256SUMS.txt').write_text(''.join(f'{b.digest(p)}  {p.name}\n' for p in sorted(b.OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt'),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--slice',choices=['controlled','broad','both'],default='both');args=parser.parse_args()
    names=['controlled','broad'] if args.slice=='both' else [args.slice]
    missing=[name for name in names if not (BASE_DATA/name/'pilot.jsonl.gz').exists() or not (BASE_OUT/name/'protocol.json').exists()]
    if missing:
        pool=get_pool();core=choose_core(pool)
        for name in missing:finish_data(copy.deepcopy(core) if name=='controlled' else select_broad(pool,core),pool,name)
    if args.prepare_only:return
    b.references=references;b.weights=task_weights;b.token_cache=token_cache
    for name in names:run(name)
    b.log('FLAGSHIP_BATCH_DONE')


if __name__=='__main__':main()
