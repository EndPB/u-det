"""Use additional flagship families as genuinely family-heldout detection probes."""
import argparse,gzip,json,re
from collections import Counter,defaultdict
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import ddet_sufficient_local_2026_10_14 as t
import ddet_two_tasks_local_2026_10_14 as b
import ddet_flagship_local_2026_10_14 as d
from ddet_flagship_clean_local_2026_10_14 import no_blank_placeholders

OUT=t.OUT;DATA=t.DATA
ALLOWED={'Mistral':{'mistral-large-2402','mistralai--Mistral-Large-Instruct-2407'},'GLM':{'glm-5.1'},'Kimi':{'kimi-k2.5'},'MiniMax':{'MiniMax-M2.7'}}


def prepare():
    path=DATA/'unseen_families.jsonl.gz'
    if path.exists():return
    main=list(b.readgz(DATA/'pilot.jsonl.gz'));extra=[];rejected=Counter()
    for r in b.readgz(b.MAIN/'samples.jsonl.gz'):
        family=r['family_label']
        if family not in ALLOWED or r['model_id'] not in ALLOWED[family] or r['representation_type']!='code':continue
        try:code=no_blank_placeholders(b.remove_docs(r['code']))
        except (SyntaxError,ValueError,RecursionError):rejected['not_parseable_complete_python']+=1;continue
        extra.append({'record_id':r['record_id'],'code':code,'family':family,'model_id':r['model_id'],'source_dataset':r['source_dataset'],'task_cluster_id':d.task_alias(r),'label_D':1})
    parent={r['task_cluster_id']:r['task_cluster_id'] for r in main+extra}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={}
    for r in main+extra:
        for key in [b.sh(r['code']),b.sh(re.sub(r'\s+','',r['code']))]:
            if key in seen:
                x,y=find(r['task_cluster_id']),find(seen[key]);parent[max(x,y)]=min(x,y)
            seen[key]=r['task_cluster_id']
    learning={find(r['task_cluster_id']) for r in main if r['split']=='train'};calibration={find(r['task_cluster_id']) for r in main if r['split']=='calibration'}
    for r in extra:
        g=find(r['task_cluster_id']);r['leakage_group']=g;r['role']='seen_training_task_ai_only' if g in learning else 'seen_calibration_task_ai_only' if g in calibration else 'unseen_task'
    wanted={r['task_cluster_id'] for r in extra if r['role']=='unseen_task'}
    references=[dict(r,role='heldout_human_reference') for r in main if r['family']=='Human' and r['split'] in {'dev','external'} and r['task_cluster_id'] in wanted]
    rows=extra+references
    for r in rows:r['content_view']=b.content_view(r['code']);r['stress_view']=b.augment(r['code'],r['record_id']+':unseen_family_stress',force=True)[0]
    with gzip.open(path,'wt',encoding='utf-8',newline='\n') as handle:
        for r in rows:handle.write(json.dumps(r,ensure_ascii=False)+'\n')
    manifest={'new_ai_rows':len(extra),'heldout_human_references':len(references),'roles':{family:dict(Counter(r['role'] for r in extra if r['family']==family)) for family in ALLOWED},'rejected':dict(rejected),'data_sha256':b.digest(path),
              'classifier_training_contains_none_of_these_families':True,'family_attribution_scope_unchanged':True,'unseen_task_auroc_requires_matching_heldout_human':True,
              'seen_training_or_calibration_tasks':'AI-only diagnostic, no independent AUROC claim',
              'official_model_sources':{'Mistral':'https://docs.mistral.ai/resources/changelogs','GLM':'https://z.ai/blog/glm-5.1','Kimi':'https://www.kimi.com/en/blog/kimi-k2-5','MiniMax':'https://www.minimax.io/models/text/m27'}}
    b.save(OUT/'unseen_families_manifest.json',manifest);b.log('EXTRA_FAMILIES_READY',manifest)


def lexical():
    target=OUT/'unseen_families_lexical_predictions.npz'
    if target.exists():return
    main=list(b.readgz(DATA/'pilot.jsonl.gz'));rows=list(b.readgz(DATA/'unseen_families.jsonl.gz'));train=[r for r in main if r['split']=='train'];y=np.array([r['label_D'] for r in train]);data={'record_ids':np.array([r['record_id'] for r in rows])}
    for analyzer,ngram,maximum in [('char',(3,5),40000),('word',(1,2),30000)]:
        vector=TfidfVectorizer(analyzer=analyzer,ngram_range=ngram,max_features=maximum,lowercase=False,sublinear_tf=True,dtype=np.float32);x=vector.fit_transform([r['code'] for r in train]);model=LinearSVC(C=1,class_weight='balanced',max_iter=3000,random_state=b.SEED).fit(x,y)
        for view,field in [('clean','code'),('stress','stress_view')]:data[f'lexical_{analyzer}_{view}']=model.decision_function(vector.transform([r[field] for r in rows]))[:,None]
        b.log('EXTRA_LEXICAL_DONE',analyzer)
    np.savez_compressed(target,**data)


def canonicalize():
    path=DATA/'unseen_families.jsonl.gz';rows=list(b.readgz(path))
    if all('is_unique_sample' in r for r in rows):return
    keys=defaultdict(list)
    for i,r in enumerate(rows):keys[(r['task_cluster_id'],r['family'],r['model_id'],b.sh(r['code']))].append(i)
    for js in keys.values():
        canonical=min(js,key=lambda i:rows[i]['record_id'])
        for i in js:rows[i]['is_unique_sample']=i==canonical;rows[i]['duplicate_of']=None if i==canonical else rows[canonical]['record_id']
    temp=path.with_suffix('.tmp.gz')
    with gzip.open(temp,'wt',encoding='utf-8',newline='\n') as handle:
        for r in rows:handle.write(json.dumps(r,ensure_ascii=False)+'\n')
    temp.replace(path);manifest=json.loads((OUT/'unseen_families_manifest.json').read_text())
    manifest['unique_ai_samples']=dict(Counter(r['family'] for r in rows if r['family']!='Human' and r['is_unique_sample']))
    manifest['duplicate_publication_rows_not_counted']=sum(not r['is_unique_sample'] for r in rows);manifest['data_sha256']=b.digest(path)
    b.save(OUT/'unseen_families_manifest.json',manifest)


def evaluate():
    from ddet_sufficient_analysis_2026_10_14 import metrics
    rows=list(b.readgz(DATA/'unseen_families.jsonl.gz'));sem=np.load(OUT/'unseen_families_semantic_features.npz');ref=np.load(DATA/'D_reference_predictors.npz');arms={}
    assert np.array_equal(sem['record_ids'],[r['record_id'] for r in rows])
    for arm in ['B0','B1']:
        z=np.load(OUT/f'unseen_families_D_{arm}_features.npz');assert np.array_equal(z['record_ids'],[r['record_id'] for r in rows]);arms[arm]=(z['base'],z['stress'])
    z=np.load(OUT/'unseen_families_D_B1_features.npz');h,hs=z['h'],z['hs'];base,stress=z['base'],z['stress'];hz=(h-ref['h_mean'])/ref['h_scale'];hsz=(hs-ref['h_mean'])/ref['h_scale']
    preds=[((sem['sem']-ref[f'mean{k}'])/ref[f'scale{k}'])@ref[f'coef{k}'].T+ref[f'intercept{k}'] for k in range(5)];sps=[((sem['stress_sem']-ref[f'mean{k}'])/ref[f'scale{k}'])@ref[f'coef{k}'].T+ref[f'intercept{k}'] for k in range(5)]
    def f(hh,g):gg=(g-ref['h_mean'])/ref['h_scale'];return np.concatenate([hh-gg,hh*gg],1)
    for arm in ['B2','B3','Ours']:
        ff=[np.concatenate([hz,hz**2],1)] if arm=='B2' else [f(hz,ref['global'])] if arm=='B3' else [f(hz,g) for g in preds]
        sf=[np.concatenate([hsz,hsz**2],1)] if arm=='B2' else [f(hsz,ref['global'])] if arm=='B3' else [f(hsz,g) for g in sps]
        for seed in [0,1,2]:
            state=torch.load(DATA/f'D_{arm}_seed{seed}.pt',weights_only=True);head=b.Correction('D').eval();head.load_state_dict(state['head'])
            with torch.no_grad():
                clean=base+np.mean([head(torch.tensor((x-state['x_mean'].numpy())/state['x_scale'].numpy(),dtype=torch.float32)).numpy() for x in ff],axis=0)
                stressed=stress+np.mean([head(torch.tensor((x-state['x_mean'].numpy())/state['x_scale'].numpy(),dtype=torch.float32)).numpy() for x in sf],axis=0)
            arms[f'{arm}_seed{seed}']=(clean,stressed)
    lex=np.load(OUT/'unseen_families_lexical_predictions.npz');assert np.array_equal(lex['record_ids'],[r['record_id'] for r in rows])
    for arm in ['lexical_char','lexical_word']:arms[arm]=(lex[arm+'_clean'],lex[arm+'_stress'])
    result={}
    for family in ALLOWED:
        human_tasks={r['task_cluster_id'] for r in rows if r['family']=='Human' and r['is_unique_sample']}
        ai=[i for i,r in enumerate(rows) if r['family']==family and r['role']=='unseen_task' and r['is_unique_sample'] and r['task_cluster_id'] in human_tasks];tasks={rows[i]['task_cluster_id'] for i in ai};human=[i for i,r in enumerate(rows) if r['family']=='Human' and r['task_cluster_id'] in tasks and r['is_unique_sample']];ix=np.array(ai+human,dtype=int)
        result[family]={'independent':{},'seen_task_ai_only':{}}
        for arm,(clean,stress) in arms.items():
            result[family]['independent'][arm]={view:metrics([rows[i] for i in ix],score[ix],'D') for view,score in [('clean',clean),('stress',stress)]} if len(ix) else None
            seen=np.array([i for i,r in enumerate(rows) if r['family']==family and r['role']!='unseen_task' and r['is_unique_sample']],dtype=int)
            result[family]['seen_task_ai_only'][arm]={'n':len(seen),'ai_tpr_at_fixed_zero':float(np.mean(clean[seen]>=0)) if len(seen) else None}
    np.savez_compressed(OUT/'unseen_families_all_predictions.npz',record_ids=np.array([r['record_id'] for r in rows]),labels=np.array([r['label_D'] for r in rows]),**{arm+'_'+view:score for arm,pair in arms.items() for view,score in zip(['clean','stress'],pair)})
    b.save(OUT/'unseen_families_metrics.json',result)
    append_report(rows,result)


def append_report(rows,result):
    doc=t.ROOT/'d-det/docx/d-det_全量旗舰双任务充分训练与归因空间分析_2026-10-14.md';text=doc.read_text(encoding='utf-8').split('\n## 7. 整个家族留出的检测测试')[0]
    lines=['','## 7. 整个家族留出的检测测试','',
           '补充使用原主数据中未进入七类归因训练标签的 Mistral、GLM、Kimi、MiniMax 旗舰输出，整家族不参与模型训练。归因仍是 Human 加六个已训练 AI 家族；这里评价的是对新家族 AI 代码的二分类检测，不要求归因模型识别从未训练过的新标签。','',
           '同一模型、同一题目、完全相同代码只计一次。主表严格限制为：题目及重复代码簇未参与训练或校准，而且该题目有留出的 Human 参考。其余输出保留为 AI 检出率诊断，不冒充同题 AUROC。','',
           '| 留出家族 | AI / Human 样本 | 普通微调 AUROC | AMMC AUROC | 普通校正 AUROC | 条件方法 AUROC | 字符词法 AUROC |','|---|---|---:|---:|---:|---:|---:|']
    def v(family,arm,key='auroc',view='clean'):
        names=[f'{arm}_seed{s}' for s in [0,1,2]] if arm in ['B2','B3','Ours'] else [arm];values=[result[family]['independent'][name][view][key] for name in names]
        return None if any(x is None for x in values) else float(np.mean(values))
    for family in ALLOWED:
        support=result[family]['independent']['B0']['clean']['support'];human=support.get('Human',0);ai=sum(support.values())-human
        lines.append(f'| {family} | {ai} / {human} | '+' | '.join('NA' if v(family,a) is None else f'{v(family,a):.4f}' for a in ['B0','B1','B2','Ours','lexical_char'])+' |')
    lines += ['', '| 留出家族 | AMMC同题平均 AUROC | 条件同题平均 AUROC | AMMC改写 AUROC | 条件改写 AUROC |','|---|---:|---:|---:|---:|']
    for family in ALLOWED:lines.append(f'| {family} | '+ ' | '.join('NA' if x is None else f'{x:.4f}' for x in [v(family,'B1','task_macro_auroc'),v(family,'Ours','task_macro_auroc'),v(family,'B1',view='stress'),v(family,'Ours',view='stress')])+' |')
    lines += ['', 'Mistral 包含历史旗舰 Large 2402/2407，并非宣称最新代际；其余模型采用原公开数据标注的 GLM-5.1、Kimi-K2.5、MiniMax-M2.7 标识。官方型号资料：', '',
              '[Mistral型号记录](https://docs.mistral.ai/resources/changelogs)、[GLM-5.1](https://z.ai/blog/glm-5.1)、[Kimi-K2.5](https://www.kimi.com/en/blog/kimi-k2-5)、[MiniMax-M2.7](https://www.minimax.io/models/text/m27)。这些资料确认产品身份，不能独立认证公开样本背后的 API 调用。','',
              '整家族留出能够检验是否仅适应已知生成家族；仍须结合题目、来源、提示协议及改写结果解释，不能单靠一列高分认定纯后训练因果机制。']
    doc.write_bytes((text+'\n'.join(lines)+'\n').encode('utf-8'));env=json.loads((OUT/'env.json').read_text());env['result_document_sha256']=b.digest(doc);b.save(OUT/'env.json',env)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--lexical-only',action='store_true');args=parser.parse_args();t.setup();prepare();canonicalize()
    if args.prepare_only:return
    lexical()
    if not args.lexical_only:evaluate()


if __name__=='__main__':main()
