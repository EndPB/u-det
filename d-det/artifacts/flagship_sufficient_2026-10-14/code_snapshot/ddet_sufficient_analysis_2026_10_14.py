"""Matched controls and descriptive attribution-space evidence for the full run."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,copy,gzip,json,math,shutil,sys,time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score,roc_auc_score,classification_report,confusion_matrix,silhouette_score
from scipy.stats import rankdata
import ddet_sufficient_local_2026_10_14 as t
import ddet_two_tasks_local_2026_10_14 as b
import ddet_flagship_local_2026_10_14 as d

OUT=t.OUT;DATA=t.DATA;ROOT=t.ROOT


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def reference_fit(rows,h,sem,task):
    tr=np.array([i for i,r in enumerate(rows) if r['split']=='train']);groups=np.array([r['leakage_group'] for r in rows])
    bytask=defaultdict(list)
    for j,i in enumerate(tr):bytask[rows[i]['task_cluster_id']].append(j)
    target=np.zeros((len(tr),512),dtype='float32');teacher=[];w=np.zeros(len(tr),dtype='float32')
    for js in bytask.values():
        families=defaultdict(list)
        for j in js:families[rows[tr[j]]['family']].append(j)
        if 'Human' not in families or len(families)<3:continue
        centers={}
        for family,ids in families.items():
            members=defaultdict(list)
            for j in ids:members[rows[tr[j]]['model_id'] or 'Human_reference'].append(j)
            centers[family]=np.mean([h[tr[jj]].mean(0) for jj in members.values()],axis=0)
            for jj in members.values():w[jj]=1/(len(families)*len(members)*len(jj))
        center=np.mean(list(centers.values()),axis=0) if task=='F' else .5*centers['Human']+.5*np.mean([v for f,v in centers.items() if f!='Human'],axis=0)
        target[js]=center;teacher.extend(js)
    teacher=np.array(teacher);global_ref=target[teacher].mean(0);target[np.setdiff1d(np.arange(len(tr)),teacher)]=global_ref
    oof=np.zeros_like(target);models=[]
    for fit,hold in GroupKFold(5).split(tr,groups=groups[tr]):
        fit=np.intersect1d(fit,teacher);sc=StandardScaler().fit(sem[tr[fit]])
        reg=Ridge(alpha=100).fit(sc.transform(sem[tr[fit]]),target[fit],sample_weight=w[fit]*len(fit)/w[fit].sum())
        oof[hold]=reg.predict(sc.transform(sem[tr[hold]]));models.append((sc,reg))
    b.save(OUT/f'{task}_reference_support.json',{'teacher_rows':len(teacher),'teacher_tasks':len({rows[tr[j]]['task_cluster_id'] for j in teacher}),'nonteacher_train_rows':len(tr)-len(teacher)})
    return tr,oof,models,target.mean(0)


def corrections(rows,task):
    dest=OUT/f'{task}_correction_predictions.npz'
    if dest.exists():return
    z=np.load(OUT/f'{task}_B1_features.npz');s=np.load(OUT/'semantic_features.npz')
    assert np.array_equal(z['row_ids'],s['row_ids'])
    h,hs,base,stress=(z[k] for k in ['h','hs','base','stress']);sem,ss=s['sem'],s['stress_sem']
    tr,oof,models,global_ref=reference_fit(rows,h,sem,task);cal=np.array([i for i,r in enumerate(rows) if r['split']=='calibration']);ev=np.array([i for i,r in enumerate(rows) if r['split']!='train'])
    hsc=StandardScaler().fit(h[tr]);hz=hsc.transform(h);hsz=hsc.transform(hs);labels=np.array([r['label_'+task] for r in rows])
    output={'indices':ev,'record_ids':np.array([rows[i]['record_id'] for i in ev]),'labels':labels[ev]};selection={}
    # Small reference predictors are serialized as numeric arrays, never external executable objects.
    numeric={'global':global_ref,'h_mean':hsc.mean_,'h_scale':hsc.scale_}
    for k,(sc,reg) in enumerate(models):numeric.update({f'mean{k}':sc.mean_,f'scale{k}':sc.scale_,f'coef{k}':reg.coef_,f'intercept{k}':reg.intercept_})
    np.savez_compressed(DATA/f'{task}_reference_predictors.npz',**numeric)
    def f(hh,ref):
        rr=hsc.transform(ref);return np.concatenate([hh-rr,hh*rr],1).astype('float32')
    calrows=[rows[i] for i in cal];weights=torch.tensor(d.task_weights([rows[i] for i in tr],task),device='cuda');y=torch.tensor(labels[tr],dtype=torch.long,device='cuda')
    for arm in ['B2','B3','Ours']:
        if arm=='B2':
            trainf=np.concatenate([hz[tr],hz[tr]**2],1);ef=[np.concatenate([hz[ev],hz[ev]**2],1)];sf=[np.concatenate([hsz[ev],hsz[ev]**2],1)]
        elif arm=='B3':
            trainf=f(hz[tr],np.broadcast_to(global_ref,(len(tr),512)));ef=[f(hz[ev],np.broadcast_to(global_ref,(len(ev),512)))];sf=[f(hsz[ev],np.broadcast_to(global_ref,(len(ev),512)))]
        else:
            trainf=f(hz[tr],oof);ef=[f(hz[ev],reg.predict(sc.transform(sem[ev]))) for sc,reg in models];sf=[f(hsz[ev],reg.predict(sc.transform(ss[ev]))) for sc,reg in models]
        xsc=StandardScaler().fit(trainf);xt=torch.tensor(xsc.transform(trainf),device='cuda');del trainf
        ef=[xsc.transform(a) for a in ef];sf=[xsc.transform(a) for a in sf];evlookup={i:j for j,i in enumerate(ev)};cj=np.array([evlookup[i] for i in cal])
        cf=[torch.tensor(a[cj],device='cuda') for a in ef];bt=torch.tensor(base[tr],device='cuda');bc=torch.tensor(base[cal],device='cuda')
        for seed in [0,1,2]:
            b.seed_all(seed);head=b.Correction(task).cuda();opt=torch.optim.AdamW(head.parameters(),lr=3e-3,weight_decay=.01);best=-np.inf;best_epoch=0
            for epoch in range(25):
                head.train();order=np.random.default_rng(seed+epoch).permutation(len(tr))
                for start in range(0,len(tr),256):
                    ix=torch.tensor(order[start:start+256],device='cuda');pred=bt[ix]+head(xt[ix])
                    loss=torch.nn.functional.cross_entropy(pred,y[ix],reduction='none') if task=='F' else torch.nn.functional.binary_cross_entropy_with_logits(pred.ravel(),y[ix].float(),reduction='none')
                    opt.zero_grad(set_to_none=True);(loss*weights[ix]).mean().backward();opt.step()
                head.eval()
                with torch.no_grad():pred=bc+torch.stack([head(a) for a in cf]).mean(0)
                score=t.quick_metric(calrows,pred.cpu().numpy(),task)
                if score>best:best=score;best_epoch=epoch+1;state=copy.deepcopy(head.state_dict())
            head.load_state_dict(state);head=head.cpu().eval();clean=base[ev].copy();stressed=stress[ev].copy()
            with torch.no_grad():
                for start in range(0,len(ev),2048):
                    end=start+2048
                    clean[start:end]+=np.mean([head(torch.from_numpy(a[start:end])).numpy() for a in ef],axis=0)
                    stressed[start:end]+=np.mean([head(torch.from_numpy(a[start:end])).numpy() for a in sf],axis=0)
            name=f'{arm}_seed{seed}';output[name+'_clean']=clean;output[name+'_stress']=stressed
            torch.save({'head':head.state_dict(),'x_mean':torch.from_numpy(xsc.mean_),'x_scale':torch.from_numpy(xsc.scale_)},DATA/f'{task}_{name}.pt')
            selection[name]={'epoch':best_epoch,'calibration_metric':best};b.log('FULL_HEAD_DONE',task,name,round(best,5))
            del head,opt
        del xt,cf,ef,sf,bt,bc;torch.cuda.empty_cache()
    np.savez_compressed(dest,**output);b.save(OUT/f'{task}_head_selection.json',selection)


def lexical(rows):
    for task in ['F','D']:
        path=OUT/f'{task}_lexical_predictions.npz'
        if path.exists():continue
        arms=b.lexical(rows,task);ix=np.array([i for i,r in enumerate(rows) if r['split']!='train']);data={'indices':ix,'record_ids':np.array([rows[i]['record_id'] for i in ix])}
        for name,(clean,stress) in arms.items():data[name+'_clean']=clean[ix];data[name+'_stress']=stress[ix]
        np.savez_compressed(path,**data);b.log('FULL_LEXICAL_DONE',task)


def metrics(rows,scores,task):
    y=np.array([r['label_'+task] for r in rows]);pred=scores.ravel()>=0 if task=='D' else scores.argmax(1);families=dict(Counter(r['family'] for r in rows));out={'n':len(rows),'support':families}
    if task=='F':
        present=sorted(set(y));out.update(macro_f1=float(f1_score(y,pred,labels=range(7),average='macro',zero_division=0)),supported_macro_f1=float(f1_score(y,pred,labels=present,average='macro',zero_division=0)),
                    ai_only_macro_f1=float(f1_score(y[y!=0],pred[y!=0],labels=range(1,7),average='macro',zero_division=0)),confusion=confusion_matrix(y,pred,labels=range(7)).tolist(),
                    report=classification_report(y,pred,labels=range(7),target_names=b.CLASSES,output_dict=True,zero_division=0))
    else:
        out.update(auroc=float(roc_auc_score(y,scores.ravel())) if len(set(y))==2 else None,macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),human_fpr=float(pred[y==0].mean()) if (y==0).any() else None,ai_tpr=float(pred[y==1].mean()) if (y==1).any() else None)
        groups=defaultdict(list)
        for i,r in enumerate(rows):groups[r['task_cluster_id']].append(i)
        auc=[]
        for js in groups.values():
            yy=y[js];n1=yy.sum();n0=len(yy)-n1
            if n1 and n0:auc.append((rankdata(scores.ravel()[js])[yy==1].sum()-n1*(n1+1)/2)/(n1*n0))
        out['task_macro_auroc']=float(np.mean(auc)) if auc else None
    return out


def all_predictions(rows,task):
    ev=np.array([i for i,r in enumerate(rows) if r['split']!='train']);arms={}
    for arm in ['B0','B1']:
        z=np.load(OUT/f'{task}_{arm}_features.npz');assert np.array_equal(z['row_ids'],[r['record_id'] for r in rows]);arms[arm]=(z['base'][ev],z['stress'][ev])
    for filename,prefixes in [(f'{task}_correction_predictions.npz',[f'{a}_seed{s}' for a in ['B2','B3','Ours'] for s in [0,1,2]]),(f'{task}_lexical_predictions.npz',['lexical_char','lexical_word','length_layout'])]:
        z=np.load(OUT/filename);assert np.array_equal(z['indices'],ev)
        for name in prefixes:arms[name]=(z[name+'_clean'],z[name+'_stress'])
    return ev,arms


def evaluate(rows):
    lengths=read(OUT/'token_lengths.json');result={};raw={}
    for task in ['F','D']:
        ev,arms=all_predictions(rows,task);raw[task]=(ev,arms);result[task]={}
        for partition in ['dev','unseen_generator','external']:
            js=[j for j,i in enumerate(ev) if rows[i]['split']==partition]
            if partition=='unseen_generator':
                tasks={rows[ev[j]]['task_cluster_id'] for j in js};js += [j for j,i in enumerate(ev) if rows[i]['split']=='dev' and rows[i]['family']=='Human' and rows[i]['task_cluster_id'] in tasks]
            js=np.array(js);subset=[rows[ev[j]] for j in js];result[task][partition]={}
            for arm,(clean,stress) in arms.items():
                entry={'clean':metrics(subset,clean[js],task),'stress':metrics(subset,stress[js],task),'lengths':{}}
                for label,low,high in [('<=512',0,512),('513-1024',512,1024),('1025-2048',1024,2048),('>2048',2048,1e9)]:
                    ix=np.array([j for j,r in enumerate(subset) if low<lengths[r['record_id']]<=high],dtype=int)
                    entry['lengths'][label]=metrics([subset[j] for j in ix],clean[js][ix],task) if len(ix) else {'n':0,'support':{},'auroc':None,'supported_macro_f1':None}
                result[task][partition][arm]=entry
                np.savez_compressed(OUT/f'{task}_{arm}_{partition}_predictions.npz',clean=clean[js],stress=stress[js],record_ids=np.array([r['record_id'] for r in subset]),labels=np.array([r['label_'+task] for r in subset]))
    b.save(OUT/'metrics.json',result)
    return result,raw


def cohort_indices(rows):
    grouped=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(rows):
        if r['split']=='dev':grouped[r['task_cluster_id']][r['family']].append(i)
    out={}
    for name,families in [('same_task_seven',b.CLASSES),('same_task_five',['Human','OpenAI','Claude','Gemini','DeepSeek'])]:
        tasks=sorted([key for key,fams in grouped.items() if all(f in fams for f in families)],key=b.sh)[:300]
        out[name]=np.array([min(grouped[key][family],key=lambda i:b.sh(rows[i]['record_id'])) for key in tasks for family in families],dtype=int)
    for name,partition in [('balanced_dev','dev'),('external','external'),('unseen_versions','unseen_generator')]:
        indices=[i for i,r in enumerate(rows) if r['split']==partition]
        if partition=='unseen_generator':
            tasks={rows[i]['task_cluster_id'] for i in indices};indices += [i for i,r in enumerate(rows) if r['split']=='dev' and r['family']=='Human' and r['task_cluster_id'] in tasks]
        fams=defaultdict(list)
        for i in indices:fams[rows[i]['family']].append(i)
        count=min(100,min(map(len,fams.values())))
        out[name]=np.array([i for family in sorted(fams) for i in sorted(fams[family],key=lambda j:b.sh(rows[j]['record_id']))[:count]],dtype=int)
    return out


def cohort_performance(rows,raw):
    cohorts=cohort_indices(rows);result={}
    for name in ['same_task_seven','same_task_five']:
        ix=cohorts[name];subset=[rows[i] for i in ix];result[name]={'rows':len(ix),'tasks':len({r['task_cluster_id'] for r in subset}),'record_ids':[r['record_id'] for r in subset],'metrics':{}}
        for task,(ev,arms) in raw.items():
            lookup={i:j for j,i in enumerate(ev)};js=np.array([lookup[i] for i in ix]);result[name]['metrics'][task]={}
            for arm,(clean,stress) in arms.items():result[name]['metrics'][task][arm]={'clean':metrics(subset,clean[js],task),'stress':metrics(subset,stress[js],task)}
    b.save(OUT/'same_task_classifier_metrics.json',result)


def geometry(x,rows):
    labels=np.array([r['label_F'] for r in rows]);tasks=np.array([r['task_cluster_id'] for r in rows]);out={'n':len(rows),'classes':dict(Counter(r['family'] for r in rows))}
    def local(xx,yy,tt):
        normalized=xx/np.maximum(np.linalg.norm(xx,axis=1,keepdims=True),1e-8);distance=np.maximum(0,1-normalized@normalized.T)
        np.fill_diagonal(distance,0);sil=float(silhouette_score(distance,yy,metric='precomputed')) if len(set(yy))>1 else None
        distance[tt[:,None]==tt[None,:]]=np.inf;k=min(10,min(np.isfinite(distance).sum(1)))
        if not k:return {'family_cosine_silhouette':sil,'leave_task_out_knn_macro_recall':None}
        nearest=np.argsort(distance,axis=1)[:,:k];pred=np.array([np.bincount(yy[js],minlength=7).argmax() for js in nearest])
        recall=float(np.mean([np.mean(pred[yy==label]==label) for label in np.unique(yy)]))
        return {'family_cosine_silhouette':sil,'leave_task_out_knn_macro_recall':recall}
    out.update(local(x,labels,tasks));ai=labels!=0
    if len(set(labels[ai]))>1:out['ai_only']=local(x[ai],labels[ai],tasks[ai])
    # Two-way decomposition only for a complete, equally represented task x family design.
    bytask=defaultdict(list)
    for i,key in enumerate(tasks):bytask[key].append(i)
    present=sorted(set(labels));complete=all(sorted(labels[js])==present for js in bytask.values())
    if complete and len(bytask)>1:
        tensor=np.stack([x[sorted(js,key=lambda i:labels[i])] for js in bytask.values()]);mean=tensor.mean((0,1));fam=tensor.mean(0)-mean;task=tensor.mean(1)-mean
        residual=tensor-mean-fam[None]-task[:,None];ss=np.array([len(tensor)*(fam*fam).sum(),len(present)*(task*task).sum(),(residual*residual).sum()])
        out['balanced_variance_fractions']={key:float(v/ss.sum()) for key,v in zip(['family','task','interaction'],ss)}
    return out


def visualizations(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from transformers import T5EncoderModel
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from ddet_t5_local_fast_2026_10_14 import install
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);cohorts=cohort_indices(rows)
    tr=np.array([i for i,r in enumerate(rows) if r['split']=='train']);anchor=np.array(sorted(tr,key=lambda i:b.sh(rows[i]['record_id']))[:2048])
    selected=np.array(sorted(set(anchor.tolist()).union(*(ix.tolist() for ix in cohorts.values()))));lookup={i:j for j,i in enumerate(selected)}
    b.save(OUT/'visual_cohorts.json',{name:{'rows':len(ix),'tasks':len({rows[i]['task_cluster_id'] for i in ix}),'record_ids':[rows[i]['record_id'] for i in ix]} for name,ix in cohorts.items()})
    prepath=OUT/'pretraining_visual_features.npz'
    if prepath.exists():
        z=np.load(prepath);assert np.array_equal(z['indices'],selected);pre=z['h']
    else:
        tok=b.tokenizer();cache={}
        for i in selected:
            ids=tok.encode(rows[i]['code'],add_special_tokens=False);cache[b.sh(rows[i]['code'])]=ids if len(ids)<=512 else ids[:384]+ids[-128:]
        model=install(T5EncoderModel.from_pretrained(b.MODEL,local_files_only=True)).cuda();pre,_=t.encode(model,None,rows,cache,indices=selected);del model;torch.cuda.empty_cache()
        np.savez_compressed(prepath,h=pre,indices=selected,record_ids=np.array([rows[i]['record_id'] for i in selected]))
    f0=np.load(OUT/'F_B0_features.npz');f1=np.load(OUT/'F_B1_features.npz');sem=np.load(OUT/'semantic_features.npz')['sem'][selected]
    ref=np.load(DATA/'F_reference_predictors.npz');predrefs=[((sem-ref[f'mean{k}'])/ref[f'scale{k}'])@ref[f'coef{k}'].T+ref[f'intercept{k}'] for k in range(5)]
    raw={'Before training':pre,'Vanilla fine-tuning':f0['h'][selected],'AMMC fine-tuning':f1['h'][selected],'Conditional residual':f1['h'][selected]-np.mean(predrefs,axis=0)}
    aj=np.array([lookup[i] for i in anchor]);stages={name:StandardScaler().fit(x[aj]).transform(x) for name,x in raw.items()}
    pca=PCA(2,random_state=b.SEED).fit(np.concatenate([x[aj] for x in stages.values()]));positions={name:pca.transform(x) for name,x in stages.items()}
    b.save(OUT/'projection_fit.json',{'fit_record_ids':[rows[i]['record_id'] for i in anchor],'fit_splits':['train'],'family_labels_used_for_axes':False,'components':pca.components_.tolist(),'mean':pca.mean_.tolist(),'stage_names':list(stages)})
    import csv
    with (OUT/'visual_points.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['cohort','stage','record_id','family','model_id','source','task','pc1','pc2'])
        for cname,ix in cohorts.items():
            for name,xy in positions.items():
                for i in ix:
                    j=lookup[i];writer.writerow([cname,name,rows[i]['record_id'],rows[i]['family'],rows[i]['model_id'],rows[i]['source_dataset'],rows[i]['task_cluster_id'],float(xy[j,0]),float(xy[j,1])])
    colors=['#6b7280','#16a34a','#d97706','#2563eb','#db2777','#9333ea','#dc2626'];stage_names=list(stages);evidence={};plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for cname in ['same_task_seven','same_task_five','balanced_dev','unseen_versions','external']:
        ix=cohorts[cname]
        if not len(ix):continue
        js=np.array([lookup[i] for i in ix]);subset=[rows[i] for i in ix];labels=np.array([r['label_F'] for r in subset]);evidence[cname]={}
        fig,axes=plt.subplots(1,4,figsize=(15,4),sharex=True,sharey=True)
        for ax,name in zip(axes,stage_names):
            xy=positions[name][js];evidence[cname][name]=geometry(stages[name][js],subset)
            for label in sorted(set(labels)):
                hit=labels==label;ax.scatter(xy[hit,0],xy[hit,1],s=14,alpha=.7,c=colors[label],label=b.CLASSES[label],linewidths=0)
            ax.set_title(name);ax.set_xlabel('Shared PC1')
        axes[0].set_ylabel('Shared PC2');handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=7,bbox_to_anchor=(.5,-.015))
        fig.suptitle(f'{cname}: fixed held-out codes, n={len(ix)}; label-blind train-fitted shared PCA',fontsize=12);fig.tight_layout(rect=[0,.06,1,.94])
        for ext in ['png','pdf','svg']:fig.savefig(figdir/f'family_space_{cname}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    # Exactly the same coordinates, now coloured by provenance, to expose source clustering.
    ix=cohorts['balanced_dev'];js=np.array([lookup[i] for i in ix]);sources=sorted({rows[i]['source_dataset'] for i in ix});source_colors=plt.get_cmap('tab10')
    fig,axes=plt.subplots(1,4,figsize=(15,4),sharex=True,sharey=True)
    for ax,name in zip(axes,stage_names):
        xy=positions[name][js]
        for j,source in enumerate(sources):
            hit=np.array([rows[i]['source_dataset']==source for i in ix]);ax.scatter(xy[hit,0],xy[hit,1],s=14,alpha=.7,color=source_colors(j),label=source,linewidths=0)
        ax.set_title(name);ax.set_xlabel('Shared PC1')
    handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.08));fig.suptitle('Same held-out points and coordinates, coloured by data source');fig.tight_layout(rect=[0,.1,1,.94]);fig.savefig(figdir/'source_space.png',dpi=180,bbox_inches='tight');fig.savefig(figdir/'source_space.pdf',bbox_inches='tight');plt.close(fig)
    # Nonlinear view is supplementary; its axes are independently fitted and not aligned.
    ix=cohorts['same_task_seven'];js=np.array([lookup[i] for i in ix]);labels=np.array([rows[i]['label_F'] for i in ix]);fig,axes=plt.subplots(1,4,figsize=(15,4))
    for ax,name in zip(axes,stage_names):
        xx=stages[name][js];xy=TSNE(n_components=2,perplexity=min(30,(len(js)-1)/3),random_state=b.SEED,init='pca',learning_rate='auto',max_iter=1000).fit_transform(xx)
        for label in sorted(set(labels)):
            hit=labels==label;ax.scatter(xy[hit,0],xy[hit,1],s=16,alpha=.75,c=colors[label],label=b.CLASSES[label],linewidths=0)
        ax.set_title(name);ax.set_xticks([]);ax.set_yticks([])
    handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=7);fig.suptitle('Same-task seven-family t-SNE; independently fitted axes, descriptive only');fig.tight_layout(rect=[0,.08,1,.94]);fig.savefig(figdir/'family_space_tsne_supplement.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # Interactive exploration uses the same PCA points, with family/source/length views.
    lengths=read(OUT/'token_lengths.json');ix=cohorts['balanced_dev'];js=np.array([lookup[i] for i in ix]);plot=make_subplots(rows=1,cols=4,subplot_titles=stage_names)
    groups=[]
    for mode in ['Family','Data source','Code length']:
        start=len(plot.data)
        values=[rows[i]['family'] if mode=='Family' else rows[i]['source_dataset'] if mode=='Data source' else '<=512' if lengths[rows[i]['record_id']]<=512 else '>512' for i in ix]
        categories=sorted(set(values))
        for col,name in enumerate(stage_names,1):
            xy=positions[name][js]
            for cat in categories:
                hit=np.array([v==cat for v in values]);picked=np.array(ix)[hit]
                colour=colors[b.CLASSES.index(cat)] if mode=='Family' else source_colors(categories.index(cat))
                if not isinstance(colour,str):colour='rgb('+','.join(str(int(v*255)) for v in colour[:3])+')'
                plot.add_trace(go.Scatter(x=xy[hit,0],y=xy[hit,1],mode='markers',name=cat,legendgroup=mode+cat,showlegend=col==1,visible=mode=='Family',marker={'size':6,'opacity':.75,'color':colour},
                              customdata=[[rows[i]['family'],rows[i]['model_id'] or 'Human reference',rows[i]['source_dataset'],rows[i]['task_cluster_id'],lengths[rows[i]['record_id']]] for i in picked],
                              hovertemplate='Family: %{customdata[0]}<br>Model: %{customdata[1]}<br>Source: %{customdata[2]}<br>Task: %{customdata[3]}<br>Tokens: %{customdata[4]}<extra></extra>'),row=1,col=col)
        groups.append((mode,start,len(plot.data)))
    plot.update_layout(title='Attribution space before and after training: fixed held-out samples',height=560,width=1500,template='plotly_white',updatemenus=[{'buttons':[{'label':mode,'method':'update','args':[{'visible':[start<=j<end for j in range(len(plot.data))]}]} for mode,start,end in groups],'direction':'right','x':0,'y':1.18}],legend={'orientation':'h','y':-.18})
    plot.write_html(figdir/'attribution_space_interactive.html',include_plotlyjs=True,full_html=True)
    # Compare the actual rank-8 correction spaces, including equal-capacity controls.
    hz=(f1['h'][selected]-ref['h_mean'])/ref['h_scale'];gz=(np.mean(predrefs,axis=0)-ref['h_mean'])/ref['h_scale'];gg=(ref['global']-ref['h_mean'])/ref['h_scale']
    inputs={'B2':np.concatenate([hz,hz**2],1),'B3':np.concatenate([hz-gg,hz*gg],1),'Ours':np.concatenate([hz-gz,hz*gz],1)};ranks={}
    for arm,x in inputs.items():
        state=torch.load(DATA/f'F_{arm}_seed0.pt',weights_only=True);scaled=(x-state['x_mean'].numpy())/state['x_scale'].numpy()
        rank=scaled@state['head']['rank.weight'].numpy().T+state['head']['rank.bias'].numpy();ranks[arm]=StandardScaler().fit(rank[aj]).transform(rank)
    rp=PCA(2,random_state=b.SEED).fit(np.concatenate([rank[aj] for rank in ranks.values()]));ix=cohorts['same_task_seven'];js=np.array([lookup[i] for i in ix]);labels=np.array([rows[i]['label_F'] for i in ix])
    fig,axes=plt.subplots(1,3,figsize=(12,4));rank_evidence={}
    for ax,(arm,rank) in zip(axes,ranks.items()):
        xy=rp.transform(rank[js]);rank_evidence[arm]=geometry(rank[js],[rows[i] for i in ix])
        for label in sorted(set(labels)):
            hit=labels==label;ax.scatter(xy[hit,0],xy[hit,1],s=16,alpha=.75,c=colors[label],label=b.CLASSES[label],linewidths=0)
        ax.set_title(arm+' rank-8 correction');ax.set_xlabel('Shared PC1')
    handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=7);fig.suptitle('Equal-capacity correction spaces; fixed same-task held-out codes');fig.tight_layout(rect=[0,.08,1,.94]);fig.savefig(figdir/'correction_space_controls.png',dpi=180,bbox_inches='tight');fig.savefig(figdir/'correction_space_controls.pdf',bbox_inches='tight');plt.close(fig)
    b.save(OUT/'correction_geometry.json',rank_evidence)
    b.save(OUT/'representation_geometry.json',{'projection':'label-blind joint PCA fitted only to fixed training anchors after per-space train scaling','pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'cohorts':evidence})
    # Learning curves expose saturation and overfitting rather than reporting only the best epoch.
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,task in enumerate(['F','D']):
        for arm,color in [('B0','#6b7280'),('B1','#2563eb')]:
            fit=read(OUT/f'{task}_{arm}_training.json');history=fit['history'];ep=[r['epoch'] for r in history]
            axes[row,0].plot(ep,[r['weighted_train_loss'] for r in history],marker='o',label=arm,color=color);axes[row,1].plot(ep,[r['calibration_metric'] for r in history],marker='o',label=arm,color=color)
            selected=next(r for r in history if r['epoch']==fit['selected_epoch']);axes[row,1].scatter([selected['epoch']],[selected['calibration_metric']],marker='*',s=170,color=color,edgecolors='white',zorder=5)
        axes[row,0].set_title(('Family attribution' if task=='F' else 'Detection')+' training loss');axes[row,1].set_title('Calibration '+('Macro-F1' if task=='F' else 'AUROC'))
        for ax in axes[row]:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Full-data learning curves; stars mark calibration-selected checkpoints');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(figdir/'training_curves.png',dpi=180);fig.savefig(figdir/'training_curves.pdf');plt.close(fig)
    b.log('VISUALIZATIONS_DONE')


def uncertainty(rows,raw):
    result={}
    for task,(ev,arms) in raw.items():
        js=np.array([j for j,i in enumerate(ev) if rows[i]['split']=='dev']);subset=[rows[ev[j]] for j in js];y=np.array([r['label_'+task] for r in subset]);groups=defaultdict(list)
        for j,r in enumerate(subset):groups[r['leakage_group']].append(j)
        group_list=list(groups.values());result[task]={}
        def measure(score,ix):return roc_auc_score(y[ix],score[ix].ravel()) if task=='D' else f1_score(y[ix],score[ix].argmax(1),labels=range(7),average='macro',zero_division=0)
        own=[arms[f'Ours_seed{s}'][0][js] for s in [0,1,2]]
        for reference in ['B0','B1','B2','B3','lexical_char','lexical_word']:
            refs=[arms[f'{reference}_seed{s}'][0][js] for s in [0,1,2]] if reference in ['B2','B3'] else [arms[reference][0][js]]
            def delta(ix):return float(np.mean([measure(a,ix) for a in own])-np.mean([measure(a,ix) for a in refs]))
            rng=np.random.default_rng(b.SEED);samples=[]
            for _ in range(200):
                ix=np.concatenate([group_list[j] for j in rng.integers(len(group_list),size=len(group_list))])
                if task=='D' and len(set(y[ix]))<2:continue
                samples.append(delta(ix))
            result[task][reference]={'mean_seed_delta':delta(np.arange(len(js))),'task_cluster_ci95':np.quantile(samples,[.025,.975]).tolist(),'repeats':len(samples),'groups':len(group_list)}
    b.save(OUT/'paired_uncertainty.json',result)
    return result


def source_probe(rows):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    tr=np.array([i for i,r in enumerate(rows) if r['split']=='train']);dev=np.array([i for i,r in enumerate(rows) if r['split']=='dev']);y=np.array([r['label_D'] for r in rows])
    vec=DictVectorizer();x=vec.fit_transform([{'source':r['source_dataset']} for r in rows if r['split']=='train']);clf=LogisticRegression(C=1,class_weight='balanced',max_iter=500).fit(x,y[tr]);score=clf.decision_function(vec.transform([{'source':rows[i]['source_dataset']} for i in dev]))
    b.save(OUT/'source_identity_diagnostic.json',{'purpose':'diagnostic only, forbidden as classifier input','dev_auroc':float(roc_auc_score(y[dev],score))})


def augmentation_support(rows):
    result={}
    for split in ['calibration','dev','unseen_generator','external']:
        result[split]={}
        for family in b.CLASSES:
            members=[r for r in rows if r['split']==split and r['family']==family]
            if not members:continue
            added=sum('local_value_0' in r['stress_view'] and 'local_value_0' not in r['code'] for r in members)
            result[split][family]={'n':len(members),'fixed_stress_rename_marker_added':added,'rate':added/len(members)}
    b.save(OUT/'augmentation_support_diagnostic.json',{'scope':'fixed heldout stress views; not exact dynamic training acceptance rates','diagnostic':'same trigger probability does not guarantee same safety-acceptance rate; fixed rename marker may be an artifact','by_split':result})
    return result


def report(rows,result,ci):
    p=read(OUT/'protocol.json');geometry_data=read(OUT/'representation_geometry.json');doc=ROOT/'d-det/docx/d-det_全量旗舰双任务充分训练与归因空间分析_2026-10-14.md'
    methods=['B0','B1','B2','B3','Ours','lexical_char','lexical_word','length_layout']
    names={'B0':'普通微调','B1':'AMMC 微调','B2':'等容量普通校正','B3':'全局参考校正','Ours':'条件参考＋残差','lexical_char':'字符词法','lexical_word':'词级词法','length_layout':'长度/布局'}
    def val(task,part,arm,view='clean',key=None):
        key=key or ('macro_f1' if task=='F' else 'auroc');arms=[f'{arm}_seed{s}' for s in [0,1,2]] if arm in ['B2','B3','Ours'] else [arm]
        v=[result[task][part][a][view].get(key) for a in arms];return None if any(x is None for x in v) else float(np.mean(v))
    ai_before=geometry_data['cohorts']['same_task_seven']['Before training']['ai_only']['leave_task_out_knn_macro_recall'];ai_after=geometry_data['cohorts']['same_task_seven']['AMMC fine-tuning']['ai_only']['leave_task_out_knn_macro_recall']
    task_variance={stage:record['balanced_variance_fractions']['task'] for stage,record in geometry_data['cohorts']['same_task_seven'].items()}
    lines=['# 全量旗舰来源数据：充分训练与归因空间分析','',
           f'**本轮主效果：条件方法开发归因 Macro-F1={val("F","dev","Ours"):.4f}，普通微调={val("F","dev","B0"):.4f}，普通校正={val("F","dev","B2"):.4f}；条件方法未超过普通校正。检测 AUROC={val("D","dev","Ours"):.4f}，相对 AMMC/普通校正的独有增量也未确认。**', '',
           f'**确定有用的部分：AMMC 将改写归因 F1 从普通微调的{val("F","dev","B0","stress"):.4f}提高到{val("F","dev","B1","stress"):.4f}；固定同题空间中，排除同题邻居后的 AI 内部家族识别由训练前{100*ai_before:.1f}%升至{100*ai_after:.1f}%。模型学到了家族信息，但这不等于干净提取后训练差异。**', '',
           f'**主要缺口：条件方法跨来源归因 F1={val("F","external","Ours",key="supported_macro_f1"):.4f}、检测 AUROC={val("D","external","Ours"):.4f}；仅来源名称的开发检测 AUROC={read(OUT/"source_identity_diagnostic.json")["dev_auroc"]:.4f}。长代码AI覆盖及增强接受率也不均衡，不能将内部高分或空间分离写成后训练因果证明。**', '',
           '仅研究 Human/AI 检测与含 Human 的家族归因。保持“基于代码条件的后训练差异”为研究假设；旗舰 base 近似共同起点是工作假设，本轮不要求不可获得的旗舰谱系数据。家族分离、空间聚类与后训练因果识别不是同一个结论。','',
           f'从 {p["eligible_pool"]:,} 份可用池代码中，{p["rows"]:,} 份进入本轮训练/评估版本，覆盖 {p["task_count"]:,} 个任务。原始主集、旧论文及上一轮小切片结果保持原样。','',
           '**先看学习曲线、同题空间和泛化结果；二维图只作解释，不用于选模型，也不单独作为学到了后训练差异的证明。**','',
           '## 1. 使用量与训练是否充分','',
           '| 类别 | 训练 | 校准 | 开发 | 未见版本AI | 新来源 |','|---|---:|---:|---:|---:|---:|']
    for family in b.CLASSES:lines.append('| '+family+' | '+' | '.join(str(p['counts'][s].get(family,0)) for s in ['train','calibration','dev','unseen_generator','external'])+' |')
    support={}
    human_tasks={r['task_cluster_id'] for r in rows if r['family']=='Human'}
    for family in b.CLASSES:
        members=[r for r in rows if r['family']==family];tasks={r['task_cluster_id'] for r in members}
        support[family]={'samples':len(members),'tasks':len(tasks),'model_identifiers':sorted({r['model_id'] for r in members if r['model_id']}),'same_task_human_tasks':len(tasks&human_tasks)}
    b.save(OUT/'dataset_use_accounting.json',{'families':support,'sources':dict(Counter(r['source_dataset'] for r in rows)),'scope':'parseable complete Python; explicit flagship whitelist; no patches or small generators','reserved_and_excluded':read(OUT/'reserved_and_excluded.json')['excluded']})
    lines += ['', '| 类别 | 本轮总样本 | 不同题目 | 模型标识数 | 有同题Human的题目 |','|---|---:|---:|---:|---:|']
    for family,s in support.items():lines.append(f'| {family} | {s["samples"]} | {s["tasks"]} | '+('—' if family=='Human' else str(len(s['model_identifiers'])))+f' | {s["same_task_human_tasks"]} |')
    lines += ['', '模型标识含版本、产品别名或推理配置，不等于独立模型谱系数量。此轮使用可解析的完整 Python 实现；补丁片段、非Python代码、未确认的模型标识及小模型未混入主比较。旗舰包含相应时期的历史旗舰，不宣称全部来自最新代际。', '',
              '| 公开来源/归档 | 本轮样本 |','|---|---:|']
    source_names={'public_same_task_full_2026-10-07':'BigCodeBench 同题公共代码归档','Zenodo:20823175':'HumanEval/MBPP 公开 H0 代码包（Zenodo 20823175）','GitHub:jamesmurdza/humaneval-results':'GitHub HumanEval 模型结果归档'}
    for source,count in sorted(Counter(r['source_dataset'] for r in rows).items()):lines.append(f'| {source_names.get(source,source)} | {count} |')
    reserved=read(OUT/'reserved_and_excluded.json')
    lines += ['', f'原主集保留185,369条记录；经过既有旗舰/完整代码筛选形成87,005份可用池。本轮另封存{reserved["reserved_count"]:,}份留出版本在学习题目上的输出，并排除{sum(reserved["excluded"].values())}份清理后跨家族完全相同的代码，因此实际使用80,751份。另有整家族留出检测补充，见§7。', '',
              '每个训练周期遍历全部训练样本；没有同题 Human 的代码同样参与分类。类权重平衡不等于真实样本数量平衡。保留版本在学习题目上的代码另行封存，不能放入独立泛化组。Human 是公开参考解/提交候选标签，未逐份认证无 AI 辅助。','',
              '| 任务 | 方法 | 已训轮数 | 选中轮次 | 最佳校准分数 | 停止原因 |','|---|---|---:|---:|---:|---|']
    saturated=[]
    for task in ['F','D']:
        for arm in ['B0','B1']:
            fit=read(OUT/f'{task}_{arm}_training.json');saturated.append(fit['validation_saturated']);lines.append(f'| {"家族归因" if task=="F" else "检测"} | {names[arm]} | {len(fit["history"])} | {fit["selected_epoch"]} | {fit["best_calibration_metric"]:.4f} | '+('校准集提升进入平台' if fit['validation_saturated'] else '达到轮数上限，尚不能确认饱和')+' |')
    lines += ['', '至少4轮、初始上限12轮；连续3轮未出现预设幅度的校准提升时停止。如触及初始上限仍提升，则在读取开发/泛化成绩前继续同一优化器，按12轮增量延长至24/36轮，扩展记录单独保存。归因按校准 Macro-F1、检测按校准 AUROC 选择最大值 checkpoint；平台期与最优轮次分开记录。停止仅表示这套训练规格下验证饱和，不证明全局最优。没有用开发/泛化结果选轮次。','',
              '判别器为 CodeT5-small 全编码器微调，输入上限512 token（前384＋后128），有效 batch32。两个任务分别训练。B0 使用原代码；B1 每轮重新抽取 AMMC 类似增强。编码器学习率3e-5、分类头1e-3、AdamW、梯度裁剪1。条件、普通和全局参考校正共用选中的 B1 编码器，均为 rank8、最多25轮及三个头 seed。','',
              '条件方法沿用已有的任务中心、家族/生成器均衡与语义代理分离思路。对有Human且至少两个AI家族的训练题目，先在每个生成器内平均，再在家族内平均得到中心。归因参考对Human及各AI家族等权；检测参考对Human与AI各分配一半权重，AI内部家族等权。', '',
              r'设微调代码表示为 $h(c)$，去文档字符串、保守改名并AST规范化后的预训练内容代理为 $s(c)$。用五折题目簇交叉拟合的岭回归预测任务中心：$\hat\mu_k(c)=q_k(s(c))$。训练校正头时，每条训练代码只使用未见其题目簇的参考预测；评估时平均五个参考器对应的校正分数。', '',
              r'在训练统计量标准化后，$\phi(h,\hat\mu)=[h-\hat\mu;\ h\odot\hat\mu]$，最终分数为 $\ell(c)=\ell_{\mathrm{B1}}(c)+\frac15\sum_{k=1}^5 a_\omega(\phi(h(c),\hat\mu_k(c)))$。检测优化带类别权重的二元交叉熵，归因优化七类交叉熵；两个任务单独训练。推断只输入代码，不输入题目ID、家族标签或数据来源。', '',
              r'B2使用$[h;h^2]$作为等容量普通校正；B3把条件参考替换成固定全局中心。参考回归正则alpha=100，只在训练题目拟合。内容代理保留部分实现和命名信息，因此不称作纯语义，残差也不预先命名为纯后训练特征。', '',
              '字符/词级控制分别使用TF-IDF 3–5字符片段/1–2词片段、40,000/30,000维词表及LinearSVC(C=1)；长度/布局控制使用字符数、行数、非空行数、tab数、空行数、行长标准差、注释行数、行尾空白数及逻辑回归(C=1)。词表及标准化仅拟合训练集。这些控制读取完整代码并按输出类别平衡，不使用动态增强；神经检测另外平衡AI内部家族权重。最直接的条件独有贡献比较是共用同一B1编码器、同一权重与同一校正头规格的B2/B3/Ours。', '',
              '安全改名会拒绝反射、嵌套作用域或不安全绑定，并回退原代码。相同触发概率不保证相同接受率；本实现使用固定改名符号local_value_0，这个符号本身可能留下增强工件。下表仅描述固定开发改写中的新增符号比例，不冒充每轮动态训练的接受率：', '',
              '| 类别 | 开发改写中新增改名符号的比例 |','|---|---:|']
    aug=read(OUT/'augmentation_support_diagnostic.json')['by_split']['dev']
    for family in b.CLASSES:lines.append(f'| {family} | {100*aug[family]["rate"]:.2f}% |')
    lines += ['', '因此，本轮增强和改写测试提供鲁棒性开发证据，不能独自证明工件已消除。原代码、同题AI内部与跨来源结果需一起判断。', '',
              f'![学习曲线]({(OUT/"figures/training_curves.png").as_posix()})','',
              '## 2. 主比较','', '| 方法 | 归因 Macro-F1 | 改写归因 F1 | 检测 AUROC | 改写检测 AUROC | 同题平均检测 AUROC |','|---|---:|---:|---:|---:|---:|']
    for arm in methods:
        v=[val('F','dev',arm),val('F','dev',arm,'stress'),val('D','dev',arm),val('D','dev',arm,'stress'),val('D','dev',arm,key='task_macro_auroc')]
        lines.append('| '+names[arm]+' | '+' | '.join('NA' if x is None else f'{x:.4f}' for x in v)+' |')
    cohorts=read(OUT/'same_task_classifier_metrics.json')
    lines += ['', '### 同题、类别平衡的留出代码比较', '', '以下使用画图前已经固定的同题代码：每道题每个家族各取一份，不让大量只有Human的题目决定比较。七类组覆盖全部标签；五类组仅比较Human/OpenAI/Claude/Gemini/DeepSeek，报告有支持类别的Macro-F1。相同题目仍可能有历史提示协议差异。', '',
              '| 同题组 | 题目 / 样本 | 方法 | 归因Macro-F1 | 改写归因F1 | 检测AUROC | 同题平均检测AUROC |','|---|---|---|---:|---:|---:|---:|']
    for cname,c in cohorts.items():
        for arm in methods:
            ns=[f'{arm}_seed{s}' for s in [0,1,2]] if arm in ['B2','B3','Ours'] else [arm]
            values=[np.mean([c['metrics'][task][a][view][key] for a in ns]) for task,view,key in [('F','clean','supported_macro_f1'),('F','stress','supported_macro_f1'),('D','clean','auroc'),('D','clean','task_macro_auroc')]]
            lines.append(f'| {"七类" if cname=="same_task_seven" else "五类"} | {c["tasks"]} / {c["rows"]} | {names[arm]} | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', '### 全部开发代码的逐类表现与检测误报', '', '以下回到全部开发代码，不限于上面的同题画图子集。', '', '| 方法 | AI内部家族 Macro-F1 | Human F1 |','|---|---:|---:|']
    for arm in ['B0','B1','B2','B3','Ours']:
        ns=[f'{arm}_seed{s}' for s in [0,1,2]] if arm in ['B2','B3','Ours'] else [arm];human=np.mean([result['F']['dev'][a]['clean']['report']['Human']['f1-score'] for a in ns])
        lines.append(f'| {names[arm]} | {val("F","dev",arm,key="ai_only_macro_f1"):.4f} | {human:.4f} |')
    lines += ['', '| 家族 | 开发样本 | 普通微调 F1 | AMMC F1 | 普通校正 F1 | 条件方法 F1 |','|---|---:|---:|---:|---:|---:|']
    for family in b.CLASSES:
        values=[]
        for arm in ['B0','B1','B2','Ours']:
            ns=[f'{arm}_seed{s}' for s in [0,1,2]] if arm in ['B2','Ours'] else [arm]
            values.append(float(np.mean([result['F']['dev'][a]['clean']['report'][family]['f1-score'] for a in ns])))
        lines.append(f'| {family} | {p["counts"]["dev"].get(family,0)} | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', '| 检测方法 | 原代码 Macro-F1 | 改写 Macro-F1 | AI检出率 | Human误报率 |','|---|---:|---:|---:|---:|']
    for arm in methods:
        values=[val('D','dev',arm,key='macro_f1'),val('D','dev',arm,view='stress',key='macro_f1'),val('D','dev',arm,key='ai_tpr'),val('D','dev',arm,key='human_fpr')]
        lines.append('| '+names[arm]+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', '检测阈值固定为分数0；未根据开发集重新选阈值。三个校正头 seed 的表格取各自指标平均，不以预测集成替代。编码器 seed 只有一个；新全量版本与旧小切片的数据构成不同，不能把跨版本分数差直接解释成扩容收益。','',
              '### 全部开发代码上的条件增量区间', '', '| 条件方法相对对照 | 归因差值及95%区间 | 检测差值及95%区间 |','|---|---:|---:|']
    for arm in ['B0','B1','B2','B3','lexical_char','lexical_word']:
        def fmt(x):return f'{100*x["mean_seed_delta"]:+.2f}pt [{100*x["task_cluster_ci95"][0]:+.4f}, {100*x["task_cluster_ci95"][1]:+.4f}]'
        lines.append('| '+names[arm]+' | '+fmt(ci['F'][arm])+' | '+fmt(ci['D'][arm])+' |')
    lines += ['', '区间按题目及重复内容联合簇重采样200次，接近0的端点仅作开发证据。“等容量”限于校正头规格；条件方法另有五个512→512的岭回归参考器，尚未做全部附加参数严格匹配的对照。正式完整词法/布局/CodeT5融合控制也尚未完成。','',
              '## 3. 泛化与长短代码','', '| 评估 | 方法 | 有支持家族 Macro-F1 | 检测 AUROC |','|---|---|---:|---:|']
    for part in ['unseen_generator','external']:
        for arm in ['B0','B1','B2','B3','Ours','lexical_char']:
            lines.append(f'| {part} | {names[arm]} | {val("F",part,arm,key="supported_macro_f1"):.4f} | {val("D",part,arm):.4f} |')
    lines += ['', '未见版本组留出 GPT-4o 产品线别名、Claude 3.5 Sonnet 20241022 和 DeepSeek-R1，并加入同题开发 Human。新来源组整体留出 HumanEval/MBPP 来源及其同题别名，既有历史旗舰，也有新一代旗舰；不把来源变化全部归因于模型代际。Qwen/Grok 只有一个旗舰成员，不支持其家族内未见版本结论。','',
              '| 长度 | Human / AI | 覆盖家族 | AMMC检测 | 条件检测 | AMMC归因 | 条件归因 |','|---|---|---:|---:|---:|---:|']
    for bucket in ['<=512','513-1024','1025-2048','>2048']:
        entry=result['D']['dev']['B1']['lengths'][bucket];support=entry['support'];human=support.get('Human',0)
        def length_value(task,arm,key):
            ns=[f'{arm}_seed{s}' for s in [0,1,2]] if arm=='Ours' else [arm];values=[result[task]['dev'][a]['lengths'][bucket].get(key) for a in ns]
            return 'NA' if any(x is None for x in values) or task=='F' and len(support)<2 else f'{np.mean(values):.4f}'
        lines.append(f'| {bucket} | {human} / {sum(support.values())-human} | {len(support)}/7 | '+ ' | '.join([length_value('D','B1','auroc'),length_value('D','Ours','auroc'),length_value('F','B1','supported_macro_f1'),length_value('F','Ours','supported_macro_f1')])+' |')
    lines += ['', '长度为截断前 token 数。模型只覆盖512 token，单类或极少样本桶不支持长代码优势结论。','',
              '## 4. 如何阅读归因空间','',
              '所有空间图使用固定的未参与训练代码。主图为相同题目覆盖全部七类的对照：','',
              f'![同题七类归因空间]({(OUT/"figures/family_space_same_task_seven.png").as_posix()})','',
              '从左到右是预训练原始表示、普通微调表示、AMMC 微调表示和预测内容参考残差。前后神经编码统一在同一远端软件实现中完成，绘图和统计在本机完成。各空间按训练样本统计量标准化；共享 PCA 仅由固定训练锚点共同拟合，不使用家族标签选轴。横纵轴不是预先定义的语义/来源坐标；不能把跨面板点的绝对移动解释成真实后训练位移。同题不等于完全相同提示协议，历史提示词差异仍是解释限制。','',
              '条件残差是解释视图，最终分类器还使用原始基线分数及残差/参考交互特征。下面对比实际 rank8 校正空间，图采用预定头seed0；主成绩仍为三个头seed指标平均。条件头在图中显示五个参考器对应的平均线性投影，最终模型则平均五个非线性校正分数。普通校正同样能产生的聚类不能当成条件方法独有贡献：','',
              f'![等容量校正空间]({(OUT/"figures/correction_space_controls.png").as_posix()})','',
              f'![数据来源着色]({(OUT/"figures/source_space.png").as_posix()})','',
              '来源图与家族图使用相同坐标，只改变着色。若分组更贴近采集来源，说明仍可能存在数据捷径。非线性 t-SNE 图仅作补充，各面板独立拟合、坐标不对齐；不得比较跨面板绝对位置。交互图支持家族/来源/长度切换及查看模型、题目：','',
              f'[打开交互归因空间]({(OUT/"figures/attribution_space_interactive.html").as_posix()})','',
              '| 同题七类空间 | 家族轮廓系数 | 排除同题邻居后的家族识别 | AI内部家族识别 |','|---|---:|---:|---:|']
    for stage,record in geometry_data['cohorts']['same_task_seven'].items():lines.append(f'| {stage} | {record["family_cosine_silhouette"]:.4f} | {record["leave_task_out_knn_macro_recall"]:.4f} | {record["ai_only"]["leave_task_out_knn_macro_recall"]:.4f} |')
    lines += ['', '| 同题七类空间 | 家族方差占比 | 题目方差占比 | 交互/剩余方差占比 |','|---|---:|---:|---:|']
    for stage,record in geometry_data['cohorts']['same_task_seven'].items():
        variance=record.get('balanced_variance_fractions')
        if variance:lines.append('| '+stage+' | '+' | '.join(f'{100*variance[key]:.2f}%' for key in ['family','task','interaction'])+' |')
    lines += ['', '上述识别率为高维空间中固定10邻居的描述性读数，查找邻居时排除相同题目；不是新的正式测试分类器。家族轮廓系数越高表示同类相对更近，负值表示家族仍混杂。AI内部读数排除 Human，因此 Human/AI 分开不能单独抬高该项。方差分解使用每题每家族各一份的完整平衡设计；交互/剩余项含实现差异及噪声，不命名为后训练因果方差。','',
              '## 5. 对后训练差异的判读','',
              '本轮依次检查：训练后是否出现可迁移的家族信息；这种信息是否在同题对照、AI内部、未见版本和改写后仍存在；条件参考是否超过普通校正与全局中心；跨来源表现是否保持。只有聚类变清晰，只能说明监督训练重组了表示空间。','']
    for task,title in [('F','家族归因'),('D','检测')]:
        unique=all(ci[task][arm]['task_cluster_ci95'][0]>0 for arm in ['B1','B2','B3','lexical_char','lexical_word'])
        lines.append(f'- {title}：'+('开发集对已测控制的独有增量获得支持，仍需结合泛化。' if unique else '相对所有已测控制的独有增量尚未确认。'))
    source=read(OUT/'source_identity_diagnostic.json');lines += ['',f'仅数据来源身份的开发检测 AUROC 为 {source["dev_auroc"]:.4f}，该信息禁止作为模型输入。这个诊断及跨来源结果共同约束对高分的解释。', '',
              f'同题七类的题目方差从预训练{100*task_variance["Before training"]:.2f}%降至AMMC微调{100*task_variance["AMMC fine-tuning"]:.2f}%，说明普通监督微调本身已削弱题目聚类；条件残差进一步降至{100*task_variance["Conditional residual"]:.2f}%，但其AI内部近邻识别反而低于AMMC表示。当前参考分离具有可见作用，却尚未带来独有分类收益。空间图中家族颜色仍混杂，来源图中分组清晰，必须同时报告这两种现象。', '',
              '下一轮应优先处理三件事：利用现有Human题目补足来源一致的旗舰AI横向覆盖，尤其补充长代码；将固定改名符号改为样本相关的安全名称并检查接受率差异；在保留任务/家族/生成器中心和AMMC经验的前提下改进条件参考的学习目标。当前这套训练规格已进入校准平台，不把继续增加轮数作为默认下一步。本轮未启动新数据生成。', '',
              '在共同 base 近似的研究假设下，本轮最多提供“与后训练行为差异一致的代码条件信号”证据；家族、提示协议、解码设置等仍可能共同影响输出。该判断不需要将 Human 当作 base，也不以缺少旗舰谱系数据作为停止实验的理由。','',
              '## 6. 证据、复现与历史','',
              'protocol.json 记录冻结数据和训练规则；每轮训练/校准曲线、所选 checkpoint、逐样本预测、条件参考数值参数、固定可视化样本ID和原始高维几何指标均保留。原始主数据未改写；不生成新AI数据、不执行下载代码、不读取历史test。格式清理仍统一删除字符串外空白行，原始格式记录继续保留。','',
              '本轮是扩大后的开发实验，公开任务有历史曝光，不冒充全新独立测试。未完全消除的来源、提示词和长度差异单列记录。以前的 v2.2 及后续实验结论不被此轮替换。']
    doc.write_bytes(('\n'.join(lines)+'\n').encode('utf-8'));b.save(OUT/'conclusions.json',{'all_encoders_validation_saturated':all(saturated),'document':str(doc),'distinct_increment':{task:all(ci[task][arm]['task_cluster_ci95'][0]>0 for arm in ['B1','B2','B3','lexical_char','lexical_word']) for task in ['F','D']}})
    return doc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lexical-only',action='store_true');args=parser.parse_args();t.setup();rows=list(b.readgz(DATA/'pilot.jsonl.gz'))
    plan={'frozen_before_eval':True,'same_task_cohort_cap':300,'balanced_visual_family_cap':100,'projection':'label-blind train-anchor shared PCA, supplementary independently fitted t-SNE','seed':b.SEED,'posttraining_interpretation':'working common-base assumption; require same-task AI-internal and heldout/augmentation evidence, plus matched-head controls; clustering alone is not proof'}
    if not (OUT/'interpretation_plan.json').exists():b.save(OUT/'interpretation_plan.json',plan)
    lexical(rows)
    if args.lexical_only:return
    for task in ['F','D']:corrections(rows,task)
    result,raw=evaluate(rows);cohort_performance(rows,raw);ci=uncertainty(rows,raw);source_probe(rows);augmentation_support(rows);visualizations(rows);doc=report(rows,result,ci)
    snapshot=OUT/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ['ddet_sufficient_local_2026_10_14.py','ddet_sufficient_analysis_2026_10_14.py','ddet_t5_local_fast_2026_10_14.py','ddet_two_tasks_local_2026_10_14.py','ddet_flagship_local_2026_10_14.py','ddet_flagship_clean_local_2026_10_14.py','ddet_sufficient_remote_2026_10_14.py','ddet_sufficient_remote_views_2026_10_14.py','ddet_sufficient_receive_2026_10_14.py','ddet_sufficient_finish_2026_10_14.py','ddet_unseen_families_2026_10_14.py','ddet_sufficient_verify_2026_10_14.py']:shutil.copy2(ROOT/'scripts'/name,snapshot/name)
    b.save(OUT/'env.json',{'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'result_document_sha256':b.digest(doc),'source_main_unchanged':b.digest(b.MAIN/'samples.jsonl.gz')==read(OUT/'protocol.json')['source_main_sha256']})
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in {'SHA256SUMS.txt','WORK_STATE.md'} and not p.name.endswith('_features.npz'))
    (OUT/'SHA256SUMS.txt').write_text(''.join(f'{b.digest(p)}  {p.relative_to(OUT).as_posix()}\n' for p in files),encoding='utf-8');b.log('SUFFICIENT_ANALYSIS_DONE',doc)


if __name__=='__main__':main()

