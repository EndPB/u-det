"""Independent saved-score and split checks; no retraining or test-set tuning."""
from pathlib import Path
from collections import defaultdict
import gzip,hashlib,json,re
import numpy as np
from sklearn.metrics import f1_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'d-det/artifacts/flagship_sufficient_2026-10-14';DATA=ROOT/'d-det/data/flagship_sufficient_2026-10-14'


def read(path):return json.loads(path.read_text(encoding='utf-8'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=read(OUT/'protocol.json');assert sha(DATA/'pilot.jsonl.gz')==p['data_sha256']
    assert sha(ROOT/'d-det/data/a_family_dcan_v1_2026-10-13/main/samples.jsonl.gz')==p['source_main_sha256']
    with gzip.open(DATA/'pilot.jsonl.gz','rt',encoding='utf-8') as handle:rows=[json.loads(line) for line in handle]
    ids={r['record_id']:r for r in rows};assert len(ids)==len(rows)==p['rows'];assert len({r['task_cluster_id'] for r in rows})==p['task_count']
    registry=read(ROOT/'d-det/artifacts/flagship_two_tasks_clean_2026-10-14/flagship_registry_and_pool.json');groups=defaultdict(set)
    for r in rows:
        groups[r['split']].add(r['leakage_group'])
        if r['family']!='Human':assert r['model_id'] in registry['allowed'][r['family']]
        if r['split'] in ['train','calibration']:assert r['model_id'] not in p['heldout_model_ids']
    for first in ['train','calibration']:
        for second in groups:
            if first!=second:assert not groups[first]&groups[second]
    reserved=read(OUT/'reserved_and_excluded.json');assert len(rows)+reserved['reserved_count']+sum(reserved['excluded'].values())==p['eligible_pool']
    m=read(OUT/'metrics.json');checked=0
    for task,partitions in m.items():
        for partition,arms in partitions.items():
            seen_ids=None
            for arm,entry in arms.items():
                z=np.load(OUT/f'{task}_{arm}_{partition}_predictions.npz',allow_pickle=False);rids=z['record_ids'].tolist()
                if seen_ids is None:seen_ids=rids
                assert rids==seen_ids;assert len(set(rids))==len(rids)
                subset=[ids[rid] for rid in rids];assert np.array_equal(z['labels'],[r['label_'+task] for r in subset])
                for r in subset:
                    assert r['split']==partition or partition=='unseen_generator' and r['split']=='dev' and r['family']=='Human'
                for view in ['clean','stress']:
                    score=z[view];assert np.isfinite(score).all();y=z['labels']
                    measured=roc_auc_score(y,score.ravel()) if task=='D' else f1_score(y,score.argmax(1),labels=range(7),average='macro',zero_division=0)
                    claimed=entry[view]['auroc' if task=='D' else 'macro_f1'];assert abs(measured-claimed)<1e-12
                checked+=1
    with gzip.open(DATA/'unseen_families.jsonl.gz','rt',encoding='utf-8') as handle:extra=[json.loads(line) for line in handle]
    supplement_manifest=read(OUT/'unseen_families_manifest.json');assert sha(DATA/'unseen_families.jsonl.gz')==supplement_manifest['data_sha256']
    assert not {r['family'] for r in extra if r['family']!='Human'}&{r['family'] for r in rows}
    learning_groups={r['leakage_group'] for r in rows if r['split'] in {'train','calibration'}}
    parent={r['task_cluster_id']:r['task_cluster_id'] for r in rows+extra}
    def find(key):
        while parent[key]!=key:parent[key]=parent[parent[key]];key=parent[key]
        return key
    hashes={}
    for r in rows+extra:
        for code in [r['code'],re.sub(r'\s+','',r['code'])]:
            key=hashlib.sha256(code.encode()).digest()
            if key in hashes:
                first,second=find(r['task_cluster_id']),find(hashes[key]);parent[max(first,second)]=min(first,second)
            hashes[key]=r['task_cluster_id']
    learning_components={find(r['task_cluster_id']) for r in rows if r['split'] in {'train','calibration'}}
    extra_scores=np.load(OUT/'unseen_families_all_predictions.npz',allow_pickle=False)
    assert extra_scores['record_ids'].tolist()==[r['record_id'] for r in extra]
    assert np.array_equal(extra_scores['labels'],[r['label_D'] for r in extra]);supplement_checked=0
    human_tasks={r['task_cluster_id'] for r in extra if r['family']=='Human' and r['is_unique_sample']}
    for family,record in read(OUT/'unseen_families_metrics.json').items():
        ai=[i for i,r in enumerate(extra) if r['family']==family and r['role']=='unseen_task' and r['is_unique_sample'] and r['task_cluster_id'] in human_tasks]
        tasks={extra[i]['task_cluster_id'] for i in ai}
        human=[i for i,r in enumerate(extra) if r['family']=='Human' and r['is_unique_sample'] and r['task_cluster_id'] in tasks]
        ix=np.array(ai+human,dtype=int);assert len(ai)>0 and len(human)>0
        assert not {extra[i]['leakage_group'] for i in ai}&learning_groups
        assert not {find(extra[i]['task_cluster_id']) for i in ix}&learning_components
        assert not {extra[i]['task_cluster_id'] for i in ix}&{r['task_cluster_id'] for r in rows if r['split'] in {'train','calibration'}}
        for arm,views in record['independent'].items():
            for view,entry in views.items():
                measured=roc_auc_score(extra_scores['labels'][ix],extra_scores[f'{arm}_{view}'][ix].ravel())
                assert abs(measured-entry['auroc'])<1e-12;assert entry['n']==len(ix);supplement_checked+=1
    cohorts=read(OUT/'visual_cohorts.json')
    for record in cohorts.values():assert all(ids[rid]['split'] in {'dev','unseen_generator','external'} for rid in record['record_ids'])
    focused_checked=0
    for name,record in read(OUT/'same_task_classifier_metrics.json').items():
        assert record['record_ids']==cohorts[name]['record_ids']
        for task,arms in record['metrics'].items():
            for arm,views in arms.items():
                z=np.load(OUT/f'{task}_{arm}_dev_predictions.npz',allow_pickle=False);lookup={rid:i for i,rid in enumerate(z['record_ids'].tolist())};ix=np.array([lookup[rid] for rid in record['record_ids']]);y=z['labels'][ix]
                for view,entry in views.items():
                    score=z[view][ix];measured=roc_auc_score(y,score.ravel()) if task=='D' else f1_score(y,score.argmax(1),labels=sorted(set(y)),average='macro',zero_division=0)
                    assert abs(measured-entry['auroc' if task=='D' else 'supported_macro_f1'])<1e-12;focused_checked+=1
    projection=read(OUT/'projection_fit.json');assert projection['family_labels_used_for_axes'] is False;assert all(ids[rid]['split']=='train' for rid in projection['fit_record_ids'])
    for path in OUT.rglob('*.json'):read(path)
    from PIL import Image
    images={}
    for path in (OUT/'figures').glob('*.png'):
        with Image.open(path) as image:image.verify()
        with Image.open(path) as image:images[path.name]=list(image.size)
    env=read(OUT/'env.json');docs=[path for path in (ROOT/'d-det/docx').glob('*2026-10-14.md') if sha(path)==env['result_document_sha256']];assert len(docs)==1
    proof={'rows':len(rows),'prediction_files_independently_recomputed':checked,'focused_same_task_metrics_recomputed':focused_checked,'supplementary_paired_aurocs_recomputed':supplement_checked,'supplementary_families_absent_from_training':True,'supplementary_paired_tasks_absent_from_learning':True,'source_main_unchanged':True,'train_calibration_eval_group_overlap':0,'heldout_versions_absent_from_learning':True,'fixed_visual_cohorts_heldout':True,'projection_fit_train_only':True,'images':images,'document_sha256_verified':True}
    (OUT/'independent_verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(proof,ensure_ascii=False))


if __name__=='__main__':main()
