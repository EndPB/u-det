"""Local task-heldout pilot: genuine Human-inclusive family attribution and detection.

No downloads, API generation, source-code execution or historical test access.
This is a small-encoder/256-token development pilot, not the formal base/2048 run.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse, ast, copy, gzip, hashlib, io, json, random, re, sys, time, tokenize
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'd-det/data/a_family_dcan_v1_2026-10-13/main'
DATA = ROOT / 'd-det/data/two_tasks_local_pilot_2026-10-14'
OUT = ROOT / 'd-det/artifacts/two_tasks_local_pilot_2026-10-14'
MODEL = ROOT / 'd-det/models/codet5-small'
CLASSES = ['Human', 'OpenAI', 'Claude', 'Gemini', 'Grok', 'Qwen', 'DeepSeek']
SEED = 20261014
MAXLEN = 256
BATCH = 4
ACCUM = 4
START = time.time()


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def sh(text): return hashlib.sha256(text.encode()).hexdigest()
def save(path, data): path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def readgz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f: yield json.loads(line)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def remove_docs(code):
    """Common implementation view: remove task docstrings, preserve other source text."""
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            # Reject inline docstrings rather than deleting adjacent implementation.
            if first.lineno == node.body[-1].end_lineno or first.col_offset != len(lines[first.lineno-1])-len(lines[first.lineno-1].lstrip()):
                continue
            for i in range(first.lineno-1, first.end_lineno): lines[i] = '\n'
    result = ''.join(lines)
    ast.parse(result)
    return result


def safe_rename(code):
    """Conservative local-only rename, no nested scope/reflection/attributes/strings."""
    tree = ast.parse(code)
    if re.search(r'\b(eval|exec|locals|globals|vars|__dict__|inspect|pickle|getattr|setattr)\b', code):
        return code, 'unsafe_reflection'
    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
        nodes = list(ast.walk(fn))
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                              ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                              ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom)) for n in nodes if n is not fn):
            continue
        args = {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)}
        assigned = sorted({n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} - args)
        for name in assigned:
            if name.startswith('__') or name in {'task_func', 'self', 'cls'}: continue
            occurrences = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == name]
            if len(occurrences) < 2 or any(n.lineno < fn.lineno or n.lineno > fn.end_lineno for n in occurrences): continue
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
            matched = [t for t in tokens if t.type == tokenize.NAME and t.string == name]
            # This excludes attributes, keyword arguments, parameter names and same-name foreign bindings.
            if len(matched) != len(occurrences): continue
            if any(name in t.string for t in tokens if t.type in {tokenize.STRING, tokenize.COMMENT}): continue
            replacement = 'local_value_0'
            if re.search(r'\b' + replacement + r'\b', code): continue
            rewritten = tokenize.untokenize([t._replace(string=replacement) if t.type == tokenize.NAME and t.string == name else t for t in tokens])
            before = copy.deepcopy(tree)
            for n in ast.walk(before):
                if isinstance(n, ast.Name) and n.id == name: n.id = replacement
            if ast.dump(before, include_attributes=False) != ast.dump(ast.parse(rewritten), include_attributes=False):
                return code, 'ast_mismatch'
            return rewritten, 'accepted'
    return code, 'no_safe_candidate'


def augment(code, key, force=False):
    rng = random.Random(int(sh(str(SEED) + key)[:16], 16))
    if not force and rng.random() >= .5: return code, []
    result = code; status = []
    if force or rng.random() < .2:
        toks = list(tokenize.generate_tokens(io.StringIO(result).readline))
        proposal = tokenize.untokenize([t._replace(string='') if t.type == tokenize.COMMENT else t for t in toks])
        if ast.dump(ast.parse(result), include_attributes=False) == ast.dump(ast.parse(proposal), include_attributes=False):
            status.append('comment:accepted' if proposal != result else 'comment:no_change'); result = proposal
    if force or rng.random() < .3:
        proposal = '\n' + '\n'.join(line.rstrip() for line in result.splitlines()) + '\n'
        if ast.dump(ast.parse(result), include_attributes=False) == ast.dump(ast.parse(proposal), include_attributes=False):
            status.append('whitespace:accepted' if proposal != result else 'whitespace:no_change'); result = proposal
    if force or rng.random() < .4:
        result, reason = safe_rename(result); status.append('rename:' + reason)
    return result, status


def content_view(code):
    renamed, _ = safe_rename(code)
    # Content proxy, deliberately not called a pure semantic representation.
    return ast.unparse(ast.parse(renamed))


def prepare():
    DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    metadata = {}
    path = ROOT / 'd-det/data/public_same_task_full_2026-10-07/bigcodebench_task_metadata.jsonl'
    with path.open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line); metadata[r['task_id']] = r
    candidates = defaultdict(list); rejected = Counter()
    for r in readgz(MAIN / 'samples.jsonl.gz'):
        if r['source_dataset'] not in {'BigCodeBench', 'public_same_task_full_2026-10-07'}: continue
        family = r['family_label']
        if family not in CLASSES: continue
        if r.get('identical_code_from_multiple_families'): rejected['raw_cross_family_identical'] += 1; continue
        model = r.get('model_id') or ''
        # This pilot uses larger public model members; the source dataset retains all models.
        lower = model.lower()
        if family != 'Human' and (any(s in lower for s in ['haiku', 'flash', 'gpt-3.5', 'coder', 'tess']) or re.search(r'(^|[-_])mini($|[-_])', lower)): continue
        code = r['code']; task = r.get('task_id') or r.get('source_task_id')
        if task not in metadata: rejected['no_metadata'] += 1; continue
        if family == 'Human':
            info = metadata[task]
            if code.strip() != info['canonical_solution'].strip(): rejected['reference_not_canonical'] += 1; continue
            code = info['code_prompt'] + info['canonical_solution']
        try:
            code = remove_docs(code)
            if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'task_func' for n in ast.parse(code).body):
                rejected['no_task_func'] += 1; continue
        except (SyntaxError, ValueError, tokenize.TokenError): rejected['parse_error'] += 1; continue
        row = {k: r.get(k) for k in ['record_id', 'task_cluster_id', 'task_center_id', 'model_id', 'source_dataset', 'code_sha256', 'prompt_protocol_status']}
        row.update(code=code, family=family, task_id=task, pilot_code_sha256=sh(code), human_reconstructed=family == 'Human')
        candidates[(r['task_cluster_id'], family)].append(row)
    selected = []
    tasks = sorted({t for t, fam in candidates if fam == 'Human'})
    for task in tasks:
        required = ['Human', 'OpenAI', 'Claude', 'Gemini', 'Qwen', 'DeepSeek']
        if any((task, fam) not in candidates for fam in required): rejected['incomplete_core_task'] += 1; continue
        for family in CLASSES:
            group = candidates.get((task, family), [])
            if not group: continue
            # Uniform across available model IDs, then deterministic output choice.
            models = sorted({r['model_id'] or 'Human' for r in group})
            chosen_model = models[int(sh(task + family)[:12], 16) % len(models)]
            eligible = [r for r in group if (r['model_id'] or 'Human') == chosen_model]
            selected.append(min(eligible, key=lambda r: sh(r['record_id'])))
    # Remove contents whose task-harmonized input is now ambiguous across families.
    by_hash = defaultdict(list)
    for r in selected: by_hash[r['pilot_code_sha256']].append(r)
    ambiguous = {h for h, rs in by_hash.items() if len({r['family'] for r in rs}) > 1}
    bad_tasks = {r['task_cluster_id'] for h in ambiguous for r in by_hash[h]}
    selected = [r for r in selected if r['task_cluster_id'] not in bad_tasks]
    rejected['tasks_with_harmonized_cross_family_duplicate'] = len(bad_tasks)
    # Group exact and whitespace-normalized content before splitting tasks.
    parent = {r['task_cluster_id']: r['task_cluster_id'] for r in selected}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    seen = {}
    for r in selected:
        for key in [r['pilot_code_sha256'], sh(re.sub(r'\s+', '', r['code']))]:
            if key in seen:
                a, b = find(r['task_cluster_id']), find(seen[key]); parent[max(a,b)] = min(a,b)
            seen[key] = r['task_cluster_id']
    groups = sorted({find(t) for t in parent}, key=lambda t: sh(str(SEED)+t))
    split = {g: 'train' if i < .7*len(groups) else 'calibration' if i < .8*len(groups) else 'dev' for i, g in enumerate(groups)}
    statuses = Counter()
    for r in selected:
        r['leakage_group'] = find(r['task_cluster_id']); r['split'] = split[r['leakage_group']]
        r['label_F'] = CLASSES.index(r['family']); r['label_D'] = int(r['family'] != 'Human')
        r['train_view'], st = augment(r['code'], r['record_id']) if r['split'] == 'train' else (r['code'], [])
        r['stress_view'], stress_st = augment(r['code'], r['record_id'] + ':stress', force=True)
        r['content_view'] = content_view(r['code'])
        statuses.update(st)
    selected.sort(key=lambda r: (r['split'], r['task_cluster_id'], r['family']))
    with gzip.open(DATA / 'pilot.jsonl.gz', 'wt', encoding='utf-8', newline='\n') as f:
        for r in selected: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    for partition in ['train', 'calibration', 'dev']:
        assert set(r['family'] for r in selected if r['split'] == partition) == set(CLASSES)
    for key in ['task_cluster_id', 'pilot_code_sha256', 'leakage_group']:
        sets = [{r[key] for r in selected if r['split'] == s} for s in ['train','calibration','dev']]
        assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    protocol = {'status': 'frozen_before_training', 'scope': 'exploratory_task_heldout_BigCodeBench_slice_of_current_main',
        'classes': CLASSES, 'rows': len(selected), 'tasks': len({r['task_cluster_id'] for r in selected}),
        'counts': {s: dict(Counter(r['family'] for r in selected if r['split'] == s)) for s in ['train','calibration','dev']},
        'models': dict(Counter(r['model_id'] or 'Human_reference' for r in selected)), 'rejections': dict(rejected),
        'encoder': 'CodeT5-small full encoder fine-tuning, 1 epoch, not CodeT5-base',
        'max_tokens': MAXLEN, 'truncation': 'head 192 plus tail 64', 'batch': BATCH, 'accumulation': ACCUM,
        'encoder_lr': 3e-5, 'head_lr': 1e-3, 'seed_encoder': SEED, 'stage2_seeds': [0,1,2],
        'reference_oof_folds': 5, 'ridge_alpha': 100, 'residual_rank': 8, 'correction_epochs': 25,
        'arms': ['B0_raw_implementation_FT','B1_AMMC_FT','B2_capacity_matched','B3_global_reference','Ours_predicted_content_reference','char_word_lexical','length_layout'],
        'human_reconstruction': 'official code_prompt + canonical_solution, checked against stored reference body',
        'common_input_view': 'remove task docstrings from all classes, retain implementation formatting/comments',
        'ammc': {'original_probability': .5, 'comment': .2, 'whitespace': .3, 'conservative_local_rename': .4, 'statuses': dict(statuses)},
        'source_main_sha256': digest(MAIN/'samples.jsonl.gz'), 'pilot_sha256': digest(DATA/'pilot.jsonl.gz'),
        'script_sha256': digest(Path(__file__)), 'model_sha256': digest(MODEL/'pytorch_model.bin'),
        'prompt_alignment': 'same task; legacy complete/instruct modes not claimed strictly identical',
        'historical_exposure': 'BigCodeBench tasks have been used previously; development evidence only',
        'domain_used': False, 'historical_test_read': False, 'new_independent_test': False,
        'generation_api': False, 'download': False, 'downloaded_code_execution': False,
        'formal_training_ready_claim': False, 'generator_heldout': False}
    save(OUT/'protocol.json', protocol)
    log('PREPARED', len(selected), 'rows', protocol['tasks'], 'tasks', protocol['counts'])
    return selected


def tokenizer():
    from transformers import RobertaTokenizer
    return RobertaTokenizer(vocab=str(MODEL/'vocab.json'), merges=str(MODEL/'merges.txt'),
        unk_token='<unk>', bos_token='<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>',
        pad_token='<pad>', mask_token='<mask>', add_prefix_space=False)


def token_cache(rows, tok):
    cache = {}; lengths = {}
    for r in rows:
        for field in ['code', 'train_view', 'stress_view', 'content_view']:
            text = r[field]; key = sh(text)
            if key not in cache:
                ids = tok.encode(text, add_special_tokens=False)
                lengths[key] = len(ids); cache[key] = ids if len(ids) <= MAXLEN else ids[:192] + ids[-64:]
    save(OUT/'token_lengths.json', {r['record_id']: lengths[sh(r['code'])] for r in rows})
    return cache, lengths


def batch_tokens(rows, ids, field, device):
    parts = [ids[sh(r[field])] for r in rows]; width = max(map(len,parts))
    tokens = torch.zeros((len(parts), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(tokens)
    for i, part in enumerate(parts):
        tokens[i,:len(part)] = torch.tensor(part, device=device); mask[i,:len(part)] = 1
    return tokens, mask


def pooled(model, ids, mask):
    hidden = model(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
    return (hidden.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)


def encode(model, rows, cache, field, device, head=None):
    model.eval()
    if head is not None: head.eval()
    hs = []; ls = []
    with torch.inference_mode():
        for start in range(0,len(rows),8):
            tokens, mask = batch_tokens(rows[start:start+8], cache, field, device)
            with torch.autocast('cuda', dtype=torch.float16):
                h = pooled(model,tokens,mask); logits = head(h) if head is not None else None
            hs.append(h.cpu().numpy())
            if logits is not None: ls.append(logits.float().cpu().numpy())
    h = np.concatenate(hs)
    assert np.isfinite(h).all()
    return h, np.concatenate(ls) if ls else None


def weights(rows, task):
    labels = np.array([r['label_'+task] for r in rows]); counts = np.bincount(labels)
    return (len(rows)/(len(counts)*counts))[labels].astype('float32')


def evaluate(rows, logits, task):
    y = np.array([r['label_'+task] for r in rows]); logits = np.asarray(logits)
    assert np.isfinite(logits).all()
    if task == 'D':
        score = logits.reshape(-1); pred = (score >= 0).astype(int)
        idx = defaultdict(list)
        for i,r in enumerate(rows): idx[r['task_cluster_id']].append(i)
        task_auc = [roc_auc_score(y[ix],score[ix]) for ix in idx.values() if len(set(y[ix]))==2]
        return {'n':len(y), 'auroc':float(roc_auc_score(y,score)), 'task_macro_auroc':float(np.mean(task_auc)) if task_auc else None,
                'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
                'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
                'human_fpr':float(pred[y==0].mean()), 'ai_tpr':float(pred[y==1].mean()), 'threshold_logit':0.0}
    pred = logits.argmax(1); labels = list(range(len(CLASSES)))
    return {'n':len(y), 'macro_f1':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
            'balanced_accuracy':float(np.mean([np.mean(pred[y==label]==label) for label in np.unique(y)])),
            'ai_only_macro_f1':float(f1_score(y[y!=0],pred[y!=0],labels=labels[1:],average='macro',zero_division=0)),
            'report':classification_report(y,pred,labels=labels,target_names=CLASSES,output_dict=True,zero_division=0),
            'confusion':confusion_matrix(y,pred,labels=labels).tolist()}


def fine_tune(rows, cache, task, arm, device, smoke=False):
    from transformers import T5EncoderModel
    seed_all(SEED)
    model = T5EncoderModel.from_pretrained(MODEL, local_files_only=True).to(device)
    model.gradient_checkpointing_enable()
    head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, len(CLASSES) if task=='F' else 1)).to(device)
    train = [i for i,r in enumerate(rows) if r['split']=='train']
    w = weights([rows[i] for i in train],task); wm = {idx:w[j] for j,idx in enumerate(train)}
    opt = torch.optim.AdamW([{'params':model.parameters(),'lr':3e-5},{'params':head.parameters(),'lr':1e-3}],weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda'); model.train(); head.train(); opt.zero_grad(set_to_none=True)
    order = np.random.default_rng(SEED).permutation(train); total = 0.; updates = 0; started = time.time()
    limit = min(len(order), BATCH*20) if smoke else len(order)
    for start in range(0,limit,BATCH):
        ix = order[start:start+BATCH]; chunk = [rows[i] for i in ix]
        tokens,mask = batch_tokens(chunk,cache,'train_view' if arm=='B1' else 'code',device)
        target = torch.tensor([r['label_'+task] for r in chunk],device=device)
        batch_w = torch.tensor([wm[int(i)] for i in ix],device=device)
        with torch.autocast('cuda',dtype=torch.float16):
            h = pooled(model,tokens,mask); logits = head(h)
            loss = torch.nn.functional.cross_entropy(logits,target,reduction='none') if task=='F' else torch.nn.functional.binary_cross_entropy_with_logits(logits.flatten(),target.float(),reduction='none')
            loss = (loss*batch_w).mean()/ACCUM
        if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
        scaler.scale(loss).backward(); total += loss.item()*ACCUM
        if (start//BATCH+1)%ACCUM==0 or start+BATCH>=limit:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1.)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); updates+=1
        if start%(BATCH*100)==0: log(task,arm,'train',start,'/',limit,'loss',round(loss.item()*ACCUM,3),'sec',round(time.time()-started))
    if smoke:
        log('SMOKE',task,arm,'seconds',time.time()-started,'peakMiB',torch.cuda.max_memory_allocated()/1048576)
        del model,head,opt; torch.cuda.empty_cache(); return None
    h, base = encode(model,rows,cache,'code',device,head)
    hs, stress = encode(model,rows,cache,'stress_view',device,head)
    result = {'task':task,'arm':arm,'optimizer_steps':updates,'seconds':time.time()-started,'mean_loss':total/(len(order)/BATCH),
              'peak_cuda_mib':torch.cuda.max_memory_allocated()/1048576}
    np.savez_compressed(OUT/f'{task}_{arm}_features.npz',h=h,base=base,hs=hs,stress=stress,
                        row_ids=np.array([r['record_id'] for r in rows]))
    torch.save({'encoder':model.state_dict(),'head':head.state_dict()},DATA/f'{task}_{arm}.pt')
    save(OUT/f'{task}_{arm}_training.json',result)
    log('TRAIN_DONE',task,arm,result)
    del model,head,opt; torch.cuda.empty_cache()
    return h,base,hs,stress


class Correction(nn.Module):
    def __init__(self, task):
        super().__init__(); self.task=task; self.rank=nn.Linear(1024,8)
        self.output=nn.Sequential(nn.Linear(16,16),nn.GELU(),nn.Linear(16,1)) if task=='D' else nn.Linear(8,len(CLASSES))
        final=self.output[-1] if task=='D' else self.output
        nn.init.zeros_(final.weight); nn.init.zeros_(final.bias)
    def forward(self,x):
        z=self.rank(x)
        if self.task=='D': return self.output(torch.cat([z,z*z],1))
        return self.output(torch.nn.functional.gelu(z))


def references(rows, h, sem, stress_sem, task):
    tr=np.array([i for i,r in enumerate(rows) if r['split']=='train']); groups=np.array([r['leakage_group'] for r in rows])
    targets=np.zeros_like(h); bytask=defaultdict(list)
    for i in tr: bytask[rows[i]['task_cluster_id']].append(i)
    for ix in bytask.values():
        if task=='F': center=h[ix].mean(0)
        else:
            hi=[i for i in ix if rows[i]['family']=='Human']; ai=[i for i in ix if rows[i]['family']!='Human']
            center=.5*h[hi].mean(0)+.5*h[ai].mean(0)
        targets[ix]=center
    oof=np.zeros_like(h); full=[]; stress=[]
    for fold,(fit,hold) in enumerate(GroupKFold(5).split(tr,groups=groups[tr])):
        fit=tr[fit];hold=tr[hold]; ss=StandardScaler().fit(sem[fit]); reg=Ridge(alpha=100).fit(ss.transform(sem[fit]),targets[fit])
        oof[hold]=reg.predict(ss.transform(sem[hold])); full.append(reg.predict(ss.transform(sem))); stress.append(reg.predict(ss.transform(stress_sem)))
    return oof,full,stress,targets


def correction_fit(rows,h,base,hs,stress,sem,stress_sem,task,device):
    tr=np.array([i for i,r in enumerate(rows) if r['split']=='train']);cal=np.array([i for i,r in enumerate(rows) if r['split']=='calibration'])
    oof,refs,stressrefs,targets=references(rows,h,sem,stress_sem,task)
    hsc=StandardScaler().fit(h[tr]); hz=hsc.transform(h); hsz=hsc.transform(hs)
    tr_weights=weights([rows[i] for i in tr],task)
    labels=np.array([r['label_'+task] for r in rows]); logits={}; configs={}
    def make_features(hh,ref):
        rz=hsc.transform(ref)
        return np.concatenate([hh-rz,hh*rz],1).astype('float32')
    for arm in ['B2','B3','Ours']:
        if arm=='B2':
            f=np.concatenate([hz,hz*hz],1).astype('float32');fs=np.concatenate([hsz,hsz*hsz],1).astype('float32')
            trainf=f[tr];evalf=[f];stressf=[fs]
        elif arm=='B3':
            global_ref=np.broadcast_to(targets[tr].mean(0),h.shape)
            f=make_features(hz,global_ref);fs=make_features(hsz,global_ref);trainf=f[tr];evalf=[f];stressf=[fs]
        else:
            trainf=make_features(hz,oof)[tr];evalf=[make_features(hz,m) for m in refs];stressf=[make_features(hsz,m) for m in stressrefs]
        xsc=StandardScaler().fit(trainf); xt=torch.tensor(xsc.transform(trainf),device=device)
        evals=[torch.tensor(xsc.transform(f),device=device) for f in evalf]
        stresses=[torch.tensor(xsc.transform(f),device=device) for f in stressf]
        yt=torch.tensor(labels[tr],device=device,dtype=torch.long);bt=torch.tensor(base[tr],device=device);wt=torch.tensor(tr_weights,device=device)
        btall=torch.tensor(base,device=device);bstress=torch.tensor(stress,device=device)
        for seed in [0,1,2]:
            seed_all(seed);head=Correction(task).to(device);opt=torch.optim.AdamW(head.parameters(),lr=3e-3,weight_decay=.01)
            best=-float('inf');best_state=None;best_epoch=None
            for epoch in range(25):
                head.train();order=np.random.default_rng(seed+epoch).permutation(len(tr))
                for start in range(0,len(tr),256):
                    ix=torch.tensor(order[start:start+256],device=device)
                    pred=bt[ix]+head(xt[ix])
                    loss=torch.nn.functional.cross_entropy(pred,yt[ix],reduction='none') if task=='F' else torch.nn.functional.binary_cross_entropy_with_logits(pred.flatten(),yt[ix].float(),reduction='none')
                    opt.zero_grad(set_to_none=True);(loss*wt[ix]).mean().backward();opt.step()
                head.eval()
                with torch.no_grad(): prediction=btall[cal]+torch.stack([head(f[cal]) for f in evals]).mean(0)
                measured=evaluate([rows[i] for i in cal],prediction.cpu().numpy(),task)
                score=measured['macro_f1' if task=='F' else 'auroc']
                if score>best:best=score;best_state=copy.deepcopy(head.state_dict());best_epoch=epoch+1
            head.load_state_dict(best_state);head.eval()
            with torch.no_grad():
                clean=(btall+torch.stack([head(f) for f in evals]).mean(0)).cpu().numpy()
                stressed=(bstress+torch.stack([head(f) for f in stresses]).mean(0)).cpu().numpy()
            name=f'{arm}_seed{seed}';logits[name]=(clean,stressed);configs[name]={'selected_epoch':best_epoch,'calibration_metric':best}
            log('HEAD_DONE',task,name,'calibration',round(best,4))
            del head,opt
    save(OUT/f'{task}_head_selection.json',configs)
    return logits


def lexical(rows,task):
    tr=np.array([i for i,r in enumerate(rows) if r['split']=='train']);cal=np.array([i for i,r in enumerate(rows) if r['split']=='calibration'])
    y=np.array([r['label_'+task] for r in rows]);texts=[r['code'] for r in rows];stress=[r['stress_view'] for r in rows]
    result={}
    for analyzer,ngram,max_features in [('char',(3,5),40000),('word',(1,2),30000)]:
        vec=TfidfVectorizer(analyzer=analyzer,ngram_range=ngram,max_features=max_features,lowercase=False,sublinear_tf=True,dtype=np.float32)
        x=vec.fit_transform([texts[i] for i in tr]);full=vec.transform(texts);xs=vec.transform(stress)
        clf=LinearSVC(C=1.,class_weight='balanced',max_iter=3000,random_state=SEED).fit(x,y[tr])
        a=clf.decision_function(full);b=clf.decision_function(xs)
        if task=='D':a=a[:,None];b=b[:,None]
        result['lexical_'+analyzer]=(a,b)
    def meta(text):
        lines=text.splitlines();lens=[len(l) for l in lines]
        return np.log1p([len(text),len(lines),sum(bool(l.strip()) for l in lines),sum(c=='\t' for c in text),
                        sum(not l.strip() for l in lines),np.std(lens) if lens else 0,
                        sum(l.lstrip().startswith('#') for l in lines),sum(len(l)-len(l.rstrip()) for l in lines)])
    x=np.array([meta(t) for t in texts]);xs=np.array([meta(t) for t in stress]);sc=StandardScaler().fit(x[tr])
    clf=LogisticRegression(C=1,class_weight='balanced',max_iter=1000).fit(sc.transform(x[tr]),y[tr])
    a=clf.decision_function(sc.transform(x));b=clf.decision_function(sc.transform(xs))
    if task=='D':a=a[:,None];b=b[:,None]
    result['length_layout']=(a,b)
    return result


def paired_ci(rows,ours,baseline,task,repeats=300):
    group=defaultdict(list)
    for i,r in enumerate(rows):group[r['leakage_group']].append(i)
    groups=list(group);rng=np.random.default_rng(SEED);y=np.array([r['label_'+task] for r in rows]);deltas=[]
    for _ in range(repeats):
        idx=np.concatenate([group[groups[i]] for i in rng.integers(len(groups),size=len(groups))])
        if task=='D':
            if len(set(y[idx]))<2:continue
            delta=roc_auc_score(y[idx],ours[idx].flatten())-roc_auc_score(y[idx],baseline[idx].flatten())
        else:
            delta=f1_score(y[idx],ours[idx].argmax(1),labels=list(range(len(CLASSES))),average='macro',zero_division=0)-f1_score(y[idx],baseline[idx].argmax(1),labels=list(range(len(CLASSES))),average='macro',zero_division=0)
        deltas.append(delta)
    return {'delta':float(evaluate(rows,ours,task)['auroc' if task=='D' else 'macro_f1']-evaluate(rows,baseline,task)['auroc' if task=='D' else 'macro_f1']),
            'ci95':np.quantile(deltas,[.025,.975]).tolist(),'bootstrap_units':len(groups),'repeats':len(deltas)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    if not (DATA/'pilot.jsonl.gz').exists() or not (OUT/'protocol.json').exists(): rows=prepare()
    else: rows=list(readgz(DATA/'pilot.jsonl.gz'))
    if args.prepare_only:return
    assert torch.cuda.is_available();device=torch.device('cuda');torch.cuda.reset_peak_memory_stats();seed_all(SEED)
    log('DEVICE',torch.cuda.get_device_name(0));tok=tokenizer();cache,lengths=token_cache(rows,tok)
    if args.smoke:fine_tune(rows,cache,'F','B1',device,smoke=True);return
    save(OUT/'env.json',{'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'script_sha256':digest(Path(__file__))})
    sempath=OUT/'semantic_features.npz'
    if sempath.exists():
        loaded=np.load(sempath);sem=loaded['sem'];stress_sem=loaded['stress_sem']
    else:
        from transformers import T5EncoderModel
        model=T5EncoderModel.from_pretrained(MODEL,local_files_only=True).to(device)
        sem,_=encode(model,rows,cache,'content_view',device)
        stressrows=[dict(r,content_view=content_view(r['stress_view'])) for r in rows]
        for r in stressrows:
            key=sh(r['content_view'])
            if key not in cache:
                ids=tok.encode(r['content_view'],add_special_tokens=False);cache[key]=ids if len(ids)<=MAXLEN else ids[:192]+ids[-64:]
        stress_sem,_=encode(model,stressrows,cache,'content_view',device)
        np.savez_compressed(sempath,sem=sem,stress_sem=stress_sem);del model;torch.cuda.empty_cache();log('SEMANTIC_ENCODED',len(rows))
    dev=np.array([i for i,r in enumerate(rows) if r['split']=='dev']);devrows=[rows[i] for i in dev]
    metrics={}
    for task in ['F','D']:
        arms=lexical(rows,task)
        for arm in ['B0','B1']:
            path=OUT/f'{task}_{arm}_features.npz'
            if path.exists():
                z=np.load(path);h,base,hs,stress=(z[k] for k in ['h','base','hs','stress'])
            else:h,base,hs,stress=fine_tune(rows,cache,task,arm,device)
            arms[arm]=(base,stress)
            if arm=='B1':arms.update(correction_fit(rows,h,base,hs,stress,sem,stress_sem,task,device))
        # Encoder seed is single; head seeds are optimization repeats, not independent datasets.
        for arm in ['Ours','B2','B3']:
            arms[arm+'_mean_head_seeds']=(np.mean([arms[f'{arm}_seed{s}'][0] for s in [0,1,2]],axis=0),np.mean([arms[f'{arm}_seed{s}'][1] for s in [0,1,2]],axis=0))
        results={}
        for name,(clean,stress) in arms.items():
            clean=clean[dev];stress=stress[dev]
            result={'clean':evaluate(devrows,clean,task),'stress':evaluate(devrows,stress,task),'lengths':{}}
            for lo,hi in [(0,128),(128,256),(256,512),(512,10000000)]:
                ix=[i for i,r in enumerate(devrows) if lo<lengths[sh(r['code'])]<=hi]
                if ix and (task=='F' or len({devrows[i]['label_D'] for i in ix})==2):result['lengths'][f'{lo+1}-{hi}']=evaluate([devrows[i] for i in ix],clean[ix],task)
            results[name]=result
            np.savez_compressed(OUT/f'{task}_{name}_dev_predictions.npz',clean=clean,stress=stress,
                labels=np.array([r['label_'+task] for r in devrows]),record_ids=np.array([r['record_id'] for r in devrows]),
                task_ids=np.array([r['task_cluster_id'] for r in devrows]))
        comparisons={name:paired_ci(devrows,arms['Ours_mean_head_seeds'][0][dev],arms[name][0][dev],task) for name in ['B0','B1','B2_mean_head_seeds','B3_mean_head_seeds','lexical_char','lexical_word']}
        metrics[task]={'arms':results,'paired_comparisons':comparisons}
        save(OUT/'metrics.json',{'status':'running','tasks':metrics});log('TASK_COMPLETE',task,{k:round(v['clean']['auroc' if task=='D' else 'macro_f1'],4) for k,v in results.items()})
    save(OUT/'metrics.json',{'status':'completed','tasks':metrics,'runtime_seconds':time.time()-START,'historical_test_read':False,'generation_api':False,'independent_test':False})
    lines=['# 本机双任务快速验证（2026-10-14）','','本轮为当前主集中 BigCodeBench 同题切片的开发实验。不是完整主集成绩，不是独立测试，也不是正式 CodeT5-base 主表。领域不参与。','',
           'Human 参考函数体依据官方 code_prompt 补全；所有来源统一移除题面文档字符串。原始主集未改。Human 为基准参考代码候选，不逐份认证无 AI 辅助。','',
           '| 方法 | 家族归因 Macro-F1 | 归因压力 Macro-F1 | 检测 AUROC | 检测压力 AUROC |','|---|---:|---:|---:|---:|']
    for name in ['length_layout','lexical_char','lexical_word','B0','B1','B2_mean_head_seeds','B3_mean_head_seeds','Ours_mean_head_seeds']:
        f=metrics['F']['arms'][name];d=metrics['D']['arms'][name]
        lines.append(f'| {name} | {f["clean"]["macro_f1"]:.4f} | {f["stress"]["macro_f1"]:.4f} | {d["clean"]["auroc"]:.4f} | {d["stress"]["auroc"]:.4f} |')
    lines+=['','归因为 Human、OpenAI、Claude、Gemini、Grok、Qwen、DeepSeek 七类，Human 完整参加训练和评分。各类支持数、混淆矩阵、AI-only F1、长度分桶、配对区间及逐样本分数在 metrics.json 与 predictions 中。','',
            'B0 为原实现视图微调；B1 为相同预算 AMMC 类增强微调；B2/B3 为等容量普通校正与全局中心；Ours 为训练侧同题中心教师、五折 OOF 预测参考和低秩残差校正，推理不读同题答案或标签。','',
            '限制：本机 CodeT5-small、256 token 头尾覆盖、全编码器仅 1 epoch；一个编码器 seed，三个校正头 seed。不是完整设计的 base/2048/多 seed 实验。词法控制为单独 char/word TF-IDF，未构建完整融合 C-P0。原始 complete/instruct 历史提示模式尚未严格统一；本轮不是未知 generator 泛化。','',
            '预先按任务及相同/空白归一化代码联合隔离 train/calibration/dev；所有模型共用划分。基线、参考器、标准化和词法拟合不使用 dev。calibration 只选择校正头 checkpoint。旧 test 未读取。','',
            '代码条件残差是来源差异的建模工具，不证明纯后训练因果。点估计及配对 CI 必须一起解释，不以点估计最高宣称稳定增量。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'commands.txt').write_text('python scripts/ddet_two_tasks_local_2026_10_14.py --prepare-only\npython scripts/ddet_two_tasks_local_2026_10_14.py --smoke\npython scripts/ddet_two_tasks_local_2026_10_14.py\n',encoding='utf-8')
    (OUT/'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt'),encoding='utf-8')
    log('BATCH_DONE','seconds',time.time()-START)


if __name__=='__main__':main()
