#!/usr/bin/env python3
"""Full local v4 diagnostic: matched raw/masked lexical and local GPU heads.

Exploratory: v4 was selected after inspecting dev, and the formal gate remains
revise_data. This script does not reopen or overwrite any closed old batch.
"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
from scipy.sparse import hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.svm import LinearSVC

from h3_gpu_diagnostic_batch_2026_10_11 import SharedHeads, encode_unique, make_tokenizer, set_seed
from h3_hardneg_revision_v3 import mask_code

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'd-det/data/h3_stacad_revision_v4'
MODEL = ROOT / 'd-det/models/codet5-small'
OUT = ROOT / 'd-det/artifacts/h3_v4_local_experiment_2026-10-11'
SEEDS = [20261011, 20261012, 20261013]
JOBS = ['detection_only', 'joint', 'joint_invariance']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def read(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(x) for x in f if x.strip()]


def task_aucs(rows, scores):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r['task_id']].append(i)
    vals = {}
    for task, idx in sorted(groups.items()):
        y = np.asarray([rows[i]['label'] for i in idx])
        if len(set(y)) == 2:
            vals[task] = float(roc_auc_score(y, np.asarray(scores)[idx]))
    return vals


def summary(rows, scores, threshold):
    y = np.asarray([r['label'] for r in rows])
    tasks = task_aucs(rows, scores)
    return {'rows': len(rows), 'all_tasks': len({r['task_id'] for r in rows}),
            'tasks_with_both_labels': len(tasks), 'row_auroc': float(roc_auc_score(y, scores)),
            'task_macro_auroc': float(np.mean(list(tasks.values()))),
            'balanced_accuracy': float(balanced_accuracy_score(y, np.asarray(scores) >= threshold))}


def save_scores(name, rows, scores, **extra):
    p = OUT / 'scores' / f'{name}.npz'
    p.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(p, scores=np.asarray(scores), labels=np.asarray([r['label'] for r in rows]),
                        row_ids=np.asarray([r['row_id'] for r in rows]),
                        task_ids=np.asarray([r['task_id'] for r in rows]), **extra)
    return {'path': p.relative_to(OUT).as_posix(), 'sha256': digest(p)}


def lexical(train, evals, fold, view):
    transform = (lambda s: s) if view == 'raw' else mask_code
    text = [transform(r['code']) for r in train]
    xs, xe = [], {k: [] for k in evals}
    for analyzer, ngram in [('char_wb', (2, 5)), ('word', (1, 2))]:
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=5, max_features=100000,
                              lowercase=False, sublinear_tf=True, token_pattern=r'[A-Za-z_][A-Za-z0-9_]*')
        xs.append(vec.fit_transform(text))
        for name, rows in evals.items():
            xe[name].append(vec.transform([transform(r['code']) for r in rows]))
    clf = LinearSVC(C=1., max_iter=3000, random_state=SEEDS[0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        clf.fit(hstack(xs).tocsr(), [r['label'] for r in train])
    result, scores = {}, {}
    for name, rows in evals.items():
        s = clf.decision_function(hstack(xe[name]).tocsr())
        scores[name] = s
        result[name] = {**summary(rows, s, 0),
                        'score_file': save_scores(f'{fold}_lexical_{view}_{name}', rows, s)}
    result['solver_iterations'] = int(clf.n_iter_)
    result['warnings'] = [str(w.message) for w in caught]
    return result, scores


def paired(task_scores, control, seed):
    # Cluster unit is a task, never a row or repeated source fold.
    keys = sorted(set(control).intersection(*(set(s) for s in task_scores)))
    a = np.mean([[s[k] for k in keys] for s in task_scores], axis=0)
    b = np.asarray([control[k] for k in keys])
    delta = a - b
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(axis=1)
    return {'tasks': len(keys), 'mean_seed_task_macro_auroc': float(a.mean()),
            'control_task_macro_auroc': float(b.mean()), 'delta': float(delta.mean()),
            'task_bootstrap_ci95': np.quantile(boot, [.025, .975]).tolist(),
            'bootstrap_repeats': 2000, 'seed': seed,
            'interpretation': 'exploratory paired task bootstrap conditional on these seeds; not seed-population CI'}


def verify_inputs(train, challenge, clean, variants, proof):
    assert {r['task_split'] for r in train} == {'train'}
    assert {r['task_split'] for r in challenge + clean} == {'dev'}
    assert not ({r['task_id'] for r in train} & {r['task_id'] for r in clean})
    assert {r['row_id'] for r in challenge} <= {r['row_id'] for r in clean}
    parents = {r['row_id']: r for r in train}
    accepted = {(r['parent_row_id'], r.get('after_sha256')) for r in proof if r['status'] == 'accepted'}
    for r in train + clean + variants:
        assert hashlib.sha256(r['code'].encode('utf8')).hexdigest() == r['code_sha256'], r['row_id']
    for v in variants:
        assert v['parent_row_id'] in parents
        assert v['task_split'] == 'train'
        assert v['task_id'] == parents[v['parent_row_id']]['task_id']
        assert (v['parent_row_id'], v['code_sha256']) in accepted
    for rows in [train, challenge]:
        counts = Counter(r['task_id'] for r in rows)
        assert set(counts.values()) == {4}
        ai = Counter(r['generator'] for r in rows if r['label'] == 1)
        assert max(ai.values()) - min(ai.values()) <= 1
    return {'passed': True, 'train_rows': len(train), 'challenge_rows': len(challenge),
            'clean_rows': len(clean), 'verified_train_variants': len(variants), 'test_opened': False,
            'variant_proof_file_sha256': digest(ROOT / 'd-det/artifacts/h3_local_revision_v1_2026-10-10/variant_records.jsonl')}


def select_ammc_views(base, variants):
    """Select a deterministic, label/transform-balanced AMMC-like view mix.

    We keep the original rows and add one balanced quota for each accepted
    comment/whitespace/rename transform.  Quotas are capped by the smallest
    transform pool within each label, and AI rows are round-robin sampled over
    the generators present in this fold.  This prevents the augmentation
    indicator itself from becoming an AI shortcut.
    """
    base_ids = {r['row_id'] for r in base}
    cand = [v for v in variants if v['parent_row_id'] in base_ids]
    transforms = ['comment', 'whitespace', 'rename']
    selected = []
    audit = {'base_rows': len(base), 'candidate_variants': len(cand), 'selected_by_label_transform': {},
             'selected_by_label_generator_transform': {}, 'policy': 'equal transform quota; AI generator round-robin'}
    for label in [0, 1]:
        pools = {t: [v for v in cand if v['label'] == label and v['transform'] == t]
                 for t in transforms}
        target = min(len(pools[t]) for t in transforms)
        for t in transforms:
            groups = defaultdict(list)
            for v in sorted(pools[t], key=lambda x: x['row_id']):
                groups[v['generator']].append(v)
            chosen = []
            gens = sorted(groups)
            while len(chosen) < target and gens:
                progressed = False
                for g in gens:
                    if groups[g] and len(chosen) < target:
                        chosen.append(groups[g].pop(0))
                        progressed = True
                if not progressed:
                    break
            selected.extend(chosen)
            audit['selected_by_label_transform'][f'{label}:{t}'] = len(chosen)
            for v in chosen:
                key = f"{label}:{v['generator']}:{t}"
                audit['selected_by_label_generator_transform'][key] = audit['selected_by_label_generator_transform'].get(key, 0) + 1
    return list(base) + selected, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--controls-only', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    paths = {'train': DATA / 'hardneg_train_v4.jsonl', 'hardneg': DATA / 'hardneg_dev_v4.jsonl',
             'clean': ROOT / 'd-det/data/h3_stacad_revision_v1/dev_clean.jsonl',
             'variants': DATA / 'train_variants_v4.jsonl'}
    train, challenge, clean, variants = [read(paths[k]) for k in ['train', 'hardneg', 'clean', 'variants']]
    proof = read(ROOT / 'd-det/artifacts/h3_local_revision_v1_2026-10-10/variant_records.jsonl')
    write(OUT / 'input_audit.json', verify_inputs(train, challenge, clean, variants, proof))
    gens = sorted({r['generator'] for r in train if r['label'] == 1})
    folds = {'fold_0': gens[:3], 'fold_1': gens[3:]}
    config = {'schema': 'h3_v4_local_exploratory_v1', 'claim_scope': 'task-heldout diagnostic, NOT formal/project-heldout',
              'formal_gpu_ready': False, 'gate': 'revise_data', 'post_dev_selection': True,
              'input_sha256': {k: digest(p) for k, p in paths.items()}, 'script_sha256': digest(Path(__file__)),
              'helper_sha256': {n: digest(ROOT / 'scripts' / n) for n in ['h3_gpu_diagnostic_batch_2026_10_11.py', 'h3_hardneg_revision_v3.py']},
              'model_sha256': digest(MODEL / 'pytorch_model.bin'), 'seeds': SEEDS, 'folds': folds,
              'neural_runs': '18 generator-heldout detection runs + 3 unique all-generator source runs; AMMC-like balanced train views',
              'encoder': 'frozen CodeT5-small; 512 head/tail tokens; mean pooling; existing char cap=24000',
              'epochs': 3, 'learning_rate': .002, 'batch_size': 256, 'encoding_batch_size': 8,
              'primary_control': 'raw char_wb(2,5)+word(1,2), min_df=5, 100k features each, LinearSVC C=1',
              'secondary_control': 'same protocol on regex-masked code (diagnostic, not a syntax-safe transform)',
              'ammc_like_training': 'original train rows plus deterministic label/transform-balanced accepted comment, whitespace, and AST-safe rename views; dev/clean untouched',
              'success_rule': '>=.01 mean task-AUC delta vs raw lexical, paired CI lower>0, both folds positive, both dev views; otherwise increment_not_observed',
              'test_read': False, 'generation': False, 'weights_downloaded': False, 'code_execution': False}
    write(OUT / 'config.json', config)  # Freeze before any fitting/encoding.
    env = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
           'numpy': np.__version__, 'sklearn': sklearn.__version__, 'cuda_available': torch.cuda.is_available(),
           'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
    write(OUT / 'env.json', env)
    controls, control_scores, splits = {}, {}, {}
    for fold, held in folds.items():
        base_fit = [r for r in train if r['label'] == 0 or r['generator'] not in held]
        fit, aug_audit = select_ammc_views(base_fit, variants)
        write(OUT / f'augmentation_{fold}.json', aug_audit)
        evals = {name: [r for r in rows if r['label'] == 0 or r['generator'] in held]
                 for name, rows in [('hardneg', challenge), ('clean', clean)]}
        splits[fold] = (fit, evals)
        controls[fold], control_scores[fold] = {}, {}
        for view in ['raw', 'masked']:
            print('CONTROL', fold, view, 'fit', len(fit), flush=True)
            controls[fold][view], control_scores[fold][view] = lexical(fit, evals, fold, view)
            write(OUT / 'control_metrics.json', controls)
    if args.controls_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('local CUDA required; controls completed but neural jobs not started')
    from transformers import T5EncoderModel
    torch.set_num_threads(4)
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    model = T5EncoderModel.from_pretrained(str(MODEL), local_files_only=True, dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    vectors, encode_info = encode_unique(train + clean + variants, make_tokenizer(), model, device, 8, 512)
    del model
    torch.cuda.empty_cache()
    write(OUT / 'encoding.json', encode_info)
    source_map = {g: i for i, g in enumerate(gens)}
    by_parent = defaultdict(list)
    for v in sorted(variants, key=lambda r: r['row_id']):
        by_parent[v['parent_row_id']].append(v)

    def tensors(rows):
        x = torch.tensor(np.stack([vectors[r['code_sha256']] for r in rows]), device=device)
        y = torch.tensor([r['label'] for r in rows], device=device, dtype=torch.float32)
        s = torch.tensor([source_map.get(r['generator'], -1) for r in rows], device=device)
        return x, y, s

    def fit_head(fit, job, seed):
        set_seed(seed)
        head = SharedHeads(len(next(iter(vectors.values()))), len(gens)).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=.002, weight_decay=1e-4)
        x, y, s = tensors(fit)
        pairs = [(r, by_parent[r['row_id']][0]) for r in fit if by_parent[r['row_id']]]
        if pairs:
            px, _, _ = tensors([p[0] for p in pairs])
            vx, _, _ = tensors([p[1] for p in pairs])
        history = []
        for epoch in range(3):
            order = np.random.permutation(len(fit))
            head.train()
            losses = []
            for start in range(0, len(fit), 256):
                idx = torch.as_tensor(order[start:start+256], device=device)
                det, src, _ = head(x[idx])
                keep = s[idx] >= 0
                if job == 'source_only':
                    loss = F.cross_entropy(src, s[idx])  # fit has AI rows only.
                else:
                    loss = F.binary_cross_entropy_with_logits(det, y[idx])
                    if job != 'detection_only' and bool(keep.any()):
                        loss = loss + .20 * F.cross_entropy(src[keep], s[idx][keep])
                    if job == 'joint_invariance' and pairs:
                        pi = torch.as_tensor(np.random.randint(0, len(pairs), min(256, len(pairs))), device=device)
                        loss = loss + .10 * F.mse_loss(head(px[pi])[0].sigmoid(), head(vx[pi])[0].sigmoid())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            history.append({'epoch': epoch+1, 'mean_batch_loss': float(np.mean(losses))})
        head.eval()
        return head, history, len(pairs)

    results, neural_tasks = [], defaultdict(list)
    for fold, (fit, evals) in splits.items():
        for job in JOBS:
            for seed in SEEDS:
                print('RUN', fold, job, seed, flush=True)
                head, history, pair_count = fit_head(fit, job, seed)
                one = {'fold': fold, 'job': job, 'seed': seed, 'train_rows': len(fit),
                       'variant_parent_pairs': pair_count, 'history': history, 'eval': {},
                       'source_eval': 'heldout generator classes unsupported; see unique source_only task-heldout runs'}
                for name, rows in evals.items():
                    x, _, _ = tensors(rows)
                    with torch.inference_mode():
                        scores = torch.cat([head(chunk)[0].sigmoid() for chunk in x.split(256)]).cpu().numpy()
                    one['eval'][name] = {**summary(rows, scores, .5),
                                         'score_file': save_scores(f'{fold}_{job}_{seed}_{name}', rows, scores)}
                    neural_tasks[(fold, job, name)].append(task_aucs(rows, scores))
                results.append(one)
                write(OUT / 'neural_metrics.json', results)
                del head
    source_runs = []
    fit_ai = [r for r in train if r['label'] == 1]
    source_fit, source_aug_audit = select_ammc_views(fit_ai, variants)
    write(OUT / 'source_augmentation.json', source_aug_audit)
    for seed in SEEDS:
        print('RUN source_only unique task-heldout', seed, flush=True)
        head, history, _ = fit_head(source_fit, 'source_only', seed)
        result = {'seed': seed, 'split': 'task-heldout-all-seven-generators', 'train_rows': len(source_fit), 'history': history, 'eval': {}}
        for name, rows in [('hardneg', challenge), ('clean', clean)]:
            rows = [r for r in rows if r['label'] == 1]
            x, _, s = tensors(rows)
            with torch.inference_mode():
                logits = torch.cat([head(chunk)[1] for chunk in x.split(256)]).cpu().numpy()
            source_labels = s.cpu().numpy()
            acc = float((logits.argmax(1) == source_labels).mean())
            result['eval'][name] = {'accuracy': acc, 'rows': len(rows), 'chance': 1/7,
                                    'score_file': save_scores(f'source_only_{seed}_{name}', rows, logits, source_labels=source_labels)}
        source_runs.append(result)
    write(OUT / 'source_metrics.json', source_runs)
    comparisons = {}
    for fold, (_, evals) in splits.items():
        comparisons[fold] = {}
        for job in JOBS:
            comparisons[fold][job] = {}
            for name, rows in evals.items():
                control = task_aucs(rows, control_scores[fold]['raw'][name])
                comparisons[fold][job][name] = paired(neural_tasks[(fold, job, name)], control, SEEDS[0])
    passed = {}
    for job in JOBS:
        checks = [comparisons[f][job][v] for f in folds for v in ['hardneg', 'clean']]
        passed[job] = all(c['delta'] >= .01 and c['task_bootstrap_ci95'][0] > 0 for c in checks)
    metrics = {'status': 'exploratory_completed', 'formal_gpu_ready': False, 'gate': 'revise_data',
               'detection_runs': len(results), 'unique_source_runs': len(source_runs), 'controls': controls,
               'comparisons_vs_raw_lexical': comparisons, 'success_by_job': passed,
               'decision': 'candidate_increment_observed_requires_independent_validation' if any(passed.values()) else 'increment_not_observed_close_v4_frozen_candidate',
               'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
               'cuda_peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2}
    write(OUT / 'metrics.json', metrics)
    lines = ['# H3 v4 本机完整对照实验', '',
             '本轮为探索性诊断：v4 使用了 dev 信息选样，正式闸门仍为 revise_data；不声称项目留出或确认性主结果。', '',
             '18 个检测运行 + 3 个独立 source-only 运行；训练集使用 label/transform-balanced 的 AMMC-like comment/whitespace/rename 视图，原始 clean-dev 与 v4 hard-negative 保持未增强。', '',
             '| fold | view | raw lexical row / task | detection task | joint task | invariance task |',
             '|---|---|---:|---:|---:|---:|']
    for f in folds:
        for view in ['hardneg', 'clean']:
            ctrl = controls[f]['raw'][view]
            vals = [comparisons[f][job][view]['mean_seed_task_macro_auroc'] for job in JOBS]
            lines.append(f"| {f} | {view} | {ctrl['row_auroc']:.4f} / {ctrl['task_macro_auroc']:.4f} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} |")
    lines += ['', '结论：`' + metrics['decision'] + '`。', '',
              '所有增量在同折、同一行集合与同一 task-macro 指标下比较，CI 使用 task 成对 bootstrap。',
              'AMMC-like 训练视图只来自 train parent，按 Human/AI 与 transform 等额采样，并在 AI generator 间轮转；source-only 也使用对应 AI balanced views；没有把变体加入 dev/clean，也没有把 transform mask 作为输入。',
              '掩码控制仅为次要诊断，不能替代 raw 词法强控制。source-only 只计 3 个独立 seed，human 不计作未知 generator。',
              '逐行预测、输入审计、变体 accepted 证明、损失历史与环境均保存；test 未读取，未生成代码，未下载模型，未执行数据集代码。']
    (OUT / 'report.md').write_bytes(('\n'.join(lines)+'\n').encode('utf8'))
    (OUT / 'commands.txt').write_bytes(b'python scripts/h3_v4_local_experiment.py\n')
    (OUT / 'SHA256SUMS.txt').write_bytes(''.join(f'{digest(p)}  {p.relative_to(OUT).as_posix()}\n' for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['SHA256SUMS.txt','run.log.txt']).encode('utf8'))
    print(json.dumps({'decision': metrics['decision'], 'runs': len(results)+len(source_runs),
                      'peak_vram_mib': metrics['cuda_peak_allocated_mib']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
