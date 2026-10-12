"""Materialize large A/B/C train/validation sets and run rapid CPU baselines.

Uses the already downloaded official AICD parquet; test files are never opened.
Labels remain the released integer IDs, with semantic names separately tracked.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import hashlib
import heapq
import json
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from format_channel_local_audit_2026_10_11 import format_features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'd-det/data/acl_attribution_collection_v1/raw/aicd'
DATA = ROOT / 'd-det/data/mainline_abc_200k_2026-10-12'
OUT = ROOT / 'd-det/artifacts/mainline_abc_200k_2026-10-12'
TASKS = {'A': ('T1', 2), 'B': ('T2', 12), 'C': ('T3', 4)}
SEED = 20261012


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, ensure_ascii=False, indent=2)+'\n').encode())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def stream(paths):
    for path in paths:
        assert not path.name.startswith('test')
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=['code', 'label']):
            data = batch.to_pydict()
            for i, (code, label) in enumerate(zip(data['code'], data['label'])):
                yield code, int(label), str(path.relative_to(ROOT)).replace('\\', '/'), offset+i
            offset += len(batch)


def quotas(counts, target):
    out = {k: 0 for k in counts}
    remaining = min(target, sum(counts.values()))
    while remaining:
        active = [k for k in counts if out[k] < counts[k]]
        n = max(1, remaining // len(active))
        for k in sorted(active):
            take = min(n, counts[k]-out[k], remaining)
            out[k] += take
            remaining -= take
            if not remaining: break
    return out


def choose(paths, target, excluded):
    # Count labels in a cheap column-only pass, then keep the smallest stable
    # code hashes per class. No score-dependent sampling or length matching.
    counts = Counter()
    for p in paths:
        for b in pq.ParquetFile(p).iter_batches(batch_size=65536, columns=['label']):
            counts.update(map(int, b.column(0).to_pylist()))
    q = quotas(counts, target)
    heaps = {k: [] for k in q}
    seen = {}
    dropped = Counter()
    for code, label, source, source_row in stream(paths):
        if not isinstance(code, str) or not code.strip():
            dropped['empty'] += 1
            continue
        sha = hashlib.sha256(code.encode()).hexdigest()
        if sha in excluded:
            dropped['train_validation_exact_overlap'] += 1
            continue
        if sha in seen:
            dropped['duplicate_same_label' if seen[sha] == label else 'duplicate_label_conflict'] += 1
            continue
        seen[sha] = label
        score = int(sha, 16)
        row = {'code': code, 'label': label, 'code_sha256': sha,
               'source_file': source, 'source_row': source_row}
        heap = heaps[label]
        item = (-score, sha, row)
        if len(heap) < q[label]: heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]: heapq.heapreplace(heap, item)
    rows = [item[2] for k in sorted(heaps) for item in heaps[k]]
    rows.sort(key=lambda r: r['code_sha256'])
    return rows, set(seen), {'raw_labels': dict(counts), 'requested_quotas': q,
                            'selected_labels': dict(Counter(r['label'] for r in rows)), 'dropped': dict(dropped)}


def prepare(train_size=200000, val_size=12000):
    if (DATA / 'manifest.json').exists():
        raise RuntimeError('Frozen package exists; use --cpu or consume manifest, do not overwrite.')
    DATA.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((RAW.parent / 'aicd_manifest.json').read_text(encoding='utf-8'))
    tasks = {}
    for task, (config, nclasses) in TASKS.items():
        log(f'PREPARE {task} {config}')
        paths = sorted((RAW/config).glob('train-*.parquet'))
        val_paths = sorted((RAW/config).glob('validation-*.parquet'))
        assert paths and val_paths
        for p in paths+val_paths:
            key = p.relative_to(RAW).as_posix()
            assert digest(p) == source_manifest['files'][key]['sha256'], key
        train, seen, tr_audit = choose(paths, train_size, set())
        val, _, va_audit = choose(val_paths, val_size, seen)
        assert len(train) >= 100000, (task, len(train))
        assert set(r['label'] for r in train) == set(range(nclasses))
        outdir = DATA/task
        outdir.mkdir(exist_ok=True)
        for split, rows in [('train', train), ('validation', val)]:
            pq.write_table(pa.Table.from_pylist(rows), outdir/f'{split}.parquet', compression='zstd')
            features = np.asarray([list(format_features(r['code']).values()) for r in rows], dtype=np.float32)
            np.savez_compressed(outdir/f'{split}_format.npz', x=features,
                                names=np.array(list(format_features('').keys())))
            log(f'{task} {split}: {len(rows)} code records')
        tasks[task] = {'config': config, 'nclasses': nclasses, 'train': tr_audit, 'validation': va_audit,
                       'raw_train_rows': sum(pq.ParquetFile(p).metadata.num_rows for p in paths),
                       'source_files': {p.relative_to(ROOT).as_posix(): digest(p) for p in paths+val_paths}}
        write(DATA/'build_progress.json', tasks)
        del train, val, seen
    manifest = {'schema': 'mainline_abc_large_v1', 'seed': SEED, 'source_revision': source_manifest['revision'],
                'tasks': tasks, 'requested_train_per_task': train_size, 'requested_validation_per_task': val_size,
                'base_assumption': 'flagship effective base differences negligible; working assumption, mechanism checks deferred',
                'label_names': {'A': 'released T1 numeric 0/1', 'B': 'released T2 IDs 0..11; family set includes Human; exact ID-to-family map pending',
                                'C': 'released T3 IDs 0..3; Human/Machine/Hybrid/Adversarial; ID naming pending'},
                'limits': ['AICD releases only code and label: no task/project/generator IDs',
                           'official train/validation preserved; exact validation matches to any raw train code removed',
                           'three task configurations overlap; do not add rows as unique examples',
                           'pilot pool selection does not use model scores; within-label duplicates keep first occurrence',
                           'rows with conflicting released labels are counted, not causally adjudicated'],
                'test_opened': False, 'script_sha256': digest(Path(__file__)),
                'files': {p.relative_to(DATA).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
                          for p in sorted(DATA.rglob('*')) if p.is_file() and p.name != 'build_progress.json'}}
    write(DATA/'manifest.json', manifest)
    write(OUT/'data_manifest.json', manifest)
    log('LARGE_DATA_READY')


def metrics(y, p):
    pred = p.argmax(1)
    out = {'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
           'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
           'per_id_recall': {str(k): float((pred[y == k] == k).mean()) for k in sorted(set(y))},
           'confusion_matrix_numeric': confusion_matrix(y, pred).tolist()}
    if p.shape[1] == 2: out['auroc_numeric_label_1'] = float(roc_auc_score(y, p[:, 1]))
    return out


def cpu():
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    out = {}
    vectorizer = HashingVectorizer(n_features=2**18, alternate_sign=False, norm='l2',
                                  ngram_range=(1, 2), lowercase=False,
                                  token_pattern=r'(?u)\b\w+\b|[^\w\s]')
    OUT.mkdir(exist_ok=True, parents=True)
    for task, (_, nclasses) in TASKS.items():
        if (OUT/f'{task}_cpu.json').exists():
            out[task] = json.loads((OUT/f'{task}_cpu.json').read_text(encoding='utf-8'))
            continue
        table = pq.read_table(DATA/task/'train.parquet').to_pydict()
        val = pq.read_table(DATA/task/'validation.parquet').to_pydict()
        xfmt = np.log1p(np.load(DATA/task/'train_format.npz', allow_pickle=False)['x'])
        vfmt = np.log1p(np.load(DATA/task/'validation_format.npz', allow_pickle=False)['x'])
        scaler = StandardScaler().fit(xfmt)
        xfmt, vfmt = scaler.transform(xfmt), scaler.transform(vfmt)
        yy = np.asarray(table['label']); yv = np.asarray(val['label'])
        weights = len(yy) / (nclasses * np.bincount(yy, minlength=nclasses))
        models = {name: SGDClassifier(loss='log_loss', alpha=1e-5, random_state=SEED,
                                     learning_rate='optimal', average=True) for name in ['lexical', 'lexical_format', 'format']}
        start = time.time()
        for epoch in range(2):
            order = np.random.default_rng(SEED+epoch).permutation(len(yy))
            for off in range(0, len(order), 512):
                ix = order[off:off+512]
                xx = vectorizer.transform([table['code'][i] for i in ix])
                inputs = {'lexical': xx, 'format': xfmt[ix],
                          'lexical_format': hstack([xx, csr_matrix(.05*xfmt[ix])], format='csr')}
                for name, model in models.items():
                    model.partial_fit(inputs[name], yy[ix], classes=np.arange(nclasses), sample_weight=weights[yy[ix]])
                if off % 51200 == 0: log(f'CPU {task} epoch={epoch+1} rows={off}/{len(order)}')
        result = {'train_rows': len(yy), 'validation_rows': len(yv), 'epochs': 2, 'arms': {},
                  'train_seconds': time.time()-start, 'test_opened': False, 'model': 'streaming hashing SGD log-loss; not CodeT5'}
        for name, model in models.items():
            pp = []
            for off in range(0, len(yv), 512):
                end = off+512
                xx = vectorizer.transform(val['code'][off:end])
                inp = xx if name == 'lexical' else vfmt[off:end] if name == 'format' else hstack([xx, csr_matrix(.05*vfmt[off:end])], format='csr')
                pp.append(model.predict_proba(inp))
            p = np.concatenate(pp)
            result['arms'][name] = metrics(yv, p)
            np.savez_compressed(OUT/f'{task}_{name}_scores.npz', y=yv, p=p, sha=np.array(val['code_sha256']))
            log(f'RESULT {task} {name} macro-F1={result["arms"][name]["macro_f1"]:.4f}')
        write(OUT/f'{task}_cpu.json', result)
        out[task] = result
    write(OUT/'cpu_metrics.json', {'tasks': out, 'status': 'complete', 'protocol': '200k scale rapid CPU controls',
                                   'data_manifest_sha256': digest(DATA/'manifest.json')})
    log('ABC_CPU_DONE')


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    args=ap.parse_args()
    with threadpool_limits(limits=4):
        if args.prepare: prepare()
        if args.cpu: cpu()
