"""Audit existing train/dev code and public task metadata without executing code.

Writes code-free static proxies and the existing 11 within-series member folds.
Neither tests nor canonical solutions are selected from the metadata Parquet.
Static proxies are deliberately not called correctness or post-training labels.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ordered_hash(values):
    return hashlib.sha256(('\n'.join(values) + '\n').encode()).hexdigest()


def static_view(code, entry_point, requirements):
    result = {'parse_ok': 0, 'entrypoint_present': 0, 'n_ast_nodes': 0,
              'n_calls': 0, 'n_branches': 0, 'n_handlers': 0,
              'n_returns': 0, 'n_import_roots': 0,
              'required_root_overlap_proxy': 0.0}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return result
    nodes = list(ast.walk(tree))
    roots = set()
    for n in nodes:
        if isinstance(n, ast.Import):
            roots.update(a.name.split('.')[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            roots.add(n.module.split('.')[0])
    result.update(parse_ok=1, n_ast_nodes=len(nodes),
                  entrypoint_present=int(any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                                             and n.name == entry_point for n in tree.body)),
                  n_calls=sum(isinstance(n, ast.Call) for n in nodes),
                  n_branches=sum(isinstance(n, (ast.If, ast.For, ast.While, ast.Try)) for n in nodes),
                  n_handlers=sum(isinstance(n, ast.ExceptHandler) for n in nodes),
                  n_returns=sum(isinstance(n, ast.Return) for n in nodes),
                  n_import_roots=len(roots),
                  required_root_overlap_proxy=len(roots & requirements) / max(1, len(requirements)))
    return result


def library_roots(value):
    if not value:
        return set()
    try:
        values = ast.literal_eval(value) if isinstance(value, str) else value
    except (ValueError, SyntaxError):
        return set()
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {str(x).split('.')[0] for x in values}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--records', type=Path, required=True)
    ap.add_argument('--tasks-parquet', type=Path, required=True)
    ap.add_argument('--series-map', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit('Refusing to overwrite a non-empty output directory')
    import pyarrow.dataset as ds

    rows = [json.loads(s) for s in a.records.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    assert rows and all(r['split'] in ('train', 'dev') for r in rows), 'train/dev-only input required'
    keys = [(r['model_id'], r['task_id']) for r in rows]
    assert len(set(keys)) == len(keys), 'duplicate model/task rows'
    role = defaultdict(set)
    for r in rows:
        role[r['task_id']].add(r['split'])
        assert hashlib.sha256(r['code'].encode()).hexdigest() == r['solution_sha256']
    assert all(len(v) == 1 for v in role.values()), 'task crosses split'
    sm = json.loads(a.series_map.read_text(encoding='utf-8-sig'))
    groups = {m['model_id']: s['series'] for s in sm['series'] for m in s['members']}
    assert set(r['model_id'] for r in rows) == set(groups), 'series support mismatch'
    assert all(r['series'] == groups[r['model_id']] for r in rows)
    columns = ['task_id', 'entry_point', 'libs', 'instruct_prompt', 'code_prompt']
    table = ds.dataset(a.tasks_parquet, format='parquet').to_table(
        columns=columns, filter=ds.field('task_id').isin(sorted(role)))
    tasks = {r['task_id']: r for r in table.to_pylist()}
    assert set(tasks) == set(role), 'missing/duplicate task metadata'
    assert len(tasks) == table.num_rows
    coverage = Counter()
    out_rows = []
    for r in rows:
        t = tasks[r['task_id']]
        v = static_view(r['code'], t['entry_point'], library_roots(t['libs']))
        coverage.update({k: int(v[k]) for k in ('parse_ok', 'entrypoint_present')})
        out_rows.append({k: r[k] for k in ('model_id', 'series', 'task_id', 'split', 'solution_sha256')} |
                        {'static_proxy': v, 'correctness': None,
                         'task_spec_sha256': hashlib.sha256(t['instruct_prompt'].encode()).hexdigest()})
    folds = []
    for h in sorted(groups):
        family = groups[h]
        tr = [r for r in rows if r['split'] == 'train' and r['model_id'] != h]
        ev = [r for r in rows if r['split'] == 'dev' and
              (r['model_id'] == h or groups[r['model_id']] != family)]
        assert h not in {r['model_id'] for r in tr}
        assert any(groups[r['model_id']] == family for r in tr)
        assert {r['model_id'] for r in ev if r['series'] == family} == {h}
        assert {r['task_id'] for r in tr}.isdisjoint(r['task_id'] for r in ev)
        folds.append({'heldout_generator_member': h, 'seen_series': family,
                      'train_rows': len(tr), 'eval_rows': len(ev),
                      'eval_positive_rows': sum(r['model_id'] == h for r in ev),
                      'ordered_train_key_sha256': ordered_hash([r['model_id']+'\t'+r['task_id'] for r in tr]),
                      'ordered_eval_key_sha256': ordered_hash([r['model_id']+'\t'+r['task_id'] for r in ev]),
                      'negative_members_seen_in_fit': True})
    a.out.mkdir(parents=True, exist_ok=True)
    def write(name, obj):
        (a.out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (a.out / 'static_proxies.jsonl').open('w', encoding='utf-8') as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    write('fold_plan.json', {'evaluation': 'seen_observed_series_unseen_positive_generator_member',
                             'claim_limit': 'size-variant transfer; family_is_confirmed remains false',
                             'row_order': 'input records order', 'folds': folds})
    write('audit.json', {'schema': 'code_conditioned_static_preflight_v1',
                        'rows': len(rows), 'tasks': len(role), 'members': len(groups),
                        'split_rows': dict(Counter(r['split'] for r in rows)),
                        'split_tasks': {s: sum(v == {s} for v in role.values()) for s in ('train', 'dev')},
                        'static_coverage': dict(coverage), 'folds': len(folds),
                        'selected_metadata_columns': columns,
                        'metadata_filter': 'train/dev task whitelist; Parquet decoder may scan pages',
                        'test_model_outputs_read': False, 'test_body_selected': False,
                        'canonical_solution_selected': False, 'code_executed': False,
                        'generation': False, 'weights_downloaded': False,
                        'correctness_labels_available_in_this_output': False,
                        'input_sha256': {str(p): digest(p) for p in (a.records, a.tasks_parquet, a.series_map)}})
    files = sorted(a.out.glob('*'))
    (a.out / 'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in files), encoding='utf-8')
    print(json.dumps({'rows': len(rows), 'tasks': len(role), 'folds': len(folds), 'coverage': dict(coverage)}))


if __name__ == '__main__':
    main()
