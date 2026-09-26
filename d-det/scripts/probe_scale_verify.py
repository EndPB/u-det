"""跨规模/跨族核查（参数化版）：s1 / s2 / 融合 + 长度对照 + 重复 CV，并存 npz。

用法：python scripts/probe_scale_verify.py --ckpt runs/v0.1.0/best.pt \
        --tag v0.1.0 --pair-file data/processed/pairs_qwen15.parquet \
        --npz runs/v0.1.0/scale_scores.npz
人类池固定为 m4-test 的 python 人类（315 条，seed 0）；AI 三组：
m4-python AI（参照）、pair 文件的 x_plus / x_minus（新族）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataio import build_dataset                    # noqa: E402
from encoders import build_encoder                  # noqa: E402
from models import build_model                      # noqa: E402
from train import apply_overrides, load_config      # noqa: E402


@torch.no_grad()
def score_one(model, tok, code, device):
    ids = tok(code, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    s1, s2 = model(ids, torch.ones_like(ids))
    return float(s1.item()), float(s2.reshape(-1)[0].item()), int(ids.shape[1])


def score_set(model, tok, codes, device):
    return np.array([score_one(model, tok, c, device) for c in codes])


def fused_rep(y, X, n_repeats=4):
    aucs = []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=0)
    for tr, te in rskf.split(X, y):
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def report(name, Xa, Xb):
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    s1 = np.r_[Xa[:, 0], Xb[:, 0]]
    s2 = np.r_[Xa[:, 1], Xb[:, 1]]
    lt = np.log1p(np.r_[Xa[:, 2], Xb[:, 2]])
    a1 = roc_auc_score(y, s1)
    a2raw = roc_auc_score(y, s2)
    a2 = max(a2raw, 1 - a2raw)
    f1, f1s = fused_rep(y, np.stack([s1, s2], 1))
    fL, _ = fused_rep(y, np.stack([s1, lt], 1))
    f1L, _ = fused_rep(y, np.stack([s1, s2, lt], 1))
    print('  %-26s n=%3d/%3d | s1=%.4f s2*=%.4f | fused=%.4f±%.4f | s1+L=%.4f s1+s2+L=%.4f'
          % (name, len(Xa), len(Xb), a1, a2, f1, f1s, fL, f1L))
    return {'name': name, 'auc_s1': a1, 'auc_s2_best': a2, 'fused_mean': f1, 'fused_std': f1s,
            'auc_s1_len': fL, 'auc_s1_s2_len': f1L}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--pair-file', required=True)
    ap.add_argument('--npz', default=None)
    ap.add_argument('--config', default='configs/ddet_base.yaml',
                    help='训练同款配置（v0.3 全参微调传 configs/ddet_v030.yaml）')
    ap.add_argument('--set', action='append', default=None)
    args = ap.parse_args()

    cfg = apply_overrides(load_config(str(ROOT / args.config)), args)
    device = 'cuda'
    enc_cfg = dict(cfg['encoder'])
    enc_cfg['path'] = str(ROOT / enc_cfg['path'])
    enc = build_encoder(**enc_cfg)
    mcfg = dict(cfg['model'])
    model = build_model(mcfg.pop('name', 'dual'), encoder=enc, dim=enc.hidden_size, **mcfg)
    ck = torch.load(str(ROOT / args.ckpt), map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state'], strict=False)
    model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(enc_cfg['path'])

    ds = build_dataset('m4', file=str(ROOT / 'data/processed/m4.parquet'), split='test')
    hum = [ds.codes[i] for i, m in enumerate(ds.meta) if ds.labels[i] == 0 and m.get('language') == 'python']
    ai4 = [ds.codes[i] for i, m in enumerate(ds.meta) if ds.labels[i] == 1 and m.get('language') == 'python']
    rng = np.random.default_rng(0)
    rng.shuffle(hum)
    rng.shuffle(ai4)
    hum, ai4 = hum[:315], ai4[:287]

    tbl = pq.read_table(str(ROOT / args.pair_file))
    plus = tbl.column('x_plus').to_pylist()
    minus = tbl.column('x_minus').to_pylist()

    H = score_set(model, tok, hum, device)
    A = score_set(model, tok, ai4, device)
    P = score_set(model, tok, plus, device)
    M = score_set(model, tok, minus, device)
    if args.npz:
        out = ROOT / args.npz
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out), hum=H, ai4=A, plus=P, minus=M)
        print('[verify] npz 已存 %s' % out)

    print('##', args.tag)
    report('ref_m4ai_vs_human', H, A)
    report('newpair_instruct_vs_human', H, P)
    report('newpair_base_vs_human', H, M)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
