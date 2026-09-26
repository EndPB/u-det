"""跨规模检测对照探针：m4 参照组 vs Qwen2.5-Coder-1.5B 生成组。

对照设计（回答「d-det 相对 CodeT5 单头基线的检测增益」）：
- 参照：m4-test 人类（python）× m4-test AI（python）——模型训练分布内；
- 新族：同一人类池 × 1.5B 生成代码（x_plus=instruct / x_minus=base）——训练未见。
指标：AUC(s1 单独) 与 AUC(融合 s1+s2，5 折 CV 逻辑回归) 对比；另有 AUC(s2，取优方向)。
- s1 单独 = 「CodeT5 单头基线」的等价物；
- 若融合增益在新族上显著，说明 s2 的跨规模检测价值；否则诚实记录为负面。
口径：全部由 code 文本重新分词（add_special_tokens=False，对齐 m4 预分词）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
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
    return float(s1.item()), float(s2.reshape(-1)[0].item())


def score_set(model, tok, codes, device, name):
    S1, S2 = [], []
    for i, c in enumerate(codes):
        s1, s2 = score_one(model, tok, c, device)
        S1.append(s1)
        S2.append(s2)
        if (i + 1) % 200 == 0:
            print('[scale] %s %d/%d' % (name, i + 1, len(codes)))
    return np.array(S1), np.array(S2)


def contrast(name, hum, ai):
    y = np.r_[np.zeros(len(hum[0])), np.ones(len(ai[0]))]
    s1 = np.r_[hum[0], ai[0]]
    s2 = np.r_[hum[1], ai[1]]
    auc1 = float(roc_auc_score(y, s1))
    a2 = roc_auc_score(y, s2)
    auc2 = float(max(a2, 1.0 - a2))
    X = np.stack([s1, s2], axis=1)
    prob = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=5,
                             method='predict_proba')[:, 1]
    aucf = float(roc_auc_score(y, prob))
    row = {'name': name, 'n_human': int((y == 0).sum()), 'n_ai': int((y == 1).sum()),
           'auc_s1': auc1, 'auc_s2_best_dir': auc2, 'auc_fused_lr5cv': aucf}
    print('[scale] %-26s n=%d/%d  AUC(s1)=%.4f  AUC(s2*)=%.4f  AUC(fused)=%.4f'
          % (name, row['n_human'], row['n_ai'], auc1, auc2, aucf))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description='跨规模检测对照（s1 / s2 / 融合）')
    ap.add_argument('--config', default='configs/ddet_base.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--pair-file', default='data/processed/pairs_qwen15.parquet')
    ap.add_argument('--n-per-class', type=int, default=350)
    ap.add_argument('--out', default=None)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--set', action='append', default=None, metavar='段.键=值')
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    enc_cfg = dict(cfg['encoder'])
    if not Path(enc_cfg['path']).is_absolute():
        enc_cfg['path'] = str(ROOT / enc_cfg['path'])
    encoder = build_encoder(**enc_cfg)
    mcfg = dict(cfg['model'])
    model = build_model(mcfg.pop('name', 'dual'), encoder=encoder,
                        dim=encoder.hidden_size, **mcfg)
    ckpt_path = Path(args.ckpt) if Path(args.ckpt).is_absolute() else ROOT / args.ckpt
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt['state'], strict=False)
    model.to(device).eval()
    print('[scale] ckpt=%s（未加载 %d 项、多余 %d 项）' % (args.ckpt, len(missing), len(unexpected)))

    tok = AutoTokenizer.from_pretrained(enc_cfg['path'])

    ds = build_dataset('m4', file=str(ROOT / cfg['data']['processed_dir'] / cfg['data']['m4_file']),
                       split='test')
    hum_c = [ds.codes[i] for i, m in enumerate(ds.meta)
             if ds.labels[i] == 0 and m.get('language') == 'python']
    ai4_c = [ds.codes[i] for i, m in enumerate(ds.meta)
             if ds.labels[i] == 1 and m.get('language') == 'python']
    rng = np.random.default_rng(0)
    rng.shuffle(hum_c)
    rng.shuffle(ai4_c)
    n = args.n_per_class
    hum_c = hum_c[:n]
    ai4_c = ai4_c[:min(n, len(ai4_c))]
    print('[scale] m4-python 可用（取前 %d/%d）' % (len(hum_c), len(ai4_c)))

    pf = Path(args.pair_file)
    if not pf.is_absolute():
        pf = ROOT / pf
    tbl = pq.read_table(pf)
    plus_c = tbl.column('x_plus').to_pylist()
    minus_c = tbl.column('x_minus').to_pylist()
    print('[scale] 1.5B pairs：plus %d / minus %d' % (len(plus_c), len(minus_c)))

    hum = score_set(model, tok, hum_c, device, 'human')
    ref = score_set(model, tok, ai4_c, device, 'm4-ai(ref)')
    pls = score_set(model, tok, plus_c, device, 'qwen15-plus')
    mns = score_set(model, tok, minus_c, device, 'qwen15-minus')

    rows = [contrast('ref_m4ai_vs_human', hum, ref),
            contrast('qwen15_instruct_vs_human', hum, pls),
            contrast('qwen15_base_vs_human', hum, mns)]

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump({'ckpt': args.ckpt, 'pair_file': str(pf), 'rows': rows},
                      f, ensure_ascii=False, indent=2)
        print('[scale] 结果已写入 %s' % out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
