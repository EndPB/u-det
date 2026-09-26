"""片段任务探针：检验检测器对代码片段（非完整函数）的判别力。

理论侧（docx/d-det-v0.2.md §1）：x 在设计文档符号表里本就是「待检代码片段，
默认函数级」；s1/s2 的读出 = 分块编码器 + token 均值池化，架构上无函数级依赖
→ 任意长度可打分。本探针用实验回答：把 m4-test 的函数切成片段后，判别力保留
多少、偏置如何。

设计：
- 选 m4-test 的 n 条（按标签 1:1 分层，seed 固定）；
- 原函数 = full 组；行数 ≥ --min-lines 的函数再生成三个片段：
  head（前 40% 行）/ mid（居中 40% 行窗）/ tail（后 40% 行）；
- 片段子集另评 full_frag（同子集完整函数）作公平基线；
- 逐条打分（s1→prob、s2 标量），报告 acc@0.5 / AUC / 两类平均 prob / 平均 token 长。

输出：runs/<tag>/fragments.json（tag 从 --ckpt 路径推断）。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataio import build_dataset                      # noqa: E402
from encoders import build_encoder                    # noqa: E402
from models import build_model                        # noqa: E402
from train import apply_overrides, load_config        # noqa: E402


def make_fragments(code: str, min_lines: int = 8, frac: float = 0.4) -> dict:
    """{full, head?, mid?, tail?}；行数不足或窗口过短时只含 full。"""
    out = {'full': code}
    lines = code.split('\n')
    n = len(lines)
    if n < int(min_lines):
        return out
    k = max(2, int(round(n * frac)))
    k = min(k, n - 1)
    if k < 2:
        return out
    mid = max(0, (n - k) // 2)
    out['head'] = '\n'.join(lines[:k])
    out['tail'] = '\n'.join(lines[n - k:])
    out['mid'] = '\n'.join(lines[mid:mid + k])
    return out


@torch.no_grad()
def scores_of(model, tokenizer, code: str, device: str):
    ids = tokenizer(code, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    s1, s2 = model(ids, torch.ones_like(ids))
    return float(s1.item()), float(s2.reshape(-1)[0].item()), int(ids.shape[1])


def summarize(rows: list) -> dict:
    n = max(len(rows), 1)
    y = [r['label'] for r in rows]
    p = [1.0 / (1.0 + math.exp(-r['s1'])) for r in rows]
    acc = sum((1 if pi >= 0.5 else 0) == yi for pi, yi in zip(p, y)) / n
    auc = None
    if len(set(y)) > 1:
        auc = float(roc_auc_score(y, p))
    ai = [i for i, yi in enumerate(y) if yi == 1]
    hu = [i for i, yi in enumerate(y) if yi == 0]
    return {
        'n': len(rows),
        'acc@0.5': acc,
        'auc': auc,
        'mean_prob_ai': sum(p[i] for i in ai) / max(len(ai), 1),
        'mean_prob_human': sum(p[i] for i in hu) / max(len(hu), 1),
        'mean_s2_ai': sum(rows[i]['s2'] for i in ai) / max(len(ai), 1),
        'mean_s2_human': sum(rows[i]['s2'] for i in hu) / max(len(hu), 1),
        'mean_tokens': sum(r['ntok'] for r in rows) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='片段任务探针（m4 函数 → 首/中/尾片段）')
    ap.add_argument('--config', default='configs/ddet_base.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--min-lines', type=int, default=8)
    ap.add_argument('--frac', type=float, default=0.4)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--dump-samples', action='store_true',
                    help='把逐样本分数落盘 fragments_samples.json（配对显著性检验用）')
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
    if missing or unexpected:
        print('[fragments] 注意：未加载 %d 项、多余 %d 项' % (len(missing), len(unexpected)))
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(enc_cfg['path'])

    data_cfg = cfg['data']
    m4_path = ROOT / data_cfg['processed_dir'] / data_cfg['m4_file']
    dataset = build_dataset('m4', file=str(m4_path), split=args.split)
    labels = list(dataset.labels)
    codes = list(dataset.codes)

    rng = random.Random(0)
    idx_h = [i for i, y in enumerate(labels) if y == 0]
    idx_a = [i for i, y in enumerate(labels) if y == 1]
    rng.shuffle(idx_h)
    rng.shuffle(idx_a)
    side = max(1, args.n // 2)
    sel = idx_h[:side] + idx_a[:side]
    print('[fragments] 抽样 %d 条（human/ai 各 <=%d；池 %d/%d）ckpt=%s'
          % (len(sel), side, len(idx_h), len(idx_a), args.ckpt))

    groups: dict = {}
    got_frag = 0
    for i in sel:
        frags = make_fragments(codes[i], args.min_lines, args.frac)
        eligible = len(frags) > 1
        got_frag += int(eligible)
        for name, text in frags.items():
            if not text.strip():
                continue
            s1, s2, ntok = scores_of(model, tokenizer, text, device)
            groups.setdefault(name, []).append(
                {'label': labels[i], 's1': s1, 's2': s2, 'ntok': ntok})
        if eligible:            # 同子集完整函数 = 公平基线
            s1, s2, ntok = scores_of(model, tokenizer, frags['full'], device)
            groups.setdefault('full_frag', []).append(
                {'label': labels[i], 's1': s1, 's2': s2, 'ntok': ntok})
    print('[fragments] 可切片段样本 %d / %d（>=%d 行）'
          % (got_frag, len(sel), args.min_lines))

    summary = {name: summarize(rows) for name, rows in groups.items()}
    for name in ['full', 'full_frag', 'head', 'mid', 'tail']:
        s = summary.get(name)
        if not s:
            continue
        auc_val = s['auc']
        auc = 'None' if auc_val is None else format(auc_val, '.4f')
        print('[%9s] n=%-4d acc=%.4f auc=%s p_ai=%.3f p_hu=%.3f s2_ai=%+.2f s2_hu=%+.2f tok=%.0f'
              % (name, s['n'], s['acc@0.5'], auc, s['mean_prob_ai'],
                 s['mean_prob_human'], s['mean_s2_ai'], s['mean_s2_human'],
                 s['mean_tokens']))

    if args.dump_samples:
        dump = ckpt_path.parent / 'fragments_samples.json'
        with open(dump, 'w', encoding='utf-8') as f:
            json.dump({'ckpt': str(args.ckpt), 'split': args.split, 'groups': groups},
                      f, ensure_ascii=False)
        print('[fragments] 逐样本分数已写入 %s' % dump)

    out = ckpt_path.parent / 'fragments.json'
    meta = {'ckpt': str(args.ckpt), 'split': args.split, 'n_requested': args.n,
            'min_lines': args.min_lines, 'frac': args.frac, 'groups': summary}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('[fragments] 结果已写入 %s' % out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
