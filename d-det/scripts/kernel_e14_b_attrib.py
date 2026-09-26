#!/usr/bin/env python
"""E14（mini）：B 上的判别式归因——多参照 r* 读数群（21 族）。

B 无配对 Δ → 形态改为：对 B val 分层样本打 3 个参照对的 r*（qw05/qw15/ds13），
用读数群做**判别式**族分类（LR/LDA，5 折），对照：NCM（无监督）、单一 r* 、s1。
- 分层：每 machine 族 ~24 条（全部 21 族 ≈ 500 条）
- 输出：runs/kernel_e14/attrib.json（含 r* 打分缓存）
用法：python scripts/kernel_e14_b_attrib.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_e1_rstar import build_samples, score_model  # noqa: E402

OUT = ROOT / "runs/kernel_e14"
PAIRS = {"qw05": (ROOT / "checkpoints/qwen2.5-coder-0.5b-base",
                  ROOT / "checkpoints/qwen2.5-coder-0.5b-instruct"),
         "qw15": (ROOT / "checkpoints/qwen2.5-coder-1.5b-base",
                  ROOT / "checkpoints/qwen2.5-coder-1.5b-instruct"),
         "ds13": (ROOT / "checkpoints/deepseek-coder-1.3b-base",
                  ROOT / "checkpoints/deepseek-coder-1.3b-instruct")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- 重建 B val 的 code/族（同 E1 链，但全量 + 按族分层）----
    from semeval_stats import FILES, RAW, replay_picks
    from transformers import AutoTokenizer
    tok5 = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    path = RAW / FILES["b"][1]
    t = pq.read_table(path, columns=["code", "label", "generator"])
    codes_all = t.column("code").to_pylist()
    labs_all = np.asarray(t.column("label").to_pylist())
    gens_all = t.column("generator").to_pylist()
    idx = replay_picks()[("b", "val")]
    kept = [i for i in idx
            if len(tok5(codes_all[i], add_special_tokens=False)["input_ids"]) >= 8]
    feat = np.load(ROOT / "runs/semeval_zeroshot/feat/b_val.npz", allow_pickle=True)
    assert len(kept) == 7817
    by_gen = defaultdict(list)
    for j, i in enumerate(kept):
        if labs_all[i] > 0:
            by_gen[str(gens_all[i])].append(j)
    rng = random.Random(0)
    sel = []
    for g in sorted(by_gen):
        sel += rng.sample(by_gen[g], min(24, len(by_gen[g])))
    sel = sorted(sel)
    codes = [codes_all[kept[j]] for j in sel]
    gens = [str(gens_all[kept[j]]) for j in sel]
    s1 = np.asarray(feat["s1"])[sel].reshape(-1)
    print(f"[e14] 分层样本 {len(sel)} 条，{len(set(gens))} 族", flush=True)

    # ---- 3 参照对打分（缓存）----
    cache = OUT / "rstars.npz"
    if cache.exists():
        d = np.load(cache)
        R = d["R"]
        np.savez_compressed(cache, R=R, gens=np.array(gens, dtype=object), s1=s1)
    else:
        cols = []
        for tag, (pb, pi) in PAIRS.items():
            fb = OUT / f"logp_{tag}_base.npz"
            fi = OUT / f"logp_{tag}_instruct.npz"
            if fb.exists() and fi.exists():
                tb = np.load(fb)["total"]
                ti = np.load(fi)["total"]
            else:
                tb, _ = score_model(pb, codes)
                ti, _ = score_model(pi, codes)
                np.savez_compressed(fb, total=tb)
                np.savez_compressed(fi, total=ti)
            cols.append(ti - tb)
            print(f"[e14] {tag} 打分完成", flush=True)
        R = np.vstack(cols).T
        np.savez_compressed(cache, R=R, gens=np.array(gens, dtype=object), s1=s1)

    z = np.array(gens)
    res = {"n": len(z), "n_families": int(len(set(z.tolist())))}

    def cv_eval(F, model_fn, name):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        acc, bacc = [], []
        for tr, va in skf.split(F, z):
            m = model_fn()
            m.fit(F[tr], z[tr])
            p = m.predict(F[va])
            acc.append(accuracy_score(z[va], p))
            bacc.append(balanced_accuracy_score(z[va], p))
        res[name] = {"acc": round(float(np.mean(acc)), 4),
                     "balanced_acc": round(float(np.mean(bacc)), 4)}
        print(f"[e14] {name}: {res[name]}", flush=True)

    Rz = (R - R.mean(0)) / (R.std(0) + 1e-9)
    cv_eval(Rz, lambda: LogisticRegression(max_iter=3000, C=1.0), "lr_R3")
    cv_eval(Rz, lambda: LinearDiscriminantAnalysis(), "lda_R3")
    for j, tag in enumerate(PAIRS):
        f = ((R[:, j:j + 1] - R[:, j:j + 1].mean(0)) / (R[:, j:j + 1].std(0) + 1e-9))
        cv_eval(f, lambda: LogisticRegression(max_iter=1000), f"lr_only_{tag}")
    s1z = ((s1 - s1.mean()) / (s1.std() + 1e-9)).reshape(-1, 1)
    cv_eval(s1z, lambda: LogisticRegression(max_iter=1000), "lr_only_s1")

    # NCM 对照（无监督）
    def ncm_acc(F):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        accs = []
        for tr, va in skf.split(F, z):
            cents, labs = [], []
            for c in set(z[tr].tolist()):
                cents.append(F[tr][z[tr] == c].mean(0))
                labs.append(c)
            cents = np.asarray(cents)
            d = ((F[va][:, None, :] - cents[None, :, :]) ** 2).sum(-1)
            pred = np.array(labs)[d.argmin(1)]
            accs.append(accuracy_score(z[va], pred))
        return float(np.mean(accs))
    res["ncm_R3_acc"] = round(ncm_acc(Rz), 4)
    print(f"[e14] ncm_R3: {res['ncm_R3_acc']}", flush=True)

    (OUT / "attrib.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("[e14] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
