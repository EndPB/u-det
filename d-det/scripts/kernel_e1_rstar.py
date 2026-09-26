#!/usr/bin/env python
"""E1（mini）：真实 log-ratio r* = log p_instruct(x) − log p_base(x) 与 s2 的关系。

RLHF 内核直接检验（用户令"尽量小的实验"）：
- 样本：B val 分层 200 条（与 s2/s1 分数行对齐；来自冻结 v1.0 特征缓存）
- 打分：Qwen2.5-Coder-1.5B base/instruct 裸文本 logprob（total 与 mean 两版）
- 分析：corr(r*, s2/s1)、r* 的人机检测 AUC、机器族间 r* 结构（Kruskal）
- 输出：runs/kernel_e1/{rstar.json, logp_{base,instruct}.npz}
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import kruskal, pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from semeval_stats import FILES, RAW, replay_picks  # noqa: E402

N = 200
MAXLEN = 1024
OUT = ROOT / "runs/kernel_e1"
MODELS = {"base": ROOT / "checkpoints/qwen2.5-coder-1.5b-base",
          "instruct": ROOT / "checkpoints/qwen2.5-coder-1.5b-instruct"}


def build_samples():
    tok5 = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    path = RAW / FILES["b"][1]  # B val
    t = pq.read_table(path, columns=["code", "label", "generator"])
    codes = t.column("code").to_pylist()
    labs = np.asarray(t.column("label").to_pylist())
    gens = t.column("generator").to_pylist()
    idx = replay_picks()[("b", "val")]
    kept = [i for i in idx
            if len(tok5(codes[i], add_special_tokens=False)["input_ids"]) >= 8]
    feat = np.load(ROOT / "runs/semeval_zeroshot/feat/b_val.npz", allow_pickle=True)
    assert len(kept) == len(feat["y"]) == 7817, (len(kept), len(feat["y"]))
    from collections import defaultdict
    by_lab = defaultdict(list)
    for j, i in enumerate(kept):
        by_lab[int(labs[i])].append(j)
    rng = random.Random(0)
    per = max(1, N // len(by_lab))
    sel = []
    for l in sorted(by_lab):
        sel += rng.sample(by_lab[l], min(per, len(by_lab[l])))
    sel = sorted(sel)
    return {"code": [codes[kept[j]] for j in sel],
            "label": np.array([int(labs[kept[j]]) for j in sel]),
            "generator": [str(gens[kept[j]]) for j in sel],
            "s1": np.asarray(feat["s1"])[sel].reshape(-1),
            "s2": np.asarray(feat["s2"])[sel].reshape(-1),
            "row": np.array(sel)}


@torch.no_grad()
def score_model(path: Path, codes, device="cuda"):
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16).to(device).eval()
    tot, mean = [], []
    for c in codes:
        ids = tok(c, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] > MAXLEN:
            ids = ids[:, :MAXLEN]
        if ids.shape[1] < 2:
            tot.append(0.0)
            mean.append(0.0)
            continue
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        tot.append(-float(out.loss) * n)
        mean.append(-float(out.loss))
    del model
    torch.cuda.empty_cache()
    return np.array(tot), np.array(mean)


def main() -> int:
    smp = build_samples()
    OUT.mkdir(parents=True, exist_ok=True)
    lp = {}
    for tag, path in MODELS.items():
        tj, mn = score_model(path, smp["code"])
        lp[tag] = (tj, mn)
        np.savez_compressed(OUT / f"logp_{tag}.npz", total=tj, mean=mn)
        print(f"[e1] {tag} 打分完成（mean logp={mn.mean():.3f}）", flush=True)

    y = (smp["label"] > 0).astype(int)
    s1, s2 = smp["s1"], smp["s2"]
    res = {"n": len(y), "n_machine": int(y.sum())}
    for name in ("total", "mean"):
        r = lp["instruct"][0 if name == "total" else 1] - lp["base"][0 if name == "total" else 1]
        np.savez_compressed(OUT / f"rstar_{name}.npz", r=r, y=y, label=smp["label"],
                            generator=np.array(smp["generator"], dtype=object),
                            row=smp["row"], s1=s1, s2=s2)
        res[name] = {
            "corr_r_s2_pearson": round(float(pearsonr(r, s2)[0]), 4),
            "corr_r_s2_spearman": round(float(spearmanr(r, s2)[0]), 4),
            "corr_r_s1_pearson": round(float(pearsonr(r, s1)[0]), 4),
            "auc_machine_vs_human": round(float(roc_auc_score(y, r)), 4),
            "r_mean": round(float(r.mean()), 3),
        }
    res["auc_s2_200"] = round(float(roc_auc_score(y, s2)), 4)
    res["auc_s1_200"] = round(float(roc_auc_score(y, s1)), 4)
    # 机器族间 r* 差异（Kruskal）
    from collections import defaultdict
    groups = defaultdict(list)
    for g, lab, r in zip(smp["generator"], smp["label"], lp["instruct"][0] - lp["base"][0]):
        if lab > 0:
            groups[g].append(float(r))
    gs = [v for v in groups.values() if len(v) >= 3]
    if len(gs) >= 3:
        stat, p = kruskal(*gs)
        res["rstar_family_kruskal"] = {"H": round(float(stat), 3), "p": float(p),
                                       "n_groups": len(gs)}
    res["family_rstar_mean"] = {g: round(float(np.mean(v)), 3) for g, v in sorted(groups.items())}
    (OUT / "rstar.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("[e1] 结果：", json.dumps(res, ensure_ascii=False)[:1500], flush=True)
    print("[e1] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
