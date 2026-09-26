#!/usr/bin/env python
"""E8（mini）：d-det 几何断言的直接检验——无训练，用现成冻结特征 h（v1.0）。

1) 法向/切向几何（§2/§3）：d_HB（人−base机）与 d_BI（instruct机−base机）夹角（原始/白化）
2) s1 的"法向坐标"检验（§2）："到 M_base 的距离"（对角白化）三组分布 + corr(距离, s1) + corr(距离, r*)
3) 谱系公共轴/族残差（注记 1 修订候选②）：各"族对"位移 d_k=instruct−base 的分解；
   w1/w2（v1.0 读出）与公共轴/残差的定位——检验"s2 学的是公共轴、族信息在残差"
输出：runs/kernel_e8/geometry.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/kernel_e8"


def kind(g: str) -> str:
    gl = g.lower()
    kw = ("instruct", "chat", "-it", "gpt", "v3-0324", "devstral", "thinking")
    return "instruct" if any(k in gl for k in kw) else "base"


def canon(g: str) -> str:
    """把 instruct/chat 变体归一到基础族名，用于族配对。"""
    gl = g.lower()
    for suf in ("-instruct", "-chat", "_instruct", "-it", "-v0.3"):
        if gl.endswith(suf):
            gl = gl[: -len(suf)]
    return gl


def cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    f = np.load(ROOT / "runs/semeval_zeroshot/feat/b_val.npz", allow_pickle=True)
    H = np.asarray(f["h"], dtype="float64")
    s1 = np.asarray(f["s1"]).reshape(-1)
    t = pq.read_table(ROOT / "data/processed/semeval/b_val.parquet",
                      columns=["label", "generator"])
    lab = np.asarray(t.column("label").to_pylist())
    gen = [str(g) for g in t.column("generator").to_pylist()]
    assert len(H) == len(lab) == len(gen)

    is_human = lab == 0
    is_base = np.array([kind(g) == "base" for g in gen]) & (lab > 0)
    is_instr = np.array([kind(g) == "instruct" for g in gen]) & (lab > 0)
    res = {"n": int(len(H)), "n_human": int(is_human.sum()),
           "n_base_machine": int(is_base.sum()), "n_instruct_machine": int(is_instr.sum())}

    mu_h, mu_b, mu_i = H[is_human].mean(0), H[is_base].mean(0), H[is_instr].mean(0)
    d_HB, d_BI = mu_h - mu_b, mu_i - mu_b
    res["geom_raw"] = {"cos_dHB_dBI": round(cos(d_HB, d_BI), 4)}

    # 白化（全体协方差）
    mu, Sig = H.mean(0), np.cov(H, rowvar=False)
    evals, evecs = np.linalg.eigh(Sig)
    evals = np.clip(evals, 1e-8, None)
    Wh = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    res["geom_whitened"] = {"cos_dHB_dBI": round(cos(Wh @ d_HB, Wh @ d_BI), 4)}

    # s1 的"到 M_base 距离"（对角白化）
    sd_b = H[is_base].std(0) + 1e-6
    dist = np.linalg.norm((H - mu_b) / sd_b, axis=1)
    res["dist_by_group"] = {
        "human": round(float(dist[is_human].mean()), 3),
        "base_machine": round(float(dist[is_base].mean()), 3),
        "instruct_machine": round(float(dist[is_instr].mean()), 3),
    }
    res["corr_dist_s1"] = round(float(np.corrcoef(dist, s1)[0, 1]), 4)
    e1 = ROOT / "runs/kernel_e1/rstar_total.npz"
    if e1.exists():
        d = np.load(e1, allow_pickle=True)
        res["corr_dist_rstar_200"] = round(float(np.corrcoef(dist[d["row"]], d["r"])[0, 1]), 4)

    # 读出方向 w1/w2（v1.0）
    ck = torch.load(str(ROOT / "runs/v0.4.1_covreg/last.pt"), map_location="cpu",
                    weights_only=False)
    w1 = ck["state"]["w1.weight"].float().numpy().reshape(-1).astype("float64")
    w2 = ck["state"]["w2.weight"].float().numpy().reshape(-1).astype("float64")
    res["readout_geom_raw"] = {
        "cos_w1_dHB": round(cos(w1, d_HB), 4), "cos_w2_dHB": round(cos(w2, d_HB), 4),
        "cos_w1_dBI": round(cos(w1, d_BI), 4), "cos_w2_dBI": round(cos(w2, d_BI), 4),
        "cos_w1_w2": round(cos(w1, w2), 4)}
    res["readout_geom_whitened"] = {
        "cos_w1_dHB": round(cos(Wh @ w1, Wh @ d_HB), 4),
        "cos_w2_dBI": round(cos(Wh @ w2, Wh @ d_BI), 4)}

    # 族对位移分解（规范化配对：同族 base+instruct 各 ≥3 样本）
    fam_b, fam_i = defaultdict(list), defaultdict(list)
    for i in range(len(H)):
        if lab[i] == 0:
            continue
        key = canon(gen[i])
        (fam_i if is_instr[i] else fam_b)[key].append(i)
    fams = [g for g in fam_b if g in fam_i and len(fam_b[g]) >= 3 and len(fam_i[g]) >= 3]
    res["paired_families"] = {g: [len(fam_b[g]), len(fam_i[g])] for g in fams}
    D = {}
    for g in fams:
        d = H[fam_i[g]].mean(0) - H[fam_b[g]].mean(0)
        D[g] = d / (np.linalg.norm(d) + 1e-12)
    if len(D) >= 3:
        A = np.mean(list(D.values()), axis=0)
        a = A / (np.linalg.norm(A) + 1e-12)
        cos_with_a = {g: round(cos(d, a), 3) for g, d in D.items()}
        res["family_disp"] = {"n_families": len(D), "cos_with_common_axis": cos_with_a,
                              "mean_cos_with_axis": round(float(np.mean(list(cos_with_a.values()))), 3)}
        # 残差间平均 |cos|（族私有程度）
        Rs = [d - np.dot(d, a) * a for d in D.values()]
        Rs = [r / (np.linalg.norm(r) + 1e-12) for r in Rs]
        cc = [abs(cos(Rs[i], Rs[j])) for i in range(len(Rs)) for j in range(i + 1, len(Rs))]
        res["family_disp"]["mean_abs_cos_residuals"] = round(float(np.mean(cc)), 3)
        res["readout_geom_raw"]["cos_w2_common_axis"] = round(cos(w2, a), 4)
        res["readout_geom_raw"]["cos_w1_common_axis"] = round(cos(w1, a), 4)

    (OUT / "geometry.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("[e8] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
