#!/usr/bin/env python
"""E25 阶段 0：特征重导 + 先验修复 + B0–B3 离线对照（不训练新编码头）。

依据 `docx/d-det_E25_服务端指导文档_2026-09-29.md`：
  · 用 E24 的 head.pt/stats.npz 与现成编码器，**同序**重导 train/val/test_seen/unseen 特征，
    保存样本映射（来源 split、generator、family、language、长度、ids 哈希）；
  · 修复训练/评估先验不一致（logπ_f 用条件先验 π_f|AI，检测比值只加一次），
    计算正确方向的拒识 AUC（同时保留原方向）；
  · 同一批特征上做 B0（raw mean）/ B1（z̄ 1537）/ B1-μ（μ 768）/ B3（r_F 1537）离线对照
    （检测=固定正则逻辑回归；家族=线性 softmax + DiscHead）；B2=修复后温度能量对照；
  · 两类置换检查（固定 H 行置换 / 原始 token 置换）与文件哈希；ids 精确去重审计。

选择只用 val；test_seen/unseen 留给阶段 1 后的统一评估。

输出：runs/flagship_e25/{features.npz, stage0.json}
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from models.disc import DiscHead  # noqa: E402
from flagship_round1 import (Doc, SetPool, lab_of, z_pass, residualize,  # noqa: E402
                             s_d_scalar, fam_post, SEEN_LABS, UNSEEN_LABS)

BIG = ROOT / "data/processed/semeval_big"
OUT = ROOT / "runs/flagship_e25"

# 旗舰/强模型候选（按 generator 名判定；主观标注，用于报告"跨实验室 vs 跨旗舰"口径）
FLAGSHIP_HINTS = ("GPT-4o", "DeepSeek-V3", "Qwen2.5-Coder-7B", "Qwen2.5-Coder-32B",
                  "Llama-3.1-405B", "Llama-3.3-70B", "Llama-4", "gemma-3-27b",
                  "phi-4", "Mistral-7B", "Devstral", "Yi-Coder-9B")


def load_corpus_tagged(args):
    """与 flagship_round1.load_corpus 完全相同的选择（同种子序列），额外标记来源。"""
    rng = np.random.RandomState(0)

    def subset(path, human_cap, fam_cap, unseen_cap, split_name):
        rows = pq.read_table(path, columns=["ids", "generator", "language"]).to_pylist()
        human, seen = [], []
        by_fam, unseen = {}, {}
        for ri, r in enumerate(rows):
            lab = lab_of(r["generator"])
            if lab == "UNKNOWN":
                continue
            ids = np.asarray(r["ids"][:args.max_len], dtype="int64")
            if len(ids) == 0:
                continue
            d = Doc(ids, int(lab != "Human"), lab, str(r["language"]))
            d.split = split_name
            d.gen = str(r["generator"])
            d.row = ri
            if lab == "Human":
                human.append(d)
            elif lab in SEEN_LABS:
                by_fam.setdefault(lab, []).append(d)
            elif lab in UNSEEN_LABS:
                unseen.setdefault(lab, []).append(d)
        rng.shuffle(human)
        out = {"human": human[:human_cap], "seen": [], "unseen": []}
        for f in SEEN_LABS:
            lst = by_fam.get(f, [])
            rng.shuffle(lst)
            out["seen"].extend(lst[:fam_cap])
        for f, lst in unseen.items():
            rng.shuffle(lst)
            out["unseen"].extend(lst[:unseen_cap])
        return out

    tr = subset(BIG / "b_train.parquet", args.train_human, args.train_fam,
                args.unseen_cap, "train")
    va = subset(BIG / "b_val.parquet", args.val_human, args.val_fam,
                args.unseen_cap, "val")
    te = subset(BIG / "b_test.parquet", args.test_human, args.test_fam,
                args.unseen_cap, "test")
    return {"train": tr["human"] + tr["seen"], "val": va["human"] + va["seen"],
            "test_seen": te["human"] + te["seen"],
            "unseen": tr["unseen"] + va["unseen"] + te["unseen"]}


def ids_md5(ids: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = Namespace(smoke=False, max_len=2048, train_human=8000, train_fam=1200,
                     val_human=1500, val_fam=500, test_human=500, test_fam=200,
                     unseen_cap=600, enc_bs=32)

    # ---- 编码器 + E24 头 ----
    with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc0 = build_encoder(**enc_cfg)
    dual = build_model("dual", encoder=enc0, dim=enc0.hidden_size,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    ck = torch.load(str(ROOT / "runs/v0.4.1_covreg/last.pt"), map_location="cpu",
                    weights_only=False)
    dual.load_state_dict(ck["state"], strict=False)
    enc = dual.encoder.to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    head = SetPool(768, 512).to(device)
    head.load_state_dict(torch.load(OUT.parent / "flagship_r1/head.pt",
                                    map_location=device)["head"])
    head.eval()
    st = dict(np.load(OUT.parent / "flagship_r1/stats.npz", allow_pickle=True))
    bD_e24 = float(torch.load(OUT.parent / "flagship_r1/head.pt",
                              map_location="cpu")["bD"])

    # ---- 同序语料 + 特征导出 ----
    corpus = load_corpus_tagged(args)
    print("[e25s0] 语料：" + " | ".join(f"{k}:{len(v)}" for k, v in corpus.items()),
          flush=True)
    feats = {}
    for name, docs in corpus.items():
        z, rm = z_pass(enc, head, docs, args.enc_bs, device, True)
        feats[name] = {"z": z, "raw_mean": rm, "docs": docs}
        print(f"[e25s0] {name}: z {z.shape} raw_mean {rm.shape}", flush=True)

    # ---- 去重审计（ids 精确） ----
    seen = {}
    dup = {"within": {}, "cross": {}}
    for name in corpus:
        hs = [ids_md5(d.ids) for d in feats[name]["docs"]]
        n_dup = len(hs) - len(set(hs))
        dup["within"][name] = n_dup
        for h in hs:
            seen.setdefault(h, []).append(name)
    pairs = {}
    for h, where in seen.items():
        u = sorted(set(where))
        if len(u) > 1:
            pairs["|".join(u)] = pairs.get("|".join(u), 0) + 1
    dup["cross"] = pairs
    print(f"[e25s0] 去重审计：within {dup['within']} / cross {pairs}", flush=True)

    # ---- 元数据与标准化 ----
    z_tr = feats["train"]["z"]
    m, sd = z_tr.mean(0), np.maximum(z_tr.std(0), 1e-6)
    meta = {}
    for name in corpus:
        docs = feats[name]["docs"]
        meta[name] = {
            "split": [d.split for d in docs], "generator": [d.gen for d in docs],
            "family": [d.fam for d in docs], "language": [d.lang for d in docs],
            "n_tokens": [int(len(d.ids)) for d in docs],
            "ids_md5": [ids_md5(d.ids) for d in docs]}
    np.savez_compressed(
        OUT / "features.npz",
        **{f"z_{k}": v["z"] for k, v in feats.items()},
        **{f"rm_{k}": v["raw_mean"] for k, v in feats.items()},
        z_mean=m, z_std=sd)
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    print("[e25s0] features.npz / meta.json 已写出", flush=True)

    def std(a):
        return (a - m) / sd

    # ---- B0–B3（val 口径；选择只看 val） ----
    val = corpus["val"]
    y_val = np.array([d.y_ai for d in val])
    ai_val = np.array([d.y_ai == 1 and d.fam in SEEN_LABS for d in val])
    fam_val = np.array([SEEN_LABS.index(d.fam) for d, s in zip(val, ai_val) if s])
    tr = corpus["train"]
    y_tr = np.array([d.y_ai for d in tr])
    ai_tr = np.array([d.y_ai == 1 for d in tr])
    fam_tr = np.array([SEEN_LABS.index(d.fam) for d in tr if d.y_ai == 1])

    rF_tr = residualize(feats["train"]["z"], st)
    rF_val = residualize(feats["val"]["z"], st)

    def det_probe(Xtr, Xva, tag):
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), y_tr)
        prob = lr.predict_proba(sc.transform(Xva))[:, 1]
        auc = float(roc_auc_score(y_val, prob))
        f1 = float(f1_score(y_val, (prob > 0.5).astype(int), average="macro"))
        print(f"[e25s0] det·{tag:<12} AUROC {auc:.4f} / F1 {f1:.4f}", flush=True)
        return {"auroc": round(auc, 4), "macro_f1": round(f1, 4)}

    def fam_probe(Xtr, Xva, tag):
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), fam_tr)
        p_lr = lr.predict(sc.transform(Xva))
        dh = DiscHead().fit(Xtr, fam_tr)
        p_dh = dh.predict(Xva)
        r = {"logreg": {"acc": round(float((p_lr == fam_val).mean()), 4),
                        "balanced_acc": round(float(balanced_accuracy_score(fam_val, p_lr)), 4)},
             "dischead": {"acc": round(float((p_dh == fam_val).mean()), 4),
                          "balanced_acc": round(float(balanced_accuracy_score(fam_val, p_dh)), 4)}}
        print(f"[e25s0] fam·{tag:<12} logreg {r['logreg']} | DiscHead {r['dischead']}",
              flush=True)
        return r

    report = {"dedup": dup, "files": {}}
    rm_tr, rm_val_ = feats["train"]["raw_mean"], feats["val"]["raw_mean"]
    ztr_ai = feats["train"]["z"][ai_tr]
    report["B0_raw_mean"] = {"det": det_probe(rm_tr, rm_val_, "raw_mean"),
                             "fam": fam_probe(rm_tr[ai_tr], rm_val_[ai_val], "raw_mean")}
    report["B1_z"] = {"det": det_probe(feats["train"]["z"], feats["val"]["z"], "z"),
                      "fam": fam_probe(ztr_ai, feats["val"]["z"][ai_val], "z")}
    report["B1_mu"] = {"det": det_probe(feats["train"]["z"][:, :768],
                                        feats["val"]["z"][:, :768], "mu"),
                       "fam": fam_probe(ztr_ai[:, :768],
                                        feats["val"]["z"][ai_val][:, :768], "mu")}
    report["B3_rF"] = {"det": det_probe(rF_tr, rF_val, "rF"),
                       "fam": fam_probe(rF_tr[ai_tr], rF_val[ai_val], "rF")}

    # ---- B2：修复后温度能量（E24 统计与头；诊断用） ----
    sD_e24 = s_d_scalar(feats["val"]["z"], st, bD_e24)
    auc = float(roc_auc_score(y_val, sD_e24))
    f1c = float(f1_score(y_val, (1 / (1 + np.exp(-np.clip(sD_e24, -50, 50))) > 0.5)
                         .astype(int), average="macro"))
    # 旧训练公式（含双重先验）的诊断对照：sD_train = sD_eval + log πAI（常数平移）
    log_pi_ai = float(np.log(st["pi"][1:].sum()))
    f1o = float(f1_score(y_val, (1 / (1 + np.exp(-np.clip(sD_e24 + log_pi_ai, -50, 50)))
                                 > 0.5).astype(int), average="macro"))
    report["B2_energy"] = {"auroc": round(auc, 4), "macro_f1_fixed": round(f1c, 4),
                           "macro_f1_oldtrain": round(f1o, 4),
                           "note": "AUROC 不变；F1 受先验修复影响（差 = −2·logπAI 平移）"}
    print(f"[e25s0] B2 energy: AUROC {auc:.4f} F1(fixed) {f1c:.4f} F1(old) {f1o:.4f}",
          flush=True)

    # ---- s_D 训练/推断一致性（torch vs numpy 同一公式） ----
    with torch.no_grad():
        zt = torch.as_tensor(feats["val"]["z"], device=device)
        mu_c = torch.as_tensor(st["mu"], device=device)
        v_c = torch.as_tensor(st["v"], device=device)
        pi = torch.as_tensor(st["pi"], device=device)
        r = zt[:, None, :] - mu_c[None]
        G = 0.5 * ((r * r) / v_c[None] + torch.log(v_c)[None]).sum(-1) / zt.shape[1]
        logpi_f = torch.log(pi[1:] / pi[1:].sum())
        g_ai = -torch.logsumexp(logpi_f - G[:, 1:], dim=1)
        sD_t = (G[:, 0] - g_ai + torch.log(pi[1:].sum() / pi[0]) + bD_e24).cpu().numpy()
    report["sD_torch_vs_numpy_maxdiff"] = float(np.abs(sD_t - sD_e24).max())
    print(f"[e25s0] sD tor/np maxdiff = {report['sD_torch_vs_numpy_maxdiff']:.2e}",
          flush=True)

    # ---- 置换检查（两种输入层级） ----
    d0 = val[0]
    with torch.no_grad():
        L = len(d0.ids)
        ids0 = torch.as_tensor(d0.ids[None, :], device=device)
        mask0 = torch.ones_like(ids0)
        h0 = enc(ids0, mask0)
        z0 = head(h0, mask0).cpu().numpy()
        perm = np.random.RandomState(0).permutation(L)
        z1 = head(h0[:, perm, :], mask0[:, perm]).cpu().numpy()
        idsp = ids0[:, perm]
        zp = head(enc(idsp, mask0), mask0).cpu().numpy()
    report["perm_fixedH_rowperm_maxdiff"] = float(np.abs(z0 - z1).max())
    report["perm_rawtoken_maxdiff"] = float(np.abs(z0 - zp).max())
    report["files"]["results_json_md5"] = hashlib.md5(
        (OUT.parent / "flagship_r1/results.json").read_bytes()).hexdigest()
    report["files"]["features_npz_md5"] = hashlib.md5(
        (OUT / "features.npz").read_bytes()).hexdigest()
    print(f"[e25s0] perm: fixedH {report['perm_fixedH_rowperm_maxdiff']:.2e} / "
          f"rawtoken {report['perm_rawtoken_maxdiff']:.4f}", flush=True)

    # ---- 拒识（两个方向；E24 能量族 + B1 logreg 族） ----
    def rej(scores_seen, scores_unseen, tag):
        lab = np.array([0] * len(scores_seen) + [1] * len(scores_unseen))
        scr = np.concatenate([scores_seen, scores_unseen])
        a1 = float(roc_auc_score(lab, scr))
        a2 = float(roc_auc_score(lab, -scr))
        print(f"[e25s0] 拒识·{tag}: AUC(maxp) {a1:.4f} / AUC(−maxp) {a2:.4f}", flush=True)
        return {"auc_maxp": round(a1, 4), "auc_neg_maxp": round(a2, 4)}

    pi_e24 = st["pi"]
    rF_val_e24 = residualize(feats["val"]["z"], st)
    rF_unseen_e24 = residualize(feats["unseen"]["z"], st)
    pv_e24 = fam_post(rF_val_e24, st["fam_res_mu"], st["fam_res_v"], pi_e24)
    pu_e24 = fam_post(rF_unseen_e24, st["fam_res_mu"], st["fam_res_v"], pi_e24)
    seen_p = [pv_e24[i].max() for i, d in enumerate(val) if d.y_ai]
    unseen_p = [pu_e24[i].max() for i in range(len(corpus["unseen"]))]
    report["rej_e24_energy"] = rej(seen_p, unseen_p, "E24能量族")

    sc = StandardScaler().fit(ztr_ai)
    lr_f = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(ztr_ai), fam_tr)
    pv_b1 = lr_f.predict_proba(sc.transform(feats["val"]["z"][ai_val]))
    pu_b1 = lr_f.predict_proba(sc.transform(feats["unseen"]["z"]))
    seen_p1 = [pv_b1[i].max() for i in range(len(pv_b1))]
    unseen_p1 = [pu_b1[i].max() for i in range(len(pu_b1))]
    report["rej_b1_logreg"] = rej(seen_p1, unseen_p1, "B1 z logreg 族")

    # ---- 家族→generator 清单（含旗舰子集计数） ----
    gen_counts = {}
    for name in ("train", "val", "test_seen", "unseen"):
        for d in corpus[name]:
            key = (name, d.fam, d.gen)
            gen_counts[key] = gen_counts.get(key, 0) + 1
    gen_table = {}
    for (split, fam, gen), n in sorted(gen_counts.items()):
        gen_table.setdefault(fam, {}).setdefault(gen, {})[split] = n
    report["generators"] = gen_table
    flag_counts = {"train": {"n": 0, "gen": {}}, "test_seen": {"n": 0, "gen": {}},
                   "val": {"n": 0, "gen": {}}, "unseen": {"n": 0, "gen": {}}}
    for name in flag_counts:
        for d in corpus[name]:
            if any(h.lower() in d.gen.lower() for h in FLAGSHIP_HINTS):
                flag_counts[name]["n"] += 1
                flag_counts[name]["gen"][d.gen] = flag_counts[name]["gen"].get(d.gen, 0) + 1
    report["flagship_subset"] = flag_counts
    print(f"[e25s0] 旗舰子集（主观判定）：" + " / ".join(
        f"{k}:{v['n']}" for k, v in flag_counts.items()), flush=True)

    # ---- 阶段 1 门控 ----
    m0_bal = report["B0_raw_mean"]["fam"]["dischead"]["balanced_acc"]
    z_best = max(report["B1_z"]["fam"]["logreg"]["balanced_acc"],
                 report["B1_z"]["fam"]["dischead"]["balanced_acc"])
    report["gate"] = {"m0_val_fam_bal": m0_bal, "z_best_val_fam_bal": z_best,
                      "pass": bool(z_best >= m0_bal - 1e-9)}
    print(f"[e25s0] 门控：z 最佳 bal {z_best:.4f} vs M0 {m0_bal:.4f} → "
          f"{'PASS（可跑阶段1）' if report['gate']['pass'] else 'FAIL（暂停阶段1）'}",
          flush=True)

    report["timing_min"] = round((time.time() - t0) / 60, 1)
    (OUT / "stage0.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                default=str))
    print(f"[e25s0] 完成 {report['timing_min']} min → {OUT/'stage0.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
