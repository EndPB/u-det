#!/usr/bin/env python
"""Round3 修正版 MLP 套件（审计修复：eval 模式 / BN buffers / 断言 / no-op 复现）。

修复点（相对 round1/round2 的 MLP 训练）：
- 每个 epoch 开始所有分支/头/adversary 调 train()；dev/test/融合 LR 表示提取前调 eval() + inference_mode；
- best state 保存并恢复完整 state_dict（含 BatchNorm buffers）；评估后继续训练前恢复 train()；
- cudnn.deterministic=True（审计可复现；仅小型 MLP）；
- 预测包含 identity（task_id / source_sha256 / model_name / y_true / family 顺序 / split hash / seed）；
- 记录 dev 曲线、每项损失梯度非零探针、正对覆盖率、数据级 family 覆盖；
- no-op 复现检查：full_disentangle(seed0) 独立重跑两遍，比较 dev 曲线与 test 概率（精确）。

臂：semantic_only / fingerprint_only / late_fusion / lf_nosupcon / full_disentangle /
    fd_nosemgrl / fd_nofpgrl / fd_noorth（各 3 seeds；dev 规则与原实现一致）。
输出：artifacts/acl_dcan_round3_audit/{mlp/, mode_assertions.json, logs/}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import dcan_four_models as r1  # noqa: E402

OUT = ROOT / "artifacts" / "acl_dcan_round3_audit"
OUT.mkdir(parents=True, exist_ok=True)

ARM_CFG = {
    "semantic_only":    dict(want_s=1, want_f=0, supcon=0.3, sem_grl=0, fp_grl=0, orth=0.00, dev="sem"),
    "fingerprint_only": dict(want_s=0, want_f=1, supcon=0.0, sem_grl=0, fp_grl=0, orth=0.00, dev="fp"),
    "late_fusion":      dict(want_s=1, want_f=1, supcon=0.3, sem_grl=0, fp_grl=0, orth=0.00, dev="fuse"),
    "lf_nosupcon":      dict(want_s=1, want_f=1, supcon=0.0, sem_grl=0, fp_grl=0, orth=0.00, dev="fuse"),
    "full_disentangle": dict(want_s=1, want_f=1, supcon=0.3, sem_grl=1, fp_grl=1, orth=0.05, dev="fp"),
    "fd_nosemgrl":      dict(want_s=1, want_f=1, supcon=0.3, sem_grl=0, fp_grl=1, orth=0.05, dev="fp"),
    "fd_nofpgrl":       dict(want_s=1, want_f=1, supcon=0.3, sem_grl=1, fp_grl=0, orth=0.05, dev="fp"),
    "fd_noorth":        dict(want_s=1, want_f=1, supcon=0.3, sem_grl=1, fp_grl=1, orth=0.00, dev="fp"),
}


def build(seed, st_dim, emb_dim):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    mods = {
        "ms": r1.MLP(emb_dim), "mf": r1.MLP(st_dim),
        "hs": nn.Linear(128, 6), "hf": nn.Linear(128, 6),
        "as": nn.Linear(128, 6), "al": nn.Linear(128, 16), "ac": nn.Linear(128, 64),
    }
    return mods


def get_state(mods):
    return {f"{n}.{k}": v.detach().cpu().clone() for n, m in mods.items() for k, v in m.state_dict().items()}


def set_state(mods, state):
    for n, m in mods.items():
        sd = m.state_dict()
        for k, v in sd.items():
            v.copy_(state[f"{n}.{k}"])


def f1(y, pred):
    import sklearn.metrics as sm
    return float(sm.f1_score(y, pred, average="macro", zero_division=0))


def run_arm(seed, arm, cfg, data, args, tag_suffix=""):
    dev_ = "cuda"
    emb, st, fam, task, split, clu, lbin, model_name = data
    fam_idx = {f: i for i, f in enumerate(r1.FAMILIES)}
    y = np.array([fam_idx[f] for f in fam])
    tr = np.array([i for i, s in enumerate(split) if s == "train"])
    dv = np.array([i for i, s in enumerate(split) if s == "dev"])
    te = np.array([i for i, s in enumerate(split) if s == "test"])
    Xe = torch.tensor(emb, dtype=torch.float32, device=dev_)
    Xs = torch.tensor(st, dtype=torch.float32, device=dev_)
    Y = torch.tensor(y, dtype=torch.long, device=dev_)
    Cc = torch.tensor(clu, dtype=torch.long, device=dev_)
    Lb = torch.tensor(lbin, dtype=torch.long, device=dev_)
    rng = np.random.default_rng(seed)
    mods = build(seed, st.shape[1], emb.shape[1])
    for m in mods.values():
        m.to(dev_)
    opt = torch.optim.AdamW([p for m in mods.values() for p in m.parameters()], lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    T = np.asarray(task)

    def forward_probs(xe=None, xs=None, eval_mode=True):
        if eval_mode:
            for m in mods.values():
                m.eval()
        with torch.inference_mode():
            ps = F.softmax(mods["hs"](mods["ms"](Xe if xe is None else xe)), 1) if cfg["want_s"] else None
            pf = F.softmax(mods["hf"](mods["mf"](Xs if xs is None else xs)), 1) if cfg["want_f"] else None
        return ps, pf

    best = {"dev": -1, "epoch": 0, "state": None}
    dev_curve, probe = [], None
    for ep in range(1, args.epochs + 1):
        for m in mods.values():
            m.train()
        perm = rng.permutation(tr)
        for bi, bidx in enumerate(r1.make_task_batches(perm, T, rng)):
            xb_e, xb_s = Xe[bidx], Xs[bidx]
            yb = Y[bidx]
            tb = torch.tensor(T[bidx], dtype=torch.long, device=dev_)
            parts = {}
            z_s = z_f = None
            if cfg["want_s"]:
                z_s = mods["ms"](xb_e)
                parts["ce_s"] = ce(mods["hs"](z_s), yb)
                if cfg["supcon"]:
                    parts["supcon"] = cfg["supcon"] * r1.supcon(z_s, tb, yb)
            if cfg["want_f"]:
                z_f = mods["mf"](xb_s)
                parts["ce_f"] = ce(mods["hf"](z_f), yb)
            if cfg["sem_grl"]:
                parts["adv_s"] = 0.3 * ce(mods["as"](r1.grl(z_s, 1.0)), yb)
            if cfg["fp_grl"]:
                parts["adv_len"] = 0.2 * ce(mods["al"](r1.grl(z_f, 1.0)), Lb[bidx])
                parts["adv_clu"] = 0.2 * ce(mods["ac"](r1.grl(z_f, 1.0)), Cc[bidx])
            if cfg["orth"]:
                parts["orth"] = cfg["orth"] * r1.cross_cov(z_s, z_f)
            total = sum(parts.values())
            if seed == 0 and ep == 1 and bi == 0:
                probe = {"components": {}, "positive_frac": None}
                for name, l in parts.items():
                    opt.zero_grad()
                    l.backward(retain_graph=True)
                    mods_key = {"ce_s": ["ms", "hs"], "supcon": ["ms"], "ce_f": ["mf", "hf"],
                                "adv_s": ["ms", "as"], "adv_len": ["mf", "al"],
                                "adv_clu": ["mf", "ac"], "orth": ["ms", "mf"]}[name]
                    g = max((float(p.grad.abs().max()) for n in mods_key for p in mods[n].parameters()
                             if p.grad is not None), default=0.0)
                    probe["components"][name] = g
                # 正对覆盖（同批 anchor 有 ≥1 正对的比例）
                same = (tb[:, None] == tb[None, :]) & (yb[:, None] != yb[None, :])
                probe["positive_frac"] = float((same.sum(1) > 0).float().mean())
                probe["batch_families"] = {r1.FAMILIES[i]: int((yb == i).sum()) for i in range(6)}
            opt.zero_grad()
            total.backward()
            opt.step()
        ps, pf = forward_probs()
        rec = {"epoch": ep}
        if ps is not None:
            rec["dev_sem"] = f1(y[dv], ps[dv].cpu().numpy().argmax(1))
        if pf is not None:
            rec["dev_fp"] = f1(y[dv], pf[dv].cpu().numpy().argmax(1))
        if cfg["dev"] == "sem":
            rec["dev_sel"] = rec["dev_sem"]
        elif cfg["dev"] == "fp":
            rec["dev_sel"] = rec["dev_fp"]
        else:
            probs = (ps[dv].cpu().numpy() + pf[dv].cpu().numpy()) / 2
            rec["dev_sel"] = f1(y[dv], probs.argmax(1))
        dev_curve.append(rec)
        if rec["dev_sel"] > best["dev"]:
            best = {"dev": rec["dev_sel"], "epoch": ep, "state": get_state(mods)}
        if ep - best["epoch"] >= 10:
            break
    set_state(mods, best["state"])
    ps, pf = forward_probs()

    result = {"seed": seed, "arm": arm, "config": cfg, "best_dev_f1": best["dev"],
              "best_epoch": best["epoch"], "dev_curve": dev_curve, "probe": probe}
    te_out = {}
    probs_pack = {}
    gen_te = [model_name[i] for i in te]
    if ps is not None:
        te_out["sem"] = r1.metric_from_probs(ps[te].cpu().numpy().astype(np.float64), y[te],
                                             {"generator": gen_te})
        probs_pack["sem"] = ps[te].cpu().numpy().astype(np.float16)
    if pf is not None:
        te_out["fp"] = r1.metric_from_probs(pf[te].cpu().numpy().astype(np.float64), y[te],
                                            {"generator": gen_te})
        probs_pack["fp"] = pf[te].cpu().numpy().astype(np.float16)
    if cfg["want_s"] and cfg["want_f"]:
        for m in mods.values():
            m.eval()
        with torch.inference_mode():
            z_tr = torch.cat([mods["ms"](Xe[tr]), mods["mf"](Xs[tr])], 1).cpu().numpy()
            z_te = torch.cat([mods["ms"](Xe[te]), mods["mf"](Xs[te])], 1).cpu().numpy()
        import warnings
        from sklearn.linear_model import LogisticRegression
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(z_tr, y[tr])
        converged = not any("converge" in str(w.message).lower() for w in wlist)
        fuse = lr.predict_proba(z_te)
        te_out["fuse"] = r1.metric_from_probs(fuse.astype(np.float64), y[te], {"generator": gen_te})
        te_out["fuse"]["lr_converged"] = bool(converged)
        probs_pack["fuse"] = fuse.astype(np.float16)
    result["test"] = te_out
    result["probs"] = probs_pack
    result["te_idx"] = te.tolist()
    return result


def mode_assertions():
    """模式断言：同输入两评一致、BN buffers 不变、批组成容差、split 不交叉。"""
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    m = r1.MLP(31).to(dev_)
    x = torch.randn(64, 31, device=dev_)
    out = {}
    m.eval()
    with torch.inference_mode():
        a = m(x.clone())
        b = m(x.clone())
    out["double_eval_exact_equal"] = bool(torch.equal(a, b))
    out["double_eval_maxdiff"] = float((a - b).abs().max())
    bn = m.net[1]
    before = (bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone())
    with torch.inference_mode():
        _ = m(x.clone())
    after = (bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone())
    out["bn_buffers_unchanged_in_eval"] = all(torch.equal(u, v) for u, v in zip(before, after))
    m.train()
    _ = m(x.clone())  # 训练模式会有 BN 更新（对照）
    out["bn_buffers_changed_in_train"] = not torch.equal(before[0], bn.running_mean)
    m.eval()
    with torch.inference_mode():
        full = m(x.clone())
        chunks = torch.cat([m(x[i:i + 16].clone()) for i in range(0, 64, 16)], 0)
    out["batch_composition_maxdiff_eval"] = float((full - chunks).abs().max())
    out["note"] = "batch_composition 差异应为数值容差级；train 模式对照证明 BN 更新来源"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--arms", default="all")
    ap.add_argument("--out", default=str(OUT / "mlp"))
    ap.add_argument("--noop-check", action="store_true")
    args = ap.parse_args()
    outd = Path(args.out)
    if (outd / "metrics.json").exists() and not args.force:
        raise SystemExit(f"refuse to overwrite existing {outd}/metrics.json (use --out)")
    outd.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = r1.load_rows()
    emb = r1.encode_semantics(rows)
    st_raw = np.array([r1.struct_features(r) for r in rows], dtype=np.float32)
    split = [r["task_split"] for r in rows]
    fam = [r["family"] for r in rows]
    task = [r["task_id"] for r in rows]
    model_name = [r["model_name"] for r in rows]
    sha = [r["source_sha256"] for r in rows]
    tmap = {t: i for i, t in enumerate(dict.fromkeys(task))}
    task_int = np.array([tmap[t] for t in task])
    tr_mask = np.array([s == "train" for s in split])
    mu, sd = st_raw[tr_mask].mean(0), st_raw[tr_mask].std(0) + 1e-6
    st = (st_raw - mu) / sd
    clu = r1.build_clusters(rows, tr_mask)
    lbin = r1.length_bins(rows, tr_mask)
    te = np.array([i for i, s in enumerate(split) if s == "test"])

    # split 断言
    t_tr = {task[i] for i in range(len(rows)) if split[i] == "train"}
    t_dv = {task[i] for i in range(len(rows)) if split[i] == "dev"}
    t_te = {task[i] for i in range(len(rows)) if split[i] == "test"}
    assert not (t_tr & t_dv) and not (t_tr & t_te) and not (t_dv & t_te), "task split overlap!"

    ma = mode_assertions()
    ma["task_splits_disjoint"] = True
    ma["test_excluded_from_selection"] = True
    (OUT / "mode_assertions.json").write_text(json.dumps(ma, ensure_ascii=False, indent=1))

    data = (emb, st, fam, task_int, split, clu, lbin, model_name)
    arms = list(ARM_CFG) if args.arms == "all" else args.arms.split(",")
    metrics = {"config": {"script": "scripts/dcan_round3_mlp.py", "arms": {a: ARM_CFG[a] for a in arms},
                          "seeds": args.seeds, "epochs": args.epochs,
                          "dev_rules": "与原实现一致（sem/fp/fuse 平均）",
                          "deterministic": "cudnn.deterministic=True, benchmark=False"},
               "data_coverage": {"train_family_counts": dict(Counter(fam[i] for i in np.where(tr_mask)[0]))},
               "runs": [], "noop_repro": None}
    dev_curves = {}
    pack = {"y_test": np.array([r1.FAMILIES.index(fam[i]) for i in te], dtype=np.int16),
            "task_id_test": np.array([task[i] for i in te], dtype=object),
            "source_sha256_test": np.array([sha[i] for i in te], dtype=object),
            "model_name_test": np.array([model_name[i] for i in te], dtype=object),
            "family_order": np.array(r1.FAMILIES, dtype=object)}
    split_blob = "\n".join(f"{task[i]}\t{sha[i]}" for i in te)
    import hashlib
    pack["split_hash_test"] = np.array(hashlib.sha256(split_blob.encode()).hexdigest())

    def run_one(seed, arm, suffix=""):
        r = run_arm(seed, arm, ARM_CFG[arm], data, args, suffix)
        dev_curves[f"{arm}_s{seed}{suffix}"] = r.pop("dev_curve")
        for k, p in r.pop("probs").items():
            pack[f"{arm}_s{seed}{suffix}_{k}"] = p
        r.pop("te_idx")
        return r

    for seed in range(args.seeds):
        for arm in arms:
            r = run_one(seed, arm)
            metrics["runs"].append(r)
            print(f"[r3] s{seed} {arm}: " + json.dumps(
                {k: round(v["macro_f1"], 4) for k, v in r["test"].items() if isinstance(v, dict)}), flush=True)

    if args.noop_check:
        r1x = run_one(0, "full_disentangle", "_noopA")
        r2x = run_one(0, "full_disentangle", "_noopB")
        da = np.array(dev_curves["full_disentangle_s0_noopA"]).tolist()
        db = np.array(dev_curves["full_disentangle_s0_noopB"]).tolist()
        pa = pack["full_disentangle_s0_noopA_fuse"].astype(np.float32)
        pb = pack["full_disentangle_s0_noopB_fuse"].astype(np.float32)
        metrics["noop_repro"] = {
            "dev_curve_equal": bool(da == db),
            "fuse_probs_maxdiff": float(np.abs(pa - pb).max()),
            "test_f1_A": r1x["test"]["fuse"]["macro_f1"], "test_f1_B": r2x["test"]["fuse"]["macro_f1"],
            "best_epoch_A": r1x["best_epoch"], "best_epoch_B": r2x["best_epoch"],
            "note": "同 seed 同配置两次独立训练；cudnn.deterministic=True"}

    (outd / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    (outd / "dev_curves.json").write_text(json.dumps(dev_curves, ensure_ascii=False, indent=1))
    np.savez_compressed(outd / "predictions.npz", **pack)
    print(f"[r3] done {round(time.time()-t0,1)}s -> {outd}")


if __name__ == "__main__":
    main()
