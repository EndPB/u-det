#!/usr/bin/env python
"""Round2 Q2：DCAN 失败首要原因——单变量消融（以 late-fusion .713 为基线）。

变体（每次只关一个机制；超参/种子/采样/评测完全同 round1）：
  lf_nosupcon   : late-fusion（双分支 CE）去掉 SupCon；
  fd_nosemgrl   : full-disentangle 去掉 semantic family-GRL；
  fd_nofpgrl    : full-disentangle 去掉 fingerprint（长度/提示簇）GRL；
  fd_noorth     : full-disentangle 去掉交叉协方差项。
参照（不重跑）：round1 late_fusion（含 SupCon）.713±.006、full_disentangle .703±.005。
输出：artifacts/acl_dcan_round2/ablations/{metrics.json, predictions.npz}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dcan_four_models as r1  # noqa: E402

OUT = ROOT / "artifacts" / "acl_dcan_round2" / "ablations"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "lf_nosupcon": {"supcon_w": 0.0, "sem_grl": False, "fp_grl": False, "orth_w": 0.0, "dev": "fuse"},
    "fd_nosemgrl": {"supcon_w": 0.3, "sem_grl": False, "fp_grl": True, "orth_w": 0.05, "dev": "fp"},
    "fd_nofpgrl": {"supcon_w": 0.3, "sem_grl": True, "fp_grl": False, "orth_w": 0.05, "dev": "fp"},
    "fd_noorth": {"supcon_w": 0.3, "sem_grl": True, "fp_grl": True, "orth_w": 0.0, "dev": "fp"},
}


def run_variant(seed, emb, st, fam, task, split, prompt_clu, len_bin, args, name, cfg, model_name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fam_idx = {f: i for i, f in enumerate(r1.FAMILIES)}
    y = np.array([fam_idx[f] for f in fam])
    tr = np.array([i for i, s in enumerate(split) if s == "train"])
    dv = np.array([i for i, s in enumerate(split) if s == "dev"])
    te = np.array([i for i, s in enumerate(split) if s == "test"])
    Xe = torch.tensor(emb, dtype=torch.float32, device=dev)
    Xs = torch.tensor(st, dtype=torch.float32, device=dev)
    Y = torch.tensor(y, dtype=torch.long, device=dev)
    T = np.asarray(task)
    Cc = torch.tensor(prompt_clu, dtype=torch.long, device=dev)
    Lb = torch.tensor(len_bin, dtype=torch.long, device=dev)
    rng = np.random.default_rng(seed)

    mlp_s, mlp_f = r1.MLP(emb.shape[1]).to(dev), r1.MLP(st.shape[1]).to(dev)
    head_s, head_f = nn.Linear(128, 6).to(dev), nn.Linear(128, 6).to(dev)
    adv_s = nn.Linear(128, 6).to(dev)
    adv_len = nn.Linear(128, int(len_bin.max()) + 1).to(dev)
    adv_clu = nn.Linear(128, int(prompt_clu.max()) + 1).to(dev)
    ce = nn.CrossEntropyLoss()
    params = (list(mlp_s.parameters()) + list(mlp_f.parameters()) + list(head_s.parameters())
              + list(head_f.parameters()) + list(adv_s.parameters()) + list(adv_len.parameters())
              + list(adv_clu.parameters()))
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    fd = name.startswith("fd_")

    best = {"f1": -1, "epoch": 0, "state": None}
    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(tr)
        for bidx in r1.make_task_batches(perm, T, rng):
            xb_e, xb_s = Xe[bidx], Xs[bidx]
            yb = Y[bidx]
            tb = torch.tensor(T[bidx], dtype=torch.long, device=dev)
            loss = torch.zeros((), device=dev)
            if fd:
                z_s = mlp_s(xb_e)
                loss = loss + ce(head_s(z_s), yb)          # 与 round1 full_disentangle 完全一致
                loss = loss + cfg["supcon_w"] * r1.supcon(z_s, tb, yb)
                if cfg["sem_grl"]:
                    loss = loss + 0.3 * ce(adv_s(r1.grl(z_s, 1.0)), yb)
                z_f = mlp_f(xb_s)
                loss = loss + ce(head_f(z_f), yb)
                if cfg["fp_grl"]:
                    loss = loss + 0.2 * ce(adv_len(r1.grl(z_f, 1.0)), Lb[bidx])
                    loss = loss + 0.2 * ce(adv_clu(r1.grl(z_f, 1.0)), Cc[bidx])
                if cfg["orth_w"]:
                    loss = loss + cfg["orth_w"] * r1.cross_cov(z_s, z_f)
            else:  # late-fusion 家族：双分支 CE（可选 SupCon）
                z_s = mlp_s(xb_e)
                z_f = mlp_f(xb_s)
                loss = loss + ce(head_s(z_s), yb) + ce(head_f(z_f), yb)
                if cfg["supcon_w"]:
                    loss = loss + cfg["supcon_w"] * r1.supcon(z_s, tb, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ps, pf = F.softmax(head_s(mlp_s(Xe[dv])), 1), F.softmax(head_f(mlp_f(Xs[dv])), 1)
            probs = pf if cfg["dev"] == "fp" else (ps + pf) / 2
            import sklearn.metrics as sm
            f1 = sm.f1_score(Y[dv].cpu().numpy(), probs.argmax(1).cpu().numpy(),
                             average="macro", zero_division=0)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in
                              {**{f"ms.{i}": p for i, p in enumerate(mlp_s.state_dict().values())},
                               **{f"mf.{i}": p for i, p in enumerate(mlp_f.state_dict().values())},
                               **{f"hs.{i}": p for i, p in enumerate(head_s.state_dict().values())},
                               **{f"hf.{i}": p for i, p in enumerate(head_f.state_dict().values())}}.items()}}
        if ep - best["epoch"] >= 10:
            break
    for mod, key in ((mlp_s, "ms"), (mlp_f, "mf"), (head_s, "hs"), (head_f, "hf")):
        for i, p in enumerate(mod.state_dict().values()):
            p.copy_(best["state"][f"{key}.{i}"])

    with torch.no_grad():
        zs_tr, zf_tr = mlp_s(Xe[tr]), mlp_f(Xs[tr])
        zs_te, zf_te = mlp_s(Xe[te]), mlp_f(Xs[te])
        ps_te, pf_te = F.softmax(head_s(zs_te), 1).cpu().numpy(), F.softmax(head_f(zf_te), 1).cpu().numpy()
        z_tr = torch.cat([zs_tr, zf_tr], 1).cpu().numpy()
        z_te = torch.cat([zs_te, zf_te], 1).cpu().numpy()
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(z_tr, y[tr])
    fuse_te = lr.predict_proba(z_te)
    groups = {"generator": [str(model_name[i]) for i in te]}
    y_te = y[te]
    res = {"seed": seed, "variant": name, "dev_rule": cfg["dev"], "best_dev_f1": best["f1"],
           "best_epoch": best["epoch"], "test": {}, "probs": {}}
    res["test"]["sem"] = r1.metric_from_probs(ps_te, y_te, groups)
    res["test"]["fp"] = r1.metric_from_probs(pf_te, y_te, groups)
    res["test"]["fuse"] = r1.metric_from_probs(fuse_te, y_te, groups)
    res["probs"]["fp"] = pf_te.astype(np.float16)
    res["probs"]["fuse"] = fuse_te.astype(np.float16)
    return res


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = 1, 8
    t0 = time.time()
    rows = r1.load_rows()
    emb = r1.encode_semantics(rows)
    st_raw = np.array([r1.struct_features(r) for r in rows], dtype=np.float32)
    split = [r["task_split"] for r in rows]
    fam = [r["family"] for r in rows]
    task = [r["task_id"] for r in rows]
    model_name = np.array([r["model_name"] for r in rows])
    tmap = {t: i for i, t in enumerate(dict.fromkeys(task))}
    task = [tmap[t] for t in task]
    tr_mask = np.array([s == "train" for s in split])
    mu, sd = st_raw[tr_mask].mean(0), st_raw[tr_mask].std(0) + 1e-6
    st = (st_raw - mu) / sd
    prompt_clu = r1.build_clusters(rows, tr_mask)
    len_bin = r1.length_bins(rows, tr_mask)

    # 参照：round1 正式结果（不重跑）
    r1m = json.loads((ROOT / "artifacts" / "acl_dcan_round1" / "dcan_four_models" / "metrics.json").read_text())
    import collections
    ref = collections.defaultdict(list)
    for run in r1m["runs"]:
        for key, m in run["test"].items():
            ref[f"{run['mode']}::{key}"].append(m["macro_f1"])
    out = {"config": {"script": "scripts/dcan_round2_ablations.py", "seeds": args.seeds,
                      "epochs": args.epochs, "variants": VARIANTS,
                      "reference_round1": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                           for k, v in ref.items()},
                      "fixed": "lr 1e-3 / wd 1e-4 / batch 48任务×≤8 / 其余与 round1 完全一致"},
           "runs": []}
    pack = {}
    for seed in range(args.seeds):
        for name, cfg in VARIANTS.items():
            r = run_variant(seed, emb, st, fam, task, split, prompt_clu, len_bin, args, name, cfg, model_name)
            for k, p in r.pop("probs").items():
                pack[f"{name}_s{seed}_{k}"] = p
            out["runs"].append(r)
            print(f"[q2] seed {seed} {name}: " + json.dumps(
                {k: round(v["macro_f1"], 4) for k, v in r["test"].items()}), flush=True)
    np.savez_compressed(OUT / "predictions.npz", **pack)
    (OUT / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[q2] done {round(time.time()-t0,1)}s -> {OUT}")


if __name__ == "__main__":
    main()
