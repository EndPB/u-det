#!/usr/bin/env python
"""E25 阶段 1：新 SetPool + 判别头（L_D + L_F）单训练臂 + 统一终评。

前置：runs/flagship_e25/stage0.json 的 gate.pass = True（否则直接退出并记录负结果）。
协议（对齐指导文档 §3–§4）：
  · 冻结 E24 同款 v1.0 编码器；**全新 SetPool**（seed 0，不得复用 E24 能量损失训练过的头）；
  · 初始化头后对训练折编码一遍 → 固定 z 的 (m, σ)（2 epoch 内不变）；
  · 训练 2 epoch：L = mean_full BCE(w_Dᵀz̄+b_D, y) + mean_AI CE(W_F z̄+b_F, fam)；
  · 每个 epoch 记录训练损失 + 该头的 val 指标（家族 val balanced 为主、检测 AUROC 并列），
    预注册规则选 epoch（bal 最高；并列取 det AUROC 高者）；
  · test_seen/unseen **只最后评估一次**：B0–B3（离线重拟合）+ E25 统一打分；
    检测阈 = 0.5 与 val 最优 F1 阈两口径；家族逐族召回；unseen 检测（vs val Human，
    含原始 split 分解）；拒识两方向；语言/长度桶分解；generator 级召回。

输出：runs/flagship_e25/{e25_final.json, head_e25.pt, z_stats_e25.npz}
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
from flagship_round1 import (SetPool, _batches, collate, z_pass,  # noqa: E402
                             residualize, s_d_scalar, SEEN_LABS,
                             LAMBDA_SHRINK)  # noqa: E402
from flagship_e25_stage0 import load_corpus_tagged  # noqa: E402
from argparse import Namespace  # noqa: E402

OUT = ROOT / "runs/flagship_e25"
ARGS = Namespace(smoke=False, max_len=2048, train_human=8000, train_fam=1200,
                 val_human=1500, val_fam=500, test_human=500, test_fam=200,
                 unseen_cap=600, enc_bs=32)


class DiscHeads(nn.Module):
    def __init__(self, dim=1537, n_fam=8):
        super().__init__()
        self.wD = nn.Linear(dim, 1)
        self.WF = nn.Linear(dim, n_fam)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stage0 = json.loads((OUT / "stage0.json").read_text())
    gate = stage0.get("gate", {})
    print(f"[e25s1] 门控：{gate}", flush=True)
    if not gate.get("pass", False):
        (OUT / "e25_final.json").write_text(json.dumps(
            {"gate": gate, "action": "skipped", "reason": "阶段0 判别式 z 未达 M0 门控"},
            ensure_ascii=False, indent=2))
        print("[e25s1] 门控未过 → 不训练（负结果已记录）", flush=True)
        return 0

    torch.manual_seed(0)
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

    corpus = load_corpus_tagged(ARGS)
    feats = dict(np.load(OUT / "features.npz", allow_pickle=True))   # E24 头特征（B0–B3）
    meta = json.loads((OUT / "meta.json").read_text())
    st_e24 = dict(np.load(OUT.parent / "flagship_r1/stats.npz", allow_pickle=True))
    bD_e24 = float(torch.load(OUT.parent / "flagship_r1/head.pt",
                              map_location="cpu")["bD"])

    head = SetPool(768, 512).to(device)          # 全新头（seed 0）
    heads = DiscHeads(1537, len(SEEN_LABS)).to(device)
    opt = torch.optim.AdamW(list(head.parameters()) + list(heads.parameters()),
                            lr=1e-3, weight_decay=1e-4)

    tr_docs = corpus["train"]
    y_tr = np.array([d.y_ai for d in tr_docs])
    fam_tr_all = np.array([SEEN_LABS.index(d.fam) if d.y_ai else -1 for d in tr_docs])

    # ---- 固定标准化（初始化头后一遍） ----
    z_tr, _ = z_pass(enc, head, tr_docs, ARGS.enc_bs, device, False)
    m, sd = z_tr.mean(0), np.maximum(z_tr.std(0), 1e-6)
    m_t = torch.as_tensor(m, device=device)
    sd_t = torch.as_tensor(sd, device=device)
    np.savez_compressed(OUT / "z_stats_e25.npz", m=m, sd=sd)
    print(f"[e25s1] z 标准化已固定（{len(tr_docs)} 条）", flush=True)

    def head_features(docs, keep_rm=False):
        z, rm = z_pass(enc, head, docs, ARGS.enc_bs, device, keep_rm)
        return (z - m) / sd, rm

    def val_metrics(ep):
        zv, _ = head_features(corpus["val"])
        with torch.no_grad():
            zt = torch.as_tensor(zv, device=device)
            sD = heads.wD(zt).squeeze(-1).cpu().numpy()
            aF = heads.WF(zt).cpu().numpy()
        yv = np.array([d.y_ai for d in corpus["val"]])
        auc = float(roc_auc_score(yv, sD))
        ai_sel = np.array([d.y_ai == 1 and d.fam in SEEN_LABS for d in corpus["val"]])
        famv = np.array([SEEN_LABS.index(d.fam) for d, s in zip(corpus["val"], ai_sel) if s])
        pred = aF[ai_sel].argmax(1)
        bal = float(balanced_accuracy_score(famv, pred))
        print(f"[e25s1] epoch {ep} val: det AUROC {auc:.4f} | fam bal {bal:.4f}",
              flush=True)
        return {"det_auroc": round(auc, 4), "fam_bal": round(bal, 4),
                "fam_acc": round(float((pred == famv).mean()), 4)}

    # ---- 训练 2 epoch ----
    epoch_log, snapshots = [], {}
    for ep in range(2):
        t_ep = time.time()
        tot = 0.0; nb = 0; ld_sum = 0.0; lf_sum = 0.0; n_all = 0; n_ai = 0
        for batch in _batches(tr_docs, 64, True, seed=ep):
            ids, mask = collate(batch, device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=device == "cuda"):
                h = enc(ids, mask)
            z = head(h, mask)
            zb = (z - m_t) / sd_t
            y = torch.as_tensor([d.y_ai for d in batch], dtype=torch.float32,
                                device=device)
            sD = heads.wD(zb).squeeze(-1)
            L_D = F.binary_cross_entropy_with_logits(sD, y)
            ai_i = [i for i, d in enumerate(batch) if d.y_ai == 1]
            if ai_i:
                fam_t = torch.as_tensor([SEEN_LABS.index(batch[i].fam) for i in ai_i],
                                        device=device)
                L_F = F.cross_entropy(heads.WF(zb)[ai_i], fam_t)
            else:
                L_F = torch.zeros((), device=device)
            loss = L_D + L_F
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); ld_sum += float(L_D.detach())
            lf_sum += float(L_F.detach()); nb += 1; n_all += len(batch); n_ai += len(ai_i)
        vm = val_metrics(ep)
        snap = {"epoch": ep, "loss": round(tot / max(nb, 1), 4),
                "L_D": round(ld_sum / max(nb, 1), 4),
                "L_F": round(lf_sum / max(nb, 1), 4),
                "n_batches": nb, "n_samples": n_all, "n_ai": n_ai,
                "val": vm, "seconds": round(time.time() - t_ep, 1),
                "head_state": {k: v.cpu().clone() for k, v in head.state_dict().items()},
                "heads_state": {k: v.cpu().clone() for k, v in heads.state_dict().items()}}
        snapshots[ep] = snap
        epoch_log.append({k: snap[k] for k in ("epoch", "loss", "L_D", "L_F",
                                               "n_batches", "n_samples", "n_ai",
                                               "val", "seconds")})
        print(f"[e25s1] epoch {ep}: loss {snap['loss']} (D {snap['L_D']} F {snap['L_F']}) "
              f"/ {snap['seconds']}s", flush=True)

    # ---- 预注册选 epoch ----
    best_ep = max(snapshots, key=lambda e: (snapshots[e]["val"]["fam_bal"],
                                            snapshots[e]["val"]["det_auroc"]))
    head.load_state_dict(snapshots[best_ep]["head_state"])
    heads.load_state_dict(snapshots[best_ep]["heads_state"])
    print(f"[e25s1] 选 epoch {best_ep}（val fam bal 最高）", flush=True)
    torch.save({"head": head.state_dict(), "heads": heads.state_dict(),
                "epoch": best_ep, "m": m, "sd": sd}, OUT / "head_e25.pt")

    # ---- 统一终评（只评一次） ----
    zv_e25, _ = head_features(corpus["val"])
    zt_e25, _ = head_features(corpus["test_seen"])
    zu_e25, _ = head_features(corpus["unseen"])

    y_val = np.array([d.y_ai for d in corpus["val"]])
    ai_val = np.array([d.y_ai == 1 and d.fam in SEEN_LABS for d in corpus["val"]])
    fam_val = np.array([SEEN_LABS.index(d.fam) for d, s in zip(corpus["val"], ai_val) if s])
    y_te = np.array([d.y_ai for d in corpus["test_seen"]])
    ai_te = np.array([d.y_ai == 1 and d.fam in SEEN_LABS for d in corpus["test_seen"]])
    fam_te = np.array([SEEN_LABS.index(d.fam) for d, s in zip(corpus["test_seen"], ai_te) if s])
    y_tr = np.array([d.y_ai for d in tr_docs])
    ai_tr = np.array([d.y_ai == 1 for d in tr_docs])
    fam_tr = np.array([SEEN_LABS.index(d.fam) for d in tr_docs if d.y_ai == 1])

    def det_arm(name, prob_tr, prob_val, prob_te):
        """检测臂：0.5 与 val 最优 F1 两口径。"""
        f1_half = float(f1_score(y_te, (prob_te > 0.5).astype(int), average="macro"))
        ths = np.linspace(0.05, 0.95, 37)
        f1s = [f1_score(y_val, (prob_val > t).astype(int), average="macro") for t in ths]
        th = float(ths[int(np.argmax(f1s))])
        f1_val_best = float(f1_score(y_te, (prob_te > th).astype(int), average="macro"))
        r = {"auroc_te": round(float(roc_auc_score(y_te, prob_te)), 4),
             "macro_f1_te_th05": round(f1_half, 4),
             "val_best_th": round(th, 3),
             "macro_f1_te_valth": round(f1_val_best, 4),
             "val_auroc": round(float(roc_auc_score(y_val, prob_val)), 4)}
        print(f"[e25s1] det·{name:<8} test AUROC {r['auroc_te']} / F1@.5 {f1_half:.4f} "
              f"/ F1@val_th({th:.2f}) {f1_val_best:.4f}", flush=True)
        return r

    def fam_arm(name, prob_te, prob_val, prob_unseen):
        pred = prob_te.argmax(1)
        acc = float((pred == fam_te).mean())
        bal = float(balanced_accuracy_score(fam_te, pred))
        rec = {SEEN_LABS[c]: round(float((pred[fam_te == c] == c).mean()), 3)
               for c in range(len(SEEN_LABS)) if (fam_te == c).any()}
        seen_p = prob_val.max(1)
        unseen_p = prob_unseen.max(1)
        lab = np.array([0] * len(seen_p) + [1] * len(unseen_p))
        scr = np.concatenate([seen_p, unseen_p])
        rej = {"auc_maxp": round(float(roc_auc_score(lab, scr)), 4),
               "auc_neg_maxp": round(float(roc_auc_score(lab, -scr)), 4)}
        r = {"acc": round(acc, 4), "balanced_acc": round(bal, 4),
             "per_family_recall": rec, "rejection": rej}
        print(f"[e25s1] fam·{name:<8} test acc {acc:.4f} / bal {bal:.4f} | 拒识 {rej}",
              flush=True)
        return r

    results = {"gate": gate, "selected_epoch": best_ep, "epoch_log": epoch_log,
               "thresholds": {}, "arms": {}}

    # E25 臂
    with torch.no_grad():
        def sc_(zarr):
            zt = torch.as_tensor(zarr, device=device)
            return (heads.wD(zt).squeeze(-1).cpu().numpy(),
                    heads.WF(zt).cpu().numpy())
        sD_val, aF_val = sc_(zv_e25)
        sD_te, aF_te = sc_(zt_e25)
        sD_un, aF_un = sc_(zu_e25)
    p_val_e25 = 1 / (1 + np.exp(-sD_val)); p_te_e25 = 1 / (1 + np.exp(-sD_te))

    def softmax_np(a):
        m = a.max(1, keepdims=True); e = np.exp(a - m)
        return e / e.sum(1, keepdims=True)

    results["arms"]["E25"] = {
        "det": det_arm("E25", None, p_val_e25, p_te_e25),
        "fam": fam_arm("E25", softmax_np(aF_te[ai_te]), softmax_np(aF_val[ai_val]),
                       softmax_np(aF_un))}

    # B0 / B1 / B3 / B2 臂（离线重拟合）
    z_tr_all = feats["z_train"]; rm_tr = feats["rm_train"]
    rF_tr = residualize(z_tr_all, st_e24)
    zv_b = feats["z_val"]; zt_b = feats["z_test_seen"]; zu_b = feats["z_unseen"]
    rv_b = residualize(zv_b, st_e24); rt_b = residualize(zt_b, st_e24)
    ru_b = residualize(zu_b, st_e24)
    rm_val_b = feats["rm_val"]; rm_te_b = feats["rm_test_seen"]; rm_un_b = feats["rm_unseen"]

    def offline_arm(name, Xtr, Xv, Xt, Xu, fam_head="both"):
        sc = StandardScaler().fit(Xtr)
        lr_d = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), y_tr)
        pv = lr_d.predict_proba(sc.transform(Xv))[:, 1]
        pt = lr_d.predict_proba(sc.transform(Xt))[:, 1]
        det = det_arm(name, None, pv, pt)
        run, arts = {}, {}
        for mode in (("logreg", "dischead") if fam_head == "both" else (fam_head,)):
            if mode == "logreg":
                scf = StandardScaler().fit(Xtr[ai_tr])
                lr_f = LogisticRegression(max_iter=3000, C=1.0).fit(
                    scf.transform(Xtr[ai_tr]), fam_tr)
                fv = lr_f.predict_proba(scf.transform(Xv[ai_val]))
                ft = lr_f.predict_proba(scf.transform(Xt[ai_te]))
                fu = lr_f.predict_proba(scf.transform(Xu))
            else:
                dh = DiscHead().fit(Xtr[ai_tr], fam_tr)
                dv = dh.transform(Xv[ai_val]); dtt = dh.transform(Xt[ai_te])
                du = dh.transform(Xu)
                c = np.asarray(dh.centers_)

                def sm(D):
                    d2 = ((D[:, None, :] - c[None]) ** 2).sum(-1)
                    e = np.exp(-(d2 - d2.min(1, keepdims=True)))
                    return e / e.sum(1, keepdims=True)
                fv, ft, fu = sm(dv), sm(dtt), sm(du)
            run[mode] = fam_arm(f"{name}·{mode}", ft, fv, fu)
            arts[mode] = {"fv": fv, "ft": ft, "fu": fu}
        return det, run, {"p_te": pt, "fam": arts}

    det_b0, fam_b0, art_b0 = offline_arm("B0", rm_tr, rm_val_b, rm_te_b, rm_un_b)
    results["arms"]["B0"] = {"det": det_b0, "fam": fam_b0}
    det_b1, fam_b1, art_b1 = offline_arm("B1", z_tr_all, zv_b, zt_b, zu_b)
    results["arms"]["B1"] = {"det": det_b1, "fam": fam_b1}
    det_b3, fam_b3, art_b3 = offline_arm("B3", rF_tr, rv_b, rt_b, ru_b)
    results["arms"]["B3"] = {"det": det_b3, "fam": fam_b3}
    sD_te_b2 = s_d_scalar(zt_b, st_e24, bD_e24)
    sD_v_b2 = s_d_scalar(zv_b, st_e24, bD_e24)
    p_v_b2 = 1 / (1 + np.exp(-np.clip(sD_v_b2, -50, 50)))
    p_t_b2 = 1 / (1 + np.exp(-np.clip(sD_te_b2, -50, 50)))
    results["arms"]["B2"] = {"det": det_arm("B2", None, p_v_b2, p_t_b2)}

    # ---- unseen 检测（vs val Human；含原始 split 分解；M0 也补） ----
    zh_val = ~y_val.astype(bool)
    def unseen_det(name, pv, pu):
        y = np.concatenate([np.zeros(int(zh_val.sum())), np.ones(len(pu))])
        s = np.concatenate([pv[zh_val], pu])
        out = {"overall_auroc": round(float(roc_auc_score(y, s)), 4)}
        splits = np.asarray(meta["unseen"]["split"])
        for sp in ("train", "val", "test"):
            msk = splits == sp
            if msk.any():
                y2 = np.concatenate([np.zeros(int(zh_val.sum())), np.ones(int(msk.sum()))])
                s2 = np.concatenate([pv[zh_val], pu[msk]])
                out[f"auroc_{sp}"] = round(float(roc_auc_score(y2, s2)), 4)
                out[f"n_{sp}"] = int(msk.sum())
        print(f"[e25s1] unseen det·{name}: {out}", flush=True)
        return out

    uns = {"E25": unseen_det("E25", p_val_e25, 1 / (1 + np.exp(-sD_un)))}
    sc0 = StandardScaler().fit(rm_tr)
    lr0 = LogisticRegression(max_iter=3000, C=1.0).fit(sc0.transform(rm_tr), y_tr)
    uns["B0"] = unseen_det("B0", lr0.predict_proba(sc0.transform(rm_val_b))[:, 1],
                           lr0.predict_proba(sc0.transform(rm_un_b))[:, 1])
    sc1 = StandardScaler().fit(z_tr_all)
    lr1 = LogisticRegression(max_iter=3000, C=1.0).fit(sc1.transform(z_tr_all), y_tr)
    uns["B1"] = unseen_det("B1", lr1.predict_proba(sc1.transform(zv_b))[:, 1],
                           lr1.predict_proba(sc1.transform(zu_b))[:, 1])
    results["unseen_detection"] = uns

    # ---- generator 级召回（test_seen；E25 与 B0/B1 的 DiscHead 版） ----
    gen_te = np.asarray(meta["test_seen"]["generator"])
    ai_idx_te = np.where(ai_te)[0]

    def gen_recall(pred_ai, name):
        out = {}
        for g in sorted(set(gen_te[ai_idx_te].tolist())):
            msk = gen_te[ai_idx_te] == g
            if msk.sum() < 8:
                continue
            out[g] = {"n": int(msk.sum()),
                      "acc": round(float((pred_ai[msk] == fam_te[msk]).mean()), 3)}
        print(f"[e25s1] gen recall·{name}: {len(out)} generators", flush=True)
        return out

    results["generator_recall_test"] = {
        "E25": gen_recall(aF_te[ai_te].argmax(1), "E25"),
        "B0": gen_recall(art_b0["fam"]["dischead"]["ft"].argmax(1), "B0"),
        "B1": gen_recall(art_b1["fam"]["dischead"]["ft"].argmax(1), "B1")}

    # ---- 语言 / 长度桶 ----
    ntok_te = np.asarray([int(x) for x in meta["test_seen"]["n_tokens"]])
    ntok_tr = np.asarray([int(x) for x in meta["train"]["n_tokens"]])
    qs = np.percentile(ntok_tr, [25, 50, 75])
    len_bucket = np.searchsorted(qs, ntok_te)
    lang_te = np.asarray(meta["test_seen"]["language"])

    def bucket_metrics(name, prob_te, fam_prob_te):
        out = {}
        combos = [("len", f"q{i}", len_bucket == i) for i in range(4)]
        combos += [("lang", lg, lang_te == lg)
                   for lg in sorted(set(lang_te.tolist()))]
        for tag, key, msk in combos:
            if msk.sum() < 30:
                continue
            entry = {"n": int(msk.sum())}
            yb = y_te[msk]
            if len(set(yb.tolist())) > 1:
                entry["det_auroc"] = round(float(roc_auc_score(yb, prob_te[msk])), 4)
            m_ai = msk[ai_idx_te]
            if m_ai.sum() >= 10:
                entry["fam_bal"] = round(float(balanced_accuracy_score(
                    fam_te[m_ai], fam_prob_te[m_ai].argmax(1))), 4)
            out[f"{tag}:{key}"] = entry
        return out

    results["buckets"] = {
        "E25": bucket_metrics("E25", p_te_e25, aF_te[ai_te]),
        "B0": bucket_metrics("B0", art_b0["p_te"],
                             art_b0["fam"]["dischead"]["ft"]),
        "B1": bucket_metrics("B1", art_b1["p_te"],
                             art_b1["fam"]["dischead"]["ft"])}

    results["timing_min"] = round((time.time() - t0) / 60, 1)
    results["params"] = {"setpool": int(sum(p.numel() for p in head.parameters())),
                         "disc_heads": int(sum(p.numel() for p in heads.parameters())),
                         "trainable_total": int(sum(p.numel() for p in head.parameters())
                                                + sum(p.numel() for p in heads.parameters()))}
    results["protocol"] = {"seed": 0, "epochs": 2, "batch": 64, "lr": 1e-3,
                           "lambda_shrink_E24": LAMBDA_SHRINK, "T_E": 1537,
                           "standardization": "训练折 z 的 (m, σ)，初始化头后一次固定",
                           "priors": "π 由训练折样本数估计；logπ_f 用条件 π_f|AI"}
    (OUT / "e25_final.json").write_text(json.dumps(results, ensure_ascii=False,
                                                   indent=2, default=str))
    print(f"[e25s1] 完成 {results['timing_min']} min → {OUT/'e25_final.json'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
