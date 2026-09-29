#!/usr/bin/env python
"""E24 诊断：首跑负结果定位——同一特征上"能量分类器 vs 判别式分类器"对照。

问题：旗舰轮首跑中家族归因（残差能量 .235 / raw-z 能量 .233）远低于 M0 判别探针
（raw-mean LDA .416）。本诊断在**完全相同的 z / r_F 特征**上换成闭式判别器（DiscHead），
回答：瓶颈在"特征/表示"还是在"能量决策规则"。

运行：OMP_NUM_THREADS=8 python scripts/flagship_r1_diag.py
输出：runs/flagship_r1/diag.json
"""
from __future__ import annotations

import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from models.disc import DiscHead  # noqa: E402
from flagship_round1 import (SetPool, load_corpus, z_pass, residualize,  # noqa: E402
                             SEEN_LABS, OUT)

from sklearn.metrics import balanced_accuracy_score  # noqa: E402


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = Namespace(smoke=False, max_len=2048, train_human=8000, train_fam=1200,
                     val_human=1500, val_fam=500, test_human=500, test_fam=200,
                     unseen_cap=600, enc_bs=32)
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
    head.load_state_dict(torch.load(OUT / "head.pt", map_location=device)["head"])
    head.eval()
    st = dict(np.load(OUT / "stats.npz", allow_pickle=True))

    corpus = load_corpus(args)["docs"]
    t0 = time.time()
    ai_tr = [d for d in corpus["train"] if d.y_ai == 1]
    z_ai, rm_ai = z_pass(enc, head, ai_tr, args.enc_bs, device, True)
    z_val, rm_val = z_pass(enc, head, corpus["val"], args.enc_bs, device, True)
    print(f"[diag] 编码完成 {time.time()-t0:.0f}s：训练 AI {len(ai_tr)}，val {len(corpus['val'])}",
          flush=True)

    rF_ai = residualize(z_ai, st)
    rF_val = residualize(z_val, st)
    fam = np.array([SEEN_LABS.index(d.fam) for d in ai_tr])
    vsel = np.array([d.y_ai == 1 and d.fam in SEEN_LABS for d in corpus["val"]])
    vib = np.where(vsel)[0]
    fam_val = np.array([SEEN_LABS.index(d.fam) for d, s in zip(corpus["val"], vsel) if s])

    def ev(tag, Xtr, Xva):
        h = DiscHead().fit(Xtr, fam)
        p = h.predict(Xva)
        acc = float((p == fam_val).mean())
        bal = float(balanced_accuracy_score(fam_val, p))
        print(f"[diag] DiscHead·{tag:<10}: acc {acc:.4f} / bal {bal:.4f}", flush=True)
        return {"acc": round(acc, 4), "balanced_acc": round(bal, 4)}

    report = {}
    report["disc_z"] = ev("z(1537)", z_ai, z_val[vib])
    report["disc_rF"] = ev("rF(1537)", rF_ai, rF_val[vib])
    report["disc_mu"] = ev("mu(768)", z_ai[:, :768], z_val[vib][:, :768])
    report["disc_rawmean"] = ev("raw-mean", rm_ai, rm_val[vib])
    report["ref_energy_rF"] = {"acc": 0.2353, "balanced_acc": 0.2456}   # 首跑 results.json
    report["ref_m0_dischead"] = {"acc": 0.4158, "balanced_acc": 0.4225}
    report["n_train_ai"] = len(ai_tr)
    (OUT / "diag.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[diag] 写出 {OUT/'diag.json'}；耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
