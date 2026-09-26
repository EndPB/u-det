#!/usr/bin/env python
"""v0.5 判别头训练/验证（冻结特征版）：3 族（Qwen0.5B/1.5B/DS1.3B）

- 数据：394 同题配对的 Δ（qwen15/ds13 复用 kernel_e11 缓存；0.5B 现场提取 CodeT5 v1.0 特征）
- 训练：models.disc.DiscHead（LDA/shrinkage 闭式）→ 保存 runs/v0.5_disc/head3.pkl
- 验证：按题 GroupKFold（3 族 + 2 族复现对照）+ in-sample 检查
输出：runs/v0.5_disc/{head3.pkl, eval.json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from models.disc import DiscHead  # noqa: E402
from kernel_e2_loss import encode_codes  # noqa: E402

OUT = ROOT / "runs/v0.5_disc"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 三族 Δ ----
    d11 = np.load(ROOT / "runs/kernel_e11/feat_cache.npz", allow_pickle=True)
    dq, dd = d11["dq"], d11["dd"]
    tasks = [str(t) for t in d11["tasks"]]  # 394 题（qwen15/ds13 对齐的公共题）
    cache05 = OUT / "d05.npz"
    if cache05.exists():
        d05 = np.load(cache05)["d05"]
    else:
        t = pq.read_table(ROOT / "data/processed/pairs.parquet")
        tm = dict(zip(t.column("task_id").to_pylist(),
                      zip(t.column("x_plus").to_pylist(), t.column("x_minus").to_pylist())))
        missing = [k for k in tasks if k not in tm]
        assert not missing, f"pairs.parquet 缺 {len(missing)} 题"
        with open(ROOT / "configs/ddet_v041.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        enc_cfg = dict(cfg["encoder"])
        enc_cfg["path"] = str(ROOT / enc_cfg["path"])
        enc = build_encoder(**enc_cfg)
        dual = build_model("dual", encoder=enc, dim=enc.hidden_size,
                           pool=cfg["model"].get("pool", "mean"),
                           s2_rank=cfg["model"].get("s2_rank", 1))
        ck = torch.load(str(ROOT / "runs/v0.4.1_covreg/last.pt"), map_location="cpu",
                        weights_only=False)
        dual.load_state_dict(ck["state"], strict=False)
        dual = dual.to(device).eval()
        tok = AutoTokenizer.from_pretrained(enc_cfg["path"])
        hp = encode_codes(dual, [tm[k][0] for k in tasks], tok, device).numpy()
        hm = encode_codes(dual, [tm[k][1] for k in tasks], tok, device).numpy()
        d05 = hp - hm
        np.savez_compressed(cache05, d05=d05)
        print("[v0.5] 0.5B Δ 提取完成", flush=True)

    n = len(tasks)
    groups = np.arange(n)              # 同题同组
    X3 = np.vstack([d05, dq, dd])
    y3 = np.array(["qwen05"] * n + ["qwen15"] * n + ["ds13"] * n)
    g3 = np.concatenate([groups, groups, groups])

    # ---- 训练 3 族头 ----
    head = DiscHead().fit(X3, y3)
    head.save(OUT / "head3.pkl")
    z_in = head.transform(X3)
    d_in = ((z_in[:, None, :] - head.centers_[None, :, :]) ** 2).sum(-1)
    res = {
        "n_pairs": n, "families": head.families_,
        "in_sample_acc": float((head.classes_[d_in.argmin(1)] == y3).mean()),
        "proj_dim": int(z_in.shape[1]),
    }
    print(f"[v0.5] 3 族 in-sample acc = {res['in_sample_acc']:.4f}（{res['proj_dim']} 维投影）", flush=True)

    # ---- 验证 ----
    res["gkf_3fam"] = {k: round(v, 4) for k, v in DiscHead().cross_val(X3, y3, g3).items()
                       if k in ("acc", "balanced_acc")}
    X2 = np.vstack([dq, dd])
    y2 = np.array(["qwen15"] * n + ["ds13"] * n)
    res["gkf_2fam_repro"] = {k: round(v, 4) for k, v in DiscHead().cross_val(X2, y2, np.concatenate([groups, groups])).items()
                             if k in ("acc", "balanced_acc")}
    print(f"[v0.5] 3 族 GroupKFold = {res['gkf_3fam']} | 2 族复现 = {res['gkf_2fam_repro']}", flush=True)

    (OUT / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("[v0.5] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
