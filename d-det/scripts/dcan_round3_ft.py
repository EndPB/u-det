#!/usr/bin/env python
"""Round3 修正版微调侧（head-only 匹配对照 + LoRA checkpoint 复算校验）。

- head-only：同 encoder/同 tokenize(512=384+128)/同线性头/同 CE/同采样/同 head-lr(1e-3)/
  同 epoch 与 early-stop（≤12ep，patience3）/3 seeds；只关 adapter。
  （修复：pad_id 由 tokenizer 提供；mask 按实际长度；梯度累积按实际窗口大小缩放；
   eval 前 model.eval()；dev 曲线保存；预测带 identity；独立 --out 与防覆盖；smoke 仅评 dev。）
- --verify-ckpt：加载 best_lora.pt 复算 test 概率并与 round2 保存预测比对；记录基座 sha256 等。
输出：artifacts/acl_dcan_round3_audit/{head_only/, checkpoint_recompute.json, logs/}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import dcan_four_models as r1  # noqa: E402
from dcan_round2_trainable import FTModel, tokenize_rows  # noqa: E402

OUT = ROOT / "artifacts" / "acl_dcan_round3_audit"


def collate(batch_ids, pad_id, device):
    L = max(len(x) for x in batch_ids)
    ids = np.full((len(batch_ids), L), pad_id, dtype=np.int64)
    mask = np.zeros((len(batch_ids), L), dtype=np.int64)
    for i, x in enumerate(batch_ids):
        ids[i, :len(x)] = x
        mask[i, :len(x)] = 1
    return torch.tensor(ids, device=device), torch.tensor(mask, device=device)


class HeadOnlyFast(FTModel):
    """head-only：冻结 encoder 前向置于 no_grad（只需 head 梯度）——与逐层冻结训练数学等价。"""

    def __init__(self):
        super().__init__("head_only")

    def forward(self, ids, mask):
        with torch.no_grad():
            h = self.enc(ids, attention_mask=mask)
        h = h.float()
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled)


def evaluate(model, ids_list, y, idx, device, pad_id, bs=32):
    model.eval()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(idx), bs):
            chunk = [ids_list[j] for j in idx[i:i + bs]]
            ids, mask = collate(chunk, pad_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask)
            probs.append(F.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(probs)


def run_head_only(seed, rows, ids_list, pad_id, args):
    import sklearn.metrics as sm
    device = "cuda"
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    fam_idx = {f: i for i, f in enumerate(r1.FAMILIES)}
    y = np.array([fam_idx[r["family"]] for r in rows])
    split = [r["task_split"] for r in rows]
    tr = np.array([i for i, s in enumerate(split) if s == "train"])
    dv = np.array([i for i, s in enumerate(split) if s == "dev"])
    te = np.array([i for i, s in enumerate(split) if s == "test"])
    model = HeadOnlyFast().to(device)
    head = [p for p in model.head.parameters()]
    opt = torch.optim.AdamW([{"params": head, "lr": 1e-3}], weight_decay=1e-4)
    best = {"dev": -1, "epoch": 0, "state": None}
    dev_curve = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        rng = np.random.RandomState(seed * 100 + ep)
        order = rng.permutation(len(tr))
        micro = [(order[i:i + args.bs]) for i in range(0, len(order), args.bs)]
        for w0 in range(0, len(micro), args.accum):
            window = micro[w0:w0 + args.accum]
            opt.zero_grad()
            for idxs in window:
                chunk = [ids_list[j] for j in tr[idxs]]
                ids, mask = collate(chunk, pad_id, device)
                yb = torch.tensor(y[tr[idxs]], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ids, mask)
                    loss = F.cross_entropy(logits.float(), yb) / len(window)  # 实际窗口缩放
                loss.backward()
            torch.nn.utils.clip_grad_norm_(head, 1.0)
            opt.step()
        dv_probs = evaluate(model, ids_list, y, dv, device, pad_id)
        f1 = float(sm.f1_score(y[dv], dv_probs.argmax(1), average="macro", zero_division=0))
        dev_curve.append({"epoch": ep, "dev_f1": f1, "sec": round(time.time() - t0, 1)})
        print(f"[r3ft] head_only s{seed} ep{ep}: dev {f1:.4f}", flush=True)
        if f1 > best["dev"]:
            best = {"dev": f1, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                              if k in [n for n, _ in model.named_parameters() if _.requires_grad]}}
        if ep - best["epoch"] >= 3:
            break
    for k, v in model.state_dict().items():
        if k in best["state"]:
            v.copy_(best["state"][k])
    res = {"scheme": "head_only", "seed": seed, "best_dev_f1": best["dev"], "best_epoch": best["epoch"],
           "dev_curve": dev_curve, "head_weights": {k: v.numpy() for k, v in best["state"].items()}}
    if not args.smoke:
        te_probs = evaluate(model, ids_list, y, te, device, pad_id)
        res["_te_probs"] = te_probs.astype(np.float16)
        res["test"] = r1.metric_from_probs(te_probs.astype(np.float64), y[te],
                                           {"generator": [rows[i]["model_name"] for i in te]})
        res["head_lr_note"] = "与 LoRA 运行一致的 head lr=1e-3；encoder 全冻结"
    return res


def verify_ckpt(args):
    ck = torch.load(args.verify_ckpt, map_location="cpu", weights_only=True)
    assert ck["scheme"] == "lora", ck["scheme"]
    rows = r1.load_rows()
    fam_idx = {f: i for i, f in enumerate(r1.FAMILIES)}
    y = np.array([fam_idx[r["family"]] for r in rows])
    split = [r["task_split"] for r in rows]
    te = np.array([i for i, s in enumerate(split) if s == "test"])
    ids_list, pad_id = tokenize_rows(rows)
    device = "cuda"
    model = FTModel("lora").to(device)
    sd = model.state_dict()
    for k, v in ck["state"].items():
        sd[k].copy_(v)
    probs = evaluate(model, ids_list, y, te, device, pad_id)
    d2 = np.load(ROOT / args.verify_round2_preds, allow_pickle=True)
    key = f"lora_s{ck['seed']}_probs"
    p2 = d2[key].astype(np.float32)
    base = ROOT / "checkpoints" / "codet5-base" / "pytorch_model.bin"
    rep = {"checkpoint": args.verify_ckpt, "scheme": ck["scheme"], "seed": ck["seed"],
           "best_dev_f1_saved": ck.get("best_dev_f1", "not-saved-in-ckpt(round2)"), "recomputed_test_macro_f1":
               r1.metric_from_probs(probs.astype(np.float64), y[te], None)["macro_f1"],
           "round2_saved_key": key,
           "max_abs_diff_vs_round2": float(np.abs(probs.astype(np.float32) - p2).max()),
           "exact_equal": bool((probs.astype(np.float16) == d2[key]).all()),
           "base_model_file": str(base), "base_model_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
           "lora_config": {"r": 8, "alpha": 16, "dropout": 0.1, "targets": ["q", "v"],
                           "task_type": "FEATURE_EXTRACTION"},
           "tokenizer": "checkpoints/codet5-base", "max_length": 512,
           "truncation": ">512 -> head384+tail128", "family_order": r1.FAMILIES,
           "env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}}
    (OUT / "checkpoint_recompute.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print("[r3ft] verify:", json.dumps({k: rep[k] for k in
          ("seed", "max_abs_diff_vs_round2", "exact_equal", "recomputed_test_macro_f1")}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="head_only", choices=["head_only"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--out", default=str(OUT / "head_only"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-ckpt", default=None)
    ap.add_argument("--verify-round2-preds",
                    default=str(ROOT / "artifacts" / "acl_dcan_round2" / "trainable" / "predictions.npz"))
    args = ap.parse_args()
    if args.verify_ckpt:
        verify_ckpt(args)
        return
    if args.smoke:
        args.seeds, args.epochs = 1, 3
    outd = Path(args.out)
    if (outd / "metrics.json").exists() and not args.force:
        raise SystemExit(f"refuse to overwrite existing {outd}/metrics.json")
    outd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = r1.load_rows()
    ids_list, pad_id = tokenize_rows(rows)
    print(f"[r3ft] pad_id={pad_id}", flush=True)
    out = {"config": {"script": "scripts/dcan_round3_ft.py", "scheme": "head_only",
                      "seeds": args.seeds, "epochs": args.epochs, "bs": args.bs, "accum": args.accum,
                      "encoder_frozen": True, "head": "Linear(768,6) lr=1e-3 wd=1e-4 CE",
                      "smoke_only_dev": args.smoke}, "runs": []}
    split = [r["task_split"] for r in rows]
    sha = [r["source_sha256"] for r in rows]
    task = [r["task_id"] for r in rows]
    te = [i for i, s in enumerate(split) if s == "test"]
    pack = {"y_test": np.array([r1.FAMILIES.index(rows[i]["family"]) for i in te], dtype=np.int16),
            "task_id_test": np.array([task[i] for i in te], dtype=object),
            "source_sha256_test": np.array([sha[i] for i in te], dtype=object),
            "model_name_test": np.array([rows[i]["model_name"] for i in te], dtype=object),
            "family_order": np.array(r1.FAMILIES, dtype=object)}
    for seed in range(args.seeds):
        r = run_head_only(seed, rows, ids_list, pad_id, args)
        if not args.smoke:
            pack[f"head_only_s{seed}_probs"] = r.pop("_te_probs")
        r.pop("head_weights", None)
        out["runs"].append(r)
        print(f"[r3ft] s{seed}: {json.dumps({k: v for k, v in r.items() if k in ('best_dev_f1','best_epoch')})}",
              flush=True)
    (outd / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    np.savez_compressed(outd / "predictions.npz", **pack) if not args.smoke else None
    print(f"[r3ft] done {round(time.time()-t0,1)}s -> {outd}")


if __name__ == "__main__":
    main()
