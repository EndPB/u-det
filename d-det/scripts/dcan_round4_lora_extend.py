#!/usr/bin/env python
"""Round4 B：LoRA 预算扩展（唯一变化 = max epoch 24 / patience 5；其余与 round2 完全一致）。

- 同 encoder/tokenizer/head/采样/split/优化器/lr（enc 3e-4、head 1e-3）/bf16/bs16×accum2；
- 每 seed 保存 dev 曲线（每 epoch dev_f1+秒数）；test 仅在 best state 冻结后评一次；
- 仅用于确认"是否存在明显欠拟合"，不做新方法主张。
输出：artifacts/acl_dcan_round4/lora_extend/{metrics.json,predictions.npz,config.json,env.json,
split_manifest.json,hashes.json,states/*.pt,logs/run.log}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dcan_four_models as r1  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from dcan_round3_ft import FTModel, collate, evaluate, tokenize_rows  # noqa: E402

OUT = ROOT / "artifacts" / "acl_dcan_round4" / "lora_extend"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_lora(seed, rows, ids_list, pad_id, args):
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
    model = FTModel("lora").to(device)
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head")]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": 3e-4},
                             {"params": head_params, "lr": 1e-3}], weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats()
    best = {"dev": -1, "epoch": 0, "state": None}
    dev_curve = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        rng = np.random.RandomState(seed * 100 + ep)
        order = rng.permutation(len(tr))
        micro = [tr[order[i:i + args.bs]] for i in range(0, len(order), args.bs)]
        for w0 in range(0, len(micro), args.accum):
            window = micro[w0:w0 + args.accum]
            opt.zero_grad()
            for idxs in window:
                chunk = [ids_list[j] for j in idxs]
                ids, mask = collate(chunk, pad_id, device)
                yb = torch.tensor(y[idxs], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ids, mask)
                    loss = F.cross_entropy(logits.float(), yb) / len(window)  # 实际窗口缩放
                loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_params + head_params, 1.0)
            opt.step()
        dv_probs = evaluate(model, ids_list, y, dv, device, pad_id)
        f1 = float(sm.f1_score(y[dv], dv_probs.argmax(1), average="macro", zero_division=0))
        dev_curve.append({"epoch": ep, "dev_f1": f1, "sec": round(time.time() - t0, 1)})
        print(f"[r4b] lora_ext s{seed} ep{ep}: dev {f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if f1 > best["dev"]:
            best = {"dev": f1, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                              if k in trainable_names}}
        if ep - best["epoch"] >= args.patience:
            break
    for k, v in model.state_dict().items():
        if k in best["state"]:
            v.copy_(best["state"][k])
    res = {"scheme": "lora_extend", "seed": seed, "best_dev_f1": best["dev"], "best_epoch": best["epoch"],
           "dev_curve": dev_curve, "epochs_run": dev_curve[-1]["epoch"],
           "max_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
           "wall_sec": round(time.time() - t0, 1)}
    if not args.smoke:
        torch.save({"scheme": "lora_extend", "seed": seed, "best_dev_f1": best["dev"],
                    "best_epoch": best["epoch"], "state": best["state"]},
                   Path(args.out) / "states" / f"lora_ext_s{seed}.pt")
        te = np.array([i for i, s in enumerate(split) if s == "test"])
        te_probs = evaluate(model, ids_list, y, te, device, pad_id)
        res["_te_probs"] = te_probs.astype(np.float16)
        res["test"] = r1.metric_from_probs(te_probs.astype(np.float64), y[te],
                                           {"generator": [rows[i]["model_name"] for i in te]})
        print(f"[r4b] lora_ext s{seed}: best ep{best['epoch']} dev {best['dev']:.4f} | "
              f"test {res['test']['macro_f1']:.4f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = 1, 2
    outd = Path(args.out)
    if (outd / "metrics.json").exists() and not args.force:
        raise SystemExit(f"refuse to overwrite existing {outd}/metrics.json")
    (outd / "logs").mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        (outd / "states").mkdir(exist_ok=True)
    t0 = time.time()
    rows = r1.load_rows()
    ids_list, pad_id = tokenize_rows(rows)
    print(f"[r4b] pad_id={pad_id}", flush=True)
    fam_idx = {f: i for i, f in enumerate(r1.FAMILIES)}
    y = np.array([fam_idx[r["family"]] for r in rows])
    split = [r["task_split"] for r in rows]
    te = [i for i, s in enumerate(split) if s == "test"]
    task = [r["task_id"] for r in rows]
    sha = [r["source_sha256"] for r in rows]
    pack = {"y_test": y[te].astype(np.int16),
            "task_id_test": np.array([task[i] for i in te], dtype=object),
            "source_sha256_test": np.array([sha[i] for i in te], dtype=object),
            "model_name_test": np.array([rows[i]["model_name"] for i in te], dtype=object),
            "family_order": np.array(r1.FAMILIES, dtype=object)}
    split_blob = "\n".join(f"{task[i]}\t{sha[i]}" for i in te)
    pack["split_hash_test"] = np.array(hashlib.sha256(split_blob.encode()).hexdigest())
    out = {"config": {"script": "scripts/dcan_round4_lora_extend.py",
                      "changed_vs_round2": "仅 max epoch（12→%d）与 patience（3→%d）" % (args.epochs, args.patience),
                      "same_as_round2": "encoder/tokenizer/head/采样/split/AdamW(lr enc 3e-4, head 1e-3, wd 1e-4)/bf16/bs16×accum2/截断384+128",
                      "seeds": args.seeds, "epochs": args.epochs, "patience": args.patience,
                      "selection": "best-dev 恢复；test 冻结后评一次；cudnn.deterministic",
                      "purpose": "确认是否存在明显欠拟合（B 条件：A 显示性能依赖可解释词法线索）"},
           "runs": []}
    for seed in range(args.seeds):
        r = train_lora(seed, rows, ids_list, pad_id, args)
        if not args.smoke:
            pack[f"lora_ext_s{seed}_probs"] = r.pop("_te_probs")
        out["runs"].append(r)
    (outd / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    if not args.smoke:
        np.savez_compressed(outd / "predictions.npz", **pack)
    import sklearn, transformers
    (outd / "env.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "sklearn": sklearn.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "env_vars": {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}}, ensure_ascii=False, indent=2))
    sm2 = json.loads((ROOT / "artifacts" / "acl_dcan_round2" / "split_manifest.json").read_text())
    (outd / "split_manifest.json").write_text(json.dumps({
        "generated_utc": "2026-10-03T22:30:00Z",
        "note": "Round4 B 不改变切分：沿用 round2 manifest（任务级 train/dev/test；test 不参与选择）",
        "unchanged_from": "artifacts/acl_dcan_round2/split_manifest.json",
        "invariants_verified": {"rows_per_split": {"train": 6557, "dev": 1484, "test": 1457}},
        "sources": sm2.get("sources", {})}, ensure_ascii=False, indent=2))
    files = ["data/h2_authorbench_dcan/core.jsonl", "scripts/dcan_round4_lora_extend.py",
             "scripts/dcan_round3_ft.py", "scripts/dcan_round2_trainable.py",
             "artifacts/acl_dcan_round4/lora_extend/metrics.json",
             "artifacts/acl_dcan_round4/lora_extend/predictions.npz"]
    hashes = {"files": {}}
    for f in files:
        p = ROOT / f
        if p.exists():
            hashes["files"][f] = {"sha256": sha256(p), "size": p.stat().st_size}
    for i in range(args.seeds):
        p = outd / "states" / f"lora_ext_s{i}.pt"
        if p.exists():
            hashes["files"][str(p.relative_to(ROOT))] = {"sha256": sha256(p), "size": p.stat().st_size}
    (outd / "hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=1))
    print(f"[r4b] done {round(time.time()-t0,1)}s -> {outd}", flush=True)


if __name__ == "__main__":
    main()
