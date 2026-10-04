#!/usr/bin/env python
"""H2 pair round2：受控编码器探针（指导 §6 步骤 1）。

- 表征：冻结 CodeT5-base 均值池化 768d（截断 384+128，与 ACL 系列一致）；
- 任务：pair classification（source 隔离；同任务同家族跨 generator 正对 vs 同任务跨家族硬负对）；
- 两臂：① 线性头 [z_L;z_R]→2（train 标准化、类逆频率、dev 选 epoch、3 seeds）；
        ② 距离基线 cosine(z_L,z_R)（dev 选阈值，无训练）——与 round1 TF-IDF 同协议对照；
- 视图：raw/ids_only/strings_only/comments_only/all（同协议，用于判断是否超越词法捷径）；
- test 只评一次；bootstrap 按 task_id 聚类。
输出：artifacts/h2_pair_round2/{metrics.json,predictions.npz,config.json,env.json,split_manifest.json,hashes.json,features/*.npz,logs/run.log}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "h2_pair_benchmark_v1"
OUT = ROOT / "artifacts" / "h2_pair_round2"
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from h2_pair_similarity_baseline import choose_threshold, cluster_bootstrap, metrics  # noqa: E402

VIEWS = ("raw", "ids_only", "strings_only", "comments_only", "all")
SEEDS = (0, 1, 2)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def read_jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def encode_view(view: str, rows: list[dict], cache: Path, device="cuda") -> dict:
    """对 view 的所有唯一端点（sha→text）编码一次，缓存 npz。"""
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return dict(zip(z["shas"].tolist(), z["feats"]))
    from transformers import AutoTokenizer
    from encoders.codet5 import CodeT5Encoder
    tok = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    enc = CodeT5Encoder(path=str(ROOT / "checkpoints/codet5-base"), dtype="float32",
                        freeze=True, max_length=512).to(device).eval()
    mapping: dict[str, str] = {}
    for r in rows:
        for side in ("left", "right"):
            mapping.setdefault(r[f"{side}_view_code_sha256"][view], r[f"{side}_views"][view])
    items = sorted(mapping.items())
    shas = [k for k, _ in items]
    feats = []
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(items), 32):
            chunk = [t for _, t in items[i:i + 32]]
            ids_list = []
            for t in chunk:
                ids = tok(t, add_special_tokens=False)["input_ids"]
                if len(ids) > 512:
                    ids = ids[:384] + ids[-128:]
                ids_list.append(np.array(ids, dtype=np.int32))
            L = max(len(x) for x in ids_list)
            arr = np.full((len(ids_list), L), tok.pad_token_id, dtype=np.int64)
            for j, x in enumerate(ids_list):
                arr[j, :len(x)] = x
            ids = torch.tensor(arr, device=device)
            mask = torch.tensor((arr != tok.pad_token_id).astype(np.int64), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = enc(ids, attention_mask=mask)
            h = h.float()
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            feats.append(pooled.cpu().numpy().astype(np.float16))
            if (i // 32) % 40 == 0:
                print(f"[r2prb] {view}: {i+len(chunk)}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    del enc
    torch.cuda.empty_cache()
    feats = np.concatenate(feats)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, shas=np.array(shas, dtype=object), feats=feats)
    return dict(zip(shas, feats))


def pair_matrix(rows: list[dict], view: str, table: dict) -> tuple[np.ndarray, np.ndarray]:
    zl = np.stack([table[r["left_view_code_sha256"][view]] for r in rows]).astype(np.float32)
    zr = np.stack([table[r["right_view_code_sha256"][view]] for r in rows]).astype(np.float32)
    return zl, zr


def train_linear(zl_tr, zr_tr, y_tr, zl_dv, zr_dv, y_dv, zl_te, zr_te, seed, device="cuda",
                 epochs=40, patience=8, bs=256):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    joint = np.concatenate([zl_tr, zr_tr])  # train only
    mu, sd = joint.mean(0), joint.std(0) + 1e-6
    norm = lambda z: (z - mu) / sd
    Xtr = torch.tensor(np.concatenate([norm(zl_tr), norm(zr_tr)], 1), device=device)
    Xdv = torch.tensor(np.concatenate([norm(zl_dv), norm(zr_dv)], 1), device=device)
    Xte = torch.tensor(np.concatenate([norm(zl_te), norm(zr_te)], 1), device=device)
    ytr = torch.tensor(y_tr, device=device); ydv = y_dv
    head = torch.nn.Linear(Xtr.shape[1], 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    cnt = Counter(y_tr.tolist()); w = torch.tensor(
        [len(y_tr) / (2 * max(1, cnt[0])), len(y_tr) / (2 * max(1, cnt[1]))], device=device)
    best = {"ba": -1, "state": None, "epoch": 0}
    dev_curve = []
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        head.train()
        rng = np.random.RandomState(seed * 100 + ep)
        order = torch.tensor(rng.permutation(n), device=device)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(head(Xtr[idx]), ytr[idx], weight=w)
            loss.backward(); opt.step()
        head.eval()
        with torch.inference_mode():
            dv = torch.softmax(head(Xdv), 1)[:, 1]
            dvp = dv.cpu().numpy()
        import sklearn.metrics as sm
        th = choose_threshold(dvp, ydv)
        ba = float(sm.balanced_accuracy_score(ydv, dvp >= th))
        dev_curve.append({"epoch": ep, "dev_ba": ba, "dev_threshold": float(th)})
        if ba > best["ba"]:
            best = {"ba": ba, "epoch": ep, "threshold": float(th),
                    "state": {k: v.detach().clone() for k, v in head.state_dict().items()}}
        if ep - best["epoch"] >= patience:
            break
    head.load_state_dict(best["state"]); head.eval()
    with torch.inference_mode():
        te_p = torch.softmax(head(Xte), 1)[:, 1].cpu().numpy()
    del Xtr, Xdv, Xte, head
    torch.cuda.empty_cache()
    return te_p, best, dev_curve


def distance_scores(zl, zr):
    num = (zl * zr).sum(1)
    den = np.linalg.norm(zl, axis=1) * np.linalg.norm(zr, axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    outd = OUT / "smoke" if args.smoke else OUT
    if (outd / "metrics.json").exists():
        raise SystemExit(f"refuse to overwrite {outd}/metrics.json")
    (outd / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = read_jsonl(DATA / "pairs.jsonl")
    fam_rows = {r["pair_id"]: r for r in read_jsonl(DATA / "family_pair_balanced_pairs.jsonl")}
    if args.smoke:
        args.seeds = 1
        tiny = []
        for s in ("authorbench_dcan", "llm_codegen_v2"):
            for sp in ("train", "dev", "test"):
                tiny += [r for r in rows if r["source"] == s and r["task_split"] == sp][:80]
        rows = tiny
    result = {"config": {"script": "scripts/h2_pair_encoder_probe.py",
                         "encoder": "checkpoints/codet5-base 冻结（mean-pool, 截断 384+128）",
                         "head": "Linear(1536,2) AdamW lr=1e-3 wd=1e-4 bs256 类逆频率；train 拟合标准化；dev 选 epoch(BA)+阈值",
                         "distance": "cosine(z_L,z_R)；dev 选阈值（无训练）",
                         "protocol": "source 隔离；test 一次；bootstrap 按 task_id 聚类(300)",
                         "views": list(VIEWS), "seeds": args.seeds},
              "sources": {}, "tfidf_ref": {}}
    try:
        ref = json.loads((ROOT / "artifacts/h2_pair_round1/similarity_baseline.json").read_text())
        for s, views in ref["sources"].items():
            result["tfidf_ref"][s] = {v: {"test_ba": views[v]["test"]["balanced_accuracy"],
                                          "test_auc": views[v]["test"]["roc_auc"],
                                          "test_f1": views[v]["test"]["f1"]} for v in VIEWS}
    except Exception as e:  # noqa: BLE001
        print("warn: tfidf_ref unavailable", e)

    pack = {}
    for source in ("authorbench_dcan", "llm_codegen_v2"):
        srows = [r for r in rows if r["source"] == source]
        res_s = {"n": len(srows), "views": {}}
        for view in VIEWS:
            cache = outd / "features" / f"{source}_{view}.npz"
            table = encode_view(view, srows, cache)
            zl, zr = pair_matrix(srows, view, table)
            y = np.array([r["pair_label"] for r in srows], dtype=int)
            splits = np.array([r["task_split"] for r in srows])
            tasks = [r["task_id"] for r in srows]
            fam_idx = np.array([r["pair_id"] in fam_rows for r in srows])
            tr, dv, te = splits == "train", splits == "dev", splits == "test"
            vres = {"n": {"train": int(tr.sum()), "dev": int(dv.sum()), "test": int(te.sum())},
                    "distance": {}, "linear": {}, "family_pair_balanced_test": {}}
            # 距离基线（无训练，dev 阈值）
            sdv = distance_scores(zl[dv], zr[dv]); ste = distance_scores(zl[te], zr[te])
            th = choose_threshold(sdv, y[dv])
            vres["distance"]["dev_threshold"] = float(th)
            vres["distance"]["dev"] = metrics(sdv, y[dv], th)
            vres["distance"]["test"] = metrics(ste, y[te], th)
            vres["distance"]["test_task_cluster_bootstrap"] = cluster_bootstrap(
                [r for r in srows if r["task_split"] == "test"], ste, y[te], th)
            fam = np.array([r["pair_id"] in fam_rows for r in srows])
            if te.sum() and fam[te].sum():
                vres["family_pair_balanced_test"]["distance"] = metrics(ste[fam[te]], y[te][fam[te]], th)
            # 线性头 3 seeds
            seeds = list(range(args.seeds))
            per_seed = []
            for seed in seeds:
                tp, best, curve = train_linear(
                    zl[tr], zr[tr], y[tr], zl[dv], zr[dv], y[dv], zl[te], zr[te], seed)
                per_seed.append({"seed": seed, "best_epoch": best["epoch"], "best_dev_ba": best["ba"],
                                 "dev_threshold": best["threshold"], "dev_curve": curve,
                                 "test": metrics(tp, y[te], best["threshold"]),
                                 "family_pair_balanced_test": metrics(tp[fam[te]], y[te][fam[te]],
                                                                      best["threshold"]) if fam[te].sum() else None,
                                 "test_task_cluster_bootstrap": cluster_bootstrap(
                                     [r for r in srows if r["task_split"] == "test"], tp, y[te],
                                     best["threshold"])})
                pack[f"{source}_{view}_s{seed}_test_probs"] = tp.astype(np.float16)
                print(f"[r2prb] {source}/{view} s{seed}: best ep{best['epoch']} devBA {best['ba']:.4f} "
                      f"testBA {per_seed[-1]['test']['balanced_accuracy']:.4f}", flush=True)
            vres["linear"]["per_seed"] = per_seed
            vres["linear"]["mean_test_ba"] = float(np.mean([x["test"]["balanced_accuracy"] for x in per_seed]))
            aucs = [x["test"]["roc_auc"] for x in per_seed if x["test"]["roc_auc"] is not None]
            vres["linear"]["mean_test_auc"] = float(np.mean(aucs)) if aucs else None
            vres["linear"]["mean_test_f1"] = float(np.mean([x["test"]["f1"] for x in per_seed]))
            if te.sum() and fam[te].sum():
                vres["family_pair_balanced_test"]["linear_mean_test_ba"] = float(
                    np.mean([p["family_pair_balanced_test"]["balanced_accuracy"] for p in per_seed
                             if p["family_pair_balanced_test"]]))
            res_s["views"][view] = vres
        result["sources"][source] = res_s
        print(f"[r2prb] {source} done ({time.time()-t0:.0f}s)", flush=True)
    (outd / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    # 预测身份（按 source 分开）
    for source in ("authorbench_dcan", "llm_codegen_v2"):
        te_rows = [r for r in rows if r["source"] == source and r["task_split"] == "test"]
        pack[f"{source}_test_y"] = np.array([r["pair_label"] for r in te_rows], dtype=np.int8)
        pack[f"{source}_test_task"] = np.array([r["task_id"] for r in te_rows], dtype=object)
        pack[f"{source}_test_pair"] = np.array([r["pair_id"] for r in te_rows], dtype=object)
    np.savez_compressed(outd / "predictions.npz", **pack)
    import sklearn, transformers
    (outd / "env.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "sklearn": sklearn.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "env_vars": {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}}, ensure_ascii=False, indent=2))
    (outd / "config.json").write_text(json.dumps(result["config"], ensure_ascii=False, indent=1))
    sm2 = json.loads((ROOT / "artifacts" / "acl_dcan_round2" / "split_manifest.json").read_text())
    (outd / "split_manifest.json").write_text(json.dumps({
        "note": "使用 h2_pair_benchmark_v1 自带 task 级 split（train/dev/test 按 task_id 不相交）；source 隔离",
        "counts": {s: {sp: sum(1 for r in rows if r["source"] == s and r["task_split"] == sp)
                       for sp in ("train", "dev", "test")} for s in ("authorbench_dcan", "llm_codegen_v2")},
        "ref": "d-det/data/h2_pair_benchmark_v1/summary.json"}, ensure_ascii=False, indent=2))
    files = ["data/h2_pair_benchmark_v1/pairs.jsonl", "scripts/h2_pair_encoder_probe.py",
             "scripts/h2_pair_similarity_baseline.py",
             str((outd / "metrics.json").relative_to(ROOT)), str((outd / "predictions.npz").relative_to(ROOT))]
    (outd / "hashes.json").write_text(json.dumps({"files": {f: {"sha256": sha(ROOT / f),
                                                                  "size": (ROOT / f).stat().st_size}
                                                              for f in files}}, ensure_ascii=False, indent=1))
    print(f"[r2prb] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
