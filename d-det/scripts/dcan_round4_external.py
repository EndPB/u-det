#!/usr/bin/env python
"""Round4 C：外部泛化审计（诊断性证据；独立于 408-task 主表）。

- STACAD：fold0 一次 file-held-out（文件级 train/子集采样；dev 按文件留出）
- Droid：fold_0 一次 generator-held-out（复用 fold_plan；dev=训练生成器的 dev 行）
- 三族模型同 test：TF-IDF（char_wb 5ep）/ head-only（冻结 CodeT5 768d 线性头）/ LoRA（round2 同超参）
- 3 seeds；dev 选择 best epoch；test 只在冻结后评一次；bootstrap 按文件/生成器聚类。
- 预算说明：neural 训练用 file-level 子集（默认 30k 行）控制成本；test 不变。诊断用途，不与主表混合。

输出：artifacts/acl_dcan_round4/external/{stacad_fold0,droid_fold0}/...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import dcan_four_models as r1  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from dcan_round2_trainable import FTModel  # noqa: E402  (FTModel 与 round3 相同)
from dcan_round3_ft import collate, evaluate  # noqa: E402

OUT = ROOT / "artifacts" / "acl_dcan_round4" / "external"
DATA = ROOT / "data"
TEXT_CAP = 6000
STACAD_MODELS = ["gemini_3_flash", "gpt5_nano", "claude_3_haiku", "qwen3_coder_30b",
                 "deepseek_r1", "gpt4o", "claude_37_sonnet"]
B = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_stacad():
    cdir = DATA / "stacad_v2" / "extracted" / "STACAD-v2" / "data" / "corpus_v2"
    folds = np.load(cdir / "folds.npy")
    langs = ["py", "java", "c", "cpp", "php", "go", "cs"]
    rows, gidx = [], 0
    for lg in langs:
        with (cdir / f"{lg}.jsonl").open() as fh:
            for line in fh:
                d = json.loads(line)
                rows.append({"text": (d.get("llm_src") or "")[:TEXT_CAP], "label": int(d["label"]) - 1,
                             "file": d["file_name"], "lang": lg, "fold": int(folds[gidx])})
                gidx += 1
    return rows


def load_droid(fold_plan):
    rows = []
    with (DATA / "h2_droid_full_selected" / "core.jsonl").open() as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("Label") != "MACHINE_GENERATED":
                continue
            rows.append({"text": (d.get("Code") or "")[:TEXT_CAP], "family": d["Model_Family"],
                         "split": d["split_source"], "generator": d["Generator"], "lang": d["Language"]})
    spec = fold_plan["folds"]["fold_0"]
    tg = {g for v in spec.values() for g in v["train_generators"]}
    hg = {g for v in spec.values() for g in v["heldout_generators"]}
    classes = sorted(spec.keys())
    cls = {c: i for i, c in enumerate(classes)}
    tr = [r for r in rows if r["split"] == "train" and r["generator"] in tg]
    dv = [r for r in rows if r["split"] == "dev" and r["generator"] in tg]
    te = [r for r in rows if r["split"] == "test" and r["generator"] in hg]
    for r in tr + dv + te:
        r["label"] = cls[r["family"]]
    return tr, dv, te, classes


def split_stacad(rows, sub_cap, dev_rows):
    te = [r for r in rows if r["fold"] == 0]
    pool = [r for r in rows if r["fold"] != 0]
    files = sorted({r["file"] for r in pool})
    rng = np.random.RandomState(0)
    rng.shuffle(files)
    by_file = {}
    for r in pool:
        by_file.setdefault(r["file"], []).append(r)
    sub_files, dev_files, n_sub, n_dev = [], [], 0, 0
    for f in files:
        if n_sub < sub_cap:
            sub_files.append(f); n_sub += len(by_file[f])
        elif n_dev < dev_rows:
            dev_files.append(f); n_dev += len(by_file[f])
        else:
            break
    assert not (set(sub_files) & set(dev_files))
    tr = [r for f in sub_files for r in by_file[f]]
    dv = [r for f in dev_files for r in by_file[f]]
    return tr, dv, te, sorted({r["label"] for r in rows})


def metrics(probs, y, extra_groups=None):
    m = r1.metric_from_probs(probs.astype(np.float64), y, extra_groups)
    return m


def tfidf_run(tr, dv, te, K, seeds):
    from sklearn.linear_model import SGDClassifier
    from p0_tfidf_baseline import fit_vectorizer, sample_stride
    import sklearn.metrics as sm
    vec = fit_vectorizer(sample_stride([r["text"] for r in tr], 100000), 300000)
    Xtr, Xdv, Xte = vec.transform([r["text"] for r in tr]), vec.transform([r["text"] for r in dv]), vec.transform([r["text"] for r in te])
    ytr = np.array([r["label"] for r in tr]); ydv = np.array([r["label"] for r in dv]); yte = np.array([r["label"] for r in te])
    counts = Counter(ytr); N = len(tr)
    w = np.array([N / (K * max(1, counts[c])) for c in ytr], dtype=np.float64)
    res = []
    for seed in seeds:
        clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=1, tol=None,
                            learning_rate="optimal", average=False, random_state=seed)
        best = {"f1": -1, "coef": None, "intercept": None, "epoch": None}
        curve = []
        for ep in range(1, 6):
            rng = np.random.RandomState(1000 * seed + 100 + ep)
            order = rng.permutation(N)
            for i in range(0, N, 10000):
                idx = order[i:i + 10000]
                clf.partial_fit(Xtr[idx], ytr[idx], classes=np.arange(K), sample_weight=w[idx])
            f = float(sm.f1_score(ydv, clf.predict(Xdv), average="macro", zero_division=0))
            curve.append({"epoch": ep, "dev_macro_f1": f})
            if f > best["f1"]:
                best = {"f1": f, "coef": clf.coef_.copy(), "intercept": clf.intercept_.copy(), "epoch": ep}
        clf.coef_, clf.intercept_ = best["coef"], best["intercept"]
        probs = clf.predict_proba(Xte)
        res.append({"seed": seed, "dev_curve": curve, "best_epoch": best["epoch"], "best_dev_f1": best["f1"],
                    "_probs": probs.astype(np.float16), "test": metrics(probs, yte)})
        print(f"[r4c] tfidf s{seed}: best ep{best['epoch']} dev {best['f1']:.4f} test {res[-1]['test']['macro_f1']:.4f}", flush=True)
    return res


def encode_features(texts):
    from transformers import AutoTokenizer
    from encoders.codet5 import CodeT5Encoder
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    enc = CodeT5Encoder(path=str(ROOT / "checkpoints/codet5-base"), dtype="float32",
                        freeze=True, max_length=512).to(device).eval()
    feats = []
    with torch.inference_mode():
        for i in range(0, len(texts), 32):
            chunk = texts[i:i + 32]
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
            feats.append(pooled.cpu().numpy().astype(np.float32))
    del enc
    torch.cuda.empty_cache()
    return np.concatenate(feats)


def head_run(Ftr, Fdv, Fte, ytr, ydv, yte, K, seeds, epochs=12, patience=3):
    import sklearn.metrics as sm
    device = "cuda"
    res = []
    for seed in seeds:
        torch.manual_seed(seed)
        head = torch.nn.Linear(Ftr.shape[1], K).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        Xtr = torch.tensor(Ftr, device=device); ytr_t = torch.tensor(ytr, device=device)
        Xdv = torch.tensor(Fdv, device=device); Xte = torch.tensor(Fte, device=device)
        best = {"f1": -1, "state": None, "epoch": None}
        curve = []
        n = len(Xtr)
        for ep in range(1, epochs + 1):
            head.train()
            rng = np.random.RandomState(seed * 100 + ep)
            order = torch.tensor(rng.permutation(n), device=device)
            for i in range(0, n, 256):
                idx = order[i:i + 256]
                opt.zero_grad()
                loss = F.cross_entropy(head(Xtr[idx]), ytr_t[idx])
                loss.backward(); opt.step()
            head.eval()
            with torch.inference_mode():
                f1 = float(sm.f1_score(ydv, head(Xdv).argmax(1).cpu().numpy(), average="macro", zero_division=0))
            curve.append({"epoch": ep, "dev_macro_f1": f1})
            if f1 > best["f1"]:
                best = {"f1": f1, "epoch": ep, "state": {k: v.detach().clone() for k, v in head.state_dict().items()}}
            if ep - best["epoch"] >= patience:
                break
        for k, v in head.state_dict().items():
            v.copy_(best["state"][k])
        head.eval()
        with torch.inference_mode():
            probs = F.softmax(head(Xte).float(), 1).cpu().numpy()
        res.append({"seed": seed, "dev_curve": curve, "best_epoch": best["epoch"], "best_dev_f1": best["f1"],
                    "_probs": probs.astype(np.float16), "test": metrics(probs, yte)})
        print(f"[r4c] head s{seed}: best ep{best['epoch']} dev {best['f1']:.4f} test {res[-1]['test']['macro_f1']:.4f}", flush=True)
        del Xtr, Xdv, Xte, head
        torch.cuda.empty_cache()
    return res


def lora_run(texts_tr, texts_dv, texts_te, ytr, ydv, yte, K, seeds, epochs=2, patience=1, bs=16, accum=2):
    import sklearn.metrics as sm
    device = "cuda"
    ids_tr = [np.array(t, dtype=np.int32) for t in texts_tr]
    ids_dv = [np.array(t, dtype=np.int32) for t in texts_dv]
    ids_te = [np.array(t, dtype=np.int32) for t in texts_te]
    res = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        model = FTModel("lora").to(device)
        if model.head.out_features != K:  # 外部源类别数 ≠ 6（DCAN）时重建分类头
            import torch.nn as nn
            model.head = nn.Linear(model.head.in_features, K).to(device)
        enc_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")]
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head")]
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW([{"params": enc_params, "lr": 3e-4}, {"params": head_params, "lr": 1e-3}],
                                weight_decay=1e-4)
        best = {"f1": -1, "state": None, "epoch": None}
        curve = []
        t0 = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            rng = np.random.RandomState(seed * 100 + ep)
            order = rng.permutation(len(ids_tr))
            micro = [order[i:i + bs] for i in range(0, len(order), bs)]
            for w0 in range(0, len(micro), accum):
                window = micro[w0:w0 + accum]
                opt.zero_grad()
                for idxs in window:
                    chunk = [ids_tr[j] for j in idxs]
                    ids, mask = collate(chunk, 0, device)
                    yb = torch.tensor(ytr[idxs], device=device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(ids, mask)
                        loss = F.cross_entropy(logits.float(), yb) / len(window)
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(enc_params + head_params, 1.0)
                opt.step()
            dv_probs = evaluate(model, ids_dv, ydv, np.arange(len(ids_dv)), device, 0)
            f1 = float(sm.f1_score(ydv, dv_probs.argmax(1), average="macro", zero_division=0))
            curve.append({"epoch": ep, "dev_macro_f1": f1, "sec": round(time.time() - t0, 1)})
            print(f"[r4c] lora s{seed} ep{ep}: dev {f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if f1 > best["f1"]:
                best = {"f1": f1, "epoch": ep,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in trainable}}
            if ep - best["epoch"] >= patience:
                break
        for k, v in model.state_dict().items():
            if k in best["state"]:
                v.copy_(best["state"][k])
        probs = evaluate(model, ids_te, yte, np.arange(len(ids_te)), device, 0)
        res.append({"seed": seed, "dev_curve": curve, "best_epoch": best["epoch"], "best_dev_f1": best["f1"],
                    "wall_sec": round(time.time() - t0, 1),
                    "_probs": probs.astype(np.float16), "test": metrics(probs, yte)})
        print(f"[r4c] lora s{seed}: best ep{best['epoch']} dev {best['f1']:.4f} test {res[-1]['test']['macro_f1']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    return res


def cluster_bootstrap(probs_a, probs_b, y, clusters, rng_seed=0):
    import sklearn.metrics as sm
    cl = np.array(clusters)
    uniq = sorted(set(cl.tolist()))
    idx_by = {c: np.where(cl == c)[0] for c in uniq}
    rng = np.random.default_rng(rng_seed)
    d0 = float(sm.f1_score(y, probs_a.argmax(1), average="macro", zero_division=0) -
               sm.f1_score(y, probs_b.argmax(1), average="macro", zero_division=0))
    deltas = np.empty(B)
    for i in range(B):
        samp = rng.choice(len(uniq), len(uniq), replace=True)
        idx = np.concatenate([idx_by[uniq[j]] for j in samp])
        deltas[i] = (sm.f1_score(y[idx], probs_a[idx].argmax(1), average="macro", zero_division=0) -
                     sm.f1_score(y[idx], probs_b[idx].argmax(1), average="macro", zero_division=0))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"diff": d0, "ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0)}


def run_source(tag, tr, dv, te, K, seeds, outd, args, cluster_key):
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "logs").mkdir(exist_ok=True)
    ytr = np.array([r["label"] for r in tr]); ydv = np.array([r["label"] for r in dv]); yte = np.array([r["label"] for r in te])
    res = {"config": {"script": "scripts/dcan_round4_external.py", "source": tag,
                      "train_rows": len(tr), "dev_rows": len(dv), "test_rows": len(te),
                      "classes": K, "seeds": seeds, "text_cap_chars": TEXT_CAP,
                      "neural_budget": {"stacad": "file-level subtrain cap", "droid": "fold_plan train"},
                      "lora": "round2 同超参（enc 3e-4/head 1e-3/bs16×accum2/bf16）；epochs ≤%d patience %d" % (args.lora_epochs, args.lora_patience),
                      "selection": "best-dev 恢复；test 冻结后评一次", "boundary": "诊断性证据，独立于 408-task 主表"},
           "models": {}, "bootstrap": {}}
    yte_arr = yte
    # TF-IDF
    tf = tfidf_run(tr, dv, te, K, seeds)
    probs_tf = {}
    for r in tf:
        probs_tf[r["seed"]] = r.pop("_probs")
    res["models"]["tfidf"] = tf
    # head-only
    cache = outd / "features.npz"
    if cache.exists():
        z = np.load(cache)
        Ftr, Fdv, Fte = z["Ftr"], z["Fdv"], z["Fte"]
    else:
        Ftr = encode_features([r["text"] for r in tr])
        Fdv = encode_features([r["text"] for r in dv])
        Fte = encode_features([r["text"] for r in te])
        np.savez_compressed(cache, Ftr=Ftr, Fdv=Fdv, Fte=Fte)
    hd = head_run(Ftr, Fdv, Fte, ytr, ydv, yte, K, seeds)
    probs_hd = {}
    for r in hd:
        probs_hd[r["seed"]] = r.pop("_probs")
    res["models"]["head_only"] = hd
    # LoRA（与 DCAN 相同 tokenize：不含 special token；>512 截 384+128）
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    def tok_ids(t):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) > 512:
            ids = ids[:384] + ids[-128:]
        return ids
    ids_tr = [tok_ids(r["text"]) for r in tr]
    ids_dv = [tok_ids(r["text"]) for r in dv]
    ids_te = [tok_ids(r["text"]) for r in te]
    lo = lora_run(ids_tr, ids_dv, ids_te, ytr, ydv, yte, K, seeds,
                  epochs=args.lora_epochs, patience=args.lora_patience)
    probs_lo = {}
    for r in lo:
        probs_lo[r["seed"]] = r.pop("_probs")
    res["models"]["lora"] = lo
    # bootstrap：tfidf vs head、tfidf vs lora、head vs lora（逐 seed，按 cluster_key 聚类）
    clusters = [r[cluster_key] for r in te]
    for a, pa, b, pb in [("tfidf", probs_tf, "head_only", probs_hd), ("tfidf", probs_tf, "lora", probs_lo),
                         ("head_only", probs_hd, "lora", probs_lo)]:
        rows_b = []
        for s in seeds:
            rows_b.append({"seed": s, **cluster_bootstrap(pa[s].astype(np.float64), pb[s].astype(np.float64),
                                                          yte_arr, clusters, rng_seed=s)})
        res["bootstrap"][f"{a} - {b}"] = {"per_seed": rows_b,
                                          "mean_diff": float(np.mean([r["diff"] for r in rows_b]))}
    pack = {"y_test": yte_arr.astype(np.int16),
            "cluster_test": np.array(clusters, dtype=object),
            "test_meta": np.array([json.dumps({k: v for k, v in r.items() if k != "text"}) for r in te], dtype=object)}
    for s in seeds:
        pack[f"tfidf_s{s}_probs"] = probs_tf[s]
        pack[f"head_only_s{s}_probs"] = probs_hd[s]
        pack[f"lora_s{s}_probs"] = probs_lo[s]
    np.savez_compressed(outd / "predictions.npz", **pack)
    (outd / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[r4c] {tag} done -> {outd}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="stacad,droid")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--subtrain-cap", type=int, default=30000)
    ap.add_argument("--dev-rows", type=int, default=3000)
    ap.add_argument("--lora-epochs", type=int, default=2)
    ap.add_argument("--lora-patience", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.subtrain_cap, args.dev_rows = 1, 1200, 300
        args.lora_epochs, args.lora_patience = 1, 1
    seeds = list(range(args.seeds))
    for token in args.sources.split(","):
        outd = Path(args.out) / ("stacad_fold0" if token == "stacad" else "droid_fold0")
        if (outd / "metrics.json").exists():
            print(f"[r4c] skip existing {outd}", flush=True)
            continue
        if token == "stacad":
            rows = load_stacad()
            tr, dv, te, classes = split_stacad(rows, args.subtrain_cap, args.dev_rows)
            K = 7
            if args.smoke:
                tr, dv, te = tr[:1200], dv[:300], te[:600]
            print(f"[r4c] stacad: train {len(tr)} dev {len(dv)} test {len(te)}", flush=True)
            run_source("stacad_fold0", tr, dv, te, K, seeds, outd, args, cluster_key="file")
        else:
            fp = json.loads((DATA / "h2_droid_full_selected" / "fold_plan.json").read_text())
            tr, dv, te, classes = load_droid(fp)
            K = len(classes)
            if args.smoke:
                tr, dv, te = tr[:1200], dv[:300], te[:600]
            print(f"[r4c] droid: train {len(tr)} dev {len(dv)} test {len(te)} classes {K}", flush=True)
            run_source("droid_fold0", tr, dv, te, K, seeds, outd, args, cluster_key="generator")
    print("[r4c] all done", flush=True)


if __name__ == "__main__":
    main()
