#!/usr/bin/env python
"""AuthorBench 任务条件归因审计（规范：`docx/d-det_AuthorBench任务条件归因实验_DeepSeek执行指导_2026-10-01.md`）。

阶段与内容：
  · A（--audit-only，不加载 encoder）：audit.json——行/任务/模型/family 计数、每任务
    恰好 8 行 8 模型、task_split 与 fold_plan 一致、source_sha256 重复与跨 split 泄漏、
    replicate_count 与元数据（char/nlines/nloc/CC/token_size）分布。
  · B 表示提取：本地**未微调** CodeT5-base（checkpoints/codet5-base，复用
    encoders/codet5.py 加载方式，冻结）+ attention-mask 均值池化 → 768 维；
    max_length=512，超长头 384+尾 128，<8 跳过并记录；bf16 仅用于 GPU 推理，缓存
    float32；逐行保存 task_id/model_name/family/source_sha256，训练前与 JSONL 复核。
  · C family 6-way：z_raw / z_center（任务内无标签均值，转导）/ z_center_std（训练任务
    标准差）/ meta_only（原始量纲）/ **meta_std（训练任务逐维标准化，补充基准；原 meta_only
    因量纲差异 lbfgs 不收敛，需以此为准）**；LR C∈{0.03,0.1} 用 dev 选一 + LDA(lsqr, shrinkage=auto) 交叉
    检查；train-only 拟合，test 终评：macro-F1 / balanced acc / 逐族 recall / 混淆矩阵 /
    任务级 bootstrap CI；机会 1/6。
  · D model 8-way 捷径诊断（raw vs center 等），机会 1/8。
  · E 描述性诊断：train-fit PCA 2D（raw/center，家族与 split 两色）、KMeans k=6
    （silhouette/ARI/purity）、LDA 2D 投影、类内/类间协方差迹比；图件标注"非 H2 证据"。
  · §5 OpenAI generator 留出辅助：3 折（留出 gpt-4o / gpt-4.1 / gpt-4o-mini），训练侧=
    其余 7 模型（train 任务）；两个口径：任务条件（dev+test 任务，兄弟可中心）与严格
    任务留出（仅 test 任务，test 不参与任何拟合）；family=OpenAI recall + model 侧
    诊断（8-way 指纹 recall + 7-way 分配占比）；raw/center 对照。

资源：OMP=MKL=2；torch 线程 2；GPU 仅小批量推理（默认 bs=8）；不联网不下载。
用法：
  python scripts/authorbench_taskaware.py --audit-only
  python scripts/authorbench_taskaware.py --smoke --batch-size 8 --max-length 512 \
      --out artifacts/authorbench_taskaware_smoke
  python scripts/authorbench_taskaware.py --batch-size 8 --max-length 512 \
      --out artifacts/authorbench_taskaware
产物：artifacts/authorbench_taskaware/{audit,config,manifest,metrics}.json +
predictions.npz + solver.log + 图件（fig_pca_raw/center.png、fig_lda.png、
fig_kmeans.png；report.md 由报告复制）；缓存 runs/authorbench_taskaware/（不入库）；
默认拒绝覆盖（--force 允许）。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score,  # noqa: E402
                             confusion_matrix, f1_score, silhouette_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA = ROOT / "data/h2_authorbench"
C_GRID = [0.03, 0.1]
MAX_ITER = 10000
TOL = 1e-6
STD_FLOOR = 1e-2
BOOT = 1000
SEED = 0
CHANCE_F = 1.0 / 6.0
CHANCE_M = 1.0 / 8.0
META_COLS = ["char_count", "num_lines", "nloc", "cyclomatic_complexity", "token_size"]
REPS = ["z_raw", "z_center", "z_center_std", "meta_only", "meta_std"]
SMOKE_TASKS = {"train": 24, "dev": 8, "test": 8}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def stream_core(path: Path):
    """逐行流式读取 core.jsonl。"""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            yield {"i": i, "task_id": str(r["task_id"]), "model_name": str(r["model_name"]),
                   "family": str(r["family"]), "split": str(r["task_split"]),
                   "sha": str(r["source_sha256"]), "code": r["code"],
                   "replicate_count": int(r["replicate_count"]),
                   "char_count": int(r["char_count"]), "num_lines": int(r["num_lines"]),
                   "nloc": int(r["nloc"]),
                   "cyclomatic_complexity": float(r["cyclomatic_complexity"]),
                   "token_size": int(r["token_size"])}


def pctiles(vals) -> dict:
    v = np.asarray(vals, dtype=np.float64)
    if len(v) == 0:
        return {}
    q = np.percentile(v, [0, 25, 50, 75, 90, 99, 100])
    return {"min": round(float(q[0]), 2), "p25": round(float(q[1]), 2),
            "p50": round(float(q[2]), 2), "p75": round(float(q[3]), 2),
            "p90": round(float(q[4]), 2), "p99": round(float(q[5]), 2),
            "max": round(float(q[6]), 2)}


def run_audit(rows, fold_plan, summary, say) -> dict:
    n = len(rows)
    tasks = sorted({r["task_id"] for r in rows})
    models = sorted({r["model_name"] for r in rows})
    fams = sorted({r["family"] for r in rows})
    assert n == 1912 and len(tasks) == 239, f"行/任务数异常 {n}/{len(tasks)}"
    assert models == sorted(summary["models"]) and len(models) == 8, "模型集合不符"
    assert fams == sorted(summary["families"]), "family 集合不符"
    # 每任务 8 行、8 个不同模型
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    for t, rs in by_task.items():
        assert len(rs) == 8 and len({x["model_name"] for x in rs}) == 8, f"{t} 非 8 模型"
    # split 一致性与计数
    split_of_task = {t: rs[0]["split"] for t, rs in by_task.items()}
    assert all(len({x["split"] for x in rs}) == 1 for rs in by_task.values())
    fp_split = fold_plan["task_split"]
    assert set(fp_split) == set(split_of_task) and all(
        fp_split[t] == split_of_task[t] for t in split_of_task), "fold_plan 与 core 划分不一致"
    split_tasks = Counter(split_of_task.values())
    assert split_tasks["train"] == 167 and split_tasks["dev"] == 36 and \
        split_tasks["test"] == 36, f"划分计数异常 {split_tasks}"
    split_rows = Counter(r["split"] for r in rows)
    # hash 检查
    sha_all = [r["sha"] for r in rows]
    dup_global = len(sha_all) - len(set(sha_all))
    sha_by_split = {sp: {r["sha"] for r in rows if r["split"] == sp}
                    for sp in ("train", "dev", "test")}
    cross = {f"{a}_{b}": len(sha_by_split[a] & sha_by_split[b])
             for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))}
    assert all(v == 0 for v in cross.values()), f"跨 split sha 重复：{cross}"
    # 计数与分布
    au = {
        "totals": {"rows": n, "tasks": len(tasks), "models": len(models),
                   "families": len(fams), "language": ["C"],
                   "split_tasks": dict(split_tasks), "split_rows": dict(split_rows)},
        "per_model": dict(Counter(r["model_name"] for r in rows)),
        "per_family": dict(Counter(r["family"] for r in rows)),
        "family_models": {f: sorted({r["model_name"] for r in rows if r["family"] == f})
                          for f in fams},
        "replicate_count": dict(Counter(r["replicate_count"] for r in rows)),
        "replicate_note": "源数据同一 prompt-model 的重复数；本子集每对仅保留 1 条，"
                          "重复不视为独立任务",
        "sha256": {"global_rows": n, "unique": len(set(sha_all)),
                   "dup_global": dup_global, "cross_split": cross,
                   "within_task_dup": sum(
                       1 for rs in by_task.values() if len({x["sha"] for x in rs}) < 8)},
        "metadata_percentiles": {c: pctiles([r[c] for r in rows]) for c in META_COLS},
        "metadata_per_model_p50": {m: {c: pctiles([r[c] for r in rows
                                                   if r["model_name"] == m])["p50"]
                                       for c in META_COLS} for m in models},
        "task_index_ok": True,
    }
    return au


def build_features(rows, args, runs_dir: Path, say):
    """未微调 CodeT5-base 冻结编码 + 均值池化（768 维）。"""
    import torch
    from transformers import AutoTokenizer
    from encoders.codet5 import CodeT5Encoder

    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    tok = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    keep, skipped = [], []
    t_tok = time.time()
    CH = 512
    for s in range(0, len(rows), CH):
        chunk = rows[s:s + CH]
        enc = tok([r["code"] for r in chunk], add_special_tokens=False)["input_ids"]
        for r, ids in zip(chunk, enc):
            if len(ids) > args.max_length:
                head = int(args.max_length * 0.75)
                ids = ids[:head] + ids[-(args.max_length - head):]
            if len(ids) < 8:
                skipped.append({"i": r["i"], "task_id": r["task_id"],
                                "model_name": r["model_name"]})
                continue
            keep.append((r, np.asarray(ids, dtype=np.int64)))
    tl = np.array([len(ids) for _, ids in keep])
    say(f"[ab] tokenize {time.time()-t_tok:.1f}s：保留 {len(keep)}，跳过(<8) "
        f"{len(skipped)}；tok 长度 p50/p90/p99/max={np.percentile(tl,50):.0f}/"
        f"{np.percentile(tl,90):.0f}/{np.percentile(tl,99):.0f}/{tl.max()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_model = CodeT5Encoder(path=str(ROOT / "checkpoints/codet5-base"),
                              dtype="float32", freeze=True,
                              max_length=args.max_length).to(device).eval()

    def encode(doc_ids):
        order = np.argsort([len(x) for x in doc_ids], kind="stable")
        z = np.empty((len(doc_ids), 768), np.float32)
        for s in range(0, len(order), args.batch_size):
            ch = order[s:s + args.batch_size]
            L = max(len(doc_ids[i]) for i in ch)
            ids = torch.zeros(len(ch), L, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, i in enumerate(ch):
                ids[j, :len(doc_ids[i])] = torch.from_numpy(doc_ids[i])
                mask[j, :len(doc_ids[i])] = 1
            ids = ids.to(device); mask = mask.to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=device == "cuda"):
                h = enc_model(ids, mask)
            h = h.float()
            nv = mask.sum(1, keepdim=True).clamp(min=1)
            z[ch] = ((h * mask.unsqueeze(-1)).sum(1) / nv).cpu().numpy()
        return z

    ids_all = [ids for _, ids in keep]
    t0 = time.time()
    z = encode(ids_all)
    dt = time.time() - t0
    maxvram = (torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else 0.0
    say(f"[ab] 编码 {len(keep)} 行 {dt:.1f}s（{len(keep)/max(dt,1e-9):.1f} 行/s，"
        f"bs={args.batch_size}，device={device}，maxVRAM {maxvram:.0f}MB）")
    assert z.shape == (len(keep), 768) and np.isfinite(z).all()
    z2a = encode(ids_all[:16])
    z2b = encode(ids_all[:16])
    rep_ok = bool(np.array_equal(z2a, z2b))
    cross_gap = float(np.abs(z2a - z[:16]).max())
    say(f"[ab] 重复编码一致性（同批序×2）：{'逐位一致' if rep_ok else '存在差异'}；"
        f"跨批组成差异 max|Δ|={cross_gap:.2e}（bf16 批核效应，记录不阻塞）")
    assert rep_ok, "同一输入（同批序）重复编码不一致"

    runs_dir.mkdir(parents=True, exist_ok=True)
    feats = {
        "z": z,
        "row_i": np.array([r["i"] for r, _ in keep], np.int64),
        "task_id": np.array([r["task_id"] for r, _ in keep]),
        "model_name": np.array([r["model_name"] for r, _ in keep]),
        "family": np.array([r["family"] for r, _ in keep]),
        "split": np.array([r["split"] for r, _ in keep]),
        "sha": np.array([r["sha"] for r, _ in keep]),
        "replicate_count": np.array([r["replicate_count"] for r, _ in keep], np.int64),
        "tok_len": np.array([len(x) for x in ids_all], np.int32),
        "meta": np.array([[r[c] for c in META_COLS] for r, _ in keep], np.float64),
        "skipped_json": json.dumps(skipped, ensure_ascii=False),
        "maxvram_mb": np.float64(maxvram),
        "dtype_saved": np.array(["float32"]),
    }
    np.savez_compressed(runs_dir / "features.npz", **feats)
    with open(runs_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r, _ in keep:
            f.write(json.dumps({"i": r["i"], "task_id": r["task_id"],
                                "model_name": r["model_name"], "family": r["family"],
                                "split": r["split"], "sha": r["sha"]},
                               ensure_ascii=False) + "\n")
    return feats


def load_features(runs_dir: Path, rows):
    z = dict(np.load(runs_dir / "features.npz", allow_pickle=True))
    by_i = {r["i"]: r for r in rows}
    for j, ri in enumerate(z["row_i"]):
        r = by_i.get(int(ri))
        assert r is not None and r["sha"] == z["sha"][j] and r["task_id"] == z["task_id"][j], \
            f"缓存与 JSONL 不符 @ {j}"
    skip = json.loads(str(z["skipped_json"]))
    assert len(z["row_i"]) + len(skip) == len(rows), "缓存行数 + 跳过 ≠ 总行数"
    return z


def make_reps(feats) -> dict:
    z = feats["z"].astype(np.float64)
    task_id, split = feats["task_id"], feats["split"]
    tr = split == "train"
    reps = {"z_raw": z.copy()}
    # 任务内无标签均值中心（转导：使用同任务 8 个输出）
    mu = {}
    for t, idx in _group(task_id):
        mu[t] = z[idx].mean(0)
    zc = z - np.array([mu[t] for t in task_id])
    reps["z_center"] = zc
    m, s = zc[tr].mean(0), np.maximum(zc[tr].std(0), STD_FLOOR)
    reps["z_center_std"] = (zc - m) / s
    reps["meta_only"] = feats["meta"].copy()
    mm, ms = feats["meta"][tr].mean(0), np.maximum(feats["meta"][tr].std(0), STD_FLOOR)
    reps["meta_std"] = (feats["meta"] - mm) / ms
    return reps


def _group(keys):
    d = defaultdict(list)
    for i, k in enumerate(keys):
        d[str(k)].append(i)
    return sorted(d.items())


def fit_lr(X, y, C):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C).fit(X, y)
    conv = not any(issubclass(x.category, ConvergenceWarning) for x in wl)
    return clf, conv


def eval_clf(y, pred, classes) -> dict:
    rec = {c: (round(float((pred[y == k] == k).mean()), 4) if int((y == k).sum()) else None)
           for k, c in enumerate(classes)}
    cm = confusion_matrix(y, pred, labels=list(range(len(classes))))
    return {"macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
            "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 4),
            "per_class_recall": rec,
            "confusion": cm.tolist(),
            "n": int(len(y))}


def task_bootstrap(y, pred, task_ids, iters=BOOT, seed=SEED):
    uniq = sorted(set(task_ids.tolist()))
    rng = np.random.RandomState(seed)
    by_t = {t: np.where(task_ids == t)[0] for t in uniq}
    f1s, bas = [], []
    for _ in range(iters):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([by_t[uniq[p]] for p in pick])
        f1s.append(f1_score(y[idx], pred[idx], average="macro"))
        bas.append(balanced_accuracy_score(y[idx], pred[idx]))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return {"macro_f1_ci95": q(f1s), "balanced_acc_ci95": q(bas)}


def run_attribution(reps, feats, label_kind, classes, say, tag):
    """C（family 6-way）与 D（model 8-way）共用：LR 选 C + LDA 交叉检查 + test 评估。"""
    y_all = None
    if label_kind == "family":
        y_all = np.array([classes.index(f) for f in feats["family"]], np.int64)
    else:
        y_all = np.array([classes.index(m) for m in feats["model_name"]], np.int64)
    split, task_id = feats["split"], feats["task_id"]
    tr, dv, te = split == "train", split == "dev", split == "test"
    out = {}
    preds_store = {}
    for rep, X in reps.items():
        sel = {}
        for C in C_GRID:
            clf, conv = fit_lr(X[tr], y_all[tr], C)
            dev = eval_clf(y_all[dv], clf.predict(X[dv]), classes)
            sel[str(C)] = {"dev_macro_f1": dev["macro_f1"],
                           "dev_balanced_acc": dev["balanced_acc"], "converged": conv}
        best_C = max(C_GRID, key=lambda C: (sel[str(C)]["dev_macro_f1"], -C))
        clf, conv = fit_lr(X[tr], y_all[tr], best_C)
        pred_te = clf.predict(X[te])
        res = eval_clf(y_all[te], pred_te, classes)
        res["bootstrap"] = task_bootstrap(y_all[te], pred_te, task_id[te])
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[tr], y_all[tr])
        res["lda_check"] = eval_clf(y_all[te], lda.predict(X[te]), classes)
        res["lda_check"].pop("confusion")
        out[rep] = {"C_selected": best_C, "C_selection_dev": sel, "lr_converged": conv,
                    **res}
        preds_store[rep] = {"test": pred_te.astype(np.int16),
                            "dev": clf.predict(X[dv]).astype(np.int16),
                            "lda_test": lda.predict(X[te]).astype(np.int16)}
        say(f"[ab] {tag}·{rep}: C*={best_C} test macroF1 {res['macro_f1']} "
            f"balAcc {res['balanced_acc']}（LDA {res['lda_check']['macro_f1']}）")
    return out, y_all, preds_store


def run_diagnostics(reps, feats, out_dir: Path, say):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split, task_id = feats["split"], feats["task_id"]
    fams = sorted(set(feats["family"].tolist()))
    tr = split == "train"
    colors = plt.get_cmap("tab10")
    fam_c = {f: colors(i) for i, f in enumerate(fams)}
    diag = {"note": "描述性诊断，不是 H2 证据"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, rep in zip(axes, ("z_raw", "z_center")):
        X = reps[rep]
        clf = _pca2().fit(X[tr])
        P = clf.transform(X)
        for f in fams:
            m = feats["family"] == f
            ax.scatter(P[m, 0], P[m, 1], s=6, alpha=0.55, label=f, color=fam_c[f])
        ax.set_title(f"PCA(2, train-fit) — {rep}")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Descriptive only — not H2 evidence", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pca_raw_center_family.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, rep in zip(axes, ("z_raw", "z_center")):
        X = reps[rep]
        P = _pca2().fit(X[tr]).transform(X)
        for sp, mk in (("train", "o"), ("dev", "^"), ("test", "s")):
            m = split == sp
            ax.scatter(P[m, 0], P[m, 1], s=6, alpha=0.55, label=sp, marker=mk)
        ax.set_title(f"PCA(2, train-fit) — {rep} by split")
        ax.legend(fontsize=7)
    fig.suptitle("Descriptive only — not H2 evidence", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pca_raw_center_split.png", dpi=140)
    plt.close(fig)

    km = {}
    for rep in ("z_raw", "z_center"):
        X = reps[rep]
        kc = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit(X[tr])
        lab = kc.labels_
        fam_y = np.array([fams.index(f) for f in feats["family"][tr]])
        cmx = confusion_matrix(fam_y, lab, labels=list(range(6)))
        purity = float(cmx.max(0).sum() / len(lab))
        km[rep] = {"silhouette_train": round(float(silhouette_score(X[tr], lab)), 4),
                   "ARI_train": round(float(adjusted_rand_score(fam_y, lab)), 4),
                   "family_purity_train": round(purity, 4)}
        say(f"[ab] KMeans k=6 {rep}: {km[rep]}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, rep in zip(axes, ("z_raw", "z_center")):
        X = reps[rep]
        P = _pca2().fit(X[tr]).transform(X)
        kc = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit(X[tr])
        lab_all = kc.predict(X)
        for k in range(6):
            m = lab_all == k
            ax.scatter(P[m, 0], P[m, 1], s=6, alpha=0.55, label=f"c{k}")
        ax.set_title(f"KMeans(k=6, train-fit) — {rep}")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Descriptive only — not H2 evidence", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_kmeans.png", dpi=140)
    plt.close(fig)

    lda_proj = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fam_y_all = np.array([fams.index(f) for f in feats["family"]])
    for ax, rep in zip(axes, ("z_raw", "z_center")):
        X = reps[rep]
        lda = LinearDiscriminantAnalysis(n_components=2, solver="svd").fit(X[tr], fam_y_all[tr])
        P = lda.transform(X)
        for f in fams:
            m = feats["family"] == f
            ax.scatter(P[m, 0], P[m, 1], s=6, alpha=0.55, label=f, color=fam_c[f])
        ax.set_title(f"LDA(2, train-fit) — {rep}")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Descriptive only — not H2 evidence", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_lda.png", dpi=140)
    plt.close(fig)

    covr = {}
    for rep in ("z_raw", "z_center"):
        X = reps[rep]
        covr[rep] = {}
        for scope, msk in (("train", tr), ("test", split == "test")):
            Xs = X[msk]; ys = fam_y_all[msk]
            mu = Xs.mean(0)
            Sw = np.zeros((Xs.shape[1], Xs.shape[1])); Sb = np.zeros_like(Sw)
            for k in range(len(fams)):
                Xk = Xs[ys == k]
                if len(Xk) == 0:
                    continue
                d = Xk - Xk.mean(0)
                Sw += d.T @ d
                dd = (Xk.mean(0) - mu)[:, None]
                Sb += len(Xk) * (dd @ dd.T)
            covr[rep][scope] = {"trace_ratio_within_over_between":
                                round(float(np.trace(Sw) / max(np.trace(Sb), 1e-12)), 4)}
    say(f"[ab] 协方差迹比: {covr}")
    return {"kmeans": km, "cov_trace": covr,
            "figures": ["fig_pca_raw_center_family.png", "fig_pca_raw_center_split.png",
                        "fig_kmeans.png", "fig_lda.png"]}


def _pca2():
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=SEED)


def run_openai_holdout(reps, feats, fold_plan, say):
    split, model = feats["split"], feats["model_name"]
    fam = feats["family"]
    fams = sorted(set(fam.tolist()))
    models = sorted(set(model.tolist()))
    tr = split == "train"
    out = {}
    for fold in fold_plan["generator_diagnostic_folds"]:
        hold = fold["heldout_generators"][0]
        side = [m for m in fold["train_generators"]]
        fam_tr = (fam != "openai") | (model != hold)
        fam_tr &= tr
        yfam = np.array([fams.index(f) for f in fam], np.int64)
        ymod8 = np.array([models.index(m) for m in model], np.int64)
        ymod7 = np.array([side.index(m) if m in side else -1 for m in model], np.int64)
        fold_out = {"heldout": hold, "train_side": side, "scopes": {}}
        for scope, scope_mask in (("task_conditioned_dev_test", (split != "train")),
                                  ("strict_test_only", split == "test")):
            ev = scope_mask & (model == hold)
            for rep in ("z_raw", "z_center"):
                X = reps[rep]
                clf_f, _ = fit_lr(X[fam_tr], yfam[fam_tr], 0.1)
                pf = clf_f.predict(X[ev])
                fam_recall = float((pf == fams.index("openai")).mean())
                clf_m8, _ = fit_lr(X[tr], ymod8[tr], 0.1)
                pm8 = clf_m8.predict(X[ev])
                m2 = float((pm8 == models.index(hold)).mean())
                clf_m7, _ = fit_lr(X[tr & (ymod7 >= 0)], ymod7[tr & (ymod7 >= 0)], 0.1)
                pm7 = clf_m7.predict(X[ev])
                shares = {side[k]: round(float((pm7 == k).mean()), 4)
                          for k in range(len(side))}
                fold_out["scopes"].setdefault(scope, {})[rep] = {
                    "n": int(ev.sum()),
                    "family_openai_recall": round(fam_recall, 4),
                    "model8way_holdout_recall(指纹诊断,未留出)": round(m2, 4),
                    "seen_model7way_assignment": shares,
                    "openai_side_share_7way": round(sum(
                        v for mname, v in shares.items() if mname in
                        ("gpt-4o", "gpt-4.1", "gpt-4o-mini")), 4)}
        out[hold] = fold_out
        say(f"[ab] §5 留出 {hold}: "
            + " | ".join(f"{sc}: famRecall raw "
                         f"{fold_out['scopes'][sc]['z_raw']['family_openai_recall']} / "
                         f"center {fold_out['scopes'][sc]['z_center']['family_openai_recall']}"
                         for sc in fold_out["scopes"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rebuild-features", action="store_true")
    a = ap.parse_args()
    OUT = Path(a.out) if a.out else ROOT / "artifacts" / (
        "authorbench_taskaware_smoke" if a.smoke else "authorbench_taskaware")
    RUNS = ROOT / "runs" / ("authorbench_taskaware_smoke" if a.smoke
                            else "authorbench_taskaware")
    if OUT.exists() and any(OUT.iterdir()) and not a.force:
        print(f"[ab] 已存在产物，默认拒绝覆盖：{OUT}（--force 可覆盖）", flush=True)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log: list[str] = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    commit = subprocess.run(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    say(f"[ab] commit {commit} | smoke={a.smoke} audit_only={a.audit_only} | out={OUT}")

    # ---- A 审计 ----
    sums = {name: {"sha256": sha256_file(DATA / name)}
            for name in ("core.jsonl", "task_index.jsonl", "fold_plan.json",
                         "summary.json", "README.md")}
    rows = list(stream_core(DATA / "core.jsonl"))
    fold_plan = json.loads((DATA / "fold_plan.json").read_text())
    summary = json.loads((DATA / "summary.json").read_text())
    audit = run_audit(rows, fold_plan, summary, say)
    audit["files"] = sums
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1))
    say(f"[ab] 审计完成：{audit['totals']['rows']} 行 / {audit['totals']['tasks']} 任务；"
        f"断言全过；audit.json 已写出")
    if a.audit_only:
        (OUT / "solver.log").write_text("\n".join(log) + "\n")
        return 0

    # ---- 特征 ----
    if a.smoke:
        task_split = {t: rs[0]["split"] for t, rs in _task_rows(rows).items()}
        keep_tasks = set()
        for sp, cap in SMOKE_TASKS.items():
            ts = sorted([t for t, s in task_split.items() if s == sp])[:cap]
            keep_tasks |= set(ts)
        rows = [r for r in rows if r["task_id"] in keep_tasks]
        say(f"[ab] smoke 子集：{len(rows)} 行 / {len(keep_tasks)} 任务")

    feats_path = RUNS / "features.npz"
    if feats_path.exists() and not a.rebuild_features:
        feats = load_features(RUNS, rows)
        say(f"[ab] 复用特征缓存并复核 ✓：{feats_path}")
    else:
        feats = build_features(rows, a, RUNS, say)

    reps = make_reps(feats)

    # ---- C family 6-way + D model 8-way ----
    fams = sorted(set(feats["family"].tolist()))
    models = sorted(set(feats["model_name"].tolist()))
    fam_res, _, fam_preds = run_attribution(reps, feats, "family", fams, say, "C-family")
    mod_res, _, mod_preds = run_attribution(reps, feats, "model", models, say, "D-model")

    # ---- E 诊断 ----
    diag = run_diagnostics(reps, feats, OUT, say)

    # ---- §5 OpenAI generator 留出 ----
    holdout = run_openai_holdout(reps, feats, fold_plan, say)

    # ---- 产物 ----
    metrics = {
        "commit": commit, "smoke": a.smoke,
        "data": {"source_archive_sha256": summary.get("source_archive_sha256"),
                 "counts": audit["totals"]},
        "chance": {"family_6way": round(CHANCE_F, 4), "model_8way": round(CHANCE_M, 4)},
        "representations": {
            "z_raw": "未微调 CodeT5-base m_raw（768 维）",
            "z_center": "z − 同任务 8 输出无标签均值（**转导**；不代表单样本部署算法）",
            "z_center_std": "z_center 再按训练任务逐维标准差标准化",
            "meta_only": f"元数据 {META_COLS}（原始量纲；LR 不收敛，仅作对照）",
            "meta_std": f"元数据 {META_COLS}（训练任务逐维均值/标准差标准化，补充基准）"},
        "family_attribution": fam_res, "model_attribution": mod_res,
        "diagnostics": diag, "openai_generator_holdout": holdout,
        "notes": ["AuthorBench 仅 OpenAI family 有多个 generator；§5 为辅助诊断，"
                  "不是多 family H2 结果",
                  "z_center 使用同任务兄弟输出（transductive）；严格口径=测试任务"
                  "不参与任何拟合",
                  "所有图件为描述性诊断，不作 H2 证据"],
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    (OUT / "config.json").write_text(json.dumps({
        "commit": commit, "C_grid": C_GRID, "selection": "dev macro-F1（并列取小 C）",
        "max_length": a.max_length, "long_rule": "头 75% + 尾 25%（512→384+128）",
        "min_tokens": 8, "batch_size": a.batch_size,
        "encoder": "未微调 checkpoints/codet5-base（冻结，均值池化，bf16 推理，缓存 "
                   "float32）", "bootstrap": {"iters": BOOT, "seed": SEED,
                                              "unit": "任务级重采样"},
        "smoke_tasks": SMOKE_TASKS, "oom_settings": {"OMP": 2, "MKL": 2,
                                                     "torch_threads": 2},
    }, ensure_ascii=False, indent=1))
    manifest = {
        "commit": commit, "smoke": a.smoke, "files": sums,
        "features_cache": {"path": str(feats_path), "sha256": sha256_file(feats_path),
                           "n": int(len(feats["row_i"])),
                           "skipped": len(json.loads(str(feats["skipped_json"]))),
                           "dtype_saved": "float32"},
        "device": "cuda" if _cuda() else "cpu",
        "max_vram_mb": float(feats.get("maxvram_mb", 0.0)),
        "threads": {"OMP": 2, "MKL": 2, "torch": 2},
        "usage": {"external_api": False, "downloads": False,
                  "transductive_center": True},
        "notes": ["C-only 数据；不外推语言/其他数据集",
                  "仅 OpenAI 有多个 generator，§5 不代表多 family H2"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    split = feats["split"]
    preds = {"task_id": feats["task_id"], "model_name": feats["model_name"],
             "family": feats["family"], "split": split,
             "tok_len": feats["tok_len"], "meta": feats["meta"],
             "classes_family": np.array(fams), "classes_model": np.array(models),
             "dev_idx": np.where(split == "dev")[0],
             "test_idx": np.where(split == "test")[0]}
    for rep in reps:
        preds[f"family_dev_{rep}"] = fam_preds[rep]["dev"]
        preds[f"family_test_{rep}"] = fam_preds[rep]["test"]
        preds[f"family_test_lda_{rep}"] = fam_preds[rep]["lda_test"]
        preds[f"model_dev_{rep}"] = mod_preds[rep]["dev"]
        preds[f"model_test_{rep}"] = mod_preds[rep]["test"]
        preds[f"model_test_lda_{rep}"] = mod_preds[rep]["lda_test"]
    np.savez_compressed(OUT / "predictions.npz", **preds)
    (OUT / "solver.log").write_text("\n".join(log) + "\n")
    say(f"[ab] 完成 {time.time()-t_start:.1f}s；产物 {OUT}")
    return 0


def _task_rows(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["task_id"]].append(r)
    return d


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    sys.exit(main())
