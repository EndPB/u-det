"""Stage D1: AuthorBench-DCAN task-aware H1 控制矩阵（同切分、test 只读一次）。

协议（与指导 §3 一致）：
- 数据：d-det/data/h2_authorbench_dcan/core.jsonl（9498 行；2715 task；6 family；C 语言）；
  主 split 保持包内 task_split；同一 task 不跨 split。
- 视图（预注册顺序）：
  1) metadata_only：长度/结构字段（char_count/num_lines/nloc/cyclomatic/token_size）——shortcut control；
  2) tfidf_word / 3) tfidf_char：词表仅 train 拟合；SGD 5ep best-dev 快照；3 seeds（0/1/2）；
  4) codet5_meanpool：冻结 CodeT5-small（包内已校验权重）均值池化 + 标准化 + LR（C 仅由 dev 选）；
  5) codet5_centered(+unscaled)：逐 task LOO 中心化（转导；无标签兄弟样本）——显式标注 transductive；
  6) P0 复用行：fusion/word/char/style_lgb/sem_lr/mean_ensemble（读既有 predictions，只重算指标）。
- 指标：macro-F1、balanced acc、per-class P/R、confusion、ECE15、NLL；500× task-cluster bootstrap CI；
  length/generator/task-size 分桶；与 P0 fusion(.8388)/word(.8200) 的 paired bootstrap 差值。
- test 不参与任何选择；test 读取时间记录在 metrics.json。

输出：d-det/artifacts/stage_d_h1_authorbench_dcan_2026-10-07/
      {config.json, metrics.json, predictions.npz, report.md, SHA256SUMS.txt, logs/}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "d-det/data/h2_authorbench_dcan/core.jsonl"
MODEL = ROOT / "d-det/models/codet5-small"
P0_NPZ = ROOT / "d-det/artifacts/acl_sota_p0/server_tracks/authorbench_dcan/predictions_all.npz"
sys.path.insert(0, str(ROOT / "d-det/scripts"))
import acl_sota_p0_baselines as p0mod  # noqa: E402

FAMILIES = ["claude", "deepseek", "gemini", "llama", "openai", "qwen"]
NC = len(FAMILIES)
SEEDS = [0, 1, 2]
BOOT = 500
BOOT_SEED = 20261007
C_GRID = (0.03, 0.1, 0.3, 1.0)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def full_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    import sklearn.metrics as sm
    pred = probs.argmax(1)
    rep = sm.classification_report(y, pred, labels=list(range(NC)), output_dict=True, zero_division=0)
    conf = sm.confusion_matrix(y, pred, labels=list(range(NC))).tolist()
    confv = probs.max(1)
    correct = (pred == y).astype(float)
    bins = np.clip(np.digitize(confv, np.linspace(0, 1, 16)) - 1, 0, 14)
    ece = float(sum(np.mean(bins == b) * abs(correct[bins == b].mean() - confv[bins == b].mean())
                    for b in range(15) if (bins == b).any())) if len(y) else 0.0
    return {
        "n": int(len(y)),
        "macro_f1": float(sm.f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_acc": float(sm.balanced_accuracy_score(y, pred)),
        "accuracy": float((pred == y).mean()),
        "per_class": {FAMILIES[c]: {"precision": rep[str(c)]["precision"],
                                    "recall": rep[str(c)]["recall"],
                                    "f1": rep[str(c)]["f1-score"],
                                    "support": int(rep[str(c)]["support"])} for c in range(NC)},
        "confusion_matrix": conf,
        "ece15": ece,
        "nll": float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)))),
    }


def macro_f1(y, pred):
    import sklearn.metrics as sm
    return sm.f1_score(y, pred, average="macro", zero_division=0)


def cluster_boot(probs, y, clusters, repeats=BOOT, seed=BOOT_SEED):
    cl = np.asarray([str(c) for c in clusters])
    uniq = np.unique(cl)
    idx_by = {c: np.where(cl == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    vals = np.empty(repeats)
    for k in range(repeats):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[uniq[p]] for p in pick])
        vals[k] = macro_f1(y[idx], probs[idx].argmax(1))
    return {"ci95_low": float(np.percentile(vals, 2.5)), "ci95_high": float(np.percentile(vals, 97.5)),
            "boot_mean": float(vals.mean())}


def paired_boot(probs_a, probs_b, y, clusters, repeats=BOOT, seed=BOOT_SEED):
    """cluster bootstrap of Δ = f1(a) - f1(b)，同一重采样索引（paired）。"""
    cl = np.asarray([str(c) for c in clusters])
    uniq = np.unique(cl)
    idx_by = {c: np.where(cl == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    d = np.empty(repeats)
    for k in range(repeats):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[uniq[p]] for p in pick])
        d[k] = macro_f1(y[idx], probs_a[idx].argmax(1)) - macro_f1(y[idx], probs_b[idx].argmax(1))
    return {"delta_mean": float(d.mean()), "ci95_low": float(np.percentile(d, 2.5)),
            "ci95_high": float(np.percentile(d, 97.5)), "frac_le_0": float((d <= 0).mean())}


def lr_select(Ftr, ytr, Fdv, ydv, standardize=True, grid=C_GRID):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr) if standardize else None
    Ztr = sc.transform(Ftr) if sc else Ftr
    Zdv = sc.transform(Fdv) if sc else Fdv
    best = (-1.0, None, None)
    for C in grid:
        clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(Ztr, ytr)
        f1 = macro_f1(ydv, clf.predict_proba(Zdv).argmax(1))
        if f1 > best[0]:
            best = (float(f1), C, clf)
    return best[2], sc, best[1], best[0]


def load_ab():
    rows = []
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    y = np.array([FAMILIES.index(r["family"]) for r in rows])
    split = np.array([r["task_split"] for r in rows])
    task = np.array([r["task_id"] for r in rows])
    gens = [r["model_name"] for r in rows]
    texts = [r["code"] for r in rows]
    lengths = np.array([float(r.get("char_count") or len(r["code"])) for r in rows])
    mask = {s: split == s for s in ("train", "dev", "test")}
    return rows, y, mask, task, gens, texts, lengths


def encode_codet5(texts, device_batch=8):
    import torch
    from transformers import RobertaTokenizer, T5EncoderModel
    torch.manual_seed(20261007)
    tok = RobertaTokenizer(vocab=str(MODEL / "vocab.json"), merges=str(MODEL / "merges.txt"),
                           unk_token="<unk>", bos_token="<s>", eos_token="</s>", sep_token="</s>",
                           cls_token="<s>", pad_token="<pad>", mask_token="<mask>",
                           add_prefix_space=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5EncoderModel.from_pretrained(
        MODEL, local_files_only=True,
        dtype=(torch.float16 if device.type == "cuda" else torch.float32))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoded = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) > 512:
            ids = ids[:384] + ids[-128:]
        encoded.append(ids)
    dim = int(model.config.d_model)
    emb = np.empty((len(encoded), dim), dtype=np.float32)
    for start in range(0, len(encoded), device_batch):
        batch = encoded[start:start + device_batch]
        ln = max(1, max(len(x) for x in batch))
        ids = torch.full((len(batch), ln), int(tok.pad_token_id), dtype=torch.long)
        am = torch.zeros((len(batch), ln), dtype=torch.long)
        for j, row in enumerate(batch):
            if row:
                ids[j, :len(row)] = torch.tensor(row, dtype=torch.long)
                am[j, :len(row)] = 1
        ids, am = ids.to(device), am.to(device)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    hidden = model(input_ids=ids, attention_mask=am, return_dict=True).last_hidden_state
            else:
                hidden = model(input_ids=ids, attention_mask=am, return_dict=True).last_hidden_state
            pooled = (hidden.float() * am.unsqueeze(-1)).sum(1) / am.sum(1, keepdim=True).clamp_min(1).float()
        emb[start:start + len(batch)] = pooled.cpu().numpy()
    return emb


def centered_loo(emb, task_ids):
    by_task = defaultdict(list)
    for i, t in enumerate(task_ids):
        by_task[t].append(i)
    zc = emb.copy()
    n_single = 0
    for t, idxs in by_task.items():
        if len(idxs) > 1:
            s = emb[idxs].sum(0)
            for i in idxs:
                zc[i] = emb[i] - (s - emb[i]) / (len(idxs) - 1)
        else:
            n_single += 1
    return zc, n_single


def bucket_metrics(probs, y, mask_sub, edges, values, name):
    out = {}
    labels = []
    if edges is None:
        for b, sel in mask_sub.items():
            labels.append((str(b), sel))
    else:
        for lo, hi in zip(edges[:-1], edges[1:]):
            labels.append((f"[{lo:.0f},{hi:.0f})", (values >= lo) & (values < hi)))
        labels.append((f">={edges[-1]:.0f}", values >= edges[-1]))
    for lab, sel in labels:
        if sel.sum() == 0:
            out[lab] = {"n": 0}
            continue
        yy, pp = y[sel], probs[sel]
        out[lab] = {"n": int(sel.sum()), "macro_f1": float(macro_f1(yy, pp.argmax(1))),
                    "accuracy": float((pp.argmax(1) == yy).mean()),
                    "classes_present": int(len(set(yy.tolist())))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "d-det/artifacts/stage_d_h1_authorbench_dcan_2026-10-07")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)
    t0 = time.time()

    rows, y, mask, task, gens, texts, lengths = load_ab()
    tr, dv, te = mask["train"], mask["dev"], mask["test"]
    print(f"rows={len(rows)} train={tr.sum()} dev={dv.sum()} test={te.sum()} "
          f"tasks={len(set(task.tolist()))} families={Counter(y.tolist())}", flush=True)

    views = {}          # name -> {"dv": probs, "te": probs, "meta": {...}}
    notes = {}
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    # ---------- 1) metadata-only ----------
    F = np.column_stack([np.log1p(lengths),
                         np.array([r["num_lines"] for r in rows], float),
                         np.array([r["nloc"] for r in rows], float),
                         np.array([r["cyclomatic_complexity"] for r in rows], float),
                         np.array([r["token_size"] for r in rows], float)])
    clf, sc, C, devf1 = lr_select(F[tr], y[tr], F[dv], y[dv])
    views["metadata_only"] = {"dv": clf.predict_proba(sc.transform(F[dv]) if sc else F[dv]),
                              "te": clf.predict_proba(sc.transform(F[te]) if sc else F[te]),
                              "meta": {"selected_C": C, "dev_macro_f1": devf1,
                                       "features": ["log1p(char_count)", "num_lines", "nloc",
                                                    "cyclomatic_complexity", "token_size"]}}
    log(f"metadata_only: dev {devf1:.4f}")

    # ---------- 2/3) TF-IDF（P0 同协议，3 seeds） ----------
    for tag, vec in (("tfidf_word", p0mod.tfidf_word()), ("tfidf_char", p0mod.tfidf_char())):
        Xtr = vec.fit_transform([texts[i] for i in np.where(tr)[0]])
        Xdv = vec.transform([texts[i] for i in np.where(dv)[0]])
        Xte = vec.transform([texts[i] for i in np.where(te)[0]])
        dvs, tes, devs = [], [], []
        for s in SEEDS:
            dvp, tep, bf1 = p0mod.sgd_snapshot(Xtr, y[tr], Xdv, y[dv], Xte, y[te], NC, s)
            dvs.append(dvp); tes.append(tep); devs.append(bf1)
        views[tag] = {"dv": np.mean(dvs, 0), "te": np.mean(tes, 0),
                      "meta": {"vocab": int(Xtr.shape[1]), "seeds": SEEDS,
                               "seed_best_dev_f1": [float(x) for x in devs]}}
        log(f"{tag}: vocab={Xtr.shape[1]} dev F1 {[round(float(x), 4) for x in devs]}")

    # ---------- 4) 冻结 CodeT5-small ----------
    emb = encode_codet5(texts)
    log(f"codet5-small embeddings: {emb.shape}")
    clf, sc, C, devf1 = lr_select(emb[tr], y[tr], emb[dv], y[dv])
    views["codet5_meanpool"] = {"dv": clf.predict_proba(sc.transform(emb[dv])), "te": clf.predict_proba(sc.transform(emb[te])),
                                "meta": {"selected_C": C, "dev_macro_f1": devf1,
                                         "encoder": "codet5-small frozen mean-pool 512d, trunc 384+128"}}
    log(f"codet5_meanpool: dev {devf1:.4f}")

    # ---------- 5) task-centered（转导诊断） ----------
    zc, n_single = centered_loo(emb, task)
    clf, sc, C, devf1 = lr_select(zc[tr], y[tr], zc[dv], y[dv])
    views["codet5_centered"] = {"dv": clf.predict_proba(sc.transform(zc[dv])), "te": clf.predict_proba(sc.transform(zc[te])),
                                "meta": {"selected_C": C, "dev_macro_f1": devf1, "transductive": True,
                                         "centering": "LOO task-sibling mean (label-free)", "singleton_tasks": n_single}}
    clf2, _, C2, devf12 = lr_select(zc[tr], y[tr], zc[dv], y[dv], standardize=False)
    views["codet5_centered_unscaled"] = {"dv": clf2.predict_proba(zc[dv]), "te": clf2.predict_proba(zc[te]),
                                         "meta": {"selected_C": C2, "dev_macro_f1": devf12, "transductive": True}}
    log(f"codet5_centered: dev {devf1:.4f} | unscaled: dev {devf12:.4f}")

    # ---------- 6) P0 复用 ----------
    z = np.load(P0_NPZ)
    assert np.array_equal(z["y_te"], y[te]), "P0 test order mismatch"
    for name in ("fusion_lr", "tfidf_word", "tfidf_char", "style_lgb", "style_lr", "sem_lr", "mean_ensemble"):
        dvk, tek = f"dv__{name}", f"te__{name}"
        if dvk in z.files and tek in z.files:
            views[f"p0_{name}"] = {"dv": np.asarray(z[dvk], np.float64), "te": np.asarray(z[tek], np.float64),
                                   "meta": {"reused": "acl_sota_p0 predictions_all.npz（只重算指标）"}}
    log(f"reused P0 rows: {[k for k in views if k.startswith('p0_')]}")

    # ---------- 指标、CI、分桶、paired 差值 ----------
    test_read_utc = datetime.now(timezone.utc).isoformat()
    train_char = lengths[tr]
    edges = np.quantile(train_char, [0.25, 0.5, 0.75]).tolist()
    tasks_per = Counter(task.tolist())
    size_of = np.array([tasks_per[t] for t in task])
    size_te = size_of[te]
    size_buckets = {"2": size_te == 2, "3": size_te == 3, "4-5": (size_te >= 4) & (size_te <= 5), "6+": size_te >= 6}
    gen_test = [gens[i] for i in np.where(te)[0]]

    results = {}
    te_probs_store = {}
    for name, v in views.items():
        p_te, p_dv = np.asarray(v["te"], np.float64), np.asarray(v["dv"], np.float64)
        m = full_metrics(p_te, y[te])
        ci = cluster_boot(p_te, y[te], task[te])
        res = {"test": m, "ci95": ci, "meta": v["meta"]}
        res["buckets"] = {
            "length_quartiles(test char_count; train-quantile edges)": bucket_metrics(
                p_te, y[te], None, [0.0] + edges + [np.inf], lengths[te], "len"),
            "generator": {g: {"n": int(sum(1 for x in gen_test if x == g)),
                              "accuracy": float((p_te[[j for j, x in enumerate(gen_test) if x == g]].argmax(1)
                                                 == y[te][[j for j, x in enumerate(gen_test) if x == g]]).mean())
                              if sum(1 for x in gen_test if x == g) else None} for g in sorted(set(gens))},
            "task_size": bucket_metrics(p_te, y[te], size_buckets, None, None, "size"),
            "language": {"C": {"n": int(te.sum())}},
        }
        if "p0_fusion_lr" in views:
            res["vs_p0_fusion"] = paired_boot(p_te, np.asarray(views["p0_fusion_lr"]["te"], np.float64), y[te], task[te])
        if "p0_tfidf_word" in views:
            res["vs_p0_word"] = paired_boot(p_te, np.asarray(views["p0_tfidf_word"]["te"], np.float64), y[te], task[te])
        results[name] = res
        te_probs_store[name] = p_te.astype(np.float16)
        log(f"[test] {name:28s} f1={m['macro_f1']:.4f} ba={m['balanced_acc']:.4f} "
            f"ci=[{ci['ci95_low']:.4f},{ci['ci95_high']:.4f}]")

    # ---------- 闸门评估（指导 §3.2） ----------
    p0_best = max(results["p0_fusion_lr"]["test"]["macro_f1"], results["p0_tfidf_word"]["test"]["macro_f1"])
    content = {k: results[k]["test"]["macro_f1"] for k in results
               if k in ("tfidf_word", "tfidf_char", "codet5_meanpool", "codet5_centered", "codet5_centered_unscaled")}
    best_c = max(content, key=content.get)
    meta_f1 = results["metadata_only"]["test"]["macro_f1"]
    gate = {
        "chance_macro_f1": 1.0 / NC,
        "best_content_view": best_c, "best_content_f1": content[best_c],
        "metadata_only_f1": meta_f1,
        "p0_best_same_split": p0_best,
        "cond1_content_above_chance_and_metadata": bool(content[best_c] > meta_f1 and content[best_c] > 2.0 / NC),
        "cond2_plus_1pt_over_p0": bool(content[best_c] - p0_best >= 0.01),
        "cond3_multiseed_or_multifold_direction": {
            "tfidf_word_seed_dev_f1": views["tfidf_word"]["meta"]["seed_best_dev_f1"],
            "tfidf_char_seed_dev_f1": views["tfidf_char"]["meta"]["seed_best_dev_f1"],
            "note": "TF-IDF 3 seeds；CodeT5 视图为确定性前向（无随机种子维度）"},
        "cond4_paired_ci_vs_p0": results[best_c].get("vs_p0_fusion"),
        "cond5_transductive_labeled": True,
        "cond6_detection_axis_not_mixed": True,
    }
    passed = all([gate["cond1_content_above_chance_and_metadata"], gate["cond2_plus_1pt_over_p0"]])
    gate["verdict"] = ("D1 通过（可进入 H2 方法实验）" if passed else
                       "D1 未通过：按指导 §3.2 —— 结论为 task-aware 可读性/任务效应审计；停止新架构探索，转入数据构造。")
    if not gate["cond2_plus_1pt_over_p0"]:
        gate["gap_to_1pt"] = float(p0_best + 0.01 - content[best_c])

    metrics = {
        "schema": "stage_d_h1_authorbench_dcan_v1",
        "protocol": {"data": "d-det/data/h2_authorbench_dcan/core.jsonl",
                     "data_sha256": sha256_file(DATA), "model": "d-det/models/codet5-small",
                     "split": "package task_split; task-isolated; test once",
                     "tfidf": "word/char_wb(2,4) min_df 3/5 sublinear; SGD log_loss 5ep best-dev; class-inverse weights",
                     "codet5": "frozen codet5-small mean-pool 512d; trunc 384+128; special tokens False; StandardScaler(train)+LR(balanced) C∈{.03,.1,.3,1} dev-select",
                     "centered": "LOO task-sibling mean, label-free; transductive diagnostic",
                     "bootstrap": {"repeats": BOOT, "cluster": "task_id", "seed": BOOT_SEED}},
        "counts": {"rows": len(rows), "train": int(tr.sum()), "dev": int(dv.sum()), "test": int(te.sum()),
                   "tasks": len(set(task.tolist())), "families": FAMILIES},
        "views": results, "gate_evaluation": gate,
        "test_read_utc": test_read_utc, "runtime_seconds": time.time() - t0,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.out / "predictions.npz", y_te=y[te].astype(np.int16),
                        task_id_te=np.array([task[i] for i in np.where(te)[0]]),
                        **{f"te__{k}": v for k, v in te_probs_store.items()})
    config = {"script": "scripts/stage_d_h1_authorbench_dcan.py", "seeds": SEEDS, "C_grid": C_GRID,
              "bootstrap": BOOT, "encoder_ckpt": "d-det/models/codet5-small",
              **metrics["protocol"]}
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "logs" / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps({"gate": {k: v for k, v in gate.items() if k.startswith(("cond1", "cond2", "verdict"))}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
