"""Stage D-N2: 候选矩阵上的最小 H1 复核（预注册于 n1_candidate_matrix/n2_protocol.json）。

- 候选子集 = 六族齐全 task（N1 manifest，剔除后矩阵；纯数据侧规则）；
- 在候选矩阵上按 P0 原配方整体复算（tfidf_char/word、sem_lr、style_lr/lgb、fusion、mean_ensemble）；
- 新视图：metadata_only、冻结 CodeT5-small meanpool、CodeT5-small task-centered（转导诊断）；
- 指标：macro-F1/BA/family recall/generator recall/500x task-cluster CI/ECE/NLL/confusion；
- 门（预注册）：cond1 内容>chance 且>metadata；cond2 best content ≥ P0_fusion+1pt；cond3 方向稳定（3 seeds）；
- test 单次读取（记录 UTC 时间）。

输出：d-det/artifacts/stage_d_data_construction_2026-10-07/n2_h1_recheck/
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "d-det/artifacts/stage_d_data_construction_2026-10-07"
OUT = BASE / "n2_h1_recheck"
N1 = BASE / "n1_candidate_matrix"
D1 = ROOT / "d-det/artifacts/stage_d_h1_authorbench_dcan_2026-10-07"
DATA = ROOT / "d-det/data/h2_authorbench_dcan/core.jsonl"
MODEL_SMALL = ROOT / "d-det/models/codet5-small"
MODEL_BASE = ROOT / "d-det/checkpoints/codet5-base"
FAMILIES = ["claude", "deepseek", "gemini", "llama", "openai", "qwen"]
NC = len(FAMILIES)
SEEDS = [0, 1, 2]
BOOT = 500
BOOT_SEED = 20261007

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "d-det/scripts"))
import acl_sota_p0_baselines as p0mod  # noqa: E402
from stage_d_h1_authorbench_dcan import (  # noqa: E402
    encode_codet5, full_metrics, cluster_boot, paired_boot, lr_select, centered_loo,
    macro_f1, C_GRID)
from stage_d_n0_diagnosis import sha256, norm_ws, norm_lex  # noqa: E402


def encode_codet5_base(texts, device_batch=8):
    """CodeT5-base 冻结均值池化（与 P0 round1 的 emb_dcan_ct5 同模型；协议=512、384+128、无特殊符、fp16）。"""
    import torch
    from transformers import RobertaTokenizer, T5EncoderModel
    torch.manual_seed(20261007)
    tok = RobertaTokenizer(vocab=str(MODEL_BASE / "vocab.json"), merges=str(MODEL_BASE / "merges.txt"),
                           unk_token="<unk>", bos_token="<s>", eos_token="</s>", sep_token="</s>",
                           cls_token="<s>", pad_token="<pad>", mask_token="<mask>", add_prefix_space=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5EncoderModel.from_pretrained(
        MODEL_BASE, local_files_only=True,
        dtype=(torch.float16 if device.type == "cuda" else torch.float32))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
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


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    protocol = json.loads((N1 / "n2_protocol.json").read_text(encoding="utf-8"))
    assert sha256(DATA.read_text(encoding="utf-8")) == protocol["inputs"]["core_sha256"]

    rows = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    manifest = [json.loads(l) for l in (N1 / "n1_manifest.jsonl").open(encoding="utf-8")]
    cand_tasks = sorted({m["task_id"] for m in manifest if m["family_coverage"] == 6})
    assert sha256("\n".join(cand_tasks)) == protocol["candidate"]["task_id_sha256"], "候选 task 集不一致"
    cand_set = set(cand_tasks)
    idx = [m["row_index"] for m in manifest if m["task_id"] in cand_set]
    split = np.array([rows[i]["task_split"] for i in idx])
    y = np.array([FAMILIES.index(rows[i]["family"]) for i in idx])
    gen = np.array([rows[i]["model_name"] for i in idx])
    task = np.array([rows[i]["task_id"] for i in idx])
    texts = [rows[i]["code"] for i in idx]
    tr, dv, te = split == "train", split == "dev", split == "test"
    counts = {"rows": len(idx), "train": int(tr.sum()), "dev": int(dv.sum()), "test": int(te.sum()),
              "tasks": len(set(task.tolist())), "test_tasks": int(len(set(task[te].tolist())))}
    assert counts["train"] == protocol["candidate"]["rows"]["train"]
    assert counts["dev"] == protocol["candidate"]["rows"]["dev"]
    assert counts["test"] == protocol["candidate"]["rows"]["test"]
    print(f"N2 candidate: {counts}", flush=True)
    log_lines = [f"N2 candidate: {counts}"]

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    # ---------- P0 复算（候选矩阵） ----------
    p0 = {}
    for tag, vec in (("tfidf_char", p0mod.tfidf_char()), ("tfidf_word", p0mod.tfidf_word())):
        Xtr = vec.fit_transform([t for t, s in zip(texts, tr) if s])
        Xdv = vec.transform([t for t, s in zip(texts, dv) if s])
        Xte = vec.transform([t for t, s in zip(texts, te) if s])
        dvs, tes, f1s = [], [], []
        for s in SEEDS:
            dvp, tep, bf1 = p0mod.sgd_snapshot(Xtr, y[tr], Xdv, y[dv], Xte, y[te], NC, s)
            dvs.append(dvp)
            tes.append(tep)
            f1s.append(float(bf1))
        p0[tag] = {"dv": np.mean(dvs, 0), "te": np.mean(tes, 0), "seed_te": tes,
                   "meta": {"vocab": int(Xtr.shape[1]), "seed_dev_f1": f1s}}
        log(f"P0 {tag}: vocab={Xtr.shape[1]} seeds dev F1 {[round(x, 4) for x in f1s]}")

    Str = np.array([p0mod.style_features(t) for t in texts], dtype=np.float32)
    dvp, tep, _ = p0mod.dense_lr(Str[tr], y[tr], Str[dv], Str[te], NC)
    p0["style_lr"] = {"dv": dvp, "te": tep, "meta": {"dim": int(Str.shape[1])}}
    dvp, tep, _ = p0mod.dense_lgb(Str[tr], y[tr], Str[dv], Str[te], NC)
    p0["style_lgb"] = {"dv": dvp, "te": tep, "meta": {"dim": int(Str.shape[1])}}
    log("P0 style_lr / style_lgb done")

    Fbase = encode_codet5_base(texts)
    dvp, tep, _ = p0mod.dense_lr(Fbase[tr].astype(np.float32), y[tr], Fbase[dv].astype(np.float32),
                                 Fbase[te].astype(np.float32), NC)
    p0["sem_lr"] = {"dv": dvp, "te": tep,
                    "meta": {"encoder": "codet5-base frozen mean-pool 768d（重编码子集，协议=512/384+128）"}}
    log(f"P0 sem_lr done (base emb {Fbase.shape})")

    fus_dv, fus_te, _, names = p0mod.fusion_dev_lr({k: v["dv"] for k, v in p0.items()},
                                                   {k: v["te"] for k, v in p0.items()}, y[dv], y[te], NC)
    p0["fusion_lr"] = {"dv": fus_dv, "te": fus_te, "meta": {"members": names}}
    mean_te = np.mean([v["te"] for v in p0.values() if v is not p0["fusion_lr"]], 0)
    mean_dv = np.mean([v["dv"] for v in p0.values() if v is not p0["fusion_lr"]], 0)
    p0["mean_ensemble"] = {"dv": mean_dv, "te": mean_te, "meta": {"members": names}}
    log(f"P0 fusion done (members {names})")

    # ---------- 新视图 ----------
    views = dict(p0)
    raws = [rows[i] for i in idx]
    Fm = np.column_stack([np.log1p(np.array([r["char_count"] for r in raws], float)),
                          np.array([r["num_lines"] for r in raws], float),
                          np.array([r["nloc"] for r in raws], float),
                          np.array([r["cyclomatic_complexity"] for r in raws], float),
                          np.array([r["token_size"] for r in raws], float)])
    clf, sc, C, devf1 = lr_select(Fm[tr], y[tr], Fm[dv], y[dv])
    views["metadata_only"] = {"dv": clf.predict_proba(sc.transform(Fm[dv])), "te": clf.predict_proba(sc.transform(Fm[te])),
                              "meta": {"selected_C": C, "dev_macro_f1": devf1}}
    log(f"metadata_only dev {devf1:.4f}")

    emb_small = encode_codet5(texts)
    clf, sc, C, devf1 = lr_select(emb_small[tr], y[tr], emb_small[dv], y[dv])
    views["codet5_small_meanpool"] = {"dv": clf.predict_proba(sc.transform(emb_small[dv])),
                                      "te": clf.predict_proba(sc.transform(emb_small[te])),
                                      "meta": {"selected_C": C, "dev_macro_f1": devf1}}
    log(f"codet5_small_meanpool dev {devf1:.4f}")

    zc, n_single = centered_loo(emb_small, task)
    clf, sc, C, devf1 = lr_select(zc[tr], y[tr], zc[dv], y[dv])
    views["codet5_small_centered"] = {"dv": clf.predict_proba(sc.transform(zc[dv])),
                                      "te": clf.predict_proba(sc.transform(zc[te])),
                                      "meta": {"selected_C": C, "dev_macro_f1": devf1, "transductive": True,
                                               "singleton_tasks": n_single}}
    clf2, _, C2, devf12 = lr_select(zc[tr], y[tr], zc[dv], y[dv], standardize=False)
    views["codet5_small_centered_unscaled"] = {"dv": clf2.predict_proba(zc[dv]), "te": clf2.predict_proba(zc[te]),
                                               "meta": {"selected_C": C2, "dev_macro_f1": devf12, "transductive": True}}
    log(f"codet5_small_centered dev {devf1:.4f} | unscaled {devf12:.4f}")

    # ---------- 指标 ----------
    test_read_utc = datetime.now(timezone.utc).isoformat()
    results = {}
    te_probs_store = {}
    for name, v in views.items():
        p_te = np.asarray(v["te"], np.float64)
        m = full_metrics(p_te, y[te])
        ci = cluster_boot(p_te, y[te], task[te], repeats=BOOT, seed=BOOT_SEED)
        res = {"test": m, "ci95": ci, "meta": v.get("meta", {})}
        if "fusion_lr" in views and name != "fusion_lr":
            res["vs_p0_fusion"] = paired_boot(p_te, np.asarray(views["fusion_lr"]["te"], np.float64),
                                              y[te], task[te], repeats=BOOT, seed=BOOT_SEED)
        if name != "tfidf_word":
            res["vs_p0_word"] = paired_boot(p_te, np.asarray(views["tfidf_word"]["te"], np.float64),
                                            y[te], task[te], repeats=BOOT, seed=BOOT_SEED)
        gacc = {}
        for g in sorted(set(gen[te].tolist())):
            m2 = gen[te] == g
            gacc[g] = {"n": int(m2.sum()), "accuracy": float((p_te[m2].argmax(1) == y[te][m2]).mean())}
        res["generator_recall"] = gacc
        results[name] = res
        te_probs_store[name] = p_te.astype(np.float16)
        log(f"[test] {name:28s} f1={m['macro_f1']:.4f} ba={m['balanced_acc']:.4f} "
            f"ci=[{ci['ci95_low']:.4f},{ci['ci95_high']:.4f}]")

    # ---------- 门（预注册） ----------
    p0_f = results["fusion_lr"]["test"]["macro_f1"]
    content_keys = ["tfidf_word", "tfidf_char", "codet5_small_meanpool", "codet5_small_centered",
                    "codet5_small_centered_unscaled"]
    best_c = max(content_keys, key=lambda k: results[k]["test"]["macro_f1"])
    best_f1 = results[best_c]["test"]["macro_f1"]
    meta_f1 = results["metadata_only"]["test"]["macro_f1"]
    # 真实 per-seed test F1
    seed_test_f1 = {}
    for tag in ("tfidf_word", "tfidf_char"):
        seed_test_f1[tag] = []
        for s_te in views[tag]["seed_te"]:
            import sklearn.metrics as sm
            seed_test_f1[tag].append(float(sm.f1_score(y[te], s_te.argmax(1), average="macro", zero_division=0)))
    tf_tag = max(seed_test_f1, key=lambda k: float(np.mean(seed_test_f1[k])))
    deltas = [f - p0_f for f in seed_test_f1[tf_tag]]
    cond3 = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
    gate = {
        "candidate": {"rows": counts["rows"], "test_rows": counts["test"], "test_tasks": counts["test_tasks"]},
        "p0_fusion_subset": p0_f,
        "d1_full_p0_fusion_ref": protocol["comparators_for_context"]["d1_full_p0_fusion"],
        "best_content_view": best_c, "best_content_f1": best_f1,
        "metadata_only_f1": meta_f1, "chance": 1.0 / NC,
        "cond1_content_above_chance_and_metadata": bool(best_f1 > meta_f1 and best_f1 > 1.0 / NC),
        "cond2_plus_1pt_over_p0": bool(best_f1 - p0_f >= 0.01),
        "cond3_direction_stable": {"tfidf_tag": tf_tag, "seed_test_f1": seed_test_f1[tf_tag],
                                   "seed_deltas_vs_p0": deltas, "pass": bool(cond3)},
        "verdict": ("N2 通过：数据构造（组成均匀化）后内容信号达到预注册门槛 → 可重新讨论 H2。"
                    if (best_f1 > meta_f1 and best_f1 > 1.0 / NC and best_f1 - p0_f >= 0.01 and cond3)
                    else "N2 未通过：组成均匀化后内容信号仍未达门槛 → 归档为数据限制/负结果。"),
    }
    metrics = {
        "schema": "stage_d_n2_h1_recheck_v1",
        "protocol": {"candidate": protocol["candidate"]["definition"],
                     "data_sha256": sha256(DATA.read_text(encoding="utf-8")),
                     "bootstrap": {"repeats": BOOT, "cluster": "task_id", "seed": BOOT_SEED},
                     "seeds": SEEDS, "C_grid": list(C_GRID)},
        "counts": counts, "views": results, "gate": gate,
        "test_read_utc": test_read_utc, "runtime_seconds": time.time() - t0,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(OUT / "predictions.npz", y_te=y[te].astype(np.int16),
                        task_id_te=np.array(task[te]), generator_te=np.array(gen[te]),
                        **{f"te__{k}": v for k, v in te_probs_store.items()})
    config = {"script": "scripts/stage_d_n2_h1_recheck.py", "seeds": SEEDS,
              "encoder_small": str(MODEL_SMALL.relative_to(ROOT)), "encoder_base": str(MODEL_BASE.relative_to(ROOT)),
              "bootstrap": BOOT, **metrics["protocol"]}
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---------- report.md ----------
    L = []
    L.append("# N2：候选矩阵上的最小 H1 复核（2026-10-07）")
    L.append("")
    L.append(f"- 候选 = {protocol['candidate']['definition']}；{counts['rows']} 行（train/dev/test="
             f"{counts['train']}/{counts['dev']}/{counts['test']}）；test {counts['test_tasks']} tasks")
    L.append(f"- test 读取（UTC）：{test_read_utc}（单次）；500× task-cluster bootstrap；运行 {time.time() - t0:.1f}s")
    L.append("- P0 在候选矩阵上整体复算（非复用 D1 预测）；原 D1 数字仅作并列参照。")
    L.append("")
    L.append("## 0 门（预注册）")
    L.append("")
    g2 = gate["cond3_direction_stable"]
    L.append(f"- cond1 内容>chance+metadata：{'✅' if gate['cond1_content_above_chance_and_metadata'] else '❌'}"
             f"（best {best_f1:.4f} vs metadata {meta_f1:.4f}，chance {1.0 / NC:.4f}）")
    L.append(f"- cond2 ≥ P0_fusion(候选)+1pt：{'✅' if gate['cond2_plus_1pt_over_p0'] else '❌'}"
             f"（best {best_f1:.4f} vs P0 {p0_f:.4f}；差 {(best_f1 - p0_f) * 100:+.2f}pt；"
             f"D1 全量参照 P0 {gate['d1_full_p0_fusion_ref']:.4f}）")
    L.append(f"- cond3 方向稳定（{g2['tfidf_tag']} 3 seeds Δ={[round(d, 4) for d in g2['seed_deltas_vs_p0']]}）："
             f"{'✅' if g2['pass'] else '❌'}")
    L.append("")
    L.append(f"**判定：{gate['verdict']}**")
    L.append("")
    L.append("## 1 视图总表（候选 test；n={}）".format(counts["test"]))
    L.append("")
    L.append("| 视图 | macro-F1 | BA | CI95 | ECE15 | NLL | Δ vs P0(候选) | Δ vs word(候选) |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    order = ["metadata_only", "tfidf_word", "tfidf_char", "codet5_small_meanpool",
             "codet5_small_centered", "codet5_small_centered_unscaled",
             "fusion_lr", "mean_ensemble", "style_lr", "style_lgb", "sem_lr"]
    for k in order:
        v = results[k]
        t, ci = v["test"], v["ci95"]
        d1 = v.get("vs_p0_fusion", {}).get("delta_mean", 0.0 if k == "fusion_lr" else float("nan"))
        d2 = v.get("vs_p0_word", {}).get("delta_mean", 0.0 if k == "tfidf_word" else float("nan"))
        L.append(f"| {k} | {t['macro_f1']:.4f} | {t['balanced_acc']:.4f} | [{ci['ci95_low']:.4f},{ci['ci95_high']:.4f}] "
                 f"| {t['ece15']:.4f} | {t['nll']:.4f} | {d1:+.4f} | {d2:+.4f} |")
    L.append("")
    L.append("## 2 family / generator 指标")
    L.append("")
    L.append("| family | support | " + " | ".join(f"{k} R" for k in ("tfidf_word", "codet5_small_meanpool", "fusion_lr")) + " |")
    L.append("| --- | --- | --- | --- | --- |")
    for f in FAMILIES:
        row = [f, str(results["tfidf_word"]["test"]["per_class"][f]["support"])]
        for k in ("tfidf_word", "codet5_small_meanpool", "fusion_lr"):
            row.append(f"{results[k]['test']['per_class'][f]['recall']:.3f}")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("| generator | n | tfidf_word acc | codet5_small_meanpool acc | fusion acc |")
    L.append("| --- | --- | --- | --- | --- |")
    for gname in sorted(set(gen[te].tolist())):
        row = [gname, results["tfidf_word"]["generator_recall"][gname]["n"]]
        for k in ("tfidf_word", "codet5_small_meanpool", "fusion_lr"):
            row.append(f"{results[k]['generator_recall'][gname]['accuracy']:.4f}")
        L.append("| " + " | ".join(str(x) for x in row) + " |")
    L.append("")
    L.append("## 3 与 D1 全量并列")
    L.append("")
    L.append(f"- D1 全量：tfidf_word 0.8200 / P0 fusion 0.8388（deliverable 目录见 stage_d_h1_authorbench_dcan_2026-10-07）")
    L.append(f"- N2 候选：best content {best_f1:.4f}（{best_c}）/ P0 fusion 0.8388→{p0_f:.4f}（候选复算）")
    L.append("")
    L.append("## 4 confusion matrix（best content view = {})".format(best_c))
    L.append("")
    cm = results[best_c]["test"]["confusion_matrix"]
    L.append("| true\\pred | " + " | ".join(FAMILIES) + " |")
    L.append("|" + "|".join(["---"] * (NC + 1)) + "|")
    for i, f in enumerate(FAMILIES):
        L.append(f"| {f} | " + " | ".join(str(x) for x in cm[i]) + " |")
    L.append("")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "logs" / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps({"gate": {"cond1": gate["cond1_content_above_chance_and_metadata"],
                               "cond2": gate["cond2_plus_1pt_over_p0"],
                               "cond3": gate["cond3_direction_stable"]["pass"],
                               "verdict": gate["verdict"]}, "best": best_c,
                      "best_f1": best_f1, "p0": p0_f}, ensure_ascii=False))


if __name__ == "__main__":
    main()
