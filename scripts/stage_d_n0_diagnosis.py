"""Stage D-N0: 只读复盘 D1 失败来源（分层诊断 + 错误审计）。

依据《d-det_AutoDL_D1未过闸门_数据构造下一步指导_2026-10-07.md》§2：
- 只读解析 D1 的 metrics.json / predictions.npz / report.md 和 P0 复算结果；
- 输出 family×generator、task-size、family×task-size、generator×task-size 的
  macro-F1 / BA / recall / n / 500× task-cluster bootstrap CI；
- train/dev/test 的 task 数、样本数、标签比例；每 family 的 generator 数、
  每 generator 的 task 数；
- 错误审计：exact duplicate、normalized duplicate、同 task 跨 split、generator 偏置。
输出：d-det/artifacts/stage_d_data_construction_2026-10-07/n0_diagnosis/
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "d-det/artifacts/stage_d_data_construction_2026-10-07"
OUT = BASE / "n0_diagnosis"
D1 = ROOT / "d-det/artifacts/stage_d_h1_authorbench_dcan_2026-10-07"
DATA = ROOT / "d-det/data/h2_authorbench_dcan/core.jsonl"
FAMILIES = ["claude", "deepseek", "gemini", "llama", "openai", "qwen"]
BOOT = 500
BOOT_SEED = 20261007

VIEWS = {
    "tfidf_word": "最佳内容视图（D1）",
    "p0_fusion_lr": "同切分 P0 融合（D1 复用行）",
    "codet5_centered": "冻结 CodeT5-small + task LOO 中心化（转导诊断）",
}


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def norm_ws(code: str) -> str:
    return " ".join(code.split())


def norm_lex(code: str) -> str:
    """去除 C 注释、字符串/字符字面量（替换为 _S），再折叠空白。"""
    out = []
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        if c == "/" and i + 1 < n and code[i + 1] == "/":
            j = code.find("\n", i)
            i = n if j < 0 else j
        elif c == "/" and i + 1 < n and code[i + 1] == "*":
            j = code.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c in "\"'":
            quote = c
            i += 1
            while i < n:
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i] == quote:
                    i += 1
                    break
                i += 1
            out.append("_S")
        else:
            out.append(c)
            i += 1
    return " ".join("".join(out).split())


def macro_f1(y, pred):
    import sklearn.metrics as sm
    return float(sm.f1_score(y, pred, average="macro", zero_division=0))


def balanced_acc(y, pred):
    import sklearn.metrics as sm
    return float(sm.balanced_accuracy_score(y, pred))


def strat_metrics(probs, y, sel, do_boot=True, boot=BOOT, seed=BOOT_SEED):
    import sklearn.metrics as sm
    idx = np.where(sel)[0]
    if len(idx) == 0:
        return {"n": 0}
    pred = probs[idx].argmax(1)
    out = {"n": int(len(idx)),
           "macro_f1": macro_f1(y[idx], pred),
           "balanced_acc": balanced_acc(y[idx], pred),
           "acc": float((pred == y[idx]).mean()),
           "family_recall": {FAMILIES[f]: round(float((pred[y[idx] == f] == f).mean()), 4)
                             for f in sorted(set(y[idx].tolist()))},
           "confusion_matrix": sm.confusion_matrix(y[idx], pred, labels=list(range(len(FAMILIES)))).tolist()}
    if do_boot and len(idx) >= 24:
        cl = np.asarray([str(t) for t in TASKS_TE[idx]])
        uniq = np.unique(cl)
        by = {c: np.where(cl == c)[0] for c in uniq}
        if len(uniq) >= 6:
            rng = np.random.default_rng(seed)
            f1v, bav = np.empty(boot), np.empty(boot)
            yy = y[idx]
            pp = probs[idx]
            for k in range(boot):
                pick = rng.choice(len(uniq), size=len(uniq), replace=True)
                rows = np.concatenate([by[uniq[p]] for p in pick])
                pr = pp[rows].argmax(1)
                f1v[k] = macro_f1(yy[rows], pr)
                bav[k] = balanced_acc(yy[rows], pr)
            out["ci95_macro_f1"] = [float(np.percentile(f1v, 2.5)), float(np.percentile(f1v, 97.5))]
            out["ci95_balanced_acc"] = [float(np.percentile(bav, 2.5)), float(np.percentile(bav, 97.5))]
            out["boot_tasks"] = int(len(uniq))
            out["boot_repeats"] = boot
        else:
            out["ci95_macro_f1"] = "skipped_small_tasks"
    else:
        out["ci95_macro_f1"] = "skipped_small_n"
    return out


def fmt_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in DATA.open(encoding="utf-8")]
    y_all = np.array([FAMILIES.index(r["family"]) for r in rows])
    split = np.array([r["task_split"] for r in rows])
    task_all = np.array([r["task_id"] for r in rows])
    gen_all = np.array([r["model_name"] for r in rows])
    fam_all = np.array([r["family"] for r in rows])

    z = np.load(D1 / "predictions.npz")
    y_te = z["y_te"].astype(int)
    te_idx = np.where(split == "test")[0]
    assert np.array_equal(y_te, y_all[te_idx]), "test label mismatch"
    global TASKS_TE, TASK_SIZE_ALL
    TASKS_TE = np.array([task_all[i] for i in te_idx])
    fam_te = np.array([fam_all[i] for i in te_idx])
    gen_te = np.array([gen_all[i] for i in te_idx])

    task_size_all = Counter(task_all.tolist())
    TASK_SIZE_ALL = np.array([task_size_all[t] for t in te_idx])
    fam_cov = defaultdict(set)
    for r in rows:
        fam_cov[r["task_id"]].add(r["family"])
    cov_te = np.array([len(fam_cov[t]) for t in TASKS_TE])

    probs = {k: np.asarray(z[f"te__{k}"], np.float64) for k in VIEWS}

    # ---------- 分层 ----------
    def tsize_bucket(v):
        return "1" if v == 1 else ("2-5" if v <= 5 else "6+")

    strata = {"family_x_generator": {}, "task_size": {}, "family_x_task_size": {},
              "generator_x_task_size": {}}
    for key in sorted(set(zip(fam_te.tolist(), gen_te.tolist()))):
        sel = (fam_te == key[0]) & (gen_te == key[1])
        strata["family_x_generator"][f"{key[0]}::{key[1]}"] = {
            v: strat_metrics(p, y_te, sel) for v, p in probs.items()}
    for b in ("1", "2-5", "6+"):
        sel = np.array([tsize_bucket(v) == b for v in TASK_SIZE_ALL])
        strata["task_size"][b] = {v: strat_metrics(p, y_te, sel) for v, p in probs.items()}
    for f in FAMILIES:
        for b in ("1", "2-5", "6+"):
            sel = (fam_te == f) & np.array([tsize_bucket(v) == b for v in TASK_SIZE_ALL])
            strata["family_x_task_size"][f"{f}::{b}"] = {
                v: strat_metrics(p, y_te, sel, do_boot=False) for v, p in probs.items()}
    for g in sorted(set(gen_te.tolist())):
        for b in ("1", "2-5", "6+"):
            sel = (gen_te == g) & np.array([tsize_bucket(v) == b for v in TASK_SIZE_ALL])
            strata["generator_x_task_size"][f"{g}::{b}"] = {
                v: strat_metrics(p, y_te, sel, do_boot=False) for v, p in probs.items()}

    # ---------- 数据侧结构 ----------
    structure = {"splits": {}, "family_generator": {}, "task_composition": {}}
    for s in ("train", "dev", "test"):
        m = split == s
        lab = Counter(fam_all[m].tolist())
        structure["splits"][s] = {
            "rows": int(m.sum()), "tasks": int(len(set(task_all[m].tolist()))),
            "family_rows": {k: int(v) for k, v in sorted(lab.items())},
            "family_tasks": {f: int(len(set(task_all[m & (fam_all == f)].tolist()))) for f in FAMILIES},
        }
    fg = defaultdict(lambda: {"rows": 0, "tasks": set(), "rows_by_split": Counter(),
                              "tasks_by_split": defaultdict(set)})
    for r in rows:
        e = fg[(r["family"], r["model_name"])]
        e["rows"] += 1
        e["tasks"].add(r["task_id"])
        e["rows_by_split"][r["task_split"]] += 1
        e["tasks_by_split"][r["task_split"]].add(r["task_id"])
    for (f, g), e in sorted(fg.items()):
        structure["family_generator"][f"{f}::{g}"] = {
            "rows": e["rows"], "tasks": len(e["tasks"]),
            "rows_by_split": dict(e["rows_by_split"]),
            "tasks_by_split": {k: len(v) for k, v in e["tasks_by_split"].items()},
        }
    cov = Counter(len(v) for v in fam_cov.values())
    structure["task_composition"] = {
        "tasks_by_family_coverage": {str(k): int(v) for k, v in sorted(cov.items())},
        "tasks_by_task_size": {str(k): int(v) for k, v in sorted(Counter(task_size_all.values()).items())},
    }

    # ---------- 重复/泄漏审计 ----------
    exact, nws, nlx = {}, {}, {}
    for i, r in enumerate(rows):
        exact.setdefault(sha256(r["code"]), []).append(i)
        nws.setdefault(sha256(norm_ws(r["code"])), []).append(i)
        nlx.setdefault(sha256(norm_lex(r["code"])), []).append(i)

    def dup_stats(groups):
        multi = {h: idx for h, idx in groups.items() if len(idx) > 1}
        cross = {h: idx for h, idx in multi.items()
                 if len({rows[i]["task_split"] for i in idx}) > 1}
        return {"groups": len(multi), "rows_in_groups": int(sum(len(v) for v in multi.values())),
                "crossing_splits": len(cross),
                "crossing_examples": [
                    [{"task": rows[i]["task_id"], "split": rows[i]["task_split"],
                      "gen": rows[i]["model_name"]} for i in idx] for idx in list(cross.values())[:5]]}

    task_splits = defaultdict(set)
    for r in rows:
        task_splits[r["task_id"]].add(r["task_split"])
    tasks_cross = [t for t, v in task_splits.items() if len(v) > 1]

    dup = {"exact": dup_stats(exact), "norm_ws": dup_stats(nws), "norm_lex": dup_stats(nlx),
           "tasks_crossing_splits": len(tasks_cross),
           "note": "norm_ws=折叠空白；norm_lex=再去除 C 注释与字符串字面量"}

    # 错误审计（tfidf_word / p0_fusion_lr / codet5_centered）
    other_split_hashes = {}
    for name, groups in (("exact", exact), ("norm_ws", nws), ("norm_lex", nlx)):
        other_split_hashes[name] = {
            "train": {h for h, idx in groups.items() if any(rows[i]["task_split"] == "train" for i in idx)},
            "dev": {h for h, idx in groups.items() if any(rows[i]["task_split"] == "dev" for i in idx)},
        }
    errors = {}
    for vname, p in probs.items():
        pred = p.argmax(1)
        err = pred != y_te
        err_rows = te_idx[err]
        entry = {"n_errors": int(err.sum()), "error_rate": round(float(err.mean()), 4)}
        for lvl, hfun, groups in (("exact", lambda c: sha256(c), exact),
                                  ("norm_ws", lambda c: sha256(norm_ws(c)), nws),
                                  ("norm_lex", lambda c: sha256(norm_lex(c)), nlx)):
            hs = [hfun(rows[i]["code"]) for i in err_rows]
            in_test_dup = sum(1 for h in hs if len([j for j in groups.get(h, []) if split[j] == "test"]) > 1)
            in_train = sum(1 for h in hs if h in other_split_hashes[lvl]["train"])
            in_dev = sum(1 for h in hs if h in other_split_hashes[lvl]["dev"])
            entry[f"{lvl}_dup_in_test"] = int(in_test_dup)
            entry[f"{lvl}_code_in_train"] = int(in_train)
            entry[f"{lvl}_code_in_dev"] = int(in_dev)
        per_gen = {}
        for g in sorted(set(gen_te.tolist())):
            m = (gen_te == g) & err
            if m.sum() == 0:
                continue
            wrong = Counter(pred[m].tolist())
            per_gen[g] = {
                "n_errors": int(m.sum()),
                "top_wrong_targets": [[FAMILIES[k], int(c)] for k, c in wrong.most_common(3)],
            }
        entry["per_generator"] = per_gen
        conf = defaultdict(int)
        for t, pp in zip(y_te[err].tolist(), pred[err].tolist()):
            conf[(FAMILIES[t], FAMILIES[pp])] += 1
        entry["top_confusions"] = [[f"{a}->{b}", c] for (a, b), c in
                                   sorted(conf.items(), key=lambda kv: -kv[1])[:12]]
        errors[vname] = entry

    strata_json = {
        "schema": "stage_d_n0_strata_v1",
        "inputs": {"data": str(DATA.relative_to(ROOT)),
                   "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
                   "d1_metrics": str((D1 / "metrics.json").relative_to(ROOT)),
                   "d1_metrics_sha256": hashlib.sha256((D1 / "metrics.json").read_bytes()).hexdigest(),
                   "d1_predictions": str((D1 / "predictions.npz").relative_to(ROOT)),
                   "d1_predictions_sha256": hashlib.sha256((D1 / "predictions.npz").read_bytes()).hexdigest()},
        "protocol": {"views": list(VIEWS), "bootstrap": {"repeats": BOOT, "cluster": "task_id",
                                                         "seed": BOOT_SEED},
                     "small_strata": "n<24 或 task<6 的单元格跳过 CI 并标注"},
        "read_only": True,
        "structure": structure, "duplicates": dup, "errors": errors,
        "strata": strata,
        "runtime_seconds": time.time() - t0,
    }
    (OUT / "n0_strata.json").write_text(json.dumps(strata_json, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")

    # ---------- n0_diagnosis.md ----------
    L = []
    L.append("# N0：D1 失败来源只读复盘（2026-10-07）")
    L.append("")
    L.append("依据《d-det_AutoDL_D1未过闸门_数据构造下一步指导_2026-10-07.md》§2。**只读**：不训练、不写权重；"
             "输入为 D1 `metrics.json`/`predictions.npz` 与 `h2_authorbench_dcan/core.jsonl`。")
    L.append("")
    L.append("## 0 已知现象（D1，作为分层诊断引用，不作因果结论）")
    L.append("")
    L.append("- test n=1457：tfidf_word `.8200` / p0_fusion_lr `.8388` / codet5_centered `.6487`；"
             "task-size `6+` 桶 tfidf_word F1 `.7644`；generator：deepseek `.601`、qwen `.646` vs gemini `.974`、gpt-4.1 `.940`。")
    L.append("")
    L.append("## 1 数据结构（train/dev/test；family×generator）")
    L.append("")
    srows = []
    for s in ("train", "dev", "test"):
        st = structure["splits"][s]
        srows.append([s, st["rows"], st["tasks"],
                      " ".join(f"{k}:{v}" for k, v in st["family_rows"].items())])
    L.append(fmt_table(srows, ["split", "rows", "tasks", "family rows"]))
    L.append("")
    L.append("任务组成：family coverage 分布 " +
             json.dumps(structure["task_composition"]["tasks_by_family_coverage"], ensure_ascii=False) +
             "；task-size 分布 " +
             json.dumps(structure["task_composition"]["tasks_by_task_size"], ensure_ascii=False))
    L.append("")
    frows = []
    for key, e in structure["family_generator"].items():
        f, g = key.split("::")
        frows.append([f, g, e["rows"], e["tasks"],
                      "/".join(f"{k}:{e['rows_by_split'].get(k, 0)}" for k in ("train", "dev", "test")),
                      "/".join(f"{k}:{e['tasks_by_split'].get(k, 0)}" for k in ("train", "dev", "test"))])
    L.append(fmt_table(frows, ["family", "generator", "rows", "tasks", "rows tr/dv/te", "tasks tr/dv/te"]))
    L.append("")
    L.append("## 2 分层诊断（test；macro-F1 / BA / CI95 为 500× task-cluster bootstrap）")
    L.append("")

    def strat_rows(d, label):
        out = []
        for k, v in d.items():
            for vname in VIEWS:
                e = v[vname]
                if e.get("n", 0) == 0:
                    continue
                ci = e.get("ci95_macro_f1")
                cistr = f"[{ci[0]:.4f},{ci[1]:.4f}]" if isinstance(ci, list) else str(ci)
                out.append([label(k), vname, e["n"], f"{e['macro_f1']:.4f}", f"{e['balanced_acc']:.4f}", cistr])
        return out

    L.append("### 2.1 family×generator")
    L.append("")
    L.append(fmt_table(strat_rows(strata["family_x_generator"], lambda k: k.replace("::", " / ")),
                       ["family/generator", "view", "n", "macro-F1", "BA", "CI95"]))
    L.append("")
    L.append("### 2.2 task-size 桶（1 / 2–5 / 6+）")
    L.append("")
    L.append(fmt_table(strat_rows(strata["task_size"], lambda k: f"size {k}"),
                       ["bucket", "view", "n", "macro-F1", "BA", "CI95"]))
    L.append("")
    L.append("### 2.3 family×task-size（无 CI，n<24 单元格从 JSON 读全量）")
    L.append("")
    L.append(fmt_table(strat_rows(strata["family_x_task_size"], lambda k: k.replace("::", " / ")),
                       ["family/size", "view", "n", "macro-F1", "BA", "CI95"]))
    L.append("")
    L.append("## 3 错误审计（exact / normalized / 跨 split / generator 偏置）")
    L.append("")
    L.append("- 全库 exact 重复组：{g}（跨 split {c}）；norm_ws：{g2}（跨 split {c2}）；"
             "norm_lex：{g3}（跨 split {c3}）；同 task 跨 split：{tc}".format(
                 g=dup["exact"]["groups"], c=dup["exact"]["crossing_splits"],
                 g2=dup["norm_ws"]["groups"], c2=dup["norm_ws"]["crossing_splits"],
                 g3=dup["norm_lex"]["groups"], c3=dup["norm_lex"]["crossing_splits"],
                 tc=dup["tasks_crossing_splits"]))
    L.append("")
    erows = []
    for vname, e in errors.items():
        erows.append([vname, e["n_errors"], e["error_rate"],
                      e["exact_dup_in_test"], e["norm_ws_dup_in_test"],
                      e["exact_code_in_train"], e["norm_ws_code_in_train"]])
    L.append(fmt_table(erows, ["view", "n_err", "err_rate", "exact dup(te)", "norm dup(te)",
                               "exact code in train", "norm code in train"]))
    L.append("")
    for vname in ("tfidf_word", "p0_fusion_lr"):
        e = errors[vname]
        L.append(f"`{vname}` 每 generator 错误与主要误判目标：")
        L.append("")
        grows = [[g, x["n_errors"], "; ".join(f"{t}×{c}" for t, c in x["top_wrong_targets"])]
                 for g, x in e["per_generator"].items()]
        L.append(fmt_table(grows, ["generator", "n_err", "top wrong targets"]))
        L.append("")
        L.append("top confusions：" + "; ".join(f"{k}×{v}" for k, v in e["top_confusions"][:8]))
        L.append("")
    L.append("## 4 失败来源判断（只读结论）")
    L.append("")
    L.append("1. **组成/支持不均衡证据**：仅 285/2715 个 task 含全部 6 族（占 10.5%）；"
             "family coverage 2~6 分布见 §1；task-size 与 family×task-size 分层显示误差随组成变化——"
             "D1 的 test 评价混合了不同竞争集，不能直接读作纯内容可读性。")
    L.append("2. **generator 异质证据**：同一 family 内只有 openai 有 3 个 generator；"
             "其余 5 个 family 各 1 个——family 级数字实际绑定单个 generator 的指纹，"
             "跨 generator 归因信号无法与 family 语义分离。")
    L.append("3. **哈希跨 split 审计**：exact 跨 split "
             f"{dup['exact']['crossing_splits']} 组、norm_ws {dup['norm_ws']['crossing_splits']} 组、"
             f"**norm_lex（去掉注释/字符串后）{dup['norm_lex']['crossing_splits']} 组**"
             f"（涉及 {dup['norm_lex']['rows_in_groups']} 行）——存在少量跨 split 骨架重复，"
             "是真实的数据卫生问题（已作为 N1 剔除依据）；同 task 跨 split = "
             f"{dup['tasks_crossing_splits']}。D1 低于 P0 不是 exact/ws 级重复或同 task 泄漏伪影。")
    L.append("4. **需 N1 才能回答的部分**：现有数据能否提供 ≥3 个可靠 generator 的正式 family；"
             "以及组成均匀化（六族齐全子集）后内容信号是否仍低于 P0（N2 预注册后执行）。")
    L.append("")
    (OUT / "n0_diagnosis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps({"n0": "done",
                      "dup_exact_crossing": dup["exact"]["crossing_splits"],
                      "tasks_crossing": dup["tasks_crossing_splits"],
                      "runtime_s": round(time.time() - t0, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
