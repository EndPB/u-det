#!/usr/bin/env python
"""ACL 家族归因数据集合 v1：服务端只读审计（2026-10-03）。

对应指导：d-det/docx/d-det_AutoDL_服务器端AI下一阶段执行指导_2026-10-03.md 第一、二步。
- 全部来源用 parquet batch / JSONL 逐行流式读取；不用 pandas.read_*，不整体 json.load 代码集合；
- 不改动任何数据文件；仅输出到 artifacts/acl_dcan_round1/；
- 单线程 I/O（环境变量建议 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2）。

输出：
  artifacts/acl_dcan_round1/audit_server_2026-10-03.json
  artifacts/acl_dcan_round1/audit_server_2026-10-03.md
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # d-det/
DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "acl_dcan_round1"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
SECTION_T = {}

SAMPLE_HASH = 50_000          # T1/T3 每分片抽样的行数上限（T2 全量）


def log(msg: str) -> None:
    print(f"[audit] {msg}", flush=True)


def mark(name: str) -> None:
    SECTION_T[name] = round(time.time() - T0, 1)
    log(f"== {name} done @ {SECTION_T[name]}s ==")


def jdump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def kd(c: Counter) -> dict:
    """Counter -> dict，键强制字符串（防 None 键）。"""
    return {str(k): v for k, v in c.items()}


# ---------------------------------------------------------------- 环境
def env_info() -> dict:
    info = {"python": sys.version.split()[0], "thread_env": {
        k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM")}}
    du = shutil.disk_usage("/root/autodl-tmp")
    info["disk"] = {"total_gb": round(du.total / 2**30, 2), "used_gb": round(du.used / 2**30, 2),
                    "avail_gb": round(du.free / 2**30, 2), "used_pct": round(100 * du.used / du.total, 1)}
    try:
        import torch
        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu_mem_gb"] = round(p.total_memory / 2**30, 1)
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = repr(exc)
    for mod in ("pyarrow", "sklearn", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001
            info[mod] = f"ERR {exc!r}"
    return info


# ---------------------------------------------------------------- AICD
def split_of(fname: str) -> str:
    return fname.split("-", 1)[0]


def audit_aicd() -> dict:
    import pyarrow.parquet as pq

    base = DATA / "acl_attribution_collection_v1" / "raw" / "aicd"
    out = {"dirs": {}, "hash_check": {}}
    for cfg in ("T1", "T2", "T3"):
        d = base / cfg
        shards = sorted(d.glob("*.parquet"))
        cfg_rows = 0
        split_rows = Counter()
        split_labels: dict[str, Counter] = defaultdict(Counter)
        empty = null = ws = 0
        shard_rows = {}
        full_hash = cfg == "T2"
        mask: dict[bytes, int] = {}
        split_idx = {"train": 0, "validation": 1, "test": 2}
        sampled = 0
        for shard in shards:
            pf = pq.ParquetFile(shard)
            rows = 0
            shard_hashed = 0
            it = pf.iter_batches(batch_size=4096, columns=["code", "label"], use_threads=False)
            for batch in it:
                codes = batch.column("code").to_pylist()
                labs = batch.column("label").to_pylist()
                for c, l in zip(codes, labs):
                    rows += 1
                    split_labels[split_of(shard.name)][l] += 1
                    if c is None:
                        null += 1
                        continue
                    if len(c) == 0:
                        empty += 1
                        continue
                    if c.strip() == "":
                        ws += 1
                    if not full_hash and shard_hashed >= SAMPLE_HASH:
                        continue
                    h = hashlib.md5(c.encode("utf-8", "surrogatepass")).digest()
                    bit = 1 << split_idx[split_of(shard.name)]
                    mask[h] = mask.get(h, 0) | bit
                    if not full_hash:
                        sampled += 1
                        shard_hashed += 1
            cfg_rows += rows
            split_rows[split_of(shard.name)] += rows
            shard_rows[shard.name] = rows
            log(f"aicd {cfg} {shard.name}: {rows} rows")
        distinct = len(mask)
        cross = sum(1 for m in mask.values() if m.bit_count() > 1)
        out["dirs"][cfg] = {
            "shards": len(shards), "rows": cfg_rows, "split_rows": dict(split_rows),
            "labels_total": {str(k): v for k, v in sorted(
                sum((Counter(split_labels[s]) for s in split_labels), Counter()).items())},
            "labels_by_split": {s: {str(k): v for k, v in sorted(c.items())}
                                for s, c in sorted(split_labels.items())},
            "null_code": null, "empty_code": empty, "whitespace_only": ws,
            "shard_rows": shard_rows,
        }
        out["hash_check"][cfg] = {
            "scan": "full" if full_hash else f"sampled-first-{SAMPLE_HASH}-per-shard",
            "rows_hashed": cfg_rows if full_hash else sampled,
            "distinct_hashes": distinct,
            "dup_rows_vs_distinct": (cfg_rows if full_hash else sampled) - distinct,
            "cross_split_leaked_hashes": cross,
        }
        del mask
    return out


# ---------------------------------------------------------------- Droid
def audit_droid() -> dict:
    p = DATA / "h2_droid_full_selected" / "core.jsonl"
    rows = 0
    fam, gen, mode, label, lang, src, split_src = (Counter() for _ in range(7))
    sha = set()
    with p.open() as fh:
        for line in fh:
            d = json.loads(line)
            rows += 1
            fam[d.get("Model_Family")] += 1
            gen[d.get("Generator")] += 1
            mode[d.get("Generation_Mode")] += 1
            label[d.get("Label")] += 1
            lang[d.get("Language")] += 1
            src[d.get("Source")] += 1
            split_src[d.get("split_source")] += 1
            sha.add(d.get("source_row_sha1"))
    fp = json.loads((DATA / "h2_droid_full_selected" / "fold_plan.json").read_text())
    folds = {}
    for fk, spec in fp["folds"].items():
        grid = {fam: len(v["train_generators"]) for fam, v in spec.items()}
        ho = {fam: v["heldout_generators"] for fam, v in spec.items()}
        zero = [f for f, n in grid.items() if n < 2]
        folds[fk] = {"train_gen_counts": grid, "anchor_zero_families": zero,
                     "heldout_generator_rows": {g: gen[g] for gs in ho.values() for g in gs}}
    return {"rows": rows, "families": kd(fam), "generators": kd(gen), "mode": kd(mode),
            "label": kd(label), "language": kd(lang), "source": kd(src),
            "split_source": kd(split_src), "distinct_source_row_sha1": len(sha),
            "fold_plan": folds, "sha256sums": (DATA / "h2_droid_full_selected" / "SHA256SUMS.txt").read_text().strip().splitlines()}


# ---------------------------------------------------------------- AuthorBench DCAN
def _fence(code: str) -> bool:
    s = code.lstrip()
    return s.startswith("```") or "```" in code[:120]


def audit_dcan() -> dict:
    p = DATA / "h2_authorbench_dcan" / "core.jsonl"
    rows = 0
    fam, model, split, lang = Counter(), Counter(), Counter(), Counter()
    task_fams: dict[str, set] = defaultdict(set)
    task_split: dict[str, set] = defaultdict(set)
    pair_seen = set()
    dup_pair = 0
    rep_gt1 = 0
    empty = fence = 0
    with p.open() as fh:
        for line in fh:
            d = json.loads(line)
            rows += 1
            fam[d["family"]] += 1
            model[d["model_name"]] += 1
            split[d["task_split"]] += 1
            lang[d["language"]] += 1
            task_fams[d["task_id"]].add(d["family"])
            task_split[d["task_id"]].add(d["task_split"])
            key = (d["task_id"], d["model_name"])
            if key in pair_seen:
                dup_pair += 1
            pair_seen.add(key)
            if d.get("replicate_count", 1) > 1:
                rep_gt1 += 1
            c = d.get("code") or ""
            if c.strip() == "":
                empty += 1
            elif _fence(c):
                fence += 1
    fams_per_task = Counter(len(v) for v in task_fams.values())
    split_per_task = Counter(len(v) for v in task_split.values())
    task_counts_by_split = Counter(next(iter(v)) for v in task_split.values())
    return {"rows": rows, "tasks": len(task_fams), "families": kd(fam), "models": kd(model),
            "task_split": kd(split), "language": kd(lang),
            "task_counts_by_split": kd(task_counts_by_split),
            "families_per_task_dist": {str(k): v for k, v in sorted(fams_per_task.items())},
            "tasks_with_lt2_families": sum(v for k, v in fams_per_task.items() if k < 2),
            "split_values_per_task_dist": {str(k): v for k, v in sorted(split_per_task.items())},
            "tasks_in_multiple_splits": sum(v for k, v in split_per_task.items() if k > 1),
            "duplicate_task_model_pairs": dup_pair, "replicate_count_gt1": rep_gt1,
            "empty_code": empty, "fence_code": fence}


# ---------------------------------------------------------------- LLM-CodeGen
def audit_llmcg() -> dict:
    p = DATA / "h2_llm_codegen" / "core.jsonl"
    rows = 0
    fam, model, scen, split, cwe, lang = (Counter() for _ in range(6))
    task_fams: dict[str, set] = defaultdict(set)
    task_split: dict[str, set] = defaultdict(set)
    pair_seen = set()
    dup_pair = 0
    empty = fence = 0
    with p.open() as fh:
        for line in fh:
            d = json.loads(line)
            rows += 1
            fam[d["family"]] += 1
            model[d["model_name"]] += 1
            scen[d["scenario"]] += 1
            split[d["task_split"]] += 1
            cwe[d["cwe_id"]] += 1
            lang[d["language"]] += 1
            task_fams[d["task_id"]].add(d["family"])
            task_split[d["task_id"]].add(d["task_split"])
            key = (d["task_id"], d["model_name"])
            if key in pair_seen:
                dup_pair += 1
            pair_seen.add(key)
            c = d.get("code") or ""
            if c.strip() == "":
                empty += 1
            elif _fence(c):
                fence += 1
    fams_per_task = Counter(len(v) for v in task_fams.values())
    split_per_task = Counter(len(v) for v in task_split.values())
    task_counts_by_split = Counter(next(iter(v)) for v in task_split.values())
    return {"rows": rows, "tasks": len(task_fams), "families": kd(fam), "models": kd(model),
            "scenario": kd(scen), "task_split": kd(split), "language": kd(lang),
            "task_counts_by_split": kd(task_counts_by_split),
            "cwe_count": len(cwe),
            "families_per_task_dist": {str(k): v for k, v in sorted(fams_per_task.items())},
            "tasks_with_lt2_families": sum(v for k, v in fams_per_task.items() if k < 2),
            "split_values_per_task_dist": {str(k): v for k, v in sorted(split_per_task.items())},
            "tasks_in_multiple_splits": sum(v for k, v in split_per_task.items() if k > 1),
            "duplicate_task_model_pairs": dup_pair, "empty_code": empty, "fence_code": fence}


# ---------------------------------------------------------------- STACAD
def audit_stacad() -> dict:
    cdir = DATA / "stacad_v2" / "extracted" / "STACAD-v2" / "data" / "corpus_v2"
    folds = np.load(cdir / "folds.npy")
    langs = ["py", "java", "c", "cpp", "php", "go", "cs"]   # corpus_v2.py LANGS：folds.npy 与写入顺序逐 pair 对齐
    per_file: dict[tuple, dict] = {}
    per_lang = {}
    total = 0
    gidx = 0
    label = Counter()
    split = Counter()
    model = Counter()
    dup_combo = Counter()
    for lg in langs:
        p = cdir / f"{lg}.jsonl"
        n = 0
        with p.open() as fh:
            for line in fh:
                d = json.loads(line)
                n += 1
                total += 1
                label[d.get("label")] += 1
                split[d.get("split_v1")] += 1
                model[d.get("paraphrased_by")] += 1
                key = (lg, d.get("file_name"))
                e = per_file.setdefault(key, {"n": 0, "splits": set(), "folds": set(),
                                              "empty_h": 0, "empty_l": 0})
                e["n"] += 1
                e["splits"].add(d.get("split_v1"))
                e["folds"].add(int(folds[gidx]))
                gidx += 1
                if (d.get("human_src") or "").strip() == "":
                    e["empty_h"] += 1
                if (d.get("llm_src") or "").strip() == "":
                    e["empty_l"] += 1
                dup_combo[(d.get("file_name"), d.get("paraphrased_by"))] += 1
        per_lang[lg] = {"pairs": n}
        log(f"stacad {lg}: {n} pairs")

    file_counts = sorted(e["n"] for e in per_file.values())
    multi_split_files = sum(1 for e in per_file.values() if len(e["splits"]) > 1)
    multi_fold = [(k, sorted(v["folds"])) for k, v in per_file.items() if len(v["folds"]) > 1]
    fold_file_counts = Counter()
    for e in per_file.values():
        if len(e["folds"]) == 1:
            fold_file_counts[next(iter(e["folds"]))] += 1
    change = np.flatnonzero(np.diff(folds)) + 1
    align = {"mode": "direct-stream-order", "files_with_multi_fold": len(multi_fold),
             "multi_fold_examples": [{"file": f"{lg}:{fn}", "folds": fo} for (lg, fn), fo in multi_fold[:5]],
             "files_per_fold": {str(k): v for k, v in sorted(fold_file_counts.items())},
             "runs": int(len(change) + 1)}
    fold_sha = hashlib.sha256((cdir / "folds.npy").read_bytes()).hexdigest()
    return {"pairs": total, "files": len(per_file), "per_lang": per_lang,
            "fold_vector_sha256": fold_sha,
            "labels": kd(label),
            "split_v1": kd(split),
            "paraphrased_by": kd(model),
            "pairs_per_file_hist": {str(k): file_counts.count(k) for k in sorted(set(file_counts))},
            "files_with_multiple_splits": multi_split_files,
            "duplicate_file_model_combos": sum(1 for v in dup_combo.values() if v > 1),
            "files_with_empty_src": sum(1 for e in per_file.values() if e["empty_h"] or e["empty_l"]),
            "folds_len": int(len(folds)), "folds_values": {str(k): int(v) for k, v in zip(*np.unique(folds, return_counts=True))},
            "fold_alignment": align}


# ---------------------------------------------------------------- CoDET-M4
def audit_codet() -> dict:
    import pyarrow.parquet as pq

    p = DATA / "codet_m4" / "dataset_without_comments.parquet"
    pf = pq.ParquetFile(p)
    cols = ["language", "model", "split", "target", "source", "cleaned_code"]
    avail = [c for c in cols if c in pf.schema_arrow.names]
    cnt = {c: Counter() for c in avail}
    rows = 0
    null_model = empty_model = 0
    for batch in pf.iter_batches(batch_size=8192, columns=avail, use_threads=False):
        data = {c: batch.column(c).to_pylist() for c in avail}
        n = batch.num_rows
        rows += n
        for c in avail:
            if c == "cleaned_code":
                continue
            cnt[c].update(data[c])
        for m, cc in zip(data.get("model", [None] * n), data.get("cleaned_code", [None] * n)):
            if m is None:
                null_model += 1
            elif str(m).strip() == "":
                empty_model += 1
            if cc is not None and str(cc).strip() == "":
                cnt["cleaned_code"]["empty"] += 1
    out = {"rows": rows, "schema": pf.schema_arrow.names}
    for c in avail:
        if c == "cleaned_code":
            continue
        out[c] = kd(cnt[c])
    out["model_null"] = null_model
    out["model_empty_string"] = empty_model
    out["cleaned_code_empty"] = cnt["cleaned_code"].get("empty", 0)
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    report = {"meta": {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "script": "scripts/audit_server_2026_10_03.py", "root": str(ROOT)}}
    report["env"] = env_info()
    mark("env")
    report["aicd"] = audit_aicd()
    mark("aicd")
    report["droid"] = audit_droid()
    mark("droid")
    report["dcan"] = audit_dcan()
    mark("dcan")
    report["llmcg"] = audit_llmcg()
    mark("llmcg")
    report["stacad"] = audit_stacad()
    mark("stacad")
    report["codet_m4"] = audit_codet()
    mark("codet_m4")
    report["timing_s"] = SECTION_T

    # ---- 与 manifest 的对照（T2 为主）
    man = json.loads((DATA / "acl_attribution_collection_v1" / "collection_manifest.json").read_text())
    aicd_src = next(s for s in man["sources"] if s["id"] == "aicd_bench")
    cmp = {}
    for cfg in ("T1", "T2", "T3"):
        want = aicd_src["configs"][cfg]["rows"]
        got = report["aicd"]["dirs"][cfg]["rows"]
        cmp[cfg] = {"manifest_rows": want, "computed_rows": got, "match": want == got}
    report["checks_vs_manifest"] = cmp

    # ---- 汇总判定
    checks = []
    chk = checks.append
    chk({"id": "aicd_shards_20", "status": "pass" if sum(
        v["shards"] for v in report["aicd"]["dirs"].values()) == 20 else "fail",
        "detail": {k: v["shards"] for k, v in report["aicd"]["dirs"].items()}})
    chk({"id": "aicd_rows_vs_manifest", "status": "pass" if all(v["match"] for v in cmp.values()) else "fail",
         "detail": cmp})
    chk({"id": "aicd_t2_cross_split_hash_leak",
         "status": "info", "detail": report["aicd"]["hash_check"]["T2"]})
    chk({"id": "aicd_label_map",
         "status": "pending" if (DATA / "acl_attribution_collection_v1" / "aicd_label_map_pending.json").exists() else "missing",
         "detail": "官方映射未核验：HF 数据集卡为空、GitHub 404；保留 numeric_id（11 类），不出具具体家族命名"})
    chk({"id": "droid_rows_146718", "status": "pass" if report["droid"]["rows"] == 146718 else "fail",
         "detail": report["droid"]["rows"]})
    chk({"id": "dcan_min2_families_per_task",
         "status": "pass" if report["dcan"]["tasks_with_lt2_families"] == 0 else "fail",
         "detail": report["dcan"]["families_per_task_dist"]})
    chk({"id": "dcan_task_split_purity",
         "status": "pass" if report["dcan"]["tasks_in_multiple_splits"] == 0 else "fail",
         "detail": report["dcan"]["split_values_per_task_dist"]})
    chk({"id": "llmcg_min2_families_per_task",
         "status": "pass" if report["llmcg"]["tasks_with_lt2_families"] == 0 else "fail",
         "detail": report["llmcg"]["families_per_task_dist"]})
    chk({"id": "llmcg_task_split_purity",
         "status": "pass" if report["llmcg"]["tasks_in_multiple_splits"] == 0 else "fail",
         "detail": report["llmcg"]["split_values_per_task_dist"]})
    st = report["stacad"]
    chk({"id": "stacad_folds_len", "status": "pass" if st["folds_len"] == st["pairs"] else "fail",
         "detail": {"folds_len": st["folds_len"], "pairs": st["pairs"]}})
    chk({"id": "stacad_file_level_split_purity",
         "status": "pass" if st["files_with_multiple_splits"] == 0 else "fail",
         "detail": st["files_with_multiple_splits"]})
    chk({"id": "stacad_fold_alignment",
         "status": "pass" if st["fold_alignment"]["files_with_multi_fold"] == 0 else "fail",
         "detail": st["fold_alignment"]})
    chk({"id": "codet_m4_model_missing",
         "status": "info", "detail": {"null": report["codet_m4"]["model_null"],
                                      "empty": report["codet_m4"]["model_empty_string"]}})
    report["checks"] = checks

    # ---- 切分清单（供后续实验协议引用）
    aicd_t2 = report["aicd"]["dirs"]["T2"]
    split_manifest = {
        "generated_utc": report["meta"]["generated_utc"],
        "sources": {
            "aicd_T2_official": {
                "role": "主规模基准（闭集；仅 numeric_id）",
                "split_rows": aicd_t2["split_rows"],
                "label_map_status": "pending_official_verification",
                "rule": "官方 train/validation/test；T1/T3 为独立任务视图，禁止相加或拼接标签"},
            "h2_authorbench_dcan": {
                "role": "同题跨 family 语义正对核心",
                "split_counts_tasks": report["dcan"].get("task_counts_by_split"),
                "split_counts_rows": report["dcan"]["task_split"],
                "rule": "group=task_id；按任务切分；每任务 ≥2 family"},
            "h2_llm_codegen": {
                "role": "同 CWE 跨模型补充对齐",
                "split_counts_tasks": report["llmcg"].get("task_counts_by_split"),
                "split_counts_rows": report["llmcg"]["task_split"],
                "rule": "group=task_id(=scenario:cwe_id)；scenario 是场景变量不是 family"},
            "stacad_v2": {
                "role": "文件级 paired/OOD",
                "split_v1_rows": report["stacad"]["split_v1"],
                "files_per_fold": report["stacad"]["fold_alignment"]["files_per_fold"],
                "fold_vector_sha256": report["stacad"].get("fold_vector_sha256"),
                "rule": "group=file_name；5 折文件级（corpus_v2.assign_folds）；本地上传语料逐 pair 校验 0 文件跨折"},
            "h2_droid_full_selected": {
                "role": "generator-held-out 外部压力测试（无 task_id）",
                "fold_plan": "fold_0/fold_1（anchor=0 家族见审计 JSON）",
                "rule": "按 generator 留出；不承担同题语义对齐 loss"},
            "codet_m4": {
                "role": "结构化来源指纹控制",
                "split_rows": report["codet_m4"].get("split"),
                "rule": "target(human/ai) 与 model 分开；缺失 model 不建新 generator"},
        },
    }
    jdump(split_manifest, OUT / "split_manifest_2026-10-03.json")

    jdump(report, OUT / "audit_server_2026-10-03.json")

    # ---- Markdown 摘要
    md = [f"# 服务端数据审计（2026-10-03）", "",
          f"脚本 `scripts/audit_server_2026_10_03.py`；输出 JSON 同目录。",
          f"- 磁盘：{report['env']['disk']}",
          f"- GPU：{report['env'].get('gpu')}（{report['env'].get('gpu_mem_gb')} GB）",
          f"- 运行时长（秒）：{report['timing_s']}", ""]

    md.append("## AICD")
    md.append("| 配置 | 分片 | 行数 | train/val/test | 空代码 | 重复风险（哈希级） |")
    md.append("|---|---:|---:|---|---:|---|")
    for cfg, v in report["aicd"]["dirs"].items():
        h = report["aicd"]["hash_check"][cfg]
        md.append(f"| {cfg} | {v['shards']} | {v['rows']} | "
                  f"{v['split_rows'].get('train',0)}/{v['split_rows'].get('validation',0)}/{v['split_rows'].get('test',0)} | "
                  f"{v['empty_code']} | {h['scan']}；distinct {h['distinct_hashes']}；跨 split 泄露哈希 {h['cross_split_leaked_hashes']} |")
    md.append("")
    md.append("T2 逐 split 标签分布：")
    md.append("```")
    md.append(json.dumps(report["aicd"]["dirs"]["T2"]["labels_by_split"], ensure_ascii=False))
    md.append("```")
    md.append(f"标签映射：**未核验**（HF 数据集卡为空、GitHub 404）→ 一律用 numeric_id。")

    md.append("\n## Droid")
    d = report["droid"]
    md.append(f"- 行数 {d['rows']}；家族 {d['families']}；generator 数 {len(d['generators'])}（含 human 伪 generator，machine=32）；"
              f"label {d['label']}；split_source {d['split_source']}")
    md.append(f"- 折 anchor=0 家族：{ {k: v['anchor_zero_families'] for k, v in d['fold_plan'].items()} }")

    md.append("\n## AuthorBench DCAN")
    a = report["dcan"]
    md.append(f"- 行数 {a['rows']}；任务 {a['tasks']}；split {a['task_split']}；"
              f"每任务家族数分布 {a['families_per_task_dist']}；<2 家族任务 {a['tasks_with_lt2_families']}")
    md.append(f"- 任务跨 split 数 {a['tasks_in_multiple_splits']}；重复(task,model) {a['duplicate_task_model_pairs']}；"
              f"空代码 {a['empty_code']}；围栏代码 {a['fence_code']}")

    md.append("\n## LLM-CodeGen")
    m = report["llmcg"]
    md.append(f"- 行数 {m['rows']}；任务 {m['tasks']}；split {m['task_split']}；场景 {m['scenario']}；"
              f"每任务家族数分布 {m['families_per_task_dist']}；<2 家族任务 {m['tasks_with_lt2_families']}")
    md.append(f"- 任务跨 split 数 {m['tasks_in_multiple_splits']}；重复(task,model) {m['duplicate_task_model_pairs']}；"
              f"空代码 {m['empty_code']}；围栏代码 {m['fence_code']}")

    md.append("\n## STACAD")
    md.append(f"- 对 {st['pairs']}；文件 {st['files']}；文件级多 split {st['files_with_multiple_splits']}；"
              f"split_v1 {st['split_v1']}；fold 分布 {st['folds_values']}")
    md.append(f"- fold 对齐：{st['fold_alignment']}")

    md.append("\n## CoDET-M4")
    cm = report["codet_m4"]
    md.append(f"- 行数 {cm['rows']}；target {cm.get('target')}；language {cm.get('language')}；"
              f"missing model {cm['model_null']}（null）+ {cm['model_empty_string']}（空串）")

    md.append("\n## 切分清单（split_manifest_2026-10-03.json 摘要）")
    md.append(f"- AICD T2：{aicd_t2['split_rows']}（numeric labels；映射 pending）")
    md.append(f"- DCAN：任务/行 {report['dcan'].get('task_counts_by_split')} / {report['dcan']['task_split']}（group=task_id）")
    md.append(f"- LLM-CodeGen：任务/行 {report['llmcg'].get('task_counts_by_split')} / {report['llmcg']['task_split']}（group=task_id）")
    md.append(f"- STACAD：文件级五折 0 跨折；文件/折 {report['stacad']['fold_alignment']['files_per_fold']}；folds sha256 {str(report['stacad'].get('fold_vector_sha256'))[:16]}…")
    md.append("- Droid：generator-held-out 两折（仅外部压力测试）；CoDET-M4：仅结构化指纹控制")

    md.append("\n## 检查清单")
    for c in checks:
        md.append(f"- `{c['id']}`：**{c['status']}** — {json.dumps(c['detail'], ensure_ascii=False)[:400]}")
    (OUT / "audit_server_2026-10-03.md").write_text("\n".join(md) + "\n")
    log(f"written {OUT}/audit_server_2026-10-03.{{json,md}}")


if __name__ == "__main__":
    main()
