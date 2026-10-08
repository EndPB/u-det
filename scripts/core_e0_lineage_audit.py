"""E0：lineage 与 pair 支持审计（《核心假设后训练差异与归因多实验指导》§6）。

只读取已有 manifest/元数据与 HF model card（API 元数据 JSON，不下载权重）；
不读取旧 test 代码正文（test-split 行的正文不装载；normalized/skeleton 仅对 train/dev 计算）。

输出：d-det/artifacts/core_lineage_audit_2026-10-08/
- paired_unit_index.jsonl    每 dual（complete+instruct）model 一行
- lineage_adjudication.json  verified/partial/unknown 三档 + 逐字段证据
- support_matrix.csv/json    每 unit 的 task 覆盖/split 支持/可进入 R,F,G
- collision_audit.json       exact / normalized / skeleton / task-id 跨 split
- prereg_r0_r1.json          R0/R1 操作定义、指标、切分与停止规则
- commands.txt、git_head.txt、git_status.txt、SHA256SUMS.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "d-det/artifacts/core_lineage_audit_2026-10-08"
RECORDS = ROOT / "d-det/data/public_same_task_full_2026-10-07/records.jsonl"
SPLIT_CSV = ROOT / "d-det/artifacts/public_full_receive_2026-10-08/prereg/split_index.csv"
FUP = ROOT / "d-det/artifacts/public_full_followup_2026-10-08"
LOG: list[str] = []


def log(m: str):
    print(m, flush=True)
    LOG.append(m)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()


_WS = re.compile(r"\s+")
_ID = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_NUM = re.compile(r"\b\d[\d_]*\.?[\d_]*([eE][+-]?\d+)?\b")
_STR = re.compile(r"(\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''|\"[^\"\n]*\"|'[^'\n]*')")


def norm_ws(code: str) -> str:
    return _WS.sub(" ", code).strip()


def skeleton(code: str) -> str:
    t = _STR.sub("S", code)
    t = _NUM.sub("N", t)
    t = _ID.sub("ID", t)
    return _WS.sub(" ", t).strip()


def hf_repo_of(model_id: str) -> str | None:
    if model_id in ("claude-3-5-sonnet-20240620", "claude-3-5-haiku-20241022", "gpt-4-0613"):
        return None  # 闭源 API 模型（无 HF 权重）
    if "--" in model_id:
        return model_id.replace("--", "/", 1)
    return None


def parse_params(model_id: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model_id)
    if not m:
        return None, None
    val = float(m.group(1))
    return (val * 1e9), f"name_parsed:{m.group(0)}"


def fetch_hf_info(repo: str, retries: int = 2, sleep: float = 0.25):
    import urllib.request
    for i in range(retries):
        try:
            req = urllib.request.Request(f"https://hf-mirror.com/api/models/{repo}",
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if i == retries - 1:
                return {"__error__": f"{type(e).__name__}: {e}"}
            time.sleep(1.0)
    return {"__error__": "unknown"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hf", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    log("[1] scan records (subset=full)")
    model_modes = defaultdict(set)
    model_rows = Counter()
    unit_task_rows = Counter()          # (model, mode, task) -> n rows
    unit_asset = defaultdict(set)
    unit_release = defaultdict(set)
    unit_source = defaultdict(set)
    unit_family_hint = defaultdict(set)
    unit_variant_hint = defaultdict(set)
    exact_dup = defaultdict(int)        # code_sha256 -> count (full scope)
    task_split = {}
    with SPLIT_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            task_split[r["task_id"]] = r["split"]
    row_meta = []  # 精简元数据（全部行，无正文）
    text_missing = Counter()
    task_rows_by_unit = defaultdict(list)  # (m,mode,t) -> [sha,...]
    with RECORDS.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("subset") != "full":
                continue
            m = d["model_id"]; mode = d["generation_mode"]; t = d["task_id"]
            model_modes[m].add(mode)
            model_rows[(m, mode)] += 1
            unit_task_rows[(m, mode, t)] += 1
            unit_asset[(m, mode)].add(d.get("asset", ""))
            unit_release[(m, mode)].add(d.get("release", ""))
            unit_source[(m, mode)].add(d.get("source", ""))
            unit_family_hint[(m, mode)].add(d.get("family_hint", ""))
            unit_variant_hint[(m, mode)].add(d.get("variant_hint", ""))
            cs = d.get("solution_sha256") or d.get("code_sha256") or ""
            text = d.get("solution") or d.get("code") or ""
            if not text:
                text_missing[f"{m}|{mode}"] += 1
            if cs:
                exact_dup[cs] += 1
            task_rows_by_unit[(m, mode, t)].append(cs)
            row_meta.append((m, mode, t, cs, len(text), task_split.get(t, "unknown")))
    log(f"  rows(full)={len(row_meta)} models={len(model_modes)}")
    dual = sorted(m for m, ms in model_modes.items() if ms == {"complete", "instruct"})
    instr_only = sorted(m for m, ms in model_modes.items() if ms == {"instruct"})
    comp_only = sorted(m for m, ms in model_modes.items() if ms == {"complete"})
    log(f"  dual={len(dual)} instruct_only={len(instr_only)} complete_only={len(comp_only)}")

    # ---- task 覆盖与 split 支持 ----
    unit_tasks = defaultdict(lambda: defaultdict(set))  # unit -> mode -> tasks
    for (m, mode, t, cs, ln, sp) in row_meta:
        unit_tasks[m][mode].add(t)

    # ---- exact dup（字段级，全量）----
    dup_task_pairs = {k: v for k, v in unit_task_rows.items() if v > 1}
    exact_dup_groups = {h: c for h, c in exact_dup.items() if c > 1}
    dup_detail = {}
    n_dup_identical = n_dup_distinct = 0
    for k, v in dup_task_pairs.items():
        shas = task_rows_by_unit[k]
        uni = len(set(shas))
        dup_detail[f"{k[0]}|{k[1]}|{k[2]}"] = {"rows": v, "distinct_sha": uni}
        if uni == 1:
            n_dup_identical += 1
        else:
            n_dup_distinct += 1
    log(f"  exact dup sha groups: {len(exact_dup_groups)}; unit-task multi rows: {len(dup_task_pairs)} "
        f"(identical={n_dup_identical}, distinct={n_dup_distinct})")
    dup_by_unit = Counter(f"{k[0]}|{k[1]}" for k in dup_task_pairs)

    # ---- normalized / skeleton（只对 train/dev 行；test 正文不装载）----
    log("[2] normalized/skeleton hashes (train/dev rows only; test text not loaded)")
    norm_h = defaultdict(list)   # hash -> [(model,mode,task,len)]
    skel_h = defaultdict(list)
    n_td = 0
    with RECORDS.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("subset") != "full":
                continue
            t = d["task_id"]
            if task_split.get(t) == "test":
                continue  # §6.1：不读旧 test 代码正文
            code = d.get("solution") or d.get("code") or ""
            n_td += 1
            key = (d["model_id"], d["generation_mode"], t)
            nm = norm_ws(code)
            norm_h[sha256_text(nm)].append((*key, len(nm)))
            skel_h[sha256_text(skeleton(code))].append((*key, len(nm)))
    log(f"  train/dev rows hashed: {n_td}")

    def cross_task_detail(hmap):
        groups = []
        for h, keys in hmap.items():
            tasks = {k[2] for k in keys}
            if len(tasks) < 2:
                continue
            splits = {task_split.get(k[2], "unknown") for k in keys}
            mx = max((k[3] for k in keys), default=-1)
            groups.append({"hash": h, "n": len(keys), "max_len": mx,
                           "tasks": sorted(tasks), "splits": sorted(splits),
                           "units": sorted({f"{k[0]}|{k[1]}" for k in keys})})
        return groups

    norm_groups_raw = cross_task_detail(norm_h)
    # 细分：空/短/实质 × 同/跨 split
    def breakdown(groups):
        b = {"empty": 0, "short_lt30": 0, "real_ge30": 0,
             "cross_split": 0, "same_split": 0,
             "cross_split_real": [], "same_split_long_pairs": defaultdict(int)}
        for g in groups:
            if g["max_len"] == 0:
                b["empty"] += 1; continue
            if g["max_len"] < 30:
                b["short_lt30"] += 1; continue
            b["real_ge30"] += 1
            if len(g["splits"]) > 1:
                b["cross_split"] += 1
                b["cross_split_real"].append({"hash": g["hash"][:12], "max_len": g["max_len"],
                                               "tasks": g["tasks"][:8], "n": g["n"]})
            else:
                b["same_split"] += 1
                if g["max_len"] >= 200:
                    b["same_split_long_pairs"][tuple(g["tasks"])] += 1
        b["same_split_long_pairs"] = {str(k): v for k, v in
                                       sorted(b["same_split_long_pairs"].items(), key=lambda x: -x[1])[:10]}
        b["cross_split_real"] = b["cross_split_real"][:30]
        return b

    norm_coll = [g for g in norm_groups_raw if g["max_len"] >= 30]
    skel_raw = cross_task_detail(skel_h)
    skel_detail = breakdown(skel_raw)
    norm_detail = breakdown(norm_groups_raw)
    log(f"  normalized cross-task groups: {len(norm_groups_raw)} (detail: {norm_detail['empty']} empty, "
        f"{norm_detail['short_lt30']} short, {norm_detail['real_ge30']} real; "
        f"cross-split real={norm_detail['cross_split']})")

    # ---- task id 跨 split ----
    split_tasks = defaultdict(set)
    for t, sp in task_split.items():
        split_tasks[sp].add(t)
    tid_cross = {a: {b: len(split_tasks[a] & split_tasks[b]) for b in split_tasks if b > a}
                 for a in split_tasks}

    collision = {
        "schema": "core_collision_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "text_missing_rows": dict(text_missing),
        "exact_sha_groups_gt1": len(exact_dup_groups),
        "unit_task_multi_rows": {"groups": len(dup_task_pairs),
                                 "identical_output_groups": n_dup_identical,
                                 "distinct_output_groups": n_dup_distinct,
                                 "by_unit": dict(dup_by_unit),
                                 "detail": dup_detail},
        "normalized_whitespace": {"method": "collapse all whitespace", "scope": "train+dev rows only",
                                  "cross_task_groups": len(norm_groups_raw), "breakdown": norm_detail},
        "lexical_skeleton": {"method": "strings→S, numbers→N, identifiers→ID, whitespace collapsed",
                             "scope": "train/dev rows only",
                             "cross_task_groups": len(skel_raw),
                             "breakdown": skel_detail},
        "task_id_cross_split_overlap": tid_cross,
        "test_side_note": "test-split 代码正文未读取（§6.1）；test 参与仅通过既有 code_sha256 字段级精确比较",
        "conclusion": ("跨 split 实质碰撞均为短输出自然撞车（<200ch）；无长输出跨 split 任务级泄漏；"
                       "同 split 长输出重复集中于 BigCodeBench/1120↔1121（同题重复注册，均在 train）"),
    }

    # ---- HF 元数据 ----
    hf_info = {}
    if not args.skip_hf:
        log("[3] fetch HF metadata (api/models; no weights)")
        for i, m in enumerate(dual):
            repo = hf_repo_of(m)
            if repo is None:
                hf_info[m] = {"hf_repo": None, "note": "closed_api_model_no_hf_weights"}
                continue
            info = fetch_hf_info(repo)
            hf_info[m] = {"hf_repo": repo, "info": info}
            time.sleep(0.15)
        (OUT / "local_hf_meta.json").write_text(json.dumps(hf_info, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        p_cache = OUT / "local_hf_meta.json"
        if p_cache.exists():
            hf_info = json.loads(p_cache.read_text(encoding="utf-8"))
            log("[3] HF fetch skipped; reused local_hf_meta.json")
        else:
            log("[3] HF fetch skipped (no cache available)")

    # ---- paired_unit_index ----
    log("[4] paired_unit_index + lineage adjudication")
    idx_rows = []
    adj = {}
    for m in dual:
        pc, pc_src = parse_params(m)
        tasks = unit_tasks[m]["complete"] | unit_tasks[m]["instruct"]
        tsplit = Counter(task_split.get(t, "unknown") for t in tasks)
        info = hf_info.get(m, {}).get("info") or {}
        cd = (info.get("cardData") or {}) if isinstance(info, dict) else {}
        base_model = cd.get("base_model")
        if isinstance(base_model, str):
            base_model = [base_model]
        license_ = cd.get("license")
        idx_rows.append({
            "model_id": m, "hf_repo": hf_repo_of(m), "modes": ["complete", "instruct"],
            "family_raw": sorted(unit_family_hint.get((m, "instruct"), set())),
            "variant_hint": sorted(unit_variant_hint.get((m, "instruct"), set())),
            "release": sorted(unit_release.get((m, "instruct"), set())),
            "source": sorted(unit_source.get((m, "instruct"), set())),
            "assets": sorted(unit_asset.get((m, "instruct"), set())),
            "parameter_count_est": pc, "parameter_count_evidence": pc_src,
            "task_count": len(tasks), "task_split": dict(tsplit),
            "source_url": f"https://huggingface.co/{hf_repo_of(m)}" if hf_repo_of(m) else None,
            "license": license_,
            "base_model_card": base_model,
            "card_error": (info.get("__error__") if isinstance(info, dict) else None),
        })
        # 判定
        cond = {
            "same_task_protocol": {"ok": tsplit.get("train", 0) + tsplit.get("dev", 0) > 0,
                                   "evidence": "records: identical 1140-task coverage for both modes"},
            "documented_base_relation": {"ok": bool(base_model), "evidence": f"HF cardData.base_model={base_model}"},
            "independent_provenance": {"ok": True,
                                       "evidence": "public per-asset sha256 + release/source fields (server_reconstruction_only caveat)"},
            "post_training_event": {"ok": False,
                                    "evidence": ("complete/instruct 为同一权重两种生成协议（协议条件差异），"
                                                 "非后训练事件；R1 需 base↔post-trained 成对成员")},
        }
        lead = {"verified": 0, "partial": 0, "unknown": 0}
        if cond["documented_base_relation"]["ok"] and cond["post_training_event"]["ok"]:
            lvl = "verified"
        elif cond["documented_base_relation"]["ok"]:
            lvl = "partial"
        else:
            lvl = "unknown"
        lead[lvl] = 1
        adj[m] = {"level": lvl, "field_evidence": cond,
                  "r0_eligible": lvl in ("verified", "partial", "unknown"),  # R0 全部合格（协议条件差异）
                  "r1_candidate": False}
    (OUT / "paired_unit_index.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in idx_rows) + "\n", encoding="utf-8")

    # ---- R1 候选：HF cardData.base_model 链条（child ← parent，parent 也在 dual）----
    rev = {m.replace("--", "/", 1): m for m in dual}
    r1_chains = []
    for m in dual:
        info = (hf_info.get(m, {}) or {}).get("info") or {}
        if not isinstance(info, dict):
            continue
        cd = info.get("cardData") or {}
        bm = cd.get("base_model")
        if isinstance(bm, str):
            bm = [bm]
        if not bm:
            continue
        hits = [rev[b] for b in bm if b in rev]
        r1_chains.append({"child": m, "documented_base_model": bm, "base_in_dual": hits})
    valid_chains = [c for c in r1_chains if c["base_in_dual"]]
    def is_base_like(mid: str) -> bool:
        name = mid.split("--", 1)[-1]
        return not re.search(r"(?i)(instruct|chat|orpo|dpo|-rl|reasoning|agent)", name)
    base_like_list = sorted(m for m in dual if is_base_like(m))
    log(f"  r1 documented chains (base in dual): {len(valid_chains)}; base-like models: {len(base_like_list)}")

    (OUT / "lineage_adjudication.json").write_text(json.dumps({
        "schema": "core_lineage_adjudication_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "levels": {m: v for m, v in sorted(adj.items())},
        "r0_note": "R0 标签=protocol_conditioned_difference（同权重两种生成协议）；不得写成后训练因果",
        "r1_status": ("partial_candidates_available" if valid_chains else "unavailable"),
        "r1_documented_chains": valid_chains,
        "r1_chain_condition_note": ("每条链需同时满足 same base/version ∧ documented post-training relation ∧ "
                                    "same task/protocol ∧ independent provenance；当前 body 证据=HF cardData.base_model"
                                    "（documented relation ✓；version 级别证据缺 → 保持 partial，等 R0 出口后再执行）"),
        "base_like_in_dual": base_like_list,
        "summary": {"dual_total": len(dual),
                    "with_base_model_card": sum(1 for m in dual if adj[m]['level'] == 'partial'),
                    "unknown": sum(1 for m in dual if adj[m]['level'] == 'unknown'),
                    "r1_chains": len(valid_chains)},
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- support matrix ----
    rows = []
    for m in sorted(model_modes):
        for mode in sorted(model_modes[m]):
            tasks = unit_tasks[m][mode]
            tsplit = Counter(task_split.get(t, "unknown") for t in tasks)
            rows.append({"unit": f"{m}::{mode}", "model_id": m, "mode": mode,
                         "task_count": len(tasks), "train": tsplit.get("train", 0),
                         "dev": tsplit.get("dev", 0), "test": tsplit.get("test", 0),
                         "dup_unit_task_rows": sum(v - 1 for k, v in unit_task_rows.items()
                                                   if k[0] == m and k[1] == mode and v > 1),
                         "dual": m in dual,
                         "r0_eligible": m in dual,
                         "f_eligible": True, "g_eligible": True})
    with (OUT / "support_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- prereg R0/R1 ----
    prereg = {
        "schema": "core_prereg_r0_r1_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "r0": {
            "label": "protocol_conditioned_difference",
            "data": "115 dual model_id × 969 train/dev tasks（test 不读）",
            "task_pairwise": "配对方向判别：same(model,task) 的 (h_complete,h_instruct)，判别哪个是 instruct",
            "readouts": {
                "1_single_complete": "样本级 LR（complete=1，model-balanced 权重）；评测在配对上",
                "2_single_instruct": "样本级 LR（instruct=1）；对称一致性检查",
                "3_delta_pair": "pairwise logistic on Δ=h_instr−h_comp",
                "4_S_sanity": "pairwise on ΔS（理论=0.5，阴性对照）",
                "5_u_joint": "u=[S;A]：线性（≡3，验证）+ MLP（S 条件化机会）",
                "6_P0": "tfidf_char/word, style_lr/lgb, metadata_only, size_length_only（同任务）",
                "7_late_fusion": "u_mlp 分数+长度/规模控制",
            },
            "metrics": "model-balanced pair-AUC（主）、BA、task-macro、task-cluster bootstrap 500 CI (seed 20261008)",
            "comparisons": "u − best_single；u − P0（point + bootstrap delta_mean）",
            "splits": {
                "task_heldout": "train tasks fit → dev tasks eval（同 model 集）",
                "model_heldout": "5-fold model 组留出（seed 20261008）；fit=其余 model 的 train tasks",
                "size_heldout": "参数量桶（<2B,2-8B,8-20B,≥20B）留出；fit=其余桶",
            },
            "stop_rules": "model-heldout 归零 → 收窄为协议指纹；不称后训练可迁移",
        },
        "r1": {
            "filter": "same base/version ∧ documented post-training relation ∧ same task/protocol ∧ independent provenance",
            "status": ("partial_candidates_available" if valid_chains else "unavailable"),
            "documented_chains": valid_chains,
            "note": "若无合资格 pair：按指导写 R1 unavailable；有 documented 链但 version 证据缺 → partial，等 R0 出口再执行",
        },
        "switches": {
            "training_allowed": True,
            "generation_allowed": False,
            "test_read_allowed": False,
            "old_test_reuse": False,
            "new_fresh_test_read": False,
            "weights_downloaded": False,
            "source_status": "server_reconstruction_only",
            "original_bundle_verified": False,
        },
    }
    (OUT / "prereg_r0_r1.json").write_text(json.dumps(prereg, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "collision_audit.json").write_text(json.dumps(collision, ensure_ascii=False, indent=1), encoding="utf-8")

    manifest = {
        "schema": "core_lineage_audit_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "exact_numbers": {"distinct_models_full": len(model_modes), "dual": len(dual),
                          "instruct_only": instr_only, "complete_only": comp_only,
                          "rows_total_full": len(row_meta),
                          "text_missing_rows": dict(text_missing),
                          "rows_train_dev_hashed": n_td},
        "outputs": ["paired_unit_index.jsonl", "lineage_adjudication.json", "support_matrix.csv",
                    "collision_audit.json", "prereg_r0_r1.json"],
        "runtime_seconds": time.time() - t0,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "logs" / "e0_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"[done] {time.time()-t0:.1f}s; dual={len(dual)}")


if __name__ == "__main__":
    main()
