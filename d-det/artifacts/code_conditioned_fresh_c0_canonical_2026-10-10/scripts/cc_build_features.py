"""Build the shared feature bundle for the code-conditioned round (server-side).

Outputs (d-det/artifacts/code_conditioned_design_2026-10-09/features/):
  - bundle.npz:
      hy_small        (10659, 512) frozen CodeT5-small mean-pool embedding of the code
                      (instruct mode), extracted from the r0 corpus cache
      style           (10659, 92)  stylometry features (r0 cache)
      meta            (10659, 10)  meta subset (r0 cache)
      sizelen         (10659, 4)   [log10 params, nchar, nlines] + NaN-filled flag
      sizeB           (10659,)     parsed parameter count (B)
      psi             (10659, 9)   static proxies from static_preflight
      member_idx      (10659,)     index into members list (ADMISSION order;\n                                   canonicalized 2026-10-10 per guidance §12)
      member_ids      (11,)        member ids themselves, in admission order
      task_idx        (10659,)     index into tasks list (row_index order)
      is_train        (10659,)     1 = train split

2026-10-10 fix: `member_idx` used to be built in endpoint-plan order
[CodeLlama, DeepSeek, Qwen] while `cc_common.fold_setup()` interprets it in
admission order [CodeLlama, Qwen, DeepSeek]. Assertions + `members_order`/
`row_mapping` hashes are now written to `feature_member_order_check.json`
BEFORE the bundle is written; any failure stops the build.
  - emb_task_small.npz: emb (969, 512) frozen CodeT5-small mean-pool embedding of
      the task `instruct_prompt` (same encoder protocol as hy).
  - features_manifest.json: shapes/hashes/protocol notes.

Encoder protocol (frozen, identical to r0 cache): CodeT5 frozen, mean-pool over
tokens, no special tokens, fp16 autocast, truncation 512 = first 384 + last 128.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import variant_transfer_stage1_execute as s1  # noqa: E402

DESIGN = ROOT / "d-det/artifacts/code_conditioned_design_2026-10-09"
INPUTS = DESIGN / "inputs"
PREFLIGHT = DESIGN / "static_preflight"
FEATS = DESIGN / "features"
R0 = ROOT / "d-det/artifacts/r0_protocol_diff_2026-10-08/local"
TASKMETA = ROOT / "d-det/data/public_same_task_full_2026-10-07/bigcodebench_task_metadata.jsonl"

PSI_FIELDS = ["parse_ok", "entrypoint_present", "n_ast_nodes", "n_calls", "n_branches",
              "n_handlers", "n_returns", "n_import_roots", "required_root_overlap_proxy"]

# Canonical admission order (2026-10-10); sha256 convention: sha256(utf-8 of
# "\n".join(members) + "\n") — same convention the local side used in its
# feature_member_order_audit.json (admission order sha 1afd29b6..., old feature
# order sha e6d7852c...).
OFFICIAL_MEMBERS_ORDER_SHA = "1afd29b6c240095269cb5f498302a0e07143e7b286869a47431872a2b2629175"
OLD_PLAN_ORDER_SHA_EXPECTED = "e6d7852c82b84d60185c7c57b78106df6f669fcad35bef2382c6bf7d31dff286"
OLD_BUNDLE_SHA256 = "be5cc9d63b489a7e3a790d654415c0d019b7d261366e0d1f6d2be3154f40a968"
ROW_MAPPING_CONVENTION = ("sha256(utf-8) over '\\n'.join(model_id+'|'+task_id+'|'+split+'|'+"
                          "solution_sha256) for all rows in record order; no trailing newline")


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    t0 = __import__("time").time()
    FEATS.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in (INPUTS / "records_train_dev.jsonl").read_text(
        encoding="utf-8").splitlines() if s.strip()]
    assert len(rows) == 10659

    idx = json.loads((R0 / "row_index.json").read_text(encoding="utf-8"))
    models_all, tasks_all = idx["models"], idx["tasks"]
    nT = len(tasks_all)

    # --- canonical members order (2026-10-10 fix, guidance §12) ------------
    # `member_idx` must index the ADMISSION members order (the same order
    # `cc_common.fold_setup()` uses). The endpoint plan is now only cross-
    # checked: until 2026-10-10 it was (mistakenly) used as the write order.
    adm = json.loads((INPUTS / "family_series_admission.json").read_text(encoding="utf-8"))
    members = [m["model_id"] for s in adm["series"] for m in s["members"]]
    assert len(members) == 11 and len(set(members)) == 11
    plan = json.loads((ROOT / "d-det/artifacts/endpoint_balanced_relation_plan_2026-10-08/"
                       "endpoint_balanced_plan.json").read_text(encoding="utf-8"))
    plan_members = [m for s in plan["series"] for m in plan["series_members"][s]]
    assert set(plan_members) == set(members), "plan/admission member sets differ"
    members_order_sha = sha256_str("\n".join(members) + "\n")
    plan_order_sha = sha256_str("\n".join(plan_members) + "\n")
    assert plan_order_sha == OLD_PLAN_ORDER_SHA_EXPECTED, "endpoint plan order changed unexpectedly"
    # STOP before writing anything if the canonical order regressed:
    assert members_order_sha == OFFICIAL_MEMBERS_ORDER_SHA, \
        f"members order != official admission order ({members_order_sha})"
    midx = {m: i for i, m in enumerate(members)}
    gmidx = {m: models_all.index(m) for m in members}  # global index in the 115-model cache
    tidx = {t: i for i, t in enumerate(tasks_all)}
    assert set(tidx) == set(r["task_id"] for r in rows)

    member_idx = np.array([midx[r["model_id"]] for r in rows], dtype=np.int64)
    member_idx_recomputed = np.array([members.index(r["model_id"]) for r in rows],
                                     dtype=np.int64)
    assert (member_idx == member_idx_recomputed).all(), "member_idx != admission-order index"
    gmodel_idx = np.array([gmidx[r["model_id"]] for r in rows], dtype=np.int64)
    task_idx = np.array([tidx[r["task_id"]] for r in rows], dtype=np.int64)
    is_train = np.array([r["split"] == "train" for r in rows], dtype=np.int64)

    # row-level member/task mapping hash (guidance §12 formula)
    row_lines = [f"{r['model_id']}|{r['task_id']}|{r['split']}|{r['solution_sha256']}"
                 for r in rows]
    row_mapping_sha = sha256_str("\n".join(row_lines))
    row_mapping_variants = {
        "nl_join_trailing_newline": sha256_str("\n".join(row_lines) + "\n"),
        "concat_no_separator": sha256_str("".join(row_lines)),
    }

    # hashes & assertions are persisted to disk BEFORE the bundle write (§12)
    (FEATS / "feature_member_order_check.json").write_text(json.dumps({
        "schema": "code_conditioned_feature_member_order_check_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "admission_file": ("d-det/artifacts/code_conditioned_design_2026-10-09/"
                           "inputs/family_series_admission.json"),
        "admission_file_sha256": sha256_file(INPUTS / "family_series_admission.json"),
        "members_order": members,
        "members_order_sha256": members_order_sha,
        "expected_members_order_sha256": OFFICIAL_MEMBERS_ORDER_SHA,
        "members_order_equals_admission": members_order_sha == OFFICIAL_MEMBERS_ORDER_SHA,
        "endpoint_plan_members_order_sha256": plan_order_sha,
        "endpoint_plan_members_order": plan_members,
        "plan_member_set_equal": True,
        "member_idx_recomputed_equal": bool((member_idx == member_idx_recomputed).all()),
        "checked_rows": len(rows),
        "row_mapping_sha256": row_mapping_sha,
        "row_mapping_convention": ROW_MAPPING_CONVENTION,
        "row_mapping_sha256_variants": row_mapping_variants,
        "old_bundle_sha256_superseded": OLD_BUNDLE_SHA256,
        "policy": ("assertions+hashes written before bundle write; any failure "
                   "stops the build (no fold fit may run)"),
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"assertions": "ok", "members_order_sha256": members_order_sha,
                      "row_mapping_sha256": row_mapping_sha}, indent=1))

    # ---- extract hy from the r0 corpus cache (model-major: gmi*nT*2 + ti*2 + instruct) ----
    r0_rows = gmodel_idx * nT * 2 + task_idx * 2 + 1
    # mapping check against the cached texts (empty codes were stored as " ")
    import gzip
    want = {int(r0_rows[i]): rows[i]["code"] for i in range(len(rows))}
    checked = ok = 0
    with gzip.open(R0 / "texts.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if int(o["i"]) in want:
                checked += 1
                ok += int(o["text"] == want[int(o["i"])] or (want[int(o["i"])] == "" and o["text"] == " "))
    assert checked == len(rows) and ok == len(rows), f"row mapping mismatch {ok}/{checked}"
    es = np.load(R0 / "emb_small.npz")["emb"]
    hy = es[r0_rows].astype(np.float32)
    sm = np.load(R0 / "style_meta.npz")
    style = sm["style"][r0_rows].astype(np.float32)
    meta = sm["meta"][r0_rows].astype(np.float32)
    sizeB = sm["sizeB"][r0_rows].astype(np.float64)

    # sizelen: mirror r0 NaN handling (parse failure -> median fill + missing flag)
    szraw = sm["sizelen"][r0_rows].astype(np.float64).copy()
    nan0 = np.isnan(szraw[:, 0])
    if nan0.any():
        med = float(np.nanmedian(szraw[:, 0]))
        fill = np.where(nan0, med, szraw[:, 0])
        sizelen = np.column_stack([fill, szraw[:, 1], szraw[:, 2], nan0.astype(np.float64)])
    else:
        sizelen = np.column_stack([szraw[:, 0], szraw[:, 1], szraw[:, 2],
                                   np.zeros(len(szraw))])

    # ---- static proxies (same order as records) ----
    psi = np.zeros((len(rows), len(PSI_FIELDS)), dtype=np.float64)
    with (PREFLIGHT / "static_proxies.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            assert d["model_id"] == rows[i]["model_id"] and d["task_id"] == rows[i]["task_id"]
            for j, k in enumerate(PSI_FIELDS):
                psi[i, j] = d["static_proxy"][k]

    np.savez_compressed(FEATS / "bundle.npz", hy_small=hy, style=style, meta=meta,
                        sizelen=sizelen, sizeB=sizeB, psi=psi,
                        member_idx=member_idx, gmodel_idx=gmodel_idx,
                        task_idx=task_idx, is_train=is_train,
                        member_ids=np.array(members))

    # ---- task prompt embeddings (instruct_prompt) with the frozen encoder ----
    prompts = {}
    with TASKMETA.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] in tidx:
                prompts[d["task_id"]] = d["instruct_prompt"]
    assert len(prompts) == len(tasks_all)
    texts = [prompts[t] for t in tasks_all]
    emb_task = s1.encode_codet5(s1.MODEL_SMALL, texts, device_batch=16)
    np.savez_compressed(FEATS / "emb_task_small.npz", emb=emb_task.astype(np.float32))

    man = {
        "schema": "code_conditioned_features_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "encoder_protocol": "CodeT5 frozen mean-pool no-special-tokens fp16; truncation 512=384+128",
        "encoder_model": str(s1.MODEL_SMALL),
        "rows": len(rows), "tasks": len(tasks_all), "members": len(members),
        "members_order": members,
        "members_order_sha256": members_order_sha,
        "expected_members_order_sha256": OFFICIAL_MEMBERS_ORDER_SHA,
        "row_mapping_sha256": row_mapping_sha,
        "row_mapping_convention": ROW_MAPPING_CONVENTION,
        "row_mapping_sha256_variants": row_mapping_variants,
        "canonicalization": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "reason": ("cross-side reproduction 2026-10-10: member_idx was built in "
                       "endpoint-plan order [CodeLlama, DeepSeek, Qwen] but interpreted "
                       "in admission order; canonicalized to admission order before "
                       "any fold fit (guidance §12)"),
            "superseded_feature_order_sha256": plan_order_sha,
            "superseded_bundle_sha256": OLD_BUNDLE_SHA256,
            "assertion_file": "feature_member_order_check.json",
        },
        "assertions": {
            "members_order_equals_admission": True,
            "member_idx_recomputed_equal": True,
            "assertions_written_before_bundle_write": True,
        },
        "script_sha256": {"cc_build_features.py": sha256_file(Path(__file__)),
                          "cc_common.py": sha256_file(ROOT / "scripts" / "cc_common.py")},
        "split_rows": {"train": int(is_train.sum()), "dev": int((1 - is_train).sum())},
        "empty_code_rows": int(sum(1 for r in rows if not r["code"])),
        "psi_fields": PSI_FIELDS,
        "row_mapping_check": {"checked": checked, "ok": ok,
                             "rule": "gmodel_idx*nT*2 + task_idx*2 + 1 (instruct) vs texts.jsonl.gz"},
        "notes": ("hy_small extracted from r0 emb_small cache; style/meta/sizelen from "
                  "style_meta.npz; psi from static_preflight; task prompts encoded fresh on server"),
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                  for p in (FEATS / "bundle.npz", FEATS / "emb_task_small.npz")},
    }
    (FEATS / "features_manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "hy": list(hy.shape), "task_emb": list(emb_task.shape),
                      "s": round(__import__("time").time() - t0, 1)}))


if __name__ == "__main__":
    main()
