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
      member_idx      (10659,)     index into members list (plan series order)
      task_idx        (10659,)     index into tasks list (row_index order)
      is_train        (10659,)     1 = train split
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


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    t0 = __import__("time").time()
    FEATS.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in (INPUTS / "records_train_dev.jsonl").read_text(
        encoding="utf-8").splitlines() if s.strip()]
    assert len(rows) == 10659

    idx = json.loads((R0 / "row_index.json").read_text(encoding="utf-8"))
    models_all, tasks_all = idx["models"], idx["tasks"]
    nT = len(tasks_all)
    plan = json.loads((ROOT / "d-det/artifacts/endpoint_balanced_relation_plan_2026-10-08/"
                       "endpoint_balanced_plan.json").read_text(encoding="utf-8"))
    members = [m for s in plan["series"] for m in plan["series_members"][s]]
    assert len(members) == 11
    midx = {m: i for i, m in enumerate(members)}
    gmidx = {m: models_all.index(m) for m in members}  # global index in the 115-model cache
    tidx = {t: i for i, t in enumerate(tasks_all)}
    assert set(tidx) == set(r["task_id"] for r in rows)

    member_idx = np.array([midx[r["model_id"]] for r in rows], dtype=np.int64)
    gmodel_idx = np.array([gmidx[r["model_id"]] for r in rows], dtype=np.int64)
    task_idx = np.array([tidx[r["task_id"]] for r in rows], dtype=np.int64)
    is_train = np.array([r["split"] == "train" for r in rows], dtype=np.int64)

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
                        task_idx=task_idx, is_train=is_train)

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
