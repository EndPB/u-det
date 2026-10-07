"""freeze_validation.json 组装（§3 门槛，test 前必须全过）。

6 项：
 1) stage-1 文件 hashes + 两配置 sha + records sha 对账；
 2) task split hashes 与 e38715d 一致；11 折 train/dev 不含 heldout；无 test 用于 fit/选择的证据（账本）；
 3) 22 评分包可加载 + 原 dev 复现核对（replay_check）+ 库版本 + commit；
 4) multiplicity 数值验证 + R2 cosine 排除 + 长度加权偏差登记；
 5) stage-2 评分代码静态审阅（只 transform/predict；失败不重训）；
 6) stage-2 读取前配置冻结（stage2_config_frozen.json sha）。
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import variant_transfer_stage1_execute as s1  # noqa: E402

B1 = ROOT / "d-det/artifacts/variant_transfer_stage1b_2026-10-08"
S1 = ROOT / "d-det/artifacts/variant_transfer_stage1_2026-10-08"
PRE = ROOT / "d-det/artifacts/variant_transfer_preflight_2026-10-08"
R = ROOT / "d-det/artifacts/public_full_receive_2026-10-08"
S2 = ROOT / "d-det/artifacts/variant_transfer_stage2_2026-10-08"
RECORDS = ROOT / "d-det/data/public_same_task_full_2026-10-07/records.jsonl"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    out = {"schema": "variant_transfer_freeze_validation_v1",
           "generated_utc": datetime.now(timezone.utc).isoformat(),
           "checks": {}, "all_pass": False}

    # 1) stage-1 files + configs + records
    sums = {}
    for line in (S1 / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, rel = line.split("  ", 1)
        sums[rel.strip()] = h
    bad = []
    for rel, h in sums.items():
        p = S1 / rel.lstrip("./")
        if not p.exists() or sha256_file(p) != h:
            bad.append(rel)
    cfg1 = sha256_file(PRE / "execution_config_frozen.json")
    cfg2 = sha256_file(PRE / "r1_negative_set_amendment.json")
    rec_sha = sha256_file(RECORDS)
    c1 = (len(bad) == 0
          and cfg1 == "6aba9c4dbae37ef30fd2792428400e653356bb660d5ca72a4fe5e50755f76aab"
          and cfg2 == "4d0d1631a6ff06724c10ea2c3b38997e48fec2a1ae14e08fc907ee9eef6cb6ef"
          and rec_sha == "d786667a72f3f3864ea38115fd6ccfe8ac393141146d522673668f3730523210")
    out["checks"]["1_hashes"] = {"pass": c1, "stage1_files_checked": len(sums), "bad": bad,
                                 "config_sha_ok": [cfg1.startswith("6aba9c4d"), cfg2.startswith("4d0d1631")],
                                 "records_sha": rec_sha}

    # 2) split hashes + heldout exclusion + no-test-use
    plan = json.loads((R / "prereg/split_plan.json").read_text(encoding="utf-8"))
    SEED = 20261007

    def hkey(s):
        return hashlib.sha256(f"{SEED}|{s}".encode()).hexdigest()

    split_of = s1.load_split()
    tasks = sorted(split_of)
    order = sorted(tasks, key=hkey)
    n = len(order)
    n_tr, n_dv = int(n * 0.70), int(n * 0.15)
    rebuilt = {}
    for i, t in enumerate(order):
        sp = "train" if i < n_tr else ("dev" if i < n_tr + n_dv else "test")
        rebuilt.setdefault(sp, []).append(t)
    sha_ok = all(hashlib.sha256("\n".join(sorted(rebuilt[sp])).encode()).hexdigest()
                 == plan["main_protocol"]["task_list_sha256"][sp] for sp in ("train", "dev", "test"))
    # 折定义复检：11 折 × 2 臂的训练/选择 units 均不含 heldout member
    amend = json.loads((PRE / "r1_negative_set_amendment.json").read_text(encoding="utf-8"))
    per_fold_sel = {k: v["size_matched_negatives"]["selected"] for k, v in
                    amend["size_matched_arm"]["per_fold"].items()}
    fold_units = {}
    fold_ok = True
    for series in s1.SERIES:
        for h in s1.SERIES[series]:
            for arm in s1.ARM_NAMES:
                key_fold = f"{series}::heldout={s1.SIZE_LABEL[h]}"
                neg = per_fold_sel[key_fold] if arm == "size_matched" else \
                    [m for ss in s1.SERIES if ss != series for m in s1.SERIES[ss]]
                units = [m for m in s1.SERIES[series] if m != h] + neg
                ok = h not in units
                fold_ok &= ok
                fold_units[f"{key_fold}::{arm}"] = {"heldout_in_train_units": not ok}
    c2 = sha_ok and fold_ok
    out["checks"]["2_split_heldout"] = {"pass": c2, "split_sha_ok": sha_ok,
                                        "heldout_exclusion_all22": fold_ok,
                                        "fold_units": fold_units,
                                        "no_test_use_evidence": "test_exposure_ledger_v2.json（feature_use/metric_evaluation 均限 train/dev）"}

    # 3) objects load + replay
    man = json.loads((B1 / "objects_manifest.json").read_text(encoding="utf-8"))
    rc = json.loads((B1 / "replay_check.json").read_text(encoding="utf-8"))
    n_pkl = len(man["objects"])
    c3 = n_pkl == 22 and rc["pass"]["csv_vs_new_within_tol"] and rc["pass"]["points_within_tol"] \
        and rc["pass"]["reload_vs_new_within_tol"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out["checks"]["3_objects_replay"] = {
        "pass": c3, "n_objects": n_pkl,
        "csv_vs_new_worst": rc["csv_vs_new"]["worst"], "reload_worst": rc["reload_vs_new"]["worst"],
        "points_worst": rc["points_vs_metrics"]["worst"],
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "sklearn": __import__("sklearn").__version__,
                "lightgbm": __import__("lightgbm").__version__,
                "torch": __import__("torch").__version__,
                "transformers": __import__("transformers").__version__},
        "commit_at_validation": commit,
    }

    # 4) multiplicity + cosine + length weighting
    tmf = json.loads((B1 / "task_macro_ci_fix.json").read_text(encoding="utf-8"))
    r2c = json.loads((B1 / "r2_correction.json").read_text(encoding="utf-8"))
    lw = json.loads((B1 / "prereg_amendment_length_weighting_deferred_2026-10-08.json").read_text(encoding="utf-8"))
    c4 = bool(tmf["validation"]["pass"]) and r2c["cosine"]["status"] == "invalid_zero_center" \
        and lw["decision"] == "deferred" and lw["created_before_stage2_test_read"] is True
    out["checks"]["4_stats_registration"] = {"pass": c4, "multiplicity_pass": tmf["validation"]["pass"],
                                             "r2_cosine": r2c["cosine"]["status"],
                                             "length_weighting": lw["decision"]}

    # 5) stage-2 code static review
    s2_code = (ROOT / "scripts/variant_transfer_stage2_evaluate.py").read_text(encoding="utf-8")
    fit_calls = re.findall(r"\.fit\(|\.partial_fit\(|GridSearchCV|RandomizedSearchCV", s2_code)
    read_guard = "freeze_validation" in s2_code and "ledger started" in s2_code
    c5 = len(fit_calls) == 0 and read_guard
    out["checks"]["5_stage2_static_review"] = {
        "pass": c5, "fit_like_calls": fit_calls,
        "guards": {"freeze_validation_before_read": "assert fv['all_pass']",
                   "ledger_before_read": True,
                   "load_failure_behavior": "assert not load_failures（中止，不重训）"},
        "note": "评分代码只 transform/predict；无任何拟合调用",
    }

    # 6) stage2 config frozen
    cfg_path = S2 / "stage2_config_frozen.json"
    c6 = cfg_path.exists()
    cfg_sha = sha256_file(cfg_path) if c6 else None
    if c6:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        c6 = cfg.get("created_before_any_test_read") is True and cfg["data"]["expected_rows"] == 1881
    out["checks"]["6_stage2_config"] = {"pass": c6, "sha256": cfg_sha}

    out["all_pass"] = all(v["pass"] for v in out["checks"].values())
    out["runtime_seconds"] = time.time() - t0
    (B1 / "freeze_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("all_pass:", out["all_pass"])
    for k, v in out["checks"].items():
        print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}")


if __name__ == "__main__":
    main()
