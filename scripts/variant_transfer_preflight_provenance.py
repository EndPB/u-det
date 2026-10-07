"""变体迁移执行前闸门（2026-10-08）§1：交接包身份核对。

生成 provenance_resolution.json：
- 对 expected_followup_manifest_via_user.txt 的 16 项逐文件核对；
- provided_original/（原件到达位置）逐一计算 observed sha256；
- 同时登记 server_rebuild/ 与服务器侧对应物（仅作旁证，不构成原件核验）；
- 政策：16/16 通过前 server_rebuild/ 不删不覆盖；重构版不得写成原件哈希。

只读：不读 records 正文、不读 test、不读模型权重。
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "d-det/artifacts/public_full_followup_2026-10-08"
R = ROOT / "d-det/artifacts/public_full_receive_2026-10-08"
MANIFEST = F / "expected_followup_manifest_via_user.txt"
PROVIDED = F / "provided_original"
SR = F / "server_rebuild"
OUT = F / "provenance_resolution.json"

# expected 清单文件 -> (server 侧对应物, relationship, note)
COUNTERPARTS: dict[str, tuple[str | None, str, str]] = {
    "family_series_admission.json": (
        "d-det/artifacts/public_full_followup_2026-10-08/server_rebuild/family_series_admission.server_rebuild.json",
        "reconstructed_from_official_sources",
        "服务器重构版（official URL + commit pin）；非原件",
    ),
    "member_protocol_and_series.jsonl": (None, "absent", "无服务器侧对应物"),
    "official_docs/CodeContests-schema.txt": (
        "d-det/artifacts/public_full_followup_2026-10-08/server_rebuild/official_docs_server_fetch/contest_problem.proto",
        "reconstructed_from_official_sources",
        "官方 proto 原始文件（commit pin fa7a4f81）",
    ),
    "official_docs/CodeLlama-Instruct.txt": (
        "d-det/artifacts/public_full_followup_2026-10-08/server_rebuild/official_docs_server_fetch/CodeLlama-MODEL_CARD.md",
        "reconstructed_from_official_sources",
        "官方 MODEL_CARD（commit pin e81b597e）；curated 文本可能不同",
    ),
    "official_docs/DeepSeek-Coder-v1-Instruct.txt": (
        "d-det/artifacts/public_full_followup_2026-10-08/server_rebuild/official_docs_server_fetch/DeepSeek-Coder-README.md",
        "reconstructed_from_official_sources",
        "官方 README（commit pin 2f9fd859）；curated 文本可能不同",
    ),
    "official_docs/Qwen2.5-Coder-Instruct.txt": (
        "d-det/artifacts/public_full_followup_2026-10-08/server_rebuild/official_docs_server_fetch/Qwen2.5-Coder-README.md",
        "reconstructed_from_official_sources",
        "官方 README（commit pin 5948e971）；curated 文本可能不同",
    ),
    "official_documentation_sources.json": (
        "d-det/artifacts/public_full_followup_2026-10-08/server_rebuild/official_docs_server_fetch/sources.json",
        "reconstructed_from_official_sources",
        "服务器侧来源记录（fetch 版）",
    ),
    "proposed_family_checkpoint_protocol.json": (None, "absent", "无服务器侧对应物"),
    "source_snapshot/cc_riegeli_extract.py": (
        "scripts/cc_riegeli_extract.py", "server_side_script_copy", "本仓脚本（上传版同源）",
    ),
    "source_snapshot/code_contests_index.json": (
        "d-det/artifacts/public_full_receive_2026-10-08/audit/code_contests_index.json",
        "server_side_receive_copy", "接收阶段产物",
    ),
    "source_snapshot/full_adjudication_audit.json": (
        "d-det/artifacts/public_full_receive_2026-10-08/audit/full_adjudication_audit.json",
        "server_side_receive_copy_modified",
        "接收后于 811ecfe 重跑更新（306 口径/D1 Δ 修正）；服务器副本 sha 已变，原始快照以期望 sha 为准",
    ),
    "source_snapshot/public_full_adjudication_audit.py": (
        "scripts/public_full_adjudication_audit.py", "server_side_script_copy", "本仓脚本（上传版同源）",
    ),
    "source_snapshot/received_files_sha256.json": (
        "d-det/artifacts/public_full_receive_2026-10-08/audit/received_files_sha256.json",
        "server_side_receive_copy", "接收阶段产物",
    ),
    "source_snapshot/split_plan.json": (
        "d-det/artifacts/public_full_receive_2026-10-08/prereg/split_plan.json",
        "server_side_receive_copy", "接收阶段产物（预注册 split）",
    ),
    "source_snapshot/unit_coverage_matrix.csv": (
        "d-det/artifacts/public_full_receive_2026-10-08/audit/unit_coverage_matrix.csv",
        "server_side_receive_copy", "接收阶段产物",
    ),
    "source_snapshot/unit_folds.json": (
        "d-det/artifacts/public_full_receive_2026-10-08/prereg/unit_folds.json",
        "server_side_receive_copy", "接收阶段产物（unit 5 折）",
    ),
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_manifest() -> list[tuple[str, str]]:
    items = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        assert len(parts) == 2, line
        items.append((parts[0].strip(), parts[1].strip()))
    return items


def main() -> None:
    t0 = time.time()
    items = parse_manifest()
    assert len(items) == 16, len(items)
    rows = []
    n_verified = n_mismatch = n_await = 0
    for exp_sha, rel in items:
        provided_path = PROVIDED / rel
        cp_rel, relationship, note = COUNTERPARTS.get(rel, (None, "absent", ""))
        cp_path = ROOT / cp_rel if cp_rel else None
        cp_sha = sha256_file(cp_path) if cp_path and cp_path.exists() else None
        row = {
            "file": rel,
            "expected_sha256": exp_sha,
            "provided_original_path": str(provided_path.relative_to(ROOT)),
            "observed_sha256": None,
            "observed_source": None,
            "observed_mtime_utc": None,
            "replacement_time": None,
            "source": None,
            "decision": None,
            "server_rebuild_counterpart": {
                "path": cp_rel,
                "sha256": cp_sha,
                "sha256_matches_expected": (cp_sha == exp_sha) if cp_sha else None,
                "relationship": relationship,
                "note": note,
                "counts_as_original_verification": False,
            },
        }
        if provided_path.exists():
            obs = sha256_file(provided_path)
            mt = datetime.fromtimestamp(provided_path.stat().st_mtime, tz=timezone.utc).isoformat()
            row.update({
                "observed_sha256": obs,
                "observed_source": "provided_original",
                "observed_mtime_utc": mt,
                "replacement_time": datetime.now(timezone.utc).isoformat(),
                "source": "user_delivered_pack",
            })
            if obs == exp_sha:
                row["decision"] = "verified_original"
                n_verified += 1
            else:
                row["decision"] = "mismatch_quarantined"
                n_mismatch += 1
        else:
            row["decision"] = "awaiting_original"
            n_await += 1
        rows.append(row)

    out = {
        "schema": "variant_transfer_provenance_resolution_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": "expected_followup_manifest_via_user.txt",
        "replacement_time_note": "replacement_time=写入本证据的时刻（原件放入 provided_original/ 后重跑本脚本更新）",
        "policy": {
            "server_rebuild_must_be_kept": "16/16 verified_original 前不得删除或覆盖 server_rebuild/",
            "no_original_hash_claim": "不得把重构版 sha（如 c079abea…）写作原件哈希",
            "long_absent_rule": "原件长期不可达时，重构版定性 server_reconstruction_only；结果不得写成复现本机交接包，也不得用原件 commit/hash 作复现证据",
        },
        "items": rows,
        "summary": {
            "total": len(rows),
            "verified_original": n_verified,
            "mismatch_quarantined": n_mismatch,
            "awaiting_original": n_await,
            "original_pack_complete": n_verified == len(rows),
            "server_reconstruction_only": n_verified < len(rows),
        },
        "gate": {
            "original_pack_complete": n_verified == len(rows),
            "note": "16/16 通过前：本轮回传中一切系列证据引用必须标注 server_reconstruction_only"
                    if n_verified < len(rows) else "原件核对完成",
        },
        "runtime_seconds": time.time() - t0,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print("summary:", json.dumps(out["summary"], ensure_ascii=False))
    for r in rows:
        cp = r["server_rebuild_counterpart"]
        print(f"[{r['decision']:>20s}] {r['file']:<48s} counterpart_match={cp['sha256_matches_expected']}")


if __name__ == "__main__":
    main()
