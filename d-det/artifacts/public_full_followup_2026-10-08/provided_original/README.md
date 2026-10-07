# provided_original/（原始交接包放置位置）

上游指导（`d-det/docx/d-det_AutoDL_变体迁移执行前闸门指导_2026-10-08.md` §1）要求：原件到达后放入本目录，
逐文件核对 `expected_followup_manifest_via_user.txt` 中的 16 项 sha256。

## 放置方式

将指导方交接包中的 16 个文件按清单相对路径放入本目录，例如：

```
provided_original/
  family_series_admission.json
  member_protocol_and_series.jsonl
  official_docs/{CodeContests-schema.txt,CodeLlama-Instruct.txt,DeepSeek-Coder-v1-Instruct.txt,Qwen2.5-Coder-Instruct.txt}
  official_documentation_sources.json
  proposed_family_checkpoint_protocol.json
  source_snapshot/{cc_riegeli_extract.py,code_contests_index.json,full_adjudication_audit.json,public_full_adjudication_audit.py,received_files_sha256.json,split_plan.json,unit_coverage_matrix.csv,unit_folds.json}
```

## 核对流程

1. 放入后重跑 `scripts/variant_transfer_preflight_provenance.py`；
2. 每项写入 `../provenance_resolution.json`：`expected_sha256` / `observed_sha256` / `source` / `replacement_time` / `decision`
   （`verified_original` | `mismatch_quarantined`）；
3. 16/16 `verified_original` 前：`server_rebuild/` 不删除、不覆盖；重构版不得写成原件哈希。

## 当前状态

**空** —— 16/16 仍缺失（2026-10-08 核）。已有服务器侧对应物仅作旁证（见 `provenance_resolution.json` 的
`server_rebuild_counterpart.sha256_matches_expected` 字段），**不构成本机交付核验**。
