"""系列准入（服务器侧重构版）：从完整语料自身登记三系列 11 成员与协议一致性。

说明：官方 build_family_series_admission.py 未随包到达（expected manifest 16/16 缺失）；
本版按指导 §2 的系列表在 records/adjudication 上做只读推导，输出
public_full_followup_2026-10-08/family_series_admission.json（server_rebuild 标注）。
family_is_confirmed 全局仍为 false；本文件只登记 model-series membership。
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import re

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "d-det/data/public_same_task_full_2026-10-07"
OUT = ROOT / "d-det/artifacts/public_full_followup_2026-10-08"

SERIES = {
    "CodeLlama-Instruct": {
        "pattern": re.compile(r"^codellama--CodeLlama-(7b|13b|34b|70b)-Instruct(-hf)?$"),
        "sizes": ["7b", "13b", "34b", "70b"],
        "docs": ["official_docs_server_fetch/CodeLlama-MODEL_CARD.md"],
    },
    "Qwen2.5-Coder-Instruct": {
        "pattern": re.compile(r"^Qwen--Qwen2\.5-Coder-(1\.5B|7B|14B|32B)-Instruct$"),
        "sizes": ["1.5B", "7B", "14B", "32B"],
        "docs": ["official_docs_server_fetch/Qwen2.5-Coder-README.md"],
    },
    "DeepSeek-Coder-v1-Instruct": {
        "pattern": re.compile(r"^deepseek-ai--deepseek-coder-(1\.3b|6\.7b|33b)-instruct$"),
        "sizes": ["1.3b", "6.7b", "33b"],
        "docs": ["official_docs_server_fetch/DeepSeek-Coder-README.md"],
    },
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    adj = [json.loads(l) for l in (FULL / "model_lineage_adjudication.jsonl").open(encoding="utf-8")]
    model2series = {}
    for s, spec in SERIES.items():
        for x in adj:
            if spec["pattern"].match(x["model_id"]):
                model2series[x["model_id"]] = s
    print("matched members:", len(model2series))

    # 单遍：四个协议切片的行数/任务/member 路径/backend
    stats = defaultdict(lambda: {"rows": defaultdict(int), "tasks": defaultdict(set),
                                 "assets": defaultdict(set), "members": defaultdict(set),
                                 "backends": defaultdict(set), "temperatures": defaultdict(set)})
    with (FULL / "records.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["source"] != "bigcodebench" or r["model_id"] not in model2series:
                continue
            key = r["model_id"]
            sub = f"{r.get('subset')}|{r.get('generation_mode')}"
            st = stats[key]
            st["rows"][sub] += 1
            st["tasks"][sub].add(r["task_id"])
            st["assets"][sub].add(r.get("asset", ""))
            st["members"][sub].add(r.get("member", ""))
            st["backends"][sub].add(str(r.get("backend", "")))
            st["temperatures"][sub].add(str(r.get("temperature", "")))

    admission = {
        "schema": "public_family_series_admission_v1",
        "source": "server_rebuild（官方包 family_series_admission.json 未上传；expected manifest 16/16 missing）",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "family_is_confirmed": False,
        "scope": "model-series membership（size/variant transfer 用），不声明 unseen family / 后训练因果 / 无污染",
        "series": {},
        "claims_supported": ["同一官方系列内的 size/variant 成员关系（据发布命名 + 官方资料 URL）",
                             "系列内留一尺寸变体迁移 pilot 的成员清单"],
        "claims_not_supported": ["unseen independent family", "post-training causal effect",
                                 "contamination-free confirmatory benchmark",
                                 "checkpoint 字节/历史生成参数的真实性（未验证）"],
        "evidence": {"fetched_docs_dir": "official_docs_server_fetch/",
                     "sources": "official_docs_server_fetch/sources.json",
                     "expected_original_pack": "expected_followup_manifest_via_user.txt（16 文件全部缺失）"},
    }
    total = 0
    for s, spec in SERIES.items():
        members = sorted([m for m, ss in model2series.items() if ss == s],
                         key=lambda m: spec["sizes"].index(spec["pattern"].match(m).group(1)))
        entry = {"members": {}, "n_members": len(members), "n_sizes": len(spec["sizes"]),
                 "docs": spec["docs"], "path_consistency": {}}
        for m in members:
            st = stats[m]
            subs = sorted(st["rows"])
            entry["members"][m] = {
                "rows_by_slice": {k: st["rows"][k] for k in subs},
                "tasks_by_slice": {k: len(st["tasks"][k]) for k in subs},
                "assets": {k: sorted(st["assets"][k]) for k in subs},
                "members_per_slice": {k: len(st["members"][k]) for k in subs},
                "backends": {k: sorted(st["backends"][k]) for k in subs},
                "temperatures": {k: sorted(st["temperatures"][k]) for k in subs},
            }
            total += 1
        # 一致性：主协议 rows==1140 & tasks==1140；每个 slice 单 asset；backend 归一化后一致
        cons = {"backend_naming_notes": [], "ok": True}
        for m in members:
            st = stats[m]
            if st["rows"].get("full|instruct", 0) != 1140 or len(st["tasks"].get("full|instruct", set())) != 1140:
                cons["ok"] = False
                cons["backend_naming_notes"].append(f"{m}: full|instruct 覆盖异常")
            bks = set()
            for k, s2 in st["backends"].items():
                bks |= {b.replace("_", "-") for b in s2 if b and b != "None"}
            if len(bks) > 1:
                cons["backend_naming_notes"].append(f"{m}: 归一化后 backend 仍多值 {sorted(bks)}")
            for k, a in st["assets"].items():
                if len(a) != 1:
                    cons["backend_naming_notes"].append(f"{m}: {k} 多 asset {sorted(a)}")
        entry["path_consistency"] = cons
        admission["series"][s] = entry
    admission["totals"] = {"members": total}
    admission["sha_self_note"] = "本文件为 server_rebuild；原版到达后以其 sha256 为准（expected: cc4a7785…）"
    (OUT / "family_series_admission.server_rebuild.json").write_text(
        json.dumps(admission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"admission": "written", "members": total,
                      "series": {s: admission["series"][s]["n_members"] for s in SERIES},
                      "runtime_s": round(time.time() - t0, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
