"""post-stage2 Phase B：主干可用性与权重 hash 登记（未执行则 not_executed）。

依据《d-det_AutoDL_后续干预与主干探针指导_2026-10-08》§6/§8.5：
- 不下载新权重；只登记服务器本地已有权重（记录字节 hash）；
- encoder-only 与 decoder-only 代码模型若本地不存在 → not_executed（等待单独授权）；
- encoder-decoder 基线 = CodeT5-small / CodeT5-base（现有）。

输出：backbone_availability.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "d-det/artifacts/post_stage2_intervention_2026-10-08"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def model_info(path: Path, label: str) -> dict:
    info = {"label": label, "path": str(path.relative_to(ROOT)), "available": path.exists()}
    if not path.exists():
        return info
    w = path / "pytorch_model.bin"
    info["weights_file"] = "pytorch_model.bin"
    info["weights_sha256"] = sha256_file(w)
    info["weights_bytes"] = w.stat().st_size
    cfg = json.loads((path / "config.json").read_text())
    info["arch"] = cfg.get("architectures")
    info["d_model"] = cfg.get("d_model")
    info["num_layers"] = cfg.get("num_layers")
    info["max_length_used_in_protocol"] = 512
    info["tokenizer"] = "roberta (CodeT5)" if (path / "vocab.json").exists() else cfg.get("tokenizer_class")
    return info


def main() -> None:
    out = {"schema": "post_stage2_backbone_availability_v1",
           "generated_utc": datetime.now(timezone.utc).isoformat(),
           "note": ("不下载新权重（指导 §2）；encoder-only/decoder-only 代码模型本地不存在 → not_executed，"
                    "等待单独授权后再执行；编码器网络访问=off（local_files_only=True）。"),
           "encoder_decoder": [
               model_info(ROOT / "d-det/models/codet5-small", "codet5-small (existing baseline)"),
               model_info(ROOT / "d-det/checkpoints/codet5-base", "codet5-base (existing baseline)"),
           ],
           "encoder_only": {"status": "not_executed",
                            "reason": "本地无 encoder-only 代码模型权重（unixcoder/codebert 等不存在）；未获下载授权"},
           "decoder_only": {"status": "not_executed",
                            "reason": "本地 HF 缓存已清理，无 decoder-only 权重（Qwen/DeepSeek/Yi/SmolLM2 均已删除）；未获下载授权"},
           "probe_design_if_authorized": {
               "minimal_2x3": "encoder-only / encoder-decoder / decoder-only × (fixed LR + style_lgb control) × (原始 + A2_comments_masked)",
               "frozen": True, "same_split": True, "same_bootstrap": "task-cluster 500 seed 20261008",
           }}
    (OUT / "backbone_availability.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    print(json.dumps(out["encoder_decoder"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
