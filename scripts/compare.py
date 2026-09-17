#!/usr/bin/env python
"""汇总 runs/ 下的实验指标（单文件脚本）。

读取每个 run 的：
    config.yaml     -> 模型/编码器/报告/流 等设置
    metrics.jsonl   -> 各 epoch 的验证指标（取最后一个 epoch）
    eval.json       -> `train.py --eval` 写出的 val/test 指标（可选）

用法::

    python scripts/compare.py                # 列出 runs/ 下所有实验
    python scripts/compare.py --sort test_m4_f1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = {"m4": "sample_f1", "hybrid": "line_f1"}      # 每个数据集的主指标


def load_run(run_dir: Path) -> dict:
    info = {"run": run_dir.name}
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        info["model"] = cfg.get("model", {}).get("name", "?")
        info["encoder"] = cfg.get("encoder", {}).get("name", "?")
        info["freeze"] = cfg.get("encoder", {}).get("freeze", None)
        info["report"] = cfg.get("report", {}).get("name", "?")
        info["streams"] = ",".join(cfg.get("train", {}).get("streams", []))

    best: dict[str, tuple] = {}                          # 数据集 -> (主指标, epoch, 全部指标)
    jsonl = run_dir / "metrics.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            for name, metrics in record.get("val", {}).items():
                score = metrics.get(PRIMARY.get(name, "sample_f1"))
                if score is None:
                    continue
                if name not in best or score > best[name][0]:
                    best[name] = (score, record.get("epoch"), metrics)
    for name, (score, epoch, metrics) in best.items():
        info[f"val_{name}_epoch"] = epoch
        info[f"val_{name}_f1"] = score
        info[f"val_{name}_acc"] = metrics.get("sample_acc")
        info[f"val_{name}_line_f1"] = metrics.get("line_f1")
        info[f"val_{name}_token_f1"] = metrics.get("token_f1")

    eval_path = run_dir / "eval.json"
    if eval_path.exists():
        blob = json.loads(eval_path.read_text(encoding="utf-8"))
        info["eval_epoch"] = blob.get("epoch")
        for key, metrics in blob.get("metrics", {}).items():
            name, _, split = key.partition("/")           # "m4/test" -> m4 / test
            info[f"{split}_{name}_f1"] = metrics.get("sample_f1", metrics.get("line_f1"))
            info[f"{split}_{name}_acc"] = metrics.get("sample_acc")
            info[f"{split}_{name}_line_f1"] = metrics.get("line_f1")
            info[f"{split}_{name}_token_f1"] = metrics.get("token_f1")
            info[f"{split}_{name}_chunk_f1"] = metrics.get("chunk_f1")
    return info


def fmt(value, width: int = 16) -> str:
    """统一单元格格式（数值保留 4 位，其余转字符串）。"""
    if value is None:
        text = "-"
    elif isinstance(value, bool):
        text = "True" if value else "False"
    elif isinstance(value, (int, float)):
        text = f"{value:.4f}"
    else:
        text = str(value)
    return f"{text[:width]:<{width}}"


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 runs/ 下的实验指标")
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"))
    parser.add_argument("--sort", default="val_m4_f1")
    args = parser.parse_args()

    run_dirs = sorted(p for p in Path(args.runs_dir).glob("*") if (p / "metrics.jsonl").exists())
    if not run_dirs:
        print(f"{args.runs_dir} 下没有完成的实验")
        return 1
    rows = [load_run(p) for p in run_dirs]
    rows.sort(key=lambda r: r.get(args.sort) or 0.0, reverse=True)

    cols = ["run", "model", "encoder", "freeze", "report", "streams",
            "val_m4_epoch", "val_m4_f1", "val_hybrid_line_f1",
            "test_m4_f1", "test_m4_acc", "test_hybrid_line_f1", "test_hybrid_token_f1"]
    print("  ".join(fmt(c) for c in cols))
    for row in rows:
        print("  ".join(fmt(row.get(c)) for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
