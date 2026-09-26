#!/usr/bin/env python
"""选择性敏感性验证（设计文档 §7 注记 2 的"防退化"协议；有卡时运行）。

对同一批代码施加扰动，看两维分数是否**选择性**响应：

    P1 删 docstring / 注释     —— "偏好规范化"维度，期望主要动 s2
    P2 紧凑排版（去空行 / 行尾空白）—— 同为排版偏好维度，期望主要动 s2

逐样本指标：

    Δ1 = |s1(x') − s1(x)|      Δ2 = ‖s2(x') − s2(x)‖（r=1 即 |Δs2|）

    selectivity = E[Δ2] / (E[Δ1] + E[Δ2])   （接近 1 = 该扰动几乎只被 s2 感知）

它回答的核心问题：两维有没有坍缩成同一个投影。若删 docstring 让 s1 与 s2 同幅度
变化 ⇒ 读出未分离（或该信号跨维），need 回到读出去相关 / 数据构造检查。

用法（有卡）::

    python scripts/probe_sensitivity.py --config configs/ddet_base.yaml \\
        --ckpt runs/v0.1.0/best.pt --n 200 --out runs/v0.1.0/sensitivity.json

自定义扰动样本对（jsonl，每行 {"code": "...", "perturbed": "..."}）::

    python scripts/probe_sensitivity.py --ckpt ... --custom-file my_cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoders import build_encoder                     # noqa: E402
from models import build_model                         # noqa: E402
from train import apply_overrides, load_config         # noqa: E402


# --------------------------------------------------------------------------- #
# 扰动（规则版；"冷门人类风格改写"类扰动需要 LLM，留到后续扩展）
# --------------------------------------------------------------------------- #
def strip_comments_docstrings(code: str) -> str:
    """删三引号块与单行注释（启发式，够探测用；不做字符串内 # 的精细处理）。"""
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    code = re.sub(r"'''[\s\S]*?'''", "", code)
    code = re.sub(r"(?m)^\s*#[^\n]*$", "", code)      # 整行注释
    code = re.sub(r"(?<=[)\w\s])\s+#[^\n]*", "", code)  # 行尾注释（保守：前面是代码字符）
    return code


def strip_blank_lines(code: str) -> str:
    """紧凑排版：删空行 + 行尾空白。"""
    lines = [line.rstrip() for line in code.split("\n")]
    return "\n".join(line for line in lines if line.strip())


def rename_params(code: str) -> str:
    """重命名 Python def 的短参数（信息保持的风格扰动；全词替换，含注释/字符串内）。

    只处理 1~3 字符的纯标识符参数，避免大范围误伤；找不到简单参数表则原样返回。
    """
    m = re.search(r"(?m)^( *)def\s+\w+\s*\(([^)]*)\)", code)
    if not m:
        return code
    names = []
    for part in m.group(2).split(","):
        name = part.strip().split("=")[0].strip().lstrip("*").strip()
        if re.fullmatch(r"[a-z_][a-z0-9_]{0,2}", name or ""):
            names.append(name)
    for i, name in enumerate(names):
        code = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                      f"arg{i}", code)
    return code


def swap_quotes(code: str) -> str:
    """双引号 → 单引号（启发式；跳过含单引号或超长/跨行的字面量）。"""

    def repl(m):
        inner = m.group(1)
        if "'" in inner or len(inner) > 200:
            return m.group(0)
        return "'" + inner + "'"

    return re.sub('"([^"\n]*)"', repl, code)


def indent_halve(code: str) -> str:
    """行首缩进 4 空格 → 2 空格（信息保持；仅当确有 4 的倍数缩进时生效）。"""
    changed = False
    out = []
    for line in code.split("\n"):
        m = re.match(r"^( +)\S", line)
        if m:
            spaces = len(m.group(1))
            if spaces >= 4 and spaces % 4 == 0:
                line = " " * (spaces // 2) + line[spaces:]
                changed = True
        out.append(line)
    return "\n".join(out) if changed else code


def add_docstring(code: str) -> str:
    """给首个 Python 函数注入一行 docstring（偏好方向的对照扰动，s2 应感知）。"""
    m = re.search(r"(?m)^( *)def\s+\w+\s*\([^)]*\)[^:\n]*: *$", code)
    if not m:
        return code
    indent = m.group(1) + "    "
    pos = m.end()
    return code[:pos] + "\n" + indent + '"""TODO: describe."""' + code[pos:]


PERTURBATIONS = {
    "strip_comments": strip_comments_docstrings,
    "compact_lines": strip_blank_lines,
    # v2（信息保持型；2026-09-23 增补，读数与解读见 docx/d-det-v0.2.md §2）
    "rename_params": rename_params,
    "swap_quotes": swap_quotes,
    "indent_halve": indent_halve,
    "add_docstring": add_docstring,
}


# --------------------------------------------------------------------------- #
def load_codes(cfg: dict, source: str, split: str, n: int) -> list:
    """取扰动样本（m4 的 code 字段 / pair 的 x_plus 文本）。"""
    from dataio import build_dataset

    data_cfg = cfg["data"]
    path = ROOT / data_cfg["processed_dir"] / data_cfg[f"{source}_file"]
    if not path.exists():
        raise SystemExit(f"[sensitivity] 找不到 {path}")
    dataset = build_dataset(source, file=str(path), split=split)
    codes = (dataset.codes if source == "m4" else dataset.codes_plus)
    step = max(1, len(codes) // max(n, 1))
    return [codes[i] for i in range(0, len(codes), step)][:n]


@torch.no_grad()
def scores_of(model, tokenizer, code: str, device: str):
    """返回 (s1 标量, s2 向量 (r,))。"""
    ids = tokenizer(code, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    s1, s2 = model(ids, torch.ones_like(ids))
    return float(s1.item()), s2.squeeze(0)


def main() -> int:
    parser = argparse.ArgumentParser(description="选择性敏感性验证（s1 / s2 防退化协议）")
    parser.add_argument("--config", default="configs/ddet_base.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--source", default="m4", choices=["m4", "pair"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=200, help="抽样条数")
    parser.add_argument("--custom-file", default=None,
                        help="jsonl 自定义扰动对（每行 {code, perturbed}）；给了就只跑它")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--set", action="append", default=None, metavar="段.键=值")
    parser.add_argument("--out", default=None, help="结果 json 路径（默认 runs 旁）")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    enc_cfg = dict(cfg["encoder"])
    if not Path(enc_cfg["path"]).is_absolute():
        enc_cfg["path"] = str(ROOT / enc_cfg["path"])        # 允许从任意 cwd 运行
    encoder = build_encoder(**enc_cfg)
    mcfg = dict(cfg["model"])
    model = build_model(mcfg.pop("name", "dual"), encoder=encoder,
                        dim=encoder.hidden_size, **mcfg)
    ckpt_path = Path(args.ckpt) if Path(args.ckpt).is_absolute() else ROOT / args.ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
    if missing or unexpected:
        print(f"[sensitivity] 注意：未加载 {len(missing)} 项、多余 {len(unexpected)} 项")
    model.to(device).eval()
    print(f"[sensitivity] 载入 {args.ckpt}（epoch {ckpt.get('epoch')}）device={device}")

    tokenizer = AutoTokenizer.from_pretrained(enc_cfg["path"])

    # ---- 组装 (原始, 扰动, 扰动名) 三元组 ----
    cases = []
    if args.custom_file:
        with open(args.custom_file, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                cases.append((row["code"], row["perturbed"], row.get("name", "custom")))
        print(f"[sensitivity] 自定义扰动对：{len(cases)} 条")
    else:
        codes = load_codes(cfg, args.source, args.split, args.n)
        for name, fn in PERTURBATIONS.items():
            for code in codes:
                perturbed = fn(code)
                if perturbed.strip() != code.strip():
                    cases.append((code, perturbed, name))
    if not cases:
        raise SystemExit("[sensitivity] 没有可用样本（检查数据源 / 扰动是否改变文本）")

    # ---- 前向对比 ----
    groups: dict = {}
    cache: dict = {}                     # 同一原样本可能被多个扰动共用 → 缓存原分数
    for code, perturbed, name in cases:
        if code not in cache:
            cache[code] = scores_of(model, tokenizer, code, device)
        s1_o, s2_o = cache[code]
        s1_p, s2_p = scores_of(model, tokenizer, perturbed, device)
        delta1 = abs(s1_p - s1_o)
        delta2 = float((s2_p - s2_o).norm())
        group = groups.setdefault(name, {"d1": [], "d2": []})
        group["d1"].append(delta1)
        group["d2"].append(delta2)

    # ---- 汇总 ----
    summary = {}
    for name, values in groups.items():
        d1, d2 = values["d1"], values["d2"]
        mean1 = statistics.fmean(d1)
        mean2 = statistics.fmean(d2)
        selectivity = mean2 / (mean1 + mean2 + 1e-12)
        summary[name] = {
            "n": len(d1),
            "delta_s1_mean": mean1,
            "delta_s1_median": statistics.median(d1),
            "delta_s2_mean": mean2,
            "delta_s2_median": statistics.median(d2),
            "selectivity": selectivity,     # →1 = 只被 s2 感知
        }
        print(f"[{name}] n={len(d1)}  Δs1(mean)={mean1:.4f}  Δs2(mean)={mean2:.4f}  "
              f"selectivity={selectivity:.3f}")

    print("[解读] selectivity 接近 1 ⇒ 该扰动几乎只被 s2 感知（两维未退化）；\n"
          "       接近 0.5 ⇒ 两维同幅响应（读出未分离，或扰动本身跨维）；\n"
          "       接近 0   ⇒ 几乎只被 s1 感知（若 P1/P2 出现这种情况，说明『偏好维』"
          "没有落到 s2 上，需要检查监督信号）。")

    out = Path(args.out) if args.out else ckpt_path.parent / "sensitivity.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ckpt": str(args.ckpt), "source": args.source, "split": args.split,
                   "groups": summary}, f, ensure_ascii=False, indent=2)
    print(f"[sensitivity] 结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
