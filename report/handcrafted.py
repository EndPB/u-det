"""手工统计报告（单文件实现）：把代码的结构/词汇/句法统计渲染成一段短文本，注入到代码前。

三类特征（对应 docx/u-det.md）：
    1. 结构偏置：空行率 R_void、缩进一致性（缩进为基本单位的比例）、尾部空行数
    2. 词汇可预测性：字符香农熵、字符二元组香农熵（bit，底数 2）
    3. 句法方差：命名惯例多样性（惯例分布的归一化熵，0=单一惯例，1=完全混用）、行长方差（字符数方差）

用法::

    from report import build_report

    report = build_report("handcrafted", tokenizer=tok, max_tokens=64)
    ids = report.ids(code)      # 直接拼在代码 token 前的报告 token（只统计一份，统计量取整文件）
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_TEMPLATE = (
    "STATS blank_ratio={void} indent_consistency={indent} trailing_blank={tail} "
    "char_entropy={charh} bigram_entropy={bigh} naming_diversity={namev} line_length_var={linev}"
)


# --------------------------------------------------------------------------- #
# 统计量
# --------------------------------------------------------------------------- #
def _entropy(counts) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def structural_bias(lines: List[str]) -> tuple[float, float, int]:
    """(空行率, 缩进一致性, 尾部空行数)。"""
    n = max(len(lines), 1)
    void = sum(1 for ln in lines if not ln.strip()) / n

    indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    unit = Counter(i for i in indents if i > 0).most_common(1)
    unit = unit[0][0] if unit else 0
    if unit > 0:
        indented = [i for i in indents if i > 0]
        indent = sum(1 for i in indented if i % unit == 0) / max(len(indented), 1)
    else:
        indent = 1.0 if not any(indents) else 0.0

    tail = 0
    for ln in reversed(lines):
        if ln.strip():
            break
        tail += 1
    return void, indent, tail


def lexical_predictability(text: str) -> tuple[float, float]:
    """(字符熵, 字符二元组熵)，单位 bit。"""
    char_h = _entropy(Counter(text).values())
    bigrams = Counter(zip(text[:-1], text[1:])).values()
    return char_h, _entropy(bigrams)


def naming_convention(code: str) -> dict[str, float]:
    """标识符命名惯例占比（snake / upper / pascal / camel / other）。"""
    buckets = {"snake": 0, "upper": 0, "pascal": 0, "camel": 0, "other": 0}
    for word in _IDENT.findall(code):
        if len(word) == 1 and not word.isalpha():
            buckets["other"] += 1
        elif word.isupper():
            buckets["upper"] += 1
        elif word.islower():
            buckets["snake"] += 1
        elif word[0].isupper():
            buckets["pascal"] += 1
        elif word[0].islower():
            buckets["camel"] += 1
        else:
            buckets["other"] += 1
    total = sum(buckets.values()) or 1
    return {k: v / total for k, v in buckets.items()}


def syntactic_variance(code: str, lines: List[str]) -> tuple[float, float]:
    """(命名惯例多样性, 行长方差)。"""
    probs = naming_convention(code)
    name_var = max(0.0, _entropy(probs.values()) / math.log2(len(probs)))   # 归一化到 [0, 1]
    lengths = [len(ln) for ln in lines]
    mean = sum(lengths) / max(len(lengths), 1)
    line_var = sum((x - mean) ** 2 for x in lengths) / max(len(lengths), 1)
    return name_var, line_var


def stats(code: str) -> dict[str, float]:
    """一次性算出报告用到的全部统计量。"""
    lines = code.split("\n")
    void, indent, tail = structural_bias(lines)
    char_h, bigram_h = lexical_predictability(code)
    name_var, line_var = syntactic_variance(code, lines)
    return {
        "void": void, "indent": indent, "tail": tail,
        "charh": char_h, "bigh": bigram_h, "namev": name_var, "linev": line_var,
    }


def text(code: str, precision: int = 3) -> str:
    """渲染成一行短报告（数值量化以控制词表规模）。"""
    s = stats(code)
    return _TEMPLATE.format(
        void=round(max(0.0, s["void"]), precision),
        indent=round(max(0.0, s["indent"]), precision),
        tail=int(s["tail"]),
        charh=round(max(0.0, s["charh"]), 2),
        bigh=round(max(0.0, s["bigh"]), 2),
        namev=round(max(0.0, s["namev"]), precision),
        linev=round(max(0.0, s["linev"]), 1),
    )


# --------------------------------------------------------------------------- #
# 报告对象（报告模块的统一下游接口）
# --------------------------------------------------------------------------- #
class HandcraftedReport:
    """把手工统计渲染成 token 前缀；统计针对整段代码（与窗口无关，保持一致）。"""

    name = "handcrafted"

    def __init__(self, tokenizer=None, max_tokens: int = 64, precision: int = 3):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.precision = precision

    def render(self, code: str) -> str:
        return text(code, self.precision)

    def ids(self, code: str) -> List[int]:
        if self.tokenizer is None:
            raise ValueError("需要 tokenizer 才能得到报告 token：build_report(..., tokenizer=tok)")
        return self.tokenizer(self.render(code), add_special_tokens=False, truncation=True,
                              max_length=self.max_tokens)["input_ids"]


class NullReport:
    """消融用：不注入报告。"""

    name = "none"

    def __init__(self, **kwargs):
        pass

    def render(self, code: str) -> str:
        return ""

    def ids(self, code: str) -> List[int]:
        return []
