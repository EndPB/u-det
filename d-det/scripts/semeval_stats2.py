#!/usr/bin/env python
"""手工艺特征 v2（在 v1 的 19 维之上再扩 ~30 维）——B/C/A 集成用。

新增：缩进轮廓（均值/熵/{1,2,4,8} 占比）、注释行细分（纯注释/inline/块注释）、
字符串风格（单双引号/三引号/f-string/转义）、运算符间距（==/:=/逗号/分号）、
命名风格（camel/snake/ALLCAPS）、行长分布（>79/>120）、行尾分号、Unicode、括号风格。
对 val/test 全部任务计算（train 用 v1 即可，集成走 val-fit）。
输出：data/processed/semeval/{task}_{split}_stats2.npz
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from semeval_stats import FILES, RAW, replay_picks, test_kept  # noqa: E402

OUT = ROOT / "data/processed/semeval"
NAMES2 = ["indent_mean", "indent_entropy", "ind1", "ind2", "ind4", "ind8",
          "pure_comment", "inline_comment", "block_comment", "comment_chars",
          "squote", "dquote", "triple_q", "fstring", "escape",
          "sp_eq", "sp_cmp", "sp_comma", "semicolon_line", "paren_if",
          "camel", "snake", "allcaps", "over79", "over120", "final_nl",
          "nonascii", "blank_before_def", "tabs_mid", "repeat_ws"]


def stats2(code: str) -> list[float]:
    lines = code.split("\n")
    n = max(len(lines), 1)
    ind = [len(ln) - len(ln.lstrip(" \t")) for ln in lines if ln.strip()]
    ind_cnt = Counter(min(x, 16) for x in ind)
    ind_ent = -sum((v / max(len(ind), 1)) * math.log2(v / max(len(ind), 1))
                   for v in ind_cnt.values()) if ind else 0.0
    pure_com = sum(1 for ln in lines if ln.lstrip().startswith(("#", "//", "*")))
    inline_com = sum(1 for ln in lines if " #" in ln or " // " in ln)
    block = 1.0 if "/*" in code else 0.0
    com_chars = sum(len(ln) - len(ln.lstrip()) + (len(ln) - len(ln.split("#")[0].rstrip()))
                    if "#" in ln else 0 for ln in lines) / max(len(code), 1)
    sq = code.count("'") / max(len(code), 1)
    dq = code.count('"') / max(len(code), 1)
    tq = 1.0 if ('"""' in code or "'''" in code) else 0.0
    fstr = len(re.findall(r"[fFrRbBuU]{1,2}['\"]", code)) / max(len(code), 1) * 100
    esc = code.count("\\") / max(len(code), 1)
    sp_eq = len(re.findall(r"[^ =!<>]= |== |!= ", code)) / max(len(code), 1) * 100
    sp_cmp = len(re.findall(r"[=!<>]= ?", code)) / max(len(code), 1) * 100
    sp_comma = len(re.findall(r",\S", code)) / max(len(code), 1) * 100
    semi = sum(1 for ln in lines if ln.rstrip().endswith(";")) / n
    paren_if = len(re.findall(r"\b(if|while|for)\s*\(", code)) / max(len(code), 1) * 100
    idents = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", code)
    nid = max(len(idents), 1)
    camel = sum(1 for x in idents if re.search(r"[a-z][A-Z]", x)) / nid
    snake = sum(1 for x in idents if "_" in x) / nid
    caps = sum(1 for x in idents if x.isupper() and len(x) > 1) / nid
    lens = [len(ln) for ln in lines]
    over79 = sum(1 for x in lens if x > 79) / n
    over120 = sum(1 for x in lens if x > 120) / n
    final_nl = 1.0 if code.endswith("\n") else 0.0
    nonascii = sum(1 for ch in code if ord(ch) > 127) / max(len(code), 1) * 100
    before_def = 0
    for i, ln in enumerate(lines):
        if re.match(r"\s*(def|class|function)\b", ln) and i > 0 and not lines[i - 1].strip():
            before_def += 1
    before_def = before_def / max(1, sum(
        1 for ln in lines if re.match(r"\s*(def|class|function)\b", ln)))
    tabs_mid = sum(1 for ln in lines if "\t" in ln.strip()) / n
    repeat_ws = len(re.findall(r" {2,}\S* {2,}|  +", code)) / max(len(code), 1) * 100
    return [float(np.mean(ind)) if ind else 0.0, ind_ent, ind_cnt.get(1, 0) / n,
            ind_cnt.get(2, 0) / n, ind_cnt.get(4, 0) / n, ind_cnt.get(8, 0) / n,
            pure_com / n, inline_com / n, block, com_chars, sq, dq, tq, fstr, esc,
            sp_eq, sp_cmp, sp_comma, semi, paren_if, camel, snake, caps,
            over79, over120, final_nl, nonascii, before_def, tabs_mid, repeat_ws]


def main() -> int:
    tok = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    picks = replay_picks()
    for task in ("a", "b", "c"):
        for split in ("val", "test"):
            try:
                path = RAW / FILES[task][("train", "val", "test").index(split)]
                npq = pq.ParquetFile(OUT / f"{task}_{split}.parquet").metadata.num_rows
                idx = test_kept(path) if split == "test" else picks[(task, split)]
                codes = pq.read_table(path, columns=["code"]).column("code").to_pylist()
                X = []
                for i in idx:
                    c = codes[i]
                    if len(tok(c, add_special_tokens=False)["input_ids"]) < 8:
                        continue
                    X.append(stats2(c))
                X = np.asarray(X, dtype="float32")
                assert X.shape[0] == npq, (task, split, X.shape[0], npq)
                np.savez_compressed(OUT / f"{task}_{split}_stats2.npz", X=X,
                                    names=np.array(NAMES2, dtype=object))
                print(f"[stats2] {task}_{split}: n={X.shape[0]} d={X.shape[1]} ✅", flush=True)
            except Exception as e:
                print(f"[stats2] {task}_{split}: FAILED {type(e).__name__}: {e}", flush=True)
    print("[stats2] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
