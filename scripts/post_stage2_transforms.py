"""post-stage2 Phase A：安全干预变换库（A1 格式规范化 / A2 注释屏蔽 / A3 字面量屏蔽）。

依据《d-det_AutoDL_后续干预与主干探针指导_2026-10-08》§3-§5、§7：
- A1：CRLF→LF、删除行尾空白、统一文件末尾换行（不改变 token 语义）；
- A2：状态机屏蔽注释（区分字符串/字符常量/模板/注释；Python 与 C-family 双模式）；
- A3：字面量屏蔽（字符串→"<str>"、数字→0），仅作敏感性分析；
- A4：默认 not_executed（指导 §3：多语言覆盖率不足时不得强行执行）。

只保存规范化后的 hash 与统计；代码正文不写入 Git（本地缓存于 local/）。

输出（d-det/artifacts/post_stage2_intervention_2026-10-08/）：
- transforms_rules.json：规则与版本；
- transforms_audit.json：全量统计（失败率、token/长度变化、hash 聚合）；
- local/transformed_texts/{tid}.jsonl.gz：变换后文本（本地，不进 git）；
- local/hash_maps/{tid}.jsonl.gz：每样本原始/变换 hash 与统计（本地）。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import variant_transfer_stage1_execute as s1  # noqa: E402

OUT = ROOT / "d-det/artifacts/post_stage2_intervention_2026-10-08"
MODEL_SMALL = s1.MODEL_SMALL
LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8", "surrogatepass")).hexdigest()


# ================================================================ 语言模式
_C_PRE_RE = re.compile(r"^\s*#\s*(include\s*[<\"]|define\b|pragma\b)", re.M)
_PY_FEAT_RE = re.compile(r"^\s*(def|class|import|from)\s|^\s*@", re.M)


def lang_mode(code: str) -> str:
    """粗判 Python / C-family。BCC 语料全部为 Python；C 模式仅为防御性路径。

    判据收紧：行首 #include<...>/#define/#pragma 且无 Python 特征行（def/class/import/@），
    或有 using namespace/std:: 且无 Python 特征。注释内 `# if ...` 不得误判为预处理。
    """
    head = code[:4000]
    py_feat = _PY_FEAT_RE.search(head)
    if not py_feat:
        if _C_PRE_RE.search(head):
            return "c"
        if re.search(r"\bstd::|^\s*using\s+namespace\b", head, re.M):
            return "c"
    return "py"


# ================================================================ 扫描器
# spans: (kind, start, end)；kind ∈ {comment_line, comment_block, string, number}
_PY_QUOTES = ('"""', "'''", '"', "'")
_C_QUOTES = ('"', "'")
_NUM_RE = re.compile(
    r"(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+"
    r"|(?:\d[\d_]*\.?[\d_]*(?:[eE][+-]?\d+)?|\.\d[\d_]*(?:[eE][+-]?\d+)?)(?:[jJ])?)"
)
_IDENT_TAIL = re.compile(r"[A-Za-z0-9_]")


def _try_string_start(code: str, i: int, mode: str):
    """若 code[i:] 处开始一个字符串/字符常量（含 Python 前缀），返回 (qlen, is_triple, prefix_len)；否则 None。"""
    n = len(code)
    j = i
    plen = 0
    if mode == "py":
        k = j
        while k < n and code[k] in "fFrRbBuU" and (k - j) < 3:
            k += 1
        if k > j and k < n and code[k] in "\"'":
            prev = code[j - 1] if j > 0 else ""
            if prev and (prev.isalnum() or prev == "_"):
                return None  # 前缀前是标识符字符 → 不是字符串前缀
            plen = k - j
            j = k
    if j >= n:
        return None
    c = code[j]
    if c not in "\"'":
        return None
    if mode == "py":
        for q in _PY_QUOTES:
            if code.startswith(q, j) and len(q) >= 1:
                if len(q) == 3:
                    return (3, True, plen)
                return (1, False, plen)
        return None
    # C：单引号=字符常量、双引号=字符串；' 单引号也当字符串处理（JS 兼容）
    if code.startswith("\\", j):
        return None
    return (1, False, plen)


def scan_spans(code: str, mode: str):
    """顺序扫描，返回 (spans, warnings)。spans 非重叠、升序。"""
    spans: list[tuple[str, int, int]] = []
    warnings: list[str] = []
    n = len(code)
    i = 0
    while i < n:
        c = code[i]
        # --- 注释
        if mode == "py" and c == "#":
            e = code.find("\n", i)
            e = n if e < 0 else e
            spans.append(("comment_line", i, e))
            i = e
            continue
        if mode == "c":
            if code.startswith("//", i):
                e = code.find("\n", i)
                e = n if e < 0 else e
                spans.append(("comment_line", i, e))
                i = e
                continue
            if code.startswith("/*", i):
                e = code.find("*/", i + 2)
                if e < 0:
                    warnings.append("unclosed_block_comment")
                    e = n
                    spans.append(("comment_block", i, e))
                    i = e
                else:
                    spans.append(("comment_block", i, e + 2))
                    i = e + 2
                continue
            if c == "\\":
                # C 续行（罕见）：跳过两字符避免误判
                i += 2
                continue
        if mode == "py" and c == "\\":
            i += 2  # 行续行；保持对齐
            continue
        # --- 字符串
        ts = _try_string_start(code, i, mode)
        if ts is not None:
            qlen, triple, plen = ts
            qstart = i + plen
            quote = code[qstart:qstart + qlen]
            raw = plen >= 1 and code[i] in "rR"
            j = qstart + qlen
            closed = False
            while j < n:
                if not raw and code[j] == "\\":
                    j += 2
                    continue
                if code.startswith(quote, j):
                    closed = True
                    j += qlen
                    break
                if not triple and code[j] == "\n":
                    break
                j += 1
            if not closed:
                warnings.append(f"unclosed_string_line")
                if triple:
                    # 三引号未闭：到 EOF 处理并告警（仍记 span，样本将标记失败）
                    spans.append(("string", i, n))
                    warnings.append("unclosed_triple_string")
                    i = n
                    continue
                spans.append(("string", i, min(j, n)))
                i = min(j, n)
                continue
            spans.append(("string", i, j))
            i = j
            continue
        # --- 数字（仅 normal 态）
        if c.isdigit() or (c == "." and i + 1 < n and code[i + 1].isdigit()):
            prev_ok = i == 0 or not (code[i - 1].isalnum() or code[i - 1] in "_")
            if prev_ok:
                m = _NUM_RE.match(code, i)
                if m:
                    e = m.end()
                    if e < n and (_IDENT_TAIL.match(code[e]) or code[e] == "."):
                        pass  # 数字后紧跟标识符/点：留给正常扫描
                    else:
                        spans.append(("number", i, e))
                        i = e
                        continue
        i += 1
    return spans, warnings


# ================================================================ A1 格式规范化
def transform_a1(code: str) -> tuple[str, dict, list[str]]:
    warns: list[str] = []
    cnt: dict = {"crlf": code.count("\r\n"), "cr": code.count("\r") - code.count("\r\n")}
    t = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")
    stripped = [l.rstrip(" \t") for l in lines]
    t = "\n".join(stripped)
    if not t.endswith("\n"):
        t += "\n"
        cnt["end_newline_added"] = 1
    else:
        t2 = re.sub(r"\n+$", "\n", t)
        if t2 != t:
            cnt["end_newlines_collapsed"] = 1
            t = t2
    cnt["trailing_ws_lines"] = sum(1 for l, s in zip(lines, stripped) if l != s)
    return t, cnt, warns


# ================================================================ A2 注释屏蔽
def transform_a2(code: str) -> tuple[str, dict, list[str]]:
    mode = lang_mode(code)
    spans, warns = scan_spans(code, mode)
    out = []
    pos = 0
    removed_chars = 0
    n_line = n_block = 0
    for kind, s, e in spans:
        if kind == "comment_line":
            out.append(code[pos:s])
            removed_chars += e - s
            n_line += 1
            pos = e
        elif kind == "comment_block":
            out.append(code[pos:s])
            blk = code[s:e]
            out.append("\n" * blk.count("\n"))
            removed_chars += (e - s) - blk.count("\n")
            n_block += 1
            pos = e
    out.append(code[pos:])
    res = "".join(out)
    cnt = {"mode": mode, "comment_line": n_line, "comment_block": n_block,
           "removed_chars": removed_chars}
    return res, cnt, warns


# ================================================================ A3 字面量屏蔽
def transform_a3(code: str) -> tuple[str, dict, list[str]]:
    mode = lang_mode(code)
    spans, warns = scan_spans(code, mode)
    out = []
    pos = 0
    n_str = n_num = 0
    for kind, s, e in spans:
        if kind == "string":
            out.append(code[pos:s])
            out.append('"<str>"')
            n_str += 1
            pos = e
        elif kind == "number":
            out.append(code[pos:s])
            out.append("0")
            n_num += 1
            pos = e
    out.append(code[pos:])
    res = "".join(out)
    cnt = {"mode": mode, "strings": n_str, "numbers": n_num}
    return res, cnt, warns


TRANSFORMS = {
    "A1_format_norm": {
        "fn": transform_a1,
        "rules": "CRLF/LF 统一；删除行尾空白（空格/制表符）；文件末尾统一为恰好一个换行。不重排缩进/括号/语句。",
        "semantic_risk": "none_expected",
    },
    "A2_comments_masked": {
        "fn": transform_a2,
        "rules": ("状态机屏蔽注释：Python 模式 # 行注释；C-family 模式 // 行注释与 /* */ 块注释；"
                  "字符串/字符常量/三引号字符串内部不误判；行注释删除至行尾（不含换行），"
                  "块注释删除但保留其中换行。失败样本保留原文并单独记录。"),
        "semantic_risk": "low（注释理论上不影响语义；docstring 按字符串处理不屏蔽）",
    },
    "A3_literals_masked": {
        "fn": transform_a3,
        "rules": ("字符串/字符常量整体替换为 \"<str>\"（含前缀）；数字替换为 0。"
                  "仅敏感性分析；可能改变任务语义与可运行性，不得命名为\"去风格后的纯代码\"。"),
        "semantic_risk": "high_sensitivity_only",
    },
}

A4_STATUS = {
    "transform_id": "A4_identifiers_masked",
    "status": "not_executed",
    "reason": ("指导 §3/§7：仅在语言 tokenizer 能稳定区分局部变量/函数名/API 名/关键字时才允许执行；"
               "无法排除误伤库 API 与类型名；多语言覆盖率不足。按指导状态写为 not_executed。"),
}


# ================================================================ 编译校验（Python）
def py_compiles(code: str) -> bool:
    import warnings as _w
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            compile(code, "<x>", "exec")
        return True
    except SyntaxError:
        return False
    except ValueError:
        return True  # 含 null 字节等罕见情况，跳过判定
    except Exception:
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="调试：只处理前 N 样本")
    args = ap.parse_args()
    if not args.run:
        print("use --run")
        return
    t0 = time.time()
    (OUT / "local/transformed_texts").mkdir(parents=True, exist_ok=True)
    (OUT / "local/hash_maps").mkdir(parents=True, exist_ok=True)

    log("[1] load samples (train/dev only)")
    split_of = s1.load_split()
    samples = s1.load_samples(split_of)
    if args.limit:
        samples = samples[:args.limit]
    log(f"  n={len(samples)}")

    log("[2] CodeT5 tokenizer for token stats")
    from transformers import RobertaTokenizer
    tok = RobertaTokenizer(vocab=str(MODEL_SMALL / "vocab.json"), merges=str(MODEL_SMALL / "merges.txt"),
                           unk_token="<unk>", bos_token="<s>", eos_token="</s>", sep_token="</s>",
                           cls_token="<s>", pad_token="<pad>", mask_token="<mask>", add_prefix_space=False)

    def ntok(t: str) -> int:
        return len(tok.encode(t, add_special_tokens=False))

    audit = {"schema": "post_stage2_transforms_audit_v1",
             "generated_utc": datetime.now(timezone.utc).isoformat(),
             "n_samples": len(samples), "order": "(unit, task) ascending (stage-1 identical)",
             "transforms": {}, "a4_status": A4_STATUS}
    rules_out = {tid: {"rules": spec["rules"], "semantic_risk": spec["semantic_risk"]}
                 for tid, spec in TRANSFORMS.items()}
    rules_out["A1_format_norm"]["version"] = "v1"
    rules_out["A2_comments_masked"]["version"] = "v1"
    rules_out["A3_literals_masked"]["version"] = "v1"

    for tid, spec in TRANSFORMS.items():
        log(f"[3] apply {tid}")
        texts_out = []
        map_rows = []
        agg = {"n_fail": 0, "fail_reasons": {}, "warnings": {}, "mode_counts": {},
               "lines_orig": 0, "lines_new": 0, "chars_orig": 0, "chars_new": 0,
               "tok_orig": 0, "tok_new": 0, "tok_trunc_orig": 0, "tok_trunc_new": 0,
               "samples_line_change": 0, "samples_compile_fail_after": 0,
               "counts": {}}
        joint = hashlib.sha256()
        for idx, r in enumerate(samples):
            code = r["code"]
            fail_reason = None
            try:
                new, cnt, warns = spec["fn"](code)
            except Exception as exc:  # 变换器自身异常 → 样本失败
                new, cnt, warns = code, {}, [f"transform_exception:{type(exc).__name__}"]
                fail_reason = "transform_exception"
            unclosed = [w for w in warns if w.startswith("unclosed")]
            if fail_reason is None and unclosed:
                fail_reason = unclosed[0]
            if fail_reason is None and py_compiles(code) and not py_compiles(new):
                fail_reason = "compile_broken"
                agg["samples_compile_fail_after"] += 1
            fail = fail_reason is not None
            if fail:
                agg["n_fail"] += 1
                agg["fail_reasons"][fail_reason] = agg["fail_reasons"].get(fail_reason, 0) + 1
                new, cnt = code, {"fail": True}  # 失败样本保留原文
            for w in warns:
                if not w.startswith("unclosed") and not w.startswith("transform_exception"):
                    agg["warnings"][w] = agg["warnings"].get(w, 0) + 1
            # ---- 统计（基于最终 new）----
            ho, hn = sha256_text(code), sha256_text(new)
            lo, ln = len(code.split("\n")), len(new.split("\n"))
            to, tn = ntok(code), ntok(new)
            agg["lines_orig"] += lo; agg["lines_new"] += ln
            agg["chars_orig"] += len(code); agg["chars_new"] += len(new)
            agg["tok_orig"] += to; agg["tok_new"] += tn
            agg["tok_trunc_orig"] += int(to > 512); agg["tok_trunc_new"] += int(tn > 512)
            if lo != ln:
                agg["samples_line_change"] += 1
            mode = lang_mode(code)
            agg["mode_counts"][mode] = agg["mode_counts"].get(mode, 0) + 1
            for k, v in cnt.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    agg["counts"][k] = agg["counts"].get(k, 0) + v
            joint.update(f"{idx}:{ho}:{hn}\n".encode())
            map_rows.append({"i": idx, "unit": r["unit"], "task": r["task"], "split": r["split"],
                             "orig": ho, "new": hn, "nchar_o": len(code), "nchar_n": len(new),
                             "nlines_o": lo, "nlines_n": ln, "tok_o": to, "tok_n": tn,
                             "fail": fail})
            texts_out.append(new)
            if (idx + 1) % 2000 == 0:
                log(f"    {idx+1}/{len(samples)}")
        agg["joint_hash_map_sha256"] = joint.hexdigest()
        agg["rule"] = spec["rules"]
        agg["semantic_risk"] = spec["semantic_risk"]
        audit["transforms"][tid] = agg
        with gzip.open(OUT / "local/transformed_texts" / f"{tid}.jsonl.gz", "wt", encoding="utf-8") as f:
            for idx, t in enumerate(texts_out):
                f.write(json.dumps({"i": idx, "text": t}, ensure_ascii=False) + "\n")
        with gzip.open(OUT / "local/hash_maps" / f"{tid}.jsonl.gz", "wt", encoding="utf-8") as f:
            for row in map_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(f"  {tid}: fail={agg['n_fail']} line_change={agg['samples_line_change']} "
            f"compile_fail_after={agg['samples_compile_fail_after']} "
            f"tok {agg['tok_orig']} -> {agg['tok_new']} ({(agg['tok_new']/max(1,agg['tok_orig'])-1)*100:+.2f}%)")

    (OUT / "transforms_rules.json").write_text(json.dumps(
        {"schema": "post_stage2_transforms_rules_v1",
         "generated_utc": audit["generated_utc"], "transforms": rules_out,
         "a4_status": A4_STATUS}, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "transforms_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    (OUT / "logs" / "transforms_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
