# -*- coding: utf-8 -*-
"""d-det-v2 v0.1 · 数据层

职责：加载 STACAD-v1（只读复用 ../../../d-det/data/）、规范化去重、task 划分
（calib / 主训练 / 5 折）、内容代理 Q 与 AMMC 变换、分词与分块。

约定：
- 独立新写，不 import d-det 旧代码；不修改原数据/原代码；
- 记录顺序全局固定（split, task_id, human 先、family 固定序），所有缓存按此对齐；
- split = task 级（原文件切分），无跨 split task 泄漏（已由数据包 audit 确认）。
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import io
import json
import random
import tokenize
import warnings
from pathlib import Path

import torch
import yaml

SRC = Path(__file__).resolve().parent

FAMILIES = ["claude", "deepseek", "gemini", "grok", "mistral", "openai", "qwen"]
CLASSES_F = ["Human"] + FAMILIES              # 8 类（Human=0）
LANG_ORDER = ["c", "cpp", "cs", "go", "java", "php", "py"]
SPLIT_ORDER = {"train": 0, "dev": 1, "test": 2}
KIND_ORDER = {"human": 0, "ai": 1}
BUILTINS = set(dir(builtins))

TRANSFORM_NAMES = ["comment", "ws", "rename"]  # AMMC/压力视图


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #

def load_cfg(path=None) -> dict:
    with open(path or (SRC / "configs.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (SRC / p).resolve()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def log(msg: str, fh=None):
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


def _safe_parse(code: str):
    """ast.parse 包装：静默学生代码自身非法转义序列引发的 SyntaxWarning。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(code)


def _py_ok(code: str) -> bool:
    try:
        _safe_parse(code)
        return True
    except SyntaxError:
        return False


# --------------------------------------------------------------------------- #
# 变换：安全空白 / 注释删除 / 作用域内改名
# --------------------------------------------------------------------------- #

def safe_whitespace(code: str) -> str:
    """安全空白：CRLF→LF、行尾空白、连续空行折叠为 1、单结尾换行。保留缩进语义。"""
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    out, blank = [], 0
    for ln in code.split("\n"):
        ln = ln.rstrip(" \t")
        if ln == "":
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(ln)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def _remove_spans(code: str, spans) -> str:
    if not spans:
        return code
    out, cur = [], 0
    for s, e in sorted(spans):
        if s < cur:
            continue
        out.append(code[cur:s])
        cur = e
    out.append(code[cur:])
    return "".join(out)


def strip_comments_py(code: str):
    """Python：tokenize 定位 COMMENT 跨度后删除；失败返回 None。"""
    try:
        src = code if code.endswith("\n") else code + "\n"
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    offs, acc = [], 0
    for ln in src.splitlines(keepends=True):
        offs.append(acc)
        acc += len(ln)
    offs.append(acc)
    spans = []
    for t in toks:
        if t.type == tokenize.COMMENT:
            try:
                s = offs[t.start[0] - 1] + t.start[1]
                e = offs[t.end[0] - 1] + t.end[1]
            except IndexError:
                continue
            spans.append((s, e))
    return _remove_spans(src, spans)


def strip_comments_clike(code: str, hash_comment=False, backtick=False, verbatim=False) -> str:
    """C 系（c/cpp/cs/go/java/php）：字符串感知的 //、/* */（php 另含 #）删除。

    处理双/单引号、反斜杠转义、go 反引号字符串、C# @"" 原义字符串。
    C++ R"(...)" 与 Java 文本块（罕见）不特殊处理（首批接受，记录说明）。
    """
    n, i, state = len(code), 0, "code"
    out = []
    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if state == "code":
            if verbatim and ch == "@" and nxt == '"':
                out.append('@"'); i += 2; state = "vdq"; continue
            if ch == '"':
                state = "dq"
            elif ch == "'":
                state = "sq"
            elif backtick and ch == "`":
                state = "bt"
            elif ch == "/" and nxt == "/":
                state = "line"; i += 2; continue
            elif ch == "/" and nxt == "*":
                state = "block"; i += 2; continue
            elif hash_comment and ch == "#":
                state = "line"; i += 1; continue
            out.append(ch)
        elif state == "vdq":
            if ch == '"' and nxt == '"':
                out.append('""'); i += 2; continue
            if ch == '"':
                state = "code"
            out.append(ch)
        elif state in ("dq", "sq"):
            out.append(ch)
            if ch == "\\":
                if i + 1 < n:
                    out.append(code[i + 1])
                i += 2
                continue
            if (state == "dq" and ch == '"') or (state == "sq" and ch == "'"):
                state = "code"
        elif state == "bt":
            out.append(ch)
            if ch == "`":
                state = "code"
        elif state == "line":
            if ch == "\n":
                out.append(ch); state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                state = "code"; i += 2; continue
        i += 1
    return "".join(out)


def strip_comments(code: str, lang: str):
    """按语言删除注释；不支持/失败返回 None（调用方回退并计数）。"""
    if lang == "py":
        return strip_comments_py(code)
    if lang == "php":
        return strip_comments_clike(code, hash_comment=True)
    if lang == "go":
        return strip_comments_clike(code, backtick=True)
    if lang == "cs":
        return strip_comments_clike(code, verbatim=True)
    if lang in ("c", "cpp", "java"):
        return strip_comments_clike(code)
    return None


_REFLECT = {"globals", "locals", "eval", "exec", "vars", "getattr", "setattr", "dir", "compile"}


def rename_private_py(code: str):
    """Python：作用域内私有标识符重命名（尽力而为）。

    规则：仅函数自身作用域内的 Store 名；排除参数、`_` 开头、builtins、
    文件级曾作属性名/关键词参数名/global/nonlocal 的名字、嵌套作用域同名（防阴影错位）、
    出现在 f-string 表达式内的名字（保守）；含反射调用的函数整体跳过。
    替换后 ast.parse 验证，失败回退 None。
    """
    try:
        tree = _safe_parse(code)
    except SyntaxError:
        return None

    banned = set()
    fstring_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            banned.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            banned.add(node.arg)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            banned.update(node.names)
        elif isinstance(node, ast.FormattedValue):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    fstring_names.add(sub.id)

    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        return None
    funcs.sort(key=lambda f: (f.end_lineno - f.lineno))  # 内层先处理

    def own_bindings(fn):
        """fn 自身作用域的 Store 名（不下钻嵌套函数/类/lambda）。"""
        store, forbidden, reflect = set(), set(), [False]

        def rec(node):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    if child is not fn:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            store.add(child.name)
                        continue
                if isinstance(child, (ast.Global, ast.Nonlocal)):
                    forbidden.update(child.names); continue
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in _REFLECT:
                    reflect[0] = True
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    store.add(child.id)
                rec(child)

        rec(fn)
        args = set()
        a = fn.args
        for grp in (a.posonlyargs, a.args, a.kwonlyargs):
            args.update(x.arg for x in grp)
        if a.vararg:
            args.add(a.vararg.arg)
        if a.kwarg:
            args.add(a.kwarg.arg)
        # 嵌套作用域存储名（保守：整名跳过）
        nested_store = set()

        def rec_nested(node):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    if child is not fn:
                        for sub in ast.walk(child):
                            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                                nested_store.add(sub.id)
                rec_nested(child)

        rec_nested(fn)
        return store, args, forbidden, reflect[0], nested_store

    # 文件内所有标识符（避免新名冲突）
    file_names = set()
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type == tokenize.NAME:
                file_names.add(t.string)
    except (tokenize.TokenError, IndentationError):
        return None

    # 按行范围收集各函数的重命名表（新名全局唯一）
    used_new = set()
    plans = []  # (lineno, end_lineno, map)
    for fn in funcs:
        store, args, forbidden, reflect, nested_store = own_bindings(fn)
        if reflect:
            continue
        m = {}
        for nm in sorted(store):
            if nm in args or nm in forbidden or nm in banned or nm in fstring_names:
                continue
            if nm in BUILTINS or nm.startswith("_") or nm in nested_store:
                continue
            if len(nm) < 1 or not nm.isidentifier():
                continue
            k = 1
            while f"v{k}" in file_names or f"v{k}" in used_new:
                k += 1
            m[nm] = f"v{k}"
            used_new.add(f"v{k}")
        if m:
            plans.append((fn.lineno, fn.end_lineno, m))
    if not plans:
        return None

    # token 级替换（内层先替换并“占用”其行区间，外层跳过）
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError):
        return None
    offs, acc = [], 0
    for ln in code.splitlines(keepends=True):
        offs.append(acc)
        acc += len(ln)
    offs.append(acc)

    claimed = []  # (s, e) 字符区间
    edits = []    # (s, e, new)
    name_toks = [(t, offs[t.start[0] - 1] + t.start[1], offs[t.end[0] - 1] + t.end[1])
                 for t in toks if t.type == tokenize.NAME]
    for lineno, end_lineno, m in plans:
        for t, s, e in name_toks:
            if not (lineno <= t.start[0] <= end_lineno):
                continue
            if any(cs <= s < ce for cs, ce in claimed):
                continue
            if t.string in m:
                edits.append((s, e, m[t.string]))
        cs = offs[lineno - 1]
        ce = offs[end_lineno]
        claimed.append((cs, ce))
    if not edits:
        return None
    edits.sort(reverse=True)
    for s, e, new in edits:
        code = code[:s] + new + code[e:]
    if not _py_ok(code):
        return None
    return code


# --------------------------------------------------------------------------- #
# 视图：内容代理 Q 与 AMMC 训练视图
# --------------------------------------------------------------------------- #

def apply_transform(code: str, lang: str, name: str):
    """单一变换。返回新代码；不支持/无效返回 None（由调用方回退计数）。"""
    if name == "comment":
        r = strip_comments(code, lang)
        if r is not None and r.strip() == code.strip():
            r = None
        if r is not None and lang == "py" and not _py_ok(r):
            r = None
        return r
    if name == "ws":
        r = safe_whitespace(code)
        return r if r.strip() != code.strip() or r != code else None
    if name == "rename":
        return rename_private_py(code) if lang == "py" else None
    raise ValueError(name)


def content_proxy(code: str, lang: str):
    """Q(x)：注释删除 + 安全空白 + 作用域内私有改名（按语言支持，尽力而为）。

    返回 (Q(x), meta)，meta 记录各步是否实际生效。
    """
    meta = {"comment": False, "ws": False, "rename": False}
    new = code
    r = strip_comments(new, lang)
    if r is not None and (lang != "py" or _py_ok(r)):
        meta["comment"] = r.strip() != new.strip()
        new = r
    w = safe_whitespace(new)
    meta["ws"] = w != new
    new = w
    if lang == "py":
        r = rename_private_py(new)
        if r is not None and r != new:
            new = r
            meta["rename"] = True
    return new, meta


def ammc_view(code: str, lang: str, decisions: dict):
    """AMMC 训练视图：aug 时按独立触发位应用 comment/ws/rename；全未生效回退原文。

    decisions 形如 {"aug": bool, "comment": bool, "ws": bool, "rename": bool}。
    返回 (视图代码, meta)；meta: fired 触发、applied 生效、fallback 回退。
    """
    meta = {"aug": bool(decisions.get("aug", False)), "fired": [], "applied": [], "fallback": False}
    if not meta["aug"]:
        return code, meta
    new = code
    for name in TRANSFORM_NAMES:
        if not decisions.get(name):
            continue
        meta["fired"].append(name)
        r = apply_transform(new, lang, name)
        if r is not None:
            new = r
            meta["applied"].append(name)
    if not meta["applied"]:
        meta["fallback"] = True
        return code, meta
    return new, meta


# --------------------------------------------------------------------------- #
# 数据加载 / 清洗 / 划分
# --------------------------------------------------------------------------- #

def load_records(cfg, limit_tasks=None):
    """加载 STACAD-v1 → 统一记录表（含 Human anchor），规范化去重。

    返回 (records, stats)。records 字段：
      rid/tid/split/lang/file/fam/gen/kind/code/sha/nsha
    """
    droot = resolve(cfg["paths"]["stacad"])
    tasks = [json.loads(l) for l in open(droot / "task_index.jsonl", encoding="utf-8")]
    core = [json.loads(l) for l in open(droot / "core.jsonl", encoding="utf-8")]
    tasks.sort(key=lambda t: t["task_id"])
    core.sort(key=lambda r: r["record_id"])

    limit_tasks = limit_tasks if limit_tasks is not None else cfg["data"].get("limit_tasks")
    if limit_tasks:
        cnt, kept = {}, []
        for t in tasks:
            c = cnt.get(t["task_split"], 0)
            if c < limit_tasks:
                kept.append(t)
                cnt[t["task_split"]] = c + 1
        tasks = kept
        keep_ids = {t["task_id"] for t in tasks}
        core = [r for r in core if r["task_id"] in keep_ids]

    recs = []
    for t in tasks:
        recs.append(dict(rid="human:" + t["task_id"], tid=t["task_id"], split=t["task_split"],
                         lang=t["language"], file=t["file_name"], fam="Human", gen="Human",
                         kind="human", code=t["human_code"], sha=t["human_sha256"]))
    for r in core:
        recs.append(dict(rid=r["record_id"], tid=r["task_id"], split=r["task_split"],
                         lang=r["language"], file=r["file_name"], fam=r["family"],
                         gen=r["generator"], kind="ai", code=r["code"], sha=r["code_sha256"]))

    # 固定顺序：split → task_id → human 先 → family 固定序
    recs.sort(key=lambda r: (SPLIT_ORDER[r["split"]], r["tid"], KIND_ORDER[r["kind"]], FAMILIES.index(r["fam"]) if r["fam"] in FAMILIES else -1))

    # 规范化去重（保留策略：先序保留；记录全部裁定）
    stats = {"raw_rows": len(recs), "norm_sha_collisions": 0, "dropped": [], "dup_exact_sha": 0}
    by_exact = {}
    by_norm = {}
    out = []
    for r in recs:
        r["nsha"] = sha256_text(safe_whitespace(r["code"]))
        if r["sha"] in by_exact:
            stats["dup_exact_sha"] += 1
        by_exact.setdefault(r["sha"], r["rid"])
        prev = by_norm.get(r["nsha"])
        if prev is not None:
            stats["norm_sha_collisions"] += 1
            stats["dropped"].append({"keep": prev, "drop": r["rid"]})
            continue
        by_norm[r["nsha"]] = r["rid"]
        out.append(r)
    recs = out
    stats["kept_rows"] = len(recs)
    stats["by_split"] = {s: sum(1 for r in recs if r["split"] == s) for s in ("train", "dev", "test")}
    stats["by_family"] = {f: sum(1 for r in recs if r["fam"] == f) for f in CLASSES_F}
    return recs, stats


def save_records(recs, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_records_file(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def assign_meta(recs, cfg):
    """划分训练内校准子集（按 task，语言分层）与 5 折（仅 train 主训练 task）。"""
    frac = cfg["data"]["calib_frac"]
    cseed = cfg["data"]["calib_seed"]
    fseed = cfg["data"]["fold_seed"]
    K = cfg["data"]["folds"]

    by_lang = {}
    for r in recs:
        if r["kind"] == "human" and r["split"] == "train":
            by_lang.setdefault(r["lang"], set()).add(r["tid"])
    calib, fold = set(), {}
    for lang in sorted(by_lang):
        ts = sorted(by_lang[lang])
        rng = random.Random(cseed + sum(map(ord, lang)))
        rng.shuffle(ts)
        n = max(1, round(frac * len(ts)))
        calib.update(ts[:n])
        main = ts[n:]
        rng2 = random.Random(fseed + sum(map(ord, lang)))
        rng2.shuffle(main)
        for i, t in enumerate(main):
            fold[t] = i % K
    for r in recs:
        r["calib"] = bool(r["tid"] in calib)
        r["fold"] = int(fold.get(r["tid"], -1))
    lang_of_task = {r["tid"]: r["lang"] for r in recs if r["kind"] == "human"}
    stats = {
        "calib_tasks": len(calib),
        "main_tasks": len(fold),
        "fold_sizes": {k: sum(1 for v in fold.values() if v == k) for k in range(K)},
        "calib_by_lang": {lang: sum(1 for t in calib if lang_of_task.get(t) == lang) for lang in sorted(by_lang)},
    }
    return stats


def split_index(recs, split):
    """记录在全局顺序中的下标（保持 records 顺序）。"""
    return [i for i, r in enumerate(recs) if r["split"] == split]


def subset_index(recs, split, calib=None, main_only=False):
    """按 split + calib 标志取下标；main_only=True 取 train 非 calib（主训练池）。"""
    out = []
    for i, r in enumerate(recs):
        if r["split"] != split:
            continue
        if calib is True and not r["calib"]:
            continue
        if calib is False and r["calib"]:
            continue
        if main_only and r["calib"]:
            continue
        out.append(i)
    return out


# --------------------------------------------------------------------------- #
# 分词与分块
# --------------------------------------------------------------------------- #

def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    path = resolve(cfg["paths"]["encoder"])
    tok = AutoTokenizer.from_pretrained(path)
    # 分块由 encode_chunks 负责：提高 model_max_length 以关闭长度告警（不截断）
    tok.model_max_length = 10 ** 9
    return tok


def encode_chunks(tok, code: str, chunk_len=512, max_chunks=4):
    """编码为 ≤max_chunks 个 512 块；超限取首 2 + 尾 2（不重复不足块内容）。"""
    ids = tok(code, add_special_tokens=True)["input_ids"]
    chunks = [ids[i:i + chunk_len] for i in range(0, len(ids), chunk_len)] or [[]]
    if len(chunks) > max_chunks:
        keep = max_chunks // 2
        chunks = chunks[:keep] + chunks[-(max_chunks - keep):]
    return [torch.tensor(c, dtype=torch.long) for c in chunks]


def compute_token_lens(tok, recs):
    """原始代码的 tokenizer token 数（长度分桶用；与分块覆盖无关）。"""
    return torch.tensor([len(tok(r["code"], add_special_tokens=True)["input_ids"]) for r in recs],
                        dtype=torch.long)


def bucket_of(n_tokens: int, edges):
    """长度分桶：≤e0 / e0+1..e1 / ... / >last。edges=[512,1024,2048]。"""
    lo = 0
    for k, e in enumerate(edges):
        if n_tokens <= e:
            return k
        lo = e
    return len(edges)


BUCKET_NAMES = ["≤512", "513-1024", "1025-2048", ">2048"]
