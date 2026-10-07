"""变体迁移 stage-1 执行（train/dev 阶段）——公开 BigCodeBench full|instruct。

依据：
- 《AutoDL 变体迁移执行前闸门指导》2026-10-08（dccd1d2）§4-§5；
- 《AutoDL 服务器重构版最小授权与补传豁免》2026-10-08 §1/§4（允许 train/dev；test 另行授权）；
- 冻结配置 execution_config_frozen.json / r1_negative_set_amendment.json（preflight 目录）。

范围（严格 train/dev，不触碰 test）：
- R1 系列成员读出（二分类）：tfidf_char / tfidf_word / sem_base / sem_small / style_lr /
  style_lgb / metadata_only / size_length_only / P0_fusion / P0_equal；
- 负集两版本：size_mix（原注册）与 size_matched（amendment）；
- R2 表示迁移：CodeT5-small/base 系列中心（欧氏/余弦）+ char TF-IDF 余弦；距离-尺寸秩相关（探索性）；
- 指标：dev AUROC / AP / task-macro AUROC + task-cluster bootstrap 500 CI；
- dev 选择口径（冻结）：C ∈ {0.03,0.1,0.3,1.0} 按 dev AUROC 最大、平局取更小 C；
  SGD 5ep best-dev、3 seeds 概率均值；LGBM(800) 固定超参。

协议防污染：
- 只加载 split ∈ {train, dev} 的行（test 行不进入内存）；
- 所有输出携带 source_status=server_reconstruction_only。

输出：d-det/artifacts/variant_transfer_stage1_2026-10-08/
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "d-det/artifacts/variant_transfer_stage1_2026-10-08"
PRE = ROOT / "d-det/artifacts/variant_transfer_preflight_2026-10-08"
FUP = ROOT / "d-det/artifacts/public_full_followup_2026-10-08"
RECORDS = ROOT / "d-det/data/public_same_task_full_2026-10-07/records.jsonl"
SPLIT_CSV = ROOT / "d-det/artifacts/public_full_receive_2026-10-08/prereg/split_index.csv"
MODEL_SMALL = ROOT / "d-det/models/codet5-small"
MODEL_BASE = ROOT / "d-det/checkpoints/codet5-base"

SEEDS = [0, 1, 2]
BOOT = 500
BOOT_SEED = 20261008
C_GRID = (0.03, 0.1, 0.3, 1.0)

SERIES = {
    "CodeLlama-Instruct": ["codellama--CodeLlama-7b-Instruct-hf", "codellama--CodeLlama-13b-Instruct-hf",
                           "codellama--CodeLlama-34b-Instruct-hf", "codellama--CodeLlama-70b-Instruct-hf"],
    "Qwen2.5-Coder-Instruct": ["Qwen--Qwen2.5-Coder-1.5B-Instruct", "Qwen--Qwen2.5-Coder-7B-Instruct",
                               "Qwen--Qwen2.5-Coder-14B-Instruct", "Qwen--Qwen2.5-Coder-32B-Instruct"],
    "DeepSeek-Coder-v1-Instruct": ["deepseek-ai--deepseek-coder-1.3b-instruct",
                                   "deepseek-ai--deepseek-coder-6.7b-instruct",
                                   "deepseek-ai--deepseek-coder-33b-instruct"],
}
PARAM_B = {"codellama--CodeLlama-7b-Instruct-hf": 7.0, "codellama--CodeLlama-13b-Instruct-hf": 13.0,
           "codellama--CodeLlama-34b-Instruct-hf": 34.0, "codellama--CodeLlama-70b-Instruct-hf": 70.0,
           "Qwen--Qwen2.5-Coder-1.5B-Instruct": 1.5, "Qwen--Qwen2.5-Coder-7B-Instruct": 7.0,
           "Qwen--Qwen2.5-Coder-14B-Instruct": 14.0, "Qwen--Qwen2.5-Coder-32B-Instruct": 32.0,
           "deepseek-ai--deepseek-coder-1.3b-instruct": 1.3, "deepseek-ai--deepseek-coder-6.7b-instruct": 6.7,
           "deepseek-ai--deepseek-coder-33b-instruct": 33.0}
SIZE_LABEL = {
    "codellama--CodeLlama-7b-Instruct-hf": "7b", "codellama--CodeLlama-13b-Instruct-hf": "13b",
    "codellama--CodeLlama-34b-Instruct-hf": "34b", "codellama--CodeLlama-70b-Instruct-hf": "70b",
    "Qwen--Qwen2.5-Coder-1.5B-Instruct": "1.5B", "Qwen--Qwen2.5-Coder-7B-Instruct": "7B",
    "Qwen--Qwen2.5-Coder-14B-Instruct": "14B", "Qwen--Qwen2.5-Coder-32B-Instruct": "32B",
    "deepseek-ai--deepseek-coder-1.3b-instruct": "1.3b", "deepseek-ai--deepseek-coder-6.7b-instruct": "6.7b",
    "deepseek-ai--deepseek-coder-33b-instruct": "33b",
}
SERIES_OF = {m: s for s, ms in SERIES.items() for m in ms}
ARM_NAMES = ("size_mix", "size_matched")
P0_MEMBERS = ("tfidf_char", "tfidf_word", "sem_base", "style_lr", "style_lgb")
LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 特征（与 P0 协议一致）
_KW = ["if", "else", "for", "while", "return", "try", "catch", "throw", "class", "struct",
       "enum", "def", "import", "include", "package", "public", "private", "static",
       "const", "void", "new", "self", "this", "print", "printf", "cout", "input",
       "lambda", "async", "await", "yield", "match"]


def style_features(code: str) -> list[float]:
    """regex stylometry（与 acl_sota_p0_baselines.style_features 同协议）。"""
    lines = code.split("\n")
    n = max(1, len(lines))
    nchar = max(1, len(code))
    lens = [len(l) for l in lines]
    stripped = [l.strip() for l in lines]
    nonblank = [l for l in stripped if l]
    indent = [len(l) - len(l.lstrip(" ")) for l in nonblank]
    idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code)
    idn = max(1, len(idents))
    names = set(idents)
    snake = sum(1 for w in idents if "_" in w and w.lower() == w)
    camel = sum(1 for w in idents if "_" not in w and any(c.isupper() for c in w[1:]))
    single = sum(1 for w in idents if len(w) == 1)
    upper_all = sum(1 for w in names if len(w) > 1 and w.isupper())
    return [
        nchar, n, np.mean(lens), np.max(lens), np.std(lens),
        sum(1 for l in lines if not l.strip()) / n,
        sum(1 for l in nonblank if l.startswith(("//", "#"))) / n,
        sum(1 for l in nonblank if l.startswith("/*") or l.startswith("*")) / n,
        code.count("/*") / n, code.count("*/") / n,
        np.mean(indent) if indent else 0.0, np.std(indent) if indent else 0.0,
        sum(1 for l in lines if "\t" in l) / n,
        sum(1 for l in lines if l != l.rstrip()) / n,
        sum(1 for l in lines if l.strip().endswith((";", "{"))) / n,
        len(idents) / n, np.mean([len(w) for w in idents]) if idents else 0.0,
        len(names) / n, single / idn, snake / idn, camel / idn, upper_all / max(1, len(names)),
        sum(1 for w in idents if w.startswith("_")) / idn,
        sum(1 for w in idents if any(ch.isdigit() for ch in w)) / idn,
        sum(c.isdigit() for c in code) / nchar, sum(c.isupper() for c in code) / nchar,
        code.count("_") / nchar, sum(1 for c in code if ord(c) > 127) / nchar,
        code.count(" ") / nchar, code.count("\t") / n,
        code.count("(") / n, code.count(")") / n, code.count("{") / n, code.count("}") / n,
        code.count("[") / n, code.count(";") / n, code.count(":") / n,
        code.count(",") / n, code.count("=") / n, code.count(".") / n,
        code.count("+") / n, code.count("-") / n, code.count("*") / n, code.count("/") / n,
        code.count("!") / n, code.count("<") / n, code.count(">") / n, code.count("&") / n,
        code.count("|") / n, code.count("%") / n,
        code.count('"') / n, code.count("'") / n, code.count("`") / n,
        code.count("\\n") / n, code.count("%s") / n, code.count("{}") / n, code.count('f"') / n,
        sum(1 for l in nonblank if l.startswith("def ") or l.startswith("func ")) / n,
        sum(1 for l in nonblank if re.match(r"^\s*(if|for|while|switch)\b", l)) / n,
        sum(1 for l in nonblank if re.match(r"^\s*(return|yield)\b", l)) / n,
    ] + [len(re.findall(r"\b" + re.escape(k) + r"\b", code)) / n for k in _KW]


META_IDX = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]  # nchar,nlines,mean/max/std len,blank,indent(mean/std),tab,trailing


def tfidf_char():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5,
                           max_features=300000, sublinear_tf=True)


def tfidf_word():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S",
                           ngram_range=(1, 2), min_df=3, max_features=200000,
                           sublinear_tf=True, lowercase=False)


def encode_codet5(model_dir: Path, texts: list[str], device_batch: int = 8) -> np.ndarray:
    """冻结 mean-pool（512=384+128、无特殊符、fp16；与 P0/N2 同协议）。"""
    import torch
    from transformers import RobertaTokenizer, T5EncoderModel
    torch.manual_seed(20261007)
    tok = RobertaTokenizer(vocab=str(model_dir / "vocab.json"), merges=str(model_dir / "merges.txt"),
                           unk_token="<unk>", bos_token="<s>", eos_token="</s>", sep_token="</s>",
                           cls_token="<s>", pad_token="<pad>", mask_token="<mask>",
                           add_prefix_space=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5EncoderModel.from_pretrained(
        model_dir, local_files_only=True,
        dtype=(torch.float16 if device.type == "cuda" else torch.float32))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    encoded = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) > 512:
            ids = ids[:384] + ids[-128:]
        encoded.append(ids)
    dim = int(model.config.d_model)
    emb = np.empty((len(encoded), dim), dtype=np.float32)
    for start in range(0, len(encoded), device_batch):
        batch = encoded[start:start + device_batch]
        ln = max(1, max(len(x) for x in batch))
        ids = torch.full((len(batch), ln), int(tok.pad_token_id), dtype=torch.long)
        am = torch.zeros((len(batch), ln), dtype=torch.long)
        for j, row in enumerate(batch):
            if row:
                ids[j, :len(row)] = torch.tensor(row, dtype=torch.long)
                am[j, :len(row)] = 1
        ids, am = ids.to(device), am.to(device)
        with torch.inference_mode():
            if device.type == "cuda":
                import contextlib
                ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
            else:
                import contextlib
                ctx = contextlib.nullcontext()
            with ctx:
                hidden = model(input_ids=ids, attention_mask=am, return_dict=True).last_hidden_state
            pooled = (hidden.float() * am.unsqueeze(-1)).sum(1) / am.sum(1, keepdim=True).clamp_min(1).float()
        emb[start:start + len(batch)] = pooled.cpu().numpy()
    del model
    import torch as _t
    if device.type == "cuda":
        _t.cuda.empty_cache()
    return emb


# ---------------------------------------------------------------- 训练组件（二分类）
def lr_select_auroc(Ftr, ytr, Fdv, ydv, grid=C_GRID):
    """StandardScaler + LogReg(balanced)；C 网格按 dev AUROC 最大、平局取更小 C。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr)
    Ztr, Zdv = sc.transform(Ftr), sc.transform(Fdv)
    best = (-1.0, None)
    for C in grid:
        clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(Ztr, ytr)
        auc = roc_auc_score(ydv, clf.predict_proba(Zdv)[:, 1])
        if auc > best[0] + 1e-12:
            best = (float(auc), C, clf)
    _, C, clf = best
    return clf.predict_proba(Ztr)[:, 1], clf.predict_proba(Zdv)[:, 1], C


def sgd_seeds(Xtr, ytr, Xdv, ydv, seeds=SEEDS, epochs=5, bs=10000, alpha=2e-6):
    """SGD log_loss 5ep best-dev（dev AUROC）× seeds；返回 seed 平均 train/dev 概率与每 seed AUC。"""
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import roc_auc_score
    classes = np.array([0, 1])
    counts = np.bincount(ytr, minlength=2).astype(float)
    wclass = len(ytr) / (2 * np.maximum(counts, 1))
    tr_list, dv_list, aucs = [], [], []
    for seed in seeds:
        clf = SGDClassifier(loss="log_loss", alpha=alpha, random_state=seed)
        rng = np.random.default_rng(seed)
        best = (-1.0, None, None)
        for ep in range(epochs):
            perm = rng.permutation(len(ytr))
            for i in range(0, len(perm), bs):
                sel = perm[i:i + bs]
                clf.partial_fit(Xtr[sel], ytr[sel], classes=classes, sample_weight=wclass[ytr[sel]])
            auc = roc_auc_score(ydv, clf.predict_proba(Xdv)[:, 1])
            if auc > best[0]:
                best = (auc, clf.coef_.copy(), clf.intercept_.copy())
        _, coef, inter = best
        clf.coef_, clf.intercept_ = coef, inter
        tr_list.append(clf.predict_proba(Xtr)[:, 1])
        dv_list.append(clf.predict_proba(Xdv)[:, 1])
        aucs.append(float(roc_auc_score(ydv, dv_list[-1])))
        log(f"      sgd(seed={seed}) dev AUC {aucs[-1]:.4f}")
    return np.mean(tr_list, 0), np.mean(dv_list, 0), aucs


def lgb_fit(Ftr, ytr, Fdv):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="binary", n_estimators=800, learning_rate=0.05,
                           num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.6, reg_lambda=1.0, class_weight="balanced",
                           random_state=0, verbose=-1, n_jobs=8)
    m.fit(Ftr, ytr)
    return m.predict_proba(Ftr)[:, 1], m.predict_proba(Fdv)[:, 1]


def fusion_lr(Ptr5: np.ndarray, ytr, Pdv5: np.ndarray):
    """P0-fusion：5 成员 log-prob 特征 LR（C=1.0，仅 train 拟合）。"""
    from sklearn.linear_model import LogisticRegression
    Ztr = np.log(np.clip(Ptr5, 1e-6, 1))
    Zdv = np.log(np.clip(Pdv5, 1e-6, 1))
    meta = LogisticRegression(max_iter=3000, C=1.0).fit(Ztr, ytr)
    return meta.predict_proba(Ztr)[:, 1], meta.predict_proba(Zdv)[:, 1]


# ---------------------------------------------------------------- 指标
def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def _ap(y, s):
    from sklearn.metrics import average_precision_score
    try:
        return float(average_precision_score(y, s))
    except ValueError:
        return float("nan")


def _fast_auc(y, s):
    """rank 和法 AUROC（与 sklearn tie 处理一致；已验证差异 0）。"""
    from scipy.stats import rankdata
    y = np.asarray(y)
    s = np.asarray(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def task_macro_auc(y, s, tasks):
    tasks = np.asarray(tasks)
    uniq, inv = np.unique(tasks, return_inverse=True)
    vals = []
    for g in range(len(uniq)):
        sel = inv == g
        yy = y[sel]
        if yy.min() == yy.max():
            continue
        v = _fast_auc(yy, s[sel])
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def boot_metrics(y, s, tasks, repeats=BOOT, seed=BOOT_SEED):
    tasks = np.asarray([str(t) for t in tasks])
    uniq = np.unique(tasks)
    idx_by = {t: np.where(tasks == t)[0] for t in uniq}
    rng = np.random.default_rng(seed)
    a = np.empty(repeats)
    p = np.empty(repeats)
    tm = np.empty(repeats)
    for k in range(repeats):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[uniq[q]] for q in pick])
        y2 = y[idx]
        s2 = s[idx]
        a[k] = _auc(y2, s2)
        p[k] = _ap(y2, s2)
        tm[k] = task_macro_auc(y2, s2, tasks[idx])
    out = {"n": int(len(y)), "n_tasks": int(len(uniq)), "pos": int(y.sum()), "neg": int(len(y) - y.sum())}
    for name, arr, point in (("auroc", a, _auc(y, s)), ("ap", p, _ap(y, s)),
                             ("task_macro_auroc", tm, task_macro_auc(y, s, tasks))):
        out[name] = {"point": point,
                     "ci95_low": float(np.nanpercentile(arr, 2.5)),
                     "ci95_high": float(np.nanpercentile(arr, 97.5)),
                     "boot_mean": float(np.nanmean(arr))}
    return out


# ---------------------------------------------------------------- 数据加载
def load_split():
    split_of = {}
    with SPLIT_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            split_of[r["task_id"]] = r["split"]
    return split_of


def load_samples(split_of):
    """只加载 11 个系列成员且 split ∈ {train,dev} 的行；test 行不进入内存。"""
    members = set(PARAM_B)
    samples = []
    n_skipped_test = 0
    with RECORDS.open(encoding="utf-8") as f:
        for line in f:
            if "codellama--CodeLlama-" not in line and "Qwen--Qwen2.5-Coder-" not in line \
                    and "deepseek-ai--deepseek-coder-" not in line:
                continue
            d = json.loads(line)
            m = d.get("model_id")
            if m not in members:
                continue
            if d.get("subset") != "full" or d.get("generation_mode") != "instruct":
                continue
            sp = split_of.get(d["task_id"])
            if sp == "test":
                n_skipped_test += 1
                continue
            if sp not in ("train", "dev"):
                continue
            text = d.get("solution")
            if text is None:
                text = d.get("code")
            if text is None:
                raise KeyError(f"no text field for {m} {d['task_id']}")
            samples.append({"unit": m, "task": d["task_id"], "split": sp, "code": text})
    samples.sort(key=lambda r: (r["unit"], r["task"]))
    seen = set()
    dups = 0
    for r in samples:
        k = (r["unit"], r["task"])
        if k in seen:
            dups += 1
        seen.add(k)
    if dups:
        log(f"  WARNING: duplicated unit-task rows={dups}")
    log(f"  samples={len(samples)} (test rows excluded={n_skipped_test})")
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="仅验证特征编码（4 单元子集）")
    ap.add_argument("--feat-only", action="store_true", help="只生成样本与特征缓存")
    ap.add_argument("--folds-limit", type=int, default=0, help=">0 时只跑前 N 个折（调试）")
    args = ap.parse_args()

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predictions").mkdir(exist_ok=True)
    (OUT / "features").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)

    cfg = json.loads((PRE / "execution_config_frozen.json").read_text(encoding="utf-8"))
    amend = json.loads((PRE / "r1_negative_set_amendment.json").read_text(encoding="utf-8"))
    per_fold_sel = {k: v["size_matched_negatives"]["selected"] for k, v in
                    amend["size_matched_arm"]["per_fold"].items()}
    src_stat = json.loads((OUT / "source_status.json").read_text(encoding="utf-8"))

    log("[1] load split + samples")
    split_of = load_split()
    samples = load_samples(split_of)
    n = len(samples)
    assert n == 11 * 969, n
    units_present = sorted({r["unit"] for r in samples})
    assert len(units_present) == 11

    feat_dir = OUT / "features"
    log("[2] style/metadata features")
    t = time.time()
    style_mat = np.array([style_features(r["code"]) for r in samples], dtype=np.float64)
    log(f"  style {style_mat.shape} in {time.time()-t:.1f}s")
    meta_mat = style_mat[:, META_IDX].copy()
    szfeat = np.array([[np.log10(PARAM_B[r["unit"]]), style_mat[i, 0], style_mat[i, 1]]
                       for i, r in enumerate(samples)], dtype=np.float64)
    np.savez_compressed(feat_dir / "style_meta.npz", style=style_mat.astype(np.float32),
                        meta=meta_mat.astype(np.float32), sizelen=szfeat.astype(np.float32))

    emb_small_path = feat_dir / "emb_codet5_small.npz"
    emb_base_path = feat_dir / "emb_codet5_base.npz"
    if args.smoke:
        subset = [i for i, r in enumerate(samples)
                  if r["unit"] in SERIES["CodeLlama-Instruct"][:2] or
                  r["unit"] in ("Qwen--Qwen2.5-Coder-7B-Instruct", "deepseek-ai--deepseek-coder-6.7b-instruct")]
        texts = [samples[i]["code"] for i in subset]
        emb_small = encode_codet5(MODEL_SMALL, texts, device_batch=16)
        emb_base = encode_codet5(MODEL_BASE, texts, device_batch=8)
        np.savez_compressed(emb_small_path, emb=emb_small)
        np.savez_compressed(emb_base_path, emb=emb_base)
        np.savez_compressed(feat_dir / "smoke_index.npz", idx=np.array(subset))
        log(f"  smoke embeddings small{emb_small.shape} base{emb_base.shape}")
        return

    log("[3] CodeT5-small embedding (GPU)")
    if emb_small_path.exists() and emb_base_path.exists() and not args.smoke:
        log("  cache hit: load embeddings from npz")
        emb_small = np.load(emb_small_path)["emb"]
        emb_base = np.load(emb_base_path)["emb"]
    else:
        t = time.time()
        emb_small = encode_codet5(MODEL_SMALL, [r["code"] for r in samples], device_batch=16)
        log(f"  small {emb_small.shape} in {time.time()-t:.1f}s")
        np.savez_compressed(emb_small_path, emb=emb_small)
        log("[4] CodeT5-base embedding (GPU)")
        t = time.time()
        emb_base = encode_codet5(MODEL_BASE, [r["code"] for r in samples], device_batch=8)
        log(f"  base {emb_base.shape} in {time.time()-t:.1f}s")
        np.savez_compressed(emb_base_path, emb=emb_base)
    manifest = {
        "schema": "variant_transfer_stage1_feature_manifest_v1",
        "samples": n, "order": "(unit, task) ascending",
        "style_meta_npz": {"path": "features/style_meta.npz", "sha256": sha256_file(feat_dir / "style_meta.npz")},
        "emb_small_npz": {"path": "features/emb_codet5_small.npz", "shape": list(emb_small.shape),
                          "sha256": sha256_file(emb_small_path)},
        "emb_base_npz": {"path": "features/emb_codet5_base.npz", "shape": list(emb_base.shape),
                         "sha256": sha256_file(emb_base_path)},
        "protocol": "codet5 frozen mean-pool 512=384+128 no special tokens fp16; style=regex stylometry (P0)",
    }
    (feat_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if args.feat_only:
        log(f"feat-only done in {time.time()-t0:.1f}s")
        return

    # ---- 索引 ----
    index = {(r["unit"], r["task"]): i for i, r in enumerate(samples)}
    unit_rows = defaultdict(list)
    for i, r in enumerate(samples):
        unit_rows[r["unit"]].append(i)

    all_folds = {}
    r2_all = {}
    log("[5] folds")
    combos = [(series, h, arm) for series in SERIES for h in SERIES[series] for arm in ARM_NAMES]
    if args.folds_limit:
        combos = combos[:args.folds_limit]
    for series, h, arm in combos:
        key_fold = f"{series}::heldout={SIZE_LABEL[h]}"
        pos = [m for m in SERIES[series] if m != h]
        neg_all = [m for ss in SERIES if ss != series for m in SERIES[ss]]
        neg = neg_all if arm == "size_mix" else per_fold_sel[key_fold]
        units = pos + [m for m in neg]
        tr_idx = np.array(sorted(i for u in units for i in unit_rows[u]
                                 if samples[i]["split"] == "train"))
        dv_idx = np.array(sorted(i for u in units for i in unit_rows[u]
                                 if samples[i]["split"] == "dev"))
        y_tr = np.array([0 if samples[i]["unit"] in neg else 1 for i in tr_idx])
        y_dv = np.array([0 if samples[i]["unit"] in neg else 1 for i in dv_idx])
        tasks_dv = [samples[i]["task"] for i in dv_idx]
        fold_key = f"{key_fold}::{arm}"
        log(f"  [{fold_key}] train={len(tr_idx)} (pos={y_tr.sum()}) dev={len(dv_idx)} (pos={y_dv.sum()})")
        res = {"readouts": {}, "seeds_detail": {}}

        # TF-IDF
        tr_texts = [samples[i]["code"] for i in tr_idx]
        dv_texts = [samples[i]["code"] for i in dv_idx]
        Xc_tr = Xc_dv = Xw_tr = Xw_dv = None
        for tag, factory in (("char", tfidf_char), ("word", tfidf_word)):
            t = time.time()
            vec = factory().fit(tr_texts)
            A = vec.transform(tr_texts)
            B = vec.transform(dv_texts)
            log(f"    tfidf_{tag} vocab={A.shape[1]} in {time.time()-t:.1f}s")
            if tag == "char":
                Xc_tr, Xc_dv = A, B
            else:
                Xw_tr, Xw_dv = A, B

        P_tr, P_dv = {}, {}

        def add_readout(name, ptr, pdv):
            P_tr[name] = ptr
            P_dv[name] = pdv

        log("    char SGD")
        ptr, pdv, aucs = sgd_seeds(Xc_tr, y_tr, Xc_dv, y_dv)
        add_readout("tfidf_char", ptr, pdv)
        res["seeds_detail"]["tfidf_char"] = aucs
        log("    word SGD")
        ptr, pdv, aucs = sgd_seeds(Xw_tr, y_tr, Xw_dv, y_dv)
        add_readout("tfidf_word", ptr, pdv)
        res["seeds_detail"]["tfidf_word"] = aucs

        st_tr, st_dv = style_mat[tr_idx], style_mat[dv_idx]
        eb_tr, eb_dv = emb_base[tr_idx], emb_base[dv_idx]
        es_tr, es_dv = emb_small[tr_idx], emb_small[dv_idx]
        mt_tr, mt_dv = meta_mat[tr_idx], meta_mat[dv_idx]
        sl_tr, sl_dv = szfeat[tr_idx], szfeat[dv_idx]

        log("    sem_base")
        ptr, pdv, Cbase = lr_select_auroc(eb_tr, y_tr, eb_dv, y_dv)
        add_readout("sem_base", ptr, pdv)
        log("    sem_small")
        ptr, pdv, Csmall = lr_select_auroc(es_tr, y_tr, es_dv, y_dv)
        add_readout("sem_small", ptr, pdv)
        log("    style_lr")
        ptr, pdv, Cstyle = lr_select_auroc(st_tr, y_tr, st_dv, y_dv)
        add_readout("style_lr", ptr, pdv)
        log("    style_lgb")
        ptr, pdv = lgb_fit(st_tr, y_tr, st_dv)
        add_readout("style_lgb", ptr, pdv)
        log("    metadata_only")
        ptr, pdv, Cmeta = lr_select_auroc(mt_tr, y_tr, mt_dv, y_dv)
        add_readout("metadata_only", ptr, pdv)
        log("    size_length_only")
        ptr, pdv, Csl = lr_select_auroc(sl_tr, y_tr, sl_dv, y_dv)
        add_readout("size_length_only", ptr, pdv)

        log("    P0_fusion / P0_equal")
        Ptr5 = np.column_stack([P_tr[k] for k in P0_MEMBERS])
        Pdv5 = np.column_stack([P_dv[k] for k in P0_MEMBERS])
        ptr, pdv = fusion_lr(Ptr5, y_tr, Pdv5)
        add_readout("P0_fusion", ptr, pdv)
        add_readout("P0_equal", Pdv5.mean(1), Pdv5.mean(1))
        res["C_selected"] = {"sem_base": Cbase, "sem_small": Csmall, "style_lr": Cstyle,
                            "metadata_only": Cmeta, "size_length_only": Csl}

        # 指标 + bootstrap + 预测
        fold_pred = []
        metrics = {}
        for name, pdv in P_dv.items():
            m = boot_metrics(y_dv, pdv, tasks_dv)
            metrics[name] = m
            fold_pred.append(pdv)
        res["metrics"] = metrics
        with gzip.open(OUT / "predictions" / f"{fold_key.replace('::','__')}.csv.gz",
                       "wt", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["readout", "unit", "task", "y", "score"])
            for name, pdv in P_dv.items():
                for j, i in enumerate(dv_idx):
                    w.writerow([name, samples[i]["unit"], samples[i]["task"],
                                int(y_dv[j]), f"{pdv[j]:.6f}"])
        # R2：系列中心
        r2 = {}
        for rep_name, (Rtr, Rdv) in {
                "codet5_base": (eb_tr, eb_dv), "codet5_small": (es_tr, es_dv)}.items():
            from sklearn.preprocessing import StandardScaler
            pos_tr_mask = y_tr == 1
            sc = StandardScaler().fit(Rtr[pos_tr_mask])
            Ztr = sc.transform(Rtr)
            Zd = sc.transform(Rdv)
            c = Ztr[pos_tr_mask].mean(0)
            d_tr = -np.linalg.norm(Ztr - c, axis=1)
            d_dv = -np.linalg.norm(Zd - c, axis=1)
            cos_tr = (Ztr @ c) / (np.linalg.norm(Ztr, axis=1) * np.linalg.norm(c) + 1e-12)
            cos_dv = (Zd @ c) / (np.linalg.norm(Zd, axis=1) * np.linalg.norm(c) + 1e-12)
            from scipy.stats import spearmanr
            sz = np.array([np.log10(PARAM_B[samples[i]["unit"]]) for i in tr_idx])
            rho_e, _ = spearmanr(d_tr, sz)
            r2[rep_name] = {
                "euclid": boot_metrics(y_dv, d_dv, tasks_dv),
                "cosine": boot_metrics(y_dv, cos_dv, tasks_dv),
                "rank_corr_train_dist_vs_logsize": {"euclid": float(rho_e)},
            }
        res["r2"] = r2
        all_folds[fold_key] = res
        r2_all[fold_key] = {
            rep: {"euclid_auroc": v["euclid"]["auroc"]["point"],
                  "euclid_ci95": [v["euclid"]["auroc"]["ci95_low"], v["euclid"]["auroc"]["ci95_high"]],
                  "cosine_auroc": v["cosine"]["auroc"]["point"],
                  "rank_corr_train_dist_vs_logsize": v["rank_corr_train_dist_vs_logsize"]["euclid"]}
            for rep, v in r2.items()}

    out = {
        "schema": "variant_transfer_stage1_train_dev_metrics_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_status": src_stat["source_status"],
        "original_bundle_verified": src_stat["original_bundle_verified"],
        "claims_of_byte_identity": src_stat["claims_of_byte_identity"],
        "authority": "《服务器重构版最小授权与补传豁免》2026-10-08 §1/§4（train/dev）",
        "config_hashes": {p.name: sha256_file(p) for p in
                          (PRE / "execution_config_frozen.json", PRE / "r1_negative_set_amendment.json")},
        "protocol": {
            "dev_selection": "C 网格按 dev AUROC 最大、平局取更小 C；SGD 5ep best-dev×3 seeds 均值；LGBM(800) 固定",
            "metrics": "AUROC / AP / task-macro AUROC + task-cluster bootstrap 500 (seed 20261008)",
            "test_read": False, "generation": False, "weights_downloaded": False,
            "negative_arms": list(ARM_NAMES),
        },
        "folds": all_folds,
        "r2_summary": r2_all,
        "runtime_seconds": time.time() - t0,
    }
    (OUT / "train_dev_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "logs" / "stage1_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
