#!/usr/bin/env python
"""DCAN 风格四模型第一轮（h2_authorbench_dcan；轻量语义/来源分离原型）。

模型（全部任务级留出评测，seen families）：
  M1 semantic-only   : z_s=MLP(冻结 CodeT5-base 均值池化 768d)；CE(family)+0.3·SupCon(同 task 不同 family 正对)
  M2 fingerprint-only: z_f=MLP(结构统计特征 ~30 维)；CE(family)
  M3 late-fusion     : 复用 M1/M2 分支输出，LR 融合（train 拟合）
  M4 full-disentangle: z_s(SupCon+GRL family adversary) + z_f(CE+GRL 长度/提示簇 adversary) + 交叉协方差正交项；
                       主输出=fingerprint head，次输出=LR([z_s,z_f]) 融合
约定：
  - 语义正对只用同 task_id 内不同 family 的样本（仅 train 任务）；**Droid 不参与任何损失**；
  - 另附 OpenAI 留出（unknown-family AUROC，1 seed；训练时从 train+dev 删除 openai 行）；
  - seed ≤3；特征缓存 runs/acl_dcan_round1/emb_dcan_ct5.npz（~29MB，不入库）。
输出：artifacts/acl_dcan_round1/dcan_four_models/{config,metrics}.json + predictions.npz
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "acl_dcan_round1" / "dcan_four_models"
CACHE = ROOT / "runs" / "acl_dcan_round1"
FAMILIES = ["claude", "deepseek", "gemini", "llama", "openai", "qwen"]


def log(msg: str) -> None:
    print(f"[dcan] {msg}", flush=True)


# ------------------------------------------------------------------ 数据
def load_rows():
    rows = []
    with (DATA / "h2_authorbench_dcan" / "core.jsonl").open() as fh:
        for line in fh:
            d = json.loads(line)
            d["code"] = d.get("code") or ""
            rows.append(d)
    return rows


def struct_features(d: dict) -> list[float]:
    code = d["code"]
    lines = code.split("\n")
    n = max(1, len(lines))
    lens = [len(l) for l in lines]
    indent = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    blank_runs = [len(list(g)) for k, g in __import__("itertools").groupby(l.strip() == "" for l in lines) if k]
    nchar = max(1, len(code))
    return [
        d.get("char_count", nchar), d.get("num_lines", n), d.get("nloc", n),
        d.get("cyclomatic_complexity", 0.0), d.get("token_size", 0),
        float(np.mean(lens)), float(np.max(lens)), sum(1 for l in lines if not l.strip()) / n,
        float(np.mean(indent)) if indent else 0.0, float(np.std(indent)) if indent else 0.0,
        sum(l.count("\t") for l in lines) / n, sum(1 for l in lines if l != l.rstrip()) / n,
        sum(1 for l in lines if l.lstrip().startswith(("//", "#"))) / n,
        code.count("{") / n, code.count(";") / n, code.count("(") / n,
        sum(c.isdigit() for c in code) / nchar, sum(c.isupper() for c in code) / nchar,
        code.count("_") / nchar, sum(1 for c in code if ord(c) > 127) / nchar,
        code.count(" ") / nchar, code.count('"') / n, sum(1 for L in lens if L > 120) / n,
        float(max(blank_runs, default=0)),
        code.count("printf") / n, code.count("scanf") / n, code.count("malloc") / n,
        code.count("return") / n, code.count("if") / n, code.count("for") / n, code.count("while") / n,
    ]


def encode_semantics(rows, batch: int = 8, max_length: int = 512):
    """冻结 CodeT5-base 均值池化（与 AuthorBench 口径一致：>512 头 384+尾 128）。"""
    cache = CACHE / "emb_dcan_ct5.npz"
    CACHE.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if len(z["emb"]) == len(rows) and list(z["sha"]) == [r["source_sha256"] for r in rows]:
            log(f"semantic cache hit: {cache}")
            return z["emb"]
    import sys
    sys.path.insert(0, str(ROOT))
    from encoders.codet5 import CodeT5Encoder
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    model = CodeT5Encoder(path=str(ROOT / "checkpoints/codet5-base"), dtype="float32",
                          freeze=True, max_length=max_length).cuda()
    model.eval()
    emb = np.zeros((len(rows), 768), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            ids = []
            for d in chunk:
                t = tok(d["code"], add_special_tokens=False, truncation=False)["input_ids"]
                if len(t) > max_length:
                    t = t[:384] + t[-128:]
                ids.append(torch.tensor(t))
            pad = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True).cuda()
            mask = (pad != tok.pad_token_id).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model(pad, attention_mask=mask)
            h = h.float()
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            emb[i:i + len(chunk)] = pooled.cpu().numpy()
            if i % 1600 == 0:
                log(f"encode {i}/{len(rows)} ({time.time() - t0:.0f}s)")
    np.savez_compressed(cache, emb=emb, sha=np.array([r["source_sha256"] for r in rows], dtype=object),
                        note="frozen CodeT5-base mean-pool 768d; bf16 inference, float32 cache")
    log(f"semantic encode done: {len(rows)} rows in {time.time() - t0:.0f}s -> {cache}")
    return emb


def build_clusters(rows, train_mask, k=64, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform([r["prompt"] for r, m in zip(rows, train_mask) if m])
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X)
    Xall = vec.transform([r["prompt"] for r in rows])
    return km.predict(Xall)


def length_bins(rows, train_mask, nb=10):
    vals = np.array([r.get("char_count", len(r["code"])) for r in rows], dtype=np.float64)
    qs = np.quantile(vals[train_mask], np.linspace(0, 1, nb + 1)[1:-1])
    return np.digitize(vals, qs)


# ------------------------------------------------------------------ 模型
class MLP(nn.Module):
    def __init__(self, d_in, d_h=256, d_out=128, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.BatchNorm1d(d_h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_h, d_out),
        )

    def forward(self, x):
        return self.net(x)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grl(x, lambd=1.0):
    return GRL.apply(x, lambd)


def supcon(z, task_ids, fam_ids, tau=0.1):
    z = F.normalize(z, dim=1)
    B = z.size(0)
    sim = z @ z.t() / tau
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    same_task = task_ids[:, None] == task_ids[None, :]
    pos = same_task & (fam_ids[:, None] != fam_ids[None, :]) & ~eye
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    cnt = pos.sum(1)
    valid = cnt > 0
    if not valid.any():
        return z.sum() * 0.0
    loss = -(log_prob.masked_fill(~pos, 0.0).sum(1) / cnt.clamp(min=1))[valid]
    return loss.mean()


def cross_cov(z_s, z_f):
    zs = z_s - z_s.mean(0, keepdim=True)
    zf = z_f - z_f.mean(0, keepdim=True)
    c = (zs.t() @ zf) / z_s.size(0)
    return (c ** 2).mean()


# ------------------------------------------------------------------ 训练/评测
def make_task_batches(idx, task_ids, rng, n_task=48, max_per_task=8):
    tasks = defaultdict(list)
    for j in idx:
        tasks[task_ids[j]].append(j)
    keys = list(tasks)
    order = rng.permutation(len(keys))
    keys = [keys[i] for i in order]
    out = []
    for i in range(0, len(keys), n_task):
        js = []
        for k in keys[i:i + n_task]:
            rows_k = tasks[k]
            take = rows_k if len(rows_k) <= max_per_task else list(rng.choice(rows_k, max_per_task, replace=False))
            js.extend(take)
        out.append(np.array(sorted(js)))
    return out


def evaluate(z, head, X, y, groups=None):
    import sklearn.metrics as sm
    logits = head(X)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    pred = probs.argmax(1)
    m = {"n": int(len(y)),
         "macro_f1": float(sm.f1_score(y, pred, average="macro", zero_division=0)),
         "balanced_acc": float(sm.balanced_accuracy_score(y, pred)),
         "per_class_recall": {FAMILIES[i]: float((pred[y == i] == i).mean()) if (y == i).any() else None
                              for i in range(len(FAMILIES))},
         "confusion_matrix": sm.confusion_matrix(y, pred, labels=list(range(len(FAMILIES)))).tolist(),
         "ece_top1_15": float(ece(probs, y))}
    if groups:
        m["groups"] = {}
        for name, arr in groups.items():
            m["groups"][name] = {}
            for g in sorted(set(arr)):
                sel = np.array([a == g for a in arr])
                m["groups"][name][g] = {"n": int(sel.sum()),
                                        "acc": float((pred[sel] == y[sel]).mean())}
    return m, probs


def ece(probs, y, bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += (m.sum() / len(y)) * abs(acc[m].mean() - conf[m].mean())
    return e


def run_seed(seed, emb, st, fam, task, split, prompt_clu, len_bin, args, mode="main", drop_family=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fam_idx = {f: i for i, f in enumerate(FAMILIES)}
    y = np.array([fam_idx[f] for f in fam])
    tr = np.array([i for i, s in enumerate(split) if s == "train"])
    dv = np.array([i for i, s in enumerate(split) if s == "dev"])
    te = np.array([i for i, s in enumerate(split) if s == "test"])
    if drop_family is not None:
        tr = np.array([i for i in tr if fam[i] != drop_family])
        dv = np.array([i for i in dv if fam[i] != drop_family])
    Xe = torch.tensor(emb, dtype=torch.float32, device=dev)
    Xs = torch.tensor(st, dtype=torch.float32, device=dev)
    Y = torch.tensor(y, dtype=torch.long, device=dev)
    T = np.array(task)
    Cc = torch.tensor(prompt_clu, dtype=torch.long, device=dev)
    Lb = torch.tensor(len_bin, dtype=torch.long, device=dev)
    rng = np.random.default_rng(seed)

    mlp_s = MLP(emb.shape[1]).to(dev)
    mlp_f = MLP(st.shape[1]).to(dev)
    head_s = nn.Linear(128, len(FAMILIES)).to(dev)
    head_f = nn.Linear(128, len(FAMILIES)).to(dev)
    adv_s = nn.Linear(128, len(FAMILIES)).to(dev)            # family adversary on z_s
    adv_len = nn.Linear(128, int(len_bin.max()) + 1).to(dev)  # length adversary on z_f
    adv_clu = nn.Linear(128, int(prompt_clu.max()) + 1).to(dev)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(list(mlp_s.parameters()) + list(mlp_f.parameters()) +
                            list(head_s.parameters()) + list(head_f.parameters()) +
                            list(adv_s.parameters()) + list(adv_len.parameters()) +
                            list(adv_clu.parameters()), lr=1e-3, weight_decay=1e-4)

    want_s = mode in ("semantic_only", "late_fusion", "full_disentangle")
    want_f = mode in ("fingerprint_only", "late_fusion", "full_disentangle")
    if mode == "semantic_only":
        want_f = False
    best = {"f1": -1, "state": None, "epoch": 0}
    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(tr)
        for bidx in make_task_batches(perm, T, rng):
            xb_e, xb_s = Xe[bidx], Xs[bidx]
            yb = Y[bidx]
            tb = torch.tensor(T[bidx], dtype=torch.long, device=dev)
            loss = torch.zeros((), device=dev)
            if want_s:
                z_s = mlp_s(xb_e)
                loss = loss + ce(head_s(z_s), yb)
                loss = loss + 0.3 * supcon(z_s, tb, yb)
                if mode == "full_disentangle":
                    loss = loss + 0.3 * ce(adv_s(grl(z_s, 1.0)), yb)
            if want_f:
                z_f = mlp_f(xb_s)
                loss = loss + ce(head_f(z_f), yb)
                if mode == "full_disentangle":
                    loss = loss + 0.2 * ce(adv_len(grl(z_f, 1.0)), Lb[bidx])
                    loss = loss + 0.2 * ce(adv_clu(grl(z_f, 1.0)), Cc[bidx])
            if mode == "full_disentangle":
                loss = loss + 0.05 * cross_cov(z_s, z_f)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            zs_e = mlp_s(Xe[dv]) if want_s else None
            zf_e = mlp_f(Xs[dv]) if want_f else None
            # dev 选择：按主输出（disentangle=fingerprint head；semantic_only=语义头；其他按可用分支）
            if mode == "full_disentangle":
                logits = head_f(zf_e)
            elif mode == "semantic_only":
                logits = head_s(zs_e)
            elif mode == "fingerprint_only":
                logits = head_f(zf_e)
            else:
                ps = F.softmax(head_s(zs_e), 1); pf = F.softmax(head_f(zf_e), 1)
                probs = (ps + pf) / 2
                logits = probs  # 仅用于选 epoch 的 argmax
            import sklearn.metrics as sm
            pred = (logits.argmax(1) if logits.shape[1] > 1 else logits).cpu().numpy()
            f1 = sm.f1_score(Y[dv].cpu().numpy(), pred, average="macro", zero_division=0)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in
                              {**{f"ms.{i}": p for i, p in enumerate(mlp_s.state_dict().values())},
                               **{f"mf.{i}": p for i, p in enumerate(mlp_f.state_dict().values())},
                               **{f"hs.{i}": p for i, p in enumerate(head_s.state_dict().values())},
                               **{f"hf.{i}": p for i, p in enumerate(head_f.state_dict().values())}}.items()}}
        if ep - best["epoch"] >= 10:
            break
    # 恢复 best state
    if best["state"] is not None:
        for i, p in enumerate(mlp_s.state_dict().values()):
            p.copy_(best["state"][f"ms.{i}"])
        for i, p in enumerate(mlp_f.state_dict().values()):
            p.copy_(best["state"][f"mf.{i}"])
        for i, p in enumerate(head_s.state_dict().values()):
            p.copy_(best["state"][f"hs.{i}"])
        for i, p in enumerate(head_f.state_dict().values()):
            p.copy_(best["state"][f"hf.{i}"])

    with torch.no_grad():
        zs_tr = mlp_s(Xe[tr]); zf_tr = mlp_f(Xs[tr])
        zs_te = mlp_s(Xe[te]); zf_te = mlp_f(Xs[te])
        outs = {}
        if mode in ("semantic_only", "late_fusion", "full_disentangle"):
            outs["sem"] = F.softmax(head_s(zs_te), 1).cpu().numpy()
            outs["sem_tr"] = F.softmax(head_s(zs_tr), 1).cpu().numpy()
        if mode in ("fingerprint_only", "late_fusion", "full_disentangle"):
            outs["fp"] = F.softmax(head_f(zf_te), 1).cpu().numpy()
            outs["fp_tr"] = F.softmax(head_f(zf_tr), 1).cpu().numpy()
        if mode in ("late_fusion", "full_disentangle"):
            outs["z_tr"] = torch.cat([zs_tr, zf_tr], 1).cpu().numpy()
            outs["z_te"] = torch.cat([zs_te, zf_te], 1).cpu().numpy()
    results = {"seed": seed, "mode": mode, "best_dev_f1": best["f1"], "best_epoch": best["epoch"],
               "test": {}, "probs": {}}
    y_te = y[te]
    groups = {"generator": [rows_meta["model_name"][i] for i in te]}
    # 主输出
    if mode == "semantic_only":
        m, p = evaluate(None, head_s, zs_te, y_te, groups); results["test"]["sem"] = m; results["probs"]["sem"] = p
    elif mode == "fingerprint_only":
        m, p = evaluate(None, head_f, zf_te, y_te, groups); results["test"]["fp"] = m; results["probs"]["fp"] = p
    elif mode == "late_fusion":
        m, p = evaluate(None, head_s, zs_te, y_te, groups); results["test"]["sem"] = m
        m2, p2 = evaluate(None, head_f, zf_te, y_te, groups); results["test"]["fp"] = m2
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(outs["z_tr"], y[tr])
        probs = lr.predict_proba(outs["z_te"])
        results["probs"]["fuse"] = probs
        results["test"]["fuse"] = metric_from_probs(probs, y_te, groups)
    else:
        m2, p2 = evaluate(None, head_f, zf_te, y_te, groups); results["test"]["fp"] = m2; results["probs"]["fp"] = p2
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(outs["z_tr"], y[tr])
        probs = lr.predict_proba(outs["z_te"])
        results["probs"]["fuse"] = probs
        results["test"]["fuse"] = metric_from_probs(probs, y_te, groups)
    results["te_idx"] = te.tolist()
    return results


def metric_from_probs(probs, y, groups=None):
    import sklearn.metrics as sm
    m = {"n": int(len(y)),
         "macro_f1": float(sm.f1_score(y, probs.argmax(1), average="macro", zero_division=0)),
         "balanced_acc": float(sm.balanced_accuracy_score(y, probs.argmax(1))),
         "ece_top1_15": float(ece(probs, y))}
    if groups:
        m["groups"] = {}
        for name, arr in groups.items():
            m["groups"][name] = {}
            for g in sorted(set(arr)):
                sel = np.array([a == g for a in arr])
                m["groups"][name][g] = {"n": int(sel.sum()), "acc": float((probs.argmax(1)[sel] == y[sel]).mean())}
    return m


rows_meta = {"model_name": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--no-aux", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = 1; args.epochs = 8
    t0 = time.time()
    rows = load_rows()
    log(f"{len(rows)} rows; splits {Counter(r['task_split'] for r in rows)}")
    rows_meta["model_name"] = [r["model_name"] for r in rows]
    emb = encode_semantics(rows)
    st_raw = np.array([struct_features(r) for r in rows], dtype=np.float32)
    split = [r["task_split"] for r in rows]
    fam = [r["family"] for r in rows]
    task_map = {t: i for i, t in enumerate(dict.fromkeys(r["task_id"] for r in rows))}
    task = [task_map[r["task_id"]] for r in rows]
    tr_mask = np.array([s == "train" for s in split])
    mu, sd = st_raw[tr_mask].mean(0), st_raw[tr_mask].std(0) + 1e-6
    st = (st_raw - mu) / sd
    prompt_clu = build_clusters(rows, tr_mask)
    len_bin = length_bins(rows, tr_mask)
    log(f"features ready: emb {emb.shape}, struct {st.shape}, clusters {len(set(prompt_clu.tolist()))}")

    results = {"config": {"script": "scripts/dcan_four_models.py", "seeds": args.seeds,
                          "epochs": args.epochs, "family_order": FAMILIES,
                          "semantic": "frozen CodeT5-base mean-pool 768d (bf16 infer)",
                          "fingerprint": f"struct features {st.shape[1]} (train z-score)",
                          "losses": {"supcon": 0.3, "adv_family_on_sem": 0.3,
                                     "adv_len+cluster_on_fp": 0.2, "orth": 0.05},
                          "semantic_positives": "same task_id, different family (train tasks only)",
                          "droid_used": False},
               "runs": [], "aux_openai_heldout": []}
    modes = ["semantic_only", "fingerprint_only", "late_fusion", "full_disentangle"]
    pack = {}
    for seed in range(args.seeds):
        for mode in modes:
            r = run_seed(seed, emb, st, fam, task, split, prompt_clu, len_bin, args, mode=mode)
            results["runs"].append({k: v for k, v in r.items() if k != "probs"})
            for k, p in r["probs"].items():
                pack[f"{mode}_s{seed}_{k}"] = p.astype(np.float16)
            log(f"seed {seed} {mode}: " + json.dumps(
                {k: round(v["macro_f1"], 4) for k, v in r["test"].items()}, ensure_ascii=False))
        pack[f"te_idx_s{seed}"] = np.array(r["te_idx"], dtype=np.int32)
    pack["y_test"] = np.array([FAMILIES.index(f) for f, s in zip(fam, split) if s == "test"],
                              dtype=np.int16)
    pack["model_name_test"] = np.array([r["model_name"] for r, s in zip(rows, split) if s == "test"],
                                       dtype=object)
    if not args.no_aux:
        for mode in modes:
            r = run_seed(0, emb, st, fam, task, split, prompt_clu, len_bin, args, mode=mode,
                         drop_family="openai")
            # unknown-family AUROC：以主输出 max-prob 作 knownness
            key = {"semantic_only": "sem", "fingerprint_only": "fp", "late_fusion": "fuse",
                   "full_disentangle": "fp"}[mode]
            probs = r["probs"][key]
            y_te = np.array([{"claude": 0, "deepseek": 1, "gemini": 2, "llama": 3, "openai": 4, "qwen": 5}[f]
                             for f in fam])[np.array(r["te_idx"])]
            unknown = (y_te == 4).astype(int)
            import sklearn.metrics as sm
            auroc = float(sm.roc_auc_score(unknown, 1 - probs.max(1))) if unknown.sum() and (1 - unknown).sum() else None
            results["aux_openai_heldout"].append({"mode": mode, "key": key, "auroc_unknown": auroc,
                                                  "n_unknown": int(unknown.sum()),
                                                  "f1_known_only": float(sm.f1_score(
                                                      y_te[unknown == 0], probs.argmax(1)[unknown == 0],
                                                      average="macro", zero_division=0))})
            log(f"aux openai-heldout {mode}: AUROC {auroc}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps(results["config"], ensure_ascii=False, indent=1))
    (OUT / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    np.savez_compressed(OUT / "predictions.npz", **pack)
    log(f"done in {time.time() - t0:.0f}s -> {OUT} (predictions {len(pack)} keys)")


if __name__ == "__main__":
    main()
