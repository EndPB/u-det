"""C3: 论文增强的一致性训练（server-side, 2026-10-09）.

三种变换（独立 Bernoulli，p=.2/.3/.4，固定 seed）：
  comment_strip : tokenize 精确删除 COMMENT（保留 shebang/编码/工具指令/docstring）
  whitespace    : 仅删行尾空白 + 折叠连续空行（多行字符串保护；AST 不变）
  rename        : AST 作用域安全改名（仅函数内局部变量，不动参数/入口/公开名；
                  反射敏感跳过；接受样本必须 AST 等价 mod rename 且接口 hash 不变）
每个尝试都记录 transform_acceptance.jsonl（含拒绝原因）。
训练：L = mean_BCE({orig}∪{accepted transforms}) + λinv·mean_JS(p_orig, p_trans)
（λinv=0.1 冻结；同 C1 full 视图与骨干 3 seeds）。
评测：C3(原码分数) vs C0 配对 Δ；变换后归因保持（transformed-eval AUROC）；
原/变换配对一致性（|Δp|、符号一致率）。
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import sys
import time
import tokenize
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cc_common as cc  # noqa: E402
import variant_transfer_stage1_execute as s1  # noqa: E402

OUT = ROOT / "d-det/artifacts/mainline_c0c3_batch_2026-10-10/c3"
C0_DIR = ROOT / "d-det/artifacts/code_conditioned_fresh_c0_canonical_2026-10-10"
TASKMETA = ROOT / "d-det/data/public_same_task_full_2026-10-07/bigcodebench_task_metadata.jsonl"
SEEDS = (0, 1, 2)
EPOCHS = 20
LAMBDA_INV = 0.1
LOG: list[str] = []
KEEP_COMMENT = re.compile(r"#\s*(!|type:|noqa|pylint|fmt:|pragma|-\*-|coding[:=]|"
                          r"vim:|isort:|mypy:|flake8)", re.I)
REFLECT = re.compile(r"\b(eval|exec|getattr|setattr|delattr|globals|locals|vars|"
                     r"__dict__|compile)\b")
PSI_FIELDS = ["parse_ok", "entrypoint_present", "n_ast_nodes", "n_calls", "n_branches",
              "n_handlers", "n_returns", "n_import_roots", "required_root_overlap_proxy"]


def log(m):
    print(m, flush=True)
    LOG.append(m)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ast_hash(code: str):
    try:
        return hashlib.sha256(ast.dump(ast.parse(code)).encode()).hexdigest()[:16]
    except Exception:
        return None


def interface_sig(code: str):
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    sigs = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sigs.append((n.name, tuple(a.arg for a in n.args.posonlyargs),
                         tuple(a.arg for a in n.args.args),
                         tuple(a.arg for a in n.args.kwonlyargs)))
    return hashlib.sha256(repr(sorted(sigs)).encode()).hexdigest()[:16]


def splice_lines(code: str, repls):
    lines = code.splitlines(keepends=True)
    for row, c0, c1, text in sorted(repls, key=lambda r: (r[0], r[1]), reverse=True):
        line = lines[row - 1]
        lines[row - 1] = line[:c0] + text + line[c1:]
    return "".join(lines)


def transform_comment(code: str):
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except Exception as e:
        return None, f"tokenize_fail:{type(e).__name__}"
    repls = []
    for t in toks:
        if t.type == tokenize.COMMENT:
            s = t.string
            if t.start[0] == 1 and s.startswith("#!"):
                continue
            if KEEP_COMMENT.match(s):
                continue
            repls.append((t.start[0], t.start[1], t.end[1], ""))
    if not repls:
        return None, "no_comment"
    new = splice_lines(code, repls)
    h1, h2 = ast_hash(code), ast_hash(new)
    if h1 is None or h2 != h1:
        return None, "ast_changed"
    return new, "ok"


def transform_whitespace(code: str):
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except Exception as e:
        return None, f"tokenize_fail:{type(e).__name__}"
    protected = set()
    for t in toks:
        if t.type == tokenize.STRING and "\n" in t.string:
            protected.update(range(t.start[0], t.end[0] + 1))
    lines = code.splitlines(keepends=True)
    out, blanks, changed = [], 0, 0
    for i, ln in enumerate(lines, start=1):
        if i in protected:
            out.append(ln)
            blanks = 0
            continue
        nl = "\n" if ln.endswith("\n") else ""
        stripped = ln.rstrip(" \t\r\n")
        if stripped == "":
            blanks += 1
            if blanks == 1:
                out.append(nl)
            else:
                changed += 1  # drop extra blank
            if ln != nl:
                changed += 1
            continue
        blanks = 0
        newln = stripped + nl
        if ln != newln:
            changed += 1
        out.append(newln)
    if changed == 0:
        return None, "no_change"
    new = "".join(out)
    h1, h2 = ast_hash(code), ast_hash(new)
    if h1 is None or h2 != h1:
        return None, "ast_changed"
    return new, "ok"


def _collect_scope_names(fn: ast.AST):
    names = set()

    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                names.add(child.id)
            walk(child)

    walk(fn)
    return names


def _ast_equal_mod_rename(a: ast.AST, b: ast.AST, ren: dict) -> bool:
    def cmp(x, y):
        if type(x) is not type(y):
            return False
        if isinstance(x, ast.Name):
            return x.id == y.id or ren.get(x.id) == y.id
        if isinstance(x, ast.arg):
            return x.arg == y.arg or ren.get(x.arg) == y.arg
        if isinstance(x, ast.Attribute) and x.attr != y.attr:
            return False
        if isinstance(x, (ast.Global, ast.Nonlocal)) and x.names != y.names:
            return False
        if isinstance(x, ast.Constant) and x.value != y.value:
            return False
        if isinstance(x, ast.alias) and (x.name != y.name or x.asname != y.asname):
            return False
        for fa in x._fields:
            va, vb = getattr(x, fa), getattr(y, fa)
            if isinstance(va, ast.AST):
                if not cmp(va, vb):
                    return False
            elif isinstance(va, list):
                if len(va) != len(vb):
                    return False
                for ua, ub in zip(va, vb):
                    if isinstance(ua, ast.AST):
                        if not cmp(ua, ub):
                            return False
                    elif ua != ub:
                        return False
            elif va != vb:
                return False
        return True

    return cmp(a, b)


def transform_rename(code: str):
    if REFLECT.search(code):
        return None, "reflection_sensitive"
    try:
        tree = ast.parse(code)
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except Exception as e:
        return None, f"tokenize_fail:{type(e).__name__}"
    all_names = {t.string for t in toks if t.type == tokenize.NAME}
    string_text = "\n".join(t.string for t in toks if t.type == tokenize.STRING)
    kw_labels = {k.arg for k in ast.walk(tree) if isinstance(k, ast.keyword) and k.arg}
    params = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in (n.args.posonlyargs + n.args.args + n.args.kwonlyargs):
                params.add(a.arg)
    repls, ren = [], {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = fn.end_lineno or fn.lineno
        for nm in _collect_scope_names(fn):
            if nm in params or nm == "_" or nm.startswith("__") or len(ren) >= 8:
                continue
            if nm in kw_labels or re.search(r"\b" + re.escape(nm) + r"\b", string_text):
                continue
            occ = [t for t in toks if t.type == tokenize.NAME and t.string == nm]
            if len(occ) < 2 or not all(fn.lineno <= t.start[0] <= end for t in occ):
                continue
            new = f"{nm}_r{len(ren)}"
            if new in all_names:
                continue
            ren[nm] = new
            for t in occ:
                repls.append((t.start[0], t.start[1], t.end[1], new))
    if not repls:
        return None, "no_safe_candidate"
    new_code = splice_lines(code, repls)
    try:
        new_tree = ast.parse(new_code)
    except Exception as e:
        return None, f"reparse_fail:{type(e).__name__}"
    if interface_sig(new_code) != interface_sig(code):
        return None, "interface_changed"
    if not _ast_equal_mod_rename(tree, new_tree, ren):
        return None, "ast_changed"
    return new_code, "ok"


TRANSFORMS = {"comment_strip": (transform_comment, 0.2),
              "whitespace": (transform_whitespace, 0.3),
              "rename": (transform_rename, 0.4)}


def generate_transforms(design):
    rows = design["rows"]
    acc_records, accepted = [], []
    for i, r in enumerate(rows):
        code = r["code"]
        for tcode, (tname, (fn, p)) in enumerate(TRANSFORMS.items()):
            rng = np.random.default_rng([20261009, i, tcode])
            rec = {"row": i, "model_id": r["model_id"], "task_id": r["task_id"],
                   "type": tname, "orig_sha256": r["solution_sha256"],
                   "ast_orig": ast_hash(code), "iface_orig": interface_sig(code),
                   "new_sha256": None, "ast_new": None, "iface_new": None}
            if code.strip() == "":
                rec.update({"attempted": False, "accepted": False, "reason": "empty_code"})
            elif rng.random() >= p:
                rec.update({"attempted": False, "accepted": False, "reason": "not_sampled"})
            else:
                rec["attempted"] = True
                new, reason = fn(code)
                if new is None:
                    rec.update({"accepted": False, "reason": reason})
                else:
                    rec.update({"accepted": True, "reason": "ok", "new_sha256": sha(new),
                                "ast_new": ast_hash(new), "iface_new": interface_sig(new)})
                    accepted.append((i, tname, new))
            acc_records.append(rec)
    return acc_records, accepted


def build_trans_features(design, accepted):
    from prepare_code_conditioned_pilot import static_view, library_roots  # noqa
    rows = design["rows"]
    taskmeta = {}
    with TASKMETA.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            taskmeta[d["task_id"]] = d
    codes = [c for _, _, c in accepted]
    t0 = time.time()
    hy = s1.encode_codet5(s1.MODEL_SMALL, codes, device_batch=16)
    log(f"[c3] transformed embeddings {hy.shape} in {time.time()-t0:.0f}s")
    psi = np.zeros((len(codes), 9), dtype=np.float64)
    for j, (i, _, c) in enumerate(accepted):
        t = taskmeta[rows[i]["task_id"]]
        v = static_view(c, t["entry_point"], library_roots(t["libs"]))
        psi[j] = [v[k] for k in PSI_FIELDS]
    return hy, psi


def build_view(design, idx, hy_override=None, psi_override=None):
    b = design["bundle"]
    ht = design["emb_task"][b["task_idx"][idx]]
    hy = b["hy_small"][idx] if hy_override is None else hy_override
    psi = b["psi"][idx] if psi_override is None else psi_override
    return np.hstack([ht, hy, ht * hy, np.abs(ht - hy), psi])


def train_inv_model(Xtr, ytr, Xj, yj, oj, seed, device, lam=LAMBDA_INV, epochs=EPOCHS):
    """Xj/yj/oj: transformed train rows (features, labels, local orig index)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    X = torch.tensor(Xtr.astype(np.float32), device=device)
    y = torch.tensor(ytr.astype(np.float32), device=device)
    j_by_orig = {}
    for j, o in enumerate(oj):
        j_by_orig.setdefault(int(o), []).append(j)
    j_by_orig_arr = {k: np.array(v) for k, v in j_by_orig.items()}
    rng = np.random.default_rng(seed)
    bs = 1024
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i in range(0, len(perm), bs):
            sel = perm[i:i + bs]
            posmap = {int(l): k for k, l in enumerate(sel)}
            logits = net(X[sel]).squeeze(-1)
            p_orig = torch.sigmoid(logits)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[sel])
            jl = [j for l in sel for j in j_by_orig_arr.get(int(l), [])]
            if jl:
                jl = np.array(jl)
                Xj_t = torch.tensor(Xj[jl].astype(np.float32), device=device)
                yj_t = torch.tensor(yj[jl].astype(np.float32), device=device)
                p_t = torch.sigmoid(net(Xj_t).squeeze(-1))
                loss = loss + torch.nn.functional.binary_cross_entropy(p_t, yj_t)
                o_pos = [posmap[int(o)] for o in oj[jl]]
                p_o = p_orig[torch.tensor(o_pos, device=device)]
                m = 0.5 * (p_o + p_t)
                eps = 1e-6
                kl = lambda a, b: (a * torch.log((a + eps) / (b + eps))
                                   + (1 - a) * torch.log((1 - a + eps) / (1 - b + eps)))
                js = 0.5 * kl(p_o, m) + 0.5 * kl(p_t, m)
                loss = loss + lam * js.mean()
            opt.zero_grad(); loss.backward(); opt.step()

    def score(M):
        with torch.inference_mode():
            return net(torch.tensor(M.astype(np.float32), device=device)).squeeze(-1).sigmoid().cpu().numpy()

    return score


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "local").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    design = cc.load_design()
    rows = design["rows"]
    b = design["bundle"]
    task_split = {r["task_id"]: r["split"] for r in rows}
    dev_tasks = sorted(t for t in design["tasks_all"] if task_split[t] == "dev")
    tpos_of_task = {t: i for i, t in enumerate(dev_tasks)}
    n_tasks = len(dev_tasks)
    device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")

    acc_records, accepted = generate_transforms(design)
    with (OUT / "transform_acceptance.jsonl").open("w", encoding="utf-8") as f:
        for rec in acc_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats = {}
    for tname in TRANSFORMS:
        rr = [r for r in acc_records if r["type"] == tname]
        att = [r for r in rr if r["attempted"]]
        okc = [r for r in att if r["accepted"]]
        reasons = {}
        for r in att:
            if not r["accepted"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        stats[tname] = {
            "total": len(rr), "attempted": len(att), "accepted": len(okc),
            "rejected": len(att) - len(okc),
            "reject_rate_of_attempted": (len(att) - len(okc)) / max(1, len(att)),
            "ast_broken_accepted": sum(1 for r in okc
                                       if tname != "rename" and r["ast_new"] != r["ast_orig"]),
            "iface_changed_accepted": sum(1 for r in okc if r["iface_new"] != r["iface_orig"]),
            "reject_reasons": reasons}
    log("[c3] transforms: " + json.dumps({k: (v["attempted"], v["accepted"])
                                          for k, v in stats.items()}))
    hy_t, psi_t = build_trans_features(design, accepted)
    acc_pos = {(i, t): j for j, (i, t, _) in enumerate(accepted)}

    labels, tposs, scores_orig = {}, {}, {}
    trans_eval = {t: {} for t in TRANSFORMS}   # type -> fid -> (row positions in eval, p_t)
    for fold in design["folds"]:
        fid = fold["heldout_generator_member"]
        tf0 = time.time()
        fit_mask, ev_mask, pos_mask, _ = cc.fold_setup(design, fold, split="dev", inner=False)
        y_fit = cc.fold_series_target(design, fold)[fit_mask]
        fit_rows = np.where(fit_mask)[0]
        ev_rows = np.where(ev_mask)[0]
        tp = np.array([tpos_of_task[rows[i]["task_id"]] for i in ev_rows])
        order = np.argsort(tp, kind="mergesort")
        ev_rows, tp = ev_rows[order], tp[order]
        y_ev = pos_mask[ev_rows].astype(int)
        labels[fid], tposs[fid] = y_ev, tp

        Xtr_raw = build_view(design, fit_rows)
        Xev_raw = build_view(design, ev_rows)
        mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0)
        sd[sd < 1e-8] = 1.0
        Xtr = ((Xtr_raw - mu) / sd).astype(np.float32)
        Xev = ((Xev_raw - mu) / sd).astype(np.float32)

        jsel_all, osel_all = [], []
        for l, i in enumerate(fit_rows):
            for tname in TRANSFORMS:
                if (i, tname) in acc_pos:
                    jsel_all.append(acc_pos[(i, tname)])
                    osel_all.append(l)
        jsel_all = np.array(jsel_all)
        osel_all = np.array(osel_all)
        ht_j = design["emb_task"][b["task_idx"][fit_rows][osel_all]]
        Xj_raw = np.hstack([ht_j, hy_t[jsel_all], ht_j * hy_t[jsel_all],
                            np.abs(ht_j - hy_t[jsel_all]), psi_t[jsel_all]])
        Xj = ((Xj_raw - mu) / sd).astype(np.float32)
        yj = cc.fold_series_target(design, fold)[fit_rows][osel_all]

        scorers = [train_inv_model(Xtr, y_fit, Xj, yj, osel_all, s, device) for s in SEEDS]
        scores_orig[fid] = np.mean([sc(Xev) for sc in scorers], axis=0)
        for tname in TRANSFORMS:
            sel = np.array([k for k, i in enumerate(ev_rows) if (i, tname) in acc_pos])
            if len(sel) == 0:
                trans_eval[tname][fid] = (np.array([], dtype=int), np.array([]))
                continue
            jsel = np.array([acc_pos[(ev_rows[k], tname)] for k in sel])
            ht_e = design["emb_task"][b["task_idx"][ev_rows[sel]]]
            Xe_raw = np.hstack([ht_e, hy_t[jsel], ht_e * hy_t[jsel],
                                np.abs(ht_e - hy_t[jsel]), psi_t[jsel]])
            Xe = ((Xe_raw - mu) / sd).astype(np.float32)
            trans_eval[tname][fid] = (sel, np.mean([sc(Xe) for sc in scorers], axis=0))
        save_kw = dict(ev_rows=ev_rows, y=y_ev, taskpos=tp, s_orig=scores_orig[fid])
        for tname in TRANSFORMS:
            sel_t, pt_t = trans_eval[tname][fid]
            save_kw[f"sel_{tname}"] = sel_t.astype(np.int32)
            save_kw[f"pt_{tname}"] = pt_t
        np.savez_compressed(OUT / "local" / f"scores_fold{fid.replace('--','_')}.npz", **save_kw)
        log(f"[c3] fold {fid[:40]:40s} done ({time.time()-tf0:.0f}s)")

    c0 = {}
    for fold in design["folds"]:
        fid = fold["heldout_generator_member"]
        c0[fid] = np.load(C0_DIR / "local" / f"scores_dev_fold{fid.replace('--','_')}.npz")["fused"]

    def agg(scores):
        row_mean = float(np.mean([cc.auroc(labels[f], scores[f]) for f in labels]))
        tm_mean = float(np.mean([cc.task_macro_auroc(labels[f], scores[f], tposs[f]) for f in labels]))
        boot = cc.bootstrap_metrics(scores, labels, tposs, n_tasks=n_tasks)
        return {"row_level_mean": row_mean, "task_macro_mean": tm_mean,
                "ci95": {"row_level": cc.ci(boot["row_level"]), "task_macro": cc.ci(boot["task_macro"])}}

    res = {"schema": "cc_c3_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "report_first_line": "C3 一致性训练（comment/whitespace/rename p=.2/.3/.4；11 折；对照 C0；train/dev only）",
           "lambda_inv": LAMBDA_INV, "transform_stats": stats,
           "c3": agg(scores_orig), "c0": agg(c0)}
    d = cc.delta_bootstrap(scores_orig, c0, labels, tposs, n_tasks=n_tasks)
    res["c3_vs_c0"] = {"row_level": {"mean": float(np.mean(d["row_level"])), "ci95": cc.ci(d["row_level"])},
                       "task_macro": {"mean": float(np.mean(d["task_macro"])), "ci95": cc.ci(d["task_macro"])}}
    res["task_macro_positive_folds_vs_c0"] = int(sum(
        cc.task_macro_auroc(labels[f], scores_orig[f], tposs[f])
        > cc.task_macro_auroc(labels[f], c0[f], tposs[f]) for f in labels))

    res["transformed_eval"] = {}
    for tname in TRANSFORMS:
        aucs, dps, agree, nrows = [], [], [], 0
        ys_all, ps_all = [], []
        for fid in labels:
            sel, pt = trans_eval[tname][fid]
            if len(sel) == 0:
                continue
            y_t = labels[fid][sel]
            aucs.append(cc.auroc(y_t, pt))
            ys_all.append(y_t)
            ps_all.append(pt)
            po = scores_orig[fid][sel]
            dp = np.abs(po - pt)
            dps.append(float(np.mean(dp)))
            agree.append(float(np.mean((po > 0.5) == (pt > 0.5))))
            nrows += len(sel)
        aucs_clean = [a for a in aucs if not np.isnan(a)]
        pooled = None
        if ys_all:
            ya = np.concatenate(ys_all)
            pa = np.concatenate(ps_all)
            if 0 < ya.sum() < len(ya):
                pooled = cc.auroc(ya, pa)
        res["transformed_eval"][tname] = {
            "n_eval_rows": int(nrows),
            "n_folds_with_auroc": len(aucs_clean),
            "n_folds_nan": len(aucs) - len(aucs_clean),
            "auroc_on_transformed_mean_over_folds": float(np.mean(aucs_clean)) if aucs_clean else None,
            "auroc_on_transformed_pooled": pooled,
            "mean_abs_delta_p": float(np.mean(dps)) if dps else None,
            "sign_agreement": float(np.mean(agree)) if agree else None}
    res["transformed_eval"]["c3_orig_row_level_auroc_for_reference"] = res["c3"]["row_level_mean"]

    (OUT / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "hypothesis.json").write_text(json.dumps({
        "schema": "cc_c3_hypothesis_v1",
        "hypothesis": ("语义安全增强的一致性训练（λinv=0.1，原/变换混合）在 task-macro 上相对 C0 出现增量，"
                       "且变换后归因保持（配对一致率高、接受样本 AST/接口零破坏）。"),
        "claim_limit": "增强仅为鲁棒性/反捷径工具；拒绝率按类型报告；不重验论文命题；dev 仅开发评测。",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({
        "schema": "cc_c3_config_v1",
        "transforms": {k: {"p": v[1]} for k, v in TRANSFORMS.items()},
        "loss": "mean_BCE({orig}+{trans}) + lambda_inv*mean_JS(p_orig,p_trans)",
        "lambda_inv": LAMBDA_INV, "heads": "同 C1 full-mlp（3 seeds, 20ep）",
        "bootstrap": {"n": 500, "seed": 20261009},
        "switches": {"train_dev_only": True, "test_read": False, "generation": False,
                     "weights_downloaded": False, "code_execution": False},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "data_role_matrix.json").write_text(json.dumps({
        "rows": "records_train_dev.jsonl（instruct）",
        "transforms": "comment_strip/whitespace/rename，独立 Bernoulli p=.2/.3/.4，固定 seed",
        "fit": "折内 train 原样本 + 已接受变换样本（混合）；JS 配对一致性",
        "eval": "dev：heldout（正）+ 其它系列（负）；原分数为主，变换分数为配对诊断",
        "never_used": ["test", "canonical_solution", "执行结果"],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# C3 一致性训练（2026-10-09）", "",
         f"性质：增强一致性（λinv={LAMBDA_INV}）；非论文复现；train/dev only。", "",
         f"- C3 row-level = {res['c3']['row_level_mean']:.4f}；task-macro = {res['c3']['task_macro_mean']:.4f}",
         f"- C0 row-level = {res['c0']['row_level_mean']:.4f}；task-macro = {res['c0']['task_macro_mean']:.4f}",
         f"- C3−C0（task-macro）: {res['c3_vs_c0']['task_macro']['mean']:+.4f} "
         f"[{res['c3_vs_c0']['task_macro']['ci95'][0]:+.4f},{res['c3_vs_c0']['task_macro']['ci95'][1]:+.4f}]，"
         f"正折 {res['task_macro_positive_folds_vs_c0']}/11", "",
         "## 变换统计", "", "| type | attempted | accepted | reject_rate | ast_broken | iface_changed |",
         "|---|---|---|---|---|---|"]
    for k, v in stats.items():
        L.append(f"| {k} | {v['attempted']} | {v['accepted']} | {v['reject_rate_of_attempted']:.3f} | "
                 f"{v['ast_broken_accepted']} | {v['iface_changed_accepted']} |")
    L += ["", "## 变换后归因保持（transformed eval）", "",
          "| type | n | AUROC_folds | AUROC_pooled | mean|Δp| | sign agreement |", "|---|---|---|---|---|---|"]
    for k in TRANSFORMS:
        v = res["transformed_eval"][k]
        af = v["auroc_on_transformed_mean_over_folds"]
        ap = v["auroc_on_transformed_pooled"]
        L.append(f"| {k} | {v['n_eval_rows']} | "
                 f"{af:.4f} | {ap:.4f} | "
                 f"{v['mean_abs_delta_p']:.4f} | {v['sign_agreement']:.4f} |")
    L += [f"（C3 原码 row-level 参考 = {res['c3']['row_level_mean']:.4f}；|Δp|=|p_orig−p_trans|）", "",
          "> 拒绝原因见 transform_acceptance.jsonl；接受样本 AST/接口零破坏。"]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "OMP_NUM_THREADS=8 /root/miniconda3/envs/udet/bin/python -W ignore scripts/cc_c3_invariance.py\n",
        encoding="utf-8")
    (OUT / "logs" / "c3_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    cc.git_meta(OUT)
    cc.write_sha256sums(OUT)
    log(f"[c3] done in {time.time()-t0:.1f}s tm={res['c3']['task_macro_mean']:.4f} "
        f"vs_c0={res['c3_vs_c0']['task_macro']['mean']:+.4f}")


if __name__ == "__main__":
    main()
