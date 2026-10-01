#!/usr/bin/env python
"""E36：DroidCollection 子集上的 H2 generator-held-out 候选实验。

依据 `docx/d-det_H2_DroidCollection_DeepSeek执行指导_2026-10-01.md`：
  · 阶段 A（--audit-only，不加载 encoder）：流式读 core.jsonl → audit.json：计数、
    重复 source_row_sha1、每折 train/held-out generator 非空且不相交等断言；
  · 阶段 B/C（--smoke）：每 (split,label,generator) 取前 6 行；特征冒烟（768 维、
    无 NaN/Inf、顺序/hash 对齐、重复编码一致）+ 协议冒烟（仅 fold_0；B0 收敛；F0/F1
    初始 logits 与 B0 逐位一致；F1 有跨 generator anchor；同 schedule；产物齐全）；
  · 阶段 D（默认）：完整两折。
  · 编码口径与 E28/E35 相同：本地 CodeT5 tokenizer（add_special_tokens=False；
    >max_length → 头 75%+尾 25%，默认 1024→768+256；<8 跳过）、v0.4.1 冻结编码器 +
    SetPool 头 + m_raw 均值池化（bf16 autocast）；特征缓存保存逐行 source_row_sha1
    并在训练前与 JSONL 复核。
  · 主实验：严格读 fold_plan.json；B0 加权多分类 LR（C=.1、tol=1e-6、max_iter=10000、
    w=N/(K|G_f|N_g)）拟合后**冻结**；残差 r=V·GELU(Uz+b1)+b2、ℓ=ℓ0+γ·W_R·r、
    γ=0.1·tanh(η)，阶段 B 修正门控初始化（W_R=0、η0=.5、U Xavier、V std=1e-3）；
    F0=L_F；F1=L_F+0.1·L_cross（同家族不同 generator 正对、τ=.1 只除一次、无正对
    anchor 跳过、分母含全批、Human/异族/同 generator 不入 L_cross）；批 7 族×18、
    2 epoch、AdamW 1e-3/wd 1e-4/clip 1；F0/F1 共享主轴/初始化/种子/schedule/标准化。
  · 指标只在 held-out test 上：BA_F/BA_G、逐族/逐 held-out generator 召回、语言/
    Source/Generation_Mode/代码长度分桶、检测 AUROC（次要，train 侧训练）；随机切分
    （同本控制）仅作迁移损失参照。预注册出口见 config.json exit_rule。
  · 不读 diagnostic_hybrid_adversarial.jsonl 参与训练/评分（仅审计计数）；test 只做
    最终评估，不用于任何选择。

用法：
  python scripts/h2_droid_e36.py --audit-only
  python scripts/h2_droid_e36.py --smoke
  python scripts/h2_droid_e36.py
运行目录产物：artifacts/h2_droid_e36/{audit,config,manifest,metrics}.json +
predictions.npz + solver.log（report.md 由报告复制）；特征缓存在 runs/h2_droid_e36/
（不入库）；默认拒绝覆盖。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flagship_e28_probe import fit_scaler, apply_scaler  # noqa: E402
from flagship_e35 import (GatedResidual, cross_pos, supcon_cross_from,  # noqa: E402
                          fam_eval)

C_FIXED = 0.1
STD_FLOOR = 1e-2
MAX_ITER = 10000
TOL = 1e-6
PER_FAM = 18
EPOCHS = 2
TAU = 0.1
LAM_C = 0.1
DELTA_THRESH = 0.01
ANCHOR_MIN = 0.15
FAM7 = ["qwen", "meta-llama", "microsoft", "01-ai", "codellama",
        "deepseek-ai", "ibm-granite"]
LEN_BUCKETS = [(0, 128), (129, 256), (257, 512), (513, 1024), (1025, 10 ** 9)]
SMOKE_PER_GROUP = 6
FIELDS = ("Code", "Generator", "Label", "Model_Family", "split_source",
          "source_row_sha1", "Language", "Source", "Generation_Mode")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def stream_rows(path: Path, tolerant: bool = False):
    """逐行流式读取 JSONL（禁止一次性 json.load 全文件）。

    tolerant=True 时字段缺失置空（诊断文件无 split_source/source_row_sha1）。
    """
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            g = (lambda k, d="": r.get(k, d)) if tolerant else (lambda k, d="": r[k])
            yield {"i": i, "code": g("Code"), "gen": str(g("Generator")),
                   "label": str(g("Label")), "fam": str(g("Model_Family")),
                   "split": str(g("split_source")), "sha1": str(g("source_row_sha1")),
                   "lang": str(g("Language") or ""), "src": str(g("Source") or ""),
                   "mode": str(g("Generation_Mode") or ""),
                   "char_len": len(str(g("Code")))}


def verify_sha256sums(data_dir: Path, say) -> dict:
    sums = {}
    for line in (data_dir / "SHA256SUMS.txt").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(None, 1)
        sums[name.strip()] = digest
    out = {}
    for name, digest in sums.items():
        real = sha256_file(data_dir / name)
        out[name] = {"sha256": real, "ok": real == digest}
    assert all(v["ok"] for v in out.values()), "SHA256SUMS 校验失败"
    say(f"[e36] SHA256SUMS 校验通过（{len(out)} 文件）")
    return out


def run_audit(rows, fold_plan, summary, data_dir, sums, diag_info, say, strict):
    n = len(rows)
    labels = Counter(r["label"] for r in rows)
    assert set(labels) <= {"MACHINE_GENERATED", "HUMAN_GENERATED"}, \
        f"core 出现非预期 Label：{labels}"
    machine = [r for r in rows if r["label"] == "MACHINE_GENERATED"]
    human = [r for r in rows if r["label"] == "HUMAN_GENERATED"]
    fams_obs = sorted({r["fam"] for r in machine})
    assert fams_obs == sorted(FAM7), f"family 集合不符：{fams_obs}"

    def sp_counts(rs):
        return dict(Counter(r["split"] for r in rs))
    by_gen = defaultdict(lambda: defaultdict(int))
    for r in machine:
        by_gen[r["gen"]][r["split"]] += 1

    # 重复 hash
    def dup_stats(rs):
        s = [r["sha1"] for r in rs]
        return {"n": len(s), "unique": len(set(s)), "dup_within": len(s) - len(set(s))}
    sha_sets = {sp: {"machine": set(), "human": set()} for sp in
                ("train", "dev", "test")}
    for r in rows:
        key = "machine" if r["label"] == "MACHINE_GENERATED" else "human"
        sha_sets[r["split"]][key].add(r["sha1"])
    cross = {}
    for key in ("machine", "human"):
        inter = {}
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
            inter[f"{a}_{b}"] = len(sha_sets[a][key] & sha_sets[b][key])
        cross[key] = inter
    if strict:
        assert all(v == 0 for v in cross["machine"].values()), \
            f"machine train/dev/test sha1 跨 split 重复：{cross['machine']}"
        assert all(v == 0 for v in cross["human"].values()), \
            f"human train/dev/test sha1 跨 split 重复：{cross['human']}"

    # 每折断言
    folds_out = {}
    for fk, spec in fold_plan["folds"].items():
        assert set(spec.keys()) == set(FAM7), f"{fk} family 集合不符"
        fo = {"families": {}, "asserts": {}}
        for f in FAM7:
            trg = list(spec[f]["train_generators"]); hog = list(spec[f]["heldout_generators"])
            obs = sorted({r["gen"] for r in machine if r["fam"] == f})
            assert trg and hog and not (set(trg) & set(hog)), f"{fk}/{f} generator 集合异常"
            if strict:
                assert set(obs) <= set(trg) | set(hog), f"{fk}/{f} 观测 generator 越界"
            cnt = {sp: 0 for sp in ("train", "dev", "test")}
            for r in machine:
                if r["fam"] != f:
                    continue
                if r["split"] == "train" and r["gen"] in trg:
                    cnt["train"] += 1
                elif r["split"] == "dev" and r["gen"] in trg:
                    cnt["dev"] += 1
                elif r["split"] == "test" and r["gen"] in hog:
                    cnt["test"] += 1
            if strict:
                assert cnt["train"] > 0 and cnt["dev"] > 0 and cnt["test"] > 0, \
                    f"{fk}/{f} 折内样本为空：{cnt}"
                for g in hog:
                    assert by_gen[g]["test"] > 0, f"{fk}/{f} held-out {g} 无 test 样本"
            fo["families"][f] = {
                "train_generators": trg, "heldout_generators": hog,
                "observed_generators": obs,
                "n_machine": {"train": cnt["train"], "dev": cnt["dev"], "test": cnt["test"]},
                "per_heldout_test": {g: by_gen[g]["test"] for g in hog},
                "per_train_gen_counts": {g: sum(by_gen[g][sp] for sp in ("train", "dev"))
                                         for g in trg}}
        folds_out[fk] = fo
    humans = {sp: sum(1 for r in human if r["split"] == sp) for sp in ("train", "dev", "test")}
    mus = {sp: sum(1 for r in machine if r["split"] == sp) for sp in ("train", "dev", "test")}

    audit = {
        "data_dir": str(data_dir), "files": sums,
        "source_revision": summary.get("source_revision"),
        "totals": {"rows": n, "machine": len(machine), "human": len(human),
                   "by_split": sp_counts(rows),
                   "machine_by_split": mus, "human_by_split": humans},
        "labels": dict(labels),
        "families": {f: {"n": sum(1 for r in machine if r["fam"] == f),
                         "generators": sorted({r["gen"] for r in machine if r["fam"] == f})}
                     for f in FAM7},
        "languages": dict(Counter(r["lang"] for r in rows).most_common()),
        "generation_modes": dict(Counter(r["mode"] for r in rows).most_common()),
        "sources_top": dict(Counter(r["src"] for r in rows).most_common(20)),
        "generator_counts": {g: dict(by_gen[g]) for g in sorted(by_gen)},
        "dup_sha1": {"machine": dup_stats(machine), "human": dup_stats(human),
                     "cross_split": cross},
        "folds": folds_out,
        "diagnostic": diag_info,
        "notes": ["DroidCollection 无公开 task_id/prompt_id：仅支持"
                  "'跨 generator 家族归因候选证据'表述",
                  "diagnostic_hybrid_adversarial.jsonl 仅计数，未参与训练/评分",
                  "test 只做最终评估（held-out generator），不用于任何选择"],
    }
    return audit


def build_features(rows, args, runs_dir: Path, say):
    import torch
    import yaml
    from transformers import AutoTokenizer
    from encoders import build_encoder
    from models import build_model
    from flagship_round1 import Doc, SetPool
    from flagship_e28_probe import export_pass

    tok = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    docs, keep_rows, skipped = [], [], []
    CH = 2048
    t_tok = time.time()
    for s in range(0, len(rows), CH):
        chunk = rows[s:s + CH]
        enc = tok([r["code"] for r in chunk], add_special_tokens=False)["input_ids"]
        for r, ids in zip(chunk, enc):
            if len(ids) > args.max_length:
                head = int(args.max_length * 0.75)
                ids = ids[:head] + ids[-(args.max_length - head):]
            if len(ids) < 8:
                skipped.append(r["i"])
                continue
            d = Doc(np.asarray(ids, dtype=np.int64),
                    int(r["label"] == "MACHINE_GENERATED"), r["fam"], r["lang"])
            d.split = r["split"]; d.gen = r["gen"]; d.row = r["i"]
            docs.append(d); keep_rows.append(r)
    tl = np.array([len(d.ids) for d in docs])
    say(f"[e36] tokenize {time.time()-t_tok:.1f}s：保留 {len(docs)}，跳过(<8) "
        f"{len(skipped)}；tok 长度 p50/p90/p99/max={np.percentile(tl,50):.0f}/"
        f"{np.percentile(tl,90):.0f}/{np.percentile(tl,99):.0f}/{tl.max()}")

    ms = torch.load(ROOT / "runs/flagship_e27/model_state.pt", map_location="cpu",
                    weights_only=False)
    cfg = yaml.safe_load((ROOT / "configs/ddet_v041.yaml").read_text())
    ec = dict(cfg["encoder"]); ec["path"] = str(ROOT / ec["path"])
    dual = build_model("dual", encoder=build_encoder(**ec), dim=768,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    dual.load_state_dict(torch.load(ROOT / "runs/v0.4.1_covreg/last.pt",
                                    map_location="cpu", weights_only=False)["state"],
                         strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = dual.encoder.to(device).eval()
    head = SetPool(768, 512).to(device)
    head.load_state_dict(torch.load(ROOT / "runs/flagship_r1/head.pt",
                                    map_location=device)["head"])
    head.eval()
    wq = torch.as_tensor(ms["wq"], device=device)
    bq = torch.as_tensor(ms["bq"], device=device)
    m_tT = torch.as_tensor(ms["m_t"], dtype=torch.float32, device=device)
    s_tT = torch.as_tensor(ms["s_t"], dtype=torch.float32, device=device)

    t0 = time.time()
    raw, mu, st = export_pass(enc, head, docs, args.batch_size, device, wq, bq,
                              m_tT, s_tT)
    dt = time.time() - t0
    say(f"[e36] 编码 {len(docs)} 行 {dt:.1f}s（{len(docs)/max(dt,1e-9):.1f} 行/s，"
        f"bs={args.batch_size}，device={device}）")
    assert raw.shape == (len(docs), 768), f"维度异常 {raw.shape}"
    assert np.isfinite(raw).all(), "特征含 NaN/Inf"
    sub = docs[:16]
    raw2a, _, _ = export_pass(enc, head, sub, args.batch_size, device, wq, bq,
                              m_tT, s_tT)
    raw2b, _, _ = export_pass(enc, head, sub, args.batch_size, device, wq, bq,
                              m_tT, s_tT)
    rep_ok = bool(np.array_equal(raw2a, raw2b))
    cross_gap = float(np.abs(raw2a - raw[:16]).max())
    say(f"[e36] 重复编码一致性（同批序×2）：{'逐位一致' if rep_ok else '存在差异'}；"
        f"跨批组成差异 max|Δ|={cross_gap:.2e}（bf16 批核效应，记录不阻塞）")
    assert rep_ok, "同一输入（同批序）重复编码不一致"

    runs_dir.mkdir(parents=True, exist_ok=True)
    feats = {
        "raw": raw,
        "row_i": np.array([r["i"] for r in keep_rows], np.int64),
        "sha1": np.array([r["sha1"] for r in keep_rows]),
        "split": np.array([r["split"] for r in keep_rows]),
        "label": np.array([r["label"] == "MACHINE_GENERATED" for r in keep_rows]),
        "family": np.array([r["fam"] for r in keep_rows]),
        "generator": np.array([r["gen"] for r in keep_rows]),
        "language": np.array([r["lang"] for r in keep_rows]),
        "mode": np.array([r["mode"] for r in keep_rows]),
        "source": np.array([r["src"] for r in keep_rows]),
        "char_len": np.array([r["char_len"] for r in keep_rows], np.int32),
        "tok_len": np.array([len(d.ids) for d in docs], np.int32),
        "skipped_row_i": np.array(skipped, np.int64),
    }
    np.savez_compressed(runs_dir / "features.npz", **feats)
    with open(runs_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in keep_rows:
            f.write(json.dumps({"i": r["i"], "sha1": r["sha1"], "split": r["split"],
                                "label": r["label"], "family": r["fam"],
                                "generator": r["gen"], "language": r["lang"],
                                "mode": r["mode"], "source": r["src"],
                                "char_len": r["char_len"]}, ensure_ascii=False) + "\n")
    return feats


def load_features(runs_dir: Path, rows):
    path = runs_dir / "features.npz"
    z = dict(np.load(path, allow_pickle=True))
    # 与 JSONL 复核：row_i → sha1 逐位一致
    by_i = {r["i"]: r for r in rows}
    ri = z["row_i"]; sh = z["sha1"]
    for j in range(len(ri)):
        r = by_i.get(int(ri[j]))
        assert r is not None and r["sha1"] == sh[j], f"特征缓存与 JSONL 不符 @ {j}"
    assert len(ri) + len(z["skipped_row_i"]) == len(rows), "缓存行数 + 跳过行数 ≠ 总行数"
    return z


def cat_map_build(feats, fold_spec):
    """机器行的 (family, generator) → train/heldout/none 分类。"""
    m = {}
    for f in FAM7:
        for g in fold_spec[f]["train_generators"]:
            m[(f, str(g))] = "train"
        for g in fold_spec[f]["heldout_generators"]:
            m[(f, str(g))] = "heldout"
    return np.array([m.get((fa, ge), "none") if is_m else "human"
                     for fa, ge, is_m in zip(feats["family"], feats["generator"],
                                             feats["label"])])


def fit_lr(X, y, w, tag, say):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        clf = LogisticRegression(max_iter=MAX_ITER, tol=TOL, C=C_FIXED).fit(
            X, y, sample_weight=w)
    conv = not any(issubclass(x.category, ConvergenceWarning) for x in wl)
    say(f"[e36] {tag}: 收敛={conv} n_iter={int(np.max(clf.n_iter_))}")
    return clf, conv


def bucket_eval(pred, y7, groups):
    out = {}
    for gval in sorted(set(groups.tolist())):
        m = groups == gval
        if int(m.sum()) == 0:
            continue
        recs = []
        for k in range(len(FAM7)):
            mk = m & (y7 == k)
            if int(mk.sum()) > 0:
                recs.append(float((pred[mk] == k).mean()))
        out[str(gval)] = {"n": int(m.sum()), "BA_F": round(float(np.mean(recs)), 4)}
    return out


def run_fold(fold_key, fold_spec, feats, args, say):
    import torch
    import torch.nn.functional as F

    n_all = len(feats["label"])
    cat = cat_map_build(feats, fold_spec)
    is_m = feats["label"]
    split = feats["split"]
    tr = np.where(is_m & (split == "train") & (cat == "train"))[0]
    va = np.where(is_m & (split == "dev") & (cat == "train"))[0]
    te = np.where(is_m & (split == "test") & (cat == "heldout"))[0]
    hum_tr = np.where(~is_m & (split == "train"))[0]
    hum_va = np.where(~is_m & (split == "dev"))[0]
    hum_te = np.where(~is_m & (split == "test"))[0]
    for tag, arr in (("train", tr), ("dev", va), ("test", te), ("hum_tr", hum_tr),
                     ("hum_dev", hum_va), ("hum_test", hum_te)):
        assert len(arr) > 0, f"{fold_key} {tag} 为空"
    fam_idx_all = np.array([FAM7.index(f) if f in FAM7 else -1
                            for f in feats["family"]], np.int8)
    y7_tr = fam_idx_all[tr].astype(np.int64)
    y7_te = fam_idx_all[te].astype(np.int64)
    gen_tr = feats["generator"][tr]
    gen_te = feats["generator"][te]
    fam7_te = feats["family"][te]
    say(f"[e36] {fold_key}: train {len(tr)} / dev {len(va)} / test {len(te)}"
        f"（human {len(hum_tr)}/{len(hum_va)}/{len(hum_te)}）")

    # ---- 标准化 + B0 主轴（冻结） ----
    m_f, s_f = fit_scaler(feats["raw"][tr])
    Z_tr = apply_scaler(feats["raw"][tr], m_f, s_f)
    Z_te = apply_scaler(feats["raw"][te], m_f, s_f)
    fam_tr = feats["family"][tr]
    fam_gens = {f: sorted({str(gen_tr[j]) for j in range(len(tr))
                           if fam_tr[j] == f}) for f in FAM7}
    gc = {g: int((gen_tr == g).sum()) for f in FAM7 for g in fam_gens[f]}
    w_sample = np.array([len(tr) / (len(FAM7) * len(fam_gens[fn]) * gc[g])
                         for fn, g in zip(fam_tr, gen_tr)])
    b0, b0_conv = fit_lr(Z_tr, y7_tr, w_sample, f"{fold_key} B0 主轴（加权）", say)
    W0, b0v = np.asarray(b0.coef_), np.asarray(b0.intercept_)

    # ---- 检测头（次要；C=.1 固定，train 侧） ----
    tr_all = np.concatenate([tr, hum_tr])
    te_all = np.concatenate([te, hum_te])
    va_all = np.concatenate([va, hum_va])
    m_d, s_d = fit_scaler(feats["raw"][tr_all])
    clf_d, _ = fit_lr(apply_scaler(feats["raw"][tr_all], m_d, s_d),
                      feats["label"][tr_all].astype(np.int64), None,
                      f"{fold_key} 检测头", say)
    p_d_te = clf_d.predict_proba(apply_scaler(feats["raw"][te_all], m_d, s_d))[:, 1]
    p_d_va = clf_d.predict_proba(apply_scaler(feats["raw"][va_all], m_d, s_d))[:, 1]
    det_te = float(roc_auc_score(feats["label"][te_all], p_d_te))
    det_va = float(roc_auc_score(feats["label"][va_all], p_d_va))
    det_f1 = float(f1_score(feats["label"][te_all], p_d_te > 0.5, average="macro"))
    say(f"[e36] {fold_key} 检测（次要）：test AUROC {det_te:.4f}（F1@.5 {det_f1:.4f}）"
        f" / dev {det_va:.4f}")

    # ---- 批调度（族→generator→样本；F0/F1 共用） ----
    gen_of = {f: {} for f in FAM7}
    for f in FAM7:
        for g in fam_gens[f]:
            gen_of[f][g] = np.where(gen_tr == g)[0]
    batch_size = PER_FAM * len(FAM7)
    steps = math.ceil(len(tr) / batch_size)
    rng = np.random.RandomState(0)
    schedule, redraw = [], 0
    for _ in range(steps * EPOCHS):
        batch = []
        for f in FAM7:
            while True:
                draws = []
                for _ in range(PER_FAM):
                    g = fam_gens[f][rng.randint(len(fam_gens[f]))]
                    draws.append(int(gen_of[f][g][rng.randint(len(gen_of[f][g]))]))
                if len(set(draws)) >= 2 or len(fam_gens[f]) < 2:
                    break
                redraw += 1
            batch += draws
        schedule.append(np.array(batch))

    # ---- 模型（F0/F1 共享初始状态；阶段 B 修正门控） ----
    torch.manual_seed(0)
    mF0 = GatedResidual(W0, b0v)
    mF1 = GatedResidual(W0, b0v)
    mF1.load_state_dict({k: v.clone() for k, v in mF0.state_dict().items()})
    Z_te_t = torch.as_tensor(Z_te)
    with torch.no_grad():
        l0_te = Z_te_t @ mF0.W0.t() + mF0.b0
        _, lg0, _ = mF0(Z_te_t)
        _, lg1, _ = mF1(Z_te_t)
        init_diff = max(float((lg0 - l0_te).abs().max()),
                        float((lg1 - l0_te).abs().max()))
    assert init_diff == 0.0, f"初始 ℓ≠ℓ0（{init_diff}）"
    say(f"[e36] {fold_key} 初始化审计：F0/F1 初始 logits 与 B0 逐位一致 ✓")

    # ---- 训练 ----
    arms = {"F0": (mF0, torch.optim.AdamW(mF0.parameters(), lr=1e-3, weight_decay=1e-4)),
            "F1": (mF1, torch.optim.AdamW(mF1.parameters(), lr=1e-3, weight_decay=1e-4))}
    Z_tr_t = torch.as_tensor(Z_tr)
    y7_tr_t = torch.as_tensor(y7_tr)
    gcode = {g: c for c, g in enumerate(sorted(set(gen_tr.tolist())))}
    gint_tr_t = torch.as_tensor(np.array([gcode[str(g)] for g in gen_tr], np.int64))
    stats = {k: [] for k in arms}
    gamma_track = {k: [] for k in arms}
    anchor = {"total": 0, "valid": 0,
              "per_family": {f: {"total": 0, "valid": 0} for f in FAM7}}
    for ep in range(EPOCHS):
        for step in range(steps):
            bidx = schedule[ep * steps + step]
            xb = Z_tr_t[bidx]; yb = y7_tr_t[bidx]
            fb = y7_tr_t[bidx]
            gb = gint_tr_t[bidx]
            with torch.no_grad():
                pos, cnt = cross_pos(fb, gb)
                cnt_np = cnt.numpy(); valid_np = cnt_np > 0
                anchor["total"] += len(bidx)
                anchor["valid"] += int(valid_np.sum())
                for k, f in enumerate(FAM7):
                    msk = y7_tr[bidx] == k
                    anchor["per_family"][f]["total"] += int(msk.sum())
                    anchor["per_family"][f]["valid"] += int((msk & valid_np).sum())
            for tag, (model, opt) in arms.items():
                r_, logits, gamma = model(xb)
                L_F = F.cross_entropy(logits, yb)
                L_C = None
                if tag == "F1":
                    L_C, _ = supcon_cross_from(r_, pos, cnt)
                loss = L_F if L_C is None else L_F + LAM_C * L_C
                opt.zero_grad(); loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0))
                opt.step()
                if step in (0, steps - 1):
                    stats[tag].append({
                        "epoch": ep, "step": step,
                        "ce": round(float(L_F.detach()), 4),
                        "lc": None if L_C is None else round(float(L_C.detach()), 4),
                        "grad_norm": round(gnorm, 3),
                        "gamma": round(float(gamma.detach()), 5)})
        for tag, (model, _) in arms.items():
            gamma_track[tag].append(round(float(
                0.1 * torch.tanh(model.eta).detach()), 5))
        say(f"[e36] {fold_key} epoch{ep}：" + " | ".join(
            f"{t} {[s for s in stats[t] if s['epoch'] == ep]}" for t in arms))

    # ---- 评估（held-out test） ----
    res = {"B0": fam_eval(l0_te.argmax(1).numpy(), y7_te, FAM7, list(gen_te))}
    preds = {"B0": l0_te.argmax(1).numpy().astype(np.int8)}
    engage, drift = {}, {}
    with torch.no_grad():
        for tag, (model, _) in arms.items():
            r_, logits, gamma = model(Z_te_t)
            corr = (gamma * model.WR(r_)).norm(dim=1).mean()
            base = l0_te.norm(dim=1).mean().clamp(min=1e-9)
            engage[tag] = {"gamma_final": round(float(gamma), 5),
                           "correction_norm_ratio": round(float(corr / base), 5)}
            preds[tag] = logits.argmax(1).numpy().astype(np.int8)
            res[tag] = fam_eval(preds[tag], y7_te, FAM7, list(gen_te))
            drift[tag] = {p: round(float(t.detach().norm()), 4)
                          for p, t in model.named_parameters()
                          if p in ("U.weight", "V.weight", "WR.weight")}
    for tag in ("B0", "F0", "F1"):
        res[tag]["language"] = bucket_eval(preds[tag], y7_te, feats["language"][te])
        res[tag]["generation_mode"] = bucket_eval(preds[tag], y7_te, feats["mode"][te])
        res[tag]["source_top"] = bucket_eval(preds[tag], y7_te, feats["source"][te])
        lb = np.array([f"{lo}-{hi}" for lo, hi in LEN_BUCKETS], dtype=object)
        buck = np.array([next((f"{lo}-{hi}" for lo, hi in LEN_BUCKETS
                               if lo <= cl <= hi), "?") for cl in feats["char_len"][te]])
        res[tag]["char_len_bucket"] = bucket_eval(preds[tag], y7_te, buck)

    dF = round(res["F1"]["BA_F"] - res["F0"]["BA_F"], 4)
    dG = round(res["F1"]["BA_G"] - res["F0"]["BA_G"], 4)
    anchor["overall_ratio"] = round(anchor["valid"] / max(anchor["total"], 1), 4)
    for f in FAM7:
        td = anchor["per_family"][f]["total"]
        anchor["per_family"][f]["ratio"] = round(
            anchor["per_family"][f]["valid"] / max(td, 1), 4)
    say(f"[e36] {fold_key} test：B0 {res['B0']['BA_F']}/{res['B0']['BA_G']}，"
        f"F0 {res['F0']['BA_F']}/{res['F0']['BA_G']}，F1 {res['F1']['BA_F']}/"
        f"{res['F1']['BA_G']}，Δ={dF:+.4f}/{dG:+.4f}；anchor "
        f"{anchor['valid']}/{anchor['total']}={anchor['overall_ratio']:.3f}")

    # ---- 随机切分参照（主轴，仅参照） ----
    atr_l, ava_l = [], []
    for f in FAM7:
        fidx = np.where(is_m & (feats["family"] == f))[0]
        n_te = int((fam7_te == f).sum())
        rs = np.random.RandomState(200 + 10 * FAM7.index(f))
        perm = rs.permutation(len(fidx))
        ava_l += fidx[perm[:n_te]].tolist()
        atr_l += fidx[perm[n_te:]].tolist()
    atr_r = np.array(sorted(atr_l)); ava_r = np.array(sorted(ava_l))
    y7_rtr = fam_idx_all[atr_r].astype(np.int64)
    y7_rva = fam_idx_all[ava_r].astype(np.int64)
    fr = feats["family"][atr_r]; gr = feats["generator"][atr_r]
    fg_r = {f: sorted({str(gr[j]) for j in range(len(gr)) if fr[j] == f})
            for f in FAM7}
    gc_r = {g: int((gr == g).sum()) for f in FAM7 for g in fg_r[f]}
    w_r = np.array([len(atr_r) / (len(FAM7) * len(fg_r[fn]) * gc_r[g])
                    for fn, g in zip(fr, gr)])
    m_r, s_r = fit_scaler(feats["raw"][atr_r])
    b0r, _ = fit_lr(apply_scaler(feats["raw"][atr_r], m_r, s_r), y7_rtr, w_r,
                    f"{fold_key} 随机参照主轴", say)
    ref_rand = fam_eval(b0r.predict(apply_scaler(feats["raw"][ava_r], m_r, s_r)),
                        y7_rva, FAM7, list(feats["generator"][ava_r]))
    say(f"[e36] {fold_key} 随机参照（主轴，仅参照）：BA_F {ref_rand['BA_F']} / "
        f"BA_G {ref_rand['BA_G']}（对照 gen B0 {res['B0']['BA_F']}/"
        f"{res['B0']['BA_G']}）")

    out = {
        "n_machine": {"train": int(len(tr)), "dev": int(len(va)), "test": int(len(te))},
        "n_human": {"train": int(len(hum_tr)), "dev": int(len(hum_va)),
                    "test": int(len(hum_te))},
        "b0_converged": b0_conv,
        "B0": res["B0"], "F0": res["F0"], "F1": res["F1"],
        "deltas": {"F1_minus_F0_BA_F": dF, "F1_minus_F0_BA_G": dG},
        "det_auroc": {"test": round(det_te, 4), "dev": round(det_va, 4)},
        "det_f1_macro@.5_test": round(det_f1, 4),
        "anchor": anchor, "engagement": engage, "param_drift": drift,
        "batch_stats": {"first_last": stats, "redraw": redraw},
        "gamma_track": gamma_track, "init_maxdiff": init_diff,
        "random_ref": {"BA_F": ref_rand["BA_F"], "BA_G": ref_rand["BA_G"],
                       "note": "同本随机控制（主轴，仅参照，不入出口）"},
        "predictions": {
            "te_row_i": feats["row_i"][te], "te_family": fam_idx_all[te],
            "te_generator": gen_te, "te_language": feats["language"][te],
            "te_char_len": feats["char_len"][te], "te_tok_len": feats["tok_len"][te],
            "te_pred_B0": preds["B0"], "te_pred_F0": preds["F0"],
            "te_pred_F1": preds["F1"],
            "te_all_row_i": feats["row_i"][te_all],
            "te_det_prob": p_d_te.astype(np.float32),
            "te_det_y": feats["label"][te_all].astype(np.int8),
            "dev_all_row_i": feats["row_i"][va_all],
            "dev_det_prob": p_d_va.astype(np.float32),
            "dev_det_y": feats["label"][va_all].astype(np.int8)},
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data/h2_droid_subset"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    OUT = Path(a.out) if a.out else ROOT / "artifacts" / (
        "h2_droid_e36_smoke" if a.smoke else "h2_droid_e36")
    RUNS = ROOT / "runs" / ("h2_droid_e36_smoke" if a.smoke else "h2_droid_e36")
    if OUT.exists() and any(OUT.iterdir()) and not a.force:
        print(f"[e36] 已存在产物，默认拒绝覆盖：{OUT}（--force 可覆盖）", flush=True)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log: list[str] = []

    def say(msg):
        print(msg, flush=True)
        log.append(msg)

    commit = subprocess.run(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    say(f"[e36] commit {commit} | smoke={a.smoke} audit_only={a.audit_only} | "
        f"data={data_dir} | out={OUT}")

    # ---- 阶段 A：审计（全量行，流式；不加载 encoder） ----
    sums = verify_sha256sums(data_dir, say)
    rows = list(stream_rows(data_dir / "core.jsonl"))
    fold_plan = json.loads((data_dir / "fold_plan.json").read_text())
    summary = json.loads((data_dir / "summary.json").read_text())
    diag_info = {"rows": 0, "counts_by_label": {}}
    dc = Counter()
    for r in stream_rows(data_dir / "diagnostic_hybrid_adversarial.jsonl",
                         tolerant=True):
        diag_info["rows"] += 1
        dc[r["label"]] += 1
    diag_info["counts_by_label"] = dict(dc)
    audit = run_audit(rows, fold_plan, summary, data_dir, sums, diag_info, say,
                      strict=True)
    (OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1))
    say(f"[e36] 审计完成：machine {audit['totals']['machine']} / human "
        f"{audit['totals']['human']}；断言全部通过；audit.json 已写出")
    if a.audit_only:
        (OUT / "solver.log").write_text("\n".join(log) + "\n")
        say(f"[e36] --audit-only 结束 {time.time()-t_start:.1f}s")
        return 0

    # ---- 阶段 B/D：特征（缓存 + 复核） ----
    feats_path = RUNS / "features.npz"
    if a.smoke:
        keep, seen = [], defaultdict(int)
        for r in rows:
            key = (r["split"], r["label"], r["gen"])
            if seen[key] < SMOKE_PER_GROUP:
                keep.append(r); seen[key] += 1
        rows = keep
        say(f"[e36] smoke 子集 {len(rows)} 行（每 (split,label,generator) ≤"
            f"{SMOKE_PER_GROUP}）")
    if feats_path.exists() and not a.force:
        feats = load_features(RUNS, rows) if not a.smoke else load_features_smoke(
            RUNS, rows)
        say(f"[e36] 复用特征缓存并复核 ✓：{feats_path}")
    else:
        feats = build_features(rows, a, RUNS, say)

    # ---- 阶段 C/D：两折（smoke 仅 fold_0） ----
    fold_keys = ["fold_0"] if a.smoke else ["fold_0", "fold_1"]
    folds_out, preds_pack = {}, {}
    for fk in fold_keys:
        out = run_fold(fk, fold_plan["folds"][fk], feats, a, say)
        folds_out[fk] = {k: v for k, v in out.items() if k != "predictions"}
        for k, v in out["predictions"].items():
            preds_pack[f"{fk}_{k}"] = v
    if a.smoke:
        assert folds_out["fold_0"]["anchor"]["valid"] > 0, "smoke 未产生跨 generator anchor"
        say(f"[e36] smoke 协议检查：anchor {folds_out['fold_0']['anchor']['valid']} "
            f"有效（>0 ✓）")
    # pooled（两折 test 互补 = 全部 machine test 行）
    if not a.smoke:
        te_row = np.concatenate([preds_pack["fold_0_te_row_i"],
                                 preds_pack["fold_1_te_row_i"]])
        y7p = np.concatenate([preds_pack["fold_0_te_family"],
                              preds_pack["fold_1_te_family"]]).astype(np.int64)
        genp = np.concatenate([preds_pack["fold_0_te_generator"],
                               preds_pack["fold_1_te_generator"]])
        pooled = {}
        for tag in ("B0", "F0", "F1"):
            pr = np.concatenate([preds_pack[f"fold_0_te_pred_{tag}"],
                                 preds_pack[f"fold_1_te_pred_{tag}"]])
            pooled[tag] = fam_eval(pr, y7p, FAM7, list(genp))
        pdF = round(pooled["F1"]["BA_F"] - pooled["F0"]["BA_F"], 4)
        pdG = round(pooled["F1"]["BA_G"] - pooled["F0"]["BA_G"], 4)
        pooled["deltas"] = {"F1_minus_F0_BA_F": pdF, "F1_minus_F0_BA_G": pdG}
        say(f"[e36] pooled test：B0 {pooled['B0']['BA_F']}/{pooled['B0']['BA_G']}，"
            f"F0 {pooled['F0']['BA_F']}/{pooled['F0']['BA_G']}，F1 "
            f"{pooled['F1']['BA_F']}/{pooled['F1']['BA_G']}，Δ {pdF:+.4f}/{pdG:+.4f}")

    # ---- 出口（预注册 §6） ----
    if a.smoke:
        exit_info = {"outcome": "smoke：仅协议验证，不做出口判定"}
    else:
        d = {fk: folds_out[fk]["deltas"] for fk in fold_keys}
        dF0 = d["fold_0"]["F1_minus_F0_BA_F"]; dG0 = d["fold_0"]["F1_minus_F0_BA_G"]
        dF1 = d["fold_1"]["F1_minus_F0_BA_F"]; dG1 = d["fold_1"]["F1_minus_F0_BA_G"]
        fold_pos = bool(dF0 > 0 and dG0 > 0 and dF1 > 0 and dG1 > 0)
        anchor_ok = bool(all(folds_out[fk]["anchor"]["overall_ratio"] >= ANCHOR_MIN
                             for fk in fold_keys))
        pooled_ok = bool(pdF >= DELTA_THRESH and pdG >= DELTA_THRESH)
        if not anchor_ok:
            outcome = ("4：有效 anchor 比例过低 ⇒ 检验不足（不得写成阴性）")
        elif fold_pos and pooled_ok:
            outcome = ("1：H2 获得初步支持（两折方向一致且 pooled ΔBA_F、ΔBA_G "
                       "均 ≥1pt）→ 由指导端决定是否做一次局部几何消融")
        elif fold_pos and (pdF > 0 or pdG > 0):
            outcome = ("2：方向一致但 pooled 增量 <1pt ⇒ 当前 Droid 子集与 768 维表示"
                       "下不支持实用 H2；停止继续堆叠损失")
        else:
            outcome = "3：方向不一致或 pooled 增量为非正 ⇒ H2 不支持"
        exit_info = {
            "rule": {"1": "两折 BA_F/BA_G 均正 且 pooled 两项 ≥1pt",
                     "2": "方向一致但 pooled <1pt",
                     "3": "方向不一致或 pooled 非正",
                     "4": f"任一折 anchor 比例 < {ANCHOR_MIN} ⇒ 检验不足",
                     "5": "只改善单个 family/role/generator ⇒ 仅报异质性"},
            "fold_deltas": d, "pooled_deltas": {"BA_F": pdF, "BA_G": pdG},
            "fold_positive": fold_pos, "anchor_ok": anchor_ok,
            "pooled_ok": pooled_ok, "outcome": outcome}
        say(f"[e36] 出口：fold_positive={fold_pos} anchor_ok={anchor_ok} "
            f"pooled_ok={pooled_ok} → {outcome}")

    # ---- 产物 ----
    np.savez_compressed(OUT / "predictions.npz", classes_=np.array(FAM7),
                        **preds_pack)
    metrics = {
        "commit": commit, "smoke": a.smoke,
        "data": {"source_revision": summary.get("source_revision"),
                 "core_sha256": sums["core.jsonl"]["sha256"],
                 "fold_plan_sha256": sums["fold_plan.json"]["sha256"]},
        "protocol": ("fold_plan.json 两折；B0 加权 LR（C=.1，w=N/(K|G_f|N_g)）冻结；"
                     "残差=阶段 B 修正门控；F0=L_F，F1=L_F+0.1·L_cross（τ=.1，"
                     "同家族不同 generator 正对）；批 7×18；2ep；AdamW 1e-3/wd 1e-4"),
        "usage": {"test_used_for_final_eval_only": True,
                  "diagnostic_used": False,
                  "diagnostic_counted_in_audit": True},
        "folds": folds_out, "exit": exit_info,
    }
    if not a.smoke:
        metrics["pooled"] = pooled
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    manifest = {
        "commit": commit, "smoke": a.smoke,
        "data_files": {k: sums[k] for k in sums},
        "features_cache": {"path": str(RUNS / "features.npz"),
                           "sha256": sha256_file(RUNS / "features.npz"),
                           "n": int(len(feats["row_i"])),
                           "skipped": int(len(feats["skipped_row_i"]))},
        "feature_spec": {"tokenizer": "checkpoints/codet5-base",
                         "add_special_tokens": False,
                         "max_length": a.max_length,
                         "long_rule": ">max_length → 头 75%+尾 25%（默认 768+256）",
                         "min_tokens": 8,
                         "encoder": "runs/v0.4.1_covreg/last.pt（冻结）+ SetPool "
                                    "runs/flagship_r1/head.pt + m_raw 均值池化",
                         "batch_size": a.batch_size},
        "counts": {"audit": audit["totals"]},
        "excluded": ["diagnostic_hybrid_adversarial.jsonl（仅审计计数）"],
        "notes": ["DroidCollection 无公开 task_id/prompt_id；结果为跨 generator "
                  "家族归因候选证据",
                  "test（held-out generator）仅用于最终评估"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (OUT / "config.json").write_text(json.dumps({
        "commit": commit, "C_fixed": C_FIXED, "lr": 1e-3, "wd": 1e-4, "tau": TAU,
        "lambda_C": LAM_C, "epochs": EPOCHS, "batch": "7 族×18", "seed": 0,
        "model": "ℓ=ℓ0+γ·W_R·r；ℓ0=加权 LR 冻结；修正门控 W_R=0、η0=.5；"
                 "r=V·GELU(Uz+b1)+b2（U Xavier、V std=1e-3、b=0）",
        "arms": {"B0": "冻结主轴（参照）", "F0": "残差 CE",
                 "F1": "残差 CE+0.1·L_cross（同家族不同 generator 正对）"},
        "exit_rule": exit_info.get("rule", {}), "anchor_min": ANCHOR_MIN,
    }, ensure_ascii=False, indent=1))
    (OUT / "solver.log").write_text("\n".join(log) + "\n")
    say(f"[e36] 完成 {time.time()-t_start:.1f}s；产物 {OUT}")
    return 0


def load_features_smoke(runs_dir: Path, rows):
    """smoke 缓存复核（子集行；row_i 为全文件行号）。"""
    z = dict(np.load(runs_dir / "features.npz", allow_pickle=True))
    by_i = {r["i"]: r for r in rows}
    ri = z["row_i"]; sh = z["sha1"]
    for j in range(len(ri)):
        r = by_i.get(int(ri[j]))
        assert r is not None and r["sha1"] == sh[j], f"smoke 特征缓存与 JSONL 不符 @ {j}"
    return z


if __name__ == "__main__":
    sys.exit(main())
