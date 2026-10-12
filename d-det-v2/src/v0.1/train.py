#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""d-det-v2 v0.1 · 训练与准备入口

子命令：
  prep    数据准备：清洗/去重/task 划分/校准子集/登记 manifest 与 data_role（CPU）
  tokens  原始代码 token 数缓存与长度分桶
  s1      阶段一：B0/B1 全参微调（CodeT5-base；B1 训练期 AMMC 视图）
  feat    阶段二特征：s(x) 缓存、冻结 B1 的 h / 基线 logits、任务中心、5 折 OOF 参考
  s2      阶段二：Ours/B2/B3/机制小头（冻结编码器，训练/推理协议见《设计指导》§5）
  smoke   短流程验证（全链路小样本，不写 artifacts 正式记录）

默认值来自 configs.yaml（《设计指导》§10.2）。所有产物：重型进 work/（gitignore），
记录（json/txt）进 artifacts/。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D          # noqa: E402
import model as M         # noqa: E402
import center as C        # noqa: E402

T0 = time.time()


# --------------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------------- #

def repo_root() -> Path:
    return D.resolve("../../..")


def git_head() -> str:
    try:
        r = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else "?"
    except Exception:
        return "?"


def env_info() -> dict:
    import transformers
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def sha256_file(path, buf=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def work_paths(cfg, work_root=None):
    root = Path(work_root) if work_root else D.resolve(cfg["paths"]["work"])
    return {"root": root, "records": root / "records.jsonl", "cache": root / "cache",
            "ckpt": root / "ckpt", "feat": root / "features", "runs": root / "runs"}


def art_paths(cfg):
    root = D.resolve(cfg["paths"]["artifacts"])
    return {"root": root, "prep": root / "prep", "runs": root / "runs", "reports": root / "reports"}


def get_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", encoding="utf-8")
    D.log(f"[log] {path}", fh)
    return fh


# --------------------------------------------------------------------------- #
# 采样计划（epoch 级，F 八类均衡 / D 人机各半、AI 内七族等权）
# --------------------------------------------------------------------------- #

def epoch_plan(task, recs, pool_idx):
    """按类/族分组，供掩码式采样；返回 {class -> [rec_idx]}。"""
    groups = {}
    if task == "F":
        for i in pool_idx:
            groups.setdefault(recs[i]["fam"], []).append(i)
    else:
        for i in pool_idx:
            key = "Human" if recs[i]["kind"] == "human" else recs[i]["fam"]
            groups.setdefault(key, []).append(i)
    return groups


def sample_epoch(task, groups, rng):
    picks = []
    if task == "F":
        n = min(len(v) for v in groups.values())
        for rows in groups.values():
            if len(rows) >= n:
                picks += rng.sample(rows, n)
            else:  # 极端小样本（冒烟）
                picks += [rng.choice(rows) for _ in range(n)]
    else:
        humans = groups.get("Human", [])
        fams = sorted(k for k in groups if k != "Human")
        n_h = len(humans)
        picks += list(humans)
        per, rem = divmod(n_h, max(len(fams), 1))
        for k, f in enumerate(fams):
            cnt = per + (1 if k < rem else 0)
            rows = groups[f]
            if cnt <= len(rows):
                picks += rng.sample(rows, cnt)
            else:
                picks += [rng.choice(rows) for _ in range(cnt)]
    rng.shuffle(picks)
    return picks


def ammc_decisions(cfg, n, rng):
    a = cfg["ammc"]
    out = []
    for _ in range(n):
        dec = {"aug": rng.random() < a["p_aug"]}
        if dec["aug"]:
            dec["comment"] = rng.random() < a["p_comment"]
            dec["ws"] = rng.random() < a["p_ws"]
            dec["rename"] = rng.random() < a["p_rename"]
        out.append(dec)
    return out


# --------------------------------------------------------------------------- #
# 阶段一：数据管道
# --------------------------------------------------------------------------- #

class S1Dataset(Dataset):
    def __init__(self, recs, picks, decisions, tok, chunk_len=512, max_chunks=4):
        self.recs, self.picks, self.decisions = recs, picks, decisions
        self.tok, self.chunk_len, self.max_chunks = tok, chunk_len, max_chunks

    def __len__(self):
        return len(self.picks)

    def __getitem__(self, k):
        i = self.picks[k]
        r = self.recs[i]
        code = r["code"]
        if self.decisions is not None:
            code, _ = D.ammc_view(code, r["lang"], self.decisions[k])
        chunks = D.encode_chunks(self.tok, code, self.chunk_len, self.max_chunks)
        C = len(chunks)
        ids = torch.zeros(C, self.chunk_len, dtype=torch.long)
        mask = torch.zeros(C, self.chunk_len, dtype=torch.long)
        for c, x in enumerate(chunks):
            n = x.shape[0]
            ids[c, :n] = x
            mask[c, :n] = 1
        return ids, mask, i


def collate_s1(batch):
    Cmax = max(b[0].shape[0] for b in batch)
    L = batch[0][0].shape[1]
    ids = torch.zeros(len(batch), Cmax, L, dtype=torch.long)
    mask = torch.zeros(len(batch), Cmax, L, dtype=torch.long)
    for k, (i_, m_, _) in enumerate(batch):
        ids[k, :i_.shape[0]] = i_
        mask[k, :m_.shape[0]] = m_
    return ids, mask


def class_ids(recs, idx, task):
    if task == "F":
        return torch.tensor([D.CLASSES_F.index(recs[i]["fam"]) for i in idx], dtype=torch.long)
    return torch.tensor([0 if recs[i]["kind"] == "human" else 1 for i in idx], dtype=torch.long)


@torch.no_grad()
def encoder_forward_sets(model, tok, recs, idx_list, chunk_len, max_chunks, batch=8):
    """在给定下标集上批量前向（分块编码器）→ (h [n,d], 原始顺序)。"""
    model.eval()
    outs = []
    for s in range(0, len(idx_list), batch):
        sub = idx_list[s:s + batch]
        items = []
        for i in sub:
            chunks = D.encode_chunks(tok, recs[i]["code"], chunk_len, max_chunks)
            C = len(chunks)
            ids = torch.zeros(C, chunk_len, dtype=torch.long)
            mask = torch.zeros(C, chunk_len, dtype=torch.long)
            for c, x in enumerate(chunks):
                n = x.shape[0]
                ids[c, :n] = x
                mask[c, :n] = 1
            items.append((ids, mask))
        Cmax = max(x[0].shape[0] for x in items)
        ids = torch.zeros(len(items), Cmax, chunk_len, dtype=torch.long)
        mask = torch.zeros(len(items), Cmax, chunk_len, dtype=torch.long)
        for k, (i_, m_) in enumerate(items):
            ids[k, :i_.shape[0]] = i_
            mask[k, :m_.shape[0]] = m_
        ids, mask = ids.cuda(), mask.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encoder(ids, mask)
        outs.append(h.float().cpu())
    return torch.cat(outs, 0) if outs else torch.zeros(0)


# --------------------------------------------------------------------------- #
# 阶段一训练
# --------------------------------------------------------------------------- #

def train_s1(cfg, task, arm, seed, work_root=None, epochs=None, max_steps=None,
             micro_batch=None, accum=None, log_fh=None, tag=""):
    wp = work_paths(cfg, work_root)
    recs = D.load_records_file(wp["records"])
    pool = D.subset_index(recs, "train", main_only=True)
    calib = D.subset_index(recs, "train", calib=True)
    groups = epoch_plan(task, recs, pool)
    seed_everything(seed)

    mb = micro_batch or cfg["stage1"]["micro_batch"]
    ac = accum or cfg["stage1"]["grad_accum"]
    ep = epochs or cfg["stage1"]["epochs"]
    n_out = 8 if task == "F" else 1
    tok = D.load_tokenizer(cfg)

    model = M.Stage1Model(cfg["paths"]["encoder"] if Path(cfg["paths"]["encoder"]).is_absolute() else str(D.resolve(cfg["paths"]["encoder"])),
                          n_out, grad_checkpointing=cfg["stage1"]["grad_checkpointing"]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["stage1"]["lr"], weight_decay=cfg["stage1"]["weight_decay"])

    rng = random.Random(seed * 100000 + 7)
    plans = [sample_epoch(task, groups, random.Random(seed * 100000 + e)) for e in range(ep)]
    steps_per_epoch = math.ceil(len(plans[0]) / (mb * ac))
    total_steps = steps_per_epoch * ep
    warmup = max(1, int(cfg["stage1"]["warmup_frac"] * total_steps))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    ckpt_dir = wp["ckpt"] / f"{task}_{arm}_s{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], {"epoch": -1, "metric": -1.0}
    step = 0
    t_start = time.time()
    for e in range(ep):
        dec = ammc_decisions(cfg, len(plans[e]), rng) if arm == "B1" else None
        ds = S1Dataset(recs, plans[e], dec, tok, cfg["model"]["chunk_len"], cfg["model"]["max_chunks"])
        dl = DataLoader(ds, batch_size=mb, shuffle=False, num_workers=cfg["stage1"]["num_workers"],
                        collate_fn=collate_s1, pin_memory=True)
        model.train()
        run_loss, nb = 0.0, 0
        for k, (ids, mask) in enumerate(dl):
            y = torch.tensor([D.CLASSES_F.index(recs[i]["fam"]) if task == "F" else (0 if recs[i]["kind"] == "human" else 1)
                              for i in plans[e][k * mb:(k + 1) * mb]], dtype=torch.long, device="cuda")
            ids, mask = ids.cuda(), mask.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(ids, mask)
                logits = logits.float()
                loss = (Fn.cross_entropy(logits, y) if task == "F" else Fn.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float()))
                loss = loss / ac
            loss.backward()
            if (k + 1) % ac == 0 or (k + 1) == len(dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for g in opt.param_groups:
                    g["lr"] = cfg["stage1"]["lr"] * lr_factor(step)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            run_loss += loss.item() * ac
            nb += 1
            if max_steps and nb >= max_steps:
                break
        # 校准子集评估（checkpoint 选择只用训练内校准子集）
        metric, _ = eval_stage1(model, tok, recs, calib, task, cfg)
        history.append({"epoch": e, "train_loss": run_loss / max(nb, 1), "calib": metric, "steps": step})
        D.log(f"[s1 {task}/{arm}/s{seed}] epoch {e}: loss={run_loss/max(nb,1):.4f} calib={metric:.4f}", log_fh)
        torch.save({"encoder": model.encoder.state_dict(), "head": model.head.state_dict(),
                    "task": task, "arm": arm, "seed": seed, "epoch": e, "calib": metric},
                   ckpt_dir / "last.pt")
        if metric > best["metric"]:
            best = {"epoch": e, "metric": metric}
            torch.save({"encoder": model.encoder.state_dict(), "head": model.head.state_dict(),
                        "task": task, "arm": arm, "seed": seed, "epoch": e, "calib": metric},
                       ckpt_dir / "best.pt")
        if e < ep - 1:
            del dl
    info = {"task": task, "arm": arm, "stage": 1, "seed": seed, "epochs": ep,
            "best_epoch": best["epoch"], "calib_metric": "macro_f1" if task == "F" else "auroc",
            "best": best["metric"], "history": history,
            "samples_per_epoch": len(plans[0]), "exposures": sum(len(p) for p in plans),
            "steps": step, "time_s": round(time.time() - t_start, 1),
            "micro_batch": mb, "grad_accum": ac, "lr": cfg["stage1"]["lr"],
            "ammc": arm == "B1", "git": git_head(), "tag": tag}
    return info


@torch.no_grad()
def eval_stage1(model, tok, recs, idx, task, cfg, batch=8):
    """校准评估：返回 (主指标, probs)。F: macro-F1；D: AUROC。"""
    if len(idx) == 0:
        return float("nan"), None
    model.eval()
    logits_all = []
    for s in range(0, len(idx), batch):
        sub = idx[s:s + batch]
        items = []
        for i in sub:
            chunks = D.encode_chunks(tok, recs[i]["code"], cfg["model"]["chunk_len"], cfg["model"]["max_chunks"])
            C = len(chunks)
            ids = torch.zeros(C, cfg["model"]["chunk_len"], dtype=torch.long)
            mask = torch.zeros(C, cfg["model"]["chunk_len"], dtype=torch.long)
            for c, x in enumerate(chunks):
                n = x.shape[0]
                ids[c, :n] = x
                mask[c, :n] = 1
            items.append((ids, mask))
        Cmax = max(x[0].shape[0] for x in items)
        ids = torch.zeros(len(items), Cmax, cfg["model"]["chunk_len"], dtype=torch.long)
        mask = torch.zeros(len(items), Cmax, cfg["model"]["chunk_len"], dtype=torch.long)
        for k, (i_, m_) in enumerate(items):
            ids[k, :i_.shape[0]] = i_
            mask[k, :m_.shape[0]] = m_
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(ids.cuda(), mask.cuda())
        logits_all.append(logits.float().cpu())
    logits = torch.cat(logits_all, 0)
    y = class_ids(recs, idx, task).numpy()
    if task == "F":
        pred = logits.argmax(-1).numpy()
        probs = torch.softmax(logits, -1).numpy()
        return float(f1_score(y, pred, average="macro")), probs
    p = torch.sigmoid(logits.squeeze(-1)).numpy()
    return float(roc_auc_score(y, p)), p


# --------------------------------------------------------------------------- #
# 特征构建（s、h、基线 logits、参考器）
# --------------------------------------------------------------------------- #

def compute_s(cfg, wp, recs, force=False, log_fh=None):
    """内容代理 s(x)=E_sem(Q(x))（冻结预训练 CodeT5，均值池化）。缓存到 cache/s.pt。"""
    path = wp["cache"] / "s.pt"
    rsha = sha256_file(wp["records"])[:16]
    if path.exists() and not force:
        obj = torch.load(path, map_location="cpu")
        if obj.get("records_sha16") == rsha and obj["s"].shape[0] == len(recs):
            D.log(f"[s] 复用缓存 {path} ({obj['s'].shape})", log_fh)
            return obj["s"]
    tok = D.load_tokenizer(cfg)
    enc = M.EncoderPool(str(D.resolve(cfg["paths"]["encoder"]))).cuda().eval()
    M.freeze_(enc)
    out = []
    t0 = time.time()
    for s in range(0, len(recs), 16):
        sub = recs[s:s + 16]
        items = []
        for r in sub:
            q, _ = D.content_proxy(r["code"], r["lang"])
            chunks = D.encode_chunks(tok, q, cfg["model"]["chunk_len"], cfg["model"]["max_chunks"])
            C = len(chunks)
            ids = torch.zeros(C, cfg["model"]["chunk_len"], dtype=torch.long)
            mask = torch.zeros(C, cfg["model"]["chunk_len"], dtype=torch.long)
            for c, x in enumerate(chunks):
                n = x.shape[0]
                ids[c, :n] = x
                mask[c, :n] = 1
            items.append((ids, mask))
        Cmax = max(x[0].shape[0] for x in items)
        ids = torch.zeros(len(items), Cmax, cfg["model"]["chunk_len"], dtype=torch.long)
        mask = torch.zeros(len(items), Cmax, cfg["model"]["chunk_len"], dtype=torch.long)
        for k, (i_, m_) in enumerate(items):
            ids[k, :i_.shape[0]] = i_
            mask[k, :m_.shape[0]] = m_
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = enc(ids.cuda(), mask.cuda())
        out.append(h.float().cpu())
        if (s // 16) % 200 == 0:
            D.log(f"[s] {s}/{len(recs)} ({time.time()-t0:.0f}s)", log_fh)
    s_all = torch.cat(out, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"s": s_all, "records_sha16": rsha}, path)
    D.log(f"[s] 完成 {s_all.shape}，{time.time()-t0:.0f}s → {path}", log_fh)
    return s_all


def compute_token_lens(cfg, wp, recs, force=False, log_fh=None):
    path = wp["cache"] / "tokens.pt"
    rsha = sha256_file(wp["records"])[:16]
    if path.exists() and not force:
        obj = torch.load(path, map_location="cpu")
        if obj.get("records_sha16") == rsha and len(obj["lens"]) == len(recs):
            D.log(f"[tokens] 复用缓存 {path}", log_fh)
            return obj["lens"]
    tok = D.load_tokenizer(cfg)
    lens = D.compute_token_lens(tok, recs)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lens": lens, "records_sha16": rsha}, path)
    D.log(f"[tokens] {len(recs)} 条已缓存 → {path}", log_fh)
    return lens


def build_features(cfg, task, seed, work_root=None, force=False, log_fh=None):
    """阶段二特征：h/基线 logits（冻结 B1）+ s + 中心与 OOF 参考。"""
    wp = work_paths(cfg, work_root)
    recs = D.load_records_file(wp["records"])
    tok = D.load_tokenizer(cfg)
    fdir = wp["feat"] / f"{task}_s{seed}"
    fdir.mkdir(parents=True, exist_ok=True)

    s_all = compute_s(cfg, wp, recs, force=force, log_fh=log_fh)

    b1 = wp["ckpt"] / f"{task}_B1_s{seed}" / "best.pt"
    assert b1.exists(), f"缺少 B1 checkpoint：{b1}"
    ck = torch.load(b1, map_location="cpu")
    model = M.Stage1Model(str(D.resolve(cfg["paths"]["encoder"])), 8 if task == "F" else 1).cuda()
    model.encoder.load_state_dict(ck["encoder"])
    model.head.load_state_dict(ck["head"])

    idx_main = D.subset_index(recs, "train", main_only=True)
    idx_calib = D.subset_index(recs, "train", calib=True)
    idx_dev = D.subset_index(recs, "dev")
    sets = {"main": idx_main, "calib": idx_calib, "dev": idx_dev}
    if cfg["data"].get("test_read"):
        sets["test"] = D.subset_index(recs, "test")

    with torch.no_grad():
        for name, idx in sets.items():
            hp = fdir / f"h_{name}.pt"
            lp = fdir / f"logits_{name}.pt"
            if hp.exists() and lp.exists() and not force:
                D.log(f"[feat] 复用 {hp.name}", log_fh)
                continue
            h = encoder_forward_sets(model, tok, recs, idx, cfg["model"]["chunk_len"], cfg["model"]["max_chunks"])
            head = model.head
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(h.cuda()).float().cpu()
            torch.save(h.half(), hp)
            torch.save(logits, lp)
            D.log(f"[feat] {task}/s{seed} {name}: h{h.shape} logits{tuple(logits.shape)}", log_fh)

    # 中心 + 5 折 OOF 参考（s/h 均按全局记录对齐）
    h_main = torch.load(fdir / "h_main.pt", map_location="cpu").float()
    h_full = torch.zeros(len(recs), h_main.shape[1])
    h_full[idx_main] = h_main
    oof, refs, m_global, centers = C.build_refs(
        task, recs, idx_main, s_all, h_full, cfg["data"]["folds"],
        cfg["stage2"]["ridge_alphas"], cfg["stage2"]["ridge_inner_folds"])
    torch.save(oof, fdir / "oof_main.pt")
    torch.save(m_global, fdir / "mglobal.pt")
    ref_meta = {"folds": sorted(refs.keys()), "alphas": {str(j): refs[j].alpha for j in refs},
                "task": task, "seed": seed, "n_main": len(idx_main)}
    for name in ("calib", "dev", "test"):
        if name not in sets:
            continue
        h = torch.load(fdir / f"h_{name}.pt", map_location="cpu").float()
        s = s_all[sets[name]]
        m5 = C.predict_refs(refs, s)  # [n,5,d]
        torch.save(m5.half(), fdir / f"m5_{name}.pt")
        D.log(f"[feat] {name} m5 {tuple(m5.shape)}", log_fh)
    np.savez(fdir / "refs.npz", **{f"W{j}": refs[j].W for j in refs},
             **{f"mu{j}": refs[j].mu for j in refs}, **{f"sd{j}": refs[j].sd for j in refs},
             **{f"ym{j}": refs[j].ym for j in refs})
    with open(fdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({**ref_meta, "b1_ckpt": str(b1), "b1_calib": ck.get("calib"),
                   "n_calib": len(idx_calib), "n_dev": len(idx_dev)}, f, ensure_ascii=False, indent=1)
    D.log(f"[feat] 完成 {fdir}；refs={ref_meta['folds']} alphas={ref_meta['alphas']}", log_fh)
    return fdir


# --------------------------------------------------------------------------- #
# 阶段二：校正头训练
# --------------------------------------------------------------------------- #

F_ARMS = ["ours", "b2", "b3", "mech_r", "mech_ref"]
D_ARMS = ["ours", "b2", "b3", "mech_r", "mech_ref", "mech_signed", "mech_mag"]


def make_head(task, arm, d, cfg):
    rank, hid = cfg["model"]["rank"], cfg["model"]["head_hidden"]
    if task == "F":
        in_dim = 2 * d if arm in ("ours", "b2", "b3") else d
        return M.FHead(in_dim, rank, n_cls=8)
    in_dim = 2 * d if arm in ("ours", "b2", "b3", "mech_signed", "mech_mag") else d
    mode = "signed" if arm == "mech_signed" else ("mag" if arm == "mech_mag" else "full")
    return M.DHead(in_dim, rank, hid, mode=mode)


def arm_inputs(arm, h, m=None):
    if arm in ("ours", "b3", "mech_signed", "mech_mag"):
        return M.feat_full(h, m)
    if arm == "b2":
        return M.feat_b2(h)
    if arm == "mech_r":
        return M.feat_r(h, m)
    if arm == "mech_ref":
        return M.feat_ref(m)
    raise ValueError(arm)


def infer_logits(task, arm, head, h, base_logits, m5=None, m_global=None):
    """推理协议：5 个参考器分别构造校正 logits 后取均值，再一次 softmax/sigmoid。"""
    dev = next(head.parameters()).device
    h = h.to(dev)
    base_logits = base_logits.to(dev)
    if base_logits.ndim == 2 and base_logits.shape[-1] == 1:  # D: (n,1) → (n,)
        base_logits = base_logits.squeeze(-1)
    if m5 is not None:
        m5 = m5.to(dev)
    if m_global is not None:
        m_global = m_global.to(dev)
    if arm in ("ours", "mech_r", "mech_ref", "mech_signed", "mech_mag"):
        assert m5 is not None
        corrs = [head(arm_inputs(arm, h, m5[:, j])) for j in range(m5.shape[1])]
        corr = torch.stack(corrs).mean(0)
    elif arm == "b3":
        corr = head(arm_inputs(arm, h, m_global))
    else:
        corr = head(arm_inputs(arm, h))
    return base_logits + corr


def train_s2(cfg, task, seed, arms, work_root=None, epochs=None, max_steps=None, log_fh=None, tag=""):
    wp = work_paths(cfg, work_root)
    recs = D.load_records_file(wp["records"])
    fdir = wp["feat"] / f"{task}_s{seed}"
    assert fdir.exists(), f"先运行 feat：{fdir}"

    idx_main = D.subset_index(recs, "train", main_only=True)
    idx_calib = D.subset_index(recs, "train", calib=True)
    h_main = torch.load(fdir / "h_main.pt", map_location="cpu").float()
    h_calib = torch.load(fdir / "h_calib.pt", map_location="cpu").float()
    base_main = torch.load(fdir / "logits_main.pt", map_location="cpu").float()
    base_calib = torch.load(fdir / "logits_calib.pt", map_location="cpu").float()
    oof = torch.load(fdir / "oof_main.pt", map_location="cpu")
    m5_calib = torch.load(fdir / "m5_calib.pt", map_location="cpu").float()
    m_global = torch.load(fdir / "mglobal.pt", map_location="cpu")
    y_main = class_ids(recs, idx_main, task)
    y_calib = class_ids(recs, idx_calib, task)
    pos = {i: k for k, i in enumerate(idx_main)}

    pool = D.subset_index(recs, "train", main_only=True)
    groups = epoch_plan(task, recs, pool)
    ep = epochs or cfg["stage2"]["epochs"]
    plans = [sample_epoch(task, groups, random.Random(seed * 100000 + e)) for e in range(ep)]

    results = []
    for arm in arms:
        seed_everything(seed + 1000)
        head = make_head(task, arm, h_main.shape[1], cfg).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=cfg["stage2"]["head_lr"], weight_decay=cfg["stage2"]["weight_decay"])
        total_steps = sum(math.ceil(len(p) / cfg["stage2"]["batch"]) for p in plans)
        warmup = max(1, int(cfg["stage2"]["warmup_frac"] * total_steps))
        step, best = 0, {"epoch": -1, "metric": -1.0}
        history = []
        t0 = time.time()
        for e in range(ep):
            picks = plans[e]
            order = torch.randperm(len(picks), generator=torch.Generator().manual_seed(seed * 100000 + 500 + e)).tolist()
            picks = [picks[k] for k in order]
            head.train()
            run, nb = 0.0, 0
            for s in range(0, len(picks), cfg["stage2"]["batch"]):
                sub = picks[s:s + cfg["stage2"]["batch"]]
                kk = torch.tensor([pos[i] for i in sub])
                x = arm_inputs(arm, h_main[kk], oof[kk]).cuda()
                corr = head(x)
                base = base_main[kk].cuda()
                y = y_main[kk].cuda()
                if task == "F":
                    loss = Fn.cross_entropy(base.detach() + corr, y)
                else:
                    loss = Fn.binary_cross_entropy_with_logits(base.detach().squeeze(-1) + corr, y.float())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                lr = cfg["stage2"]["head_lr"]
                if step < warmup:
                    lr *= (step + 1) / warmup
                else:
                    prog = (step - warmup) / max(1, total_steps - warmup)
                    lr *= 0.5 * (1 + math.cos(math.pi * prog))
                for g in opt.param_groups:
                    g["lr"] = lr
                opt.step()
                run += loss.item(); nb += 1; step += 1
                if max_steps and nb >= max_steps:
                    break
            # 校准评估（与推理同协议）
            head.eval()
            with torch.no_grad():
                logits = infer_logits(task, arm, head, h_calib, base_calib, m5_calib, m_global)
                logits = logits.detach().float().cpu()
                if task == "F":
                    metric = float(f1_score(y_calib.numpy(), logits.argmax(-1).numpy(), average="macro"))
                else:
                    metric = float(roc_auc_score(y_calib.numpy(), logits.squeeze(-1).numpy()))
            history.append({"epoch": e, "train_loss": run / max(nb, 1), "calib": metric})
            D.log(f"[s2 {task}/{arm}/s{seed}] epoch {e}: loss={run/max(nb,1):.4f} calib={metric:.4f}", log_fh)
            if metric > best["metric"]:
                best = {"epoch": e, "metric": metric}
                cdir = wp["ckpt"] / f"{task}_{arm}_s{seed}"
                cdir.mkdir(parents=True, exist_ok=True)
                torch.save({"head": head.state_dict(), "task": task, "arm": arm, "seed": seed,
                            "epoch": e, "calib": metric,
                            "in_dim": h_main.shape[1], "rank": cfg["model"]["rank"]}, cdir / "best.pt")
        info = {"task": task, "arm": arm, "stage": 2, "seed": seed, "epochs": ep,
                "best_epoch": best["epoch"], "best": best["metric"], "history": history,
                "samples_per_epoch": len(plans[0]), "exposures": sum(len(p) for p in plans),
                "time_s": round(time.time() - t0, 1), "head_lr": cfg["stage2"]["head_lr"],
                "git": git_head(), "tag": tag}
        results.append(info)
        D.log(f"[s2 {task}/{arm}/s{seed}] best epoch {best['epoch']} calib {best['metric']:.4f}", log_fh)
    return results


# --------------------------------------------------------------------------- #
# prep / tokens / smoke
# --------------------------------------------------------------------------- #

def transform_acceptance(recs, log_fh=None):
    """预置变换接受率：单变换与 Q，按语言（rename 另按 family）统计请求/接受。"""
    from collections import defaultdict
    stats = defaultdict(lambda: defaultdict(lambda: {"n": 0, "applied": 0}))
    qstats = defaultdict(lambda: defaultdict(lambda: {"n": 0, "applied": 0}))
    ren_fam = defaultdict(lambda: {"n": 0, "applied": 0})
    for r in recs:
        if r["split"] != "train":
            continue
        lang = r["lang"]
        for name in ("comment", "ws", "rename"):
            s = stats[lang][name]; s["n"] += 1
            ok = D.apply_transform(r["code"], lang, name) is not None
            s["applied"] += int(ok)
            if name == "rename":
                rf = ren_fam[r["fam"]]; rf["n"] += 1; rf["applied"] += int(ok)
        _, meta = D.content_proxy(r["code"], lang)
        for name in ("comment", "ws", "rename"):
            s = qstats[lang][name]; s["n"] += 1; s["applied"] += int(bool(meta.get(name)))
    out = {"ammc_single": {k: {n: dict(v) for n, v in d.items()} for k, d in stats.items()},
           "content_proxy": {k: {n: dict(v) for n, v in d.items()} for k, d in qstats.items()},
           "rename_by_family": {k: dict(v) for k, v in ren_fam.items()}}
    if log_fh is not None:
        for lang in sorted(out["ammc_single"]):
            row = out["ammc_single"][lang]
            D.log(f"[prep] 变换接受 {lang}: " + " ".join(
                f"{n}={row[n]['applied']}/{row[n]['n']}" for n in row), log_fh)
    return out


def cmd_prep(cfg, work_root=None, limit_tasks=None, log_fh=None):
    wp = work_paths(cfg, work_root)
    recs, stats = D.load_records(cfg, limit_tasks=limit_tasks)
    mst = D.assign_meta(recs, cfg)
    D.save_records(recs, wp["records"])
    tstats = transform_acceptance(recs, log_fh=log_fh)
    droot = D.resolve(cfg["paths"]["stacad"])
    hashes = {f: sha256_file(droot / f) for f in ["core.jsonl", "task_index.jsonl", "split_manifest.json", "summary.json"]}
    cfg_sha = sha256_file(D.SRC / "configs.yaml")
    manifest = {
        "dataset": "h2_stacad_alignment_v1", "source_hashes": hashes,
        "records": {"raw": stats["raw_rows"], "kept": stats["kept_rows"],
                    "dup_exact_sha": stats["dup_exact_sha"], "dup_norm": stats["norm_sha_collisions"],
                    "dropped": stats["dropped"],
                    "by_split": stats["by_split"], "by_family": stats["by_family"]},
        "splits": mst, "limit_tasks": limit_tasks,
        "label_space": D.CLASSES_F, "folds": cfg["data"]["folds"],
        "calib_frac": cfg["data"]["calib_frac"], "calib_seed": cfg["data"]["calib_seed"],
        "calib_tasks": sorted({r["tid"] for r in recs if r["calib"]}),
        "transform_acceptance": tstats,
        "git": git_head(), "configs_sha16": cfg_sha[:16], "env": env_info(),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    ap = art_paths(cfg)
    ap["prep"].mkdir(parents=True, exist_ok=True)
    if limit_tasks is None:
        with open(ap["prep"] / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        with open(ap["prep"] / "calib_tasks.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(manifest["calib_tasks"]) + "\n")
    D.log(f"[prep] records={stats['kept_rows']} splits={stats['by_split']} calib_tasks={mst['calib_tasks']} "
          f"main_tasks={mst['main_tasks']} folds={mst['fold_sizes']}", log_fh)
    D.log(f"[prep] 去重：exact={stats['dup_exact_sha']} norm={stats['norm_sha_collisions']}（保留先序）", log_fh)
    return {"records": len(recs), "stats": stats, "meta": mst}


def cmd_tokens(cfg, work_root=None, force=False, log_fh=None):
    wp = work_paths(cfg, work_root)
    recs = D.load_records_file(wp["records"])
    lens = compute_token_lens(cfg, wp, recs, force=force, log_fh=log_fh)
    edges = cfg["eval"]["buckets"]
    names = D.BUCKET_NAMES
    stats = {}
    for split in ("all", "train", "dev", "test"):
        cnt = {nm: 0 for nm in names}
        for i, r in enumerate(recs):
            if split == "all" or r["split"] == split:
                cnt[names[D.bucket_of(int(lens[i]), edges)]] += 1
        stats[split] = cnt
    D.log(f"[tokens] all={stats['all']}", log_fh)
    if work_root is None:  # 正式运行：记录进 artifacts
        tp = art_paths(cfg)["prep"] / "token_buckets.json"
        tp.parent.mkdir(parents=True, exist_ok=True)
        with open(tp, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        D.log(f"[tokens] 记录 → {tp}", log_fh)
    return stats


def cmd_smoke(cfg, log_fh=None):
    """短流程验证：小样本全链路（prep→tokens→s1→feat→s2）。"""
    work = D.resolve(cfg["paths"]["work"]) / "smoke"
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["data"]["limit_tasks"] = 8
    cfg2["data"]["calib_frac"] = 0.2
    D.log("[smoke] prep (limit_tasks=8)", log_fh)
    cmd_prep(cfg2, work_root=work, limit_tasks=8, log_fh=log_fh)
    cmd_tokens(cfg2, work_root=work, force=True, log_fh=log_fh)
    for task in ("F", "D"):
        D.log(f"[smoke] s1 {task} B1", log_fh)
        train_s1(cfg2, task, "B1", 0, work_root=work, epochs=1, max_steps=4, micro_batch=2, accum=1, log_fh=log_fh, tag="smoke")
        D.log(f"[smoke] feat {task}", log_fh)
        build_features(cfg2, task, 0, work_root=work, force=True, log_fh=log_fh)
        arms = ["ours", "b2", "b3", "mech_r", "mech_ref"] if task == "F" else ["ours", "mech_signed", "mech_mag"]
        D.log(f"[smoke] s2 {task} {arms}", log_fh)
        train_s2(cfg2, task, 0, arms, work_root=work, epochs=1, max_steps=3, log_fh=log_fh, tag="smoke")
    D.log("[smoke] 全链路通过", log_fh)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="d-det-v2 v0.1 训练与准备")
    ap.add_argument("cmd", choices=["prep", "tokens", "s1", "feat", "s2", "smoke"])
    ap.add_argument("--task", choices=["F", "D"])
    ap.add_argument("--arm")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--micro-batch", type=int)
    ap.add_argument("--accum", type=int)
    ap.add_argument("--limit-tasks", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = D.load_cfg()
    wp = work_paths(cfg)
    wp["root"].mkdir(parents=True, exist_ok=True)
    log_fh = get_logger(wp["runs"] / f"{args.cmd}_s{args.seed}.log.txt")

    if args.cmd == "prep":
        cmd_prep(cfg, limit_tasks=args.limit_tasks, log_fh=log_fh)
    elif args.cmd == "tokens":
        cmd_tokens(cfg, force=args.force, log_fh=log_fh)
    elif args.cmd == "s1":
        info = train_s1(cfg, args.task, args.arm, args.seed, epochs=args.epochs,
                        max_steps=args.max_steps, micro_batch=args.micro_batch, accum=args.accum, log_fh=log_fh)
        apath = art_paths(cfg)["runs"] / f"{args.task}_{args.arm}_s{args.seed}_stage1.json"
        apath.parent.mkdir(parents=True, exist_ok=True)
        with open(apath, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=1)
        D.log(f"[s1] 记录 → {apath}", log_fh)
    elif args.cmd == "feat":
        build_features(cfg, args.task, args.seed, force=args.force, log_fh=log_fh)
    elif args.cmd == "s2":
        arms = args.arms or (F_ARMS if args.task == "F" else D_ARMS)
        infos = train_s2(cfg, args.task, args.seed, arms, epochs=args.epochs,
                         max_steps=args.max_steps, log_fh=log_fh)
        apath = art_paths(cfg)["runs"] / f"{args.task}_stage2_s{args.seed}.json"
        apath.parent.mkdir(parents=True, exist_ok=True)
        with open(apath, "w", encoding="utf-8") as f:
            json.dump(infos, f, ensure_ascii=False, indent=1)
        D.log(f"[s2] 记录 → {apath}", log_fh)
    elif args.cmd == "smoke":
        cmd_smoke(cfg, log_fh=log_fh)

    D.log(f"[done] {args.cmd} 用时 {time.time() - T0:.0f}s", log_fh)


if __name__ == "__main__":
    main()
