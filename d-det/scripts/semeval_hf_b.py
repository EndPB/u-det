#!/usr/bin/env python
"""在 HF Preprocessed-SemEval-2026-Task13（官方 B test 500K 重打包，标签已扣留）上运行 B 臂模型。

流程：4 分片全量读取语言列 → 按语言比例抽样（总量 N，稀有语言保底）→ 分词（头 768 + 尾 256）
→ runs/semeval_b 模型预测 → 输出预测分布并与官方真实构成对比（distribution-level 验证）。

输出：runs/semeval_hf_b/preds.parquet + summary.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import build_encoder  # noqa: E402
from models import build_model  # noqa: E402
from semeval_finetune import TaskModel  # noqa: E402

HF = ROOT / "data/raw/hf_preprocessed"
OUT = ROOT / "runs/semeval_hf_b"

FAMILIES = {0: "Human", 1: "DeepSeek", 2: "Qwen", 3: "Yi", 4: "StarCoder",
            5: "Gemma", 6: "Phi", 7: "Llama", 8: "IBM-Granite", 9: "Mistral", 10: "GPT"}
# 官方 B test 真实构成（semeval.md；500K）
TRUE_COUNTS = {0: 243769, 1: 9674, 2: 26459, 3: 8471, 4: 3631, 5: 21913,
               6: 20981, 7: 35411, 8: 11202, 9: 14185, 10: 104304}


class CodeList(Dataset):
    def __init__(self, codes: list, dummy: int = 0):
        self.codes, self.dummy = codes, dummy

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        return {"code": self.codes[i], "label": self.dummy}


def make_collate(tok, head=768, tail=256, max_t=1024):
    def collate(batch):
        ids_list = []
        for b in batch:
            ids = tok(b["code"], add_special_tokens=False)["input_ids"]
            if len(ids) > max_t:
                ids = ids[:head] + ids[-tail:]
            if not ids:
                ids = [0]
            ids_list.append(ids)
        length = max(len(x) for x in ids_list)
        ids = torch.zeros(len(ids_list), length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, x in enumerate(ids_list):
            ids[r, :len(x)] = torch.tensor(x, dtype=torch.long)
            mask[r, :len(x)] = 1
        return {"input_ids": ids, "attention_mask": mask}
    return collate


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--min-per-lang", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ckpt", default="runs/semeval_b/best.pt")
    ap.add_argument("--config", default="configs/ddet_v041.yaml")
    args = ap.parse_args()

    # ---------- 1) 语言分布（全 500K）与抽样 ----------
    shard_langs = []
    for i in range(4):
        t = pq.read_table(HF / f"test-{i}.parquet", columns=["language"])
        shard_langs.append(t.column("language").to_pylist())
    lang_counts = Counter(lg for ls in shard_langs for lg in ls)
    total = sum(lang_counts.values())
    targets = {lg: min(max(args.min_per_lang, round(args.n * c / total)), c)
               for lg, c in lang_counts.items()}
    print(f"[sample] 语言分布={dict(lang_counts.most_common())} → targets={targets}", flush=True)

    rng = random.Random(args.seed)
    sel = {i: set() for i in range(4)}          # shard -> 选中的行位置
    for lg, want in targets.items():
        cand = [(i, pos) for i, ls in enumerate(shard_langs)
                for pos, x in enumerate(ls) if x == lg]
        pick = rng.sample(cand, min(want, len(cand)))
        for i, pos in pick:
            sel[i].add(pos)

    rows = []
    for i in range(4):
        t = pq.read_table(HF / f"test-{i}.parquet", columns=["ID", "language", "code"])
        ids = t.column("ID").to_pylist()
        langs = t.column("language").to_pylist()
        codes = t.column("code").to_pylist()
        for pos in sorted(sel[i]):
            rows.append({"ID": ids[pos], "language": langs[pos], "code": codes[pos]})
    print(f"[sample] 共抽取 {len(rows)} 行（语言={dict(Counter(r['language'] for r in rows).most_common())}）", flush=True)

    # ---------- 2) 模型 ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enc_cfg = dict(cfg["encoder"])
    enc_cfg["path"] = str(ROOT / enc_cfg["path"])
    enc = build_encoder(**enc_cfg)
    dual = build_model("dual", encoder=enc, dim=enc.hidden_size,
                       pool=cfg["model"].get("pool", "mean"),
                       s2_rank=cfg["model"].get("s2_rank", 1))
    model = TaskModel(dual, "b", 11).to(device)
    ck = torch.load(str(ROOT / args.ckpt), map_location="cpu", weights_only=False)
    miss, unexp = model.load_state_dict(ck["state"], strict=False)
    print(f"[ckpt] {args.ckpt}（epoch {ck.get('epoch')}）missing={len(miss)} unexpected={len(unexp)}", flush=True)
    model.eval()

    tok = AutoTokenizer.from_pretrained(enc_cfg["path"])
    loader = DataLoader(CodeList([r["code"] for r in rows]), batch_size=args.batch,
                        collate_fn=make_collate(tok), num_workers=0)
    probs = []
    with torch.no_grad():
        for b in tqdm(loader, desc="predict", unit="it"):
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model.head(model.dual.features(ids, mask))
            probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
    probs = np.concatenate(probs, 0)
    preds = probs.argmax(1)

    # ---------- 3) 分布统计与对比 ----------
    pred_counts = Counter(preds.tolist())
    tn = sum(TRUE_COUNTS.values())
    per_class = {}
    tv = 0.0
    print("\n== 预测分布 vs 官方真实构成（B test 500K）==")
    print(f"{'class':<14}{'pred%':>8}{'true%':>8}{'diff':>8}")
    for c in range(11):
        p = pred_counts.get(c, 0) / len(preds)
        t_ = TRUE_COUNTS[c] / tn
        tv += abs(p - t_)
        per_class[FAMILIES[c]] = {"pred_n": int(pred_counts.get(c, 0)),
                                  "pred_pct": round(100 * p, 2),
                                  "true_pct": round(100 * t_, 2),
                                  "diff_pct": round(100 * (p - t_), 2)}
        print(f"{FAMILIES[c]:<14}{100*p:>7.2f}%{100*t_:>7.2f}%{100*(p-t_):>+7.2f}%")
    print(f"总变差 TV = {tv/2:.4f}")

    by_lang = {}
    for lg in sorted(set(r["language"] for r in rows)):
        m = np.array([r["language"] == lg for r in rows])
        cc = Counter(preds[m].tolist())
        top = {FAMILIES[c]: round(cnt / m.sum(), 3) for c, cnt in cc.most_common(4)}
        by_lang[lg] = {"n": int(m.sum()), "top_pred": top}

    OUT.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "ID": [r["ID"] for r in rows],
        "language": [r["language"] for r in rows],
        "pred": preds.astype("int16"),
        "prob1": probs.max(1).astype("float32"),
    }), OUT / "preds.parquet")
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "n": len(rows), "seed": args.seed,
                   "per_class": per_class, "tv": round(tv / 2, 4),
                   "by_language": by_lang}, f, ensure_ascii=False, indent=2)
    print(f"[out] {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
