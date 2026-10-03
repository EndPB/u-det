#!/usr/bin/env python
"""Round2 Q3：仅在现有 CodeT5 资产上测试可训练表示（任务级划分；bf16；小 batch + 梯度累积）。

方案：
  head_only（参照，复用 round1）：冻结 encoder mean-pool 768d → MLP/线性 head；
  last_block：解冻 encoder.block[-1] + final_layer_norm + 新 head（mean-pool）；
  lora      ：peft LoRA(r=8, alpha=16, q/v) + 新 head（主体冻结）。
流程：先冒烟（--smoke，1 seed/3ep）→ 择优 → 3 seed 复现（--scheme name）。
输出：artifacts/acl_dcan_round2/trainable/{metrics.json, config.json, predictions.npz}
checkpoint：仅最佳方案最佳 seed 的可训练参数 → runs/acl_dcan_round2/best_<scheme>.pt（不入库）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import dcan_four_models as r1  # noqa: E402

OUT = ROOT / "artifacts" / "acl_dcan_round2" / "trainable"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = ROOT / "runs" / "acl_dcan_round2"
CKPT.mkdir(parents=True, exist_ok=True)
MAXLEN = 512


def tokenize_rows(rows):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "checkpoints/codet5-base"))
    ids = []
    for r in rows:
        t = tok(r["code"], add_special_tokens=False)["input_ids"]
        if len(t) > MAXLEN:
            t = t[:384] + t[-128:]
        ids.append(np.array(t, dtype=np.int32))
    return ids, tok.pad_token_id


class FTModel(nn.Module):
    def __init__(self, scheme: str):
        super().__init__()
        from encoders.codet5 import CodeT5Encoder
        self.scheme = scheme
        self.enc = CodeT5Encoder(path=str(ROOT / "checkpoints/codet5-base"), dtype="float32",
                                 freeze=True, max_length=MAXLEN)
        if scheme == "last_block":
            for p in self.enc.model.encoder.block[-1].parameters():
                p.requires_grad = True
            for p in self.enc.model.encoder.final_layer_norm.parameters():
                p.requires_grad = True
        elif scheme == "lora":
            from peft import LoraConfig, TaskType, get_peft_model
            cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q", "v"],
                             task_type=TaskType.FEATURE_EXTRACTION)
            self.enc.model = get_peft_model(self.enc.model, cfg)
        self.head = nn.Linear(768, 6)

    def forward(self, ids, mask):
        if self.scheme == "lora":
            out = self.enc.model(input_ids=ids, attention_mask=mask)
            h = getattr(out, "last_hidden_state", None)
            if h is None:
                h = out[0]
        else:
            h = self.enc(ids, attention_mask=mask)
        h = h.float() if h.dtype != torch.float32 else h
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled)


def collate(batch_ids, pad_id, device):
    L = max(len(x) for x in batch_ids)
    arr = np.zeros((len(batch_ids), L), dtype=np.int64)
    for i, x in enumerate(batch_ids):
        arr[i, :len(x)] = x
    ids = torch.tensor(arr, device=device)
    mask = (ids != pad_id).long()
    return ids, mask


def evaluate(model, ids_list, y, idx, device, bs=32):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            chunk = [ids_list[j] for j in idx[i:i + bs]]
            ids, mask = collate(chunk, 0, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask)
            probs.append(F.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(probs)


def train_scheme(scheme, seed, rows, ids_list, pad_id, args):
    import sklearn.metrics as sm
    device = "cuda"
    torch.manual_seed(seed); np.random.seed(seed)
    fam_idx = {f: i for i, f in enumerate(r1.FAMILIES)}
    y = np.array([fam_idx[r["family"]] for r in rows])
    split = [r["task_split"] for r in rows]
    tr = np.array([i for i, s in enumerate(split) if s == "train"])
    dv = np.array([i for i, s in enumerate(split) if s == "dev"])
    te = np.array([i for i, s in enumerate(split) if s == "test"])
    model = FTModel(scheme).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head")]
    lr_enc = 2e-5 if scheme == "last_block" else 3e-4
    opt = torch.optim.AdamW([{"params": enc_params, "lr": lr_enc},
                             {"params": head_params, "lr": 1e-3}], weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats()
    best = {"dev": -1, "epoch": 0, "state": None}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        rng = np.random.RandomState(seed * 100 + ep)
        order = rng.permutation(len(tr))
        opt.zero_grad()
        for step, i in enumerate(range(0, len(order), args.bs)):
            idxs = tr[order[i:i + args.bs]]
            chunk = [ids_list[j] for j in idxs]
            ids, mask = collate(chunk, pad_id, device)
            yb = torch.tensor(y[idxs], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, mask)
                loss = F.cross_entropy(logits.float(), yb) / args.accum
            loss.backward()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
        dv_probs = evaluate(model, ids_list, y, dv, device)
        f1 = float(sm.f1_score(y[dv], dv_probs.argmax(1), average="macro", zero_division=0))
        print(f"[q3] {scheme} s{seed} ep{ep}: dev {f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if f1 > best["dev"]:
            best = {"dev": f1, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                              if k in trainable_names}}
        if ep - best["epoch"] >= 3:
            break
    for k, v in model.state_dict().items():
        if k in best["state"]:
            v.copy_(best["state"][k])
    te_probs = evaluate(model, ids_list, y, te, device)
    m = r1.metric_from_probs(te_probs.astype(np.float64), y[te],
                             {"generator": [rows[i]["model_name"] for i in te]})
    vram = torch.cuda.max_memory_allocated() / 1e6
    return {"scheme": scheme, "seed": seed, "best_dev_f1": best["dev"], "best_epoch": best["epoch"],
            "trainable_params": int(n_train), "max_vram_mb": round(vram, 1),
            "wall_sec": round(time.time() - t0, 1),
            "test": m, "te_probs": te_probs.astype(np.float16), "state": best["state"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["last_block", "lora", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = 1, 3
    t0 = time.time()
    rows = r1.load_rows()
    ids_list, pad_id = tokenize_rows(rows)
    schemes = ["last_block", "lora"] if args.scheme == "both" else [args.scheme]
    out = {"config": {"script": "scripts/dcan_round2_trainable.py", "schemes": schemes,
                      "seeds": args.seeds, "epochs": args.epochs, "bs": args.bs, "accum": args.accum,
                      "encoder": "checkpoints/codet5-base（未微调基座；last_block=block[-1]+final_layer_norm；lora=r8/q,v）",
                      "head_only_ref": "round1 semantic_only（冻结 mean-pool+MLP/SupCon+.688）与 Q1 codeT5_raw LR 探针",
                      "amp": "bf16 autocast；任务的 test 不参与任何选择"},
           "runs": []}
    pack = {}
    best_overall = {"dev": -1}
    for scheme in schemes:
        for seed in range(args.seeds):
            r = train_scheme(scheme, seed, rows, ids_list, pad_id, args)
            test_short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["test"].items()
                          if k in ("macro_f1", "balanced_acc", "ece_top1_15")}
            print(f"[q3] {scheme} s{seed}: {test_short}", flush=True)
            pack[f"{scheme}_s{seed}_probs"] = r.pop("te_probs")
            state = r.pop("state")
            if r["best_dev_f1"] > best_overall["dev"]:
                best_overall = {"dev": r["best_dev_f1"], "scheme": scheme, "seed": seed, "state": state}
            out["runs"].append(r)
    if args.smoke:
        # 冒烟：不落 checkpoint
        pass
    elif best_overall["state"]:
        torch.save({"scheme": best_overall["scheme"], "seed": best_overall["seed"],
                    "state": best_overall["state"]}, CKPT / f"best_{best_overall['scheme']}.pt")
        print(f"[q3] saved ckpt runs/acl_dcan_round2/best_{best_overall['scheme']}.pt", flush=True)
    np.savez_compressed(OUT / "predictions.npz", **pack)
    out["env"] = {"wall_sec": round(time.time() - t0, 1)}
    (OUT / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (OUT / "config.json").write_text(json.dumps(out["config"], ensure_ascii=False, indent=1))
    print(f"[q3] done {round(time.time()-t0,1)}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
