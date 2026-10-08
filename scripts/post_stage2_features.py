"""post-stage2 Phase A：变换后文本的特征提取（style/meta/size-length + CodeT5 small/base 冻结嵌入）。

依据《d-det_AutoDL_后续干预与主干探针指导_2026-10-08》§3-§5：
- 与 stage-1 完全同式特征（style regex stylometry、meta 布局子集、size-length、CodeT5 冻结 mean-pool 512=384+128 fp16）；
- 每变换（A1/A2/A3）输出独立特征缓存（本地，不进 git），并记录 sha 到 feature_manifest；
- A0 原始对照直接复用 stage-1 缓存（features/），本脚本不重复生成。

输出：local/features/{tid}/{style_meta.npz, emb_codet5_small.npz, emb_codet5_base.npz, feature_manifest.json}
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import variant_transfer_stage1_execute as s1  # noqa: E402

OUT = ROOT / "d-det/artifacts/post_stage2_intervention_2026-10-08"
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


def load_transformed(tid: str) -> list[str]:
    p = OUT / "local/transformed_texts" / f"{tid}.jsonl.gz"
    texts = []
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", required=True)
    ap.add_argument("--no-emb", action="store_true")
    args = ap.parse_args()
    tid = args.transform
    t0 = time.time()
    feat_dir = OUT / "local/features" / tid
    feat_dir.mkdir(parents=True, exist_ok=True)

    log("[1] samples + transformed texts")
    split_of = s1.load_split()
    samples = s1.load_samples(split_of)
    texts = load_transformed(tid)
    assert len(texts) == len(samples) == 10659, (len(texts), len(samples))

    man = {"schema": "post_stage2_feature_manifest_v1", "transform_id": tid,
           "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_samples": len(texts), "order": "(unit, task) ascending (stage-1 identical)",
           "protocol": "style=regex stylometry (stage-1 identical); emb=codet5 frozen mean-pool 512=384+128 no-special fp16"}

    log("[2] style/meta/size-length (CPU)")
    t = time.time()
    style_mat = np.array([s1.style_features(tx) for tx in texts], dtype=np.float64)
    meta_mat = style_mat[:, s1.META_IDX].copy()
    szfeat = np.array([[np.log10(s1.PARAM_B[r["unit"]]), style_mat[i, 0], style_mat[i, 1]]
                       for i, r in enumerate(samples)], dtype=np.float64)
    np.savez_compressed(feat_dir / "style_meta.npz", style=style_mat.astype(np.float32),
                        meta=meta_mat.astype(np.float32), sizelen=szfeat.astype(np.float32))
    log(f"  style {style_mat.shape} in {time.time()-t:.1f}s")
    man["style_meta"] = {"sha256": sha256_file(feat_dir / "style_meta.npz")}

    if not args.no_emb:
        log("[3] CodeT5-small embedding (GPU)")
        t = time.time()
        emb_small = s1.encode_codet5(s1.MODEL_SMALL, texts, device_batch=16)
        np.savez_compressed(feat_dir / "emb_codet5_small.npz", emb=emb_small)
        log(f"  small {emb_small.shape} in {time.time()-t:.1f}s")
        log("[4] CodeT5-base embedding (GPU)")
        t = time.time()
        emb_base = s1.encode_codet5(s1.MODEL_BASE, texts, device_batch=8)
        np.savez_compressed(feat_dir / "emb_codet5_base.npz", emb=emb_base)
        log(f"  base {emb_base.shape} in {time.time()-t:.1f}s")
        man["emb_small"] = {"shape": list(emb_small.shape), "sha256": sha256_file(feat_dir / "emb_codet5_small.npz")}
        man["emb_base"] = {"shape": list(emb_base.shape), "sha256": sha256_file(feat_dir / "emb_codet5_base.npz")}

    (feat_dir / "feature_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    (OUT / "logs" / f"features_{tid}.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
