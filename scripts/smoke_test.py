#!/usr/bin/env python
"""端到端冒烟测试：编码器 + U-Net 前向，验证环境与代码是否就绪（CPU 可跑）。

    python scripts/smoke_test.py            # 使用 checkpoints/codet5-base（若已下载）
    python scripts/smoke_test.py --tiny     # 使用随机初始化的迷你 T5，无需权重

输出示例：
    tokenizer: vocab=32100
    encoder  : hidden=768, layers=12, params=...
    unet1d   : logits=(1, 2, L)  params=...
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from udet.encoders import build_encoder, list_encoders  # noqa: E402
from udet.models import build_unet, list_unets  # noqa: E402
from udet.utils import count_parameters, human_format, load_config, set_seed  # noqa: E402

SAMPLE_CODE = '''def add(a, b):
    return a + b
'''


def make_tiny_codet5_dir() -> str:
    """构造一个随机初始化的迷你 T5 编码器目录（用于无权重时的冒烟测试）。"""
    from transformers import T5Config, T5EncoderModel

    cfg = T5Config(
        vocab_size=1000,
        d_model=64,
        d_ff=128,
        d_kv=16,
        num_layers=2,
        num_heads=4,
        decoder_start_token_id=0,
        pad_token_id=0,
    )
    tmpdir = tempfile.mkdtemp(prefix="tiny-codet5-")
    T5EncoderModel(cfg).save_pretrained(tmpdir)
    return tmpdir


def main() -> int:
    parser = argparse.ArgumentParser(description="U-Det 冒烟测试")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "udet_base.yaml"))
    parser.add_argument("--tiny", action="store_true", help="使用迷你随机权重，跳过真实权重")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    print(f"已注册编码器: {list_encoders()}")
    print(f"已注册网络  : {list_unets()}")

    cfg = load_config(args.config)
    enc_cfg = dict(cfg["encoder"])
    model_cfg = dict(cfg["model"])

    # ---- 编码器 ----
    tiny_dir = None
    if args.tiny or not Path(enc_cfg["model_name_or_path"]).exists():
        if not args.tiny:
            print(f"[提示] 未找到本地权重 {enc_cfg['model_name_or_path']}，自动切换到迷你权重。")
            print("       真实权重下载: python scripts/download_encoder.py")
        tiny_dir = make_tiny_codet5_dir()
        enc_cfg["model_name_or_path"] = tiny_dir
        enc_cfg["torch_dtype"] = "float32"

    encoder = build_encoder(enc_cfg).to(device).eval()
    print(
        f"encoder  : {enc_cfg['name']} hidden={encoder.hidden_size} layers={encoder.num_layers} "
        f"params={human_format(count_parameters(encoder))}"
    )

    # ---- 分词 ----
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(enc_cfg["model_name_or_path"], local_files_only=True)
    except Exception:  # noqa: BLE001 - 迷你模型没有 tokenizer，退化为字符级伪分词
        print("[提示] 未找到 tokenizer，使用字符级伪编码进行形状验证。")
        tokenizer = None

    if tokenizer is not None:
        batch = tokenizer(SAMPLE_CODE, return_tensors="pt", truncation=True, max_length=cfg["data"]["max_length"])
        print(f"tokenizer: vocab={tokenizer.vocab_size}, tokens={batch['input_ids'].shape[-1]}")
    else:
        ids = torch.randint(0, 100, (1, 32))
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    batch = {k: v.to(device) for k, v in batch.items()}

    # ---- 编码器前向 ----
    with torch.no_grad():
        enc_out = encoder(**batch)
    print(
        f"encoder 前向: last_hidden_state={tuple(enc_out.last_hidden_state.shape)} "
        f"hidden_states={len(enc_out.hidden_states) if enc_out.hidden_states else 0} 层 "
        f"pooled={tuple(enc_out.pooled_output.shape)}"
    )

    # ---- U-Net 前向 ----
    model_cfg = {**model_cfg, "in_channels": encoder.hidden_size}  # 保证通道匹配
    unet = build_unet(model_cfg).to(device).eval()
    feats = enc_out.last_hidden_state.transpose(1, 2)  # (B, D, L)
    with torch.no_grad():
        logits = unet(feats, mask=batch.get("attention_mask"))
    print(
        f"{model_cfg['name']:>8} : logits={tuple(logits.shape)} "
        f"params={human_format(count_parameters(unet))}"
    )

    # ---- 2D U-Net 形状自检 ----
    unet2d = build_unet("unet2d", in_channels=3, out_channels=2, features=[16, 32, 64]).to(device).eval()
    with torch.no_grad():
        out2d = unet2d(torch.randn(2, 3, 65, 37, device=device))
    print(f"  unet2d : logits={tuple(out2d.shape)}（奇数尺寸输入已自动对齐）")

    print("冒烟测试通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
