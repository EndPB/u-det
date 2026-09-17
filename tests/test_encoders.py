"""编码器单元测试：注册表、CodeT5 封装、通用 HF 封装（使用迷你随机权重，无需下载）。"""

from __future__ import annotations

import pytest
import torch

from udet.encoders import build_encoder, list_encoders
from udet.encoders.codet5 import CodeT5Encoder
from udet.encoders.hf_encoder import HuggingFaceEncoder


# --------------------------------------------------------------------------- #
# 迷你模型 fixture
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_t5_dir(tmp_path_factory) -> str:
    from transformers import T5Config, T5EncoderModel

    path = tmp_path_factory.mktemp("tiny-t5")
    cfg = T5Config(
        vocab_size=128, d_model=32, d_ff=64, d_kv=8, num_layers=3, num_heads=4,
        pad_token_id=0, decoder_start_token_id=0,
    )
    T5EncoderModel(cfg).save_pretrained(str(path))
    return str(path)


@pytest.fixture(scope="module")
def tiny_bert_dir(tmp_path_factory) -> str:
    from transformers import BertConfig, BertModel

    path = tmp_path_factory.mktemp("tiny-bert")
    cfg = BertConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64)
    BertModel(cfg).save_pretrained(str(path))
    return str(path)


def _batch(batch_size: int = 2, length: int = 16) -> dict:
    input_ids = torch.randint(1, 100, (batch_size, length))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, length // 2 :] = 0
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
def test_list_encoders():
    names = list_encoders()
    assert "codet5" in names and "hf" in names


def test_build_encoder_from_dict(tiny_t5_dir):
    enc = build_encoder({"name": "codet5", "model_name_or_path": tiny_t5_dir, "torch_dtype": "float32"})
    assert isinstance(enc, CodeT5Encoder)
    assert enc.hidden_size == 32
    assert enc.num_layers == 3


def test_build_encoder_unknown_name():
    with pytest.raises(KeyError):
        build_encoder({"name": "not-exist"})


# --------------------------------------------------------------------------- #
# CodeT5 封装
# --------------------------------------------------------------------------- #
def test_codet5_forward(tiny_t5_dir):
    enc = CodeT5Encoder(tiny_t5_dir).eval()
    out = enc(**_batch())
    assert out.last_hidden_state.shape == (2, 16, 32)
    assert out.pooled_output.shape == (2, 32)
    assert out.attention_mask.shape == (2, 16)
    assert torch.isfinite(out.last_hidden_state).all()


def test_codet5_layer_selection(tiny_t5_dir):
    all_enc = CodeT5Encoder(tiny_t5_dir, layer_selection="all").eval()
    out = all_enc(**_batch())
    assert len(out.hidden_states) == all_enc.num_layers + 1  # 含嵌入层

    subset = CodeT5Encoder(tiny_t5_dir, layer_selection=[0, 2]).eval()
    out2 = subset(**_batch())
    assert len(out2.hidden_states) == 2
    assert out2.last_hidden_state.shape == (2, 16, 32)

    with pytest.raises(IndexError):
        CodeT5Encoder(tiny_t5_dir, layer_selection=[99]).eval()(**_batch())


def test_codet5_no_hidden_states(tiny_t5_dir):
    enc = CodeT5Encoder(tiny_t5_dir, output_hidden_states=False).eval()
    out = enc(**_batch())
    assert out.hidden_states is None
    assert out.last_hidden_state.shape == (2, 16, 32)


def test_codet5_freeze(tiny_t5_dir):
    enc = CodeT5Encoder(tiny_t5_dir, freeze=True)
    assert enc.num_trainable_parameters() == 0

    enc2 = CodeT5Encoder(tiny_t5_dir, freeze_layers=1)
    trainable = enc2.num_trainable_parameters()
    assert 0 < trainable < enc2.num_parameters()


def test_codet5_backward(tiny_t5_dir):
    enc = CodeT5Encoder(tiny_t5_dir).train()
    out = enc(**_batch())
    loss = out.last_hidden_state.mean()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert torch.isfinite(torch.stack([g.float().abs().sum() for g in grads])).all()


def test_codet5_pooling_max(tiny_t5_dir):
    enc = CodeT5Encoder(tiny_t5_dir, pooling="max").eval()
    out = enc(**_batch())
    assert out.pooled_output.shape == (2, 32)


def test_codet5_inputs_embeds(tiny_t5_dir):
    enc = CodeT5Encoder(tiny_t5_dir).eval()
    emb_layer = enc.get_input_embeddings()
    ids = torch.randint(1, 100, (1, 10))
    with torch.no_grad():
        embeds = emb_layer(ids)
        out = enc(inputs_embeds=embeds, attention_mask=torch.ones_like(ids))
    assert out.last_hidden_state.shape == (1, 10, 32)


# --------------------------------------------------------------------------- #
# 通用 HF 封装
# --------------------------------------------------------------------------- #
def test_hf_encoder(tiny_bert_dir):
    enc = build_encoder("hf", model_name_or_path=tiny_bert_dir)
    assert isinstance(enc, HuggingFaceEncoder)
    assert enc.hidden_size == 32
    out = enc(**_batch())
    assert out.last_hidden_state.shape == (2, 16, 32)
    assert out.pooled_output.shape == (2, 32)


def test_hf_encoder_freeze(tiny_bert_dir):
    enc = build_encoder("hf", model_name_or_path=tiny_bert_dir, freeze=True)
    assert enc.num_trainable_parameters() == 0
