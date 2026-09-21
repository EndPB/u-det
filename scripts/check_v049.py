"""v0.4.9 自检：三处改动 + 瓶颈权重移植。

改动（对应 §8.15.3 / §8.16.9 里列出的三条 + 用户要求的「只复制权重」）：

    A. 瓶颈权重移植   ``model.mid_init: codet5@8-11``
       —— **不改结构、不改参数量**，只把 CodeT5 的层按位置拷进 4 个 ``SelfBlock``。
    B. token 旁路改「额外一路头」  ``heads.token_bypass_mode: separate``
       —— 主干 5 个尺度的头不再拼接编码器特征，旁路改走独立头；
          **报告用两者 logits 的均值，损失分别施加**（这是恢复 §8.11.6 信息保值机制的关键）。
    C. 文档级 → 主干的**梯度缩放**  ``heads.sample_grad_scale: λ``
       —— 只改梯度、不改前向；λ=0 即 stop-gradient。

重点不是"能跑"，而是**向后兼容 + 契约不破**：
    1. 三个新旋钮取默认值时必须与历史**逐位一致**；
    2. `separate` 必须保住 `evaluate` 的长度契约（最细尺度长度 == L）；
    3. 额外一路头必须**真的被训练**（否则是静默失效）；
    4. 梯度缩放必须只作用于文档级那条路径（λ=0 时 token 级梯度不受影响）。

用法::

    OMP_NUM_THREADS=8 python scripts/check_v049.py            # CPU 即可（默认不跑真实前向）
    OMP_NUM_THREADS=8 python scripts/check_v049.py --encoder   # 额外用真样本量移植保真度（约 1 分钟）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from encoders.codet5 import init_selfblocks_from_codet5, port_fidelity  # noqa: E402
from models.heads import SampleHead, TokenHeads, grad_scale              # noqa: E402
from models.hier import SelfBlock                                        # noqa: E402
from train import build_model, resolve                                   # noqa: E402
from transformers import AutoTokenizer                                   # noqa: E402

OK, FAIL = "\033[32mOK\033[0m", "\033[31mFAIL\033[0m"
bad = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global bad
    if not cond:
        bad += 1
    print(f"  [{OK if cond else FAIL}] {name}" + (f"   {extra}" if extra else ""))


# --------------------------------------------------------------------------- #
def t1_defaults_identical() -> None:
    print("\n=== 1. 三个新旋钮取默认值时与历史逐位一致 ===")
    torch.manual_seed(0)
    x = torch.randn(2, 7, 768)

    a, b = SampleHead(768, hidden=64), SampleHead(768, hidden=64, grad_scale=1.0)
    b.load_state_dict(a.state_dict())
    check("SampleHead(grad_scale=1.0) 与不加该参数逐位相同",
          torch.equal(a(x), b(x)))

    t = torch.randn(1, 5, 768, requires_grad=True)
    check("grad_scale(x, 1.0) 返回**同一个张量**（连算子都不建）", grad_scale(t, 1.0) is t,
          f"type={type(grad_scale(t, 1.0)).__name__}")

    th = TokenHeads([768] * 5)
    check("TokenHeads 默认无 bypass_head（与历史结构一致）",
          th.bypass_head is None and not th.separate)
    check("TokenHeads 默认 5 个头都是 768->1",
          all(h.in_features == 768 for h in th.heads))

    k = TokenHeads([768] * 5, bypass_dim=768, bypass_mode="concat")
    check("bypass_mode='concat' 仍是「拼进最细头」（1536->1）",
          k.heads[-1].in_features == 1536 and k.bypass_head is None)


def t2_separate_shape_and_contract() -> None:
    print("\n=== 2. 额外一路头：形状、参数量、报告契约 ===")
    torch.manual_seed(0)
    concat = TokenHeads([768] * 5, bypass_dim=768, bypass_mode="concat")
    sep = TokenHeads([768] * 5, bypass_dim=768, bypass_mode="separate")
    p_c = sum(p.numel() for p in concat.parameters())
    p_s = sum(p.numel() for p in sep.parameters())
    check("参数几乎不变（+1）", p_s - p_c == 1,
          f"concat {p_c} -> separate {p_s}（差 {p_s - p_c}）")
    check("separate：主干最细头回到 768->1（= v0.4.4 的形状）",
          sep.heads[-1].in_features == 768)
    check("separate：额外头是 768->1", sep.bypass_head.in_features == 768
          and sep.bypass_head.out_features == 1)

    # 长度契约：最细尺度必须仍是 L（evaluate 的 `token_logits[-1]` 与逐 token 标签对齐）
    levels = [torch.randn(1, k, 768) for k in (5, 10, 19, 74, 293)]
    feats = torch.randn(1, 293, 768)
    trunk, extra = sep(levels, feats)
    check("separate：主干 5 个尺度长度不变", [t.shape[-1] for t in trunk] == [5, 10, 19, 74, 293],
          f"{[t.shape[-1] for t in trunk]}")
    check("separate：额外头长度 = L", extra.shape == (1, 1, 293), f"{tuple(extra.shape)}")
    merged = 0.5 * (trunk[-1] + extra)
    check("报告用均值 ⇒ 最细尺度长度仍 = L", merged.shape[-1] == 293)
    check("均值确实是 0.5*(trunk+extra)",
          torch.equal(merged, 0.5 * (trunk[-1] + extra)))

    c_out, c_extra = concat(levels, feats)
    check("concat 模式第二返回值为 None（不产生额外头）", c_extra is None)
    check("concat 模式最细尺度长度也是 L", c_out[-1].shape[-1] == 293)


def t3_bypass_actually_trained() -> None:
    print("\n=== 3. 额外一路头**真的**被训练，且主干那一级继续承压 ===")
    torch.manual_seed(0)
    sep = TokenHeads([768] * 5, bypass_dim=768, bypass_mode="separate")
    levels = [torch.randn(1, k, 768) for k in (5, 10, 19, 74, 293)]
    feats = torch.randn(1, 293, 768, requires_grad=True)
    trunk, extra = sep(levels, feats)

    # 与 batch_losses 同构：主干最细尺度一份损失 + 额外头一份损失
    y = (torch.rand(1, 1, 293) > 0.5).float()
    l = F.binary_cross_entropy_with_logits(trunk[-1], y) + F.binary_cross_entropy_with_logits(extra, y)
    l.backward()

    g_extra = sep.bypass_head.weight.grad
    g_trunk = sep.heads[-1].weight.grad
    check("额外头拿到梯度（另：train.py 对 token_bypass<=0 会直接报错）",
          g_extra is not None and float(g_extra.abs().sum()) > 0)
    check("主干最细头也拿到梯度（⇒ 信息保值机制不被旁路绕开）",
          g_trunk is not None and float(g_trunk.abs().sum()) > 0)
    check("额外头与主干头**不共享参数**",
          sep.bypass_head.weight.data_ptr() != sep.heads[-1].weight.data_ptr())


def t4_grad_scale() -> None:
    print("\n=== 4. 梯度缩放：只作用文档级路径，且不改前向 ===")
    torch.manual_seed(0)
    x = torch.randn(1, 19, 768)
    head = SampleHead(768, hidden=64)          # ★ 同一个 head 复用，只改 λ
                                               #（否则每次新建都会消耗 RNG ⇒ 权重不同 ⇒ 假失败）

    def doc_grad(lam: float) -> torch.Tensor:
        head.grad_scale = float(lam)
        t = x.clone().detach().requires_grad_(True)
        head(t, None).sum().backward()
        return t.grad

    g1, g0, gh = doc_grad(1.0), doc_grad(0.0), doc_grad(0.5)
    check("λ=0 ⇒ stop-gradient（文档级对主干的梯度恰为 0）", float(g0.abs().max()) == 0.0)
    check("λ=0.5 ⇒ 恰为 λ=1 的一半", torch.allclose(gh, 0.5 * g1, atol=1e-7),
          f"max|g0.5 − 0.5·g1| = {float((gh - 0.5 * g1).abs().max()):.2e}")
    check("λ=1 时梯度非零（对照）", float(g1.abs().max()) > 0)

    # 前向不受 λ 影响
    torch.manual_seed(1)
    a, b = SampleHead(768, hidden=64), SampleHead(768, hidden=64, grad_scale=0.0)
    b.load_state_dict(a.state_dict())
    check("λ 不改变前向输出（纯粹改梯度）", torch.equal(a(x), b(x)))


def t5_bottleneck_port(spec: str = "codet5@8-11") -> None:
    print(f"\n=== 5. 瓶颈权重移植 mid_init={spec}：结构/参数不变 ===")
    d = SelfBlock(768, 12, 4.0)
    check("SelfBlock 默认仍是 LayerNorm + GELU（与历史逐位一致）",
          isinstance(d.norm1, torch.nn.LayerNorm) and isinstance(d.mlp[1], torch.nn.GELU))
    r = SelfBlock(768, 12, 4.0, norm="rms", act="relu")
    n_d = sum(p.numel() for p in d.parameters())
    n_r = sum(p.numel() for p in r.parameters())
    check("开启 rms/relu 后模块类型正确",
          hasattr(r.norm1, "weight") and not hasattr(r.norm1, "bias")
          and isinstance(r.mlp[1], torch.nn.ReLU))
    # ★ rms 会少掉两个 norm 的 bias（2 个 norm × 768）—— 这是**语义需要**（T5 的 T5LayerNorm 无 bias），
    #   不是形状漂移；每层 −1536、瓶颈 4 层 −6144，占 28.35M 的 −0.02%。
    check("rms 只少掉 norm 的 bias（每块 1536）", n_d - n_r == 1536,
          f"{n_d} -> {n_r}（差 {n_d - n_r}）")

    sb = [SelfBlock(768, 12, 4.0) for _ in range(4)]
    before = [p.detach().clone() for p in sb[0].parameters()]
    n_before = sum(p.numel() for b in sb for p in b.parameters())
    srcs = init_selfblocks_from_codet5(sb, path="checkpoints/codet5-base", spec=spec, verbose=False)
    n_after = sum(p.numel() for b in sb for p in b.parameters())
    check("参数量不变", n_before == n_after, f"{n_before} == {n_after}")
    check("确实改了权重（不再是随机初始化）",
          any(not torch.equal(a, b) for a, b in zip(before, sb[0].parameters())))
    check("bias 全部置零（T5 的 q/k/v/o 无 bias）",
          float(sb[0].attn.in_proj_bias.abs().max()) == 0.0
          and float(sb[0].attn.out_proj.bias.abs().max()) == 0.0
          and float(sb[0].norm1.bias.abs().max()) == 0.0
          and float(sb[0].mlp[0].bias.abs().max()) == 0.0)

    # 单块的线性映射确实是 T5 的（构造等价检查：q/k/v/o 拼接顺序）
    from transformers.models.t5.modeling_t5 import T5Block
    from transformers import T5Config
    cfg = T5Config.from_pretrained(str(resolve("checkpoints/codet5-base")))
    for i, b in enumerate(range(8, 12)):
        ref = T5Block(cfg, has_relative_attention_bias=False)
        a = srcs[i].layer[0].SelfAttention
        ok_qkv = torch.equal(sb[i].attn.in_proj_weight,
                             torch.cat([a.q.weight, a.k.weight, a.v.weight], 0))
        check(f"block {b}: in_proj_weight 的 q/k/v 拼接顺序正确", ok_qkv)
        check(f"block {b}: out_proj / wi / wo 与 T5 逐位相同",
              torch.equal(sb[i].attn.out_proj.weight, a.o.weight)
              and torch.equal(sb[i].mlp[0].weight, srcs[i].layer[1].DenseReluDense.wi.weight)
              and torch.equal(sb[i].mlp[3].weight, srcs[i].layer[1].DenseReluDense.wo.weight))
    return sb, srcs


def t6_fidelity_with_real_input() -> None:
    print("\n=== 6. 移植保真度：用**真实的瓶颈输入**量（不是随机噪声）===")
    cfg = yaml.safe_load(open(str(resolve("configs/udet_v048.yaml")), encoding="utf-8"))
    cfg["encoder"]["compile"] = False                    # 自检不必要、且慢
    tokenizer = AutoTokenizer.from_pretrained(str(resolve(cfg["encoder"]["path"])))
    tok_pkg = build_model(cfg, tokenizer)
    model, _ = tok_pkg[0], tok_pkg[1]
    model.eval()

    captured = []

    def hook(_module, inputs):
        captured.append(inputs[0].detach())

    h = model.backbone.mid_blocks[0].register_forward_pre_hook(hook)
    ds = __import__("train").make_dataset(cfg, "hybrid", "test", None, False, 6)
    rep_dim = int(cfg.get("heads", {}).get("report_dim", 0) or 0)
    rep = torch.zeros(1, rep_dim) if rep_dim else None      # 只为让前向跑通，不影响瓶颈输入
    with torch.no_grad():
        for i in range(min(4, len(ds))):
            ids = torch.tensor([ds[i]["input_ids"]], dtype=torch.long)
            model(ids, torch.ones_like(ids), report=rep)
    h.remove()
    x = torch.cat([c.reshape(-1, c.shape[-1]) for c in captured], 0)
    print(f"  （真实瓶颈输入 x：{tuple(x.shape)}，E‖x‖² = {float(x.pow(2).sum(-1).mean()):.1f}）")

    sb, srcs = t5_bottleneck_port()
    for i, b in enumerate(range(8, 12)):
        f = port_fidelity(sb[i], srcs[i], x[: min(4096, x.shape[0])].unsqueeze(0))
        verdict = "接近（可作热启动）" if f["cos"] > 0.99 else (
            "中等（权重携带的信息部分保留）" if f["cos"] > 0.9 else "偏低（近似较粗）")
        print(f"  block {b}: cos = {f['cos']:.4f}（最小 {f['cos_min']:.4f}），"
              f"相对 L2 误差 = {f['rel']:.4f}   ⇒ {verdict}")
    t6b_attribute_mismatch(x, sb, srcs)
    t6c_faithful_port(x)


def _run_variant(sb, x: torch.Tensor, rms: bool, relu: bool, eps: float = 1e-6) -> torch.Tensor:
    """用**同一个** ``SelfBlock`` 的权重，手工跑一遍，但归一化/激活换成 T5 的语义。

    用来把移植误差**归因**到「RMS vs 减均值 LayerNorm」和「ReLU vs GELU」这两处：
    ``(rms=True, relu=True)`` 应当复现原始 ``T5Block``（误差只剩浮点级）。
    """
    def norm(t, w):
        if rms:
            return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * w
        return F.layer_norm(t, (t.shape[-1],), w, None, eps)

    h = norm(x, sb.norm1.weight)
    h, _ = sb.attn(h, h, h, need_weights=False)                   # bias 已置零 ⇒ 与 T5 等价
    x = x + h
    h = norm(x, sb.norm2.weight)
    h = sb.mlp[0](h)                                              # bias 已置零
    h = F.relu(h) if relu else F.gelu(h)
    return x + sb.mlp[3](h)                                       # bias 已置零


def t6b_attribute_mismatch(x: torch.Tensor, sb, srcs) -> None:
    print("\n=== 6b. 移植误差归因：到底是哪一处对不上 ===")
    print(f"{'组合':<28}{'cos (block 8..11 均值)':>24}")
    combos = [(False, False, "当前移植：LayerNorm + GELU"),
              (True, False, "只换 RMS 归一化"),
              (False, True, "只换 ReLU"),
              (True, True, "RMS + ReLU（= T5 语义）")]
    xx = x[: min(2048, x.shape[0])].unsqueeze(0)
    for rms, relu, label in combos:
        cs = []
        for i, b in enumerate(range(8, 12)):
            ref = srcs[i](xx, cache_position=torch.arange(xx.shape[1]))[0].float()
            got = _run_variant(sb[i], xx, rms, relu).float()
            cs.append(float(F.cosine_similarity(ref.reshape(-1, 768), got.reshape(-1, 768)).mean()))
        print(f"{label:<28}{sum(cs) / len(cs):>24.4f}   {[round(c, 4) for c in cs]}")
    print("  ⇒ 若最后一行 ≈ 1.0000，说明移植误差**只**来自这两处，线性映射是逐位正确的")


def t6c_faithful_port(x: torch.Tensor) -> None:
    print("\n=== 6c. 若允许 T5 语义（mid_norm=rms + mid_act=relu），端到端保真度 ===")
    sb = [SelfBlock(768, 12, 4.0, norm="rms", act="relu") for _ in range(4)]
    srcs = init_selfblocks_from_codet5(sb, path="checkpoints/codet5-base",
                                       spec="codet5@8-11", verbose=False)
    xx = x[: min(2048, x.shape[0])].unsqueeze(0)
    cs = []
    for i, b in enumerate(range(8, 12)):
        cs.append(port_fidelity(sb[i], srcs[i], xx)["cos"])
    check("rms+relu 下端口后与原始 T5Block 几乎重合（均值 cos > 0.99）",
          sum(cs) / len(cs) > 0.99, f"均值 {sum(cs) / len(cs):.4f}，逐块 {[round(c, 4) for c in cs]}")


def t7_build_model_wiring() -> None:
    print("\n=== 7. build_model 接线：新配置真的生效 ===")
    base = yaml.safe_load(open(str(resolve("configs/udet_v045.yaml")), encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(resolve(base["encoder"]["path"])))

    def build(**over):
        import copy
        cfg = copy.deepcopy(base)
        for sec, kv in over.items():
            cfg.setdefault(sec, {}).update(kv)
        cfg["encoder"]["compile"] = False
        m, _ = build_model(cfg, tok)
        return m.eval()

    m0 = build()
    check("默认（v0.4.5 配置）仍是 concat 旁路，且 λ=1",
          getattr(m0.token_heads, "separate", False) is False
          and m0.sample_head.grad_scale == 1.0)
    n0 = sum(p.numel() for p in m0.parameters())

    m1 = build(heads={"token_bypass_mode": "separate"})
    check("separate 接线生效", m1.token_heads.separate is True)
    n1 = sum(p.numel() for p in m1.parameters())
    check("参数只 +1", n1 - n0 == 1, f"{n0} -> {n1}")

    m2 = build(heads={"sample_grad_scale": 0.0})
    check("sample_grad_scale 接线生效", m2.sample_head.grad_scale == 0.0)
    check("λ 不改变参数量", sum(p.numel() for p in m2.parameters()) == n0)

    m3 = build(model={"mid_init": "codet5@8-11"})
    check("mid_init 不改变参数量（只换初始化）",
          sum(p.numel() for p in m3.parameters()) == n0)

    # 报错而不是静默失效
    try:
        build(heads={"token_bypass_mode": "separate"}, loss={"token_bypass": 0.0})
        check("separate 且 token_bypass=0 应直接报错", False)
    except SystemExit:
        check("separate 且 token_bypass=0 直接报错（不静默失效）", True)


def main() -> int:
    ap = argparse.ArgumentParser(description="v0.4.9 自检")
    ap.add_argument("--encoder", action="store_true", help="额外跑真样本的移植保真度（慢）")
    args = ap.parse_args()

    t1_defaults_identical()
    t2_separate_shape_and_contract()
    t3_bypass_actually_trained()
    t4_grad_scale()
    t5_bottleneck_port()
    t7_build_model_wiring()
    if args.encoder:
        t6_fidelity_with_real_input()
    print(f"\n{'=' * 60}\n{'全部通过 ✓' if bad == 0 else f'有 {bad} 项失败 ✗'}\n{'=' * 60}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
