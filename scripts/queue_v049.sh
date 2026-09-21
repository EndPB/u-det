#!/usr/bin/env bash
# v0.4.9 ~ v0.4.12：**针对「结构性耦合」的四条单变量实验**（每条 2 epoch）。
#
# 全部以 `configs/udet_v049.yaml` 为底，用 `--set` 逐个只改一个量
# （lessons A7：单变量；用 --set 而不是抄四份配置，避免配置漂移）：
#
#   v0.4.9   额外一路头（相对 v0.4.5 单变量）           —— 恢复「token 级替主干保值」
#   v0.4.10  + heads.sample_grad_scale=0.3             —— 文档级→主干的梯度缩放 λ
#   v0.4.11  + loss.token=2.0                          —— 让信息保值机制更强
#   v0.4.12  + model.mid_init=codet5@8-11 (+rms+relu)  —— 瓶颈权重移植（不改结构，只换初始化）
#
# ★ 每条都是「相对 v0.4.9 只动一个量」，所以**不能同时开**；这里顺序跑。
# ★ ⚠️ 2 epoch 只能**排方向**，不能下结论（lessons A1 已在本项目六次翻车）。
#   v0.4.5 同时有 2ep 与 4ep 的读数，所以 2ep 结果可直接与 v0.4.5@2ep 对方向；
#   胜出者**必须续到 4 epoch** 才能引用（`--resume last.pt --epochs 2` + 同样的 `--set`）。
#
# 预计：4 × ≈2.2 h ≈ **9 小时**（每条 0.25 s/step × 16069 step × 2 epoch）。
#
# 用法：
#   setsid nohup bash scripts/queue_v049.sh > /tmp/queue_v049.log 2>&1 &
#   只跑某几条： bash scripts/queue_v049.sh v0.4.9 v0.4.10
set -u
cd /root/autodl-tmp/u-det

PY=/root/miniconda3/envs/udet/bin/python
CFG=configs/udet_v049.yaml

say() { echo "[q49] $(date +%H:%M:%S) $*"; }

# 变体表：tag → --set 参数（用 | 分隔，避免数组在 set -u 下的坑）
variant_sets() {
  case "$1" in
    v0.4.9)  echo "" ;;
    v0.4.10) echo "heads.sample_grad_scale=0.3" ;;
    v0.4.11) echo "loss.token=2.0" ;;
    v0.4.12) echo "model.mid_init=codet5@8-11|model.mid_norm=rms|model.mid_act=relu" ;;
    *)       echo "__UNKNOWN__" ;;
  esac
}
variant_desc() {
  case "$1" in
    v0.4.9)  echo "额外一路头（相对 v0.4.5 单变量）" ;;
    v0.4.10) echo "额外一路头 + 文档级梯度缩放 λ=0.3" ;;
    v0.4.11) echo "额外一路头 + loss.token=2.0" ;;
    v0.4.12) echo "额外一路头 + 瓶颈权重移植（CodeT5 blocks 8-11，RMS+ReLU 保真）" ;;
  esac
}

TAGS=("$@")
if [ ${#TAGS[@]} -eq 0 ]; then
  TAGS=(v0.4.9 v0.4.10 v0.4.11 v0.4.12)
fi

say "已登记：${TAGS[*]}（每条 2 epoch）"
while pgrep -f 'python train\.py' >/dev/null 2>&1; do
  say "有训练在跑，等 60s…"
  sleep 60
done
say "队列开始"

for TAG in "${TAGS[@]}"; do
  SPEC=$(variant_sets "$TAG")
  if [ "$SPEC" = "__UNKNOWN__" ]; then
    say "✗ 未知变体 $TAG ⇒ 跳过"
    continue
  fi
  SETARGS=()
  if [ -n "$SPEC" ]; then
    IFS='|' read -r -a _parts <<< "$SPEC"
    for p in "${_parts[@]}"; do SETARGS+=(--set "$p"); done
  fi
  say "=== $TAG：$(variant_desc "$TAG")（--set ${SPEC:-无}）==="

  if [ -f "runs/$TAG/eval.json" ]; then
    say "⚠ runs/$TAG/eval.json 已存在 ⇒ 跳过（不覆盖正式产物，B2）"
    continue
  fi

  OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" "${SETARGS[@]}" \
      > "/tmp/${TAG//./_}.log" 2>&1
  rc=$?
  say "$TAG 训练退出码 $rc"
  if [ "$rc" -ne 0 ]; then
    say "✗ $TAG 训练失败（详见 /tmp/${TAG//./_}.log）⇒ **中止队列**，后面的不盲跑"
    exit "$rc"
  fi

  OMP_NUM_THREADS=8 "$PY" train.py --config "$CFG" --tag "$TAG" \
      --eval --ckpt "runs/$TAG/best.pt" --dump-raw "${SETARGS[@]}" \
      > "/tmp/${TAG//./_}_eval.log" 2>&1
  rc=$?
  say "$TAG 评测退出码 $rc"
  if [ "$rc" -ne 0 ]; then
    say "✗ $TAG 评测失败（详见 /tmp/${TAG//./_}_eval.log）⇒ 中止队列"
    exit "$rc"
  fi

  # 2 轮快照：以后续跑到 4 轮时不会被覆盖（lessons B2 的教训）
  cp "runs/$TAG/eval.json" "runs/$TAG/eval_2ep.json"
  cp "runs/$TAG/best.pt" "runs/$TAG/best_2ep.pt"
  for stream in m4 hybrid; do
    for split in val test; do
      src="runs/$TAG/raw_${stream}_${split}.pt"
      [ -f "$src" ] && cp "$src" "runs/$TAG/raw_${stream}_${split}_2ep.pt"
    done
  done
  say "✓ $TAG 完成（已存 2 轮快照）：$(cat "runs/$TAG/eval.json" | tr -d '\n ' | head -c 400)"
done

say "队列全部结束"
say "下一步：胜出者用同样 --set 续跑 2 轮到 4 epoch 再引用（A1/A3）；"
say "  --resume runs/<TAG>/last.pt --epochs 2 <同样的 --set>"
