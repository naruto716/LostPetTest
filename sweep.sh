#!/usr/bin/env bash
set -e

# 你可以改这个根目录
ROOT_OUT=./outputs/sweeps_$(date +%Y%m%d_%H%M%S)
EPOCHS=60

mkdir -p "$ROOT_OUT"

# 小网格：3 个 LR × 1 个 margin（可解开更多） × BN 开关
LRS=("0.0002" "0.0003" "0.0005")
MARGINS=("0.3")
BNs=("true" "false")

i=0
for lr in "${LRS[@]}"; do
  for m in "${MARGINS[@]}"; do
    for bn in "${BNs[@]}"; do
      i=$((i+1))
      RUN=lr${lr}_m${m}_bn${bn}
      OUTDIR="${ROOT_OUT}/${RUN}"
      echo ">>> [${i}] Running ${RUN}"
      CUDA_VISIBLE_DEVICES=0 python launch.py \
        --output_dir "${OUTDIR}" \
        --epochs ${EPOCHS} \
        --base_lr ${lr} \
        --triplet_margin ${m} \
        --bn_neck ${bn} \
        --use_center_loss false \
        --freeze_backbone true
    done
  done
done

# 可选：再跑 CenterLoss 两组
for cw in 0.0005 0.001; do
  RUN=center_cw${cw}
  OUTDIR="${ROOT_OUT}/${RUN}"
  echo ">>> Running ${RUN}"
  CUDA_VISIBLE_DEVICES=0 python launch.py \
    --output_dir "${OUTDIR}" \
    --epochs ${EPOCHS} \
    --base_lr 0.0003 \
    --triplet_margin 0.3 \
    --bn_neck true \
    --use_center_loss true \
    --center_w ${cw} \
    --freeze_backbone true
done

echo "All runs finished under: ${ROOT_OUT}"
