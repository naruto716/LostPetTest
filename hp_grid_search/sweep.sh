#!/usr/bin/env bash
set -Eeuo pipefail

# =========================
# Hyperparameter sweep runner
# - Comma-separated grids (epochs/LR/margin/BN)
# - Optional Center Loss sweep
# - Per-run logging
# - Dry-run mode (prints commands only)
# - Extra flags appended to every run
# =========================
#
# Examples:
#   bash sweep.sh --gpu 1 \
#     --epochs 20,40,60 \
#     --lrs 0.0002,0.0003,0.0005 \
#     --margins 0.3,0.5,0.8 \
#     --bns true,false \
#     --center-sweep true --center-ws 0.0005,0.001 \
#     --root-out ./outputs/my_sweeps \
#     --eval-split val \
#     --extra "--weight_decay 0.05 --warmup_iters 500"
#
# To preview without executing:
#   bash sweep.sh --dry-run

DATE_STR="$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="./outputs/sweeps_${DATE_STR}"
GPU="0"

# Grids (CSV lists)
EPOCHS_CSV="20"
LRS_CSV="0.0002,0.0003,0.0005"
MARGINS_CSV="0.3,0.5,0.8"
BNS_CSV="true,false"

# Center Loss sweep
CENTER_SWEEP="true"
CENTER_WS_CSV="0.0005,0.001"

# Runner options
EVAL_SPLIT="val"       # "val" or "test"
EXTRA_FLAGS=""         # appended to every run
DRY_RUN="false"        # if "true", only print commands
RESUME=""              # checkpoint path
EVAL_ONLY="false"      # if "true", evaluation only

usage() {
  cat <<'EOF'
Usage: sweep.sh [options]

General:
  --gpu <id>                 GPU id (default: 0)
  --root-out <dir>           Root output directory (default: ./outputs/sweeps_<ts>)
  --dry-run                  Print commands without executing them

Grids (CSV lists):
  --epochs <csv>             e.g. "20,40,60" (default: 20)
  --lrs <csv>                e.g. "0.0002,0.0003,0.0005"
  --margins <csv>            e.g. "0.3,0.5,0.8"
  --bns <csv>                e.g. "true,false"

Center Loss:
  --center-sweep <bool>      Enable center-loss sweeps (default: true)
  --center-ws <csv>          e.g. "0.0005,0.001"

Runner flags:
  --eval-split <val|test>    Evaluation split (default: val)
  --resume <ckpt>            Checkpoint path to resume from
  --eval-only                Evaluation only

Extras:
  --extra "<args>"           Additional args appended to every run (quoted)

  -h, --help                 Show this help
EOF
}

# -------- Parse CLI flags --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --root-out) ROOT_OUT="$2"; shift 2;;
    --epochs) EPOCHS_CSV="$2"; shift 2;;
    --lrs) LRS_CSV="$2"; shift 2;;
    --margins) MARGINS_CSV="$2"; shift 2;;
    --bns) BNS_CSV="$2"; shift 2;;
    --center-sweep) CENTER_SWEEP="$2"; shift 2;;
    --center-ws) CENTER_WS_CSV="$2"; shift 2;;
    --eval-split) EVAL_SPLIT="$2"; shift 2;;
    --resume) RESUME="$2"; shift 2;;
    --eval-only) EVAL_ONLY="true"; shift 1;;
    --extra) EXTRA_FLAGS="$2"; shift 2;;
    --dry-run) DRY_RUN="true"; shift 1;;
    -h|--help) usage; exit 0;;
    --) shift; break;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

# -------- Expand CSVs into arrays --------
IFS=',' read -r -a EPOCHS <<< "$EPOCHS_CSV"
IFS=',' read -r -a LRS <<< "$LRS_CSV"
IFS=',' read -r -a MARGINS <<< "$MARGINS_CSV"
IFS=',' read -r -a BNs <<< "$BNS_CSV"
IFS=',' read -r -a CENTER_WS <<< "$CENTER_WS_CSV"

mkdir -p "$ROOT_OUT"

echo "Root output: $ROOT_OUT"
echo "GPU: $GPU"
echo "Epochs: ${EPOCHS[*]}"
echo "LRs: ${LRS[*]}"
echo "Margins: ${MARGINS[*]}"
echo "BN neck: ${BNs[*]}"
echo "Center sweep: ${CENTER_SWEEP}  weights: ${CENTER_WS[*]}"
echo "Eval split: ${EVAL_SPLIT}"
echo "Resume: ${RESUME:-<none>}"
echo "Eval-only: ${EVAL_ONLY}"
echo "Extra flags: ${EXTRA_FLAGS:-<none>}"
echo

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY] $*"
  else
    echo "[$(date +%H:%M:%S)] $*"
    eval "$@"
  fi
}

i=0
for epochs in "${EPOCHS[@]}"; do
  # -------- Main grid (no center loss) --------
  for lr in "${LRS[@]}"; do
    for m in "${MARGINS[@]}"; do
      for bn in "${BNs[@]}"; do
        i=$((i+1))
        RUN="lr${lr}_m${m}_bn${bn}_e${epochs}"
        OUTDIR="${ROOT_OUT}/${RUN}"
        LOG="${OUTDIR}/run.log"
        mkdir -p "${OUTDIR}"

        CMD="CUDA_VISIBLE_DEVICES=${GPU} python launch.py \
          --output_dir \"${OUTDIR}\" \
          --epochs ${epochs} \
          --base_lr ${lr} \
          --triplet_margin ${m} \
          --bn_neck ${bn} \
          --use_center_loss false \
          --freeze_backbone true \
          --eval_split ${EVAL_SPLIT} \
          ${EXTRA_FLAGS}"

        # Append resume/eval-only only when set
        if [[ -n "$RESUME" ]]; then
          CMD+=" --resume \"${RESUME}\""
        fi
        if [[ "$EVAL_ONLY" == "true" ]]; then
          CMD+=" --eval_only"
        fi

        echo ">>> [${i}] ${RUN}"
        run_cmd "${CMD} 2>&1 | tee -a \"${LOG}\""
      done
    done
  done

  # -------- Optional center-loss sweeps --------
  if [[ "${CENTER_SWEEP}" == "true" ]]; then
    for cw in "${CENTER_WS[@]}"; do
      RUN="center_cw${cw}_e${epochs}"
      OUTDIR="${ROOT_OUT}/${RUN}"
      LOG="${OUTDIR}/run.log"
      mkdir -p "${OUTDIR}"

      CMD="CUDA_VISIBLE_DEVICES=${GPU} python launch.py \
        --output_dir \"${OUTDIR}\" \
        --epochs ${epochs} \
        --base_lr 0.0003 \
        --triplet_margin 0.3 \
        --bn_neck true \
        --use_center_loss true \
        --center_w ${cw} \
        --freeze_backbone true \
        --eval_split ${EVAL_SPLIT} \
        ${EXTRA_FLAGS}"

      if [[ -n "$RESUME" ]]; then
        CMD+=" --resume \"${RESUME}\""
      fi
      if [[ "$EVAL_ONLY" == "true" ]]; then
        CMD+=" --eval_only"
      fi

      echo ">>> [center] ${RUN}"
      run_cmd "${CMD} 2>&1 | tee -a \"${LOG}\""
    done
  fi
done

echo "All runs finished under: ${ROOT_OUT}"
