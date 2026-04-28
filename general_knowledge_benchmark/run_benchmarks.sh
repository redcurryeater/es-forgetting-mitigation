#!/usr/bin/env bash
# Evaluate ONE model on MMLU / HellaSwag / RACE via lm-evaluation-harness.
#
# The model arg can be:
#   - a HuggingFace Hub id            e.g. Qwen/Qwen2.5-1.5B-Instruct
#   - a local HF model directory      e.g. ./bench_out/hf_beta0
#   - a vLLM-named state dict (.pth)  e.g. ../logs/beta0_iter150/pytorch_model.pth
#     (auto-converted to an HF dir via export_checkpoint_to_hf.py)
#
# Prereqs (one-time):
#   pip install "lm-eval[vllm]"
#
# Usage:
#   ./run_benchmarks.sh <model> [label]
#
# Examples:
#   ./run_benchmarks.sh Qwen/Qwen2.5-1.5B-Instruct base
#   ./run_benchmarks.sh ../logs/beta0_iter150/pytorch_model.pth beta0
#   ./run_benchmarks.sh ../logs/beta1_iter150/pytorch_model-001.pth beta1
#
# Env overrides:
#   BASE_MODEL  HF id of the base architecture (only used when input is a .pth)
#   OUT_DIR     where to write converted HF dirs and lm-eval result JSONs
#   TASKS       comma-separated lm-eval task names
#   BATCH_SIZE  lm-eval batch size (default: auto)
#   DTYPE       float16 | bfloat16 | float32

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model> [label]"
  echo "  <model> is a HF id, a local HF dir, or a .pth checkpoint."
  exit 1
fi

MODEL="$1"
LABEL="${2:-$(basename "$MODEL" .pth)}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
OUT_DIR="${OUT_DIR:-./bench_out}"
TASKS="${TASKS:-mmlu,hellaswag,race}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
DTYPE="${DTYPE:-float16}"

mkdir -p "$OUT_DIR/results"

# If the input is a .pth, convert it to an HF directory first.
if [[ "$MODEL" == *.pth ]]; then
  HF_DIR="$OUT_DIR/hf_$LABEL"
  echo "=== [convert] $MODEL -> $HF_DIR ==="
  python export_checkpoint_to_hf.py \
    --base_model "$BASE_MODEL" \
    --ckpt "$MODEL" \
    --out "$HF_DIR" \
    --dtype "$DTYPE"
  PRETRAINED="$HF_DIR"
else
  PRETRAINED="$MODEL"
fi

echo "=== [eval] $LABEL ($PRETRAINED) ==="
lm_eval \
  --model vllm \
  --model_args "pretrained=${PRETRAINED},dtype=${DTYPE},gpu_memory_utilization=0.9,max_model_len=2048" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUT_DIR/results/$LABEL"

echo "Done. Results under $OUT_DIR/results/$LABEL/"
