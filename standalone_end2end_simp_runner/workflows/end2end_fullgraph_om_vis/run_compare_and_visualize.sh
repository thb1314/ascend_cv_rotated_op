#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OM_PATH="${OM_PATH:?set OM_PATH}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"

IMAGE_DIR="${IMAGE_DIR:-$ROOT_DIR/images}"
LIMIT="${LIMIT:-10}"
DEVICE_ID="${DEVICE_ID:-0}"

BENCH_WARMUP_OM="${BENCH_WARMUP_OM:-1}"
BENCH_ITERS_OM="${BENCH_ITERS_OM:-3}"
BENCH_WARMUP_BOTH="${BENCH_WARMUP_BOTH:-0}"
BENCH_ITERS_BOTH="${BENCH_ITERS_BOTH:-1}"
ORT_INTRA_OP_THREADS="${ORT_INTRA_OP_THREADS:-8}"
ORT_INTER_OP_THREADS="${ORT_INTER_OP_THREADS:-1}"

mkdir -p "$OUT_DIR"

cd "$ROOT_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source "$ROOT_DIR/env.sh"

python "$ROOT_DIR/scripts/run_om_only_with_cached_ort.py" \
  --om "$OM_PATH" \
  --image-dir "$IMAGE_DIR" \
  --limit "$LIMIT" \
  --cache-root "$ROOT_DIR/output" \
  --out-dir "$OUT_DIR" \
  --device-id "$DEVICE_ID" \
  --bench-warmup "$BENCH_WARMUP_OM" \
  --bench-iters "$BENCH_ITERS_OM"

python "$ROOT_DIR/scripts/fill_ort_vis_and_benchmark.py" \
  --work-dir "$OUT_DIR" \
  --om "$OM_PATH" \
  --image-dir "$IMAGE_DIR" \
  --device-id "$DEVICE_ID" \
  --bench-warmup "$BENCH_WARMUP_BOTH" \
  --bench-iters "$BENCH_ITERS_BOTH" \
  --ort-intra-op-threads "$ORT_INTRA_OP_THREADS" \
  --ort-inter-op-threads "$ORT_INTER_OP_THREADS"

echo
echo "[DONE] Key outputs:"
echo "  $OUT_DIR/summary_om_only.csv"
echo "  $OUT_DIR/summary_ort_vs_om.csv"
echo "  $OUT_DIR/report_ort_vs_om.json"
echo "  $OUT_DIR/<image_stem>/vis_side_by_side.jpg"

