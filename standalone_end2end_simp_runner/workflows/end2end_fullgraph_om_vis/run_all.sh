#!/usr/bin/env bash
set -euo pipefail

WF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$WF_DIR/../.." && pwd)"

IN_ONNX="${IN_ONNX:-$ROOT_DIR/ascend_c_onnx/end2end_simp.onnx}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/output/fullgraph_pipeline_$(date +%Y%m%d_%H%M%S)}"
SOC_VERSION="${SOC_VERSION:-Ascend310B4}"
PRECISION_MODE="${PRECISION_MODE:-allow_fp32_to_fp16}"
OM_BASENAME="${OM_BASENAME:-full_where2_gather40slice_allowfp32tofp16}"
COMPARE_DIR="${COMPARE_DIR:-$ROOT_DIR/output/fullgraph_compare_$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-10}"

mkdir -p "$WORK_DIR"

cd "$ROOT_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source "$ROOT_DIR/env.sh"

echo "[1/3] Prepare ONNX (atc_ready_full + optional rewrites)"
python "$WF_DIR/prepare_fullgraph_onnx.py" \
  --in-onnx "$IN_ONNX" \
  --out-dir "$WORK_DIR"

FINAL_ONNX="$(WORK_DIR="$WORK_DIR" python - <<'PY'
import os
import json
from pathlib import Path
work = Path(os.environ["WORK_DIR"])
man=work/"fullgraph_prepare_manifest.json"
print(json.loads(man.read_text(encoding="utf-8"))["final_onnx"])
PY
)"
echo "[INFO] FINAL_ONNX=$FINAL_ONNX"

echo "[2/3] Build OM via screen+atc"
WORK_DIR="$WORK_DIR" \
MODEL_ONNX="$FINAL_ONNX" \
SOC_VERSION="$SOC_VERSION" \
PRECISION_MODE="$PRECISION_MODE" \
OM_BASENAME="$OM_BASENAME" \
bash "$WF_DIR/build_full_om_in_screen.sh"

echo
echo "[ACTION REQUIRED] Wait ATC to finish first, then run step 3:"
echo "  OM_PATH=$WORK_DIR/$OM_BASENAME.om OUT_DIR=$COMPARE_DIR LIMIT=$LIMIT bash $WF_DIR/run_compare_and_visualize.sh"
