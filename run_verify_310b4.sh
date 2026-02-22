#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPE="$ROOT_DIR/standalone_end2end_simp_runner"

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base

rm -rf "$PIPE/.ascend_custom_opp_mmrotate_all_ops"

cd "$PIPE"
IN_ONNX="$PIPE/ascend_c_onnx/end2end_simp.onnx" \
WORK_DIR="$ROOT_DIR/verify_run/work_fullgraph_minset" \
COMPARE_DIR="$ROOT_DIR/verify_run/compare_fullgraph_minset" \
SOC_VERSION="${SOC_VERSION:-Ascend310B4}" \
PRECISION_MODE="${PRECISION_MODE:-allow_fp32_to_fp16}" \
LIMIT="${LIMIT:-10}" \
REUSE_OM_IF_EXISTS=0 \
bash workflows/end2end_fullgraph_om_vis/run_all_minset_auto.sh
