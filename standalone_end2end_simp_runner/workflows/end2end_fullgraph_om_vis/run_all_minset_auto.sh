#!/usr/bin/env bash
set -euo pipefail

WF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$WF_DIR/../.." && pwd)"

IN_ONNX="${IN_ONNX:-$ROOT_DIR/ascend_c_onnx/end2end_simp.onnx}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/output/fullgraph_pipeline_minset_auto_$(date +%Y%m%d_%H%M%S)}"
COMPARE_DIR="${COMPARE_DIR:-$ROOT_DIR/output/fullgraph_compare_minset_auto_$(date +%Y%m%d_%H%M%S)}"

SOC_VERSION="${SOC_VERSION:-Ascend310B4}"
PRECISION_MODE="${PRECISION_MODE:-allow_fp32_to_fp16}"
OM_BASENAME="${OM_BASENAME:-full_minset_where2_gather40slice_allowfp32tofp16}"
DEVICE_ID="${DEVICE_ID:-0}"
LIMIT="${LIMIT:-10}"

ATC_EXTRA_ARGS="${ATC_EXTRA_ARGS:-}"
ORT_INTRA_OP_THREADS="${ORT_INTRA_OP_THREADS:-8}"
ORT_INTER_OP_THREADS="${ORT_INTER_OP_THREADS:-1}"
BENCH_WARMUP_OM="${BENCH_WARMUP_OM:-1}"
BENCH_ITERS_OM="${BENCH_ITERS_OM:-3}"
BENCH_WARMUP_BOTH="${BENCH_WARMUP_BOTH:-0}"
BENCH_ITERS_BOTH="${BENCH_ITERS_BOTH:-1}"

POLL_SEC="${POLL_SEC:-15}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-0}"
REUSE_OM_IF_EXISTS="${REUSE_OM_IF_EXISTS:-1}"

mkdir -p "$WORK_DIR" "$COMPARE_DIR"

cd "$ROOT_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source "$ROOT_DIR/env.sh"

echo "[1/4] Prepare ONNX (minset: decompose NPULayerNorm + where2 + gather40)"
python "$WF_DIR/prepare_fullgraph_onnx.py" \
  --in-onnx "$IN_ONNX" \
  --out-dir "$WORK_DIR" \
  --enable-where2-maskarith \
  --enable-gather40-slice \
  --decompose-npu-layernorm

FINAL_ONNX="$(WORK_DIR="$WORK_DIR" python - <<'PY'
import json, os
from pathlib import Path
manifest = Path(os.environ["WORK_DIR"]) / "fullgraph_prepare_manifest.json"
obj = json.loads(manifest.read_text(encoding="utf-8"))
print(obj["final_onnx"])
PY
)"
echo "[INFO] FINAL_ONNX=$FINAL_ONNX"

if [[ ! -f "$FINAL_ONNX" ]]; then
  echo "[ERROR] missing final onnx: $FINAL_ONNX" >&2
  exit 2
fi

USED_OPS_TXT="$WORK_DIR/used_custom_ops.txt"
python - <<'PY' "$FINAL_ONNX" "$USED_OPS_TXT"
import onnx
import sys
from collections import Counter
from pathlib import Path

inp = Path(sys.argv[1])
outp = Path(sys.argv[2])
m = onnx.load(str(inp))
ops = Counter(n.op_type for n in m.graph.node)
used = [(k, ops[k]) for k in sorted(ops) if k.startswith("NPU") or k.startswith("Ascend") or k in {"GridPriorsNPU"}]
outp.write_text("\n".join(f"{k}\t{v}" for k, v in used) + "\n", encoding="utf-8")
print(f"[INFO] wrote {outp}")
PY

SESSION_NAME="${SESSION_NAME:-atc_${OM_BASENAME}_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="$WORK_DIR/screen_atc_${OM_BASENAME}.log"
OM_PATH="$WORK_DIR/$OM_BASENAME.om"

echo "[2/4] Build OM via screen+ATC"
if [[ -f "$OM_PATH" && "$REUSE_OM_IF_EXISTS" == "1" ]]; then
  echo "[SKIP] reuse existing OM: $OM_PATH"
else
  WORK_DIR="$WORK_DIR" \
  MODEL_ONNX="$FINAL_ONNX" \
  SOC_VERSION="$SOC_VERSION" \
  PRECISION_MODE="$PRECISION_MODE" \
  OM_BASENAME="$OM_BASENAME" \
  DEVICE_ID="$DEVICE_ID" \
  SESSION_NAME="$SESSION_NAME" \
  ATC_EXTRA_ARGS="$ATC_EXTRA_ARGS" \
  bash "$WF_DIR/build_full_om_in_screen.sh"

  echo "[WAIT] waiting for OM: $OM_PATH"
  start_ts="$(date +%s)"
  while [[ ! -f "$OM_PATH" ]]; do
    if ! (screen -ls 2>/dev/null | grep -q "$SESSION_NAME"); then
      echo "[ERROR] screen session exited before OM generated: $SESSION_NAME" >&2
      if [[ -f "$LOG_FILE" ]]; then
        tail -n 200 "$LOG_FILE" || true
      fi
      exit 3
    fi
    if [[ "$MAX_WAIT_SEC" -gt 0 ]]; then
      now_ts="$(date +%s)"
      elapsed="$((now_ts - start_ts))"
      if [[ "$elapsed" -gt "$MAX_WAIT_SEC" ]]; then
        echo "[ERROR] wait OM timeout: ${elapsed}s > ${MAX_WAIT_SEC}s" >&2
        exit 4
      fi
    fi
    sleep "$POLL_SEC"
  done
  echo "[OK] OM generated: $OM_PATH"
fi

echo "[3/4] OM/ORT compare + visualization"
OM_PATH="$OM_PATH" \
OUT_DIR="$COMPARE_DIR" \
LIMIT="$LIMIT" \
DEVICE_ID="$DEVICE_ID" \
ORT_INTRA_OP_THREADS="$ORT_INTRA_OP_THREADS" \
ORT_INTER_OP_THREADS="$ORT_INTER_OP_THREADS" \
BENCH_WARMUP_OM="$BENCH_WARMUP_OM" \
BENCH_ITERS_OM="$BENCH_ITERS_OM" \
BENCH_WARMUP_BOTH="$BENCH_WARMUP_BOTH" \
BENCH_ITERS_BOTH="$BENCH_ITERS_BOTH" \
bash "$WF_DIR/run_compare_and_visualize.sh"

echo "[4/4] Summary"
python - <<'PY' "$COMPARE_DIR"
import json
import csv
import glob
import sys
from pathlib import Path

wd = Path(sys.argv[1])
report = json.loads((wd / "report_ort_vs_om.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader((wd / "summary_ort_vs_om.csv").open()))
worst = max(rows, key=lambda r: float(r["dets_max_abs"])) if rows else None
vis_cnt = len(glob.glob(str(wd / "*" / "vis_side_by_side.jpg")))

print(f"[RESULT] compare_dir={wd}")
print(f"[RESULT] om_mean_ms={report['aggregate_latency_ms']['om_from_summary']['mean']}")
print(f"[RESULT] ort_mean_ms={report['aggregate_latency_ms']['onnx_from_run']['mean']}")
print(f"[RESULT] ort_over_om={report['aggregate_compare']['ort_over_om_ratio_by_mean']}")
print(f"[RESULT] vis_side_by_side_count={vis_cnt}")
if worst is not None:
    print(f"[RESULT] worst_dets_max_abs={worst['image']} {worst['dets_max_abs']}")
PY

echo "[DONE]"
echo "WORK_DIR=$WORK_DIR"
echo "COMPARE_DIR=$COMPARE_DIR"
echo "USED_CUSTOM_OPS=$USED_OPS_TXT"
