#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WORK_DIR="${WORK_DIR:?set WORK_DIR}"
MODEL_ONNX="${MODEL_ONNX:?set MODEL_ONNX}"

SOC_VERSION="${SOC_VERSION:-Ascend310B4}"
OM_BASENAME="${OM_BASENAME:-full_where2_gather40slice_allowfp32tofp16}"
PRECISION_MODE="${PRECISION_MODE:-allow_fp32_to_fp16}"
DEVICE_ID="${DEVICE_ID:-0}"

# Add additional atc params by environment, e.g.
# export ATC_EXTRA_ARGS="--deterministic=1 --op_select_implmode=high_precision_for_all"
ATC_EXTRA_ARGS="${ATC_EXTRA_ARGS:-}"

mkdir -p "$WORK_DIR"

CMD_SH="$WORK_DIR/atc_cmd_${OM_BASENAME}.sh"
LOG_FILE="$WORK_DIR/screen_atc_${OM_BASENAME}.log"
SESSION_NAME="${SESSION_NAME:-atc_${OM_BASENAME}_$(date +%Y%m%d_%H%M%S)}"

cat >"$CMD_SH" <<EOF
#!/usr/bin/env bash
set -euo pipefail

/usr/local/Ascend/cann-8.5.0/bin/atc \\
  --model=${MODEL_ONNX} \\
  --framework=5 \\
  --soc_version=${SOC_VERSION} \\
  --output=${WORK_DIR}/${OM_BASENAME} \\
  --input_format=ND \\
  --input_shape=input:1,3,672,1024 \\
  --log=error \\
  --precision_mode=${PRECISION_MODE} \\
  ${ATC_EXTRA_ARGS}
EOF
chmod +x "$CMD_SH"

echo "[INFO] generated $CMD_SH"
echo "[INFO] running atc with screen"

export WORK_DIR
export CMD_SH
export LOG_FILE
export SESSION_NAME
export DEVICE_ID
# run_atc_in_screen.sh runner does not force timeout unless CMD_SH uses timeout
unset ATC_TIMEOUT_SEC || true

bash "$ROOT_DIR/tools/run_atc_in_screen.sh"

echo
echo "[NEXT] Monitor:"
echo "  screen -ls"
echo "  tail -f $LOG_FILE"
echo
echo "[NEXT] Output OM:"
echo "  ${WORK_DIR}/${OM_BASENAME}.om"

