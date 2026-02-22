#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WORK_DIR="${WORK_DIR:?set WORK_DIR}"
CMD_SH="${CMD_SH:-$WORK_DIR/atc_cmds.sh}"
SESSION_NAME="${SESSION_NAME:-atc_$(basename "$WORK_DIR")_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-$WORK_DIR/screen_${SESSION_NAME}.log}"
ATC_TIMEOUT_SEC="${ATC_TIMEOUT_SEC:-1800}"
DEVICE_ID="${DEVICE_ID:-0}"

if [[ ! -f "$CMD_SH" ]]; then
  echo "missing CMD_SH: $CMD_SH" >&2
  exit 2
fi

mkdir -p "$WORK_DIR"

screen -wipe >/dev/null 2>&1 || true

RUNNER="$WORK_DIR/.screen_runner_${SESSION_NAME}.sh"
cat >"$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

exec >>"$LOG_FILE" 2>&1

echo "[runner] start $(date -Is)"

cd "$ROOT_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate base
source "$ROOT_DIR/env.sh"

export DEVICE_ID="$DEVICE_ID"
export ATC_TIMEOUT_SEC="$ATC_TIMEOUT_SEC"

pkill -9 -f atc.bin >/dev/null 2>&1 || true
pkill -9 -f ccec >/dev/null 2>&1 || true
if [[ -f "$ROOT_DIR/tools/clear_npu_mem.sh" ]]; then
  bash "$ROOT_DIR/tools/clear_npu_mem.sh" "$DEVICE_ID" --force >/dev/null 2>&1 || true
fi
sync >/dev/null 2>&1 || true
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

bash "$CMD_SH"

echo "[runner] done $(date -Is)"
EOF
chmod +x "$RUNNER"

touch "$LOG_FILE"

screen -dmS "$SESSION_NAME" bash "$RUNNER"

echo "OK: session=$SESSION_NAME"
echo "OK: log=$LOG_FILE"
