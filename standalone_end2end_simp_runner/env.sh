#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOC_VERSION="${SOC_VERSION:-Ascend910B2}"
DEVICE_ID="${DEVICE_ID:-0}"

ASCEND_SETENV="/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash"
if [[ -f "$ASCEND_SETENV" ]]; then
  source "$ASCEND_SETENV" >/dev/null 2>&1 || true
fi

INSTALL_BASE="$ROOT_DIR/.ascend_custom_opp_mmrotate_all_ops"
CUSTOM_DIR="$INSTALL_BASE/vendors/customize"

RUN_PKG="$ROOT_DIR/opp_pkgs/mmrotate_all_ops_frameworklaunch_${SOC_VERSION}_custom_opp_ubuntu_aarch64.run"
if [[ ! -d "$CUSTOM_DIR" ]]; then
  if [[ ! -f "$RUN_PKG" ]]; then
    echo "[ERROR] missing custom opp run pkg: $RUN_PKG" 1>&2
    exit 2
  fi
  mkdir -p "$INSTALL_BASE"
  "$RUN_PKG" --quiet --install-path="$INSTALL_BASE"
fi

export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_DIR"

ORT_LIB_DIR="$ROOT_DIR/onnxruntime-linux-aarch64-1.11.0/lib"
export LD_LIBRARY_PATH="$CUSTOM_DIR/op_api/lib:$CUSTOM_DIR/framework/onnx:$ORT_LIB_DIR:${LD_LIBRARY_PATH:-}"

export MAX_COMPILE_CORE_NUMBER="${MAX_COMPILE_CORE_NUMBER:-1}"
