#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
LOGS_DIR="${LOGS_DIR:-logs/test_scripts}"

# all | avp_binary | avp_ternary | avgp_binary | avgp_ternary | gp_binary | gp_ternary
MODE="${MODE:-all}"

run_script() {
  local script_path="$1"
  echo
  echo "============================================================"
  echo "Running test script: ${script_path}"
  echo "device=${DEVICE}"
  echo "logs_dir=${LOGS_DIR}"
  echo "============================================================"
  PYTHON_BIN="${PYTHON_BIN}" \
  DEVICE="${DEVICE}" \
  LOGS_DIR="${LOGS_DIR}" \
  bash "${script_path}"
}

run_avp_binary() {
  run_script "${SCRIPT_DIR}/A-V-P/run_binary.sh"
}

run_avp_ternary() {
  run_script "${SCRIPT_DIR}/A-V-P/run_ternary.sh"
}

run_avgp_binary() {
  run_script "${SCRIPT_DIR}/A-V-G+P/run_binary.sh"
}

run_avgp_ternary() {
  run_script "${SCRIPT_DIR}/A-V-G+P/run_ternary.sh"
}

run_gp_binary() {
  run_script "${SCRIPT_DIR}/G-P/run_binary.sh"
}

run_gp_ternary() {
  run_script "${SCRIPT_DIR}/G-P/run_ternary.sh"
}

case "${MODE}" in
  all)
    run_avp_binary
    run_avp_ternary
    run_avgp_binary
    run_avgp_ternary
    run_gp_binary
    run_gp_ternary
    ;;
  avp_binary)
    run_avp_binary
    ;;
  avp_ternary)
    run_avp_ternary
    ;;
  avgp_binary)
    run_avgp_binary
    ;;
  avgp_ternary)
    run_avgp_ternary
    ;;
  gp_binary)
    run_gp_binary
    ;;
  gp_ternary)
    run_gp_ternary
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use one of: all, avp_binary, avp_ternary, avgp_binary, avgp_ternary, gp_binary, gp_ternary" >&2
    exit 1
    ;;
esac
