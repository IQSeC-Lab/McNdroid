#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-.}"

echo "Scanning for personal/machine-specific identifiers in ${ROOT_DIR}"
PATTERN='msrahman3|Nowmi|erivas6|2026NeurIPS|/home/shared-datasets/McNdroid|/work/[^[:space:]]*Mcndroid|/home/[^[:space:]"'"'"']+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|ghp_[A-Za-z0-9]{36}|AKIA[0-9A-Z]{16}'

if command -v rg >/dev/null 2>&1; then
  rg -n --hidden --glob '!.git' --glob '!anonymity_audit.sh' "${PATTERN}" "${ROOT_DIR}" || true
else
  grep -RInE --exclude-dir=.git --exclude=anonymity_audit.sh "${PATTERN}" "${ROOT_DIR}" || true
fi
