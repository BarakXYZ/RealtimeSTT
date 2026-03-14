#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <model-alias>"
  echo "Example: $0 base.en"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHISPER_CPP_DIR="${ROOT_DIR}/third_party/whisper.cpp"

if [[ ! -d "${WHISPER_CPP_DIR}" ]]; then
  echo "third_party/whisper.cpp is missing"
  exit 1
fi

cd "${ROOT_DIR}"
"${WHISPER_CPP_DIR}/models/generate-coreml-model.sh" "$1"
