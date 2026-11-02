#!/usr/bin/env bash
set -euo pipefail

# Lightweight WebShop-small setup for an environment where full WebShop
# already runs. It downloads only the catalog files and builds indexes_1k; it
# does not reinstall PyTorch, VERL, Java, spaCy, or the WebShop requirements.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEBSHOP_ROOT="${WEBSHOP_ROOT:-${ROOT}/WebShop}"
DATA_DIR="${WEBSHOP_ROOT}/data"
SEARCH_DIR="${WEBSHOP_ROOT}/search_engine"
PRODUCTS_FILE="${DATA_DIR}/items_shuffle_1000.json"
ATTRIBUTES_FILE="${DATA_DIR}/items_ins_v2_1000.json"
HUMAN_ATTRIBUTES_FILE="${DATA_DIR}/items_human_ins.json"
RESOURCE_FILE="${SEARCH_DIR}/resources_1k/documents.jsonl"
INDEX_DIR="${SEARCH_DIR}/indexes_1k"

if [[ ! -d "${WEBSHOP_ROOT}/web_agent_site" ]]; then
  echo "Invalid WEBSHOP_ROOT: ${WEBSHOP_ROOT}" >&2
  exit 1
fi

if ! python3 -c 'import pyserini' >/dev/null 2>&1; then
  echo "pyserini is missing from the active WebShop/VERL environment." >&2
  echo "Activate the same environment that already runs your full WebShop experiment." >&2
  exit 1
fi

mkdir -p "${DATA_DIR}" "${SEARCH_DIR}/resources_1k"

download_if_missing() {
  local source_name="$1"
  local target="$2"
  if [[ -s "${target}" ]]; then
    echo "Reusing ${target}"
  else
    if ! command -v curl >/dev/null 2>&1; then
      echo "Missing ${target} and curl is unavailable." >&2
      echo "Upload the WebShop-small JSON files into ${DATA_DIR} before running this script." >&2
      exit 1
    fi
    curl -L --fail --retry 3 --retry-delay 2 \
      "${HF_ENDPOINT:-https://hf-mirror.com}/datasets/HongbangYuan/webshop/resolve/main/${source_name}" \
      -o "${target}"
  fi
}

download_if_missing "items_shuffle_1000.json" "${PRODUCTS_FILE}"
download_if_missing "items_ins_v2_1000.json" "${ATTRIBUTES_FILE}"
# The upstream WebShop loader imports this file even for synthetic goals.
download_if_missing "items_human_ins.json" "${HUMAN_ATTRIBUTES_FILE}"

python3 "${ROOT}/examples/agent_rubric2/build_webshop_small_index.py" \
  --webshop-root "${WEBSHOP_ROOT}" \
  --products "${PRODUCTS_FILE}" \
  --attributes "${ATTRIBUTES_FILE}" \
  --output "${RESOURCE_FILE}"

if [[ -d "${INDEX_DIR}" && "${REBUILD_WEBSHOP_SMALL_INDEX:-0}" != "1" ]]; then
  echo "Reusing existing WebShop-small index: ${INDEX_DIR}"
  echo "Set REBUILD_WEBSHOP_SMALL_INDEX=1 to rebuild it from the downloaded 1k catalog."
else
  BUILD_DIR="$(mktemp -d "${SEARCH_DIR}/indexes_1k.build.XXXXXX")"
  python3 -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "${SEARCH_DIR}/resources_1k" \
    --index "${BUILD_DIR}" \
    --generator DefaultLuceneDocumentGenerator \
    --threads "${WEBSHOP_INDEX_THREADS:-4}" \
    --storePositions --storeDocvectors --storeRaw

  if [[ -d "${INDEX_DIR}" ]]; then
    BACKUP_DIR="${INDEX_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
    mv "${INDEX_DIR}" "${BACKUP_DIR}"
    echo "Previous index moved to ${BACKUP_DIR}"
  fi
  mv "${BUILD_DIR}" "${INDEX_DIR}"
  echo "Built WebShop-small index: ${INDEX_DIR}"
fi

echo "WebShop-small assets are ready. The existing Python/Java environment was not changed."
