#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_models.sh
#
# Downloads WAN 2.1 I2V model weights to a RunPod network volume.
# Run this on a RunPod *pod* (not serverless) that has the network volume
# mounted at /runpod-volume.
#
# Usage:
#   bash scripts/download_models.sh --model 5b
#   bash scripts/download_models.sh --model 14b-480p
#   bash scripts/download_models.sh --model 14b-720p
#   bash scripts/download_models.sh --model 14b-480p \
#       --lora 1234567 --civitai-token YOUR_TOKEN
#
# Options:
#   --model          5b | 14b-480p | 14b-720p   (required)
#   --lora           CivitAI model ID to download as a LoRA (optional)
#   --civitai-token  CivitAI API token (required when --lora is set)
#   --models-dir     Override model storage path (default: /runpod-volume/models)
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
MODEL_VARIANT=""
LORA_ID=""
CIVITAI_TOKEN=""
MODELS_DIR="${MODEL_PATH:-/runpod-volume/models}"

# ---- Argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)         MODEL_VARIANT="$2"; shift 2 ;;
    --lora)          LORA_ID="$2";        shift 2 ;;
    --civitai-token) CIVITAI_TOKEN="$2";  shift 2 ;;
    --models-dir)    MODELS_DIR="$2";     shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL_VARIANT" ]]; then
  echo "Error: --model is required (5b | 14b-480p | 14b-720p)" >&2
  exit 1
fi

# ---- Model map --------------------------------------------------------------
case "$MODEL_VARIANT" in
  5b)       HF_REPO="Wan-AI/Wan2.1-I2V-5B"       ; LOCAL_DIR="wan-i2v-5b"       ;;
  14b-480p) HF_REPO="Wan-AI/Wan2.1-I2V-14B-480P" ; LOCAL_DIR="wan-i2v-14b-480p" ;;
  14b-720p) HF_REPO="Wan-AI/Wan2.1-I2V-14B-720P" ; LOCAL_DIR="wan-i2v-14b-720p" ;;
  *)
    echo "Unknown model variant: $MODEL_VARIANT" >&2
    echo "Valid options: 5b, 14b-480p, 14b-720p" >&2
    exit 1
    ;;
esac

DEST="${MODELS_DIR}/${LOCAL_DIR}"

# ---- Download model from Hugging Face ---------------------------------------
echo "==> Downloading ${HF_REPO} to ${DEST} ..."
mkdir -p "$DEST"

pip install -q --no-cache-dir huggingface_hub

python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_REPO}",
    local_dir="${DEST}",
    local_dir_use_symlinks=False,
)
print("Model download complete.")
PYEOF

echo "==> Model saved to: ${DEST}"

# ---- Optionally download a LoRA from CivitAI --------------------------------
if [[ -n "$LORA_ID" ]]; then
  if [[ -z "$CIVITAI_TOKEN" ]]; then
    echo "Error: --civitai-token is required when using --lora" >&2
    exit 1
  fi

  LORAS_DIR="${MODELS_DIR}/loras"
  mkdir -p "$LORAS_DIR"

  echo "==> Fetching LoRA metadata for model ID: ${LORA_ID} ..."
  METADATA=$(curl -fsSL \
    -H "Authorization: Bearer ${CIVITAI_TOKEN}" \
    "https://civitai.com/api/v1/models/${LORA_ID}")

  # Extract the download URL and filename for the primary model file
  read -r DOWNLOAD_URL LORA_FILENAME < <(echo "$METADATA" | python3 -c "
import sys, json
data = json.load(sys.stdin)
version = data['modelVersions'][0]
files = version.get('files', [])
file_info = next(
    (f for f in files if f.get('primary', False) or f.get('type') == 'Model'),
    None,
)
if file_info is None:
    sys.stderr.write('Error: No suitable model file found in LoRA metadata\n')
    sys.exit(1)
print(file_info['downloadUrl'], file_info['name'])
")

  echo "==> Downloading LoRA: ${LORA_FILENAME} ..."
  curl -fL \
    -H "Authorization: Bearer ${CIVITAI_TOKEN}" \
    -o "${LORAS_DIR}/${LORA_FILENAME}" \
    "$DOWNLOAD_URL"

  echo "==> LoRA saved to: ${LORAS_DIR}/${LORA_FILENAME}"
fi

echo "==> All downloads complete."
