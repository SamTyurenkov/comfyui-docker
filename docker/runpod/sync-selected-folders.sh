#!/bin/bash
set -euo pipefail
set -x

export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y rclone

# Source storage
SRC_BUCKET="${SRC_BUCKET:-}"
SRC_REGION="${SRC_REGION:-EU-RO-1}"
SRC_ENDPOINT="${SRC_ENDPOINT:-https://s3api-eu-ro-1.runpod.io}"
RUNPOD_USERID="${RUNPOD_USERID:-}"
RUNPOD_TOKEN="${RUNPOD_TOKEN:-}"

# Local destination root
LOCAL_ROOT="${LOCAL_ROOT:-/workspace}"
CONFIG_DIR="${CONFIG_DIR:-/root/.config/rclone}"
RCLONE_LOG="${RCLONE_LOG:-/var/log/rclone_sync.log}"

# Space-separated list of folder prefixes to sync.
# Example:
#   FOLDERS_TO_SYNC="models/loras/wan models/vae models/text_encoders"
FOLDERS_TO_SYNC="${FOLDERS_TO_SYNC:-}"
FILES_TO_SYNC="${FILES_TO_SYNC:-}"
EXCLUDE_FROM_SYNC="${EXCLUDE_FROM_SYNC:-}"

if [[ -z "${SRC_BUCKET}" || -z "${RUNPOD_USERID}" || -z "${RUNPOD_TOKEN}" ]]; then
  echo "Missing required env vars."
  echo "Required: SRC_BUCKET, RUNPOD_USERID, RUNPOD_TOKEN"
  echo "Provide at least one of: FOLDERS_TO_SYNC or FILES_TO_SYNC"
  echo "Optional: SRC_REGION, SRC_ENDPOINT, LOCAL_ROOT, CONFIG_DIR, RCLONE_LOG, EXCLUDE_FROM_SYNC"
  exit 1
fi

if [[ -z "${FOLDERS_TO_SYNC}" && -z "${FILES_TO_SYNC}" ]]; then
  echo "Nothing to sync."
  echo "Provide at least one of: FOLDERS_TO_SYNC or FILES_TO_SYNC"
  exit 1
fi

mkdir -p "${CONFIG_DIR}"
cat > "${CONFIG_DIR}/rclone.conf" <<EOL
[runpod]
type = s3
provider = Runpod
access_key_id = ${RUNPOD_USERID}
secret_access_key = ${RUNPOD_TOKEN}
endpoint = ${SRC_ENDPOINT}
region = ${SRC_REGION}
acl = private
EOL

path_excluded() {
  local path="$1"
  local ex
  [[ -z "${EXCLUDE_FROM_SYNC}" ]] && return 1
  for ex in ${EXCLUDE_FROM_SYNC}; do
    [[ "${path}" == "${ex}" ]] && return 0
    [[ "${path}" == "${ex}/"* ]] && return 0
  done
  return 1
}

sync_one_folder() {
  local folder="$1"
  local local_path
  local f ex rel
  local -a subtree_ex

  local_path="${LOCAL_ROOT%/}/${folder}"
  f="${folder%/}"
  subtree_ex=()
  if [[ -n "${EXCLUDE_FROM_SYNC}" ]]; then
    for ex in ${EXCLUDE_FROM_SYNC}; do
      ex="${ex%/}"
      [[ -z "${ex}" ]] && continue
      if [[ "${ex}" == "${f}/"* ]]; then
        rel="${ex#${f}/}"
        [[ -z "${rel}" ]] && continue
        subtree_ex+=( --exclude "${rel}" --exclude "${rel}/**" )
      fi
    done
  fi


  mkdir -p "${local_path}"
  echo "Syncing folder: ${folder}"
  echo "  Source: runpod:${SRC_BUCKET}/${folder}"
  echo "  Dest:   ${local_path}"

  /usr/bin/rclone sync "runpod:${SRC_BUCKET}/${folder}" "${local_path}" \
    --config "${CONFIG_DIR}/rclone.conf" \
    --log-level DEBUG \
    --buffer-size 64M \
    --exclude '*.sha256' \
    "${subtree_ex[@]}" \
    --use-server-modtime \
    --stats 5s \
    --stats-one-line \
    --progress \
    2>&1 | tee -a "${RCLONE_LOG}"
  return "${PIPESTATUS[0]}"
}

sync_one_file() {
  local file_path="$1"
  local local_file_path
  local local_parent_dir
  local remote_parent_dir
  local filename

  local_file_path="${LOCAL_ROOT%/}/${file_path}"
  local_parent_dir="$(dirname "${local_file_path}")"
  remote_parent_dir="$(dirname "${file_path}")"
  filename="$(basename "${file_path}")"
  mkdir -p "${local_parent_dir}"
  echo "Syncing file: ${file_path}"
  echo "  Source: runpod:${SRC_BUCKET}/${file_path}"
  echo "  Dest:   ${local_file_path}"

  /usr/bin/rclone copy "runpod:${SRC_BUCKET}/${remote_parent_dir}" "${local_parent_dir}/" \
    --config "${CONFIG_DIR}/rclone.conf" \
    --include "${filename}" \
    --log-level DEBUG \
    --buffer-size 64M \
    --use-server-modtime \
    --stats 5s \
    --stats-one-line \
    --progress \
    2>&1 | tee -a "${RCLONE_LOG}"
  return "${PIPESTATUS[0]}"
}

sync_errors=0

if [[ -n "${FOLDERS_TO_SYNC}" ]]; then
  for folder in ${FOLDERS_TO_SYNC}; do
    if path_excluded "${folder}"; then
      echo "Skipping folder (EXCLUDE_FROM_SYNC): ${folder}"
      continue
    fi
    sync_one_folder "${folder}" || sync_errors=$((sync_errors + 1))
  done
fi

if [[ -n "${FILES_TO_SYNC}" ]]; then
  for file_path in ${FILES_TO_SYNC}; do
    sync_one_file "${file_path}" || sync_errors=$((sync_errors + 1))
  done
fi

# chmod +x /workspace/venv_cc12_cuda129/bin/python
# find /workspace/venv_cc12_cuda129/bin -maxdepth 1 -type f -exec chmod +x {} \;

if [[ "${sync_errors}" -eq 0 ]]; then
  echo "Sync complete."
else
  echo "Sync finished with ${sync_errors} failed job(s). Check ${RCLONE_LOG}."
  exit 1
fi
