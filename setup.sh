#!/bin/bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Starting MiniMax H3 Template Setup ==="
echo "LICENCE: MiniMax H3 open weights are NOT licensed for use in the US, EU, UK or"
echo "South Korea (MiniMax H3 Community License). See README.md before using this pod."

# Refuse a disposable workspace before doing installation work.
# The image's /start.sh fixes ComfyUI at this path.
export COMFYUI_PATH="${COMFYUI_PATH:-/workspace/runpod-slim/ComfyUI}"
if [ "$COMFYUI_PATH" != /workspace/runpod-slim/ComfyUI ]; then
  echo "FATAL: this image's /start.sh uses /workspace/runpod-slim/ComfyUI. Leave COMFYUI_PATH unset."
  exit 1
fi
python3 "$SCRIPT_DIR/custom_nodes/comfyui-h3-longform/storage.py" preflight "$COMFYUI_PATH"

# Apply before the image starts Jupyter; an empty JUPYTER_PASSWORD alone can
# trigger automatic token generation in /start.sh.
if [ "${JUPYTER_NO_AUTH:-1}" = "1" ]; then
  echo "Jupyter authentication disabled: anyone with access to its URL can run commands."
  if ! sed -i -E 's/--IdentityProvider\.token="\$\{JUPYTER_PASSWORD(:-)?\}"/--IdentityProvider.token="" --PasswordIdentityProvider.hashed_password=""/' /start.sh ||
     ! grep -Fq -- '--IdentityProvider.token="" --PasswordIdentityProvider.hashed_password=""' /start.sh; then
    echo "FATAL: could not configure Jupyter authentication in /start.sh."
    echo "Check the image startup command, or set JUPYTER_NO_AUTH=0 to keep image authentication."
    exit 1
  fi
fi

echo "=== Ensuring System Dependencies are Installed ==="
apt-get update
apt-get install -y wget ca-certificates util-linux
# ffmpeg does the per-chunk encoding and the final stitch.
command -v ffmpeg >/dev/null || apt-get install -y ffmpeg

# Jupyter serves /workspace in the base image. Keep its settings and user data on
# that persistent volume too; /root belongs to the disposable container disk.
export JUPYTER_CONFIG_DIR=/workspace/.jupyter
export JUPYTER_DATA_DIR=/workspace/.local/share/jupyter
export IPYTHONDIR=/workspace/.ipython
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$IPYTHONDIR"

# Global Volumes use a separate object-backed mount, never a replacement for the
# working filesystem containing Python, caches and ffmpeg's temporary files.
export H3_GLOBAL_MOUNT="${H3_GLOBAL_MOUNT:-/workspace-global}"
case "${H3_GLOBAL_STORAGE:-auto}" in
  auto) if mountpoint -q "$H3_GLOBAL_MOUNT"; then H3_GLOBAL_STORAGE=1; else H3_GLOBAL_STORAGE=0; fi ;;
  0|1) ;;
  *) echo "FATAL: H3_GLOBAL_STORAGE must be auto, 0 or 1."; exit 1 ;;
esac
if [ "$H3_GLOBAL_STORAGE" = "1" ]; then
  mountpoint -q "$H3_GLOBAL_MOUNT" || { echo "FATAL: attach a Global Volume and set its mount path to $H3_GLOBAL_MOUNT (not /workspace)."; exit 1; }
  export H3_GLOBAL_ROOT="${H3_GLOBAL_ROOT:-$H3_GLOBAL_MOUNT/minimax-h3}"
  case "$COMFYUI_PATH/" in "$H3_GLOBAL_MOUNT/"*) echo "FATAL: ComfyUI must stay on the working disk."; exit 1 ;; esac
else
  unset H3_GLOBAL_ROOT
fi

# The long-form workflow needs ComfyUI >= 0.35.0: MiniMaxH3AddGuide (which pins the
# voiceover) first appeared in 0.34.0, and 0.35.0 is the first runpod/comfyui build
# after it. This script deliberately does NOT update ComfyUI core - on these images
# `pip install -r requirements.txt` can reinstall torch and break the baked CUDA build.
#   Recommended image: runpod/comfyui:1.4.0-rc.164-comfyuiv0.35.0-cuda12.8

# Comfy-Org/MiniMax-H3 is ungated, so no token is needed. Set HF_TOKEN anyway if you
# have one: anonymous requests are rate-limited (~120/h vs ~1000/h) and slower.
HF_TOKEN="${HF_TOKEN:-}"
[ -n "$HF_TOKEN" ] && export HF_TOKEN || echo "NOTE: HF_TOKEN not set - downloads may be slower and rate-limited."
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
HF_REPO="Comfy-Org/MiniMax-H3"
# Same files, same layout, published by Comfy-Org on ModelScope. Used only if HF fails.
MS_BASE="https://modelscope.cn/models/Comfy-Org/MiniMax-H3/resolve/master"

H3_TEXT_ENCODER="${H3_TEXT_ENCODER:-nvfp4}"
DOWNLOAD_TURBO_LORA="${DOWNLOAD_TURBO_LORA:-1}"
DOWNLOAD_REF2VA="${DOWNLOAD_REF2VA:-0}"

# Self-healing check prevents directory collisions and fixes broken folders
if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  echo "First time setup: Copying baked ComfyUI to workspace..."
  if [ -e "$COMFYUI_PATH" ]; then
    echo "FATAL: existing ComfyUI directory has no main.py; refusing to delete user files."
    exit 1
  fi
  mkdir -p "$(dirname "$COMFYUI_PATH")"
  cp -r /opt/comfyui-baked "$COMFYUI_PATH"
fi

if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  echo "FATAL: no ComfyUI at $COMFYUI_PATH and /opt/comfyui-baked did not provide one."
  echo "       Check the container image."
  exit 1
fi
echo "ComfyUI: $COMFYUI_PATH"

# ComfyUI runs from its own venv; a bare `pip` would install where it never looks.
PY=""
for cand in "$COMFYUI_PATH/.venv-cu128/bin/python" "$COMFYUI_PATH/.venv/bin/python"; do
  [ -x "$cand" ] && { PY="$cand"; break; }
done
[ -n "$PY" ] || PY="$(command -v python3)"
echo "ComfyUI Python: $PY"

# Fail loudly, not 20 minutes into a render: without this node the workflow cannot
# pin the voiceover and will not load.
if ! grep -q 'node_id="MiniMaxH3AddGuide"' "$COMFYUI_PATH/comfy_extras/nodes_minimax_h3.py" 2>/dev/null; then
  echo "FATAL: this ComfyUI has no MiniMaxH3AddGuide node - it is older than 0.34."
  echo "         The bundled long-form workflow will not load. Use the image"
  echo "         runpod/comfyui:1.4.0-rc.164-comfyuiv0.35.0-cuda12.8 (or newer)."
  exit 1
fi

# 1. Custom nodes: the small H3 Longform pack bundled in this repo (no dependencies
# beyond ffmpeg). Everything else the workflow uses is core ComfyUI.
echo "Installing custom nodes..."
mkdir -p "$COMFYUI_PATH/custom_nodes"
if [ -d "$SCRIPT_DIR/custom_nodes" ]; then
  cp -r "$SCRIPT_DIR/custom_nodes/"* "$COMFYUI_PATH/custom_nodes/"
fi
# The bundled nodes only use image-provided torch/numpy and system ffmpeg. Do not
# reinstall every baked custom node's dependencies and risk replacing CUDA torch.

STORAGE_HELPER="$COMFYUI_PATH/custom_nodes/comfyui-h3-longform/storage.py"
if [ "$H3_GLOBAL_STORAGE" = "1" ]; then
  "$PY" "$STORAGE_HELPER" configure "$COMFYUI_PATH"
fi

# 2. Model folders
echo "Preparing model directories..."
for d in diffusion_models text_encoders vae loras; do
  mkdir -p "$COMFYUI_PATH/models/$d"
done

# -------------------------------------------------------------------
# Download helpers
# -------------------------------------------------------------------
# `hf` is only a downloader - ComfyUI never imports it - so it lives in its own venv.
# Upgrading huggingface_hub inside ComfyUI's environment breaks transformers' version
# pin, and ComfyUI then fails to start with "from transformers import CLIPTokenizer".
# Its Xet backend is much faster than wget on 20GB files but cannot resume a partial
# file, so wget stays as the fallback.
echo "Installing fast downloader (optional)..."
HF_VENV="/workspace/.hf-cli"
HF_BIN=""
if [ ! -x "$HF_VENV/bin/hf" ]; then
  if python3 -m venv "$HF_VENV" 2>/dev/null; then
    "$HF_VENV/bin/pip" install -q -U "huggingface_hub[hf-xet]" 2>/dev/null || true
  fi
fi
for cand in "$HF_VENV/bin/hf" "$HF_VENV/bin/huggingface-cli"; do
  [ -x "$cand" ] && { HF_BIN="$cand"; break; }
done
if [ -n "$HF_BIN" ]; then
  echo "Fast downloader: $HF_BIN (isolated venv)"
else
  echo "Fast downloader unavailable, using wget."
fi

# Check the safetensors header as well as a successful transfer exit status.
file_ok() {
  [ -f "$1" ] && [ "$(stat -c%s "$1")" -ge 1000000 ] || return 1
  "$PY" - "$1" <<'PY'
import json, os, struct, sys
try:
    with open(sys.argv[1], 'rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        if not 2 <= size <= min(os.path.getsize(sys.argv[1]) - 8, 100_000_000):
            raise ValueError('invalid header length')
        header = json.loads(f.read(size))
        end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
        if end + size + 8 != os.path.getsize(sys.argv[1]):
            raise ValueError('incomplete tensor data')
except (OSError, ValueError, KeyError, TypeError, struct.error):
    sys.exit(1)
PY
}

hf_fast() {
  # $1 = destination file, $2 = path inside the repo
  local dest="$1" rpath="$2" tmp
  [ -n "$HF_BIN" ] || return 1
  # Stage beside the destination so the final move is a rename, not a 20GB copy.
  tmp="$(mktemp -d "$(dirname "$dest")/.hfdl.XXXXXX")" || return 1
  if HF_XET_HIGH_PERFORMANCE=1 "$HF_BIN" download "$HF_REPO" "$rpath" --local-dir "$tmp"; then
    file_ok "$tmp/$rpath" && mv -f "$tmp/$rpath" "$dest"
  fi
  rm -rf "$tmp"
  file_ok "$dest"
}

fetch_url() {
  # $1 = destination file, $2 = URL, $3 = tag for the partial file
  local dest="$1" url="$2" part="$1.$3.part"
  # Partial files are keyed to the source: resuming one host's partial download
  # against another host's URL would splice two files together.
  if [ -n "$HF_TOKEN" ] && [ "$3" = "hf" ]; then
    wget -c -q --show-progress --tries=3 --read-timeout=120 \
      --header="Authorization: Bearer $HF_TOKEN" -O "$part" "$url" || return 1
  else
    wget -c -q --show-progress --tries=3 --read-timeout=120 -O "$part" "$url" || return 1
  fi
  file_ok "$part" && mv -f "$part" "$dest"
}

download_local() {
  # $1 = path inside the repo, e.g. vae/minimax_h3_audio_vae_fp32.safetensors
  local rpath="$1" dest="$2" fname
  fname="$(basename "$rpath")"
  if file_ok "$dest"; then
    echo "$fname already exists, skipping."
    return 0
  fi
  echo "Downloading $fname ..."
  hf_fast "$dest" "$rpath" && { echo "$fname done."; return 0; }
  echo "  falling back to wget..."
  fetch_url "$dest" "https://huggingface.co/$HF_REPO/resolve/main/$rpath" hf || true
  file_ok "$dest" || { echo "  trying the ModelScope mirror..."; fetch_url "$dest" "$MS_BASE/$rpath" ms || true; }
  if ! file_ok "$dest"; then
    rm -f "$dest"
    echo "ERROR: could not download $fname."
    return 1
  fi
  echo "$fname done."
}

download_h3() {
  local rpath="$1" stage
  if [ "$H3_GLOBAL_STORAGE" != "1" ]; then
    download_local "$rpath" "$COMFYUI_PATH/models/$rpath"
    return $?
  fi
  if ! "$PY" "$STORAGE_HELPER" model-ready "$rpath"; then
    # Reuse a complete local model when migrating an existing pod. Otherwise stage
    # one download at a time locally: HF locks and wget resume never touch FUSE.
    stage="$COMFYUI_PATH/.h3-downloads/$rpath"
    mkdir -p "$(dirname "$stage")" || return 1
    if [ ! -L "$COMFYUI_PATH/models/$rpath" ] && file_ok "$COMFYUI_PATH/models/$rpath"; then
      "$PY" "$STORAGE_HELPER" publish-model "$COMFYUI_PATH/models/$rpath" "$rpath" || return 1
    else
      download_local "$rpath" "$stage" || return 1
      "$PY" "$STORAGE_HELPER" publish-model "$stage" "$rpath" || return 1
      rm -f "$stage"
    fi
  fi
  "$PY" "$STORAGE_HELPER" link-model "$COMFYUI_PATH" "$rpath"
}

# -------------------------------------------------------------------
# 3. Model downloads (all from Comfy-Org/MiniMax-H3)
# -------------------------------------------------------------------
FAILED=0

# FL2VA diffusion model: first/last-frame + audio-guided video (~21GB)
download_h3 "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" || FAILED=1

# Text encoder: Qwen3-VL 32B. nvfp4 (~15.7GB) is what the official templates use;
# int8 (~27GB) is the alternative if nvfp4 misbehaves on your GPU.
if [ "$H3_TEXT_ENCODER" = "int8" ]; then
  download_h3 "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" || FAILED=1
  echo "NOTE: H3_TEXT_ENCODER=int8 - pick qwen3vl_32b_minimax_h3_int8_convrot in the"
  echo "      workflow's CLIPLoader, it defaults to the nvfp4 file."
else
  download_h3 "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" || FAILED=1
fi

# Video VAE (~2.8GB) and stereo audio VAE (~0.6GB, also encodes your voiceover)
download_h3 "vae/minimax_h3_video_vae_int8_convrot.safetensors" || FAILED=1
download_h3 "vae/minimax_h3_audio_vae_fp32.safetensors" || FAILED=1

# Turbo LoRA: 8 steps instead of 20 (~2GB). The workflow has it switched on.
if [ "$DOWNLOAD_TURBO_LORA" = "1" ]; then
  download_h3 "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" || FAILED=1
fi

# OPTIONAL: Ref2VA model (~21GB) for ComfyUI's own "Reference to Video" template.
# Not used by the bundled workflow.
if [ "$DOWNLOAD_REF2VA" = "1" ]; then
  download_h3 "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" || FAILED=1
  download_h3 "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" || FAILED=1
fi

if [ "$FAILED" = "1" ]; then
  echo "FATAL: one or more model downloads/backups failed (see ERROR lines above)."
  echo "         Restart the pod to retry - finished files are kept and skipped."
  exit 1
fi

# 4. Install bundled workflows into ComfyUI's Workflows sidebar
echo "Installing workflows..."
WF_DEST="$COMFYUI_PATH/user/default/workflows"
mkdir -p "$WF_DEST"
if [ -d "$SCRIPT_DIR/workflows" ]; then
  # Never overwrite a workflow already edited on this pod or Global Volume.
  for workflow in "$SCRIPT_DIR/workflows/"*.json; do
    [ -e "$WF_DEST/$(basename "$workflow")" ] || cp "$workflow" "$WF_DEST/"
  done
  ls "$WF_DEST" | sed 's/^/  workflow: /'
else
  echo "No workflows directory found in repo, skipping."
fi
if [ "$H3_GLOBAL_STORAGE" = "1" ]; then
  "$PY" "$STORAGE_HELPER" backup-workflows "$COMFYUI_PATH"
fi

# 5. Clean up the temporary git folder
echo "Cleaning up temp files..."
if [ "$SCRIPT_DIR" = /tmp/temp_repo ]; then rm -rf /tmp/temp_repo; fi

# 6. Start ComfyUI using the official RunPod entrypoint
echo "Setup complete! Handing over to start script..."
exec /start.sh
