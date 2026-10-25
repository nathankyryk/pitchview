# Shared settings for the pod scripts (source it, don't run it).
# Caches live on the network volume, so a new pod reuses Python, wheels and model
# weights instead of downloading them again.

# SSH sessions may not inherit the container's environment (RUNPOD_POD_ID, RUNPOD_API_KEY);
# Runpod keeps a copy in /etc/rp_environment.
if [ -z "${RUNPOD_POD_ID:-}" ] && [ -f /etc/rp_environment ]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/rp_environment
  set +a
fi

export WORKSPACE="${WORKSPACE:-/workspace}"
export REPO_DIR="$WORKSPACE/pitchview"
export LOG_DIR="$WORKSPACE/logs"

export UV_INSTALL_DIR="$WORKSPACE/.cache/uv-bin"
export UV_CACHE_DIR="$WORKSPACE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE/.cache/uv-python"
export UV_LINK_MODE=copy  # network volumes may not support hard links

export HF_HOME="$WORKSPACE/.cache/huggingface"    # SigLIP
export TORCH_HOME="$WORKSPACE/.cache/torch"       # PARSeq, torchvision weights
export YOLO_CONFIG_DIR="$WORKSPACE/.cache/ultralytics"

export PATH="$UV_INSTALL_DIR:$PATH"
