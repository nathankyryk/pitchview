#!/usr/bin/env bash
# Set up a fresh Runpod pod that has the `pitchview` network volume mounted at /workspace.
#
# First pod (repo not on the volume yet):
#   curl -fsSL https://raw.githubusercontent.com/nathankyryk/pitchview/master/scripts/setup_pod.sh | bash
# Any later pod:
#   bash /workspace/pitchview/scripts/setup_pod.sh
#
# Safe to re-run: every step checks what is already there. Python, packages, data and
# model weights are kept on the volume, so a second pod is ready in a minute or two.
# Ends with `pitchview doctor`; a non-zero exit means something needs attention.

set -euo pipefail

REPO_URL="https://github.com/nathankyryk/pitchview.git"
UV_VERSION="0.13.1"  # same uv that wrote uv.lock
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/pitchview"
SAM_BYTES=3502755717  # size of facebook/sam3.1 sam3.1_multiplex.pt
PITCH_URL="https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"  # Roboflow sports, pitch keypoints

mkdir -p "$WORKSPACE/logs"
LOG="$WORKSPACE/logs/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n== %s ==\n' "$*"; }
fail() { printf '!! %s\n' "$*" >&2; exit 1; }

step "GPU and driver"
command -v nvidia-smi >/dev/null || fail "no nvidia-smi: this is not a GPU pod"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n1)
echo "driver supports CUDA $cuda"
awk -v v="$cuda" 'BEGIN { exit !(v + 0 >= 12.8) }' ||
  fail "PyTorch here is built for CUDA 12.8; redeploy the pod with the CUDA 12.8+ filter"

step "System tools"
missing=()
for tool in git curl tmux; do command -v "$tool" >/dev/null || missing+=("$tool"); done
if ((${#missing[@]})); then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
fi
echo "git, curl and tmux present"

step "Repository"
OLD_DIR="$WORKSPACE/pitchstate"  # the project's name until 2026-10-25
if [ -d "$OLD_DIR/.git" ] && [ ! -e "$REPO_DIR" ]; then
  # Move the old checkout rather than clone again: its environment, weights and outputs stay.
  mv "$OLD_DIR" "$REPO_DIR"
  git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
  # Scripts in the environment name the old folder in their first line; point them at the new one.
  { grep -lIs "^#!$OLD_DIR/" "$REPO_DIR"/.venv/bin/* || true; } | xargs -r sed -i "1s|^#!$OLD_DIR/|#!$REPO_DIR/|"
  echo "moved $OLD_DIR -> $REPO_DIR"
fi
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" log --oneline -1
# shellcheck source=scripts/pod_env.sh
source "$REPO_DIR/scripts/pod_env.sh"
cd "$REPO_DIR"

step "uv $UV_VERSION"
if ! command -v uv >/dev/null || [ "$(uv --version | cut -d' ' -f2)" != "$UV_VERSION" ]; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version

step "Python environment (exactly as uv.lock)"
uv sync --frozen
uv run python - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name(0)}")
EOF

step "Model caches on the volume"
# RF-DETR downloads its pretrained weights to ~/.roboflow; keep them on the volume.
mkdir -p "$WORKSPACE/.cache/roboflow"
if [ ! -L "$HOME/.roboflow" ]; then
  rm -rf "$HOME/.roboflow"
  ln -s "$WORKSPACE/.cache/roboflow" "$HOME/.roboflow"
fi
echo "$HOME/.roboflow -> $(readlink "$HOME/.roboflow")"

step "SoccerNet GSR data"
# The 3 zips (~28 GB) live on the volume: few large files, fast, and they persist.
# The ~123,000 frames are unpacked to this pod's local disk, which writes small files
# ~17x faster than the network volume (834 vs 49 files/s measured). Local disk is wiped
# when the pod stops, so each new pod unpacks again (a few minutes).
zips="$WORKSPACE/datasets/SoccerNetGS"
local_data="/root/pitchview-data"
mkdir -p "$zips/gamestate-2024" "$local_data"
if [ -d "$REPO_DIR/data" ] && [ ! -L "$REPO_DIR/data" ]; then
  # Older layout (data on the volume): keep any downloaded zips, drop the slow partial unpack.
  find "$REPO_DIR/data" -maxdepth 3 -name '*.zip' -exec mv -n {} "$zips/gamestate-2024/" \;
  rm -rf "$REPO_DIR/data"
fi
ln -sfn "$local_data" "$REPO_DIR/data"  # code reads data/..., which now lives on local disk
# GSR_SPLITS="valid test" skips train (~20 GB) on a pod that only runs inference
read -ra splits <<<"${GSR_SPLITS:-train valid test}"
uv run python scripts/download_gsr.py --zips "$zips" --out "$REPO_DIR/data/SoccerNetGS" --splits "${splits[@]}"
data="$REPO_DIR/data/SoccerNetGS/gamestate-2024"
for split in "${splits[@]}"; do
  echo "$split: $(find "$data/$split" -mindepth 1 -maxdepth 1 -type d | wc -l) clips"
done
df -h "$local_data" | tail -n 1

step "Model weights"
mkdir -p weights
if [ ! -f weights/football-pitch-detection.pt ]; then
  uvx gdown -q -O weights/football-pitch-detection.pt "$PITCH_URL" ||
    echo "!! pitch weights download failed: copy weights/football-pitch-detection.pt from the laptop"
fi
if [ -f weights/sam3.1_multiplex.pt ]; then
  size=$(stat -c %s weights/sam3.1_multiplex.pt)
  [ "$size" = "$SAM_BYTES" ] || fail "weights/sam3.1_multiplex.pt is $size bytes, expected $SAM_BYTES: copy it again"
else
  echo "!! SAM 3.1 weights missing (gated, so they come from the laptop). In PowerShell on the laptop:"
  printf '   scp -P <PORT> %s root@<POD_IP>:%s/weights/\n' \
    'C:\Users\natha\code\pitchview\weights\sam3.1_multiplex.pt' "$REPO_DIR"
  echo "   (<POD_IP> and <PORT>: the pod's 'SSH over exposed TCP' line in the Runpod console)"
fi
ls -lh weights

step "pitchview doctor"
if uv run pitchview doctor; then
  printf '\nReady. Next: bash scripts/run_gpu_jobs.sh check   (log: %s)\n' "$LOG"
else
  printf '\nNot ready: fix the !! lines above, then re-run this script. (log: %s)\n' "$LOG"
  exit 1
fi
