#!/usr/bin/env bash
# Run the GPU stages on the pod (after scripts/setup_pod.sh).
#
#   bash scripts/run_gpu_jobs.sh check   # attended, ~30 min: 1-clip test of every job + timing runs
#   bash scripts/run_gpu_jobs.sh train   # unattended, ~4.75 h; the pod stops itself at the end
#   bash scripts/run_gpu_jobs.sh infer   # unattended, ~3.25 h; the pod stops itself at the end
#
# train and infer re-launch themselves inside tmux, so a dropped SSH connection can't kill
# them, and they stop the pod when they end (succeeded or not), so a finished or crashed
# job never bills overnight. Every step is timed; a failed step doesn't stop the others.
# Logs and outputs live on the volume: /workspace/logs and /workspace/pitchview/outputs.
#
# Settings (environment variables):
#   DETECTOR_SIZE=medium DETECTOR_RES=1120 DETECTOR_BATCH=4 DETECTOR_ACCUM=4 DETECTOR_EPOCHS=20
#   PITCH_EPOCHS=20      MASK_CLIPS=49 (test clips whose full masks are kept)
#   NO_STOP=1 keeps the pod running after train/infer

set -euo pipefail

stage="${1:-}"
case "$stage" in
  check | train | infer) ;;
  *) echo "usage: $0 check|train|infer" >&2; exit 2 ;;
esac

# shellcheck source=scripts/pod_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pod_env.sh"
cd "$REPO_DIR"

DETECTOR_SIZE="${DETECTOR_SIZE:-medium}"
DETECTOR_RES="${DETECTOR_RES:-1120}"
DETECTOR_BATCH="${DETECTOR_BATCH:-4}"
DETECTOR_ACCUM="${DETECTOR_ACCUM:-4}"
DETECTOR_EPOCHS="${DETECTOR_EPOCHS:-20}"
PITCH_EPOCHS="${PITCH_EPOCHS:-20}"
MASK_CLIPS="${MASK_CLIPS:-49}"  # test clips whose full masks are kept (test has 49)
NO_STOP="${NO_STOP:-0}"

# Long stages: re-launch inside tmux, passing the settings explicitly (a running tmux
# server would otherwise hand the new session its own, older environment).
if [ "$stage" != check ] && [ -z "${INSIDE_JOB:-}" ]; then
  if tmux has-session -t "$stage" 2>/dev/null; then
    echo "'$stage' is already running. Watch it: tmux attach -t $stage"
    exit 1
  fi
  tmux new-session -d -s "$stage" \
    "env INSIDE_JOB=1 DETECTOR_SIZE=$DETECTOR_SIZE DETECTOR_RES=$DETECTOR_RES DETECTOR_BATCH=$DETECTOR_BATCH \
DETECTOR_ACCUM=$DETECTOR_ACCUM DETECTOR_EPOCHS=$DETECTOR_EPOCHS PITCH_EPOCHS=$PITCH_EPOCHS MASK_CLIPS=$MASK_CLIPS NO_STOP=$NO_STOP \
bash $REPO_DIR/scripts/run_gpu_jobs.sh $stage; exec bash"
  echo "Started '$stage' in tmux. Watch it: tmux attach -t $stage   (leave it running: Ctrl-b then d)"
  echo "Status so far: cat $LOG_DIR/${stage}_status.txt"
  exit 0
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${stage}_$(date +%Y%m%d_%H%M%S).log"
STATUS="$LOG_DIR/${stage}_status.txt"
exec > >(tee -a "$LOG") 2>&1
: >"$STATUS"
failures=0

# GPU use every 30 s: shows afterwards whether the GPU ever sat idle waiting on data.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 30 \
  >>"$LOG_DIR/${stage}_gpu.csv" 2>/dev/null &
sampler=$!

run() { # run <label> <command...>: time it, record ok/FAILED, never abort the stage
  local label=$1 start=$SECONDS
  shift
  printf '\n== %s ==\n$ %s\n' "$label" "$*"
  if "$@"; then
    printf 'ok      %-42s %6ds\n' "$label" $((SECONDS - start)) | tee -a "$STATUS"
  else
    printf 'FAILED  %-42s %6ds\n' "$label" $((SECONDS - start)) | tee -a "$STATUS"
    failures=$((failures + 1))
  fi
}

seconds_of() { # seconds recorded for a step label in the status file
  grep -F "$1" "$STATUS" | awk '{ print $NF }' | tr -d 's' | tail -n1
}

finish() {
  kill "$sampler" 2>/dev/null || true
  printf '\n%s finished %s with %d failed step(s):\n' "$stage" "$(date)" "$failures"
  cat "$STATUS"
  printf 'log: %s\n' "$LOG"
  [ "$stage" = check ] && return
  if [ "$NO_STOP" = 1 ]; then
    echo "NO_STOP=1: the pod keeps running (and billing)."
    return
  fi
  if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "!! RUNPOD_POD_ID unknown: stop the pod in the Runpod console"
    return
  fi
  sync
  echo "Stopping pod $RUNPOD_POD_ID. Outputs and logs stay on the volume."
  if command -v runpodctl >/dev/null; then
    # The pod image ships runpodctl v1 ("stop pod"). The pod's own key may stop its pod
    # (verified 2026-10-22) even though it cannot read pod details.
    { runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"; } && return
  fi
  curl -fsS -X POST -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
    "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" && return
  echo "!! automatic stop failed: stop the pod in the Runpod console"
}
trap finish EXIT

can_stop_self() { # train/infer stop the pod with runpodctl and the pod's own key
  # A real stop can't be tested without stopping, and the pod key is refused for reads
  # (HTTP 403), so check the pieces that `runpodctl stop pod` needs (verified 2026-10-22).
  [ -n "${RUNPOD_POD_ID:-}" ] || { echo "RUNPOD_POD_ID is not set"; return 1; }
  [ -n "${RUNPOD_API_KEY:-}" ] || { echo "RUNPOD_API_KEY is not set"; return 1; }
  command -v runpodctl >/dev/null || { echo "runpodctl is missing"; return 1; }
  echo "pod $RUNPOD_POD_ID: runpodctl and the pod key are present"
}

run "update code" git pull --ff-only
run "sync environment" uv sync --frozen

case "$stage" in
  check)
    # Separate output folder: these 1-epoch / 1-clip results must never look like finished work.
    export PITCHVIEW_OUT=outputs/check
    run "doctor" uv run pitchview doctor
    run "pod can stop itself" can_stop_self
    run "detector timing: 1 epoch" uv run pitchview train-detector --epochs 1 --size "$DETECTOR_SIZE" \
      --resolution "$DETECTOR_RES" --batch-size "$DETECTOR_BATCH" --grad-accum "$DETECTOR_ACCUM"
    run "detect: 1 clip" uv run pitchview detect --split valid --limit 1 --size "$DETECTOR_SIZE" \
      --resolution "$DETECTOR_RES"
    run "SAM 3.1: 1 clip" uv run pitchview sam --split valid --limit 1 --demo 1
    run "keypoints: 1 clip" uv run pitchview keypoints --split valid --limit 1 --tag offtheshelf
    run "pitch timing: 1 epoch" uv run pitchview train-pitch --epochs 1
    run "jerseys: 1 clip" uv run pitchview jerseys --split valid --limit 1

    epoch=$(seconds_of "detector timing")
    sam_clip=$(seconds_of "SAM 3.1: 1 clip")
    pitch_epoch=$(seconds_of "pitch timing")
    if [ -n "$epoch" ] && [ "$epoch" -gt 0 ]; then
      suggested=$((14400 / epoch))  # fit detector training in ~4 h
      suggested=$((suggested < 5 ? 5 : suggested > 30 ? 30 : suggested))
      printf '\nDetector: ~%ds per epoch (incl. setup) -> suggest DETECTOR_EPOCHS=%d (~4 h)\n' "$epoch" "$suggested"
    fi
    [ -n "$pitch_epoch" ] && printf 'Pitch model: ~%ds per epoch -> %d epochs ~ %d min\n' \
      "$pitch_epoch" "$PITCH_EPOCHS" $((pitch_epoch * PITCH_EPOCHS / 60))
    [ -n "$sam_clip" ] && printf 'SAM 3.1: ~%ds per clip -> 54 clips ~ %d min\n' "$sam_clip" $((sam_clip * 54 / 60))
    printf '\nIf everything above is ok:  DETECTOR_EPOCHS=<n> bash scripts/run_gpu_jobs.sh train\n'
    ;;
  train)
    run "train detector ($DETECTOR_EPOCHS epochs)" uv run pitchview train-detector --epochs "$DETECTOR_EPOCHS" \
      --size "$DETECTOR_SIZE" --resolution "$DETECTOR_RES" --batch-size "$DETECTOR_BATCH" --grad-accum "$DETECTOR_ACCUM"
    run "train pitch model ($PITCH_EPOCHS epochs)" uv run pitchview train-pitch --epochs "$PITCH_EPOCHS"
    ;;
  infer)
    run "detect: test" uv run pitchview detect --split test --size "$DETECTOR_SIZE" --resolution "$DETECTOR_RES"
    run "detect: valid" uv run pitchview detect --split valid --size "$DETECTOR_SIZE" --resolution "$DETECTOR_RES"
    run "keypoints: test (fine-tuned)" uv run pitchview keypoints --split test
    run "keypoints: test (off-the-shelf)" uv run pitchview keypoints --split test --tag offtheshelf
    run "keypoints: valid x5 (fine-tuned)" uv run pitchview keypoints --split valid --limit 5
    # Masks can't be remade once the volume is gone, so keep every one (RLE, ~10 MB a clip):
    # the viewer can then show and toggle each player's mask on any test clip.
    run "SAM 3.1: test (masks of $MASK_CLIPS clips)" uv run pitchview sam --split test --demo "$MASK_CLIPS"
    run "SAM 3.1: valid x5" uv run pitchview sam --split valid --limit 5 --demo 5
    run "jerseys: test, ByteTrack tracks" uv run pitchview jerseys --split test
    run "jerseys: test, SAM tracks" uv run pitchview jerseys --split test --tracker sam
    ;;
esac
