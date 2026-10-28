#!/usr/bin/env bash
# Run the GPU stages on the pod (after scripts/setup_pod.sh).
#
#   bash scripts/run_gpu_jobs.sh check   # attended, ~30 min: 1-clip test of every job + timing runs
#   bash scripts/run_gpu_jobs.sh train   # unattended, ~4.75 h; the pod stops itself at the end
#   bash scripts/run_gpu_jobs.sh infer   # unattended: detection, pitch keypoints, jerseys (ByteTrack)
#   bash scripts/run_gpu_jobs.sh all     # unattended, on the 4090: train, then infer, then the pod stops
#   bash scripts/run_gpu_jobs.sh sam     # unattended, on a 48 GB+ card: SAM 3.1 masks and tracks
#   bash scripts/run_gpu_jobs.sh retrain # unattended, ~3 h on a 24 GB+ card: detector v2, its detections, speed table
#   bash scripts/run_gpu_jobs.sh yolo    # unattended, ~2 h on a 24 GB+ card: YOLO26 on the same data, its detections, speed
#   bash scripts/run_gpu_jobs.sh jersey  # unattended, ~45 min on any card: fine-tune PARSeq two ways, read valid + test
#
# Every stage but check re-launches themselves inside tmux, so a dropped SSH connection can't kill
# them, and they stop the pod when they end (succeeded or not), so a finished or crashed
# job never bills overnight. Every step is timed; a failed step doesn't stop the others.
# Logs and outputs live on the volume: /workspace/logs and /workspace/pitchview/outputs.
#
# Settings (environment variables):
#   DETECTOR_SIZE=medium DETECTOR_RES=1120 DETECTOR_BATCH=4 DETECTOR_ACCUM=4 DETECTOR_EPOCHS=20
#   PITCH_EPOCHS=20      MASK_CLIPS=49 (test clips whose full masks are kept)
#   NO_STOP=1 keeps the pod running after train/infer
#   retrain only: DETECTOR_NAME=v2 DETECTOR_LR=5e-5 DETECTOR_LR_ENCODER=7.5e-5 DETECTOR_SCHEDULE=cosine
#     DETECTOR_WARMUP=1 DETECTOR_MIN_FACTOR=0.05 DETECTOR_PATIENCE=10, and DETECTOR_RES=1280
#   yolo only: YOLO_NAME=yolo26 YOLO_MODEL=yolo26m.pt YOLO_IMGSZ=1280 YOLO_EPOCHS=12 YOLO_BATCH=8
#   jersey only: JERSEY_EPOCHS=4, and HOLD_MINUTES=45: after the steps the pod waits this long for
#     $LOG_DIR/<stage>_copied (touch it once the outputs are copied home) before it stops itself

set -euo pipefail

stage="${1:-}"
case "$stage" in
  check | train | infer | sam | all | retrain | yolo | jersey) ;;
  *) echo "usage: $0 check|train|infer|sam|all|retrain|yolo|jersey" >&2; exit 2 ;;
esac

# shellcheck source=scripts/pod_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pod_env.sh"
cd "$REPO_DIR"

# retrain (review feedback, 2026-10-26): Sunday's run kept the learning rate flat (RF-DETR's step
# schedule first drops at epoch 100) and peaked early. v2: lower rates, a 1-epoch warmup, a cosine
# decay, and a larger input for the ~13 px ball. Its own folders (detector_v2, detections_v2).
if [ "$stage" = retrain ]; then
  DETECTOR_RES="${DETECTOR_RES:-1280}"
  DETECTOR_LR="${DETECTOR_LR:-5e-5}"
  DETECTOR_LR_ENCODER="${DETECTOR_LR_ENCODER:-7.5e-5}"
  DETECTOR_SCHEDULE="${DETECTOR_SCHEDULE:-cosine}"
  DETECTOR_WARMUP="${DETECTOR_WARMUP:-1}"
  DETECTOR_MIN_FACTOR="${DETECTOR_MIN_FACTOR:-0.05}"
  DETECTOR_PATIENCE="${DETECTOR_PATIENCE:-10}"  # let the cosine decay finish
fi
DETECTOR_NAME="${DETECTOR_NAME:-v2}"
DETECTOR_LR="${DETECTOR_LR:-1e-4}"
DETECTOR_LR_ENCODER="${DETECTOR_LR_ENCODER:-1.5e-4}"
DETECTOR_SCHEDULE="${DETECTOR_SCHEDULE:-step}"
DETECTOR_WARMUP="${DETECTOR_WARMUP:-0}"
DETECTOR_MIN_FACTOR="${DETECTOR_MIN_FACTOR:-0}"
DETECTOR_PATIENCE="${DETECTOR_PATIENCE:-5}"
DETECTOR_SIZE="${DETECTOR_SIZE:-medium}"
DETECTOR_RES="${DETECTOR_RES:-1120}"
DETECTOR_BATCH="${DETECTOR_BATCH:-4}"
DETECTOR_ACCUM="${DETECTOR_ACCUM:-4}"
DETECTOR_EPOCHS="${DETECTOR_EPOCHS:-20}"
PITCH_EPOCHS="${PITCH_EPOCHS:-20}"
MASK_CLIPS="${MASK_CLIPS:-49}"  # test clips whose full masks are kept (test has 49)
NO_STOP="${NO_STOP:-0}"
# yolo: the second specialist, on detector v2's data, resolution and epochs
YOLO_NAME="${YOLO_NAME:-yolo26}"
YOLO_MODEL="${YOLO_MODEL:-yolo26m.pt}"
YOLO_IMGSZ="${YOLO_IMGSZ:-1280}"
YOLO_EPOCHS="${YOLO_EPOCHS:-12}"
YOLO_BATCH="${YOLO_BATCH:-8}"
JERSEY_EPOCHS="${JERSEY_EPOCHS:-4}"
HOLD_MINUTES="${HOLD_MINUTES:-0}"

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
DETECTOR_NAME=$DETECTOR_NAME DETECTOR_LR=$DETECTOR_LR DETECTOR_LR_ENCODER=$DETECTOR_LR_ENCODER \
DETECTOR_SCHEDULE=$DETECTOR_SCHEDULE DETECTOR_WARMUP=$DETECTOR_WARMUP DETECTOR_MIN_FACTOR=$DETECTOR_MIN_FACTOR \
DETECTOR_PATIENCE=$DETECTOR_PATIENCE YOLO_NAME=$YOLO_NAME YOLO_MODEL=$YOLO_MODEL YOLO_IMGSZ=$YOLO_IMGSZ \
YOLO_EPOCHS=$YOLO_EPOCHS YOLO_BATCH=$YOLO_BATCH JERSEY_EPOCHS=$JERSEY_EPOCHS HOLD_MINUTES=$HOLD_MINUTES \
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
  if [ "$HOLD_MINUTES" -gt 0 ]; then
    # Outputs live on this pod's disk; a stopped pod may not start again (its host's GPU taken),
    # so wait for the laptop to copy them, then stop.
    touch "$LOG_DIR/${stage}_done"
    echo "Waiting up to $HOLD_MINUTES min for $LOG_DIR/${stage}_copied before stopping."
    local waited=0
    while [ ! -f "$LOG_DIR/${stage}_copied" ] && [ "$waited" -lt $((HOLD_MINUTES * 60)) ]; do
      sleep 30
      waited=$((waited + 30))
    done
  fi
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

train_steps() {
  run "train detector ($DETECTOR_EPOCHS epochs)" uv run pitchview train-detector --epochs "$DETECTOR_EPOCHS" \
    --size "$DETECTOR_SIZE" --resolution "$DETECTOR_RES" --batch-size "$DETECTOR_BATCH" --grad-accum "$DETECTOR_ACCUM"
  run "train pitch model ($PITCH_EPOCHS epochs)" uv run pitchview train-pitch --epochs "$PITCH_EPOCHS"
}

infer_steps() {
  run "detect: test" uv run pitchview detect --split test --size "$DETECTOR_SIZE" --resolution "$DETECTOR_RES"
  # valid: only the 5 tuning clips, like keypoints and SAM
  run "detect: valid x5" uv run pitchview detect --split valid --limit 5 --size "$DETECTOR_SIZE" \
    --resolution "$DETECTOR_RES"
  run "keypoints: test (fine-tuned)" uv run pitchview keypoints --split test
  run "keypoints: test (off-the-shelf)" uv run pitchview keypoints --split test --tag offtheshelf
  run "keypoints: valid x5 (fine-tuned)" uv run pitchview keypoints --split valid --limit 5
  run "jerseys: test, ByteTrack tracks" uv run pitchview jerseys --split test
}

retrain_steps() {
  run "train detector $DETECTOR_NAME ($DETECTOR_EPOCHS epochs, res $DETECTOR_RES, $DETECTOR_SCHEDULE lr $DETECTOR_LR)" \
    uv run pitchview train-detector --name "$DETECTOR_NAME" --epochs "$DETECTOR_EPOCHS" --size "$DETECTOR_SIZE" \
    --resolution "$DETECTOR_RES" --batch-size "$DETECTOR_BATCH" --grad-accum "$DETECTOR_ACCUM" --lr "$DETECTOR_LR" \
    --lr-encoder "$DETECTOR_LR_ENCODER" --schedule "$DETECTOR_SCHEDULE" --warmup "$DETECTOR_WARMUP" \
    --min-factor "$DETECTOR_MIN_FACTOR" --patience "$DETECTOR_PATIENCE"
  run "detect: test ($DETECTOR_NAME)" uv run pitchview detect --split test --dets "$DETECTOR_NAME" \
    --size "$DETECTOR_SIZE" --resolution "$DETECTOR_RES"
  run "detect: valid x5 ($DETECTOR_NAME)" uv run pitchview detect --split valid --limit 5 --dets "$DETECTOR_NAME" \
    --size "$DETECTOR_SIZE" --resolution "$DETECTOR_RES"
  # ~10 min: every model one frame at a time on this GPU (the original plan's speed table)
  local gpu
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_')
  run "speed benchmark" uv run pitchview bench --detectors model:1120 --detectors "$DETECTOR_NAME:$DETECTOR_RES" \
    --save "outputs/bench/bench_${gpu}.json"
}

# YOLO26 (Ultralytics, COCO-pretrained) on the frames RF-DETR trained on, for the accuracy-vs-speed
# table and a second specialist in the viewer. Ultralytics' defaults otherwise (SGD/auto optimizer,
# mosaic); the frames are converted once to YOLO's layout on local disk (data/yolo_gsr).
yolo_steps() {
  run "train $YOLO_NAME ($YOLO_MODEL, $YOLO_EPOCHS epochs, $YOLO_IMGSZ px)" uv run pitchview train-yolo \
    --name "$YOLO_NAME" --model "$YOLO_MODEL" --imgsz "$YOLO_IMGSZ" --epochs "$YOLO_EPOCHS" --batch "$YOLO_BATCH"
  run "detect: test ($YOLO_NAME)" uv run pitchview detect --split test --dets "$YOLO_NAME" --resolution "$YOLO_IMGSZ"
  run "detect: valid x5 ($YOLO_NAME)" uv run pitchview detect --split valid --limit 5 --dets "$YOLO_NAME" \
    --resolution "$YOLO_IMGSZ"
  # Sunday's RF-DETR again as the reference, so this table lines up with the retrain stage's (SAM is there)
  local gpu
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_')
  run "speed benchmark" uv run pitchview bench --detectors model:1120 --detectors "$YOLO_NAME:$YOLO_IMGSZ" --no-sam \
    --save "outputs/bench/bench_${gpu}_yolo.json"
}

# PARSeq fine-tuned on GSR train's shirt crops (jersey_train), two ways: only crops where zero-shot
# PARSeq reads a digit of the number, and every crop. Each epoch's checkpoint and zero-shot PARSeq
# then read all 58 valid clips (perfect boxes, every 9th frame) to pick one with jersey-score, and
# the SAM 3.1 tracks of test and valid x5 (every 10th frame), so the pick needs no second pod.
jersey_steps() {
  run "train jersey reader: agree ($JERSEY_EPOCHS epochs)" uv run pitchview train-jersey --name gsr_agree --keep agree \
    --epochs "$JERSEY_EPOCHS"
  run "train jersey reader: all ($JERSEY_EPOCHS epochs)" uv run pitchview train-jersey --name gsr_all --keep all \
    --epochs "$JERSEY_EPOCHS"
  run "perfect boxes: valid" uv run pitchview detect --split valid --oracle
  local readers=(zeroshot) reader e
  for e in $(seq 1 "$JERSEY_EPOCHS"); do readers+=("gsr_agree_e$e" "gsr_all_e$e"); done
  for reader in "${readers[@]}"; do
    run "jerseys: valid x58, perfect boxes ($reader)" uv run pitchview jerseys --split valid --tracker bytetrack \
      --dets oracle --stride 9 --reader "$reader"
  done
  for reader in "${readers[@]:1}"; do
    run "jerseys: test, SAM tracks ($reader)" uv run pitchview jerseys --split test --tracker sam --reader "$reader"
    run "jerseys: valid x5, SAM tracks ($reader)" uv run pitchview jerseys --split valid --limit 5 --tracker sam \
      --reader "$reader"
  done
}

# SAM 3.1 keeps every past frame in GPU memory (~27 MB a frame), so a 750-frame clip needs
# ~40 GB: it runs as its own stage on a 48 GB+ card, alongside training on the 4090.
sam_steps() {
  # Masks can't be remade once the volume is gone, so keep every one (RLE, ~10 MB a clip):
  # the viewer can then show and toggle each player's mask on any test clip.
  run "SAM 3.1: test (masks of $MASK_CLIPS clips)" uv run pitchview sam --split test --demo "$MASK_CLIPS"
  run "SAM 3.1: valid x5" uv run pitchview sam --split valid --limit 5 --demo 5
  run "jerseys: test, SAM tracks" uv run pitchview jerseys --split test --tracker sam
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
    printf '\nIf everything above is ok:  DETECTOR_EPOCHS=<n> bash scripts/run_gpu_jobs.sh all\n'
    ;;
  train) train_steps ;;
  infer) infer_steps ;;
  sam) sam_steps ;;
  retrain) retrain_steps ;;
  yolo) yolo_steps ;;
  jersey) jersey_steps ;;
  all)
    train_steps
    # Pick up fixes pushed while training ran.
    run "update code" git pull --ff-only
    run "sync environment" uv sync --frozen
    infer_steps
    ;;
esac
