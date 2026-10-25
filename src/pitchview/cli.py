"""`pitchview` command line. Every GPU stage has a --smoke mode that runs on a laptop CPU.

--smoke uses 2 clips x 10 frames, tiny model settings, and ground-truth stand-ins where the
real step needs CUDA (SAM 3.1), so wiring bugs show up locally instead of on a paid pod.
Stage outputs are written one file per clip; a re-run skips clips that are already done.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import typer

from pitchview.gsr import DEFAULT_ROOT, list_clips

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
OUT = Path(os.environ.get("PITCHVIEW_OUT", "outputs"))  # the pod's check stage writes to outputs/check
WEIGHTS = Path("weights")
SAM_WEIGHTS = WEIGHTS / "sam3.1_multiplex.pt"
PITCH_BASE = WEIGHTS / "football-pitch-detection.pt"
SMOKE_CLIPS, SMOKE_FRAMES = 2, 10


def _out(smoke: bool) -> Path:
    return OUT / "smoke" if smoke else OUT


def _clips(split: str, smoke: bool, limit: int | None = None):
    clips = list_clips(split, DEFAULT_ROOT)
    return clips[:SMOKE_CLIPS] if smoke else (clips[:limit] if limit else clips)


def _todo(path: Path, name: str) -> bool:
    if path.exists():
        typer.echo(f"  {name}: done already, skipping")
        return False
    return True


@app.command()
def doctor() -> None:
    """Check the machine: GPU, bf16, data, weights, SAM 3.1 package, disk."""
    import torch

    failures = 0

    def check(name: str, ok: bool, detail: str = "", critical: bool = True) -> None:
        nonlocal failures
        failures += (not ok) and critical
        typer.echo(f"{'OK ' if ok else '!! '} {name:18s} {detail}")

    cuda = torch.cuda.is_available()
    check("CUDA GPU", cuda, torch.cuda.get_device_name(0) if cuda else "none (fine on the laptop)", critical=False)
    if cuda:
        check("bf16", torch.cuda.is_bf16_supported(), "SAM 3.1 runs in bf16")
        _, total = torch.cuda.mem_get_info()
        check("GPU memory", total > 20e9, f"{total / 1e9:.0f} GB")
    check("GSR data", (DEFAULT_ROOT / "test").is_dir(), str(DEFAULT_ROOT))
    check("SAM 3.1 weights", SAM_WEIGHTS.is_file(), str(SAM_WEIGHTS))
    check("pitch weights", PITCH_BASE.is_file(), str(PITCH_BASE))
    try:
        import sam3  # noqa: F401

        check("sam3 package", True, "importable")
    except ImportError:
        check("sam3 package", False, "Linux pod only", critical=cuda)
    # On the pod the frames sit on local disk and outputs on the network volume: check both.
    data_gb, out_gb = _free_gb(DEFAULT_ROOT), _free_gb(OUT)
    check("disk (data)", data_gb > 5, f"{data_gb:.0f} GB free at {DEFAULT_ROOT.resolve()}", critical=False)
    check("disk (outputs)", out_gb > 20, f"{out_gb:.0f} GB free at {OUT.resolve()}")
    raise typer.Exit(1 if failures else 0)


def _free_gb(path: Path) -> float:
    """Free space on the disk holding `path` (or its nearest existing parent)."""
    path = path.resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


@app.command("train-detector")
def train_detector(
    smoke: bool = False, size: str = "medium", resolution: int = 1120, epochs: int = 20,
    batch_size: int = 4, grad_accum: int = 4, workers: int = 8,
) -> None:  # fmt: skip
    """Fine-tune RF-DETR on GSR (converted to COCO, every 5th frame)."""
    from pitchview.coco import convert_split, make_subset
    from pitchview.detect import TrainSettings, train

    out = _out(smoke)
    full = Path("data/rfdetr_gsr")
    # RF-DETR scores itself on `valid` after every epoch: every 25th frame (1,740 images)
    # keeps that to about a minute. Final accuracy is measured separately on the test split.
    for split, stride in (("train", 5), ("valid", 25), ("test", 25)):
        if not (full / split / "_annotations.coco.json").exists():
            typer.echo(f"converting {split} to COCO (every {stride}th frame)")
            convert_split(DEFAULT_ROOT, split, full / split, stride=stride)
    dataset = full
    if smoke:
        dataset = out / "rfdetr_subset"
        for split, n in (("train", 8), ("valid", 4), ("test", 4)):
            make_subset(full / split, dataset / split, n)
        size, resolution, epochs, batch_size, grad_accum, workers = "nano", 384, 1, 2, 1, 0
    train(TrainSettings(dataset_dir=dataset, output_dir=out / "detector", size=size, resolution=resolution,
                        epochs=epochs, batch_size=batch_size, grad_accum_steps=grad_accum, num_workers=workers))  # fmt: skip


@app.command("train-pitch")
def train_pitch(smoke: bool = False, epochs: int = 20, imgsz: int = 640, batch: int = 16, workers: int = 8) -> None:
    """Fine-tune the pitch-keypoint model on frames auto-labelled from GSR."""
    from pitchview.pitch_train import finetune, write_yolo_pose_dataset

    out = _out(smoke)
    strides, cap = ({"train": 50, "valid": 150}, 8) if smoke else ({"train": 5, "valid": 25}, None)
    # The training set (~8,000 frames) goes under data/: local disk on the pod, where the
    # frames are, so it is hard links rather than slow copies onto the network volume.
    dataset = out / "pitch_dataset" if smoke else Path("data/pitch_dataset")
    counts = write_yolo_pose_dataset(DEFAULT_ROOT, dataset, strides, cap)
    typer.echo(f"labelled frames: {counts}")
    if smoke:
        epochs, imgsz, batch, workers = 1, 320, 2, 0
    best = finetune(PITCH_BASE, dataset / "data.yaml", out / "pitch_model", epochs, imgsz, batch,
                    "cpu" if smoke else None, workers)  # fmt: skip
    typer.echo(f"best weights: {best}")


def _oracle_detections(clip, max_frames: int | None) -> pd.DataFrame:
    """GSR's labelled boxes as if a perfect detector produced them."""
    last = max_frames or len(clip.frames)
    rows = [
        {"clip": clip.name, "frame": b.frame, "x1": b.x, "y1": b.y, "x2": b.x + b.w, "y2": b.y + b.h,
         "score": 1.0, "class_name": b.role}
        for b in clip.boxes if b.role in ("player", "goalkeeper", "referee", "ball") and b.frame <= last
    ]  # fmt: skip
    return pd.DataFrame(rows)


@app.command()
def detect(split: str = "test", smoke: bool = False, size: str = "medium", resolution: int = 1120,
           batch_size: int = 10, limit: int | None = None, oracle: bool = False) -> None:  # fmt: skip
    """Run the fine-tuned detector over every frame of each clip.
    --oracle writes the ground-truth boxes instead ("perfect detection", for error attribution)."""
    out = _out(smoke)
    if oracle:
        for clip in _clips(split, smoke, limit):
            path = out / "detections_oracle" / split / f"{clip.name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            _oracle_detections(clip, SMOKE_FRAMES if smoke else None).to_parquet(path)
        typer.echo("  oracle detections written")
        return

    import torch

    from pitchview.detect import load, predict_clip

    if smoke:
        size, resolution = "nano", 384
    model = load(out / "detector" / "checkpoint_best_total.pth", size, resolution)
    if torch.cuda.is_available():
        model.inference(compile=False, dtype=torch.float16)  # half precision; no trace, so any batch size works
    for clip in _clips(split, smoke, limit):
        path = out / "detections" / split / f"{clip.name}.parquet"
        if _todo(path, clip.name):
            n = predict_clip(model, clip, path, batch_size=batch_size, max_frames=SMOKE_FRAMES if smoke else None)
            typer.echo(f"  {clip.name}: {n} boxes")


@app.command()
def sam(split: str = "test", smoke: bool = False, limit: int | None = None, prompt: str = "person",
        embed_stride: int = 5, demo: int = 0) -> None:  # fmt: skip
    """SAM 3.1 'person' masks and tracks, plus box and masked jersey embeddings.
    --demo N also keeps every mask of the first N clips (RLE), for the demo video."""
    import time

    from pitchview.segment import GroundTruthTracks, MaskWriter, Sam31, write_tracks
    from pitchview.teams import Embedder, EmbeddingCollector

    out = _out(smoke)
    tracker = GroundTruthTracks() if smoke else Sam31(SAM_WEIGHTS)
    embedder = Embedder()
    for i, clip in enumerate(_clips(split, smoke, limit)):
        path = out / "sam" / split / f"{clip.name}.parquet"
        if not _todo(path, clip.name):
            continue
        start = time.time()
        collector = EmbeddingCollector(clip, embedder, stride=embed_stride)
        masks = MaskWriter() if i < demo else None

        def hook(frame_index, frame_out, collector=collector, masks=masks):
            collector(frame_index, frame_out)
            if masks is not None:
                masks(frame_index, frame_out)

        n = write_tracks(tracker, clip, path, prompt=prompt, max_frames=SMOKE_FRAMES if smoke else None,
                         on_frame=hook)  # fmt: skip
        collector.save(path.with_suffix(".npz"))
        if masks is not None:
            masks.save(out / "sam_masks" / split / f"{clip.name}.jsonl")
        typer.echo(f"  {clip.name}: {n} masks, {len(collector.frame)} embedded crops ({time.time() - start:.0f}s)")


@app.command()
def keypoints(split: str = "test", smoke: bool = False, weights: Path | None = None, tag: str = "finetuned",
              limit: int | None = None) -> None:  # fmt: skip
    """Pitch keypoints for every frame. --tag offtheshelf with Roboflow's weights for the ablation."""
    from pitchview.pitch import PitchModel, write_keypoints

    out = _out(smoke)
    w = weights or (PITCH_BASE if tag == "offtheshelf" else out / "pitch_model" / "weights" / "best.pt")
    model = PitchModel(w, device="cpu" if smoke else None)
    for clip in _clips(split, smoke, limit):
        path = out / f"keypoints_{tag}" / split / f"{clip.name}.parquet"
        if _todo(path, clip.name):
            n = write_keypoints(model, clip, path, max_frames=SMOKE_FRAMES if smoke else None)
            typer.echo(f"  {clip.name}: {n} frames")


def _detections(out: Path, split: str, clip, dets: str) -> pd.DataFrame:
    folder = "detections" if dets == "model" else "detections_oracle"
    return pq.read_table(out / folder / split / f"{clip.name}.parquet").to_pandas()


def _tracks(out: Path, split: str, clip, tracker: str, dets: str) -> pd.DataFrame:
    """The tracks jersey reading looks at: ByteTrack on detections, or SAM's own."""
    from pitchview.track import bytetrack

    if tracker == "sam":
        return pq.read_table(out / "sam" / split / f"{clip.name}.parquet").to_pandas().rename(columns={"obj_id": "track_id"})
    tracked = bytetrack(_detections(out, split, clip, dets), len(clip.frames), clip.fps)
    return tracked[tracked["class_name"] == "player"]


@app.command()
def jerseys(split: str = "test", smoke: bool = False, tracker: str = "bytetrack", dets: str = "model",
            stride: int = 10, limit: int | None = None) -> None:  # fmt: skip
    """Read jersey numbers per track with PARSeq (CPU is fine: ~40 crops/s)."""
    from PIL import Image

    from pitchview.jersey import JerseyReader, shirt_region, vote

    out = _out(smoke)
    if smoke:
        stride = 2  # only 10 frames, and tracks are not confirmed on frame 1
    reader = JerseyReader()
    for clip in _clips(split, smoke, limit):
        path = out / f"jerseys_{tracker}_{dets}" / split / f"{clip.name}.json"
        if not _todo(path, clip.name):
            continue
        tracks = _tracks(out, split, clip, tracker, dets)
        frames = {f.index: f for f in clip.frames}
        reads = []
        for frame, rows in tracks[(tracks["frame"] - 1) % stride == 0].groupby("frame"):
            image = Image.open(frames[int(frame)].path)
            crops = [image.crop(shirt_region(b)) for b in rows[["x1", "y1", "x2", "y2"]].itertuples(index=False)]
            for tid, (number, conf) in zip(rows["track_id"], reader.read(crops)):
                reads.append((int(tid), number, conf))
        numbers = vote(reads)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({str(k): v for k, v in numbers.items()}))
        typer.echo(f"  {clip.name}: {sum(v is not None for v in numbers.values())}/{len(numbers)} tracks read")


@app.command()
def assemble(
    split: str = "test", smoke: bool = False, name: str = "main", tracker: str = "bytetrack", foot: str = "mask",
    team_crops: str = "mask", jersey: bool = True, smooth: int = 5, pitch: str = "finetuned", dets: str = "model",
    limit: int | None = None,
) -> None:  # fmt: skip
    """Join every stage into the game state and write official prediction files.
    --pitch oracle (perfect calibration) and --dets oracle (perfect detection) attribute errors."""
    from pitchview.evaluate import image_ids, write_predictions
    from pitchview.gamestate import Variant, ball_positions, to_predictions
    from pitchview.gamestate import assemble as build
    from pitchview.pitch import frame_homographies, oracle_homography

    out = _out(smoke)
    v = Variant(tracker=tracker, foot=foot, team_crops=team_crops, jersey=jersey, smooth=smooth)
    for clip in _clips(split, smoke, limit):
        detections = _detections(out, split, clip, dets)
        sam_path = out / "sam" / split / f"{clip.name}.parquet"
        sam_tracks = pq.read_table(sam_path).to_pandas()
        emb = dict(np.load(sam_path.with_suffix(".npz")))
        if pitch == "oracle":
            by_frame = clip.boxes_by_frame()
            Hs = {f.index: H for f in clip.frames if (H := oracle_homography(by_frame.get(f.index, []))) is not None}
        else:
            kp = pq.read_table(out / f"keypoints_{pitch}" / split / f"{clip.name}.parquet").to_pandas()
            Hs = frame_homographies(kp)
        jersey_path = out / f"jerseys_{tracker}_{dets}" / split / f"{clip.name}.json"
        numbers = {int(k): v for k, v in json.loads(jersey_path.read_text()).items()} if jersey_path.exists() else None
        state = build(detections, sam_tracks, Hs, emb, numbers, len(clip.frames), clip.fps, v)
        state_path = out / "gamestate" / name / split / f"{clip.name}.parquet"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state.to_parquet(state_path)
        ball_positions(detections, Hs).to_parquet(state_path.with_name(f"{clip.name}_ball.parquet"))
        write_predictions(out / "trackers", split, name, clip.name, to_predictions(state, image_ids(clip)))
        typer.echo(f"  {clip.name}: {len(state)} rows, {state['track_id'].nunique()} tracks")


@app.command()
def stats(name: str = "main", split: str = "test", smoke: bool = False, limit: int | None = None) -> None:
    """Match analytics per clip (possession, speeds, team shape) as JSON, for the demo."""
    from pitchview.analytics import summarise

    out = _out(smoke)
    for clip in _clips(split, smoke, limit):
        state_path = out / "gamestate" / name / split / f"{clip.name}.parquet"
        state = pd.read_parquet(state_path)
        ball = pd.read_parquet(state_path.with_name(f"{clip.name}_ball.parquet"))
        summary = summarise(state, ball, clip.fps)
        path = out / "stats" / name / split / f"{clip.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        held = summary.get("possession_pct", {})
        typer.echo(f"  {clip.name}: possession {held}, fastest {summary.get('fastest', {}).get('top_speed_kmh')} km/h")


@app.command("eval")
def evaluate(names: list[str], split: str = "test", smoke: bool = False, limit: int | None = None) -> None:
    """GS-HOTA for each named run, plus tracking-only HOTA (role, team and jersey checks off)."""
    from pitchview.evaluate import gs_hota

    out = _out(smoke)
    clips = [c.name for c in _clips(split, smoke, limit)]
    typer.echo(f"{'run':28s} {'GS-HOTA':>8s} {'DetA':>6s} {'AssA':>6s} {'track-HOTA':>11s}")
    for name in names:
        full = gs_hota(DEFAULT_ROOT, out / "trackers", split, name, clips)
        bare = gs_hota(DEFAULT_ROOT, out / "trackers", split, name, clips, use_jersey=False, use_teams=False,
                       use_roles=False)  # fmt: skip
        typer.echo(f"{name:28s} {full['HOTA']:8.1f} {full['DetA']:6.1f} {full['AssA']:6.1f} {bare['HOTA']:11.1f}")


if __name__ == "__main__":
    app()
