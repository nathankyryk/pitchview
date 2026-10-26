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
from pitchview.jersey import MIN_CONF

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
OUT = Path(os.environ.get("PITCHVIEW_OUT", "outputs"))  # the pod's check stage writes to outputs/check
WEIGHTS = Path("weights")
SAM_WEIGHTS = WEIGHTS / "sam3.1_multiplex.pt"
PITCH_BASE = WEIGHTS / "football-pitch-detection.pt"
SMOKE_CLIPS, SMOKE_FRAMES = 2, 10


def _out(smoke: bool) -> Path:
    return OUT / "smoke" if smoke else OUT


def _clips(split: str, smoke: bool, limit: int | None = None, names: list[str] | None = None):
    clips = list_clips(split, DEFAULT_ROOT)
    if names:
        missing = set(names) - {c.name for c in clips}
        if missing:
            raise typer.BadParameter(f"not in {split}: {', '.join(sorted(missing))}")
        clips = [c for c in clips if c.name in names]
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
    batch_size: int = 4, grad_accum: int = 4, workers: int = 8, name: str = "model", lr: float = 1e-4,
    lr_encoder: float = 1.5e-4, schedule: str = "step", warmup: float = 0.0, min_factor: float = 0.0,
    patience: int = 5,
) -> None:  # fmt: skip
    """Fine-tune RF-DETR on GSR (converted to COCO, every 5th frame).
    --name writes to its own folder (detector_<name>), so runs stay comparable; the defaults are
    Sunday's run. Use the same name with `detect --dets <name>`."""
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
    train(TrainSettings(dataset_dir=dataset, output_dir=out / _named("detector", name), size=size,
                        resolution=resolution, epochs=epochs, batch_size=batch_size, grad_accum_steps=grad_accum,
                        lr=lr, lr_encoder=lr_encoder, schedule=schedule, warmup_epochs=warmup, min_factor=min_factor,
                        early_stopping_patience=patience, num_workers=workers))  # fmt: skip


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
           batch_size: int = 10, limit: int | None = None, oracle: bool = False, dets: str = "model") -> None:  # fmt: skip
    """Run the fine-tuned detector over every frame of each clip.
    --oracle writes the ground-truth boxes instead ("perfect detection", for error attribution).
    --dets <name> uses the detector trained with `train-detector --name <name>`."""
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
    model = load(out / _named("detector", dets) / "checkpoint_best_total.pth", size, resolution)
    if torch.cuda.is_available():
        model.inference(compile=False, dtype=torch.float16)  # half precision; no trace, so any batch size works
    for clip in _clips(split, smoke, limit):
        path = out / _named("detections", dets) / split / f"{clip.name}.parquet"
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


def _named(base: str, name: str) -> str:
    """Folder of a named variant: Sunday's detector is "model" (plain folder); others get a suffix."""
    return base if name == "model" else f"{base}_{name}"


def _detections(out: Path, split: str, clip, dets: str) -> pd.DataFrame:
    """dets: "model" (Sunday's detector), "oracle" (ground truth) or a re-trained detector's name."""
    return pq.read_table(out / _named("detections", dets) / split / f"{clip.name}.parquet").to_pandas()


def _tracks(out: Path, split: str, clip, tracker: str, dets: str) -> pd.DataFrame:
    """The tracks jersey reading looks at: ByteTrack on detections, or SAM's own."""
    from pitchview.track import bytetrack

    if tracker == "sam":
        return pq.read_table(out / "sam" / split / f"{clip.name}.parquet").to_pandas().rename(columns={"obj_id": "track_id"})
    tracked = bytetrack(_detections(out, split, clip, dets), len(clip.frames), clip.fps)
    return tracked[tracked["class_name"] == "player"]


JERSEY_STRIDE = 10  # read every 10th frame


def _jersey_dir(out: Path, kind: str, tracker: str, dets: str, stride: int, crops: str = "box") -> Path:
    """Where `jerseys` saves raw reads (kind "jersey_reads") or voted numbers ("jerseys"). Every
    setting but the first one used (every 10th frame, box crops) gets its own folder, for comparison."""
    suffix = ("" if stride == 10 else f"_s{stride}") + ("" if crops == "box" else f"_{crops}")
    return out / f"{kind}_{tracker}_{dets}{suffix}"


def _masks(path: Path, frames: set[int]) -> dict[tuple[int, int], dict]:
    """SAM's saved masks (COCO RLE) for the given frames, by (frame, object id)."""
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["frame"] in frames:
                out[(r["frame"], r["obj_id"])] = {"size": r["size"], "counts": r["counts"]}
    return out


@app.command()
def jerseys(split: str = "test", smoke: bool = False, tracker: str = "bytetrack", dets: str = "model",
            stride: int = JERSEY_STRIDE, limit: int | None = None, min_conf: float = MIN_CONF,
            allow_one: bool = False, merge_partial: bool = False, crops: str = "box") -> None:  # fmt: skip
    """Read jersey numbers per track with PARSeq (CPU is fine: ~40 crops/s), then vote per track.

    Every raw read is saved (jersey_reads_*). A clip whose reads are saved is only re-voted, in
    seconds, so vote settings can change without reading again (tune them with jersey-score).
    --crops mask (SAM tracks only) greys out everything in the shirt crop but the player's mask."""
    from PIL import Image
    from pycocotools import mask as mask_utils

    from pitchview.jersey import NOT_PLAYER_NUMBERS, JerseyReader, load_reads, shirt_crop, vote

    if crops not in ("box", "mask") or (crops == "mask" and tracker != "sam"):
        raise typer.BadParameter("--crops is box, or mask with --tracker sam")

    out = _out(smoke)
    if smoke:
        stride = 2  # only 10 frames, and tracks are not confirmed on frame 1
    exclude = frozenset() if allow_one else NOT_PLAYER_NUMBERS
    reader = None
    for clip in _clips(split, smoke, limit):
        path = _jersey_dir(out, "jerseys", tracker, dets, stride, crops) / split / f"{clip.name}.json"
        reads_path = _jersey_dir(out, "jersey_reads", tracker, dets, stride, crops) / split / f"{clip.name}.parquet"
        if reads_path.exists():
            reads = load_reads(reads_path)
        else:
            reader = reader or JerseyReader()
            tracks = _tracks(out, split, clip, tracker, dets)
            frames = {f.index: f for f in clip.frames}
            tracks = tracks[(tracks["frame"] - 1) % stride == 0]
            mask_path = out / "sam_masks" / split / f"{clip.name}.jsonl"
            masks = _masks(mask_path, set(tracks["frame"])) if crops == "mask" else {}
            raw = []
            for frame, rows in tracks.groupby("frame"):
                image = Image.open(frames[int(frame)].path)
                shirts = []
                for tid, box in zip(rows["track_id"], rows[["x1", "y1", "x2", "y2"]].itertuples(index=False)):
                    rle = masks.get((int(frame), int(tid)))
                    shirts.append(shirt_crop(image, box, mask_utils.decode(rle).astype(bool) if rle else None))
                for tid, (number, conf) in zip(rows["track_id"], reader.read(shirts)):
                    raw.append((int(tid), int(frame), number, conf))
            reads_path.parent.mkdir(parents=True, exist_ok=True)
            table = pd.DataFrame(raw, columns=["track_id", "frame", "number", "conf"])
            table.astype({"track_id": "int64", "frame": "int64", "conf": "float64"}).to_parquet(reads_path)
            reads = [(tid, number, conf) for tid, _, number, conf in raw]
        numbers = vote(reads, min_conf=min_conf, exclude=exclude, merge_partial=merge_partial)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({str(k): v for k, v in numbers.items()}))
        typer.echo(f"  {clip.name}: {sum(v is not None for v in numbers.values())}/{len(numbers)} tracks read")


@app.command()
def baseline(archive: Path = OUT / "baseline" / "gamestate-prtreid-strongsort-test-compressed.pklz",
             split: str = "test", limit: int | None = None) -> None:  # fmt: skip
    """Write the official GSR baseline's published predictions as run "baseline", to score with `eval`.
    Download the archive first (766 MB): see pitchview.baseline.TEST_URL."""
    from pitchview.baseline import TEST_URL, video_predictions
    from pitchview.evaluate import write_predictions

    if not archive.exists():
        raise typer.Exit(f"{archive} not found; download it from {TEST_URL}")
    for clip in _clips(split, False, limit):
        preds = video_predictions(archive, clip.name)
        write_predictions(OUT / "trackers", split, "baseline", clip.name, preds)
        typer.echo(f"  {clip.name}: {len(preds)} predictions")


@app.command("roster-template")
def roster_template(clips: list[str], split: str = "test") -> None:
    """Write rosters/<clip>.json to fill in by hand: the kit numbers seen on each side (from the labels)
    and each side's shirt colour. Names go in as "Firstname Lastname"; the nametag shortens them."""
    from PIL import Image

    from pitchview.jersey import shirt_region

    by_name = {c.name: c for c in _clips(split, False)}
    for name in clips:
        clip = by_name[name]
        numbers: dict = {"left": set(), "right": set()}
        colours: dict = {"left": [], "right": []}
        by_frame = clip.boxes_by_frame()
        for frame in clip.frames[::75]:  # 10 frames for the colour
            image = Image.open(frame.path).convert("RGB")
            for b in by_frame.get(frame.index, []):
                if b.role == "player" and b.team in colours:
                    crop = image.crop(shirt_region((b.x, b.y, b.x + b.w, b.y + b.h)))
                    colours[b.team].append(np.asarray(crop).reshape(-1, 3).mean(axis=0))
        for b in clip.boxes:
            if b.role == "player" and b.team in numbers and b.jersey:
                numbers[b.team].add(b.jersey)
        kits = {side: "rgb({:.0f}, {:.0f}, {:.0f})".format(*np.median(c, axis=0)) if c else "" for side, c in colours.items()}
        roster = {
            "match": "",  # "home vs away, competition, date"
            "kits": kits,
            **{side: {n: "" for n in sorted(nums, key=int)} for side, nums in numbers.items()},
        }
        path = Path("rosters") / f"{name}.json"
        if path.exists():
            typer.echo(f"  {name}: {path} exists, left as is")
            continue
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(roster, indent=2) + "\n")
        typer.echo(f"  {name}: {len(roster['left'])} left / {len(roster['right'])} right numbers, kits {roster['kits']}")


@app.command("detector-score")
def detector_score(split: str = "test", dets: str = "model", limit: int | None = None, stride: int = 5) -> None:
    """Box AP per class (ball separate) on the frames and rules the detector trained with."""
    from pitchview.evaluate import detection_ap

    clips = _clips(split, False, limit)
    scores = detection_ap(clips, {c.name: _detections(OUT, split, c, dets) for c in clips}, stride)
    typer.echo(f"{split}: {len(clips)} clips, every {stride}th labelled frame")
    typer.echo(f"{'class':12s} {'mAP50-95':>9s} {'AP50':>6s}")
    for name, s in scores.items():
        typer.echo(f"{name:12s} {s['ap']:9.3f} {s['ap50']:6.3f}")


@app.command("jersey-score")
def jersey_score(split: str = "valid", tracker: str = "bytetrack", dets: str = "model", limit: int | None = None,
                 stride: int = JERSEY_STRIDE, crops: str = "box", min_conf: float = MIN_CONF, min_reads: int = 1, allow_one: bool = False, merge_partial: bool = False,
                 grid: bool = False) -> None:  # fmt: skip
    """Score per-track jersey numbers against ground truth, re-voting from the saved raw reads.
    --grid tries a range of vote settings. Tune on valid, then report the chosen one on test."""
    from pitchview.evaluate import jersey_summary, jersey_truth
    from pitchview.jersey import NOT_PLAYER_NUMBERS, load_reads, vote

    clips = []
    for clip in _clips(split, False, limit):
        reads_path = _jersey_dir(OUT, "jersey_reads", tracker, dets, stride, crops) / split / f"{clip.name}.parquet"
        if reads_path.exists():
            clips.append((load_reads(reads_path), jersey_truth(_tracks(OUT, split, clip, tracker, dets), clip)))
    if not clips:
        raise typer.Exit("no saved reads: run `pitchview jerseys` first")

    def score(conf: float, reads_needed: int, one: bool, merge: bool) -> dict:
        predicted, truth, frames = [], [], []
        for reads, t in clips:
            numbers = vote(reads, conf, reads_needed, frozenset() if one else NOT_PLAYER_NUMBERS, merge)
            predicted += [numbers.get(int(k)) for k in t["track_id"]]
            truth += list(t["gt_jersey"])
            frames += list(t["frames"])
        return jersey_summary(predicted, truth, frames)

    settings = (
        [(c, r, o, m) for m in (False, True) for o in (True, False) for r in (1, 2, 3, 4, 5) for c in (0.5, 0.6, 0.7, 0.8, 0.9)]
        if grid
        else [(min_conf, min_reads, allow_one, merge_partial)]
    )
    typer.echo(f"{split} / {tracker}: {len(clips)} clips, {sum(len(t) for _, t in clips)} player tracks")
    typer.echo(f"{'min_conf':>8} {'min_reads':>9} {'allow 1':>7} {'merge':>5} {'agree':>6} {'read':>5} {'precision':>9} {'accuracy':>8}")
    for c, r, o, m in settings:
        s = score(c, r, o, m)
        typer.echo(f"{c:8.1f} {r:9d} {o!s:>7} {m!s:>5} {s['agree']:6.1%} {s['read']:5.0%} {s['precision']:9.1%} {s['accuracy']:8.1%}")


@app.command()
def assemble(
    split: str = "test", smoke: bool = False, name: str = "main", tracker: str = "bytetrack", foot: str = "mask",
    team_crops: str = "mask", jersey: bool = True, smooth: int = 25, pitch: str = "finetuned", dets: str = "model",
    limit: int | None = None, pitch_check: bool = True, jersey_stride: int = JERSEY_STRIDE,
    jersey_crops: str = "box", clips: list[str] | None = None,
) -> None:  # fmt: skip
    """Join every stage into the game state and write official prediction files.
    --pitch oracle (perfect calibration) and --dets oracle (perfect detection) attribute errors.
    --no-pitch-check keeps every fitted pitch mapping, even implausible or jumpy ones (pitch.checked_homographies).
    --clips SNGS-118 --clips SNGS-121 assembles only those clips."""
    from pitchview.evaluate import image_ids, write_predictions
    from pitchview.gamestate import Variant, ball_positions, to_predictions
    from pitchview.gamestate import assemble as build
    from pitchview.pitch import checked_homographies, frame_homographies, oracle_homography

    out = _out(smoke)
    v = Variant(tracker=tracker, foot=foot, team_crops=team_crops, jersey=jersey, smooth=smooth)
    for clip in _clips(split, smoke, limit, clips):
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
            if pitch_check:
                Hs = checked_homographies(Hs, width=clip.frames[0].width, height=clip.frames[0].height)
        jersey_dir = _jersey_dir(out, "jerseys", tracker, dets, jersey_stride, jersey_crops)
        jersey_path = jersey_dir / split / f"{clip.name}.json"
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


@app.command()
def render(clips: list[str], split: str = "test", run: str = "main", tracker: str = "bytetrack",
           dets: str = "model", backfill: bool = False, max_frames: int | None = None,
           out_dir: Path = OUT / "videos") -> None:  # fmt: skip
    """Annotated MP4 of each clip: team-coloured masks, nametags and a radar.

    Nametags show "#?" until the number is first read, then the name from rosters/<clip>.json
    ("L. Messi"), or "#10" without one; --backfill labels each track from its first frame.
    Needs the run's game state (`assemble`), SAM's saved masks and the jersey reads."""
    from pitchview.render import first_reads, load_roster, render_clip

    for clip in _clips(split, False, None, clips):
        state = pd.read_parquet(OUT / "gamestate" / run / split / f"{clip.name}.parquet")
        if "sam_obj_id" not in state:
            raise typer.Exit(f"{run} was assembled before boxes were kept: re-run `assemble --clips {clip.name}`")
        ball_path = OUT / "gamestate" / run / split / f"{clip.name}_ball.parquet"
        ball = pd.read_parquet(ball_path) if ball_path.exists() else None
        mask_path = OUT / "sam_masks" / split / f"{clip.name}.jsonl"
        masks = _masks(mask_path, {f.index for f in clip.frames}) if mask_path.exists() else {}
        reads_path = _jersey_dir(OUT, "jersey_reads", tracker, dets, JERSEY_STRIDE) / split / f"{clip.name}.parquet"
        numbers = state.dropna(subset=["jersey"]).groupby("track_id")["jersey"].first().to_dict()
        first = first_reads(pd.read_parquet(reads_path), numbers, MIN_CONF) if reads_path.exists() else {}
        roster = load_roster(Path("rosters") / f"{clip.name}.json")
        path = out_dir / run / f"{clip.name}.mp4"
        n = render_clip(clip.frames, state, masks, ball, roster, first, path, clip.fps, backfill,
                        title=f"PitchView  |  {clip.name}", max_frames=max_frames)  # fmt: skip
        typer.echo(f"  {clip.name}: {n} frames -> {path} ({len(roster.get('left', {})) + len(roster.get('right', {}))} names)")


def _torch_module(model):
    """The nn.Module inside a model wrapper (RF-DETR and Ultralytics nest it), for counting parameters."""
    for candidate in (model, getattr(model, "model", None), getattr(getattr(model, "model", None), "model", None)):
        if candidate is not None and hasattr(candidate, "parameters") and callable(candidate.parameters):
            try:
                next(iter(candidate.parameters()))
                return candidate
            except (StopIteration, TypeError):
                continue
    return None


@app.command()
def bench(smoke: bool = False, frames: int = 200, warmup: int = 20, clip: str | None = None,
          detectors: list[str] | None = None, sam: bool = True, save: Path | None = None) -> None:  # fmt: skip
    """Speed of each model on this machine's GPU: one frame at a time, half precision, after a
    warm-up, on frames already decoded in memory. --detectors name:resolution (repeatable), e.g.
    model:1120 (Sunday's RF-DETR, the default) and v2:1280. SAM 3.1 tracks the clip's first --frames frames."""
    import torch
    from PIL import Image

    from pitchview.bench import bench_detector, bench_pitch, bench_reader, bench_sam, parameters, size_mb
    from pitchview.detect import load
    from pitchview.jersey import JerseyReader
    from pitchview.pitch import PitchModel

    out = _out(smoke)
    detectors = detectors or ["model:1120"]
    cuda = torch.cuda.is_available()
    target = next(c for c in _clips("test", smoke, None) if clip in (None, c.name))
    n = min(3 if smoke else frames, len(target.frames))
    warmup = 1 if smoke else warmup
    images = [Image.open(f.path).convert("RGB") for f in target.frames[:n]]
    gt = target.boxes_by_frame()
    people = [
        [(b.x, b.y, b.x + b.w, b.y + b.h) for b in gt.get(f.index, []) if b.role in ("player", "goalkeeper")]
        for f in target.frames[:n]
    ]
    report = {"gpu": torch.cuda.get_device_name(0) if cuda else "cpu", "torch": torch.__version__,
              "clip": target.name, "frames": n, "warmup": warmup, "models": {}}  # fmt: skip
    rows = report["models"]

    def add(name: str, result: dict, model, weights: Path | None) -> None:
        module = _torch_module(model)
        result.update(parameters_m=round(parameters(module) / 1e6, 1) if module is not None else None,
                      file_mb=size_mb(weights))  # fmt: skip
        rows[name] = result
        typer.echo(f"  {name}: {result['fps']:.1f} fps, peak {result['peak_memory_gb'] or 0:.1f} GB")

    for spec in detectors:
        name, _, res = spec.partition(":")
        weights = out / _named("detector", name) / "checkpoint_best_total.pth"
        if not weights.is_file():
            typer.echo(f"  skip RF-DETR {name}: no {weights}")
            continue
        size, resolution = ("nano", 384) if smoke else ("medium", int(res or 1120))
        model = load(weights, size, resolution)
        if cuda:
            model.inference(compile=False, dtype=torch.float16)
        add(f"RF-DETR {name} ({resolution} px)", bench_detector(model, images, warmup), model, weights)
        del model

    pitch_weights = out / "pitch_model" / "weights" / "best.pt"
    pitch_weights = pitch_weights if pitch_weights.is_file() else PITCH_BASE
    device = None if cuda else "cpu"
    pitch = PitchModel(pitch_weights, device=device)
    add("pitch keypoints (YOLOv8x-pose)", bench_pitch(pitch.model, images, warmup, device), pitch.model, pitch_weights)
    del pitch

    reader = JerseyReader()
    add("jersey reader (PARSeq)", bench_reader(reader, images, people, warmup), reader.model, None)
    del reader

    if sam and cuda and SAM_WEIGHTS.is_file():
        from pitchview.segment import Sam31

        tracker = Sam31(SAM_WEIGHTS)
        add("SAM 3.1 (video tracking)", bench_sam(tracker, target, n), tracker.predictor.model, SAM_WEIGHTS)
        del tracker
    elif sam:
        typer.echo("  skip SAM 3.1: needs CUDA and its weights")

    path = save or out / "bench.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    typer.echo(f"\n{report['gpu']}, {n} frames of {target.name}")
    typer.echo(f"{'model':34s} {'fps':>6s} {'ms':>6s} {'GB':>5s} {'params M':>8s} {'file MB':>7s}")
    for name, r in rows.items():
        ms = f"{r['ms_median']:6.1f}" if r["ms_median"] is not None else f"{'-':>6s}"
        gb = f"{r['peak_memory_gb']:5.1f}" if r["peak_memory_gb"] is not None else f"{'-':>5s}"
        params = f"{r['parameters_m']:8.1f}" if r["parameters_m"] is not None else f"{'-':>8s}"
        file_mb = f"{r['file_mb']:7.1f}" if r["file_mb"] is not None else f"{'-':>7s}"
        typer.echo(f"{name:34s} {r['fps']:6.1f} {ms} {gb} {params} {file_mb}")
    typer.echo(f"saved {path}")


@app.command("eval")
def evaluate(names: list[str], split: str = "test", smoke: bool = False, limit: int | None = None,
             detail: bool = False, save: Path | None = None) -> None:  # fmt: skip
    """GS-HOTA for each named run, plus tracking-only HOTA (role, team and jersey checks off).

    --detail adds the error breakdown: GS-HOTA with each attribute check switched off in turn (the
    gain is what that attribute's mistakes cost), and the tracking errors behind track-HOTA: missed
    and extra people (DetRe, DetPr), split and merged tracks (AssRe, AssPr), IDF1 and ID switches.
    --save writes every score, per clip too, to a JSON file."""
    from pitchview.evaluate import gs_hota

    out = _out(smoke)
    clips = [c.name for c in _clips(split, smoke, limit)]
    off = {"use_jersey": False, "use_teams": False, "use_roles": False}
    report = {}
    typer.echo(f"{'run':28s} {'GS-HOTA':>8s} {'DetA':>6s} {'AssA':>6s} {'track-HOTA':>11s}")
    for name in names:
        row = {
            "gs": gs_hota(DEFAULT_ROOT, out / "trackers", split, name, clips, per_clip=True),
            "track": gs_hota(DEFAULT_ROOT, out / "trackers", split, name, clips, per_clip=True, detail=detail, **off),
        }
        if detail:
            for check in off:
                row[f"no_{check[4:]}"] = gs_hota(DEFAULT_ROOT, out / "trackers", split, name, clips, **{check: False})
        report[name] = row
        full, bare = row["gs"], row["track"]
        typer.echo(f"{name:28s} {full['HOTA']:8.1f} {full['DetA']:6.1f} {full['AssA']:6.1f} {bare['HOTA']:11.1f}")
    if detail:
        typer.echo("\nGS-HOTA with one check off, then the tracking-only errors")
        typer.echo(f"{'run':28s} {'-jersey':>7s} {'-team':>6s} {'-role':>6s} {'DetRe':>6s} {'DetPr':>6s} {'AssRe':>6s} "
                   f"{'AssPr':>6s} {'LocA':>5s} {'IDF1':>5s} {'IDSW':>5s}")  # fmt: skip
        for name, row in report.items():
            t = row["track"]
            typer.echo(
                f"{name:28s} {row['no_jersey']['HOTA']:7.1f} {row['no_teams']['HOTA']:6.1f} {row['no_roles']['HOTA']:6.1f} "
                f"{t['DetRe']:6.1f} {t['DetPr']:6.1f} {t['AssRe']:6.1f} {t['AssPr']:6.1f} {t['LocA']:5.1f} "
                f"{t['IDF1']:5.1f} {t['IDSW']:5d}"
            )
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(json.dumps({"split": split, "clips": clips, "runs": report}, indent=1))
        typer.echo(f"saved {save}")

if __name__ == "__main__":
    app()
