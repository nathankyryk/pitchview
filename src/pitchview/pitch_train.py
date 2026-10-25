"""Fine-tune the pitch-keypoint model on SoccerNet frames with labels generated from GSR.

Off the shelf, Roboflow's pitch model places keypoints 5-10 m away from the real markings
on SoccerNet broadcast frames. GSR has no keypoint labels, but its metric player positions
give an exact homography per frame (pitch.oracle_homography), which projects the 32
template keypoints into the image: free, pixel-accurate labels for ~97% of frames.
"""

from __future__ import annotations

from pathlib import Path

from pitchview.coco import link_or_copy
from pitchview.gsr import list_clips
from pitchview.pitch import N_KEYPOINTS, keypoint_labels, oracle_homography


def write_yolo_pose_dataset(
    root: Path, out: Path, strides: dict[str, int] | None = None, max_frames_per_split: int | None = None
) -> dict[str, int]:
    """Write images/<split> and labels/<split> in YOLO pose format plus data.yaml.

    One object per image: the pitch, with a full-frame box and 32 keypoints (x, y, visible).
    Returns the number of labelled frames per split.
    """
    strides = strides or {"train": 5, "valid": 25}
    counts = {}
    for split, stride in strides.items():
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for clip in list_clips(split, root):
            by_frame = clip.boxes_by_frame()
            for frame in clip.frames[::stride]:
                if max_frames_per_split and n >= max_frames_per_split:
                    break
                H = oracle_homography(by_frame.get(frame.index, []))
                if H is None:
                    continue
                xy, visible = keypoint_labels(H, frame.width, frame.height)
                if visible.sum() < 4:
                    continue  # not enough of the pitch markings in view to learn from
                stem = f"{clip.name}_{frame.index:06d}"
                fields = ["0", "0.5", "0.5", "1", "1"]  # class 0 (pitch), full-frame box
                for (x, y), v in zip(xy, visible):
                    if v:
                        fields += [f"{x / frame.width:.6f}", f"{y / frame.height:.6f}", "2"]
                    else:
                        fields += ["0", "0", "0"]
                (lbl_dir / f"{stem}.txt").write_text(" ".join(fields) + "\n")
                link_or_copy(frame.path, img_dir / f"{stem}.jpg")
                n += 1
        counts[split] = n
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/valid\n"
        f"kpt_shape: [{N_KEYPOINTS}, 3]\n"
        "names:\n  0: pitch\n"
    )
    return counts


def finetune(
    base_weights: Path,
    data_yaml: Path,
    out_dir: Path,
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    workers: int = 8,
) -> Path:
    """Fine-tune from Roboflow's weights. Returns the path of the best checkpoint."""
    from ultralytics import YOLO

    model = YOLO(str(base_weights))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(out_dir.parent.resolve()),
        name=out_dir.name,
        exist_ok=True,
        patience=5,
        fliplr=0.0,  # a mirrored frame would need left/right keypoint ids swapped
        mosaic=0.0,  # Roboflow found mosaic hurts pitch keypoints
        verbose=False,
    )
    return out_dir / "weights" / "best.pt"
