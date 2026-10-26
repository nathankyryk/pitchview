"""YOLO26 on the same data as RF-DETR: the second specialist detector, for the accuracy-vs-speed table.

The training set is the one RF-DETR uses (data/rfdetr_gsr: every 5th train frame, every 25th
valid frame), rewritten in YOLO's layout: images/<split>/ (hard links) and labels/<split>/,
one line per box: class x_centre y_centre width height, all relative to the image size.
Detections are written in RF-DETR's Parquet format, so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from pitchview.coco import CLASS_NAMES, link_or_copy
from pitchview.detect import DETECTION_SCHEMA
from pitchview.gsr import Clip

YOLO_CLASSES = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]  # player, goalkeeper, referee, ball -> 0-3


def yolo_lines(coco: dict) -> dict[str, list[str]]:
    """YOLO label lines per image file name, from a COCO dict with category ids 1-4."""
    images = {im["id"]: im for im in coco["images"]}
    lines: dict[str, list[str]] = {im["file_name"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        if a["category_id"] not in CLASS_NAMES:
            continue
        im = images[a["image_id"]]
        x, y, w, h = a["bbox"]
        W, H = im["width"], im["height"]
        cls = sorted(CLASS_NAMES).index(a["category_id"])
        lines[im["file_name"]].append(f"{cls} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
    return lines


def convert(coco_root: Path, out: Path, splits: tuple[str, ...] = ("train", "valid", "test")) -> Path:
    """Write the YOLO layout next to RF-DETR's COCO folders. Returns the data.yaml path."""
    for split in splits:
        coco = json.loads((coco_root / split / "_annotations.coco.json").read_text())
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for name, rows in yolo_lines(coco).items():
            link_or_copy(coco_root / split / name, out / "images" / split / name)
            (out / "labels" / split / name).with_suffix(".txt").write_text("".join(f"{r}\n" for r in rows))
    yaml = out / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(YOLO_CLASSES))
    yaml.write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/valid\ntest: images/test\nnames:\n{names}\n")
    return yaml


@dataclass
class YoloSettings:
    data_yaml: Path
    output_dir: Path  # weights land in <output_dir>/weights/best.pt
    model: str = "yolo26m.pt"  # COCO-pretrained, downloaded by Ultralytics
    imgsz: int = 1280
    epochs: int = 12
    batch: int = 8
    patience: int = 10
    workers: int = 8
    device: str | None = None


def train(s: YoloSettings) -> Path:
    from ultralytics import YOLO

    model = YOLO(s.model)
    model.train(
        data=str(s.data_yaml),
        imgsz=s.imgsz,
        epochs=s.epochs,
        batch=s.batch,
        patience=s.patience,
        workers=s.workers,
        device=s.device,
        project=str(s.output_dir.parent.resolve()),
        name=s.output_dir.name,
        exist_ok=True,
        plots=True,
    )
    return s.output_dir / "weights" / "best.pt"


def predict_clip(model, clip: Clip, out_path: Path, imgsz: int = 1280, threshold: float = 0.1, batch_size: int = 8,
                 max_frames: int | None = None, device: str | None = None) -> int:  # fmt: skip
    """Detect every frame of a clip and write RF-DETR's Parquet format. Returns the number of boxes."""
    frames = clip.frames[:max_frames] if max_frames else clip.frames
    precision = {} if device == "cpu" else {"quantize": "fp16"}
    rows: dict[str, list] = {name: [] for name in DETECTION_SCHEMA.names}
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        images = [Image.open(f.path).convert("RGB") for f in batch]
        results = model.predict(images, imgsz=imgsz, conf=threshold, verbose=False, device=device, **precision)
        for frame, r in zip(batch, results):
            boxes = r.boxes
            for (x1, y1, x2, y2), score, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                rows["clip"].append(clip.name)
                rows["frame"].append(frame.index)
                rows["x1"].append(float(x1))
                rows["y1"].append(float(y1))
                rows["x2"].append(float(x2))
                rows["y2"].append(float(y2))
                rows["score"].append(float(score))
                rows["class_name"].append(YOLO_CLASSES[int(cls)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    pq.write_table(pa.table(rows, schema=DETECTION_SCHEMA), tmp)
    tmp.replace(out_path)
    return len(rows["clip"])
