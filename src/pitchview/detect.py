"""Fine-tune RF-DETR on GSR and run it over clips.

Classes: player, goalkeeper, referee, ball (see pitchview.coco.CATEGORIES).
Predictions go to one Parquet file per clip, so an interrupted run can resume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from pitchview.coco import CLASS_NAMES
from pitchview.gsr import Clip

SIZES = ("nano", "small", "medium")

DETECTION_SCHEMA = pa.schema(
    [
        ("clip", pa.string()),
        ("frame", pa.int32()),
        ("x1", pa.float32()),
        ("y1", pa.float32()),
        ("x2", pa.float32()),
        ("y2", pa.float32()),
        ("score", pa.float32()),
        ("class_name", pa.string()),
    ]
)


def _model_class(size: str):
    import rfdetr  # heavy import; kept lazy so the CLI starts fast

    return {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[size]


@dataclass
class TrainSettings:
    dataset_dir: Path
    output_dir: Path
    size: str = "medium"
    resolution: int = 1120  # multiple of 32; the ball is only ~13 px wide in a 1920 px frame
    epochs: int = 20
    batch_size: int = 4
    grad_accum_steps: int = 4  # effective batch 16, the size RF-DETR recommends
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4  # RF-DETR's default for the backbone
    schedule: str = "step"  # RF-DETR's default; its first drop is at epoch 100, so the rate stays flat
    warmup_epochs: float = 0.0
    min_factor: float = 0.0  # cosine: the rate ends at lr * min_factor
    num_workers: int = 8
    early_stopping_patience: int = 5


def train(s: TrainSettings) -> None:
    model = _model_class(s.size)(resolution=s.resolution)
    model.train(
        dataset_dir=str(s.dataset_dir),
        output_dir=str(s.output_dir),
        epochs=s.epochs,
        batch_size=s.batch_size,
        grad_accum_steps=s.grad_accum_steps,
        lr=s.lr,
        lr_encoder=s.lr_encoder,
        lr_scheduler=s.schedule,
        lr_scheduler_kwargs={"min_factor": s.min_factor} if s.schedule == "cosine" else {},
        warmup_epochs=s.warmup_epochs,
        num_workers=s.num_workers,
        checkpoint_interval=1,  # a crash or a stopped pod loses at most one epoch
        log_per_class_metrics=True,  # the ball gets its own mAP
        early_stopping=True,
        early_stopping_patience=s.early_stopping_patience,
    )


def load(checkpoint: Path | None, size: str, resolution: int):
    """Load fine-tuned weights, or the COCO-pretrained model when checkpoint is None."""
    cls = _model_class(size)
    if checkpoint is None:
        return cls(resolution=resolution)
    return cls(pretrain_weights=str(checkpoint), resolution=resolution, num_classes=len(CLASS_NAMES))


def label_names(model) -> dict[int, str]:
    """Map the model's output class ids to our class names."""
    names = getattr(model, "class_names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)) and len(names) == len(CLASS_NAMES):
        return dict(enumerate(names))
    return {i: name for i, name in enumerate(CLASS_NAMES.values())}


def predict_clip(
    model,
    clip: Clip,
    out_path: Path,
    threshold: float = 0.1,  # keep low-score boxes: ByteTrack uses them in its second pass
    batch_size: int = 8,
    max_frames: int | None = None,
) -> int:
    """Detect every frame of a clip and write a Parquet file. Returns the number of boxes."""
    frames = clip.frames[:max_frames] if max_frames else clip.frames
    names = label_names(model)
    rows: dict[str, list] = {name: [] for name in DETECTION_SCHEMA.names}
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        images = [Image.open(f.path).convert("RGB") for f in batch]
        detections = model.predict(images, threshold=threshold)
        if not isinstance(detections, list):
            detections = [detections]
        for frame, det in zip(batch, detections):
            for (x1, y1, x2, y2), score, class_id in zip(det.xyxy, det.confidence, det.class_id):
                rows["clip"].append(clip.name)
                rows["frame"].append(frame.index)
                rows["x1"].append(float(x1))
                rows["y1"].append(float(y1))
                rows["x2"].append(float(x2))
                rows["y2"].append(float(y2))
                rows["score"].append(float(score))
                rows["class_name"].append(names.get(int(class_id), str(class_id)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    pq.write_table(pa.table(rows, schema=DETECTION_SCHEMA), tmp)
    tmp.replace(out_path)  # atomic: a half-written file never looks finished
    return len(rows["clip"])
