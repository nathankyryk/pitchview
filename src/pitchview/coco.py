"""Convert GSR clips to the COCO layout RF-DETR trains on.

Writes <out>/_annotations.coco.json next to the selected frames. Frames are
hard-linked when possible, so the converted set takes no extra disk space.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from pitchview.gsr import Box, list_clips

# Same layout as a Roboflow COCO export: an unannotated parent category (id 0)
# plus the real classes. RF-DETR drops the parent and maps ids 1-4 to labels 0-3.
# GSR uses the same ids for these classes; its id 7 ("other") is dropped.
CATEGORIES = [
    {"id": 0, "name": "soccer", "supercategory": "none"},
    {"id": 1, "name": "player", "supercategory": "soccer"},
    {"id": 2, "name": "goalkeeper", "supercategory": "soccer"},
    {"id": 3, "name": "referee", "supercategory": "soccer"},
    {"id": 4, "name": "ball", "supercategory": "soccer"},
]
CLASS_NAMES = {c["id"]: c["name"] for c in CATEGORIES if c["id"] != 0}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:  # different drive, or a filesystem without hard links
        shutil.copy2(src, dst)


def clip_box(box: Box, width: int, height: int) -> list[float] | None:
    """Clip a box to the image; None if less than a pixel survives."""
    x0, y0 = max(0.0, box.x), max(0.0, box.y)
    x1, y1 = min(float(width), box.x + box.w), min(float(height), box.y + box.h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def convert_split(root: Path, split: str, out: Path, stride: int = 5) -> Counter:
    """Convert one split, keeping every `stride`-th frame (neighbours are near-duplicates)."""
    out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    images: list[dict] = []
    annotations: list[dict] = []

    for clip in list_clips(split, root):
        stats["clips"] += 1
        by_frame = clip.boxes_by_frame()
        for frame in clip.frames:
            if (frame.index - 1) % stride or not frame.has_labeled_person:
                continue
            if frame.has_ignore_regions:
                # Unlabeled people sit inside ignore regions; training on them would teach misses.
                stats["frames_skipped_ignore_region"] += 1
                continue
            image_id = len(images) + 1
            file_name = f"{clip.name}_{frame.path.name}"
            link_or_copy(frame.path, out / file_name)
            images.append({"id": image_id, "file_name": file_name, "width": frame.width, "height": frame.height})

            for box in by_frame.get(frame.index, []):
                if box.category_id not in CLASS_NAMES:
                    stats["boxes_dropped_other"] += 1
                    continue
                bbox = clip_box(box, frame.width, frame.height)
                if bbox is None:
                    stats["boxes_dropped_degenerate"] += 1
                    continue
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": box.category_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                stats[f"boxes_{CLASS_NAMES[box.category_id]}"] += 1

    stats["frames"] = len(images)
    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    (out / "_annotations.coco.json").write_text(json.dumps(coco))
    return stats


def make_subset(src: Path, dst: Path, n_images: int, seed: int = 0) -> int:
    """Copy a small random sample of a converted split, for smoke tests. Returns the image count."""
    coco = json.loads((src / "_annotations.coco.json").read_text())
    k = min(n_images, len(coco["images"]))
    images = sorted(random.Random(seed).sample(coco["images"], k=k), key=lambda im: im["id"])
    keep = {im["id"] for im in images}
    dst.mkdir(parents=True, exist_ok=True)
    for im in images:
        link_or_copy(src / im["file_name"], dst / im["file_name"])
    subset = {
        "images": images,
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
        "categories": coco["categories"],
    }
    (dst / "_annotations.coco.json").write_text(json.dumps(subset))
    return len(images)
