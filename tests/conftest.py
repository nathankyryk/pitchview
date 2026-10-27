"""A tiny fake GSR clip, so tests run without the 28 GB dataset."""

import json
from pathlib import Path

import pytest
from PIL import Image

WIDTH, HEIGHT = 64, 36


def _object(ann_id, frame, cat, role, x, y, w, h, team=None, pitch=(1.0, 2.0)):
    return {
        "id": str(ann_id),
        "image_id": f"99900000{frame}",
        "track_id": ann_id,
        "supercategory": "object",
        "category_id": cat,
        "attributes": {"role": role, "jersey": "7" if role == "player" else None, "team": team},
        "bbox_image": {"x": x, "y": y, "w": w, "h": h},
        "bbox_pitch": {"x_bottom_middle": pitch[0], "y_bottom_middle": pitch[1]},
    }


@pytest.fixture
def gsr_root(tmp_path: Path) -> Path:
    """One valid-split clip with 6 frames; frame 3 has an ignore region."""
    clip = tmp_path / "valid" / "SNGS-999"
    (clip / "img1").mkdir(parents=True)
    images = []
    for i in range(1, 7):
        Image.new("RGB", (WIDTH, HEIGHT), "green").save(clip / "img1" / f"{i:06d}.jpg")
        images.append(
            {
                "image_id": f"99900000{i}",
                "file_name": f"{i:06d}.jpg",
                "width": WIDTH,
                "height": HEIGHT,
                "has_labeled_person": True,
                "ignore_regions_x": [[1, 2, 3]] if i == 3 else [],
                "ignore_regions_y": [[1, 2, 3]] if i == 3 else [],
            }
        )
    annotations = []
    for i in range(1, 7):
        annotations += [
            _object(10 * i + 1, i, 1, "player", 10, 5, 8, 20, team="left", pitch=(-3.5, 12.0)),
            _object(10 * i + 2, i, 4, "ball", 30, 10, 2, 2, pitch=(0.0, 0.0)),
            _object(10 * i + 3, i, 7, "other", 40, 5, 6, 15),
            _object(10 * i + 4, i, 3, "referee", 60, 30, 10, 10),  # runs off the right and bottom edges
            _object(10 * i + 5, i, 1, "player", 20, 20, 0.5, 10),  # degenerate: under a pixel wide
        ]
    annotations.append(
        {"id": "pitch1", "image_id": "999000001", "supercategory": "pitch", "category_id": 5, "lines": {}}
    )
    labels = {
        "info": {"name": "SNGS-999", "im_dir": "img1", "frame_rate": 25, "seq_length": 6},
        "images": images,
        "annotations": annotations,
        "categories": [],
    }
    (clip / "Labels-GameState.json").write_text(json.dumps(labels))
    return tmp_path
