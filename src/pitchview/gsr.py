"""Read SoccerNet Game State Reconstruction (GSR) clips.

Each clip is 30 s of broadcast video (750 JPEG frames at 25 fps) plus one
Labels-GameState.json. For every frame it lists each person's and the ball's
box, role, team, jersey number and ground-truth position on the pitch.

Pitch coordinates are metres with the origin at the centre spot: x runs along
the length of the pitch (about ±52.5) and y across its width (about ±34).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

DEFAULT_ROOT = Path("data/SoccerNetGS/gamestate-2024")
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Frame:
    index: int  # 1-based, matches the JPEG file name
    path: Path
    width: int
    height: int
    has_labeled_person: bool
    has_ignore_regions: bool


@dataclass(frozen=True)
class Box:
    frame: int
    track_id: int | None
    category_id: int  # 1 player, 2 goalkeeper, 3 referee, 4 ball, 7 other
    role: str
    team: str | None  # "left" or "right"; None for referees and the ball
    jersey: str | None
    x: float  # top-left corner and size, in image pixels
    y: float
    w: float
    h: float
    pitch_x: float | None  # bottom-middle of the box on the pitch, in metres
    pitch_y: float | None


class Clip:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = self.path.name

    def __repr__(self) -> str:
        return f"Clip({self.name})"

    @cached_property
    def labels(self) -> dict:
        return json.loads((self.path / "Labels-GameState.json").read_text())

    @property
    def fps(self) -> float:
        return float(self.labels["info"].get("frame_rate", 25))

    @cached_property
    def frames(self) -> list[Frame]:
        frames_dir = self.path / self.labels["info"].get("im_dir", "img1")
        frames = [
            Frame(
                index=int(Path(im["file_name"]).stem),
                path=frames_dir / im["file_name"],
                width=int(im["width"]),
                height=int(im["height"]),
                has_labeled_person=bool(im.get("has_labeled_person")),
                has_ignore_regions=bool(im.get("ignore_regions_x")),
            )
            for im in self.labels["images"]
        ]
        return sorted(frames, key=lambda f: f.index)

    @cached_property
    def boxes(self) -> list[Box]:
        frame_of = {im["image_id"]: int(Path(im["file_name"]).stem) for im in self.labels["images"]}
        boxes = []
        for ann in self.labels["annotations"]:
            if ann.get("supercategory") != "object":
                continue  # pitch lines and camera parameters live in the same list
            attrs = ann.get("attributes") or {}
            bbox = ann["bbox_image"]
            pitch = ann.get("bbox_pitch") or {}
            boxes.append(
                Box(
                    frame=frame_of[ann["image_id"]],
                    track_id=ann.get("track_id"),
                    category_id=int(ann["category_id"]),
                    role=attrs.get("role") or "other",
                    team=attrs.get("team"),
                    jersey=attrs.get("jersey"),
                    x=float(bbox["x"]),
                    y=float(bbox["y"]),
                    w=float(bbox["w"]),
                    h=float(bbox["h"]),
                    pitch_x=pitch.get("x_bottom_middle"),
                    pitch_y=pitch.get("y_bottom_middle"),
                )
            )
        return boxes

    def boxes_by_frame(self) -> dict[int, list[Box]]:
        grouped: dict[int, list[Box]] = defaultdict(list)
        for box in self.boxes:
            grouped[box.frame].append(box)
        return dict(grouped)


def list_clips(split: str, root: Path = DEFAULT_ROOT) -> list[Clip]:
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"{split_dir} not found; run scripts/download_gsr.py first")
    return [Clip(p) for p in sorted(split_dir.iterdir()) if (p / "Labels-GameState.json").is_file()]
