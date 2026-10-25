"""Pitch calibration: keypoints on the field markings -> homography -> metres.

Coordinates follow GSR: metres, origin at the centre spot, x along the length (left goal
line at -52.5) and y across the width (far touchline at -34, camera side at +34).

Keypoints come from Roboflow's pretrained pitch model: 32 points (corners, box corners,
penalty spots, centre circle). We keep its point order but place the points on a real
105 x 68 m FIFA pitch. Roboflow's own template is 120 x 70 m with a 20.15 m-deep penalty
box, which would bend every position.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pitchview.gsr import Clip

LENGTH, WIDTH = 105.0, 68.0
PENALTY_BOX_LENGTH, PENALTY_BOX_WIDTH = 16.5, 40.32
GOAL_BOX_LENGTH, GOAL_BOX_WIDTH = 5.5, 18.32
CENTRE_CIRCLE_RADIUS = 9.15
PENALTY_SPOT = 11.0
N_KEYPOINTS = 32


def _template() -> np.ndarray:
    """The 32 keypoints in Roboflow's order, in metres from the far-left corner, then centred."""
    L, W = LENGTH, WIDTH
    pbl, pbw = PENALTY_BOX_LENGTH, PENALTY_BOX_WIDTH
    gbl, gbw = GOAL_BOX_LENGTH, GOAL_BOX_WIDTH
    r, spot = CENTRE_CIRCLE_RADIUS, PENALTY_SPOT
    corner_origin = [
        (0, 0), (0, (W - pbw) / 2), (0, (W - gbw) / 2), (0, (W + gbw) / 2), (0, (W + pbw) / 2), (0, W),
        (gbl, (W - gbw) / 2), (gbl, (W + gbw) / 2),
        (spot, W / 2),
        (pbl, (W - pbw) / 2), (pbl, (W - gbw) / 2), (pbl, (W + gbw) / 2), (pbl, (W + pbw) / 2),
        (L / 2, 0), (L / 2, W / 2 - r), (L / 2, W / 2 + r), (L / 2, W),
        (L - pbl, (W - pbw) / 2), (L - pbl, (W - gbw) / 2), (L - pbl, (W + gbw) / 2), (L - pbl, (W + pbw) / 2),
        (L - spot, W / 2),
        (L - gbl, (W - gbw) / 2), (L - gbl, (W + gbw) / 2),
        (L, 0), (L, (W - pbw) / 2), (L, (W - gbw) / 2), (L, (W + gbw) / 2), (L, (W + pbw) / 2), (L, W),
        (L / 2 - r, W / 2), (L / 2 + r, W / 2),
    ]  # fmt: skip
    return np.array(corner_origin, dtype=np.float64) - (L / 2, W / 2)


PITCH_KEYPOINTS = _template()  # (32, 2), metres, GSR coordinates


def fit_homography(image_xy: np.ndarray, conf: np.ndarray, min_conf: float = 0.5) -> np.ndarray | None:
    """Image pixels -> pitch metres, from the confident keypoints. None if fewer than 4 survive."""
    keep = conf >= min_conf
    if keep.sum() < 4:
        return None
    H, _ = cv2.findHomography(
        image_xy[keep].astype(np.float64),
        PITCH_KEYPOINTS[keep],
        cv2.RANSAC,
        1.0,  # inlier tolerance, in metres on the pitch
    )
    return H


def to_pitch(H: np.ndarray, image_xy: np.ndarray) -> np.ndarray:
    """Map (N, 2) image points to (N, 2) pitch metres."""
    pts = np.asarray(image_xy, dtype=np.float64).reshape(-1, 1, 2)
    if pts.shape[0] == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def to_image(H: np.ndarray, pitch_xy: np.ndarray) -> np.ndarray:
    """Map (N, 2) pitch metres back to image pixels (inverse of to_pitch)."""
    return to_pitch(np.linalg.inv(H), pitch_xy)


def oracle_homography(people, min_people: int = 8, min_spread_m: float = 15.0) -> np.ndarray | None:
    """Exact image->pitch homography from GSR's ground truth for one frame.

    Each labelled person gives a correspondence: box bottom-centre in the image, and GSR's
    pitch position of that same point. GSR's positions are metric and planar-consistent
    (residual ~0.1 m), so with enough well-spread people this recovers the camera's mapping.
    Returns None when the people are too few or too bunched to pin it down.
    """
    people = [b for b in people if b.role != "ball" and b.pitch_x is not None]
    if len(people) < min_people:
        return None
    truth = np.array([[b.pitch_x, b.pitch_y] for b in people], dtype=np.float64)
    if np.ptp(truth[:, 0]) < min_spread_m and np.ptp(truth[:, 1]) < min_spread_m:
        return None
    feet = np.array([[b.x + b.w / 2, b.y + b.h] for b in people], dtype=np.float64)
    H, _ = cv2.findHomography(feet, truth, 0)
    return H


def keypoint_labels(H: np.ndarray, width: int, height: int, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The 32 template keypoints projected into the image, and whether each one is visible."""
    xy = to_image(H, PITCH_KEYPOINTS)
    visible = (
        (xy[:, 0] >= -margin) & (xy[:, 0] < width + margin) & (xy[:, 1] >= -margin) & (xy[:, 1] < height + margin)
    )
    return xy, visible


def frame_homographies(keypoints, min_conf: float = 0.5) -> dict[int, np.ndarray]:
    """Fit one homography per frame from a keypoints table (columns frame, x, y, conf)."""
    out = {}
    for row in keypoints.itertuples(index=False):
        H = fit_homography(np.column_stack([row.x, row.y]), np.asarray(row.conf), min_conf)
        if H is not None:
            out[int(row.frame)] = H
    return out


def nearest_homography(Hs: dict[int, np.ndarray], frame: int, max_gap: int = 12) -> np.ndarray | None:
    """The homography of `frame`, or of the closest frame within max_gap (camera moves slowly)."""
    if frame in Hs:
        return Hs[frame]
    for gap in range(1, max_gap + 1):
        for f in (frame - gap, frame + gap):
            if f in Hs:
                return Hs[f]
    return None


KEYPOINT_SCHEMA = pa.schema(
    [
        ("clip", pa.string()),
        ("frame", pa.int32()),
        ("x", pa.list_(pa.float32(), N_KEYPOINTS)),
        ("y", pa.list_(pa.float32(), N_KEYPOINTS)),
        ("conf", pa.list_(pa.float32(), N_KEYPOINTS)),
    ]
)


class PitchModel:
    """Roboflow's YOLOv8-pose pitch-keypoint model."""

    def __init__(self, weights: Path, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device

    def keypoints(self, paths: list[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per image: (32, 2) pixel positions and (32,) confidences. Zeros when nothing is found."""
        results = self.model.predict([str(p) for p in paths], verbose=False, device=self.device)
        out = []
        for r in results:
            kp = r.keypoints
            if kp is None or kp.xy.shape[0] == 0 or kp.conf is None:
                out.append((np.zeros((N_KEYPOINTS, 2)), np.zeros(N_KEYPOINTS)))
                continue
            best = int(r.boxes.conf.argmax()) if r.boxes is not None and len(r.boxes) else 0
            out.append((kp.xy[best].cpu().numpy(), kp.conf[best].cpu().numpy()))
        return out


def write_keypoints(model: PitchModel, clip: Clip, out_path: Path, batch_size: int = 16, max_frames: int | None = None) -> int:
    """Store raw keypoints per frame. Homographies are fitted later, on the laptop, where
    thresholds and smoothing can be tuned without paying for the GPU again."""
    frames = clip.frames[:max_frames] if max_frames else clip.frames
    rows: dict[str, list] = {name: [] for name in KEYPOINT_SCHEMA.names}
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        for frame, (xy, conf) in zip(batch, model.keypoints([f.path for f in batch])):
            rows["clip"].append(clip.name)
            rows["frame"].append(frame.index)
            rows["x"].append(xy[:, 0].tolist())
            rows["y"].append(xy[:, 1].tolist())
            rows["conf"].append(conf.tolist())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    pq.write_table(pa.table(rows, schema=KEYPOINT_SCHEMA), tmp)
    tmp.replace(out_path)
    return len(rows["clip"])
