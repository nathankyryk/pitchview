"""People masks and tracks from SAM 3.1, with a CPU stand-in for smoke tests.

One SAM 3.1 detail shapes this module. In this release a *box* prompt is a concept
exemplar ("find everything like this box"), and prompts for a single object accept
points only (see sam3_multiplex_tracking.py, add_prompt). So we run SAM 3.1 the way it
is designed: one text prompt per clip ("person"). It detects, segments and tracks every
person by itself, which is also the zero-shot side of the foundation-vs-specialist
comparison. RF-DETR's boxes then borrow these masks through IoU matching (attach_masks).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment

from pitchview.gsr import Clip

# SAM's per-frame output: out_obj_ids (N,), out_probs (N,), out_binary_masks (N, H, W) bool
FrameOutput = dict

TRACK_SCHEMA = pa.schema(
    [
        ("clip", pa.string()),
        ("frame", pa.int32()),
        ("obj_id", pa.int32()),
        ("prob", pa.float32()),
        ("x1", pa.float32()),
        ("y1", pa.float32()),
        ("x2", pa.float32()),
        ("y2", pa.float32()),
        ("foot_x", pa.float32()),
        ("foot_y", pa.float32()),
        ("area", pa.int32()),
    ]
)


def mask_features(mask: np.ndarray) -> tuple[float, float, float, float, float, float, int] | None:
    """Box, foot point and pixel area of one binary mask, or None for an empty mask.

    The foot point is the median x of the lowest few rows of the mask: where the
    player touches the grass. A box's bottom-centre is often off the body.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    top, bottom = int(ys.min()), int(ys.max())
    band = max(2, int(0.03 * (bottom - top + 1)))
    lowest = ys >= bottom - band
    foot_x = float(np.median(xs[lowest]))
    return float(xs.min()), float(top), float(xs.max() + 1), float(bottom + 1), foot_x, float(bottom + 1), int(ys.size)


def drop_offload_state_arg(model) -> None:
    """Work around a mismatch in sam3 (commit 7f473cfd): the shared start_session always
    passes offload_state_to_cpu, but the SAM 3.1 multiplex model's init_state doesn't take
    it, so every session crashes on start. Accept the argument and drop it."""
    init_state = model.init_state

    def init_state_compat(*args, offload_state_to_cpu=False, **kwargs):
        return init_state(*args, **kwargs)

    model.init_state = init_state_compat


class Sam31:
    """SAM 3.1 multiplex video predictor. CUDA only, so it runs on the pod."""

    def __init__(self, checkpoint: Path, max_num_objects: int = 48, detect_batch: int | None = None):
        import torch
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        if detect_batch is None:
            # Meta's 16-frame detection batches on a big card; 4 on a 24 GB one.
            detect_batch = 16 if torch.cuda.get_device_properties(0).total_memory >= 40e9 else 4

        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint),
            max_num_objects=max_num_objects,  # the default (16) is below a busy frame's head count
            use_fa3=False,  # FlashAttention 3 needs Hopper (H100); the RTX 4090 is Ada
        )
        drop_offload_state_arg(self.predictor.model)
        # The builder detects 16 frames per batch, sized for an 80 GB card: on the 24 GB
        # RTX 4090 the second batch ran out of memory while the first was still cached.
        self.predictor.model.batched_grounding_batch_size = detect_batch

    def track(
        self, clip: Clip, prompt: str = "person", prob_thresh: float = 0.5, max_frames: int | None = None
    ) -> Iterator[tuple[int, FrameOutput]]:
        """Yield (GSR frame index, SAM output) for each frame, in order."""
        frames_dir = clip.frames[0].path.parent
        start = self.predictor.handle_request(
            {"type": "start_session", "resource_path": str(frames_dir), "offload_video_to_cpu": True}
        )
        session_id = start["session_id"]
        try:
            self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt,
                    "output_prob_thresh": prob_thresh,
                }
            )
            request = {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": 0,
                "output_prob_thresh": prob_thresh,
                "evict_cached_frame_outputs": True,  # keeps memory flat over 750 frames
            }
            if max_frames:
                request["max_frame_num_to_track"] = max_frames - 1  # SAM's range is inclusive
            for response in self.predictor.handle_stream_request(request):
                yield response["frame_index"] + 1, response["outputs"]  # SAM counts frames from 0, GSR from 1
        finally:
            self.predictor.handle_request({"type": "close_session", "session_id": session_id})


class GroundTruthTracks:
    """CPU stand-in with SAM's output shape: GSR's labelled people as rectangle masks.

    It lets every downstream step run in --smoke mode on a laptop, and doubles as an
    oracle ("perfect masks and tracks") when attributing errors.
    """

    def track(
        self, clip: Clip, prompt: str = "person", prob_thresh: float = 0.5, max_frames: int | None = None
    ) -> Iterator[tuple[int, FrameOutput]]:
        by_frame = clip.boxes_by_frame()
        frames = clip.frames[:max_frames] if max_frames else clip.frames
        for frame in frames:
            people = [b for b in by_frame.get(frame.index, []) if b.role != "ball" and b.track_id is not None]
            masks = np.zeros((len(people), frame.height, frame.width), dtype=bool)
            for i, b in enumerate(people):
                x0, y0 = max(0, int(b.x)), max(0, int(b.y))
                x1, y1 = min(frame.width, int(b.x + b.w)), min(frame.height, int(b.y + b.h))
                masks[i, y0:y1, x0:x1] = True
            yield frame.index, {
                "out_obj_ids": np.array([b.track_id for b in people], dtype=np.int64),
                "out_probs": np.ones(len(people), dtype=np.float32),
                "out_binary_masks": masks,
            }


class MaskWriter:
    """A `write_tracks` hook that keeps every mask of a clip as COCO run-length encoding.

    Raw masks are ~50 MB per frame, RLE a few KB, so every test clip can keep them.
    """

    def __init__(self):
        self.rows: list[dict] = []

    def __call__(self, frame_index: int, out: FrameOutput) -> None:
        from pycocotools import mask as mask_utils

        masks = np.asarray(out["out_binary_masks"])
        if masks.shape[0] == 0:
            return
        rles = mask_utils.encode(np.asfortranarray(masks.transpose(1, 2, 0).astype(np.uint8)))
        for obj_id, rle in zip(out["out_obj_ids"], rles):
            self.rows.append(
                {"frame": frame_index, "obj_id": int(obj_id), "size": rle["size"], "counts": rle["counts"].decode()}
            )

    def save(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in self.rows) + "\n")


def write_tracks(
    tracker,
    clip: Clip,
    out_path: Path,
    prompt: str = "person",
    max_frames: int | None = None,
    on_frame: Callable[[int, FrameOutput], None] | None = None,
) -> int:
    """Run a tracker over a clip and write one row per object per frame. Returns the row count.

    `on_frame` sees every frame's raw masks while they are in memory, which is where
    jersey embeddings and demo masks are computed (masks are too big to keep).
    """
    rows: dict[str, list] = {name: [] for name in TRACK_SCHEMA.names}
    for frame_index, out in tracker.track(clip, prompt=prompt, max_frames=max_frames):
        for obj_id, prob, mask in zip(out["out_obj_ids"], out["out_probs"], out["out_binary_masks"]):
            features = mask_features(mask)
            if features is None:
                continue
            x1, y1, x2, y2, foot_x, foot_y, area = features
            for name, value in zip(
                TRACK_SCHEMA.names,
                (clip.name, frame_index, int(obj_id), float(prob), x1, y1, x2, y2, foot_x, foot_y, area),
            ):
                rows[name].append(value)
        if on_frame is not None:
            on_frame(frame_index, out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    pq.write_table(pa.table(rows, schema=TRACK_SCHEMA), tmp)
    tmp.replace(out_path)
    return len(rows["clip"])


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) boxes in x1, y1, x2, y2 order."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


BOX = ["x1", "y1", "x2", "y2"]


def attach_masks(dets: pd.DataFrame, tracks: pd.DataFrame, min_iou: float = 0.5) -> pd.DataFrame:
    """Give each detection the foot point of the SAM mask it matches (one-to-one, IoU >= min_iou).

    Unmatched detections, and the ball, keep the box's bottom-centre. `has_mask` records
    which, so the box-vs-mask ablation can compare like with like.
    """
    out = dets.copy()
    out["foot_x"] = (out["x1"] + out["x2"]) / 2
    out["foot_y"] = out["y2"]
    out["has_mask"] = False
    out["sam_obj_id"] = -1
    tracks_by_frame = {frame: t for frame, t in tracks.groupby("frame")}
    for frame, d in out[out["class_name"] != "ball"].groupby("frame"):
        t = tracks_by_frame.get(frame)
        if t is None or t.empty:
            continue
        iou = box_iou(d[BOX].to_numpy(dtype=float), t[BOX].to_numpy(dtype=float))
        rows, cols = linear_sum_assignment(-iou)
        good = iou[rows, cols] >= min_iou
        det_idx, trk = d.index[rows[good]], t.iloc[cols[good]]
        out.loc[det_idx, "foot_x"] = trk["foot_x"].to_numpy()
        out.loc[det_idx, "foot_y"] = trk["foot_y"].to_numpy()
        out.loc[det_idx, "sam_obj_id"] = trk["obj_id"].to_numpy()
        out.loc[det_idx, "has_mask"] = True
    return out
