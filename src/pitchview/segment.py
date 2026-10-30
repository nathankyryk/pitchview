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
RESTART_ID_STEP = 10_000  # ids after a SAM 3.1 restart (see Sam31.track); a clip creates a few hundred at most
MAX_RESTARTS = 20

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


def clean_mask(mask: np.ndarray, near: float = 0.25, min_share: float = 0.1) -> np.ndarray:
    """The person without SAM's stray pieces: the largest connected part, plus any part at least
    `min_share` of its size that touches its box grown by `near` x its height (a leg cut off by
    an occluder). A third of SAM 3.1's masks carry a stray piece elsewhere in the frame, which
    stretches the box, moves the foot point and spoils the crops."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = 1 + int(np.argmax(areas))
    x, y, w, h = stats[big, :4]
    pad = near * h
    keep = [big]
    for i in range(1, count):
        px, py, pw, ph, area = stats[i]
        touches = px < x + w + pad and px + pw > x - pad and py < y + h + pad and py + ph > y - pad
        if i != big and touches and area >= min_share * areas[big - 1]:
            keep.append(i)
    return np.isin(labels, keep)


def mask_features(mask: np.ndarray) -> tuple[float, float, float, float, float, float, int] | None:
    """Box, foot point and pixel area of one binary mask (stray pieces removed first, see
    clean_mask), or None for an empty mask.

    The foot point is the median x of the lowest few rows of the mask: where the
    player touches the grass. A box's bottom-centre is often off the body.
    """
    return _features(clean_mask(mask))


def _features(mask: np.ndarray) -> tuple[float, float, float, float, float, float, int] | None:
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

    def __init__(self, checkpoint: Path, max_num_objects: int = 128, detect_batch: int | None = None):
        import torch
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        if detect_batch is None:
            # Meta's 16-frame detection batches on a big card; 4 on a 24 GB one.
            detect_batch = 16 if torch.cuda.get_device_properties(0).total_memory >= 40e9 else 4

        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint),
            # The builder's default (16) is below a busy frame's head count. SAM 3.1 never lets go of a
            # player who leaves the picture, and once it holds this many tracks it drops new people: at 48,
            # 38 of 49 test clips filled up (SNGS-145 lost everyone after a pan). 128 is sam3's own default.
            max_num_objects=max_num_objects,
            use_fa3=False,  # FlashAttention 3 needs Hopper (H100); the RTX 4090 is Ada
        )
        drop_offload_state_arg(self.predictor.model)
        # The builder detects 16 frames per batch, sized for an 80 GB card: on the 24 GB
        # RTX 4090 the second batch ran out of memory while the first was still cached.
        self.predictor.model.batched_grounding_batch_size = detect_batch

    def track(
        self, clip: Clip, prompt: str = "person", prob_thresh: float = 0.5, max_frames: int | None = None
    ) -> Iterator[tuple[int, FrameOutput]]:
        """Yield (GSR frame index, SAM output) for each frame, in order.

        SAM 3.1 can crash part-way through a clip ("No points are provided"): when it drops a
        player, it can also clear the first frame that the other players in that group are
        tracked from (SNGS-130, frame 428, at the 128-track limit). A new session then starts
        on the frame that crashed, with ids moved up by RESTART_ID_STEP so they can't clash
        with the old ones. assemble's hand-off linking joins each player's two pieces."""
        n_frames = min(max_frames, len(clip.frames)) if max_frames else len(clip.frames)
        last = n_frames - 1 if max_frames else None  # None: SAM runs to the end of the clip
        start, restarts = 0, 0
        while start < n_frames:
            reached = start
            try:
                for sam_index, out in self._session(clip, prompt, prob_thresh, start, last):
                    reached = sam_index + 1
                    if restarts:
                        out = {**out, "out_obj_ids": np.asarray(out["out_obj_ids"]) + restarts * RESTART_ID_STEP}
                    yield sam_index + 1, out  # SAM counts frames from 0, GSR from 1
                return
            except RuntimeError as error:
                if "No points are provided" not in str(error) or restarts >= MAX_RESTARTS:
                    raise
            restarts += 1
            start = reached if reached > start else start + 1  # no frame came out: skip the one that crashed
            print(f"  {clip.name}: SAM 3.1 crashed; new session from frame {start + 1}", flush=True)

    def _session(
        self, clip: Clip, prompt: str, prob_thresh: float, start: int, last: int | None
    ) -> Iterator[tuple[int, FrameOutput]]:
        """One SAM session: the text prompt on frame `start` (SAM counts from 0), then forward to
        frame `last` (the clip's end when None). Yields (SAM frame index, output)."""
        frames_dir = clip.frames[0].path.parent
        session_id = self.predictor.handle_request(
            {"type": "start_session", "resource_path": str(frames_dir), "offload_video_to_cpu": True}
        )["session_id"]
        try:
            self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": start,
                    "text": prompt,
                    "output_prob_thresh": prob_thresh,
                }
            )
            request = {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": start,
                "output_prob_thresh": prob_thresh,
                "evict_cached_frame_outputs": True,  # keeps memory flat over 750 frames
            }
            if last is not None:
                request["max_frame_num_to_track"] = last - start  # SAM's range is inclusive
            for response in self.predictor.handle_stream_request(request):
                yield response["frame_index"], response["outputs"]
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
            yield (
                frame.index,
                {
                    "out_obj_ids": np.array([b.track_id for b in people], dtype=np.int64),
                    "out_probs": np.ones(len(people), dtype=np.float32),
                    "out_binary_masks": masks,
                },
            )


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


def clean_clip(
    raw_tracks: Path, raw_masks: Path, out_tracks: Path, out_masks: Path,
    raw_emb: Path | None = None, out_emb: Path | None = None,
) -> tuple[int, int]:  # fmt: skip
    """Re-derive one clip's saved SAM tracks and masks with clean_mask: new boxes, foot points and
    areas, and cleaned masks. Jersey embeddings (raw_emb -> out_emb) were cropped from the raw
    mask's box, so those of masks whose box changed are dropped. Returns (masks read, masks that
    lost a piece)."""
    import json

    from pycocotools import mask as mask_utils

    tracks = pq.read_table(raw_tracks).to_pandas()
    row_of = {(int(f), int(o)): i for i, (f, o) in enumerate(zip(tracks["frame"], tracks["obj_id"]))}
    cols = ["x1", "y1", "x2", "y2", "foot_x", "foot_y", "area"]
    values = tracks[cols].to_numpy(dtype=float)
    keep = np.ones(len(tracks), dtype=bool)
    lines, n, changed = [], 0, 0
    with Path(raw_masks).open() as f:
        for line in f:
            r = json.loads(line)
            raw = mask_utils.decode({"size": r["size"], "counts": r["counts"]}).astype(bool)
            mask = clean_mask(raw)
            n += 1
            if mask.sum() != raw.sum():
                changed += 1
                r["counts"] = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode()
            lines.append(json.dumps(r))
            i = row_of.get((int(r["frame"]), int(r["obj_id"])))
            if i is not None:
                features = _features(mask)
                if features is None:
                    keep[i] = False
                else:
                    values[i] = features
    moved = np.abs(values[:, :4] - tracks[cols[:4]].to_numpy(dtype=float)).max(axis=1) > 1
    stale = set(zip(tracks.loc[moved, "frame"].astype(int), tracks.loc[moved, "obj_id"].astype(int)))
    tracks[cols] = values
    tracks["area"] = tracks["area"].astype("int32")
    tracks = tracks[keep]
    out_tracks.parent.mkdir(parents=True, exist_ok=True)
    out_masks.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(tracks, schema=TRACK_SCHEMA, preserve_index=False), out_tracks)
    Path(out_masks).write_text("\n".join(lines) + "\n")
    if raw_emb is not None and out_emb is not None and Path(raw_emb).exists():
        emb = dict(np.load(raw_emb))
        ok = np.array([(int(f), int(o)) not in stale for f, o in zip(emb["frame"], emb["obj_id"])], dtype=bool)
        np.savez_compressed(out_emb, **{k: v[ok] for k, v in emb.items()})
    return n, changed


def write_tracks(
    tracker,
    clip: Clip,
    out_path: Path,
    prompt: str = "person",
    max_frames: int | None = None,
    on_frame: Callable[[int, FrameOutput], None] | None = None,
) -> int:
    """Run a tracker over a clip and write one row per object per frame. Returns the row count.

    `on_frame` sees every frame's masks while they are in memory, which is where
    jersey embeddings and demo masks are computed (masks are too big to keep).
    Masks are cleaned first (clean_mask), so features, embeddings and saved masks agree.
    """
    rows: dict[str, list] = {name: [] for name in TRACK_SCHEMA.names}
    for frame_index, out in tracker.track(clip, prompt=prompt, max_frames=max_frames):
        masks = [clean_mask(m) for m in np.asarray(out["out_binary_masks"])]
        out = {**out, "out_binary_masks": np.stack(masks) if masks else out["out_binary_masks"]}
        for obj_id, prob, mask in zip(out["out_obj_ids"], out["out_probs"], out["out_binary_masks"]):
            features = _features(mask)
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
