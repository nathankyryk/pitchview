"""ByteTrack over RF-DETR detections: the specialist tracking baseline SAM 3.1 is compared with."""

from __future__ import annotations

import numpy as np
import pandas as pd
import supervision as sv
from trackers import ByteTrackTracker

PEOPLE = ("player", "goalkeeper", "referee")


def bytetrack(dets: pd.DataFrame, n_frames: int, fps: float = 25.0) -> pd.DataFrame:
    """Track people across one clip. Returns the input rows that joined a track, with `track_id`.

    The ball is not tracked here: one ball per frame is picked by score elsewhere.
    Every frame is fed to the tracker, including empty ones, so lost tracks age properly.
    """
    people = dets[dets["class_name"].isin(PEOPLE)]
    by_frame = {frame: d for frame, d in people.groupby("frame")}
    tracker = ByteTrackTracker(
        frame_rate=fps,
        lost_track_buffer=int(fps * 2),  # keep a lost player for 2 s (occlusions, camera cuts)
        track_activation_threshold=0.5,
        high_conf_det_threshold=0.5,  # boxes below this only extend existing tracks (ByteTrack's 2nd pass)
        minimum_consecutive_frames=1,
    )
    kept = []
    for frame in range(1, n_frames + 1):
        d = by_frame.get(frame)
        if d is None:
            tracker.update(sv.Detections.empty())
            continue
        detections = sv.Detections(
            xyxy=d[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32),
            confidence=d["score"].to_numpy(dtype=np.float32),
            class_id=np.zeros(len(d), dtype=int),  # one class: the role comes from the detector row
            data={"row": d.index.to_numpy()},
        )
        tracked = tracker.update(detections)
        confirmed = tracked[tracked.tracker_id >= 0] if len(tracked) else tracked
        if len(confirmed):
            rows = people.loc[confirmed.data["row"]].copy()
            rows["track_id"] = confirmed.tracker_id
            kept.append(rows)
    if not kept:
        return people.iloc[0:0].assign(track_id=pd.Series(dtype=int))
    return pd.concat(kept)
