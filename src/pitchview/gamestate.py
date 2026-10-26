"""Assemble the game state of one clip: who is where on the pitch, with role, team and number.

Inputs are the stage outputs (RF-DETR detections, SAM 3.1 tracks, pitch keypoints, jersey
embeddings, jersey reads). `Variant` switches each design choice, so every row of the
results table is the same code with different switches.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitchview.evaluate import prediction
from pitchview.pitch import nearest_homography, to_pitch
from pitchview.segment import attach_masks, box_iou
from pitchview.teams import cluster_teams, name_sides
from pitchview.track import bytetrack

BOX = ["x1", "y1", "x2", "y2"]


@dataclass(frozen=True)
class Variant:
    tracker: str = "bytetrack"  # "bytetrack" on RF-DETR boxes, or "sam" (SAM 3.1's own tracks)
    foot: str = "mask"  # where the player touches the grass: "mask" (lowest mask pixels) or "box" (bottom-centre)
    team_crops: str = "mask"  # jersey embeddings from "mask" or "box" crops
    jersey: bool = True
    smooth: int = 25  # rolling-median window over a track's positions, in frames; 1 = off (25 won on valid x5)


def _majority(values) -> str | None:
    values = [v for v in values if v is not None]
    return Counter(values).most_common(1)[0][0] if values else None


def _people(dets: pd.DataFrame, sam: pd.DataFrame, n_frames: int, fps: float, v: Variant) -> pd.DataFrame:
    """Rows of (frame, track_id, role, box, foot point, sam_obj_id) for the chosen tracker."""
    if v.tracker == "bytetrack":
        people = attach_masks(bytetrack(dets, n_frames, fps), sam)
        people["role"] = people["class_name"]
        if v.foot == "box":
            people["foot_x"] = (people["x1"] + people["x2"]) / 2
            people["foot_y"] = people["y2"]
        return people
    # SAM 3.1 tracks: masks give the foot point; roles come from overlapping RF-DETR boxes.
    people = sam.rename(columns={"obj_id": "track_id"}).copy()
    people["sam_obj_id"] = people["track_id"]
    people["role"] = None
    dets_by_frame = {f: d for f, d in dets[dets["class_name"] != "ball"].groupby("frame")}
    for frame, rows in people.groupby("frame"):
        d = dets_by_frame.get(frame)
        if d is None:
            continue
        iou = box_iou(rows[BOX].to_numpy(float), d[BOX].to_numpy(float))
        best = iou.argmax(1)
        ok = iou.max(1) >= 0.5
        people.loc[rows.index[ok], "role"] = d["class_name"].to_numpy()[best[ok]]
    if v.foot == "box":
        people["foot_x"] = (people["x1"] + people["x2"]) / 2
        people["foot_y"] = people["y2"]
    return people


def _to_metres(people: pd.DataFrame, homographies: dict[int, np.ndarray]) -> pd.DataFrame:
    xs, ys = np.full(len(people), np.nan), np.full(len(people), np.nan)
    for frame, idx in people.groupby("frame").indices.items():
        H = nearest_homography(homographies, int(frame))
        if H is not None:
            pts = to_pitch(H, people.iloc[idx][["foot_x", "foot_y"]].to_numpy(float))
            xs[idx], ys[idx] = pts[:, 0], pts[:, 1]
    out = people.assign(x_m=xs, y_m=ys)
    return out.dropna(subset=["x_m", "y_m"])


def _teams(people: pd.DataFrame, embeddings: dict | None, v: Variant) -> dict[int, str]:
    """Players: SigLIP clusters named by side. Goalkeepers: the half they stand in."""
    teams: dict[int, str] = {}
    mean_x = people.groupby("track_id")["x_m"].mean().to_dict()
    for track, role in people.groupby("track_id")["role"].agg(_majority).items():
        if role == "goalkeeper":
            teams[track] = "left" if mean_x[track] < 0 else "right"
    if embeddings is None or len(embeddings["frame"]) == 0:
        return teams
    vectors = embeddings[v.team_crops].astype(np.float32)
    emb_index = {(int(f), int(o)): i for i, (f, o) in enumerate(zip(embeddings["frame"], embeddings["obj_id"]))}
    players = people[(people["role"] == "player") & (people["sam_obj_id"] >= 0)]
    rows, tracks = [], []
    for frame, obj, track in zip(players["frame"], players["sam_obj_id"], players["track_id"]):
        i = emb_index.get((int(frame), int(obj)))
        if i is not None:
            rows.append(i)
            tracks.append(int(track))
    if len(set(tracks)) < 2:
        return teams
    clusters = cluster_teams(vectors[rows], np.array(tracks))
    teams.update(name_sides(clusters, {t: mean_x[t] for t in clusters}))
    return teams


def assemble(
    dets: pd.DataFrame,
    sam: pd.DataFrame,
    homographies: dict[int, np.ndarray],
    embeddings: dict | None,
    jerseys: dict[int, str | None] | None,
    n_frames: int,
    fps: float = 25.0,
    v: Variant = Variant(),  # noqa: B008  (frozen dataclass, safe as a default)
) -> pd.DataFrame:
    """One row per person per frame: track_id, role, team, jersey, x_m, y_m, score."""
    people = _to_metres(_people(dets, sam, n_frames, fps, v), homographies)
    if people.empty:
        return pd.DataFrame(columns=["frame", "track_id", "role", "team", "jersey", "x_m", "y_m", "score"])
    if v.smooth > 1:
        people = people.sort_values(["track_id", "frame"])
        for col in ("x_m", "y_m"):
            people[col] = people.groupby("track_id")[col].transform(
                lambda s: s.rolling(v.smooth, center=True, min_periods=1).median()
            )
    roles = people.groupby("track_id")["role"].agg(_majority).to_dict()
    teams = _teams(people.assign(role=people["track_id"].map(roles)), embeddings, v)
    out = people.assign(role=people["track_id"].map(roles))
    out = out[out["role"].notna()]
    out["team"] = out["track_id"].map(teams)
    out.loc[~out["role"].isin(["player", "goalkeeper"]), "team"] = None
    out["jersey"] = out["track_id"].map(jerseys or {}) if v.jersey else None
    out.loc[out["role"] != "player", "jersey"] = None
    score = out["score"] if "score" in out else out.get("prob", 1.0)
    out = out.assign(score=score)
    return out[["frame", "track_id", "role", "team", "jersey", "x_m", "y_m", "score"]].reset_index(drop=True)


def ball_positions(dets: pd.DataFrame, homographies: dict[int, np.ndarray], min_score: float = 0.3,
                   max_gap: int = 10) -> pd.DataFrame:  # fmt: skip
    """The ball on the pitch per frame: the most confident ball box, bottom-centre, in metres.
    Gaps up to `max_gap` frames are filled linearly. (A ball in the air projects too far
    from the camera; fine for possession, which only looks at balls near players.)"""
    balls = dets[(dets["class_name"] == "ball") & (dets["score"] >= min_score)]
    if balls.empty:
        return pd.DataFrame(columns=["frame", "x_m", "y_m"])
    best = balls.loc[balls.groupby("frame")["score"].idxmax()]
    rows = []
    for r in best.itertuples(index=False):
        H = nearest_homography(homographies, int(r.frame))
        if H is not None:
            x, y = to_pitch(H, np.array([[(r.x1 + r.x2) / 2, r.y2]]))[0]
            rows.append({"frame": int(r.frame), "x_m": x, "y_m": y})
    if not rows:
        return pd.DataFrame(columns=["frame", "x_m", "y_m"])
    ball = pd.DataFrame(rows).set_index("frame")
    frames = np.arange(ball.index.min(), ball.index.max() + 1)
    ball = ball.reindex(frames).interpolate(limit=max_gap, limit_area="inside")
    return ball.dropna().rename_axis("frame").reset_index()


def to_predictions(state: pd.DataFrame, image_ids: dict[int, str]) -> list[dict]:
    """GSR prediction records for the official evaluator."""
    return [
        prediction(image_ids[int(r.frame)], int(r.track_id), r.role, float(r.x_m), float(r.y_m),
                   r.team if isinstance(r.team, str) else None,
                   r.jersey if isinstance(r.jersey, str) else None, float(r.score))
        for r in state.itertuples(index=False) if int(r.frame) in image_ids
    ]  # fmt: skip
