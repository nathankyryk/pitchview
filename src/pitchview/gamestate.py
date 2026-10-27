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
from pitchview.track import PEOPLE, bytetrack

BOX = ["x1", "y1", "x2", "y2"]
STATE_COLUMNS = ["frame", "track_id", "role", "team", "jersey", "x_m", "y_m", "score", *BOX, "sam_obj_id"]


@dataclass(frozen=True)
class Variant:
    tracker: str = "bytetrack"  # "bytetrack" on RF-DETR boxes, or "sam" (SAM 3.1's own tracks)
    foot: str = "mask"  # where the player touches the grass: "mask" (lowest mask pixels) or "box" (bottom-centre)
    team_crops: str = "mask"  # jersey embeddings from "mask" or "box" crops
    jersey: bool = True
    smooth: int = 25  # rolling-median window over a track's positions, in frames; 1 = off (25 won on valid x5)
    # One id for track pieces with the same team and number that never overlap.
    # Valid x5: ByteTrack GS-HOTA 35.2 -> 36.7 (AssA 50.7 -> 54.9); SAM tracks unchanged.
    merge_by_jersey: bool = True
    # SAM tracks only: keep a person only in frames within this many frames of a match to one of
    # RF-DETR's player, goalkeeper or referee boxes; negative keeps every frame SAM tracked.
    # Valid x5 GS-HOTA: every frame 25.7; gap 0 24.1, 3 27.7, 10 27.7, 25 27.4.
    sam_box_gap: int = 3
    # Split a track where its jersey reads switch number for good (jersey.split_votes): the
    # tracker jumped to another player. Needs `pieces` in assemble(). Valid x5 GS-HOTA: SAM tracks
    # 36.3 -> 36.4, ByteTrack 36.5 -> 36.6 (8 and 4 tracks split): about even; it fixes the demo's names.
    split_by_jersey: bool = True


def _near(frames: np.ndarray, anchors: np.ndarray, gap: int) -> np.ndarray:
    """True where a frame is within `gap` frames of an anchor frame (anchors sorted)."""
    if len(anchors) == 0:
        return np.zeros(len(frames), dtype=bool)
    i = np.searchsorted(anchors, frames)
    left = anchors[np.clip(i - 1, 0, len(anchors) - 1)]
    right = anchors[np.clip(i, 0, len(anchors) - 1)]
    return np.minimum(np.abs(frames - left), np.abs(frames - right)) <= gap


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
    dets_by_frame = {f: d for f, d in dets[dets["class_name"].isin(PEOPLE)].groupby("frame")}
    for frame, rows in people.groupby("frame"):
        d = dets_by_frame.get(frame)
        if d is None:
            continue
        iou = box_iou(rows[BOX].to_numpy(float), d[BOX].to_numpy(float))
        best = iou.argmax(1)
        ok = iou.max(1) >= 0.5
        people.loc[rows.index[ok], "role"] = d["class_name"].to_numpy()[best[ok]]
    people["box_role"] = people["role"]  # this frame's match; "role" later becomes the track's majority
    if v.sam_box_gap >= 0:
        keep = pd.Series(False, index=people.index)
        for _, rows in people.groupby("track_id"):
            anchors = np.sort(rows.loc[rows["role"].notna(), "frame"].to_numpy())
            keep[rows.index] = _near(rows["frame"].to_numpy(), anchors, v.sam_box_gap)
        people = people[keep]
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
    if "box_role" in players:  # SAM tracks: only crops from frames where the mask is in a player box
        players = players[players["box_role"] == "player"]
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


def merge_by_jersey(state: pd.DataFrame) -> pd.DataFrame:
    """Give one track id to pieces of the same player: tracks with the same team and jersey number
    that never share a frame (a tracker often splits a player at an occlusion or a camera cut).
    Pieces are chained in time order; each joins the first chain that has ended before it starts."""
    keyed = state.dropna(subset=["team", "jersey"])
    spans = keyed.groupby("track_id").agg(
        team=("team", "first"), jersey=("jersey", "first"), start=("frame", "min"), end=("frame", "max")
    )
    new_id: dict[int, int] = {}
    for _, group in spans.groupby(["team", "jersey"]):
        chains: list[list[int]] = []  # [first track id, last frame so far]
        for track, row in group.sort_values("start").iterrows():
            chain = next((c for c in chains if row["start"] > c[1]), None)
            if chain is None:
                chains.append([track, row["end"]])
            else:
                new_id[track] = chain[0]
                chain[1] = row["end"]
    return state.assign(track_id=state["track_id"].map(lambda t: new_id.get(t, t)))


def split_by_jersey(state: pd.DataFrame, pieces: dict[int, list[tuple[int, str | None]]]) -> pd.DataFrame:
    """One track per piece of jersey.split_votes(): pieces after the first get new ids, and each piece
    takes its own number. Pieces of one player are then joined again by merge_by_jersey."""
    out = state.copy()
    next_id = int(out["track_id"].max()) + 1 if len(out) else 0
    for track, parts in pieces.items():
        if len(parts) < 2:
            continue
        rows = (out["track_id"] == track).to_numpy()
        frames = out["frame"].to_numpy()
        for i, (start, number) in enumerate(parts):
            end = parts[i + 1][0] if i + 1 < len(parts) else np.inf
            piece = rows & (frames >= start) & (frames < end)
            out.loc[piece, "jersey"] = number
            if i > 0:
                out.loc[piece, "track_id"] = next_id
                next_id += 1
    return out


def assemble(
    dets: pd.DataFrame,
    sam: pd.DataFrame,
    homographies: dict[int, np.ndarray],
    embeddings: dict | None,
    jerseys: dict[int, str | None] | None,
    n_frames: int,
    fps: float = 25.0,
    v: Variant = Variant(),  # noqa: B008  (frozen dataclass, safe as a default)
    pieces: dict[int, list[tuple[int, str | None]]] | None = None,
) -> pd.DataFrame:
    """One row per person per frame: track_id, role, team, jersey, x_m, y_m, score, and for the
    video renderer the image box (x1, y1, x2, y2) and SAM's mask id (sam_obj_id, -1 if none).
    pieces (jersey.split_votes) splits tracks whose number changes, with v.split_by_jersey."""
    people = _to_metres(_people(dets, sam, n_frames, fps, v), homographies)
    if people.empty:
        return pd.DataFrame(columns=STATE_COLUMNS)
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
    if v.jersey and v.split_by_jersey and pieces:
        out = split_by_jersey(out, pieces)
    out.loc[out["role"] != "player", "jersey"] = None
    score = out["score"] if "score" in out else out.get("prob", 1.0)
    out = out.assign(score=score, sam_obj_id=out["sam_obj_id"].fillna(-1).astype(int))
    if v.merge_by_jersey:
        out = merge_by_jersey(out)
    return out[STATE_COLUMNS].reset_index(drop=True)


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
