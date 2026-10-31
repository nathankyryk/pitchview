"""Assemble the game state of one clip: who is where on the pitch, with role, team and number.

Inputs are the stage outputs (RF-DETR detections, SAM 3.1 tracks, pitch keypoints, jersey
embeddings, jersey reads). `Variant` switches each design choice, so every row of the
results table is the same code with different switches.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

from pitchview.evaluate import prediction
from pitchview.jersey import MIN_CONF, NOT_PLAYER_NUMBERS, vote
from pitchview.pitch import LENGTH, WIDTH, nearest_homography, to_pitch
from pitchview.segment import attach_masks, box_iou
from pitchview.teams import cluster_teams, crop_clusters, majority_cluster, name_sides
from pitchview.track import PEOPLE, bytetrack

BOX = ["x1", "y1", "x2", "y2"]
# sam_parts: ids of further SAM 3.1 masks that are the same person in that frame (join_parts), space-separated
STATE_COLUMNS = ["frame", "track_id", "role", "team", "jersey", "x_m", "y_m", "score", *BOX, "sam_obj_id", "sam_parts"]


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
    # One player per number per team per frame (unique_numbers). Needs `evidence` in assemble().
    # Valid x5 GS-HOTA: SAM tracks 36.4 -> 36.7, ByteTrack 36.6 -> 36.5. 4-12% of numbered player
    # frames had a second player with the same number (two nametags with one name in the demo).
    unique_numbers: bool = True
    # unique_numbers: clash frames up to this many frames apart form one clash, with one winner
    # throughout (None: each frame on its own, which let names jump between two players and back).
    # Valid x5, SAM tracks: GS-HOTA 36.7 -> 36.9 with 1, 5 or 25; right-number player frames
    # 52.2% -> 52.8% (1, 5) / 52.7% (25); 1000 (one winner per clip) 52.1%. ByteTrack: 36.5 -> 36.4.
    episode_gap: int | None = 5
    # SAM tracks only: join a track that lets go of a player to the new track SAM 3.1 starts on them
    # a moment later (link_handoffs); the earlier piece takes the later one's number. Valid x5: GS-HOTA
    # 44.3 either way, IDF1 63.9 -> 64.0, ID switches 277 -> 275, right-number player frames 70.4% -> 70.5%.
    link_handoffs: bool = True
    # SAM tracks only: cut a track where SAM 3.1 jumps to another person (jump_cuts): its crops' kit
    # cluster changes for at least `jump_team_run` crops (5 frames apart), its detector box changes
    # between referee and player for at least `jump_role_run` matched frames, or its sure jersey reads
    # change number (split_by_jersey). Role, team and number are then decided per piece.
    # Valid x5 GS-HOTA (45.7 with none of these four fixes): this one alone 46.0; all four 47.1, and
    # 47.1 with team runs 3 / 6 or role runs 5 / 20 too. ID switches 284 -> 311 (cuts not rejoined).
    split_jumps: bool = True
    jump_team_run: int = 4
    jump_role_run: int = 10
    # SAM tracks only: a person SAM 3.1 holds as two or more masks (torso and legs) is one person
    # where the masks together fill a detector box that no single mask matches (_join_parts).
    # Valid x5: 45.7 alone (even); kept for the video, where such players had no mask at all.
    join_parts: bool = True
    # Drop people standing more than this many metres outside the pitch lines (staff, firefighters);
    # negative keeps everyone. Valid x5: 46.0 alone; with the other fixes 3 m 47.0, 5 m 47.1, 7 m 47.1, 10 m 47.0.
    off_pitch: float = 5.0
    # SAM tracks only: a player piece in a keeper's kit becomes a goalkeeper (keepers_by_kit). Needs
    # `keeper_kits` in assemble() (`pitchview keeper-kits`). Valid x5 has no case (its 4 keepers are all
    # boxed as keepers), so it was checked on test without tuning (README).
    keeper_by_kit: bool = True


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
    people["sam_parts"] = ""
    dets_by_frame = {f: d for f, d in dets[dets["class_name"].isin(PEOPLE)].groupby("frame")}
    unmatched = []  # per frame: masks and detector boxes left without a match, for _join_parts
    for frame, rows in people.groupby("frame"):
        d = dets_by_frame.get(frame)
        if d is None:
            continue
        iou = box_iou(rows[BOX].to_numpy(float), d[BOX].to_numpy(float))
        best = iou.argmax(1)
        ok = iou.max(1) >= 0.5
        people.loc[rows.index[ok], "role"] = d["class_name"].to_numpy()[best[ok]]
        lonely = iou.max(0) < 0.5
        if v.join_parts and lonely.any() and (~ok).sum() >= 2:
            unmatched.append((rows.index[~ok], d[lonely]))
    if unmatched:
        people = _join_parts(people, unmatched)
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


def _join_parts(people: pd.DataFrame, unmatched: list, inside: float = 0.8, min_iou: float = 0.5) -> pd.DataFrame:
    """SAM 3.1 sometimes holds one person as two masks, say the torso and the legs. Neither overlaps
    the person's detector box by IoU 0.5, so neither gets a role and sam_box_gap drops the person
    (test: 0.6% of detector people, 347 in SNGS-123 alone). In each frame, masks without a match that
    lie at least `inside` within a detector box that no single mask matches are one person when their
    union overlaps that box by IoU >= `min_iou`. The mask whose track matched boxes on its own most
    often keeps the row, with the union box, the lowest piece's foot point and the box's role; the
    other pieces' rows go, and their ids are kept in sam_parts for the video."""
    matched = people.groupby("track_id")["role"].count()  # frames each mask matched a box on its own
    drop: list = []
    for idx, boxes in unmatched:
        masks = people.loc[idx]
        mb = masks[BOX].to_numpy(float)
        area = (mb[:, 2] - mb[:, 0]) * (mb[:, 3] - mb[:, 1])
        free = np.ones(len(mb), dtype=bool)
        for det in boxes.itertuples(index=False):
            db = np.array([det.x1, det.y1, det.x2, det.y2], dtype=float)
            w = np.clip(np.minimum(mb[:, 2], db[2]) - np.maximum(mb[:, 0], db[0]), 0, None)
            h = np.clip(np.minimum(mb[:, 3], db[3]) - np.maximum(mb[:, 1], db[1]), 0, None)
            part = free & (area > 0) & (w * h >= inside * area)
            if part.sum() < 2:
                continue
            union = np.array([[mb[part, 0].min(), mb[part, 1].min(), mb[part, 2].max(), mb[part, 3].max()]])
            if box_iou(db[None], union)[0, 0] < min_iou:
                continue
            free &= ~part
            pieces = masks.index[part]
            keeper = max(pieces, key=lambda i: (matched.get(people.at[i, "track_id"], 0), -people.at[i, "track_id"]))
            lowest = people.loc[pieces, "foot_y"].idxmax()
            people.loc[keeper, BOX] = union[0]
            people.loc[keeper, ["foot_x", "foot_y"]] = people.loc[lowest, ["foot_x", "foot_y"]].to_numpy()
            people.loc[keeper, "role"] = det.class_name
            others = [i for i in pieces if i != keeper]
            people.loc[keeper, "sam_parts"] = " ".join(str(int(people.at[i, "sam_obj_id"])) for i in others)
            drop.extend(others)
    return people.drop(index=drop)


def _to_metres(people: pd.DataFrame, homographies: dict[int, np.ndarray]) -> pd.DataFrame:
    return _to_metres_all(people, homographies).dropna(subset=["x_m", "y_m"])


def _to_metres_all(people: pd.DataFrame, homographies: dict[int, np.ndarray]) -> pd.DataFrame:
    """Every row, with x_m and y_m NaN where no pitch mapping is near."""
    xs, ys = np.full(len(people), np.nan), np.full(len(people), np.nan)
    for frame, idx in people.groupby("frame").indices.items():
        H = nearest_homography(homographies, int(frame))
        if H is not None:
            pts = to_pitch(H, people.iloc[idx][["foot_x", "foot_y"]].to_numpy(float))
            xs[idx], ys[idx] = pts[:, 0], pts[:, 1]
    return people.assign(x_m=xs, y_m=ys)


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


def on_pitch(people: pd.DataFrame, margin: float) -> np.ndarray:
    """True for rows standing at most `margin` metres outside the pitch lines. In GSR's train labels
    4.1% of people stand outside (throw-ins, assistant referees): 99.9% within 3.6 m, all within 6.7 m."""
    return ((people["x_m"].abs() <= LENGTH / 2 + margin) & (people["y_m"].abs() <= WIDTH / 2 + margin)).to_numpy()


def jump_cuts(frames, labels, min_run: int) -> list[int]:
    """Where a track's per-frame labels change for good (its kit, or referee vs player): runs shorter
    than `min_run` are noise and take the label of their longer neighbour (the shortest run first).
    Each change left is a cut at the frame halfway between the old run's last frame and the new
    run's first; the new piece starts at the cut."""
    frames, labels = np.asarray(frames), list(labels)
    runs: list[list] = []  # [label, first index, last index]
    for i, label in enumerate(labels):
        if runs and runs[-1][0] == label:
            runs[-1][2] = i
        else:
            runs.append([label, i, i])
    while len(runs) > 1:
        size = [last - first + 1 for _, first, last in runs]
        k = min(range(len(runs)), key=size.__getitem__)
        if size[k] >= min_run:
            break
        left, right = k - 1, k + 1
        if right >= len(runs) or (left >= 0 and size[left] >= size[right]):
            runs[k][0] = runs[left][0]
        else:
            runs[k][0] = runs[right][0]
        merged: list[list] = []
        for run in runs:
            if merged and merged[-1][0] == run[0]:
                merged[-1][2] = run[2]
            else:
                merged.append(run)
        runs = merged
    return [int((frames[a[2]] + frames[b[1]]) // 2 + 1) for a, b in pairwise(runs)]


def split_tracks(state: pd.DataFrame, cuts: dict[int, list[int]]) -> pd.DataFrame:
    """Cut tracks at the given frames: each piece after a track's first gets a new id."""
    track, frames = state["track_id"].to_numpy(), state["frame"].to_numpy()
    new = track.copy()
    next_id = int(track.max()) + 1 if len(track) else 0
    for t, at in cuts.items():
        rows = np.flatnonzero(track == t)
        if not len(rows) or not at:
            continue
        piece = np.searchsorted(np.sort(at), frames[rows], side="right")
        for k in np.unique(piece)[1:]:
            new[rows[piece == k]] = next_id
            next_id += 1
    return state.assign(track_id=new)


def _crop_labels(people: pd.DataFrame, embeddings: dict | None, v: Variant) -> pd.DataFrame | None:
    """SAM people's jersey crops from frames inside a player box, each with its kit cluster (0/1, one
    clustering per clip): frame, sam_obj_id, emb (row in the embeddings), cluster."""
    if embeddings is None or len(embeddings["frame"]) == 0:
        return None
    emb_index = {(int(f), int(o)): i for i, (f, o) in enumerate(zip(embeddings["frame"], embeddings["obj_id"]))}
    players = people[(people["box_role"] == "player") & (people["sam_obj_id"] >= 0)]
    rows = [
        (int(f), int(o), i)
        for f, o in zip(players["frame"], players["sam_obj_id"])
        if (i := emb_index.get((int(f), int(o)))) is not None
    ]
    if len({o for _, o, _ in rows}) < 2:
        return None
    crops = pd.DataFrame(rows, columns=["frame", "sam_obj_id", "emb"])
    vectors = embeddings[v.team_crops].astype(np.float32)[crops["emb"].to_numpy()]
    return crops.assign(cluster=crop_clusters(vectors))


def _jump_cuts_by_track(people: pd.DataFrame, labels: pd.DataFrame | None, pieces: dict | None, v: Variant,
                        min_gap: int = 10) -> dict[int, list[int]]:  # fmt: skip
    """Cut frames per SAM track: sure jersey reads switching number (pieces from jersey.split_votes),
    the kit cluster switching, and the detector box switching between referee and player. Cuts closer
    than `min_gap` frames to an earlier one are the same jump."""
    found: dict[int, list[int]] = {}
    for t, parts in (pieces or {}).items():
        found.setdefault(int(t), []).extend(start for start, _ in parts[1:])
    if labels is not None:
        for t, g in labels.sort_values("frame").groupby("sam_obj_id"):
            found.setdefault(int(t), []).extend(jump_cuts(g["frame"], g["cluster"], v.jump_team_run))
    matched = people[people["box_role"].notna()].sort_values("frame")
    for t, g in matched.groupby("track_id"):
        kind = np.where(g["box_role"] == "referee", "referee", "player")
        found.setdefault(int(t), []).extend(jump_cuts(g["frame"], kind, v.jump_role_run))
    cuts = {}
    for t, at in found.items():
        kept: list[int] = []
        for c in sorted(at):
            if not kept or c - kept[-1] >= min_gap:
                kept.append(c)
        if kept:
            cuts[t] = kept
    return cuts


def _piece_numbers(people: pd.DataFrame, cuts: dict[int, list[int]], jerseys: dict | None,
                   pieces: dict | None, reads: pd.DataFrame | None) -> dict[int, str | None]:  # fmt: skip
    """Each piece's jersey number. A track that wasn't cut keeps its vote; a cut track's pieces each
    vote on their own reads, with the rule `pitchview jerseys` uses (or, without the reads, take
    the jersey piece they start in)."""
    out: dict[int, str | None] = {}
    starts = people.groupby("track_id").agg(orig=("read_track", "first"), start=("frame", "min"))
    for piece, (orig, start) in starts.iterrows():
        at = cuts.get(int(orig))
        if not at:
            out[int(piece)] = (jerseys or {}).get(int(orig))
            continue
        bounds = [-np.inf, *at, np.inf]
        k = int(np.searchsorted(at, start, side="right"))
        lo, hi = bounds[k], bounds[k + 1]
        if reads is not None:
            mine = reads[(reads["track_id"] == orig) & (reads["frame"] >= lo) & (reads["frame"] < hi)]
            own = [(0, n if isinstance(n, str) else None, c) for n, c in zip(mine["number"], mine["conf"])]
            out[int(piece)] = vote(own, min_conf=MIN_CONF, exclude=NOT_PLAYER_NUMBERS).get(0)
        else:
            parts = (pieces or {}).get(int(orig)) or [(0, (jerseys or {}).get(int(orig)))]
            out[int(piece)] = next((n for s, n in reversed(parts) if s <= start), parts[0][1])
    return out


def _teams_from_labels(people: pd.DataFrame, labels: pd.DataFrame, roles: dict) -> dict[int, str]:
    """_teams for cut SAM tracks: each player piece's majority kit cluster among its own crops,
    named by side; goalkeepers by the half they stand in."""
    mean_x = people.groupby("track_id")["x_m"].mean().to_dict()
    teams = {t: ("left" if mean_x[t] < 0 else "right") for t, r in roles.items() if r == "goalkeeper"}
    crops = labels.merge(people[["frame", "sam_obj_id", "track_id"]], on=["frame", "sam_obj_id"])
    crops = crops[crops["track_id"].map(roles) == "player"]
    if crops["track_id"].nunique() < 2:
        return teams
    clusters = majority_cluster(crops["cluster"].to_numpy(), crops["track_id"].to_numpy())
    teams.update(name_sides(clusters, {t: mean_x[t] for t in clusters}))
    return teams


def clip_half(info: dict) -> int | None:
    """The half a GSR clip comes from, from its info ("2 - 23:00" -> 2); None if missing."""
    try:
        return int(str(info["game_time_start"]).split(" - ")[0])
    except (KeyError, ValueError):
        return None


def side_evidence(state: pd.DataFrame, embeddings: dict | None, crops: str = "mask") -> dict | None:
    """Each named team's kit and place in one clip, for pooled_sides: per side ("left", "right"), the
    mean unit embedding of its players' crops and their mean x in metres. None unless both sides have crops."""
    if embeddings is None or len(embeddings["frame"]) == 0 or state.empty:
        return None
    index = {(int(f), int(o)): i for i, (f, o) in enumerate(zip(embeddings["frame"], embeddings["obj_id"]))}
    vectors = embeddings[crops].astype(np.float32)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    players = state[(state["role"] == "player") & state["team"].isin(["left", "right"]) & (state["sam_obj_id"] >= 0)]
    found = {}
    for side, g in players.groupby("team"):
        rows = [i for f, o in zip(g["frame"], g["sam_obj_id"]) if (i := index.get((int(f), int(o)))) is not None]
        if rows:
            kit = vectors[rows].mean(axis=0)
            found[side] = (kit / max(float(np.linalg.norm(kit)), 1e-9), float(g["x_m"].mean()))
    return found if len(found) == 2 else None


def pooled_sides(evidence: dict[str, dict | None], groups: dict[str, object]) -> set[str]:
    """Teams change ends only at half-time, so which kit defends the left goal is one fact for every
    clip of a match half. A clip's own naming (name_sides: the team standing further left) is a coin
    flip when play stays in one half. Pooled instead: each clip's two kits are lined up with those of
    the clip whose teams stand furthest apart (embedding similarity), every clip's evidence (the right
    team's mean x minus the left team's, in metres) is summed, and the sum names the kits.
    groups: clip -> its match half, e.g. (game id, half). Returns the clips whose own naming
    disagrees: their players' sides are to be swapped (swap_sides)."""
    by_group: dict[object, list[str]] = {}
    for clip, ev in evidence.items():
        if ev is not None and groups.get(clip) is not None:
            by_group.setdefault(groups[clip], []).append(clip)
    flips: set[str] = set()
    for clips in by_group.values():
        if len(clips) < 2:
            continue
        gap = {c: evidence[c]["right"][1] - evidence[c]["left"][1] for c in clips}
        ref = max(clips, key=lambda c: abs(gap[c]))
        a, b = evidence[ref]["left"][0], evidence[ref]["right"][0]
        same = {}  # the clip's "left" team wears kit a
        for c in clips:
            left, right = evidence[c]["left"][0], evidence[c]["right"][0]
            same[c] = float(left @ a + right @ b) >= float(left @ b + right @ a)
        total = sum(gap[c] if same[c] else -gap[c] for c in clips)  # > 0: kit a defends the left goal
        if total != 0:
            flips.update(c for c in clips if same[c] != (total > 0))
    return flips


def swap_sides(state: pd.DataFrame) -> pd.DataFrame:
    """Swap the players' sides; goalkeepers keep theirs (named by the half they stand in)."""
    team = state["team"].copy()
    players = (state["role"] == "player") & team.isin(["left", "right"])
    team.loc[players] = team[players].map({"left": "right", "right": "left"})
    return state.assign(team=team)


def keepers_by_kit(people: pd.DataFrame, labels: pd.DataFrame, embeddings: dict, roles: dict, teams: dict,
                   keeper_kits: np.ndarray, v: Variant, min_crops: int = 20,
                   min_share: float = 0.8) -> tuple[dict, dict]:  # fmt: skip
    """The detector sometimes calls a keeper a player for a whole track (SNGS-145: St. Gallen's keeper
    in 111 of 113 frames), so he gets a team but no keeper name. A player piece becomes the goalkeeper
    of the half he stands in when at least `min_crops` of its crops, and `min_share` of them, are nearer
    one of the match's keeper kits (`pitchview keeper-kits`) than to the clip's two team kits and its
    referees, and no goalkeeper of that half is in the picture at the same time.
    keeper_kits: (2, D) centres in embedding space. Returns updated (roles, teams)."""
    vectors = embeddings[v.team_crops].astype(np.float32)

    def unit(a):
        return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)

    crops = labels.merge(people[["frame", "sam_obj_id", "track_id"]], on=["frame", "sam_obj_id"])
    crops = crops[crops["track_id"].map(roles) == "player"]
    if crops.empty:
        return roles, teams
    own = unit(vectors[crops["emb"].to_numpy()])
    centres = [unit(own[crops["cluster"].to_numpy() == k].mean(0)) for k in (0, 1) if (crops["cluster"] == k).any()]
    emb_index = {(int(f), int(o)): i for i, (f, o) in enumerate(zip(embeddings["frame"], embeddings["obj_id"]))}
    refs = people[people["box_role"] == "referee"]
    ref_rows = [
        i for f, o in zip(refs["frame"], refs["sam_obj_id"]) if (i := emb_index.get((int(f), int(o)))) is not None
    ]
    if ref_rows:
        centres.append(unit(unit(vectors[ref_rows]).mean(0)))
    keepers = unit(np.asarray(keeper_kits, dtype=np.float32))
    nearest_keeper = (own @ keepers.T).max(1) > (own @ np.stack(centres).T).max(1)
    share = pd.Series(nearest_keeper, index=crops.index).groupby(crops["track_id"]).agg(["size", "mean"])
    span = people.groupby("track_id")["frame"].agg(["min", "max"])
    mean_x = people.groupby("track_id")["x_m"].mean()
    roles, teams = dict(roles), dict(teams)
    for t, (n, s) in share.iterrows():
        if n < min_crops or s < min_share:
            continue
        side = "left" if mean_x[t] < 0 else "right"
        rivals = [g for g, r in roles.items() if r == "goalkeeper" and teams.get(g) == side and g != t]
        if any(span.at[g, "min"] <= span.at[t, "max"] and span.at[t, "min"] <= span.at[g, "max"] for g in rivals):
            continue
        roles[t], teams[t] = "goalkeeper", side
    return roles, teams


def keeper_crops(dets: pd.DataFrame, sam: pd.DataFrame, embeddings: dict, n_frames: int,
                 fps: float = 25.0) -> dict[str, np.ndarray]:  # fmt: skip
    """The jersey embeddings ("mask" and "box" crops) of SAM 3.1 people inside the detector's goalkeeper
    boxes, for `pitchview keeper-kits` (pooled over a match, since kits don't change within one)."""
    people = _people(dets, sam, n_frames, fps, Variant(tracker="sam", sam_box_gap=-1))
    keepers = people[people["box_role"] == "goalkeeper"]
    emb_index = {(int(f), int(o)): i for i, (f, o) in enumerate(zip(embeddings["frame"], embeddings["obj_id"]))}
    rows = [
        i for f, o in zip(keepers["frame"], keepers["sam_obj_id"]) if (i := emb_index.get((int(f), int(o)))) is not None
    ]
    return {crops: np.asarray(embeddings[crops], dtype=np.float32)[rows] for crops in ("mask", "box")}


def merge_by_jersey(state: pd.DataFrame) -> pd.DataFrame:
    """Give one track id to pieces of the same player: tracks with the same team and jersey number
    that never share a frame (a tracker often splits a player at an occlusion or a camera cut).
    Pieces are chained in time order; each joins the first chain that has ended before it starts."""
    keyed = state.dropna(subset=["team", "jersey"])
    # a track's span counts all its frames, also those where it shows no number (unique_numbers)
    spans = keyed.groupby("track_id").agg(team=("team", "first"), jersey=("jersey", "first"))
    spans = spans.join(state.groupby("track_id")["frame"].agg(start="min", end="max"))
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


def link_handoffs(state: pd.DataFrame, max_gap: int = 10, min_iou: float = 0.5, max_overlap: int = 5) -> pd.DataFrame:
    """Join SAM 3.1 hand-offs: in a tackle or an occlusion SAM can let go of a player and start a new
    track on them a moment later, while the old track drifts off (and sam_box_gap drops it). Player
    track A hands off to B when both are on one team, B starts after A, A ends no later than
    `max_overlap` frames into B, B starts at most `max_gap` frames after A's last frame before it, and
    that box of A overlaps B's first box by IoU >= `min_iou`. Each track has at most one successor and
    one predecessor (largest overlap first), and a link that would join two numbers is skipped. A chain
    takes its earliest track's id and its number; A's frames from B's start on (the drift) are dropped.
    Chosen on valid x5 (gap 10, IoU 0.5: 2 links, both one labelled player; IoU 0.3 also joined two
    players). Test, held out: 13 links, all one player; 6 pass a number on."""
    players = state[(state["role"] == "player") & state["team"].notna()].sort_values("frame")
    if players.empty:
        return state
    frames, boxes, team, number = {}, {}, {}, {}
    for track, rows in players.groupby("track_id"):
        frames[track] = rows["frame"].to_numpy()
        boxes[track] = rows[BOX].to_numpy(float)
        team[track] = rows["team"].iloc[0]
        number[track] = next((j for j in rows["jersey"] if isinstance(j, str) and j), None)
    start = {t: f[0] for t, f in frames.items()}
    candidates = []
    for b in frames:
        for a, seen in frames.items():
            if a == b or team[a] != team[b] or start[a] >= start[b] or seen[-1] > start[b] + max_overlap:
                continue
            i = np.searchsorted(seen, start[b]) - 1  # A's last frame before B starts
            if i < 0 or start[b] - seen[i] > max_gap:
                continue
            iou = box_iou(boxes[a][i : i + 1], boxes[b][:1])[0, 0]
            if iou >= min_iou:
                candidates.append((iou, a, b))
    root = {t: t for t in frames}

    def find(t):
        while root[t] != t:
            t = root[t]
        return t

    succ, pred, chain_number = set(), set(), dict(number)
    for _, a, b in sorted(candidates, key=lambda c: -c[0]):
        ra, rb = find(a), find(b)
        if a in succ or b in pred or ra == rb:
            continue
        na, nb = chain_number[ra], chain_number[rb]
        if na is not None and nb is not None and na != nb:
            continue
        succ.add(a)
        pred.add(b)
        root[rb] = ra
        chain_number[ra] = na if na is not None else nb
        drift = ((state["track_id"] == a) & (state["frame"] >= start[b])).to_numpy()
        state = state[~drift]
    chains: dict[int, list] = {}
    for t in frames:
        chains.setdefault(find(t), []).append(t)
    out = state.copy()
    for r, members in chains.items():
        if len(members) < 2:
            continue
        first = min(members, key=lambda t: (start[t], t))
        rows = out["track_id"].isin(members)
        if chain_number[r] is not None:
            out.loc[rows & out["track_id"].map(lambda t: number.get(t) is None), "jersey"] = chain_number[r]
        out.loc[rows, "track_id"] = first
    return out


def unique_numbers(state: pd.DataFrame, evidence: dict, episode_gap: int | None = None) -> pd.DataFrame:
    """A number belongs to one player per team at a time. Where tracks of one team claim the same
    number in a frame, the track with a sure read of it nearest in time keeps it (more reads break a
    tie) and the others get no number there. evidence: jersey.read_frames(), keyed by the reader's
    track ids, which the state carries in `read_track` (split_by_jersey gives pieces new ids).
    episode_gap g: clash frames of one number at most g frames apart form one clash, and the track
    that wins most of its frames keeps the number through all of them (where it is there), so the
    name doesn't jump between two players and back. None decides each frame on its own."""
    keyed = state.dropna(subset=["team", "jersey"])
    claims = keyed.groupby(["frame", "team", "jersey"])["track_id"].transform("nunique")
    clash = keyed[claims > 1]
    if clash.empty:
        return state
    dist = np.full(len(clash), np.inf)
    reads = np.zeros(len(clash))
    keys = list(zip(clash["read_track"].astype(int), clash["jersey"].astype(str)))
    frames = clash["frame"].to_numpy()
    for i, (key, frame) in enumerate(zip(keys, frames)):
        seen = evidence.get(key)
        if seen is not None and len(seen):
            j = np.searchsorted(seen, frame)
            near = seen[max(j - 1, 0) : j + 1]
            dist[i], reads[i] = np.abs(near - frame).min(), len(seen)
    ranked = clash.assign(_dist=dist, _reads=-reads).sort_values(["_dist", "_reads", "track_id"])
    if episode_gap is not None:
        won = ranked.groupby(["frame", "team", "jersey"]).head(1).sort_values(["team", "jersey", "frame"])
        new = (won["team"] != won["team"].shift()) | (won["jersey"] != won["jersey"].shift())
        episode = (new | (won["frame"].diff() > episode_gap)).cumsum()
        winner = won.groupby(episode)["track_id"].agg(lambda t: t.value_counts().index[0])
        key = pd.MultiIndex.from_frame(won[["frame", "team", "jersey"]])
        episode_of = pd.Series(episode.to_numpy(), index=key)
        rows = pd.MultiIndex.from_frame(ranked[["frame", "team", "jersey"]])
        favourite = winner.reindex(episode_of.reindex(rows).to_numpy()).to_numpy()
        ranked = ranked.assign(_lost=ranked["track_id"].to_numpy() != favourite)
        ranked = ranked.sort_values(["_lost", "_dist", "_reads", "track_id"], kind="stable")
    keep = ranked.groupby(["frame", "team", "jersey"]).head(1).index
    out = state.copy()
    out.loc[clash.index.difference(keep), "jersey"] = None
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
    evidence: dict | None = None,
    reads: pd.DataFrame | None = None,
    keeper_kits: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row per person per frame: track_id, role, team, jersey, x_m, y_m, score, and for the
    video renderer the image box (x1, y1, x2, y2), SAM's mask id (sam_obj_id, -1 if none) and any
    further masks of the same person (sam_parts).
    pieces (jersey.split_votes) splits tracks whose number changes, with v.split_by_jersey;
    evidence (jersey.read_frames) settles shared numbers, with v.unique_numbers; reads (the saved
    jersey reads) give each piece of a cut SAM track its own number, with v.split_jumps;
    keeper_kits (`pitchview keeper-kits`, this match) finds keepers called players, with v.keeper_by_kit."""
    people = _to_metres(_people(dets, sam, n_frames, fps, v), homographies)
    if people.empty:
        return pd.DataFrame(columns=STATE_COLUMNS)
    if v.smooth > 1:
        people = people.sort_values(["track_id", "frame"])
        for col in ("x_m", "y_m"):
            people[col] = people.groupby("track_id")[col].transform(
                lambda s: s.rolling(v.smooth, center=True, min_periods=1).median()
            )
    if v.off_pitch >= 0:
        people = people[on_pitch(people, v.off_pitch)]
        if people.empty:
            return pd.DataFrame(columns=STATE_COLUMNS)
    if "sam_parts" not in people:
        people = people.assign(sam_parts="")
    people = people.assign(read_track=people["track_id"])
    numbers, labels = None, None
    if v.tracker == "sam" and v.split_jumps:
        labels = _crop_labels(people, embeddings, v)
        cuts = _jump_cuts_by_track(people, labels, pieces if v.jersey and v.split_by_jersey else None, v)
        people = split_tracks(people, cuts)
        numbers = _piece_numbers(people, cuts, jerseys, pieces, reads)
    roles = people.groupby("track_id")["role"].agg(_majority).to_dict()
    if labels is not None:
        teams = _teams_from_labels(people, labels, roles)
        if v.keeper_by_kit and keeper_kits is not None:
            roles, teams = keepers_by_kit(people, labels, embeddings, roles, teams, keeper_kits, v)
    else:
        teams = _teams(people.assign(role=people["track_id"].map(roles)), embeddings, v)
    out = people.assign(role=people["track_id"].map(roles))
    out = out[out["role"].notna()]
    out["team"] = out["track_id"].map(teams)
    out.loc[~out["role"].isin(["player", "goalkeeper"]), "team"] = None
    if not v.jersey:
        out["jersey"] = None
    elif numbers is not None:
        out["jersey"] = out["track_id"].map(numbers)
    else:
        out["jersey"] = out["track_id"].map(jerseys or {})
        if v.split_by_jersey and pieces:
            out = split_by_jersey(out, pieces)
    out.loc[out["role"] != "player", "jersey"] = None
    if v.tracker == "sam" and v.link_handoffs:
        out = link_handoffs(out)
    if v.jersey and v.unique_numbers and evidence is not None:
        out = unique_numbers(out, evidence, v.episode_gap)
    score = out["score"] if "score" in out else out.get("prob", 1.0)
    out = out.assign(score=score, sam_obj_id=out["sam_obj_id"].fillna(-1).astype(int))
    if v.merge_by_jersey:
        out = merge_by_jersey(out)
    return out[STATE_COLUMNS].reset_index(drop=True)


def unplaced(
    dets: pd.DataFrame,
    sam: pd.DataFrame,
    homographies: dict[int, np.ndarray],
    state: pd.DataFrame,
    n_frames: int,
    fps: float = 25.0,
    v: Variant = Variant(),  # noqa: B008  (frozen dataclass, safe as a default)
) -> pd.DataFrame:
    """SAM 3.1 people the game state leaves out only because their frame has no pitch mapping (a
    close-up, a fast pan), for the video. Each takes the track id, role, team and jersey of the same
    mask's nearest frame in the state; x_m and y_m stay NaN, so nothing scored or measured changes.
    Masks the state never has stay out."""
    people = _to_metres_all(_people(dets, sam, n_frames, fps, v), homographies)
    people = people[people["x_m"].isna() & people["sam_obj_id"].isin(state["sam_obj_id"])]
    if people.empty or state.empty:
        return pd.DataFrame(columns=STATE_COLUMNS)
    keys = {"frame": "int64", "sam_obj_id": "int64"}
    known = state[["frame", "sam_obj_id", "track_id", "role", "team", "jersey"]].astype(keys).sort_values("frame")
    if "sam_parts" not in people:
        people = people.assign(sam_parts="")
    rows = people[["frame", "sam_obj_id", "x_m", "y_m", "score" if "score" in people else "prob", *BOX, "sam_parts"]]
    rows = rows.rename(columns={"prob": "score"}).astype(keys).sort_values("frame")
    out = pd.merge_asof(rows, known.rename(columns={"frame": "near"}), left_on="frame", right_on="near",
                        by="sam_obj_id", direction="nearest")  # fmt: skip
    return out[STATE_COLUMNS].sort_values(["frame", "track_id"]).reset_index(drop=True)


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
