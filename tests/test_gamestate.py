import numpy as np
import pandas as pd

from pitchview.gamestate import Variant, assemble, ball_positions, to_predictions

# Pixels -> metres: x_m = x / 10 - 50, y_m = y / 10 - 30
H = np.array([[0.1, 0, -50], [0, 0.1, -30], [0, 0, 1]], dtype=float)
FRAMES = 12
NO_SAM = pd.DataFrame(columns=["frame", "obj_id", "prob", "x1", "y1", "x2", "y2", "foot_x", "foot_y", "area"])


def _all_frames(n=FRAMES):
    return {f: H for f in range(1, n + 1)}


def _box(frame, x1, y1, w, h, class_name, score=0.9):
    return {"frame": frame, "x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h, "score": score, "class_name": class_name}


def _dets(n=FRAMES):
    rows = []
    for f in range(1, n + 1):
        rows.append(_box(f, 100 + 5 * f, 100, 40, 100, "player"))
        rows.append(_box(f, 600, 300, 40, 100, "referee"))
        if f not in (5, 6):  # the ball goes missing for two frames
            rows.append(_box(f, 300 + f, 395, 10, 10, "ball", score=0.8))
    return pd.DataFrame(rows)


def _sam_row(frame, obj_id, box, foot_x=None):
    x1, y1, x2, y2 = box
    foot_x = (x1 + x2) / 2 if foot_x is None else foot_x
    return {"frame": frame, "obj_id": obj_id, "prob": 0.95, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "foot_x": foot_x, "foot_y": y2, "area": int((x2 - x1) * (y2 - y1))}  # fmt: skip


def test_assemble_maps_people_to_metres_with_roles():
    state = assemble(_dets(), NO_SAM, _all_frames(), None, {}, FRAMES, v=Variant(smooth=1))
    assert set(state["role"]) == {"player", "referee"}
    referee = state[state["role"] == "referee"]
    assert np.allclose(referee[["x_m", "y_m"]].to_numpy(), [12.0, 10.0])  # (620, 400) px, box bottom-centre
    assert referee["team"].isna().all() and referee["jersey"].isna().all()
    player = state[state["role"] == "player"].sort_values("frame")
    assert np.allclose(np.diff(player["x_m"]), 0.5)  # 5 px per frame -> 0.5 m per frame


def test_assemble_without_homographies_is_empty():
    assert assemble(_dets(), NO_SAM, {}, None, None, FRAMES).empty


def test_mask_foot_point_used_when_a_mask_matches():
    dets = pd.DataFrame([_box(f, 200, 100, 40, 100, "player") for f in range(1, FRAMES + 1)])
    # The mask says the planted foot is 10 px right of the box centre (220 -> 230 px).
    sam = pd.DataFrame([_sam_row(f, 7, (201, 101, 239, 200), foot_x=230.0) for f in range(1, FRAMES + 1)])
    with_mask = assemble(dets, sam, _all_frames(), None, None, FRAMES, v=Variant(foot="mask", smooth=1))
    with_box = assemble(dets, sam, _all_frames(), None, None, FRAMES, v=Variant(foot="box", smooth=1))
    assert np.allclose(with_mask["x_m"], 230 / 10 - 50)
    assert np.allclose(with_box["x_m"], 220 / 10 - 50)


def test_sam_tracker_takes_ids_from_sam_and_roles_from_detections():
    dets = _dets()
    sam = pd.DataFrame(
        [_sam_row(f, 3, (100 + 5 * f, 100, 140 + 5 * f, 200)) for f in range(1, FRAMES + 1)]
        + [_sam_row(f, 8, (600, 300, 640, 400)) for f in range(1, FRAMES + 1)]
        + [_sam_row(f, 9, (900, 300, 930, 380)) for f in range(1, FRAMES + 1)]  # nobody RF-DETR saw: dropped
    )
    state = assemble(dets, sam, _all_frames(), None, None, FRAMES, v=Variant(tracker="sam", smooth=1))
    assert state.groupby("track_id")["role"].first().to_dict() == {3: "player", 8: "referee"}
    assert len(state) == 2 * FRAMES  # every frame, no ByteTrack warm-up


def test_teams_from_jersey_embeddings_and_goalkeeper_half():
    rng = np.random.default_rng(0)
    red_x, blue_x = (250, 300, 350), (650, 700, 750)  # -25..-15 m vs +15..+25 m
    people = [(x, 100 + 150 * i, "player", "red") for i, x in enumerate(red_x)]
    people += [(x, 100 + 150 * i, "player", "blue") for i, x in enumerate(blue_x)]
    people.append((50, 300, "goalkeeper", "keeper"))  # stands in the left half (x = -45 m)
    dets, sam, emb = [], [], {"frame": [], "obj_id": [], "box": [], "mask": []}
    kit = {"red": np.eye(16)[0], "blue": np.eye(16)[1], "keeper": np.eye(16)[2]}
    for f in range(1, FRAMES + 1):
        for i, (x, y, role, colour) in enumerate(people):
            dets.append(_box(f, x, y, 30, 60, role))
            sam.append(_sam_row(f, 100 + i, (x, y, x + 30, y + 60)))
            vec = kit[colour] + rng.normal(0, 0.03, 16)
            for key, value in (("frame", f), ("obj_id", 100 + i), ("box", vec), ("mask", vec)):
                emb[key].append(value)
    emb = {k: np.array(v) for k, v in emb.items()}
    state = assemble(pd.DataFrame(dets), pd.DataFrame(sam), _all_frames(), emb, None, FRAMES, v=Variant(smooth=1))
    team_by_x = state.groupby("track_id").agg(x=("x_m", "mean"), team=("team", "first"), role=("role", "first"))
    players = team_by_x[team_by_x["role"] == "player"]
    assert set(players[players["x"] < 0]["team"]) == {"left"}
    assert set(players[players["x"] > 0]["team"]) == {"right"}
    assert set(team_by_x[team_by_x["role"] == "goalkeeper"]["team"]) == {"left"}


def test_jersey_numbers_go_to_players_only():
    first = assemble(_dets(), NO_SAM, _all_frames(), None, None, FRAMES, v=Variant(smooth=1))
    numbers = {int(t): "10" for t in first["track_id"].unique()}  # a number for every track, referee included
    state = assemble(_dets(), NO_SAM, _all_frames(), None, numbers, FRAMES, v=Variant(smooth=1))
    assert set(state.loc[state["role"] == "player", "jersey"]) == {"10"}
    assert state.loc[state["role"] == "referee", "jersey"].isna().all()
    off = assemble(_dets(), NO_SAM, _all_frames(), None, numbers, FRAMES, v=Variant(jersey=False, smooth=1))
    assert off["jersey"].isna().all()


def test_frames_borrow_the_nearest_homography_within_twelve_frames():
    n = 20
    state = assemble(_dets(n), NO_SAM, {1: H}, None, None, n, v=Variant(smooth=1))
    referee_frames = set(state.loc[state["role"] == "referee", "frame"])
    assert max(referee_frames) == 13  # frame 13 is 12 frames from frame 1; 14+ have no calibration


def test_ball_positions_fill_short_gaps():
    ball = ball_positions(_dets(), _all_frames()).set_index("frame")
    assert list(ball.index) == list(range(1, FRAMES + 1))  # frames 5-6 interpolated
    assert np.isclose(ball.loc[5, "x_m"], (305 + 5) / 10 - 50)
    assert np.isclose(ball.loc[1, "y_m"], 405 / 10 - 30)


def test_to_predictions_uses_gsr_image_ids():
    state = pd.DataFrame([{"frame": 3, "track_id": 1, "role": "player", "team": "left", "jersey": "9",
                           "x_m": 1.0, "y_m": 2.0, "score": 0.8}])  # fmt: skip
    (pred,) = to_predictions(state, {3: "2021000003"})
    assert pred["image_id"] == "2021000003"
    assert pred["attributes"] == {"role": "player", "team": "left", "jersey": "9"}
    assert pred["bbox_pitch"]["x_bottom_middle"] == 1.0
