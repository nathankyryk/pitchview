import numpy as np
import pandas as pd

from pitchview.gamestate import (
    Variant,
    assemble,
    ball_positions,
    merge_by_jersey,
    split_by_jersey,
    to_predictions,
    unique_numbers,
)

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
    assert (state["sam_obj_id"] == -1).all()  # no masks in this test
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
    # the renderer's link back to the image: the tracked box and the mask that matched it
    assert (with_mask["sam_obj_id"] == 7).all() and (with_mask["x1"] == 200).all()


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
    assert (state["sam_obj_id"] == state["track_id"]).all()


def test_sam_box_gap_keeps_sam_people_only_near_detector_boxes():
    dets = pd.DataFrame([_box(f, 200, 100, 40, 100, "player") for f in range(1, 5)])  # boxes in frames 1-4 only
    sam = pd.DataFrame([_sam_row(f, 3, (200, 100, 240, 200)) for f in range(1, FRAMES + 1)])
    kept = {}
    for gap in (-1, 0, 2):
        state = assemble(
            dets, sam, _all_frames(), None, None, FRAMES, v=Variant(tracker="sam", smooth=1, sam_box_gap=gap)
        )
        kept[gap] = sorted(state["frame"])
    assert kept[-1] == list(range(1, FRAMES + 1))  # off: one matched frame keeps the whole track
    assert kept[0] == [1, 2, 3, 4]
    assert kept[2] == [1, 2, 3, 4, 5, 6]


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


def test_sam_teams_use_only_crops_inside_player_boxes():
    # SAM keeps tracking everyone after the detector's boxes stop (frames 5-12), but those crops show
    # something else (say a spectator in front). They must not decide the teams.
    rng = np.random.default_rng(0)
    people = [(250, 100, "red"), (300, 250, "red"), (650, 100, "blue"), (700, 250, "blue")]
    kit = {"red": np.eye(16)[0], "blue": np.eye(16)[1], "other": np.eye(16)[2]}
    dets, sam, emb = [], [], {"frame": [], "obj_id": [], "box": [], "mask": []}
    for f in range(1, FRAMES + 1):
        for i, (x, y, colour) in enumerate(people):
            if f <= 4:
                dets.append(_box(f, x, y, 30, 60, "player"))
            sam.append(_sam_row(f, 100 + i, (x, y, x + 30, y + 60)))
            vec = kit[colour if f <= 4 else "other"] + rng.normal(0, 0.03, 16)
            for key, value in (("frame", f), ("obj_id", 100 + i), ("box", vec), ("mask", vec)):
                emb[key].append(value)
    emb = {k: np.array(v) for k, v in emb.items()}
    v = Variant(tracker="sam", smooth=1, merge_by_jersey=False)
    state = assemble(pd.DataFrame(dets), pd.DataFrame(sam), _all_frames(), emb, None, FRAMES, v=v)
    teams = state.groupby("track_id")["team"].first().to_dict()
    assert teams == {100: "left", 101: "left", 102: "right", 103: "right"}


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


def test_merge_by_jersey_joins_pieces_that_never_overlap():
    def piece(track, frames, team="left", jersey="10"):
        return [{"frame": f, "track_id": track, "team": team, "jersey": jersey} for f in frames]

    state = pd.DataFrame(
        piece(1, range(1, 11))
        + piece(2, range(15, 30))  # same player after a gap: joins 1
        + piece(3, range(20, 25))  # overlaps 2 in time: a different player (a misread), kept apart
        + piece(4, range(31, 40))  # after 2 ends: joins the first chain
        + piece(5, range(1, 40), team="right")  # other team, same number
        + piece(6, range(12, 14), jersey=None)  # no number: never merged
    )
    merged = merge_by_jersey(state).groupby("track_id")["frame"].agg(["min", "max"]).to_dict("index")
    assert merged == {1: {"min": 1, "max": 39}, 3: {"min": 20, "max": 24}, 5: {"min": 1, "max": 39},
                      6: {"min": 12, "max": 13}}  # fmt: skip


def test_split_by_jersey_gives_each_piece_its_own_id_and_number():
    state = pd.DataFrame(
        [{"frame": f, "track_id": 5, "jersey": "31"} for f in range(1, 11)]
        + [{"frame": f, "track_id": 7, "jersey": "9"} for f in range(1, 11)]
    )
    out = split_by_jersey(state, {5: [(0, "25"), (4, "31"), (8, "25")], 7: [(0, "9")]})
    five = out[out["frame"].isin(range(1, 11)) & (state["track_id"] == 5)]
    assert five["jersey"].tolist() == ["25"] * 3 + ["31"] * 4 + ["25"] * 3
    assert five["track_id"].tolist() == [5] * 3 + [8] * 4 + [9] * 3  # new ids after the largest one
    assert (out.loc[state["track_id"] == 7, ["track_id", "jersey"]] == [7, "9"]).all().all()
    rejoined = merge_by_jersey(five.assign(team="right"))  # the two #25 pieces are one player again
    assert rejoined["track_id"].tolist() == [5] * 3 + [8] * 4 + [5] * 3


def test_unique_numbers_keeps_a_number_on_the_player_read_nearest_in_time():
    def rows(track, frames, jersey="31", team="right", read_track=None):
        return [{"frame": f, "track_id": track, "team": team, "jersey": jersey,
                 "read_track": track if read_track is None else read_track} for f in frames]  # fmt: skip

    state = pd.DataFrame(
        rows(5, range(1, 11))  # read as 31 at frame 2
        + rows(30, range(1, 11))  # read as 31 at frame 9
        + rows(7, range(1, 11), team="left")  # the other team's 31: no clash
        + rows(8, range(1, 11), jersey="9")  # another number: no clash
    )
    evidence = {(5, "31"): np.array([2]), (30, "31"): np.array([9]), (7, "31"): np.array([5])}
    out = unique_numbers(state, evidence)
    shows = out[out["jersey"] == "31"].groupby("frame")["track_id"].apply(set).to_dict()
    assert all(shows[f] == ({5, 7} if f <= 5 else {30, 7}) for f in range(1, 11))  # f = 5.5 is the midpoint
    assert out.loc[out["track_id"] == 8, "jersey"].eq("9").all()


def test_merge_by_jersey_spans_count_frames_without_a_number():
    # track 2 shows its number only from frame 6, but it exists from frame 1, alongside track 1
    state = pd.DataFrame(
        [{"frame": f, "track_id": 1, "team": "left", "jersey": "10"} for f in range(1, 5)]
        + [{"frame": f, "track_id": 2, "team": "left", "jersey": None if f < 6 else "10"} for f in range(1, 11)]
    )
    assert set(merge_by_jersey(state)["track_id"]) == {1, 2}  # overlapping: two players, not one
