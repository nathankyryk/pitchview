import numpy as np
import pandas as pd

from pitchview.gamestate import (
    Variant,
    assemble,
    ball_positions,
    clip_half,
    jump_cuts,
    link_handoffs,
    merge_by_jersey,
    pooled_sides,
    side_evidence,
    split_by_jersey,
    split_tracks,
    swap_sides,
    to_predictions,
    unique_numbers,
    unplaced,
)
from pitchview.jersey import read_frames, split_votes

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


def test_unplaced_keeps_people_in_frames_without_a_pitch_mapping_for_the_video():
    n = 30
    dets = _dets(n)
    sam = pd.DataFrame(
        [_sam_row(f, 3, (100 + 5 * f, 100, 140 + 5 * f, 200)) for f in range(1, n + 1)]
        + [_sam_row(f, 9, (900, 300, 930, 380)) for f in range(1, n + 1)]  # nobody RF-DETR saw: never shown
    )
    Hs = {1: H, 2: H}  # then a close-up: frames more than 12 from frame 2 have no pitch mapping
    v = Variant(tracker="sam", smooth=1)
    state = assemble(dets, sam, Hs, None, {3: "14"}, n, v=v)
    assert sorted(state["frame"]) == list(range(1, 15))
    extra = unplaced(dets, sam, Hs, state, n, v=v)
    assert sorted(extra["frame"]) == list(range(15, n + 1))
    assert (extra["track_id"] == 3).all() and (extra["role"] == "player").all() and (extra["jersey"] == "14").all()
    assert extra["x_m"].isna().all() and (extra["sam_obj_id"] == 3).all()
    assert unplaced(dets, sam, {f: H for f in range(1, n + 1)}, state, n, v=v).empty


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


def test_unique_numbers_one_winner_per_clash():
    # SNGS-145: the real #14 is read 14 time after time; another player is read as 14 once, at frame 9.
    # Frame by frame, the stray read wins the frames nearest it (8-10) and the name jumps across;
    # with episodes, whoever wins most of a clash keeps the number through all of it.
    def rows(track, frames):
        return [{"frame": f, "track_id": track, "team": "right", "jersey": "14", "read_track": track} for f in frames]

    frames = [*range(1, 11), *range(30, 36)]  # two clashes, 20 frames apart
    state = pd.DataFrame(rows(18, frames) + rows(19, frames))
    evidence = {(18, "14"): np.array([1, 2, 3, 4, 5, 6]), (19, "14"): np.array([9, 33])}

    def holder(out):
        return out[out["jersey"] == "14"].set_index("frame")["track_id"].to_dict()

    per_frame = holder(unique_numbers(state, evidence))
    assert [per_frame[f] for f in range(1, 11)] == [18] * 7 + [19] * 3
    steady = holder(unique_numbers(state, evidence, episode_gap=5))
    assert [steady[f] for f in range(1, 11)] == [18] * 10  # 18 won 7 of the 10 frames
    assert [steady[f] for f in range(30, 36)] == [19] * 6  # a separate clash, which 19 wins
    joined = holder(unique_numbers(state, evidence, episode_gap=25))  # one clash: 18 won 7 of 16
    assert set(joined.values()) == {19}


def test_merge_by_jersey_spans_count_frames_without_a_number():
    # track 2 shows its number only from frame 6, but it exists from frame 1, alongside track 1
    state = pd.DataFrame(
        [{"frame": f, "track_id": 1, "team": "left", "jersey": "10"} for f in range(1, 5)]
        + [{"frame": f, "track_id": 2, "team": "left", "jersey": None if f < 6 else "10"} for f in range(1, 11)]
    )
    assert set(merge_by_jersey(state)["track_id"]) == {1, 2}  # overlapping: two players, not one


def _track(track, frames, x, team="right", jersey=None, role="player"):
    return [{"frame": f, "track_id": track, "role": role, "team": team, "jersey": jersey,
             "x1": x, "y1": 100.0, "x2": x + 40, "y2": 200.0} for f in frames]  # fmt: skip


def test_link_handoffs_joins_a_new_track_that_starts_where_an_old_one_let_go():
    state = pd.DataFrame(
        _track(0, range(1, 13), 100)  # lets go of the player at frame 10 and drifts (frames 10-12)
        + _track(38, range(10, 30), 102, jersey="25")  # SAM's new track on the same player
        + _track(5, range(1, 30), 400, jersey="9")  # someone else
    )
    out = link_handoffs(state)
    player = out[out["track_id"] == 0]
    assert set(out["track_id"]) == {0, 5}
    assert player["frame"].tolist() == list(range(1, 30))  # the drift is dropped: one row a frame
    assert set(player["jersey"]) == {"25"}  # the number now shows from frame 1


def test_link_handoffs_needs_one_team_close_boxes_a_short_gap_and_no_second_number():
    def ids(*tracks):
        return set(link_handoffs(pd.DataFrame([row for t in tracks for row in t]))["track_id"])

    assert ids(_track(0, range(1, 10), 100), _track(1, range(10, 20), 100)) == {0}  # joined
    assert ids(_track(0, range(1, 10), 100), _track(1, range(10, 20), 100, team="left")) == {0, 1}
    assert ids(_track(0, range(1, 10), 100), _track(1, range(10, 20), 300)) == {0, 1}  # far apart
    assert ids(_track(0, range(1, 10), 100), _track(1, range(25, 30), 100)) == {0, 1}  # 16 frames later
    assert ids(_track(0, range(1, 30), 100), _track(1, range(10, 20), 100)) == {0, 1}  # both on: two players
    assert ids(_track(0, range(1, 10), 100, jersey="14"), _track(1, range(10, 20), 100, jersey="25")) == {0, 1}
    assert ids(_track(0, range(1, 10), 100), _track(1, range(10, 20), 100, role="referee")) == {0, 1}


def test_link_handoffs_chains_keep_one_number():
    state = pd.DataFrame(
        _track(0, range(1, 10), 100, jersey="25")
        + _track(1, range(10, 20), 100)
        + _track(2, range(20, 30), 110, jersey="14")  # would put two numbers on one chain: not joined
    )
    out = link_handoffs(state)
    assert out.groupby("track_id")["jersey"].agg(set).to_dict() == {0: {"25"}, 2: {"14"}}


def test_jump_cuts_finds_lasting_changes_only():
    frames = list(range(1, 101, 5))  # 20 crops, 5 frames apart
    assert jump_cuts(frames, ["L"] * 10 + ["R"] * 10, 4) == [49]  # halfway from frame 46 to 51
    assert jump_cuts(frames, ["L"] * 9 + ["R"] * 2 + ["L"] * 9, 4) == []  # a short run is noise
    assert jump_cuts(frames, ["L"] * 6 + ["R"] * 6 + ["L"] * 8, 4) == [29, 59]  # there and back
    assert jump_cuts(frames, ["L"] * 18 + ["R"] * 2, 4) == []  # too short at the end too
    assert jump_cuts([], [], 4) == [] and jump_cuts([1], ["L"], 4) == []


def test_split_tracks_gives_each_later_piece_a_new_id():
    state = pd.DataFrame({"frame": [1, 2, 3, 4, 1, 2], "track_id": [5, 5, 5, 5, 7, 7]})
    assert split_tracks(state, {5: [3]})["track_id"].tolist() == [5, 5, 8, 8, 7, 7]
    assert split_tracks(state, {5: [2, 4], 7: [9]})["track_id"].tolist() == [5, 8, 8, 9, 7, 7]


def _swap_clip(n=40):
    """Red #8 (A, left) and blue #36 (B, right). SAM 3.1's masks 100 and 101 swap them at frame 21;
    masks 102 (red) and 103 (blue) stay on their players."""
    rng = np.random.default_rng(0)
    kit = {"red": np.eye(16)[0], "blue": np.eye(16)[1]}
    spots = {"A": (250, 100, "red"), "B": (650, 100, "blue"), "C": (300, 250, "red"), "D": (700, 250, "blue")}
    dets, sam, reads, emb = [], [], [], {"frame": [], "obj_id": [], "box": [], "mask": []}
    for f in range(1, n + 1):
        on = {100: "A" if f <= 20 else "B", 101: "B" if f <= 20 else "A", 102: "C", 103: "D"}
        for obj, who in on.items():
            x, y, colour = spots[who]
            dets.append(_box(f, x, y, 30, 60, "player"))
            sam.append(_sam_row(f, obj, (x, y, x + 30, y + 60)))
            vec = kit[colour] + rng.normal(0, 0.03, 16)
            for key, value in (("frame", f), ("obj_id", obj), ("box", vec), ("mask", vec)):
                emb[key].append(value)
            number = {"A": "8", "B": "36"}.get(who)
            if number:
                reads.append({"track_id": obj, "frame": f, "number": number, "conf": 0.95})
    emb = {k: np.array(v) for k, v in emb.items()}
    return pd.DataFrame(dets), pd.DataFrame(sam), pd.DataFrame(reads), emb


def test_split_jumps_gives_each_player_their_own_team_and_number_after_a_swap():
    n = 40
    Hs = {f: H for f in range(1, n + 1)}
    dets, sam, reads, emb = _swap_clip(n)
    v = Variant(tracker="sam", smooth=1)
    state = assemble(dets, sam, Hs, emb, {100: "8", 101: "36"}, n, v=v, pieces=split_votes(reads),
                     evidence=read_frames(reads), reads=reads)  # fmt: skip
    a, b = state[state["x1"] == 250], state[state["x1"] == 650]
    assert len(a) == len(b) == n
    assert set(a["team"]) == {"left"} and set(a["jersey"]) == {"8"}
    assert set(b["team"]) == {"right"} and set(b["jersey"]) == {"36"}
    assert a["track_id"].nunique() == b["track_id"].nunique() == 1  # the pieces of each player rejoin
    # Without any jersey reads, the kit alone cuts the swapped masks.
    state = assemble(dets, sam, Hs, emb, None, n, v=v)
    assert set(state.loc[state["x1"] == 250, "team"]) == {"left"}
    assert set(state.loc[state["x1"] == 650, "team"]) == {"right"}
    # Before: one team per mask, so each player shows the wrong team half the time.
    old = assemble(dets, sam, Hs, emb, None, n, v=Variant(tracker="sam", smooth=1, split_jumps=False))
    assert old.loc[old["x1"] == 250, "team"].nunique() == 2


def test_split_jumps_cuts_a_referee_mask_that_moves_onto_a_player():
    n = 30
    Hs = {f: H for f in range(1, n + 1)}
    dets = pd.DataFrame(
        [_box(f, 600, 300, 40, 100, "referee") for f in range(1, n + 1)]
        + [_box(f, 200, 100, 40, 100, "player") for f in range(1, n + 1)]
    )
    on = {f: (600, 300, 640, 400) if f <= 15 else (200, 100, 240, 200) for f in range(1, n + 1)}
    sam = pd.DataFrame([_sam_row(f, 8, box) for f, box in on.items()])
    state = assemble(dets, sam, Hs, None, None, n, v=Variant(tracker="sam", smooth=1))
    pieces = state.groupby("track_id").agg(role=("role", "first"), first=("frame", "min"), last=("frame", "max"))
    assert sorted(map(tuple, pieces.to_numpy().tolist())) == [("player", 16, 30), ("referee", 1, 15)]
    old = assemble(dets, sam, Hs, None, None, n, v=Variant(tracker="sam", smooth=1, split_jumps=False))
    assert old["track_id"].nunique() == 1


def test_join_parts_makes_one_person_of_a_torso_mask_and_a_legs_mask():
    dets = pd.DataFrame([_box(f, 200, 100, 40, 100, "player") for f in range(1, FRAMES + 1)])
    sam = pd.DataFrame(
        [_sam_row(f, 3, (202, 100, 238, 150), foot_x=215.0) for f in range(1, FRAMES + 1)]  # torso: IoU 0.45
        + [_sam_row(f, 4, (204, 150, 236, 200), foot_x=222.0) for f in range(1, FRAMES + 1)]  # legs: IoU 0.4
    )
    state = assemble(dets, sam, _all_frames(), None, None, FRAMES, v=Variant(tracker="sam", smooth=1))
    assert len(state) == FRAMES and set(state["track_id"]) == {3} and set(state["role"]) == {"player"}
    assert set(state["sam_parts"]) == {"4"}
    assert (state[["x1", "y1", "x2", "y2"]].to_numpy() == [202, 100, 238, 200]).all()  # the union box
    assert np.allclose(state["x_m"], 222 / 10 - 50) and np.allclose(state["y_m"], 200 / 10 - 30)  # legs' foot
    v = Variant(tracker="sam", smooth=1, join_parts=False)
    assert assemble(dets, sam, _all_frames(), None, None, FRAMES, v=v).empty  # neither half matched: gone


def test_join_parts_leaves_masks_that_match_a_box_on_their_own():
    dets = pd.DataFrame([_box(f, 200, 100, 40, 100, "player") for f in range(1, FRAMES + 1)])
    sam = pd.DataFrame(
        [_sam_row(f, 3, (201, 101, 239, 200)) for f in range(1, FRAMES + 1)]  # the whole player
        + [_sam_row(f, 4, (210, 120, 230, 140)) for f in range(1, FRAMES + 1)]  # a stray bit inside the box
    )
    state = assemble(dets, sam, _all_frames(), None, None, FRAMES, v=Variant(tracker="sam", smooth=1))
    assert set(state["track_id"]) == {3} and set(state["sam_parts"]) == {""}


def test_people_well_off_the_pitch_are_dropped():
    # x_m = x / 10 - 50 and y_m = y / 10 - 30; the lines are at x = +-52.5 m and y = +-34 m
    dets = pd.DataFrame(
        [_box(f, 1130, 300, 40, 100, "goalkeeper") for f in range(1, FRAMES + 1)]  # x = 65 m, behind the goal
        + [_box(f, 280, 550, 40, 100, "player") for f in range(1, FRAMES + 1)]  # y = 35 m: 1 m over the touchline
    )
    state = assemble(dets, NO_SAM, _all_frames(), None, None, FRAMES, v=Variant(smooth=1))
    assert set(state["role"]) == {"player"}
    kept = assemble(dets, NO_SAM, _all_frames(), None, None, FRAMES, v=Variant(smooth=1, off_pitch=-1))
    assert set(kept["role"]) == {"player", "goalkeeper"}


def _keeper_clip(n=30, detected_keeper=False):
    """Two red and two blue players, and a keeper in green (x = -45 m) the detector calls a player."""
    rng = np.random.default_rng(0)
    kit = {"red": np.eye(16)[0], "blue": np.eye(16)[1], "green": np.eye(16)[2]}
    people = [(250, 100, "red", "player"), (300, 250, "red", "player"), (650, 100, "blue", "player"),
              (700, 250, "blue", "player"), (50, 300, "green", "player")]  # fmt: skip
    if detected_keeper:  # a keeper the detector does see, in that half at the same time
        people.append((30, 150, "green", "goalkeeper"))
    dets, sam, emb = [], [], {"frame": [], "obj_id": [], "box": [], "mask": []}
    for f in range(1, n + 1):
        for i, (x, y, colour, role) in enumerate(people):
            dets.append(_box(f, x, y, 30, 60, role))
            sam.append(_sam_row(f, 100 + i, (x, y, x + 30, y + 60)))
            vec = kit[colour] + rng.normal(0, 0.03, 16)
            for key, value in (("frame", f), ("obj_id", 100 + i), ("box", vec), ("mask", vec)):
                emb[key].append(value)
    emb = {k: np.array(v) for k, v in emb.items()}
    return pd.DataFrame(dets), pd.DataFrame(sam), emb


def test_keeper_by_kit_makes_a_player_in_a_keeper_kit_the_goalkeeper_of_his_half():
    n = 30
    Hs = {f: H for f in range(1, n + 1)}
    kits = np.stack([np.eye(16)[2], np.eye(16)[3]])  # the match's two keeper kits: green and yellow
    v = Variant(tracker="sam", smooth=1)
    dets, sam, emb = _keeper_clip(n)
    state = assemble(dets, sam, Hs, emb, None, n, v=v, keeper_kits=kits)
    keeper = state[state["x1"] == 50]
    assert set(keeper["role"]) == {"goalkeeper"} and set(keeper["team"]) == {"left"}
    assert set(state.loc[state["x1"] != 50, "role"]) == {"player"}
    without = assemble(dets, sam, Hs, emb, None, n, v=v)
    assert set(without.loc[without["x1"] == 50, "role"]) == {"player"}
    dets10, sam10, emb10 = _keeper_clip(10)
    short = assemble(dets10, sam10, {f: H for f in range(1, 11)}, emb10, None, 10, v=v, keeper_kits=kits)
    assert set(short.loc[short["x1"] == 50, "role"]) == {"player"}  # 10 crops: too few to be sure
    dets, sam, emb = _keeper_clip(n, detected_keeper=True)
    state = assemble(dets, sam, Hs, emb, None, n, v=v, keeper_kits=kits)
    assert set(state.loc[state["x1"] == 50, "role"]) == {"player"}  # his half already has a keeper


def _sides(left_kit, left_x, right_kit, right_x):
    unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)
    return {"left": (unit(left_kit), left_x), "right": (unit(right_kit), right_x)}


def test_clip_half_reads_the_half_from_the_clip_info():
    assert clip_half({"game_time_start": "2 - 23:00"}) == 2
    assert clip_half({"game_time_start": "1 - 00:30"}) == 1
    assert clip_half({}) is None


def test_side_evidence_gives_each_team_its_kit_and_mean_x():
    state = pd.DataFrame({"frame": [1, 1, 2, 1], "sam_obj_id": [5, 6, 5, 7], "role": ["player"] * 3 + ["goalkeeper"],
                          "team": ["left", "right", "left", "right"], "x_m": [-10.0, 4.0, -6.0, 50.0]})  # fmt: skip
    emb = {"frame": np.array([1, 1, 2, 1]), "obj_id": np.array([5, 6, 5, 7]),
           "mask": np.array([[2.0, 0], [0, 3.0], [4.0, 0], [1.0, 1.0]])}  # fmt: skip
    ev = side_evidence(state, emb)
    assert np.allclose(ev["left"][0], [1, 0]) and ev["left"][1] == -8.0
    assert np.allclose(ev["right"][0], [0, 1]) and ev["right"][1] == 4.0  # the keeper isn't a player
    assert side_evidence(state[state["team"] == "left"], emb) is None


def test_pooled_sides_names_a_coin_flip_clip_by_its_match_half():
    red, white = [1.0, 0.1], [0.1, 1.0]
    evidence = {
        "a": _sides(red, -12.0, white, 9.0),  # red defends the left goal, clearly
        "b": _sides(red, -3.0, white, 5.0),
        "c": _sides(white, 30.2, red, 30.6),  # play in one half: this clip's own guess is wrong
        "d": _sides(white, -20.0, red, 15.0),  # the other half: teams changed ends
        "e": _sides(red, 2.0, white, 1.0),  # alone in its half: its own naming stands
    }
    groups = {"a": ("7", 1), "b": ("7", 1), "c": ("7", 1), "d": ("7", 2), "e": ("8", 1)}
    assert pooled_sides(evidence, groups) == {"c"}
    assert pooled_sides({**evidence, "f": None}, {**groups, "f": ("7", 1)}) == {"c"}


def test_swap_sides_swaps_players_only():
    state = pd.DataFrame({"role": ["player", "player", "goalkeeper", "referee", "player"],
                          "team": ["left", "right", "left", None, None]})  # fmt: skip
    assert swap_sides(state)["team"].fillna("-").tolist() == ["right", "left", "left", "-", "-"]
