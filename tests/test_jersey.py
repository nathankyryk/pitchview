import numpy as np
import pandas as pd
from PIL import Image

from pitchview.evaluate import jersey_summary, jersey_truth, position_errors
from pitchview.gsr import Box
from pitchview.jersey import (
    NOT_PLAYER_NUMBERS,
    load_reads,
    parse_number,
    read_frames,
    shirt_crop,
    shirt_region,
    split_votes,
    vote,
)


def test_shirt_region():
    assert shirt_region((100, 200, 140, 300)) == (104, 215, 136, 260)


def test_shirt_crop_greys_out_everything_but_the_mask():
    image = Image.fromarray(np.full((300, 200, 3), 200, dtype=np.uint8))
    box = (100, 200, 140, 300)  # shirt region (104, 215, 136, 260)
    assert shirt_crop(image, box).size == (32, 45)
    mask = np.zeros((300, 200), dtype=bool)
    mask[:, :120] = True  # the player covers the region's left 16 columns
    pixels = np.array(shirt_crop(image, box, mask))
    assert (pixels[:, :16] == 200).all() and (pixels[:, 16:] == 128).all()
    edge = shirt_crop(image, (180, 250, 220, 350), np.ones((300, 200), dtype=bool))  # runs off the image
    assert edge.size == (32, 45) and (np.array(edge)[:, -16:] == 128).all()


def test_parse_number():
    assert parse_number("10") == "10"
    assert parse_number(" 7.") == "7"
    assert parse_number("abc") is None
    assert parse_number("1234") is None


def test_vote_weighs_confident_reads_and_abstains():
    reads = [
        ("a", "10", 0.9), ("a", "18", 0.6), ("a", "18", 0.55), ("a", "70", 0.3),  # 10: 0.9 < 18: 1.15
        ("b", "7", 0.4), ("b", None, 0.0),  # nothing confident
        ("c", "23", 0.95),
    ]  # fmt: skip
    assert vote(reads) == {"a": "18", "b": None, "c": "23"}
    assert vote(reads, min_conf=0.7) == {"a": "10", "b": None, "c": "23"}


def test_vote_needs_agreeing_reads_and_skips_excluded_numbers():
    reads = [
        ("a", "1", 0.9), ("a", "1", 0.8), ("a", "7", 0.6),  # "1" wins unless excluded
        ("b", "23", 0.95),  # a single read
    ]  # fmt: skip
    assert vote(reads) == {"a": "1", "b": "23"}
    assert vote(reads, exclude=NOT_PLAYER_NUMBERS) == {"a": "7", "b": "23"}
    assert vote(reads, min_reads=2) == {"a": "1", "b": None}
    assert vote(reads, min_reads=2, exclude=NOT_PLAYER_NUMBERS) == {"a": None, "b": None}


def test_load_reads_round_trip(tmp_path):
    path = tmp_path / "reads.parquet"
    pd.DataFrame({"track_id": [3, 3], "frame": [1, 11], "number": ["10", None], "conf": [0.9, 0.1]}).to_parquet(path)
    assert load_reads(path) == [(3, "10", 0.9), (3, None, 0.1)]  # None, not NaN: vote() must skip it


def _gt(frame, track, x, role="player", jersey="10"):
    return Box(frame, track, 1, role, "left", jersey, x, 0, 10, 20, None, None)


class _FakeClip:
    def __init__(self, boxes):
        self.boxes = boxes

    def boxes_by_frame(self):
        out = {}
        for b in self.boxes:
            out.setdefault(b.frame, []).append(b)
        return out


def test_jersey_truth_follows_the_most_matched_player():
    clip = _FakeClip(
        [_gt(f, 1, 0) for f in (1, 2, 3)]
        + [_gt(f, 2, 100, jersey=None) for f in (1, 2, 3)]
        + [_gt(f, 3, 200, role="referee", jersey=None) for f in (1, 2, 3)]
    )
    tracks = pd.DataFrame(
        [  # track 7 sits on player 1 twice and on player 2 once; track 8 on player 2; track 9 on the referee
            (1, 7, 0), (2, 7, 0), (3, 7, 100),
            (1, 8, 100), (2, 8, 100),
            (1, 9, 200),
            (3, 6, 500),  # matches nobody
        ],
        columns=["frame", "track_id", "x1"],
    )  # fmt: skip
    tracks = tracks.assign(y1=0.0, x2=tracks["x1"] + 10, y2=20.0)
    truth = jersey_truth(tracks, clip).set_index("track_id")
    assert truth.to_dict("index") == {7: {"gt_jersey": "10", "frames": 2}, 8: {"gt_jersey": None, "frames": 2}}


def test_position_errors_measure_matched_people_in_metres():
    gt = [
        Box(1, 1, 1, "player", "left", "10", 0, 0, 10, 20, 3.0, 4.0),
        Box(1, 2, 1, "player", "left", "7", 100, 0, 10, 20, -5.0, 0.0),
        Box(1, 3, 4, "ball", None, None, 200, 0, 5, 5, 1.0, 1.0),  # the ball is not a person: never matched
        Box(2, 1, 1, "player", "left", "10", 0, 0, 10, 20, 3.0, 4.0),
    ]
    state = pd.DataFrame(
        [  # frame, x1, x_m, y_m
            (1, 0, 0.0, 0.0),  # player 1, 5 m off
            (1, 100, -5.0, 1.0),  # player 2, 1 m off
            (1, 200, 1.0, 1.0),  # on the ball
            (2, 0, float("nan"), float("nan")),  # player 1, no pitch mapping this frame
            (2, 400, 3.0, 4.0),  # nobody there
        ],
        columns=["frame", "x1", "x_m", "y_m"],
    )
    state = state.assign(y1=0.0, x2=state["x1"] + 10, y2=20.0)
    errors = position_errors(state, _FakeClip(gt))
    assert errors["error_m"].iloc[:2].tolist() == [5.0, 1.0]
    assert len(errors) == 3 and np.isnan(errors["error_m"].iloc[2])


def test_jersey_summary_counts_no_number_as_right_when_none_is_visible():
    s = jersey_summary(["10", None, "4", None], ["10", None, "7", "9"], [10, 30, 20, 40])
    assert s["agree"] == 0.4  # 10 + 30 of 100 frames
    assert s["read"] == 0.5 and s["precision"] == 0.5
    assert s["accuracy"] == 1 / 3  # of the three numbered players, only "10" was read


def test_vote_merges_a_lone_digit_into_the_two_digit_number_it_is_part_of():
    reads = [("a", "4", 0.9), ("a", "4", 0.8), ("a", "14", 0.75), ("b", "4", 0.9), ("c", "1", 0.9), ("c", "17", 0.6)]
    assert vote(reads) == {"a": "4", "b": "4", "c": "1"}
    merged = vote(reads, merge_partial=True, exclude=NOT_PLAYER_NUMBERS)
    assert merged == {"a": "14", "b": "4", "c": "17"}  # a stray "1" now supports 17 instead of being lost


def _reads(track, pattern):
    """pattern: (frame, number or None, confidence) tuples for one track."""
    return [{"track_id": track, "frame": f, "number": n, "conf": c} for f, n, c in pattern]


def test_split_votes_cuts_a_track_that_jumps_to_another_player():
    # SNGS-123's track 5: #25 (two sure reads, one unsure), then #31 for good
    jump = [(141, "25", 0.98), (201, "7", 0.2), (271, "25", 0.83), (281, "25", 0.35), (451, "1", 0.9),
            (501, "31", 0.95), (511, "31", 0.83), (531, "31", 0.97), (581, "21", 0.8), (601, "31", 0.98)]  # fmt: skip
    # one player throughout: a stray "4" (half of "14"), and a lone sure misread with nothing behind it
    steady = [(1, "14", 0.9), (81, "4", 0.8), (261, "4", 0.3), (361, "14", 0.9), (371, "14", 0.95)]
    lone = [(10, "7", 0.9), (100, "17", 0.9), (110, "17", 0.9), (120, "17", 0.8)]
    # the earlier number is read again after the later one starts: no clean switch, no cut
    mixed = [(241, "33", 0.52), (261, "33", 0.95), (321, "23", 0.15), (381, "23", 0.85), (391, "33", 0.43)]
    # SNGS-145's #16: four sure reads, then "10" once sure and twice unsure: a misread, not a jump
    misread = [(1, "16", 0.94), (141, "16", 0.99), (151, "16", 0.99), (221, "16", 0.98), (231, "10", 0.98),
               (291, "10", 0.21), (351, "10", 0.27)]  # fmt: skip
    tracks = _reads(5, jump) + _reads(10, steady) + _reads(11, lone) + _reads(33, mixed) + _reads(40, [])
    reads = pd.DataFrame(tracks + _reads(16, misread))
    pieces = split_votes(reads)
    assert pieces[5] == [(0, "25"), (392, "31")]  # halfway between the last 25 (281) and the first 31 (501)
    assert pieces[10] == [(0, "14")] and pieces[11] == [(0, "17")] and pieces[33] == [(0, "33")]
    assert pieces[16] == [(0, "16")]
    unsplit = vote(load_reads(reads), min_conf=0.7, exclude=NOT_PLAYER_NUMBERS)
    assert {t: p[0][1] for t, p in pieces.items() if len(p) == 1} == {t: unsplit[t] for t in (10, 11, 16, 33)}


def test_split_votes_finds_a_second_jump():
    pattern = [(f, "8", 0.9) for f in (1, 11, 21)] + [(f, "20", 0.9) for f in (101, 111)]
    pattern += [(f, "5", 0.9) for f in (201, 211, 221)]
    assert split_votes(pd.DataFrame(_reads(1, pattern))) == {1: [(0, "8"), (62, "20"), (157, "5")]}


def test_read_frames_keeps_sure_reads_per_track_and_number():
    reads = pd.DataFrame(
        _reads(5, [(30, "31", 0.9), (10, "31", 0.8), (20, "31", 0.5), (40, "1", 0.99), (50, None, 0.9)])
    )
    assert {k: v.tolist() for k, v in read_frames(reads).items()} == {(5, "31"): [10, 30]}
