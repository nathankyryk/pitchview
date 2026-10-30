import json

import numpy as np
import pandas as pd

from pitchview.render import (
    FALLBACK_KITS,
    Radar,
    anchors,
    fading_tags,
    first_reads,
    kit_colours,
    load_roster,
    short_name,
    tag_position,
    tag_text,
)


def test_short_name():
    assert short_name("Kevin De Bruyne") == "K. De Bruyne"
    assert short_name("Lionel Messi") == "L. Messi"
    assert short_name("Pedri") == "Pedri"
    assert short_name("  Pedri ") == "Pedri"


def test_roster_lookup_and_tag_text(tmp_path):
    path = tmp_path / "SNGS-001.json"
    path.write_text(
        json.dumps({"match": "x", "left": {"10": "Lionel Messi", "7": ""}, "right": {"9": "Erling Haaland"}})
    )
    roster = load_roster(path)
    assert roster == {"left": {"10": "Lionel Messi", "7": ""}, "right": {"9": "Erling Haaland"}, "goalkeeper": {}}
    assert tag_text("player", "left", "10", True, roster) == "L. Messi"
    assert tag_text("player", "left", "7", True, roster) == "#7"  # on the sheet, no name filled in yet
    assert tag_text("player", "right", "10", True, roster) == "#?"  # not on this side's sheet: a misread
    assert tag_text("player", "right", "10", True, roster, sheet_only=False) == "#10"  # the labels, as they are
    assert tag_text("player", "right", "10", True, {}) == "#10"  # no sheet: show what was read
    assert tag_text("player", "left", "10", False, roster) == "#?"  # not read yet
    assert tag_text("player", "left", None, True, roster) == "#?"
    assert tag_text("goalkeeper", "left", None, True, roster) == "GK"
    assert tag_text("referee", None, None, True, roster) is None
    assert load_roster(tmp_path / "missing.json") == {}


def test_goalkeeper_tag_comes_from_the_sheet(tmp_path):
    # GSR doesn't number goalkeepers; a team has one keeper on the pitch, so role and side name him
    path = tmp_path / "SNGS-118.json"
    path.write_text(json.dumps({"goalkeeper": {"left": "32", "right": 1},
                                "left": {"32": "Dejan Stojanovic"}, "right": {"10": "B. Stillhart"}}))  # fmt: skip
    roster = load_roster(path)
    assert roster["goalkeeper"] == {"left": "32", "right": "1"}
    assert tag_text("goalkeeper", "left", None, True, roster) == "D. Stojanovic"
    assert tag_text("goalkeeper", "right", None, True, roster) == "#1"  # keeper's number, no name on the sheet
    assert tag_text("goalkeeper", None, None, True, roster) == "GK"  # side unknown


def test_first_reads_waits_for_the_voted_number():
    reads = pd.DataFrame(
        [(1, 1, "4", 0.9), (1, 11, "14", 0.6), (1, 21, "14", 0.8), (1, 31, "14", 0.9), (2, 1, None, 0.0)],
        columns=["track_id", "frame", "number", "conf"],
    )
    assert first_reads(reads, {1: "14", 2: None}, min_conf=0.7) == {1: 21}  # frame 11 was not confident


def test_tag_stays_above_the_player_and_inside_the_frame():
    assert tag_position(500, 300, 60, 30, 1920, 1080) == (470, 262)
    assert tag_position(5, 10, 60, 30, 1920, 1080) == (2, 2)  # top-left corner
    assert tag_position(1919, 300, 60, 30, 1920, 1080) == (1858, 262)  # right edge


def test_anchors_smooth_along_each_track():
    state = pd.DataFrame({"frame": [1, 2, 3], "track_id": 7, "x1": [100.0, 130.0, 100.0], "x2": [120.0, 150.0, 120.0],
                          "y1": [50.0, 50.0, 50.0]})  # fmt: skip
    a = anchors(state, window=3).set_index("frame")
    assert a.loc[2, "ax"] == np.mean([110, 140, 110])  # the jump at frame 2 is damped


def _state(frames=(1,)):
    rows = []
    for f in frames:
        rows.append({"frame": f, "role": "player", "team": "left", "x1": 0, "y1": 0, "x2": 40, "y2": 100})
        rows.append({"frame": f, "role": "player", "team": "right", "x1": 100, "y1": 0, "x2": 140, "y2": 100})
    return pd.DataFrame(rows)


def test_kit_colours_ignore_grass_and_fall_back_when_alike():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :] = (40, 140, 50)  # grass
    image[15:60, 4:36] = (200, 30, 30)  # red shirt on the left player
    image[15:60, 104:136] = (230, 230, 230)  # white shirt on the right player
    image[15:20, 4:36] = (40, 140, 50)  # some grass inside the left crop
    kits = kit_colours({1: image}, _state())
    assert kits["left"][0] > 200 and kits["left"][1] < 80  # vivid red
    assert min(kits["right"]) > 220  # white
    same = image.copy()
    same[15:60, 104:136] = (200, 30, 30)
    assert kit_colours({1: same}, _state()) == FALLBACK_KITS


def test_radar_puts_the_camera_side_at_the_bottom():
    radar = Radar(scale=5.0, margin=10)
    assert radar.px(-52.5, -34) == (10, 10)
    assert radar.px(52.5, 34) == (535, 350)
    pane = radar.draw([(0.0, 0.0, (255, 0, 0), "10")], ball=(10.0, 5.0))
    assert pane.shape == (360, 545, 3)


def test_truth_state_turns_labels_into_a_game_state():
    from pitchview.gsr import Box
    from pitchview.render import truth_state

    boxes = [
        Box(1, 5, 1, "player", "left", "10", 10, 20, 30, 40, 1.0, 2.0),
        Box(1, None, 7, "other", None, None, 0, 0, 5, 5, None, None),  # no track: skipped
        Box(1, 9, 4, "ball", None, None, 50, 50, 4, 4, 3.0, -1.0),
    ]
    state, ball = truth_state(boxes)
    assert state[["track_id", "role", "team", "jersey", "x1", "y2", "x_m"]].values.tolist() == [
        [5, "player", "left", "10", 10, 60, 1.0]
    ]
    assert ball.values.tolist() == [[1, 3.0, -1.0]]


def test_track_colour_uses_the_role_then_the_track():
    from pitchview.render import ROLE_COLOURS, TRACK_PALETTE, track_colour

    assert track_colour("referee", 3) == ROLE_COLOURS["referee"]
    assert track_colour("person", 3) == TRACK_PALETTE[3]
    assert track_colour("person", 3 + len(TRACK_PALETTE)) == TRACK_PALETTE[3]


def test_faststart_moves_the_index_first(tmp_path):
    import av

    from pitchview.render import faststart, index_first

    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as out:  # PyAV writes the index last by default
        stream = out.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(10):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 20, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    assert not index_first(path)
    faststart(path)
    assert index_first(path)
    with av.open(str(path)) as c:
        assert sum(1 for _ in c.decode(video=0)) == 10


def test_shrink_reencodes_smaller_with_the_index_first(tmp_path):
    import av

    from pitchview.render import index_first, shrink

    src = tmp_path / "big.mp4"
    with av.open(str(src), "w") as out:
        stream = out.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 128, 72, "yuv420p"
        for i in range(8):
            frame = av.VideoFrame.from_ndarray(np.full((72, 128, 3), i * 25, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    dst = tmp_path / "small" / "clip.mp4"
    shrink(src, dst, height=36)
    assert index_first(dst)
    with av.open(str(dst)) as c:
        s = c.streams.video[0]
        assert (s.width, s.height) == (64, 36)
        assert sum(1 for _ in c.decode(video=0)) == 8


def test_a_tag_fades_out_after_its_track_ends():
    seen = {7: (10, 100.0, 50.0, (255, 0, 0), "L. Messi")}
    ((ax, ay, colour, text, alpha),) = fading_tags(seen, {}, 11, fade=10)
    assert (ax, ay, colour, text) == (100.0, 50.0, (255, 0, 0), "L. Messi")
    assert 0.9 < alpha < 1
    assert fading_tags(seen, {}, 20, fade=10)[0][4] < 0.1
    assert fading_tags(seen, {}, 21, fade=10) == []  # gone after `fade` frames
    assert fading_tags(seen, {}, 11, fade=0) == []  # fade 0: tags vanish at once


def test_a_tag_drawn_again_is_live_not_fading():
    seen = {7: (10, 100.0, 50.0, (255, 0, 0), "#10")}
    assert fading_tags(seen, {7: "#10"}, 12, fade=10) == []


def test_a_fading_name_gives_way_to_the_same_name_on_another_track():
    seen = {7: (10, 100.0, 50.0, (255, 0, 0), "L. Messi"), 8: (10, 300.0, 50.0, (255, 0, 0), "#?")}
    faded = fading_tags(seen, {9: "L. Messi", 10: "#?"}, 11, fade=10)
    assert [t[3] for t in faded] == ["#?"]  # one "L. Messi" on screen; "#?" is not a name
