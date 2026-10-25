import pandas as pd

from pitchview.track import bytetrack


def _box(frame, x1, y1, w, h, class_name):
    return {"frame": frame, "x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h, "score": 0.9, "class_name": class_name}


def test_bytetrack_keeps_two_people_apart():
    rows = []
    for frame in range(1, 11):
        rows.append(_box(frame, 100 + 2 * frame, 100, 40, 100, "player"))
        rows.append(_box(frame, 500 - 2 * frame, 120, 40, 100, "referee"))
        rows.append(_box(frame, 300, 300, 10, 10, "ball"))
    tracked = bytetrack(pd.DataFrame(rows), n_frames=12)

    assert set(tracked["class_name"]) == {"player", "referee"}  # the ball is not tracked
    ids = tracked.groupby("class_name")["track_id"].unique()
    assert len(ids["player"]) == 1 and len(ids["referee"]) == 1  # each keeps one id throughout
    assert ids["player"][0] != ids["referee"][0]
    assert len(tracked) >= 18  # nearly every detection joins a track
