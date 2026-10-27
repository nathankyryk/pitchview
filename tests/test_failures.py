import numpy as np
import pandas as pd

from pitchview.failures import _on_pitch, _touching_person, ball_case, match
from pitchview.gsr import Box


def _box(role, x, y, w, h, pitch=(0.0, 0.0), track=1):
    return Box(1, track, 4 if role == "ball" else 1, role, None, None, x, y, w, h, *pitch)


def test_match_pairs_boxes_one_to_one_above_the_iou_floor():
    pred = np.array([[0, 0, 10, 10], [100, 100, 110, 110], [50, 50, 60, 60]], float)
    truth = np.array([[101, 101, 111, 111], [0, 0, 10, 10]], float)
    assert sorted(match(pred, truth)) == [(0, 1), (1, 0)]
    assert match(pred[:0], truth) == []


def test_loose_ball_rules():
    player = _box("player", 100, 100, 20, 60)
    assert _touching_person(_box("ball", 105, 160, 6, 6), [player])  # at the player's feet
    assert not _touching_person(_box("ball", 300, 300, 6, 6), [player])
    assert _on_pitch(_box("ball", 0, 0, 6, 6, pitch=(10.0, -20.0)))
    assert not _on_pitch(_box("ball", 0, 0, 6, 6, pitch=(10.0, 34.5)))  # over the touchline
    assert not _on_pitch(_box("ball", 0, 0, 6, 6, pitch=(None, None)))


class _Frame:
    def __init__(self, index):
        self.index = index


class _Clip:
    name, fps = "c", 25.0

    def __init__(self, boxes_by_frame, n):
        self._b, self.frames = boxes_by_frame, [_Frame(i) for i in range(1, n + 1)]

    def boxes_by_frame(self):
        return self._b


def test_ball_case_finds_the_longest_run_of_missed_loose_balls():
    ball = _box("ball", 500, 500, 10, 10, pitch=(5.0, 5.0))
    clip = _Clip({f: [ball] for f in range(1, 11)}, 10)
    found = [2, 6]  # detected at frames 2 and 6 only: longest miss is frames 7-10
    dets = pd.DataFrame({"frame": found, "x1": 500.0, "y1": 500.0, "x2": 510.0, "y2": 510.0, "score": 0.9,
                         "class_name": "ball"})  # fmt: skip
    case = ball_case([clip], {"c": dets})
    assert case.frames == [9] and "4 frames" in case.caption
