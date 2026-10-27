import math

import pandas as pd

from pitchview.coco import CLASS_NAMES, clip_box
from pitchview.evaluate import detection_ap
from pitchview.gsr import list_clips


def _perfect_detections(clip, drop_ball=False):
    """The clip's own boxes as detections, clipped like the training labels."""
    rows = []
    for frame in clip.frames:
        for box in clip.boxes_by_frame().get(frame.index, []):
            name = CLASS_NAMES.get(box.category_id)
            bbox = clip_box(box, frame.width, frame.height) if name else None
            if bbox is None or (drop_ball and name == "ball"):
                continue
            x, y, w, h = bbox
            rows.append({"clip": clip.name, "frame": frame.index, "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                         "score": 0.9, "class_name": name})  # fmt: skip
    return pd.DataFrame(rows)


def test_detection_ap_is_perfect_for_the_labels_and_zero_for_a_missed_class(gsr_root):
    clip = list_clips("valid", gsr_root)[0]
    perfect = detection_ap([clip], {clip.name: _perfect_detections(clip)}, stride=1)
    assert perfect["player"]["ap"] > 0.99 and perfect["referee"]["ap"] > 0.99 and perfect["ball"]["ap"] > 0.99
    assert math.isnan(perfect["goalkeeper"]["ap"])  # no goalkeeper in the clip: no score, not zero

    no_ball = detection_ap([clip], {clip.name: _perfect_detections(clip, drop_ball=True)}, stride=1)
    assert no_ball["ball"]["ap"] == 0 and no_ball["player"]["ap"] > 0.99

    unnamed = _perfect_detections(clip)  # a box from the detector's unnamed extra class is skipped
    unnamed = pd.concat([unnamed, unnamed.head(1).assign(class_name="4", score=0.2)], ignore_index=True)
    assert detection_ap([clip], {clip.name: unnamed}, stride=1)["player"]["ap"] > 0.99
