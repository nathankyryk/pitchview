import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pitchview.gsr import list_clips
from pitchview.segment import GroundTruthTracks, MaskWriter, attach_masks, box_iou, mask_features, write_tracks


def test_mask_features_rectangle():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 40:60] = True
    x1, y1, x2, y2, foot_x, foot_y, area = mask_features(mask)
    assert (x1, y1, x2, y2) == (40, 20, 60, 80)
    assert foot_x == 49.5 and foot_y == 80
    assert area == 60 * 20


def test_mask_features_foot_follows_the_legs():
    # A body whose lowest pixels (a planted foot) sit right of the box centre.
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:70, 30:50] = True  # torso
    mask[70:90, 44:50] = True  # leg down to the grass
    _, _, _, y2, foot_x, foot_y, _ = mask_features(mask)
    assert y2 == 90 and foot_y == 90
    assert foot_x > 44  # the box centre (40) would be 4+ px off the foot


def test_mask_features_empty():
    assert mask_features(np.zeros((5, 5), dtype=bool)) is None


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    np.testing.assert_allclose(box_iou(a, b), [[1.0, 1 / 3, 0.0]])


def test_attach_masks_matches_one_to_one():
    dets = pd.DataFrame(
        {
            "frame": [1, 1, 1],
            "x1": [10.0, 100.0, 50.0],
            "y1": [10.0, 10.0, 50.0],
            "x2": [30.0, 120.0, 52.0],
            "y2": [60.0, 60.0, 52.0],
            "class_name": ["player", "player", "ball"],
        }
    )
    tracks = pd.DataFrame(
        {
            "frame": [1],
            "obj_id": [7],
            "x1": [11.0],
            "y1": [12.0],
            "x2": [29.0],
            "y2": [61.0],
            "foot_x": [24.0],
            "foot_y": [61.0],
        }
    )
    out = attach_masks(dets, tracks)
    assert out["has_mask"].tolist() == [True, False, False]
    assert out.loc[0, ["foot_x", "foot_y", "sam_obj_id"]].tolist() == [24.0, 61.0, 7]
    assert out.loc[1, ["foot_x", "foot_y"]].tolist() == [110.0, 60.0]  # box bottom-centre


def test_write_tracks_with_ground_truth_stand_in(gsr_root, tmp_path):
    (clip,) = list_clips("valid", gsr_root)
    seen = []
    n = write_tracks(GroundTruthTracks(), clip, tmp_path / "t.parquet", on_frame=lambda f, out: seen.append(f))
    # per frame: player, "other" and referee; the ball is skipped and the sub-pixel box is empty
    assert n == 18
    assert seen == [1, 2, 3, 4, 5, 6]
    table = pq.read_table(tmp_path / "t.parquet").to_pandas()
    referee = table[(table.frame == 1) & (table.obj_id == 14)].iloc[0]
    assert (referee.x2, referee.y2) == (64, 36)  # clipped to the 64x36 frame


def test_mask_writer_round_trips_rle(gsr_root, tmp_path):
    import json

    from pycocotools import mask as mask_utils

    (clip,) = list_clips("valid", gsr_root)
    writer = MaskWriter()
    write_tracks(GroundTruthTracks(), clip, tmp_path / "t.parquet", max_frames=2, on_frame=writer)
    writer.save(tmp_path / "m.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert len(rows) == 2 * 4  # 4 people per frame (the sub-pixel one encodes as an empty mask)
    player = next(r for r in rows if r["frame"] == 1 and r["obj_id"] == 11)
    decoded = mask_utils.decode({"size": player["size"], "counts": player["counts"].encode()})
    assert decoded.shape == (36, 64) and decoded.sum() == 8 * 20  # the 8x20 player box
