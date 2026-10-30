from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from pitchview.gsr import list_clips
from pitchview.segment import (
    RESTART_ID_STEP,
    GroundTruthTracks,
    MaskWriter,
    Sam31,
    attach_masks,
    box_iou,
    clean_clip,
    clean_mask,
    drop_offload_state_arg,
    mask_features,
    match_masks,
    rles,
    write_tracks,
)


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


def test_clean_mask_drops_stray_pieces_and_keeps_a_cut_off_leg():
    mask = np.zeros((200, 200), dtype=bool)
    mask[20:70, 40:60] = True  # the body
    mask[74:94, 48:54] = True  # a leg below a gap (an occluder): near and big enough, kept
    mask[150:160, 150:170] = True  # a stray piece far away: dropped
    mask[18:19, 70:71] = True  # a speck right by the body but tiny: dropped
    clean = clean_mask(mask)
    assert clean[20:70, 40:60].all() and clean[74:94, 48:54].all()
    assert not clean[150:160, 150:170].any() and not clean[18, 70]
    x1, y1, x2, y2, _, foot_y, _ = mask_features(mask)
    assert (x1, y1, x2, y2) == (40, 20, 60, 94) and foot_y == 94  # not stretched to (170, 160)


def test_clean_clip_rewrites_tracks_and_masks(tmp_path):
    import json

    from pycocotools import mask as mask_utils

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:60, 20:40] = 1
    mask[90:95, 90:95] = 1  # stray
    rle = mask_utils.encode(np.asfortranarray(mask))
    (tmp_path / "raw.jsonl").write_text(
        json.dumps({"frame": 1, "obj_id": 4, "size": rle["size"], "counts": rle["counts"].decode()}) + "\n"
    )
    raw = pd.DataFrame([{"clip": "c", "frame": 1, "obj_id": 4, "prob": 0.9, "x1": 20.0, "y1": 10.0, "x2": 95.0,
                         "y2": 95.0, "foot_x": 92.0, "foot_y": 95.0, "area": 1025}])  # fmt: skip
    raw.to_parquet(tmp_path / "raw.parquet")
    emb = {"frame": np.array([1, 6]), "obj_id": np.array([4, 4]), "box": np.eye(2), "mask": np.eye(2)}
    np.savez(tmp_path / "raw.npz", **emb)  # frame 1 was cropped from the stretched box; frame 6 is untouched
    n, changed = clean_clip(
        tmp_path / "raw.parquet", tmp_path / "raw.jsonl", tmp_path / "t.parquet", tmp_path / "m.jsonl",
        tmp_path / "raw.npz", tmp_path / "e.npz",
    )  # fmt: skip
    assert (n, changed) == (1, 1)
    assert list(np.load(tmp_path / "e.npz")["frame"]) == [6]
    row = pd.read_parquet(tmp_path / "t.parquet").iloc[0]
    assert (row.x1, row.y1, row.x2, row.y2, row.foot_y, row.area) == (20, 10, 40, 60, 60, 1000)
    saved = json.loads((tmp_path / "m.jsonl").read_text())
    assert mask_utils.area({"size": saved["size"], "counts": saved["counts"]}) == 1000


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


def test_sam_session_start_survives_the_extra_argument():
    class Model:  # SAM 3.1's init_state: no offload_state_to_cpu parameter
        def init_state(self, resource_path, offload_video_to_cpu=False, async_loading_frames=False):
            return {"path": resource_path, "offload_video": offload_video_to_cpu}

    model = Model()
    drop_offload_state_arg(model)
    # what sam3's shared start_session sends
    state = model.init_state(
        resource_path="frames/", offload_video_to_cpu=True, offload_state_to_cpu=False, async_loading_frames=True
    )
    assert state == {"path": "frames/", "offload_video": True}


class CrashingSam:
    """Stand-in for SAM 3.1's predictor: session k crashes on the frame in crash_at[k] (None: no crash)."""

    def __init__(self, crash_at, n_frames=10, error="No points are provided; please add points first"):
        self.crash_at, self.n_frames, self.error = crash_at, n_frames, error
        self.prompts, self.closed, self.sessions = [], [], 0

    def handle_request(self, request):
        if request["type"] == "start_session":
            self.sessions += 1
            return {"session_id": self.sessions - 1}
        if request["type"] == "add_prompt":
            self.prompts.append(request["frame_index"])
        if request["type"] == "close_session":
            self.closed.append(request["session_id"])
        return {}

    def handle_stream_request(self, request):
        crash = self.crash_at[request["session_id"]]
        for frame in range(request["start_frame_index"], self.n_frames):
            if frame == crash:
                raise RuntimeError(self.error)
            # SAM's id 0 stays on one person; id 1 is someone else after a restart (no overlap)
            masks = np.zeros((2, 8, 8), bool)
            masks[0, :4, :4] = True
            masks[1, 4:, 4:] = True
            if request["session_id"] > 0:
                masks[1] = np.roll(masks[1], -4, axis=0)
            yield {"frame_index": frame, "outputs": {"out_obj_ids": np.array([0, 1]), "out_binary_masks": masks}}


def _sam(predictor):
    tracker = Sam31.__new__(Sam31)  # no weights or GPU: only the session logic is under test
    tracker.predictor = predictor
    clip = SimpleNamespace(name="SNGS-000", frames=[SimpleNamespace(path=Path("frames/000001.jpg"))] * 10)
    return tracker, clip


def test_sam_crash_restarts_from_the_crashed_frame_and_keeps_ids_that_overlap():
    predictor = CrashingSam(crash_at=[4, 4, None])  # the second session crashes at once: frame 5 is skipped
    tracker, clip = _sam(predictor)
    out = list(tracker.track(clip))
    assert [f for f, _ in out] == [1, 2, 3, 4, 6, 7, 8, 9, 10]  # GSR counts from 1
    assert list(out[3][1]["out_obj_ids"]) == [0, 1]
    # the same person keeps id 0; the new one gets a fresh id; and the map holds for the session
    assert list(out[4][1]["out_obj_ids"]) == [0, 2 * RESTART_ID_STEP + 1]
    assert list(out[-1][1]["out_obj_ids"]) == [0, 2 * RESTART_ID_STEP + 1]
    assert predictor.prompts == [0, 4, 5]
    assert predictor.closed == [0, 1, 2]


def test_sam_other_errors_are_not_swallowed():
    tracker, clip = _sam(CrashingSam(crash_at=[3], error="CUDA out of memory"))
    with pytest.raises(RuntimeError, match="out of memory"):
        list(tracker.track(clip))


def test_match_masks_is_one_to_one_and_needs_overlap():
    prev = np.zeros((2, 10, 10), bool)
    prev[0, :5, :5] = True
    prev[1, 5:, 5:] = True
    new = np.zeros((3, 10, 10), bool)
    new[0, 5:, 5:] = True  # person 1
    new[1, :5, :4] = True  # person 0, a little smaller (IoU 0.8)
    new[2, :5, :5] = True  # also person 0, exactly: it wins
    assert match_masks([7, 9], rles(prev), [20, 21, 22], rles(new)) == {20: 9, 22: 7}
    assert match_masks([7], rles(prev[:1]), [], []) == {}
