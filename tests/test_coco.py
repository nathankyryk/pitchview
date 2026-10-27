import json

from pitchview.coco import CATEGORIES, convert_split


def test_convert_every_frame(gsr_root, tmp_path):
    out = tmp_path / "coco" / "valid"
    stats = convert_split(gsr_root, "valid", out, stride=1)

    # frame 3 is skipped for its ignore region
    assert stats["frames"] == 5
    assert stats["frames_skipped_ignore_region"] == 1
    assert stats["boxes_dropped_other"] == 5
    assert stats["boxes_dropped_degenerate"] == 5
    assert (stats["boxes_player"], stats["boxes_ball"], stats["boxes_referee"]) == (5, 5, 5)

    coco = json.loads((out / "_annotations.coco.json").read_text())
    assert coco["categories"] == CATEGORIES
    assert {im["file_name"] for im in coco["images"]} == {f"SNGS-999_{i:06d}.jpg" for i in (1, 2, 4, 5, 6)}
    assert all((out / im["file_name"]).is_file() for im in coco["images"])

    # the referee box is clipped to the 64x36 frame
    referee = next(a for a in coco["annotations"] if a["category_id"] == 3)
    assert referee["bbox"] == [60.0, 30.0, 4.0, 6.0]
    assert referee["area"] == 24.0


def test_stride_keeps_every_nth_frame(gsr_root, tmp_path):
    stats = convert_split(gsr_root, "valid", tmp_path / "coco", stride=2)
    # frames 1, 3, 5 are selected; 3 is dropped for its ignore region
    assert stats["frames"] == 2
