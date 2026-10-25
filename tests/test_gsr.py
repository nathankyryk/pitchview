import pytest

from pitchview.gsr import list_clips


def test_list_clips_reads_frames_and_boxes(gsr_root):
    (clip,) = list_clips("valid", gsr_root)
    assert clip.name == "SNGS-999"
    assert clip.fps == 25
    assert [f.index for f in clip.frames] == [1, 2, 3, 4, 5, 6]
    assert [f.has_ignore_regions for f in clip.frames] == [False, False, True, False, False, False]
    assert clip.frames[0].path.name == "000001.jpg"

    # 5 objects per frame; the pitch annotation is skipped
    assert len(clip.boxes) == 30
    first = clip.boxes_by_frame()[1]
    player = next(b for b in first if b.role == "player" and b.w > 1)
    assert (player.team, player.jersey, player.pitch_x, player.pitch_y) == ("left", "7", -3.5, 12.0)
    assert {b.role for b in first} == {"player", "ball", "other", "referee"}


def test_missing_split_raises(gsr_root):
    with pytest.raises(FileNotFoundError):
        list_clips("train", gsr_root)
