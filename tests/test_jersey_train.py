import torch

from pitchview.cli import _jersey_dir
from pitchview.jersey_train import agrees, augment


def test_agrees_means_zero_shot_saw_part_of_the_number():
    assert agrees("14", "14")
    assert agrees("4", "14")  # half of a two-digit number
    assert agrees("41", "14")
    assert not agrees("7", "14")
    assert not agrees(None, "14")


def test_augment_keeps_shape_and_range():
    g = torch.Generator().manual_seed(0)
    x = torch.rand((16, 3, 32, 128), generator=g)
    y = augment(x.clone(), generator=g)
    assert y.shape == x.shape
    assert float(y.min()) >= 0 and float(y.max()) <= 1
    assert not torch.equal(x, y)


def test_reader_gets_its_own_folders(tmp_path):
    assert _jersey_dir(tmp_path, "jersey_reads", "sam", "model", 10).name == "jersey_reads_sam_model"
    assert (
        _jersey_dir(tmp_path, "jerseys", "sam", "model", 10, reader="gsr_all_e2").name == "jerseys_sam_model_gsr_all_e2"
    )
    assert _jersey_dir(tmp_path, "jersey_reads", "bytetrack", "oracle", 9, reader="zeroshot").name == (
        "jersey_reads_bytetrack_oracle_s9"
    )
