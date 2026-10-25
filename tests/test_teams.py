import numpy as np

from pitchview.teams import GREY, cluster_teams, jersey_crop, name_sides


def test_jersey_crop_takes_shirt_band_and_greys_background():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = (0, 200, 0)  # grass
    frame[20:80, 40:60] = (255, 0, 0)  # red player
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 40:60] = True
    box = (30, 20, 70, 80)  # wider than the player, so grass gets in

    plain = np.asarray(jersey_crop(frame, box))
    masked = np.asarray(jersey_crop(frame, box, mask))
    assert plain.shape == masked.shape == (27, 40, 3)  # rows 29..56 of a 60 px tall box
    assert (plain == (0, 200, 0)).all(axis=-1).any()  # the box crop contains grass
    assert not (masked == (0, 200, 0)).all(axis=-1).any()  # the masked crop does not
    assert (masked == GREY).all(axis=-1).any()


def test_cluster_teams_separates_two_kits_and_votes_per_track():
    rng = np.random.default_rng(0)
    red = rng.normal(0.0, 0.05, size=(30, 16)) + np.eye(16)[0]
    blue = rng.normal(0.0, 0.05, size=(30, 16)) + np.eye(16)[1]
    embeddings = np.vstack([red, blue])
    track_ids = np.array([1] * 10 + [2] * 10 + [3] * 10 + [4] * 10 + [5] * 10 + [6] * 10)
    teams = cluster_teams(embeddings, track_ids)
    assert teams[1] == teams[2] == teams[3]
    assert teams[4] == teams[5] == teams[6]
    assert teams[1] != teams[4]


def test_name_sides_uses_pitch_position():
    clusters = {1: 0, 2: 0, 3: 1, 4: 1}
    mean_x = {1: 20.0, 2: 30.0, 3: -25.0, 4: -10.0}  # cluster 1 stands on the left (x < 0)
    assert name_sides(clusters, mean_x) == {1: "right", 2: "right", 3: "left", 4: "left"}
