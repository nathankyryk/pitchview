import os
import pickle
import zipfile

import numpy as np
import pandas as pd
import pytest

from pitchview.baseline import load_table, predictions


class _Exploit:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def _archive(path, obj):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("1.pkl", pickle.dumps(obj))
    return zipfile.ZipFile(path)


def test_load_table_reads_dataframes_and_refuses_code(tmp_path):
    # Object strings, as in the 2024 archive; modern pandas would store strings via pyarrow, which stays blocked.
    df = pd.DataFrame({"a": [1.0, np.nan], "b": pd.Series(["x", None], dtype=object)})
    df.columns = pd.Index(["a", "b"], dtype=object)
    pd.testing.assert_frame_equal(load_table(_archive(tmp_path / "ok.pklz", df), "1.pkl"), df)
    with pytest.raises(pickle.UnpicklingError, match="refusing to load"):
        load_table(_archive(tmp_path / "bad.pklz", _Exploit()), "1.pkl")


def test_predictions_follow_tracklab_rules():
    pitch = {k: 1.0 for k in ("x_bottom_left", "y_bottom_left", "x_bottom_middle", "y_bottom_middle",
                              "x_bottom_right", "y_bottom_right")}  # fmt: skip
    dets = pd.DataFrame(
        {
            "image_id": [3116000001, 3116000001, 3116000002, 3116000002],
            "track_id": [1.0, np.nan, 2.0, 3.0],  # an untracked detection is dropped
            "bbox_ltwh": [[0, 0, 1, 1]] * 4,
            "bbox_pitch": [pitch, pitch, pitch, None],  # so is one without a pitch position
            "category_id": [1.0, 1.0, 1.0, 1.0],
            "role": ["player", "player", "referee", "player"],
            "team": ["left", "left", np.nan, "right"],
            "jersey_number": [33.0, 7, np.nan, "10"],
        }
    )
    preds = predictions(dets)
    assert [p["track_id"] for p in preds] == [1, 2]
    assert preds[0]["image_id"] == "3116000001"
    assert preds[0]["attributes"] == {"role": "player", "team": "left", "jersey": "33"}  # 33.0 would never match "33"
    assert preds[1]["attributes"] == {"role": "referee", "team": None, "jersey": None}
    assert preds[0]["bbox_pitch"]["x_bottom_middle"] == 1.0
