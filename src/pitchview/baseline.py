"""Score SoccerNet's official GSR baseline with our own evaluator, from its published predictions.

The baseline's predictions ("tracker state", Zenodo record 11065177) are a zip of pickled pandas
tables, one per video. Pickles can run code when loaded, so they are read with an allow-list that
only accepts pandas and numpy objects. The conversion mirrors Tracklab's own save_for_eval
(tracklab/wrappers/dataset/soccernet/soccernet_game_state.py): detections without a track id,
image box or pitch position are dropped, and role, team and jersey become the attributes.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path

import pandas as pd

TEST_URL = "https://zenodo.org/records/11065177/files/gamestate-prtreid-strongsort-test-compressed.pklz?download=1"

_ALLOWED_PACKAGES = {"pandas", "numpy", "collections", "datetime", "copyreg"}
_ALLOWED_BUILTINS = {"slice", "frozenset", "set", "complex", "range", "bytearray", "object",
                     "list", "dict", "tuple", "int", "float", "str", "bool"}  # fmt: skip


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module == "builtins" and name in _ALLOWED_BUILTINS) or module.split(".")[0] in _ALLOWED_PACKAGES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def load_table(archive: zipfile.ZipFile, entry: str) -> pd.DataFrame:
    return _DataOnlyUnpickler(io.BytesIO(archive.read(entry))).load()


def _value(v):
    """None for a missing value, otherwise the value."""
    return None if v is None or (not isinstance(v, (dict, list)) and pd.isna(v)) else v


def _jersey(v) -> str | None:
    """A digit string, as in the labels: a float 33.0 would never match "33"."""
    v = _value(v)
    if v is None:
        return None
    text = str(v)
    try:
        return str(int(float(text)))
    except ValueError:
        return None


def predictions(detections: pd.DataFrame) -> list[dict]:
    """One video's baseline detections in the GSR prediction format."""
    kept = detections.dropna(subset=["track_id", "bbox_ltwh", "bbox_pitch"])
    out = []
    for row in kept.itertuples(index=False):
        out.append(
            {
                "image_id": str(row.image_id),
                "track_id": int(row.track_id),
                "supercategory": "object",
                "category_id": int(row.category_id) if _value(row.category_id) is not None else None,
                "attributes": {"role": _value(row.role), "team": _value(row.team), "jersey": _jersey(row.jersey_number)},
                "bbox_pitch": {k: float(v) for k, v in row.bbox_pitch.items()},
            }
        )
    return out


def video_predictions(archive_path: Path, clip_name: str) -> list[dict]:
    """Predictions for one GSR clip ("SNGS-116" is the archive's video "116")."""
    with zipfile.ZipFile(archive_path) as archive:
        return predictions(load_table(archive, f"{int(clip_name.split('-')[1])}.pkl"))
