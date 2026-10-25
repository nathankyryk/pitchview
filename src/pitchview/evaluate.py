"""Official GS-HOTA scoring via SoccerNet's sn-trackeval, plus the prediction file format.

GS-HOTA matches a prediction to a ground-truth person only when both are within ~5 m on
the pitch (Gaussian similarity) AND role, team and jersey number all agree exactly.
Switching those checks off one at a time shows which component costs the most.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pitchview.gsr import Clip

BENCHMARK = "SoccerNetGS"


def prediction(
    image_id: str,
    track_id: int,
    role: str,
    x_m: float,
    y_m: float,
    team: str | None = None,
    jersey: str | None = None,
    confidence: float = 1.0,
) -> dict:
    """One prediction in GSR's format. Only the bottom-middle point is used for matching."""
    return {
        "image_id": image_id,
        "track_id": int(track_id),
        "supercategory": "object",
        "category_id": {"player": 1, "goalkeeper": 2, "referee": 3, "ball": 4}.get(role, 7),
        "attributes": {"role": role, "team": team, "jersey": jersey},
        "bbox_pitch": {
            "x_bottom_left": x_m,
            "y_bottom_left": y_m,
            "x_bottom_middle": x_m,
            "y_bottom_middle": y_m,
            "x_bottom_right": x_m,
            "y_bottom_right": y_m,
        },
        "confidence": float(confidence),
    }


def image_ids(clip: Clip) -> dict[int, str]:
    """GSR frame index -> the image_id string the evaluator keys on."""
    return {int(Path(im["file_name"]).stem): im["image_id"] for im in clip.labels["images"]}


def write_predictions(trackers_root: Path, split: str, tracker: str, clip_name: str, predictions: list[dict]) -> Path:
    path = trackers_root / f"{BENCHMARK}-{split}" / tracker / "data" / f"{clip_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"predictions": predictions}))
    return path


def gs_hota(
    gt_root: Path,
    trackers_root: Path,
    split: str,
    tracker: str,
    clips: list[str],
    use_jersey: bool = True,
    use_teams: bool = True,
    use_roles: bool = True,
) -> dict[str, float]:
    """Run the official evaluator. Returns GS-HOTA with its detection (DetA) and association
    (AssA) parts, in percent, over the given clips."""
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
         "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "DISPLAY_LESS_PROGRESS": True}
    )  # fmt: skip
    dataset_config = trackeval.datasets.SoccerNetGS.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(trackers_root),
            "SPLIT_TO_EVAL": split,
            "TRACKERS_TO_EVAL": [tracker],
            "SEQ_INFO": {name: None for name in clips},
            "USE_JERSEY_NUMBERS": use_jersey,
            "USE_TEAMS": use_teams,
            "USE_ROLES": use_roles,
            "PRINT_CONFIG": False,
        }
    )
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.SoccerNetGS(dataset_config)
    results, _ = evaluator.evaluate([dataset], [trackeval.metrics.HOTA({"THRESHOLD": 0.5})])
    hota = results[BENCHMARK][tracker]["COMBINED_SEQ"]["person"]["HOTA"]
    return {k: float(np.mean(hota[k]) * 100) for k in ("HOTA", "DetA", "AssA")}
