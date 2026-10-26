"""Official GS-HOTA scoring via SoccerNet's sn-trackeval, plus the prediction file format.

GS-HOTA matches a prediction to a ground-truth person only when both are within ~5 m on
the pitch (Gaussian similarity) AND role, team and jersey number all agree exactly.
Switching those checks off one at a time shows which component costs the most.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def jersey_truth(tracks: pd.DataFrame, clip: Clip, min_iou: float = 0.5) -> pd.DataFrame:
    """The true number behind each predicted track that follows a player, for scoring jersey reads.

    Per frame, predicted and true boxes are matched one-to-one (IoU >= min_iou), and a track
    follows the person it matched most often. One row per such track: track_id, gt_jersey
    (None when the number is never visible) and frames (how often it matched that player).
    """
    from scipy.optimize import linear_sum_assignment

    from pitchview.segment import box_iou

    by_frame = clip.boxes_by_frame()
    people: dict = {}
    hits = []
    for frame, rows in tracks.groupby("frame"):
        gt = [b for b in by_frame.get(int(frame), []) if b.role != "ball" and b.track_id is not None]
        if not gt:
            continue
        iou = box_iou(
            rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float),
            np.array([[b.x, b.y, b.x + b.w, b.y + b.h] for b in gt]),
        )
        for i, j in zip(*linear_sum_assignment(-iou)):
            if iou[i, j] >= min_iou:
                hits.append((int(rows["track_id"].iloc[i]), gt[j].track_id))
                people[gt[j].track_id] = gt[j]
    if not hits:
        return pd.DataFrame({"track_id": [], "gt_jersey": [], "frames": []})
    counts = pd.DataFrame(hits, columns=["track_id", "gt_track"]).value_counts().rename("frames").reset_index()
    best = counts.sort_values("frames", ascending=False, kind="stable").drop_duplicates("track_id")
    best = best[[people[t].role == "player" for t in best["gt_track"]]]
    return pd.DataFrame(
        {
            "track_id": best["track_id"].to_numpy(),
            "gt_jersey": pd.Series([people[t].jersey for t in best["gt_track"]], dtype=object),  # keep None
            "frames": best["frames"].to_numpy(),
        }
    )


def jersey_summary(predicted: list, truth: list, frames) -> dict[str, float]:
    """Scores for per-track numbers (None = no number) against the truth.

    agree:     share of player frames whose number is exactly right, "no number" included.
               This is what GS-HOTA rewards: a wrong number and a missing one both break the match.
    read:      share of tracks given a number.
    precision: share of those numbers that are right.
    accuracy:  share of numbered players read right (the zero-shot PARSeq baseline was 40%).
    """
    frames = np.asarray(frames, dtype=float)
    predicted = [None if pd.isna(p) else p for p in predicted]  # pandas may hand back NaN for None
    truth = [None if pd.isna(t) else t for t in truth]
    right = np.array([p == t for p, t in zip(predicted, truth)], dtype=bool)
    read = np.array([p is not None for p in predicted], dtype=bool)
    numbered = np.array([t is not None for t in truth], dtype=bool)
    return {
        "agree": float(frames[right].sum() / max(frames.sum(), 1)),
        "read": float(read.mean()) if len(read) else 0.0,
        "precision": float((right & read).sum() / max(read.sum(), 1)),
        "accuracy": float((right & numbered).sum() / max(numbered.sum(), 1)),
    }


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
