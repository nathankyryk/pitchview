from pitchview.cli import _no_nan
from pitchview.viewer import build, clip_label, score_card, stats_card, video_path

MANIFEST = {
    "overall": {"split": "test", "clips": 49, "gs_hota": 28.3, "baseline": 23.1},
    "tabs": {"full": "main"},
    "clips": {
        "SNGS-118": {
            "game": "7",
            "scores": {"full": {"gs_hota": 66.24, "baseline": 53.64}},
            "stats": {
                "possession_pct": {"left": 61.2, "right": 38.8},
                "fastest": {"team": "right", "jersey": "10", "top_speed_kmh": 27.34},
                "median_top_speed_kmh": 19.0,
                "shape": {"left": {"width_m": 40.2, "depth_m": 30.1, "line_height_m": 20.0}},
            },
        },
        "SNGS-121": {
            "game": "7",
            "scores": {},
            "stats": {"fastest": {"team": None, "jersey": None, "top_speed_kmh": 20.0}},
        },
    },
}


def test_cards_show_the_clip_scores_and_stats():
    assert "GS-HOTA 66.2" in score_card(MANIFEST, "SNGS-118", "full")
    assert "Official baseline: 53.6" in score_card(MANIFEST, "SNGS-118", "full")
    assert score_card(MANIFEST, "SNGS-121", "full") == ""  # no scores: no card
    stats = stats_card(MANIFEST, "SNGS-118")
    assert "Left team 61%" in stats and "#10 (right team), 27.3 km/h" in stats and "40 m wide" in stats
    assert "a player (unknown team)" in stats_card(MANIFEST, "SNGS-121")
    assert clip_label(MANIFEST, "SNGS-118") == "SNGS-118 (game 7)"


def test_app_builds_from_a_bundle(tmp_path):
    import json

    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    (tmp_path / "videos" / "full").mkdir(parents=True)
    (tmp_path / "videos" / "full" / "SNGS-118.mp4").write_bytes(b"")
    assert video_path(tmp_path, "full", "SNGS-118").endswith("SNGS-118.mp4")
    assert video_path(tmp_path, "full", "SNGS-121") is None
    build(tmp_path)  # constructs the Blocks without launching


def test_no_nan_makes_stats_json_safe():
    assert _no_nan({"a": [float("nan"), 1.5], "b": {"c": float("nan")}}) == {"a": [None, 1.5], "b": {"c": None}}
