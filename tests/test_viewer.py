from pitchview.cli import _no_nan
from pitchview.viewer import build, clip_label, score_card, stats_card, video_path

MANIFEST = {
    "overall": {"split": "test", "clips": 49, "gs_hota": 28.3, "baseline": 23.1},
    "tabs": {"full": "main"},
    "clips": {
        "SNGS-118": {
            "game": "7",
            "seconds": 30.0,
            "teams": {"left": "FC St. Gallen", "right": "FC Thun"},
            "scores": {"full": {"gs_hota": 66.24, "baseline": 53.64}},
            "stats": {
                "possession_pct": {"left": 61.2, "right": 38.8},
                "fastest": {"team": "right", "jersey": "10", "top_speed_kmh": 27.34, "name": "B. Stillhart"},
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
    assert "### Stats for this clip" in stats and "this clip's 30 seconds only, not the whole match" in stats
    assert "FC St. Gallen 61%" in stats and "B. Stillhart (#10, FC Thun), 27.3 km/h" in stats
    assert "FC St. Gallen 40 m wide" in stats
    other = stats_card(MANIFEST, "SNGS-121")
    assert "a player (unknown team)" in other and "From this clip only" in other
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


def test_compare_tab_appears_with_two_tabs(tmp_path):
    import json

    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    for tab in ("foundation", "full"):
        (tmp_path / "videos" / tab).mkdir(parents=True)
        (tmp_path / "videos" / tab / "SNGS-118.mp4").write_bytes(b"")
    app = build(tmp_path)
    labels = {c.get("props", {}).get("label") for c in app.get_config_file()["components"]}
    assert {"Compare", "Foundation", "Full system", "Left", "Right"} <= labels


def test_space_folder_has_app_readme_viewer_and_bundle(tmp_path):
    import json

    from pitchview.viewer import CREDITS, make_space

    bundle = tmp_path / "bundle"
    (bundle / "videos" / "full").mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps(MANIFEST))
    (bundle / "videos" / "full" / "SNGS-118.mp4").write_bytes(b"0" * 10)
    out = tmp_path / "space"
    size = make_space(bundle, out, "6.30.0")
    assert (out / "demo" / "videos" / "full" / "SNGS-118.mp4").exists()
    assert "from viewer import build" in (out / "app.py").read_text()
    assert "def make_space" in (out / "viewer.py").read_text(encoding="utf-8")
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\ntitle: PitchView") and "sdk_version: 6.30.0" in readme
    assert "SAM 3.1 (Meta, SAM License)" in CREDITS and CREDITS in readme
    assert size > 10
    try:
        make_space(tmp_path / "missing", out, "6.30.0")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a folder without manifest.json should be refused")
