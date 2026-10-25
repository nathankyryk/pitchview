import numpy as np
import pandas as pd
import pytest

from pitchview.analytics import player_stats, possession, summarise, team_shape


def _row(frame, track, x, y, team="left", role="player", jersey="7"):
    return {"frame": frame, "track_id": track, "role": role, "team": team, "jersey": jersey, "x_m": x, "y_m": y}


def test_player_stats_constant_speed():
    # 5 m/s along x for 3 s at 25 fps, plus one teleport that must not count as running
    rows = [_row(f, 1, 0.2 * f, 0.0) for f in range(75)]
    rows[40]["x_m"] += 30.0
    stats = player_stats(pd.DataFrame(rows)).iloc[0]
    assert stats["distance_m"] == pytest.approx(0.2 * 74, rel=0.15)
    assert stats["top_speed_kmh"] == pytest.approx(18.0, rel=0.1)


def test_possession_needs_a_sustained_change():
    ball = pd.DataFrame({"frame": range(30), "x_m": 0.0, "y_m": 0.0})
    rows = []
    for f in range(30):
        # left player sits on the ball; a right player brushes past on frames 10-11,
        # then wins it for good from frame 20
        rows.append(_row(f, 1, 0.5 if f < 20 else 5.0, 0.0, team="left"))
        near = f in (10, 11) or f >= 20
        rows.append(_row(f, 2, 0.1 if near else 8.0, 0.0, team="right"))
    held = possession(pd.DataFrame(rows), ball)
    assert held[5] == "left"
    assert held[11] == "left"  # two frames of contact is not a change of possession
    assert held[23] == "left" and held[24] == "right"  # the 5th straight frame flips it


def test_team_shape_and_defensive_line():
    xs = [-40, -38, -36, -34, -20, -10, 0, 5]
    ys = [-20, -10, 10, 20, -15, 0, 15, 5]
    state = pd.DataFrame([_row(1, i, x, y) for i, (x, y) in enumerate(zip(xs, ys))])
    shape = team_shape(state).iloc[0]
    assert shape["players"] == 8
    assert shape["line_height_m"] == pytest.approx(np.mean([12.5, 14.5, 16.5, 18.5]))  # from x = -52.5
    assert shape["width_m"] == pytest.approx(np.percentile(ys, 90) - np.percentile(ys, 10))
    assert team_shape(state.iloc[:5]).empty  # too few visible players


def test_summarise_is_json_ready():
    import json

    rows = [_row(f, t, 0.1 * f + t, t, team="left" if t < 7 else "right") for f in range(150) for t in range(14)]
    ball = pd.DataFrame({"frame": range(150), "x_m": [0.1 * f for f in range(150)], "y_m": 0.0})
    summary = summarise(pd.DataFrame(rows), ball)
    json.dumps(summary)
    assert summary["players_tracked"] == 14
    assert summary["possession_pct"]["left"] == 100.0
    assert set(summary["shape"]) == {"left", "right"}
