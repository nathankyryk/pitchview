"""Match analytics from a clip's game state: what the demo's stats panel shows.

Works on any game state with columns frame, track_id, role, team, jersey, x_m, y_m
(our pipeline's output, or GSR ground truth). Broadcast video shows part of the pitch,
so team-shape numbers use visible players only and need a minimum head count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_SPEED = 12.0  # m/s; faster steps are tracking or calibration noise (sprinters peak near 10)
TEAMS = ("left", "right")


def _smooth_track(track: pd.DataFrame, max_gap: int = 5, window: int = 5) -> pd.DataFrame:
    """One track on a contiguous frame axis: short gaps filled in, then a rolling mean."""
    track = track.sort_values("frame").set_index("frame")[["x_m", "y_m"]]
    frames = np.arange(track.index.min(), track.index.max() + 1)
    track = track.reindex(frames).interpolate(limit=max_gap, limit_area="inside")
    return track.rolling(window, center=True, min_periods=1).mean()


def player_stats(state: pd.DataFrame, fps: float = 25.0) -> pd.DataFrame:
    """Distance covered (m) and top speed (km/h, best 1-second average) per track."""
    rows = []
    for track_id, track in state.groupby("track_id"):
        pos = _smooth_track(track)
        steps = np.hypot(pos["x_m"].diff(), pos["y_m"].diff())
        steps[steps * fps > MAX_SPEED] = np.nan  # a jump, not running
        speed = (steps * fps).rolling(int(fps), min_periods=int(fps) // 2).mean()
        first = track.iloc[0]
        rows.append(
            {
                "track_id": track_id,
                "role": first["role"],
                "team": first["team"],
                "jersey": first["jersey"],
                "seconds": len(track) / fps,
                "distance_m": float(np.nansum(steps)),
                "top_speed_kmh": float(np.nanmax(speed) * 3.6) if speed.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def possession(state: pd.DataFrame, ball: pd.DataFrame, radius: float = 2.0, min_frames: int = 5) -> pd.Series:
    """Team in possession per frame: the nearest player within `radius` m of the ball.

    Possession only changes after the other team is nearest for `min_frames` frames in a row,
    so a ball rolling past an opponent doesn't flip it. While nobody is close (a pass in
    flight) the last team keeps possession. Frames before anyone has the ball are None.
    """
    players = state[state["role"].isin(["player", "goalkeeper"]) & state["team"].isin(TEAMS)]
    ball_at = ball.set_index("frame")[["x_m", "y_m"]]
    by_frame = {f: g for f, g in players.groupby("frame")}
    current, challenger, streak = None, None, 0
    out = {}
    for frame in range(int(ball_at.index.min()), int(ball_at.index.max()) + 1) if len(ball_at) else []:
        nearest = None
        if frame in ball_at.index and frame in by_frame:
            g = by_frame[frame]
            d = np.hypot(g["x_m"] - ball_at.at[frame, "x_m"], g["y_m"] - ball_at.at[frame, "y_m"])
            if d.min() <= radius:
                nearest = g["team"].iloc[int(np.argmin(d.to_numpy()))]
        if nearest is not None and nearest != current:
            streak = streak + 1 if nearest == challenger else 1
            challenger = nearest
            if current is None or streak >= min_frames:
                current, streak = nearest, 0
        elif nearest == current:
            streak = 0
        out[frame] = current
    return pd.Series(out, name="team", dtype=object)


def team_shape(state: pd.DataFrame, min_players: int = 6) -> pd.DataFrame:
    """Per frame and team, from visible outfield players: centroid, width, depth, and how
    far the defensive line (mean of the 4 deepest players) stands from its own goal (m)."""
    outfield = state[(state["role"] == "player") & state["team"].isin(TEAMS)]
    rows = []
    for (frame, team), g in outfield.groupby(["frame", "team"]):
        if len(g) < min_players:
            continue
        x, y = g["x_m"].to_numpy(), g["y_m"].to_numpy()
        own_goal = -52.5 if team == "left" else 52.5
        deepest = np.sort(np.abs(x - own_goal))[:4]
        rows.append(
            {
                "frame": frame,
                "team": team,
                "centroid_x": float(x.mean()),
                "centroid_y": float(y.mean()),
                "width_m": float(np.percentile(y, 90) - np.percentile(y, 10)),
                "depth_m": float(np.percentile(x, 90) - np.percentile(x, 10)),
                "line_height_m": float(deepest.mean()),
                "players": len(g),
            }
        )
    return pd.DataFrame(rows)


def summarise(state: pd.DataFrame, ball: pd.DataFrame | None, fps: float = 25.0) -> dict:
    """The clip's headline numbers, JSON-ready."""
    stats = player_stats(state[state["role"] == "player"], fps)
    long_tracks = stats[stats["seconds"] >= 5]  # top speeds from short fragments are unreliable
    summary: dict = {"players_tracked": len(stats)}
    if not long_tracks.empty:
        best = long_tracks.sort_values("top_speed_kmh", ascending=False).iloc[0]
        summary["fastest"] = {
            "team": best["team"], "jersey": best["jersey"], "top_speed_kmh": round(best["top_speed_kmh"], 1)
        }
        summary["median_top_speed_kmh"] = round(float(long_tracks["top_speed_kmh"].median()), 1)
    if ball is not None and len(ball):
        held = possession(state, ball).dropna()
        if len(held):
            share = held.value_counts(normalize=True)
            summary["possession_pct"] = {team: round(float(share.get(team, 0.0)) * 100, 1) for team in TEAMS}
    shape = team_shape(state)
    if not shape.empty:
        means = shape.groupby("team")[["width_m", "depth_m", "line_height_m"]].mean().round(1)
        summary["shape"] = {team: means.loc[team].to_dict() for team in means.index}
    return summary
