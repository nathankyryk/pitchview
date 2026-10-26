"""PitchView's demo viewer: pre-rendered clips with their scores and match stats.

It reads a bundle folder made by `pitchview bundle` and runs no model, so it can be served
from a free Hugging Face Space:

    <bundle>/manifest.json            tabs, clips, scores and stats
    <bundle>/videos/<tab>/<clip>.mp4  one rendered video per tab and clip

Only gradio and the standard library are used here, so the Space needs nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

# Every tab the viewer knows, in display order: (title, what it shows). A tab appears once the
# bundle has its videos. Only "full" is built so far.
TABS = {
    "full": (
        "Full system",
        (
            "Fine-tuned RF-DETR finds players, goalkeepers, referees and the ball. SAM 3.1 masks are matched "
            "to its boxes, ByteTrack follows each person, teams come from kit colours, jersey numbers are "
            "read and voted per track, and the fine-tuned pitch model places everyone on the radar."
        ),
    ),
}
TEAM_NAMES = {"left": "Left team", "right": "Right team"}


def load_manifest(bundle: Path) -> dict:
    return json.loads((Path(bundle) / "manifest.json").read_text())


def video_path(bundle: Path, tab: str, clip: str) -> str | None:
    path = Path(bundle) / "videos" / tab / f"{clip}.mp4"
    return str(path) if path.is_file() else None


def clip_label(manifest: dict, clip: str) -> str:
    game = manifest["clips"][clip].get("game")
    return f"{clip} (game {game})" if game else clip


def header(manifest: dict) -> str:
    o = manifest["overall"]
    return (
        "# PitchView\n"
        "Broadcast soccer video in, game state out: every player's team, role, jersey number and position "
        "on the pitch, frame by frame.\n\n"
        f"On the SoccerNet GSR {o['split']} split ({o['clips']} clips), PitchView scores **{o['gs_hota']:.1f} "
        f"GS-HOTA**, against {o['baseline']:.1f} for the official baseline, scored with the same evaluator."
    )


def score_card(manifest: dict, clip: str, tab: str) -> str:
    s = manifest["clips"][clip].get("scores", {}).get(tab)
    if not s:
        return ""
    lines = [f"### This clip\n**GS-HOTA {s['gs_hota']:.1f}**"]
    if s.get("baseline") is not None:
        lines.append(f"Official baseline: {s['baseline']:.1f}")
    return "\n\n".join(lines)


def stats_card(manifest: dict, clip: str) -> str:
    st = manifest["clips"][clip].get("stats") or {}
    lines = ["### Match stats"]
    if share := st.get("possession_pct"):
        lines.append("Possession: " + ", ".join(f"{TEAM_NAMES.get(t, t)} {p:.0f}%" for t, p in share.items()))
    if fast := st.get("fastest"):
        who = f"#{fast['jersey']}" if fast.get("jersey") else "a player"
        lines.append(f"Fastest: {who} ({TEAM_NAMES.get(fast.get('team'), 'unknown team').lower()}), "
                     f"{fast['top_speed_kmh']:.1f} km/h")  # fmt: skip
    if "median_top_speed_kmh" in st:
        lines.append(f"Median top speed: {st['median_top_speed_kmh']:.1f} km/h")
    for team, shape in (st.get("shape") or {}).items():
        lines.append(
            f"{TEAM_NAMES.get(team, team)} shape: {shape['width_m']:.0f} m wide, {shape['depth_m']:.0f} m deep"
        )
    lines.append("Positions are the players in view, so team shape covers only what the camera shows.")
    return "\n\n".join(lines)


def build(bundle: Path):
    """The Gradio app for a bundle. Call .launch() on the result."""
    import gradio as gr

    bundle = Path(bundle)
    manifest = load_manifest(bundle)
    clips = list(manifest["clips"])
    tabs = [t for t in TABS if any(video_path(bundle, t, c) for c in clips)]
    first = clips[0]

    with gr.Blocks(title="PitchView") as app:
        gr.Markdown(header(manifest))
        clip = gr.Dropdown(choices=[(clip_label(manifest, c), c) for c in clips], value=first, label="Clip")
        widgets = []
        for tab in tabs:
            title, blurb = TABS[tab]
            with gr.Tab(title):
                gr.Markdown(blurb)
                with gr.Row():
                    video = gr.Video(value=video_path(bundle, tab, first), show_label=False, scale=3)
                    with gr.Column(scale=1):
                        score = gr.Markdown(score_card(manifest, first, tab))
                        stats = gr.Markdown(stats_card(manifest, first))
            widgets.append((tab, video, score, stats))

        def show(name: str) -> list:
            out = []
            for tab, *_ in widgets:
                out += [video_path(bundle, tab, name), score_card(manifest, name, tab), stats_card(manifest, name)]
            return out

        clip.change(show, clip, [w for _, *ws in widgets for w in ws])
    return app
