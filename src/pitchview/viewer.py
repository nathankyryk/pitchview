"""PitchView's demo viewer: pre-rendered clips with their scores and match stats.

It reads a bundle folder made by `pitchview bundle` and runs no model, so it can be served
from a free Hugging Face Space:

    <bundle>/manifest.json            tabs, clips, scores and stats
    <bundle>/videos/<tab>/<clip>.mp4  one rendered video per tab and clip

Only gradio and the standard library are used here, so the Space needs nothing else.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# Every tab the viewer knows, in display order: (title, what it shows). A tab appears once the
# bundle has its videos; the viewer opens on "full".
TABS = {
    "truth": (
        "Ground truth",
        (
            "SoccerNet's labels: every person's box, role, team, jersey number and pitch position. "
            "Every other tab is scored against these."
        ),
    ),
    "specialist": (
        "Specialist",
        (
            "The fine-tuned RF-DETR detector with ByteTrack: boxes for players, goalkeepers, referees and "
            "the ball, each with a track id. No teams, numbers or pitch positions yet."
        ),
    ),
    "foundation": (
        "Foundation",
        (
            'SAM 3.1 out of the box, with no training: one text prompt, "person", gives a mask and a track '
            "for everyone it finds, spectators included. No roles, ball, teams or numbers."
        ),
    ),
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


REPO = "https://github.com/nathankyryk/pitchview"

# Shown under the app and in the Space's README. The SAM License asks that published results
# acknowledge SAM; the clips' broadcast rights stay with the leagues and broadcasters.
CREDITS = (
    "**Credits.** Clips: SoccerNet Game State Reconstruction (SN-GSR-2024). The footage belongs to its "
    "leagues and broadcasters and is shown in short, compressed, annotated excerpts for research and "
    "demonstration. Models: SAM 3.1 (Meta, SAM License), RF-DETR (Roboflow, Apache-2.0), a pitch keypoint "
    "model fine-tuned from Roboflow's sports weights with Ultralytics (AGPL-3.0), PARSeq (Apache-2.0) and "
    "SigLIP (Google, Apache-2.0). Baseline: sn-gamestate (GPL-3.0), rescored with SoccerNet's sn-trackeval. "
    f"Code and full results: {REPO}"
)

SPACE_README = """---
title: PitchView
emoji: ⚽
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: {gradio}
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Soccer game state from broadcast video, on SoccerNet GSR
---

# PitchView

Broadcast soccer video in, game state out: every player's team, role, jersey number and position
on the pitch, frame by frame, scored on SoccerNet's Game State Reconstruction benchmark.

This Space plays pre-rendered test clips and runs no model. Code, method and full results: {repo}

{credits}
"""

SPACE_APP = '''"""PitchView's demo: pre-rendered clips with their scores. It runs no model. Made by `pitchview space`."""

from pathlib import Path

from viewer import build

build(Path(__file__).parent / "demo").launch()
'''


def make_space(bundle: Path, out: Path, gradio_version: str) -> int:
    """A Hugging Face Space folder: app.py, a README with the Space settings and credits, this
    module as viewer.py, the bundle as demo/, and Git LFS rules for the videos. Returns its size in bytes."""
    bundle, out = Path(bundle), Path(out)
    if not (bundle / "manifest.json").exists():
        raise FileNotFoundError(f"{bundle} has no manifest.json: run `pitchview bundle ... --out {bundle}` first")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle, out / "demo", dirs_exist_ok=True)
    shutil.copyfile(__file__, out / "viewer.py")
    files = {
        "app.py": SPACE_APP,
        "README.md": SPACE_README.format(gradio=gradio_version, repo=REPO, credits=CREDITS),
        ".gitattributes": "*.mp4 filter=lfs diff=lfs merge=lfs -text\n",
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    return sum(f.stat().st_size for f in out.rglob("*") if f.is_file())


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
    if s.get("gs_hota") is None:
        return ""
    lines = [f"### This clip\n**GS-HOTA {s['gs_hota']:.1f}**"]
    if s.get("note"):
        lines.append(s["note"])
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


def _tab(gr, bundle: Path, manifest: dict, tab: str, clip: str) -> tuple:
    title, blurb = TABS[tab]
    with gr.Tab(title, id=tab):
        gr.Markdown(blurb)
        with gr.Row():
            video = gr.Video(value=video_path(bundle, tab, clip), show_label=False, scale=3)
            with gr.Column(scale=1):
                score = gr.Markdown(score_card(manifest, clip, tab))
                stats = gr.Markdown(stats_card(manifest, clip))
    return tab, video, score, stats


# Restart both compare videos together (they are separate players, so this is how they line up).
PLAY_BOTH = """() => {
  const videos = ["#compare-a video", "#compare-b video"].map((s) => document.querySelector(s)).filter(Boolean);
  videos.forEach((v) => { v.pause(); v.currentTime = 0; });
  videos.forEach((v) => v.play());
}"""
PAUSE_BOTH = """() => {
  ["#compare-a video", "#compare-b video"].forEach((s) => document.querySelector(s)?.pause());
}"""


def _compare(gr, bundle: Path, tabs: list[str], clip: str) -> tuple:
    """Any two tabs side by side on the same clip."""
    titles = [(TABS[t][0], t) for t in tabs]
    left, right = (tabs[0], tabs[-1]) if "foundation" not in tabs else ("foundation", tabs[-1])
    with gr.Tab("Compare", id="compare"):
        gr.Markdown("Any two tabs side by side on the same clip.")
        with gr.Row():
            pick_a = gr.Dropdown(choices=titles, value=left, label="Left")
            pick_b = gr.Dropdown(choices=titles, value=right, label="Right")
        with gr.Row():
            video_a = gr.Video(value=video_path(bundle, left, clip), show_label=False, elem_id="compare-a")
            video_b = gr.Video(value=video_path(bundle, right, clip), show_label=False, elem_id="compare-b")
        with gr.Row():
            play = gr.Button("Play both from the start", variant="primary")
            pause = gr.Button("Pause both")
    play.click(None, js=PLAY_BOTH)
    pause.click(None, js=PAUSE_BOTH)
    return pick_a, pick_b, video_a, video_b


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
        with gr.Tabs(selected="full" if "full" in tabs else None):
            widgets = [_tab(gr, bundle, manifest, tab, first) for tab in tabs]
            compare = _compare(gr, bundle, tabs, first) if len(tabs) > 1 else None

        def show(name: str, *picks: str) -> list:
            out = []
            for tab, *_ in widgets:
                out += [video_path(bundle, tab, name), score_card(manifest, name, tab), stats_card(manifest, name)]
            return out + [video_path(bundle, tab, name) for tab in picks]

        outputs = [w for _, *ws in widgets for w in ws]
        if compare:
            pick_a, pick_b, video_a, video_b = compare
            clip.change(show, [clip, pick_a, pick_b], outputs + [video_a, video_b])
            pick_a.change(lambda name, tab: video_path(bundle, tab, name), [clip, pick_a], video_a)
            pick_b.change(lambda name, tab: video_path(bundle, tab, name), [clip, pick_b], video_b)
        else:
            clip.change(show, clip, outputs)
        gr.Markdown(CREDITS)
    return app
