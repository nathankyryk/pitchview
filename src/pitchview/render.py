"""Draw a clip's game state onto its video: team-coloured masks, nametags and a radar.

- Each person's SAM 3.1 mask is filled in their team's kit colour (ByteTrack people without a
  matching mask get a box outline instead).
- A nametag sits above each player: the name from the clip's roster ("L. Messi"), or "#10" when
  the roster has no name. The number is voted over the whole track, so it shows from the track's
  first frame (backfill); without backfill the tag reads "#?" until the video first reads it.
- A radar in the corner shows everyone on a 105 x 68 m pitch, plus the ball.

style="tracks" draws perception only, for the viewer's specialist and foundation tabs: each box or
mask in its role's colour (or its track's, when there is no role) with an "id N" tag, and no
teams, numbers or radar.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pitchview.jersey import shirt_region
from pitchview.pitch import LENGTH, WIDTH

REFEREE = (250, 204, 21)  # yellow
UNKNOWN = (170, 170, 170)  # grey: no team yet
FALLBACK_KITS = {"left": (66, 133, 244), "right": (234, 67, 53)}
MASK_ALPHA = 0.5
ROLE_COLOURS = {"player": (66, 165, 245), "goalkeeper": (186, 104, 200), "referee": REFEREE, "ball": (255, 152, 0)}
TRACK_PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
                 (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
                 (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195)]  # fmt: skip
STATE_DEFAULTS = {"team": None, "jersey": None, "x_m": np.nan, "y_m": np.nan, "sam_obj_id": -1}


def short_name(full: str) -> str:
    """Tag format: first initial and the rest ("Kevin De Bruyne" -> "K. De Bruyne"); one-word names stay."""
    parts = full.split()
    if len(parts) <= 1:
        return full.strip()
    return f"{parts[0][0]}. {' '.join(parts[1:])}"


def load_roster(path: Path) -> dict[str, dict[str, str]]:
    """{"left": {"10": "Full Name"}, "right": {...}, "goalkeeper": {"left": "32", "right": "1"}} from
    rosters/<clip>.json (the team sheet); {} if there is none. A number with a blank name is on the
    sheet, its name not filled in yet. "goalkeeper" gives each side's keeper's number, if known."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    roster = {side: {str(k): v or "" for k, v in data.get(side, {}).items()} for side in ("left", "right")}
    roster["goalkeeper"] = {side: str(n) for side, n in (data.get("goalkeeper") or {}).items() if n is not None}
    return roster


def keeper_number(roster: dict, team: str | None) -> str | None:
    """The team's goalkeeper's number from the sheet: GSR doesn't number goalkeepers, but a team
    has one on the pitch, so role and side name him."""
    return roster.get("goalkeeper", {}).get(team or "")


def tag_text(role: str, team: str | None, number: str | None, revealed: bool, roster: dict,
             sheet_only: bool = True) -> str | None:  # fmt: skip
    """What the nametag says. Referees get none. Goalkeepers are unnumbered in GSR, so a keeper's tag
    comes from the sheet's goalkeeper for his side ("GK" without one). With sheet_only, a number that
    is not on the team's sheet (when there is one) is a misread, so it shows as "#?" (the
    ground-truth tab turns this off, to show the labels as they are)."""
    if role == "referee":
        return None
    if role == "goalkeeper":
        keeper = keeper_number(roster, team)
        if keeper is None:
            return "GK"
        name = roster.get(team or "", {}).get(keeper)
        return short_name(name) if name else f"#{keeper}"
    if number is None or not revealed:
        return "#?"
    sheet = roster.get(team or "", {})
    name = sheet.get(str(number))
    if name:
        return short_name(name)
    if sheet_only and sheet and str(number) not in sheet:
        return "#?"
    return f"#{number}"


def track_colour(role: str, track_id: int) -> tuple[int, int, int]:
    """tracks style: the role's colour, or one per track when there is no role (SAM's "person")."""
    return ROLE_COLOURS.get(role) or TRACK_PALETTE[int(track_id) % len(TRACK_PALETTE)]


def truth_state(boxes: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ground-truth labels as a game state (people) and ball positions, for the truth tab."""
    people = [b for b in boxes if b.role != "ball" and b.track_id is not None]
    state = pd.DataFrame(
        {
            "frame": [b.frame for b in people],
            "track_id": [b.track_id for b in people],
            "role": [b.role for b in people],
            "team": pd.Series([b.team for b in people], dtype=object),
            "jersey": pd.Series([b.jersey for b in people], dtype=object),
            "x_m": [np.nan if b.pitch_x is None else b.pitch_x for b in people],
            "y_m": [np.nan if b.pitch_y is None else b.pitch_y for b in people],
            "x1": [b.x for b in people],
            "y1": [b.y for b in people],
            "x2": [b.x + b.w for b in people],
            "y2": [b.y + b.h for b in people],
            "sam_obj_id": -1,
        }
    )
    balls = [b for b in boxes if b.role == "ball" and b.pitch_x is not None]
    ball = pd.DataFrame({"frame": [b.frame for b in balls], "x_m": [b.pitch_x for b in balls],
                         "y_m": [b.pitch_y for b in balls]})  # fmt: skip
    return state, ball


def first_reads(reads: pd.DataFrame, numbers: dict, min_conf: float) -> dict[int, int]:
    """Per track, the first frame where the reader saw the number the vote settled on."""
    voted = reads["track_id"].map(numbers)
    hits = reads[(reads["number"] == voted) & (reads["conf"] >= min_conf)]
    return hits.groupby("track_id")["frame"].min().astype(int).to_dict()


def _not_grass(pixels: np.ndarray) -> np.ndarray:
    """Drop grass-green and near-black pixels from (N, 3) RGB."""
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] > 50)  # OpenCV hue runs 0-180
    return pixels[~grass & (hsv[:, 2] > 50)]


def vivid(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """A display version of a kit colour: saturated colours made bright, whites and greys lifted."""
    h, s, v = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0].astype(int)
    if s > 70:
        s, v = max(s, 190), max(v, 210)
    else:
        v = max(v, 225)
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0, 0])


def kit_colours(images: dict[int, np.ndarray], state: pd.DataFrame) -> dict[str, tuple[int, int, int]]:
    """Each side's kit colour: the median non-grass shirt pixel of its players in the sampled frames,
    made vivid for display. If the two sides come out alike, fixed blue and red are used instead."""
    pixels: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    players = state[(state["role"] == "player") & state["team"].isin(["left", "right"])]
    for row in players[players["frame"].isin(list(images))].itertuples(index=False):
        x1, y1, x2, y2 = shirt_region((row.x1, row.y1, row.x2, row.y2))
        crop = images[int(row.frame)][max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]
        if crop.size:
            pixels[row.team].append(_not_grass(crop.reshape(-1, 3)))
    kits = {}
    for side, p in pixels.items():
        p = np.concatenate(p) if p else np.zeros((0, 3), np.uint8)
        kits[side] = vivid(tuple(int(c) for c in np.median(p, axis=0))) if len(p) else FALLBACK_KITS[side]
    if np.linalg.norm(np.subtract(kits["left"], kits["right"])) < 80:
        return dict(FALLBACK_KITS)
    return kits


def anchors(state: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Where each tag sits: above the box's top centre, smoothed along the track so it doesn't jitter."""
    s = state.sort_values(["track_id", "frame"]).copy()
    s["ax"] = (s["x1"] + s["x2"]) / 2
    s["ay"] = s["y1"]
    for col in ("ax", "ay"):
        s[col] = s.groupby("track_id")[col].transform(lambda v: v.rolling(window, center=True, min_periods=1).mean())
    return s


def tag_position(ax: float, ay: float, w: int, h: int, width: int, height: int, gap: int = 8) -> tuple[int, int]:
    """Top-left corner of a w x h tag centred above (ax, ay), kept inside the frame."""
    x0 = int(min(max(ax - w / 2, 2), width - w - 2))
    y0 = int(min(max(ay - h - gap, 2), height - h - 2))
    return x0, y0


def text_colour(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black on light badges, white on dark ones."""
    r, g, b = rgb
    return (0, 0, 0) if 0.299 * r + 0.587 * g + 0.114 * b > 150 else (255, 255, 255)


class Radar:
    """A top-down pitch at `scale` px per metre, camera side at the bottom (GSR's +y)."""

    def __init__(self, scale: float = 5.0, margin: int = 12):
        self.scale, self.margin = scale, margin
        self.size = (int(LENGTH * scale) + 2 * margin, int(WIDTH * scale) + 2 * margin)  # width, height
        self.base = self._pitch()
        self.font = ImageFont.load_default(size=max(10, int(scale * 2.2)))

    def px(self, x_m: float, y_m: float) -> tuple[int, int]:
        return (int(self.margin + (x_m + LENGTH / 2) * self.scale), int(self.margin + (y_m + WIDTH / 2) * self.scale))

    def _pitch(self) -> Image.Image:
        img = Image.new("RGB", self.size, (34, 110, 52))
        d = ImageDraw.Draw(img)
        line, w = (235, 235, 235), 2
        d.rectangle([self.px(-52.5, -34), self.px(52.5, 34)], outline=line, width=w)
        d.line([self.px(0, -34), self.px(0, 34)], fill=line, width=w)
        r = 9.15 * self.scale
        cx, cy = self.px(0, 0)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=line, width=w)
        for side in (-1, 1):
            goal_x = side * 52.5
            d.rectangle(
                sorted_box(self.px(goal_x, -20.16), self.px(goal_x - side * 16.5, 20.16)), outline=line, width=w
            )
            d.rectangle(sorted_box(self.px(goal_x, -9.16), self.px(goal_x - side * 5.5, 9.16)), outline=line, width=w)
        return img

    def draw(
        self, people: list[tuple[float, float, tuple, str | None]], ball: tuple[float, float] | None
    ) -> np.ndarray:
        """people: (x_m, y_m, colour, label). Returns an RGB array."""
        img = self.base.copy()
        d = ImageDraw.Draw(img)
        r = max(5, int(self.scale * 1.6))
        for x, y, colour, label in people:
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            cx, cy = self.px(x, y)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour, outline=(20, 20, 20), width=1)
            if label:
                d.text((cx, cy), label, fill=text_colour(colour), font=self.font, anchor="mm")
        if ball is not None and np.all(np.isfinite(ball)):
            bx, by = self.px(*ball)
            d.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=(255, 255, 255), outline=(0, 0, 0), width=1)
        return np.asarray(img)


def sorted_box(a: tuple[int, int], b: tuple[int, int]) -> list[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]


def index_first(path: Path) -> bool:
    """Whether an MP4's index (moov) comes before its data (mdat), so a browser can start at once."""
    with open(path, "rb") as f:
        head = f.read(4096)
    return b"moov" in head and (b"mdat" not in head or head.index(b"moov") < head.index(b"mdat"))


def faststart(path: Path) -> None:
    """Move an MP4's index to the front without re-encoding (videos rendered before render_clip did it)."""
    import av

    path = Path(path)
    if index_first(path):
        return
    tmp = path.with_suffix(".fs.mp4")
    with av.open(str(path)) as src, av.open(str(tmp), "w", options={"movflags": "faststart"}) as dst:
        stream = dst.add_stream_from_template(src.streams.video[0])
        for packet in src.demux(src.streams.video[0]):
            if packet.dts is None:  # the demuxer's final flush packet
                continue
            packet.stream = stream
            dst.mux(packet)
    tmp.replace(path)


def shrink(src: Path, dst: Path, height: int = 720, crf: int = 26) -> None:
    """Re-encode an MP4 at a smaller height (width keeps the aspect ratio), index first: for the
    Hugging Face Space, where 1080p clips would be ~38 MB each."""
    import av

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    with av.open(str(src)) as inp, av.open(str(tmp), "w", options={"movflags": "faststart"}) as out:
        source = inp.streams.video[0]
        width = round(source.width * height / source.height / 2) * 2  # H.264 needs even sizes
        stream = out.add_stream("libx264", rate=source.average_rate)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": str(crf), "preset": "medium"}
        for frame in inp.decode(source):
            for packet in stream.encode(frame.reformat(width=width, height=height, format="yuv420p")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    tmp.replace(dst)


def blend_mask(frame: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], alpha: float = MASK_ALPHA) -> None:
    """Tint the mask's pixels and outline it, in place."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    region, m = frame[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1]
    region[m] = (region[m] * (1 - alpha) + np.array(colour) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(region, contours, -1, colour, 2)
    frame[y0:y1, x0:x1] = region


def render_clip(
    frames: list,  # gsr Frame objects, in order
    state: pd.DataFrame,
    masks: dict[tuple[int, int], dict],
    ball: pd.DataFrame | None,
    roster: dict,
    first_frame: dict[int, int],
    out_path: Path,
    fps: float = 25.0,
    backfill: bool = False,
    title: str = "",
    max_frames: int | None = None,
    style: str = "full",
    sheet_only: bool = True,
) -> int:
    """Write the annotated clip as H.264 MP4. Returns the number of frames written.
    style "full": team colours, nametags and the radar; "tracks": perception only (see the module notes)."""
    import av
    from pycocotools import mask as mask_utils

    frames = frames[:max_frames] if max_frames else frames
    state = state.assign(**{c: v for c, v in STATE_DEFAULTS.items() if c not in state})
    tracks_only = style == "tracks"
    if tracks_only:
        kits: dict = {}
    else:
        sample = {f.index: np.asarray(Image.open(f.path).convert("RGB")) for f in frames[:: max(1, len(frames) // 25)]}
        kits = kit_colours(sample, state)
    placed = anchors(state)
    by_frame = {f: rows for f, rows in placed.groupby("frame")}
    ball_at = {} if ball is None else {int(r.frame): (r.x_m, r.y_m) for r in ball.itertuples(index=False)}
    radar = Radar(scale=5.0)
    tag_font = ImageFont.load_default(size=22)
    title_font = ImageFont.load_default(size=24)

    def colour_of(role: str, team: str | None, track_id: int) -> tuple[int, int, int]:
        if tracks_only:
            return track_colour(role, track_id)
        if role == "referee":
            return REFEREE
        return kits.get(team, UNKNOWN) if team in kits else UNKNOWN

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")
    container = av.open(str(tmp), "w", options={"movflags": "faststart"})  # index first: browsers can start at once
    width, height = frames[0].width, frames[0].height
    stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": "23", "preset": "medium"}
    for f in frames:
        img = np.array(Image.open(f.path).convert("RGB"))
        rows = by_frame.get(f.index)
        people, tags = [], []
        if rows is not None:
            for r in rows.itertuples(index=False):
                colour = colour_of(r.role, r.team, r.track_id)
                rle = masks.get((f.index, int(r.sam_obj_id))) if r.sam_obj_id >= 0 else None
                if rle is not None:
                    blend_mask(img, mask_utils.decode(rle).astype(bool), colour)
                else:
                    cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), colour, 2)
                number = r.jersey if isinstance(r.jersey, str) and r.jersey else None
                if r.role == "goalkeeper" and not tracks_only:
                    number = keeper_number(roster, r.team)
                revealed = backfill or (int(r.track_id) in first_frame and f.index >= first_frame[int(r.track_id)])
                if tracks_only:
                    text = None if r.role == "ball" else f"id {int(r.track_id)}"
                else:
                    text = tag_text(r.role, r.team, number, revealed, roster, sheet_only)
                if text:
                    tags.append((r.ax, r.ay, colour, text))
                shown = revealed and (tracks_only or text != "#?")  # the radar agrees with the tag
                if np.isfinite(r.x_m) and np.isfinite(r.y_m):  # no dot without a pitch position
                    people.append((r.x_m, r.y_m, colour, number if shown else None))
        canvas = Image.fromarray(img)
        draw = ImageDraw.Draw(canvas)
        tags.sort(key=lambda t: t[3] != "#?")  # names on top, so a "#?" never covers one
        for ax, ay, colour, text in tags:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=tag_font)
            w, h = right - left + 14, bottom - top + 10
            x0, y0 = tag_position(ax, ay, w, h, width, height)
            draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=6, fill=colour, outline=(20, 20, 20))
            draw.text((x0 + w / 2, y0 + h / 2), text, fill=text_colour(colour), font=tag_font, anchor="mm")
        if title:
            draw.rounded_rectangle([16, 16, 30 + draw.textlength(title, font=title_font), 54], radius=6, fill=(0, 0, 0))
            draw.text((23, 35), title, fill=(255, 255, 255), font=title_font, anchor="lm")
        frame_out = np.asarray(canvas).copy()
        if not tracks_only:
            pane = radar.draw(people, ball_at.get(f.index))
            ph, pw = pane.shape[:2]
            y0, x0 = height - ph - 16, width - pw - 16
            frame_out[y0 : y0 + ph, x0 : x0 + pw] = (frame_out[y0 : y0 + ph, x0 : x0 + pw] * 0.15 + pane * 0.85).astype(
                np.uint8
            )
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame_out, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    tmp.replace(out_path)
    return len(frames)
