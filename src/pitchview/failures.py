"""Failure examples for the README, found from the data rather than picked by hand.

Each finder applies one stated rule to a run's game state and returns a case; `draw_case` turns
it into a PNG.
- calibration: on the clip with the worst median position error, the frame whose error is that
  clip's median (a typical frame, not a freak one). True positions vs predicted, on the radar.
- jersey: the wrongly numbered player track seen for the most frames, with the shirt crops whose
  reads produced the wrong number.
- ball: the middle of the longest run of frames where the ball is labelled but not detected.
- id_switch: a player whose track id changes from one long track to another within a few frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pitchview.jersey import MIN_CONF, shirt_region

BALL_MIN_SCORE = 0.3  # the confidence the game state uses for the ball (gamestate.ball_positions)


@dataclass
class Case:
    kind: str
    clip: str
    frames: list[int]
    caption: str
    data: dict = field(default_factory=dict)


def match(pred: np.ndarray, truth: np.ndarray, min_iou: float = 0.5) -> list[tuple[int, int]]:
    """One-to-one (pred, truth) index pairs of x1,y1,x2,y2 boxes with IoU >= min_iou."""
    from scipy.optimize import linear_sum_assignment

    from pitchview.segment import box_iou

    if len(pred) == 0 or len(truth) == 0:
        return []
    iou = box_iou(pred, truth)
    return [(int(i), int(j)) for i, j in zip(*linear_sum_assignment(-iou)) if iou[i, j] >= min_iou]


def _people(boxes: list) -> list:
    return [b for b in boxes if b.role != "ball" and b.track_id is not None]


def _xyxy(boxes: list) -> np.ndarray:
    return np.array([[b.x, b.y, b.x + b.w, b.y + b.h] for b in boxes], dtype=float).reshape(-1, 4)


def _frame_pairs(state: pd.DataFrame, clip) -> dict[int, list[tuple[pd.Series, object]]]:
    """Per frame, each game-state row matched to the true person it covers."""
    truth = clip.boxes_by_frame()
    out = {}
    for frame, rows in state.groupby("frame"):
        gt = _people(truth.get(int(frame), []))
        pairs = match(rows[["x1", "y1", "x2", "y2"]].to_numpy(float), _xyxy(gt))
        out[int(frame)] = [(rows.iloc[i], gt[j]) for i, j in pairs]
    return out


def calibration_case(clips: list, states: dict[str, pd.DataFrame], game: str) -> Case | None:
    best = None
    for clip in clips:
        if str(clip.labels["info"].get("game_id")) != game or clip.name not in states:
            continue
        per_frame = {}
        for frame, pairs in _frame_pairs(states[clip.name], clip).items():
            pairs = [(r, b) for r, b in pairs if b.pitch_x is not None and np.isfinite(r.x_m)]
            if len(pairs) >= 6:
                per_frame[frame] = (float(np.median([np.hypot(r.x_m - b.pitch_x, r.y_m - b.pitch_y) for r, b in pairs])),
                                    pairs)  # fmt: skip
        if not per_frame:
            continue
        clip_median = float(np.median([e for e, _ in per_frame.values()]))
        if best is None or clip_median > best[0]:
            frame = min(per_frame, key=lambda f: abs(per_frame[f][0] - clip_median))
            best = (clip_median, clip, frame, per_frame[frame][1])
    if best is None:
        return None
    median, clip, frame, pairs = best
    points = [((r.x_m, r.y_m), (b.pitch_x, b.pitch_y), b.team) for r, b in pairs]
    return Case("calibration", clip.name, [frame],
                f"Unseen stadium (game {game}), {clip.name} frame {frame}: the pitch model's mapping is off by "
                f"{median:.1f} m (median) on this clip. Hollow: true positions; filled: predicted.",
                {"points": points, "boxes": [(r.x1, r.y1, r.x2, r.y2, b.team) for r, b in pairs]})  # fmt: skip


def jersey_case(clips: list, tracks: dict[str, pd.DataFrame], numbers: dict[str, dict],
                reads: dict[str, pd.DataFrame]) -> Case | None:  # fmt: skip
    """tracks, numbers and reads are per clip and share the reader's track ids (the tracker's, before
    any linking), so the misread crops can be found again."""
    from pitchview.evaluate import jersey_truth

    best = None
    for clip in clips:
        if clip.name not in tracks or clip.name not in reads:
            continue
        voted = numbers.get(clip.name, {})
        for t in jersey_truth(tracks[clip.name], clip).itertuples(index=False):
            number = voted.get(int(t.track_id))
            if number is None or t.gt_jersey is None or number == t.gt_jersey:
                continue
            if best is None or t.frames > best[0]:
                best = (t.frames, clip, int(t.track_id), number, t.gt_jersey)
    if best is None:
        return None
    frames_seen, clip, track, number, true = best
    r = reads[clip.name]
    total = int((r["track_id"] == track).sum())
    wrong = r[(r["track_id"] == track) & (r["number"] == number) & (r["conf"] >= MIN_CONF)]
    rows = tracks[clip.name].set_index(["track_id", "frame"])
    crops = []
    for frame in wrong["frame"].astype(int).tolist()[:4]:
        if (track, frame) in rows.index:
            row = rows.loc[(track, frame)]
            crops.append((frame, shirt_region((row.x1, row.y1, row.x2, row.y2))))
    return Case("jersey", clip.name, [f for f, _ in crops],
                f"{clip.name}: a player wearing {true} was read as {number}. {len(wrong)} of the track's {total} "
                f"reads said {number} with enough confidence to vote, and {number} won the vote, so {frames_seen} "
                "frames carry the wrong number. The shirt crop behind the misread:",
                {"crops": crops, "read": number, "true": true})  # fmt: skip


def ball_case(clips: list, detections: dict[str, pd.DataFrame]) -> Case | None:
    best = None
    for clip in clips:
        if clip.name not in detections:
            continue
        d = detections[clip.name]
        balls = d[(d["class_name"] == "ball") & (d["score"] >= BALL_MIN_SCORE)]
        found = {int(f): g for f, g in balls.groupby("frame")}
        truth = clip.boxes_by_frame()
        run: list[tuple[int, object]] = []
        for frame in clip.frames:
            boxes = truth.get(frame.index, [])
            gt = [b for b in boxes if b.role == "ball"]
            # in play: on the pitch, and not held or at someone's feet
            free = bool(gt) and _on_pitch(gt[0]) and not _touching_person(gt[0], boxes)
            missed = free and not _ball_near(found.get(frame.index), gt[0])
            if missed:
                run.append((frame.index, gt[0]))
            if run and (not missed or frame is clip.frames[-1]):
                if best is None or len(run) > len(best[1]):
                    best = (clip, list(run))
                run = []
    if best is None:
        return None
    clip, run = best
    frame, box = run[len(run) // 2]
    return Case("ball", clip.name, [frame],
                f"{clip.name}: the longest run of frames where a loose ball in play (on the pitch, away from every "
                f"player) goes undetected: {len(run)} frames ({len(run) / clip.fps:.1f} s). The white circle is the "
                f"labelled ball ({box.w:.0f} x {box.h:.0f} px); crop around frame {frame}, enlarged 2x.",
                {"ball": (box.x, box.y, box.w, box.h)})  # fmt: skip


def _on_pitch(ball, inside_m: float = 1.0) -> bool:
    """Whether the ball's labelled pitch position is at least `inside_m` inside the touchlines and goal lines."""
    from pitchview.pitch import LENGTH, WIDTH

    if ball.pitch_x is None or ball.pitch_y is None:
        return False
    return abs(ball.pitch_x) <= LENGTH / 2 - inside_m and abs(ball.pitch_y) <= WIDTH / 2 - inside_m


def _touching_person(ball, boxes: list, margin: float = 0.5) -> bool:
    """Whether the ball's centre is inside a person's box widened by `margin` of its size."""
    cx, cy = ball.x + ball.w / 2, ball.y + ball.h / 2
    for b in boxes:
        if b.role == "ball":
            continue
        mx, my = b.w * margin, b.h * margin
        if b.x - mx <= cx <= b.x + b.w + mx and b.y - my <= cy <= b.y + b.h + my:
            return True
    return False


def _ball_near(found: pd.DataFrame | None, gt, max_px: float = 20.0) -> bool:
    if found is None:
        return False
    cx, cy = gt.x + gt.w / 2, gt.y + gt.h / 2
    dx = (found["x1"] + found["x2"]) / 2 - cx
    dy = (found["y1"] + found["y2"]) / 2 - cy
    return bool((np.hypot(dx, dy) <= max_px).any())


def id_switch_case(clips: list, states: dict[str, pd.DataFrame], min_run: int = 25, max_gap: int = 5) -> Case | None:
    """A true player covered by track A for min_run+ frames, then by track B within max_gap frames."""
    best = None
    for clip in clips:
        if clip.name not in states:
            continue
        follow: dict[int, list[tuple[int, int, tuple]]] = {}
        for frame, pairs in _frame_pairs(states[clip.name], clip).items():
            for r, b in pairs:
                follow.setdefault(b.track_id, []).append((frame, int(r.track_id), (r.x1, r.y1, r.x2, r.y2)))
        for person, seq in follow.items():
            seq.sort()
            for k in range(1, len(seq)):
                (f0, a, box_a), (f1, b, box_b) = seq[k - 1], seq[k]
                if a == b or f1 - f0 > max_gap:
                    continue
                before = sum(1 for _, t, _ in seq[:k] if t == a)
                after = sum(1 for _, t, _ in seq[k:] if t == b)
                score = min(before, after)
                if score >= min_run and (best is None or score > best[0]):
                    best = (score, clip, person, (f0, a, box_a), (f1, b, box_b))
    if best is None:
        return None
    _, clip, _, (f0, a, box_a), (f1, b, box_b) = best
    return Case("id_switch", clip.name, [f0, f1],
                f"{clip.name}: one player is track {a} until frame {f0}, then track {b} from frame {f1}: an ID switch "
                f"between two long tracks (left: before, right: after).",
                {"boxes": [(f0, a, box_a), (f1, b, box_b)]})  # fmt: skip


def draw_case(case: Case, clip, path) -> None:
    """The case as a PNG: frames, crops or a radar, depending on its kind."""
    images = {f.index: f.path for f in clip.frames}
    font = ImageFont.load_default(size=26)
    if case.kind == "calibration":
        canvas = _calibration_figure(case, Image.open(images[case.frames[0]]).convert("RGB"), font)
    elif case.kind == "jersey":
        crops = []
        for frame, (x1, y1, x2, y2) in case.data["crops"]:
            crop = Image.open(images[frame]).convert("RGB").crop((x1, y1, x2, y2))
            crops.append(crop.resize((max(1, crop.width * 300 // max(crop.height, 1)), 300), Image.LANCZOS))
        canvas = _strip(crops, [f"read {case.data['read']}, true {case.data['true']}"] * len(crops), font)
    elif case.kind == "ball":
        frame = Image.open(images[case.frames[0]]).convert("RGB")
        x, y, w, h = case.data["ball"]
        cx, cy = x + w / 2, y + h / 2
        left = int(min(max(cx - 240, 0), frame.width - 480))  # a 480 x 270 window kept inside the frame
        top = int(min(max(cy - 135, 0), frame.height - 270))
        crop = frame.crop((left, top, left + 480, top + 270)).resize((960, 540), Image.LANCZOS)
        d = ImageDraw.Draw(crop)
        bx, by, r = (cx - left) * 2, (cy - top) * 2, max(w, h) + 14
        d.ellipse([bx - r, by - r, bx + r, by + r], outline=(255, 255, 255), width=4)
        canvas = crop
    else:  # id_switch
        crops, labels = [], []
        for frame, track, (x1, y1, x2, y2) in case.data["boxes"]:
            img = Image.open(images[frame]).convert("RGB")
            pad = max(x2 - x1, y2 - y1) * 1.5
            left, top = int(x1 - pad), int(y1 - pad)
            crop = img.crop((left, top, int(x2 + pad), int(y2 + pad)))
            # mark the player, since other players can share the crop
            ImageDraw.Draw(crop).rectangle([x1 - left - 3, y1 - top - 3, x2 - left + 3, y2 - top + 3],
                                           outline=(255, 255, 255), width=2)  # fmt: skip
            crops.append(crop.resize((max(1, crop.width * 360 // max(crop.height, 1)), 360), Image.LANCZOS))
            labels.append(f"frame {frame}: track {track}")
        canvas = _strip(crops, labels, font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _strip(images: list[Image.Image], labels: list[str], font) -> Image.Image:
    gap, label_h = 12, 40
    width = sum(im.width for im in images) + gap * (len(images) + 1)
    height = max(im.height for im in images) + label_h + 2 * gap
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    x = gap
    for im, label in zip(images, labels):
        canvas.paste(im, (x, gap))
        d.text((x + im.width / 2, gap + im.height + label_h / 2), label, fill=(240, 240, 240), font=font, anchor="mm")
        x += im.width + gap
    return canvas


def _calibration_figure(case: Case, frame: Image.Image, font) -> Image.Image:
    from pitchview.render import REFEREE, Radar, kit_colours

    people = pd.DataFrame(
        [(case.frames[0], "player", team, x1, y1, x2, y2) for x1, y1, x2, y2, team in case.data["boxes"]],
        columns=["frame", "role", "team", "x1", "y1", "x2", "y2"],
    )
    colours = {**kit_colours({case.frames[0]: np.asarray(frame)}, people), None: REFEREE}
    d = ImageDraw.Draw(frame)
    for x1, y1, x2, y2, team in case.data["boxes"]:
        d.rectangle([x1, y1, x2, y2], outline=colours.get(team, REFEREE), width=3)
    frame = frame.resize((960, 540), Image.LANCZOS)
    radar = Radar(scale=8.0)
    pitch = radar.base.copy()
    p = ImageDraw.Draw(pitch)
    r = 8
    for pred, true, team in case.data["points"]:
        colour = colours.get(team, REFEREE)
        (px, py), (tx, ty) = radar.px(*pred), radar.px(*true)
        p.line([px, py, tx, ty], fill=(255, 255, 255), width=2)
        p.ellipse([tx - r, ty - r, tx + r, ty + r], outline=colour, width=3)
        p.ellipse([px - r, py - r, px + r, py + r], fill=colour, outline=(20, 20, 20))
    pitch = pitch.resize((int(pitch.width * 540 / pitch.height), 540), Image.LANCZOS)
    canvas = Image.new("RGB", (frame.width + pitch.width + 36, 540 + 24), (20, 20, 20))
    canvas.paste(frame, (12, 12))
    canvas.paste(pitch, (frame.width + 24, 12))
    return canvas
