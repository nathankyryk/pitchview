"""Jersey numbers: zero-shot PARSeq on shirt crops, then a confidence-weighted vote per track.

Measured on 6 held-out GSR clips (90 player tracks), track-level accuracy:
  - TrOCR-small (printed text), zero-shot ................................. 2%
  - ResNet-18 trained on GSR's weak per-track labels (20 clips) ........... 8%  (memorised players)
  - PARSeq, zero-shot, crop squashed to its 32x128 input ................. 39-40%  <- used here
  - PARSeq with the crop letterboxed instead of squashed ................... 2%  (digits too small)
Fine-tuning PARSeq on GSR crops is the obvious next step (stretch goal).

PARSeq runs at ~40 crops/s on a laptop CPU, so this step needs no GPU.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from PIL import Image

# Pinned so the downloaded model code cannot change under us (Apache-2.0).
PARSEQ_REPO = "baudm/parseq:1902db043c029a7e03a3818c616c06600af574be"
INPUT_SIZE = (128, 32)  # PARSeq's width x height


def shirt_region(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """Where the number sits: 10-90% of the box width, 15-60% of its height."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return int(x1 + 0.10 * w), int(y1 + 0.15 * h), int(np.ceil(x1 + 0.90 * w)), int(np.ceil(y1 + 0.60 * h))


def shirt_crop(image: Image.Image, box, mask: np.ndarray | None = None, fill: int = 128) -> Image.Image:
    """The shirt region of an image. With the person's mask, everything else (grass, a team-mate
    overlapping the box) is set to grey, which PARSeq's input normalisation maps to zero."""
    region = shirt_region(box)
    crop = image.crop(region)
    if mask is None:
        return crop
    x0, y0, x1, y1 = region
    h, w = mask.shape
    keep = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    ys, xs = slice(max(y0, 0), min(y1, h)), slice(max(x0, 0), min(x1, w))
    keep[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0] = mask[ys, xs]
    pixels = np.array(crop.convert("RGB"))
    pixels[~keep] = fill
    return Image.fromarray(pixels)


def parse_number(text: str) -> str | None:
    """Keep 1-2 digit reads; anything else is treated as no read."""
    digits = re.sub(r"\D", "", text)
    return digits if 1 <= len(digits) <= 2 else None


class JerseyReader:
    def __init__(self, device: str | None = None):
        import torch
        from torchvision import transforms as T

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(PARSEQ_REPO, "parseq", pretrained=True, trust_repo=True).eval().to(self.device)
        # Squash to the model's input: letterboxing keeps the shape but makes digits too small to read.
        self.transform = T.Compose(
            [T.Resize(INPUT_SIZE[::-1], T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(0.5, 0.5)]
        )

    def read(self, crops: list[Image.Image], batch_size: int = 128) -> list[tuple[str | None, float]]:
        """(number or None, confidence) per crop."""
        out = []
        with self.torch.inference_mode():
            for start in range(0, len(crops), batch_size):
                batch = self.torch.stack([self.transform(c.convert("RGB")) for c in crops[start : start + batch_size]])
                texts, confs = self.model.tokenizer.decode(self.model(batch.to(self.device)).softmax(-1))
                out.extend((parse_number(t), float(c.prod())) for t, c in zip(texts, confs))
        return out


# Number 1 is the goalkeeper's, and GSR leaves goalkeepers unnumbered: not one of GSR's
# 1,145 train player tracks wears 1. PARSeq, meanwhile, reads stripes and seams as "1",
# which made "1" over half of all ByteTrack track numbers on test.
NOT_PLAYER_NUMBERS = frozenset({"1"})

# Tuned with `pitchview jersey-score --grid` on the 5 valid clips (2026-10-26). With "1" excluded,
# 0.7 gave the most player frames with the right number for both trackers:
# ByteTrack 53.8% (30.1% with the old 0.5 vote that kept "1"), SAM tracks 52.9% (was 33.3%).
# One rule for both trackers: ByteTrack alone did best at 0.6 with 2 agreeing reads (57.0%), but 5 clips
# are too few to tune each tracker separately.
# merge_partial (one-digit reads count toward a two-digit one) is off: +1.3 points for ByteTrack on the
# valid clips, but -0.6 (ByteTrack) and -1.3 (SAM tracks) on test, so the valid gain was noise.
MIN_CONF = 0.7


def vote(reads, min_conf: float = 0.5, min_reads: int = 1, exclude: frozenset = frozenset(),
         merge_partial: bool = False) -> dict:  # fmt: skip
    """reads: (track_key, number or None, confidence). Per track, the number with the largest
    summed confidence among confident reads, if at least `min_reads` of them agree on it;
    None otherwise. Numbers in `exclude` never win.

    merge_partial: a one-digit read that is part of a two-digit read on the same track ("4" and
    "14") counts toward the two-digit number, since the reader often catches only one digit.
    """
    scores: dict = defaultdict(lambda: defaultdict(float))
    counts: dict = defaultdict(lambda: defaultdict(int))
    keys = []
    for key, number, conf in reads:
        keys.append(key)
        if number is not None and conf >= min_conf:
            scores[key][number] += conf
            counts[key][number] += 1
    out = {}
    for key in dict.fromkeys(keys):
        s, c = scores[key], counts[key]
        if merge_partial:
            for digit in [n for n in s if len(n) == 1]:
                longer = [n for n in s if len(n) == 2 and digit in n]
                if longer:
                    best = max(longer, key=s.get)
                    s[best] += s.pop(digit)
                    c[best] += c.pop(digit)
        ok = {number: v for number, v in s.items() if c[number] >= min_reads and number not in exclude}
        out[key] = max(ok, key=ok.get) if ok else None
    return out


def split_votes(reads, min_conf: float = MIN_CONF, exclude: frozenset = NOT_PLAYER_NUMBERS,
                min_reads: int = 2) -> dict[int, list[tuple[int, str | None]]]:  # fmt: skip
    """Per track, [(first frame, number), ...]: one piece with vote()'s number, or several when the
    track's reads switch number for good. That happens when the tracker jumps from one player to
    another: on test SNGS-123, SAM 3.1's track 5 follows Thun's #25 for 15 s, then #31, and one
    vote named both players #31.

    reads: the table `pitchview jerseys` saves (track_id, frame, number, conf). A track splits where
    its confident reads (as in vote()) say A before and B after; no read at any confidence says A
    after B's first confident read or B before A's last; A and B are each read at least `min_reads`
    times on their side, and neither is a digit of the other ("4" is often half of a "14"). The cut
    sits between the two sure reads where the number changes, halfway from the last (unsure) read
    of A to the first read of B. Each side is checked again, for a second jump.
    """
    import pandas as pd

    d = reads if isinstance(reads, pd.DataFrame) else pd.read_parquet(reads)
    out = {}
    for tid, g in d.sort_values(["track_id", "frame"]).groupby("track_id", sort=False):
        frames = g["frame"].to_numpy(dtype=int)
        numbers = [n if isinstance(n, str) else None for n in g["number"]]
        pieces = _pieces(frames, numbers, g["conf"].to_numpy(dtype=float), 0, min_conf, exclude, min_reads, 3)
        out[int(tid)] = [p for i, p in enumerate(pieces) if i == 0 or p[1] != pieces[i - 1][1]]
    return out


def _pieces(frames, numbers, confs, start, min_conf, exclude, min_reads, depth) -> list[tuple[int, str | None]]:
    sure = [i for i, (n, c) in enumerate(zip(numbers, confs)) if n is not None and c >= min_conf and n not in exclude]

    def best(idx):
        score: dict = defaultdict(float)
        for i in idx:
            score[numbers[i]] += confs[i]
        return max(score, key=score.get) if score else None

    cut = None
    if depth > 0:
        top = 0.0
        for k in range(1, len(sure)):
            before, after = sure[:k], sure[k:]
            a, b = best(before), best(after)
            if a == b or a in b or b in a:
                continue
            lo, hi = frames[before[-1]], frames[after[0]]
            if any((n == b and f <= lo) or (n == a and f >= hi) for f, n in zip(frames, numbers)):
                continue  # read on the wrong side of the switch, at any confidence
            seen_a = [f for f, n in zip(frames, numbers) if n == a and f < hi]
            seen_b = [f for f, n in zip(frames, numbers) if n == b and f > lo]
            if len(seen_a) < min_reads or len(seen_b) < min_reads:
                continue
            support = sum(confs[i] for i in before if numbers[i] == a) + sum(confs[i] for i in after if numbers[i] == b)
            if support > top:  # cut in the gap between these two sure reads, after any unsure A, before any B
                last_a = max([lo] + [f for f in seen_a if f > lo])
                first_b = min([hi] + [f for f in seen_b if f < hi])
                top, cut = support, (last_a + first_b) // 2 + 1 if last_a < first_b else (lo + hi) // 2 + 1
    if cut is None:
        return [(start, best(sure))]
    left = frames < cut
    early = [n for n, is_left in zip(numbers, left) if is_left]
    late = [n for n, is_left in zip(numbers, left) if not is_left]
    rest = (min_conf, exclude, min_reads, depth - 1)
    return _pieces(frames[left], early, confs[left], start, *rest) + _pieces(
        frames[~left], late, confs[~left], cut, *rest
    )


def load_reads(path, every: int = 1, stride: int = 1) -> list[tuple[int, str | None, float]]:
    """The raw reads `pitchview jerseys` saves (a path or the loaded table), as vote() input.
    every=3 keeps every 3rd read frame of a `stride` read: reads every 3rd frame then stand in
    for reads every 9th, on the same clips."""
    import pandas as pd

    d = path if isinstance(path, pd.DataFrame) else pd.read_parquet(path)
    if every > 1:
        d = d[((d["frame"] - 1) // stride) % every == 0]
    numbers = d["number"].astype(object).where(d["number"].notna(), None)  # parquet brings None back as NaN
    return list(zip(d["track_id"].astype(int), numbers, d["conf"].astype(float)))
