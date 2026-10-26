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
MIN_CONF = 0.7


def vote(reads, min_conf: float = 0.5, min_reads: int = 1, exclude: frozenset = frozenset()) -> dict:
    """reads: (track_key, number or None, confidence). Per track, the number with the largest
    summed confidence among confident reads, if at least `min_reads` of them agree on it;
    None otherwise. Numbers in `exclude` are ignored."""
    scores: dict = defaultdict(lambda: defaultdict(float))
    counts: dict = defaultdict(lambda: defaultdict(int))
    keys = []
    for key, number, conf in reads:
        keys.append(key)
        if number is not None and number not in exclude and conf >= min_conf:
            scores[key][number] += conf
            counts[key][number] += 1
    out = {}
    for key in dict.fromkeys(keys):
        ok = {number: s for number, s in scores[key].items() if counts[key][number] >= min_reads}
        out[key] = max(ok, key=ok.get) if ok else None
    return out


def load_reads(path) -> list[tuple[int, str | None, float]]:
    """The raw reads `pitchview jerseys` saves, as vote() input."""
    import pandas as pd

    d = pd.read_parquet(path)
    numbers = d["number"].astype(object).where(d["number"].notna(), None)  # parquet brings None back as NaN
    return list(zip(d["track_id"].astype(int), numbers, d["conf"].astype(float)))
