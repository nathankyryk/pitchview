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


def vote(reads, min_conf: float = 0.5) -> dict:
    """reads: (track_key, number or None, confidence). Per track, the number with the largest
    summed confidence among confident reads; None when the track never had one."""
    scores: dict = defaultdict(lambda: defaultdict(float))
    keys = []
    for key, number, conf in reads:
        keys.append(key)
        if number is not None and conf >= min_conf:
            scores[key][number] += conf
    return {key: (max(scores[key], key=scores[key].get) if scores[key] else None) for key in dict.fromkeys(keys)}
