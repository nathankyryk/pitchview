"""Speed benchmark: how fast each model runs on one GPU, one frame at a time.

Each model gets the same frames, already decoded in memory (JPEG decoding is not counted), a
warm-up, and then a timed run with the GPU synchronised around every frame. Reported per model:
frames per second, median milliseconds per frame, peak GPU memory, parameters and file size.

SAM 3.1 is a video model: it tracks a whole clip in one stream, so it is timed over the first
N frames of a clip, with loading the clip left out.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from pitchview.gsr import Clip
from pitchview.jersey import shirt_crop


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_gb() -> float | None:
    import torch

    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None


def time_per_frame(run: Callable[[int], object], n: int, warmup: int) -> dict:
    """Call run(i) for frames 0..n-1 after `warmup` untimed calls; GPU-synchronised timings."""
    for i in range(warmup):
        run(i % n)
    _sync()
    _reset_memory()
    times = []
    for i in range(n):
        start = time.perf_counter()
        run(i)
        _sync()
        times.append(time.perf_counter() - start)
    times = np.array(times)
    return {
        "frames": n,
        "fps": float(n / times.sum()),
        "ms_median": float(np.median(times) * 1000),
        "ms_p90": float(np.percentile(times, 90) * 1000),
        "peak_memory_gb": _peak_memory_gb(),
    }


def parameters(module) -> int | None:
    """Parameter count of a torch module, or None when the model hides it."""
    try:
        return int(sum(p.numel() for p in module.parameters()))
    except AttributeError:
        return None


def size_mb(path: Path | None) -> float | None:
    return round(path.stat().st_size / 1e6, 1) if path and path.is_file() else None


def bench_detector(model, images: list, warmup: int, threshold: float = 0.1) -> dict:
    """RF-DETR, one image per call, with its own pre- and post-processing."""
    return time_per_frame(lambda i: model.predict(images[i], threshold=threshold), len(images), warmup)


def bench_ultralytics(model, images: list, warmup: int, device: str | None, imgsz: int | None = None) -> dict:
    """An Ultralytics model (the pitch-keypoint model, YOLO26), one image per call."""
    options = {} if device == "cpu" else {"quantize": "fp16"}  # Ultralytics 8.4 replaced half=True
    if imgsz:
        options["imgsz"] = imgsz
    return time_per_frame(lambda i: model.predict(images[i], verbose=False, device=device, **options), len(images), warmup)


def bench_reader(reader, images: list, boxes: list[list], warmup: int) -> dict:
    """PARSeq on every player's shirt in a frame, as one batch per frame."""
    crops = [[shirt_crop(im, b) for b in bs] for im, bs in zip(images, boxes)]
    out = time_per_frame(lambda i: reader.read(crops[i]) if crops[i] else None, len(images), warmup)
    out["crops_per_frame"] = float(np.mean([len(c) for c in crops]))
    return out


def bench_sam(tracker, clip: Clip, n: int) -> dict:
    """SAM 3.1 tracking the clip's first n frames in one stream (clip loading not timed)."""
    _reset_memory()
    stream = tracker.track(clip, max_frames=n)
    first = next(stream)  # session start, clip load and the first frame's detection
    _sync()
    start = time.perf_counter()
    count = 1 + sum(1 for _ in stream)
    _sync()
    elapsed = time.perf_counter() - start
    del first
    return {
        "frames": count - 1,
        "fps": float((count - 1) / elapsed),
        "ms_median": None,
        "ms_p90": None,
        "peak_memory_gb": _peak_memory_gb(),
        "note": "video stream, first frame (prompt) excluded",
    }
