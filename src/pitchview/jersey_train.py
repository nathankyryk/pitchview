"""Fine-tune PARSeq to read GSR's jersey numbers.

Zero-shot PARSeq reads the right number for ~43% of player frames on all 58 valid clips. Its
misses are the reason names come and go in the demo: SAM 3.1 hands a player to a mask whose
reads never settle on a number. GSR labels every player box with its number, so the train
split gives ~140,000 labelled shirt crops (every 5th frame) for free.

The crops are the reader's own (jersey.shirt_crop, squashed to PARSeq's 128 x 32 input), so
training sees exactly what inference sees. Labels are weak: a box keeps its number when the
player turns and the number can't be seen. --keep agree trains only on crops where zero-shot
PARSeq already sees part of the number (a digit of it, at any confidence); --keep all trains on
every crop. Both are scored on valid with `jersey-score`, and the better one is kept.

Training uses PARSeq's own loss (permuted autoregressive decoding, strhub's training_step),
AdamW with a one-cycle schedule, and mild augmentation on the GPU (colour, blur, low
resolution, small rotations and shifts). A checkpoint is saved after every epoch.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from pitchview.jersey import INPUT_SIZE, shirt_crop


def _clip_crops(clip, stride: int, min_height: float, max_frames: int | None) -> tuple[np.ndarray, list[str]]:
    from PIL import Image

    by_frame = clip.boxes_by_frame()
    crops, labels = [], []
    frames = clip.frames[:max_frames] if max_frames else clip.frames
    for frame in frames[::stride]:
        boxes = [b for b in by_frame.get(frame.index, []) if b.role == "player" and b.jersey and b.h >= min_height]
        if not boxes:
            continue
        image = Image.open(frame.path).convert("RGB")
        for b in boxes:
            crop = shirt_crop(image, (b.x, b.y, b.x + b.w, b.y + b.h)).resize(INPUT_SIZE, Image.BICUBIC)
            crops.append(np.asarray(crop).transpose(2, 0, 1))
            labels.append(str(b.jersey))
    empty = np.zeros((0, 3, INPUT_SIZE[1], INPUT_SIZE[0]), np.uint8)
    return (np.stack(crops) if crops else empty), labels


def training_crops(clips, stride: int = 5, min_height: float = 0, max_frames: int | None = None,
                   workers: int = 8) -> tuple[np.ndarray, list[str]]:  # fmt: skip
    """Every `stride`th frame's labelled player crops, as uint8 (N, 3, 32, 128), and their numbers."""
    with ProcessPoolExecutor(workers) as pool:
        work = partial(_clip_crops, stride=stride, min_height=min_height, max_frames=max_frames)
        parts = list(pool.map(work, clips))
    crops = np.concatenate([p[0] for p in parts]) if parts else np.zeros((0, 3, 32, 128), np.uint8)
    return crops, [label for p in parts for label in p[1]]


def agrees(read: str | None, label: str) -> bool:
    """Zero-shot sees part of the number: its read shares a digit with the label."""
    return bool(read) and bool(set(read) & set(label))


def augment(x, generator=None):
    """Mild, per-crop augmentation of a float batch in [0, 1] (B, 3, 32, 128), on its device."""
    import torch
    import torch.nn.functional as F

    b, device = x.shape[0], x.device

    def u(lo, hi, *shape):
        return torch.rand((b, *shape), device=device, generator=generator) * (hi - lo) + lo

    # small rotation, scale and shift
    angle = u(-5, 5) * math.pi / 180
    scale = u(0.92, 1.08)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack(
        [torch.stack([cos, -sin * 0.25, u(-0.06, 0.06)], 1), torch.stack([sin * 4, cos, u(-0.08, 0.08)], 1)], 1
    )  # the 4:1 aspect ratio, so a rotation stays a rotation in pixels
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    # colour: brightness, contrast, saturation
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * u(0.7, 1.3, 1, 1, 1) + mean * u(0.8, 1.2, 1, 1, 1)
    grey = x.mean(dim=1, keepdim=True)
    x = grey + (x - grey) * u(0.6, 1.4, 1, 1, 1)
    # low resolution or blur for some crops, as on small, fast players
    low = torch.rand(b, device=device, generator=generator) < 0.3
    if low.any():
        small = F.interpolate(x[low], scale_factor=0.5, mode="bilinear", align_corners=False)
        x[low] = F.interpolate(small, size=x.shape[-2:], mode="bilinear", align_corners=False)
    blur = torch.rand(b, device=device, generator=generator) < 0.2
    if blur.any():
        x[blur] = F.avg_pool2d(x[blur], 3, stride=1, padding=1, count_include_pad=False)
    x = x + torch.randn(x.shape, device=device, generator=generator) * 0.02
    return x.clamp(0, 1)


def finetune(crops: np.ndarray, labels: list[str], out_dir: Path, name: str, epochs: int = 5,
             batch_size: int = 256, lr: float = 1e-4, weight_decay: float = 1e-2, device: str | None = None,
             seed: int = 0, max_steps: int | None = None, log=print) -> list[Path]:  # fmt: skip
    """Fine-tune the hub PARSeq on (crops, labels); saves its state dict after every epoch as
    <out_dir>/parseq_<name>_e<k>.pt and returns the paths."""
    import torch

    from pitchview.jersey import PARSEQ_REPO

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(PARSEQ_REPO, "parseq", pretrained=True, trust_repo=True).to(device)
    model.log = lambda *args, **kwargs: None  # training_step logs to a Lightning trainer; there is none
    model.train()
    data = torch.from_numpy(crops)  # uint8, stays on the CPU until each batch
    n = len(data)
    steps_per_epoch = math.ceil(n / batch_size)
    total = steps_per_epoch * epochs if max_steps is None else min(max_steps, steps_per_epoch * epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    saved, step, history = [], 0, []
    start = time.time()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        losses = []
        for i in range(0, n, batch_size):
            if step >= total:
                break
            idx = order[i : i + batch_size]
            x = data[idx].to(device, non_blocking=True).float() / 255
            x = (augment(x) - 0.5) / 0.5  # PARSeq's Normalize(0.5, 0.5)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = model.training_step((x, [labels[j] for j in idx]), step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # strhub trains with gradient_clip_val 20
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                log(
                    f"  epoch {epoch} step {step}/{total} loss {np.mean(losses[-50:]):.3f} ({time.time() - start:.0f} s)"
                )
        path = out_dir / f"parseq_{name}_e{epoch}.pt"
        torch.save(model.state_dict(), path)
        saved.append(path)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else None, "steps": step})
        log(f"epoch {epoch}: mean loss {history[-1]['loss']}, saved {path}")
        if step >= total:
            break
    (out_dir / f"parseq_{name}.json").write_text(
        json.dumps({"crops": n, "epochs": epochs, "batch_size": batch_size, "lr": lr, "history": history}, indent=1)
    )
    return saved
