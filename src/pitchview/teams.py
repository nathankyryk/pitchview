"""Team assignment: SigLIP embeddings of jersey crops, UMAP, then KMeans with k=2.

Crops come in two flavours for the box-vs-mask ablation: the plain box, and the same box
with every non-player pixel greyed out using the SAM 3.1 mask (no grass, no teammates).
Clusters are named "left"/"right" the way GSR labels teams: by the half each team defends.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

SIGLIP = "google/siglip-base-patch16-224"
GREY = 128


def jersey_crop(frame: np.ndarray, box: tuple[float, float, float, float], mask: np.ndarray | None = None) -> Image.Image:
    """The shirt region of one person: from 15% to 60% of the box height.

    With a mask, pixels outside the person are set to grey so only jersey colours remain.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    top = y1 + 0.15 * (y2 - y1)
    bottom = y1 + 0.60 * (y2 - y1)
    c0, c1 = max(0, int(x1)), min(w, int(np.ceil(x2)))
    r0, r1 = max(0, int(top)), min(h, int(np.ceil(bottom)))
    crop = frame[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)].copy()
    if mask is not None:
        keep = mask[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)]
        crop[~keep] = GREY
    return Image.fromarray(crop)


class Embedder:
    """SigLIP image embeddings (the vision tower only)."""

    def __init__(self, model_id: str = SIGLIP, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel
        from transformers.utils import logging

        logging.set_verbosity_error()  # loading only the vision half lists every text weight as "unexpected"
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = SiglipVisionModel.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for start in range(0, len(images), batch_size):
                inputs = self.processor(images=images[start : start + batch_size], return_tensors="pt").to(self.device)
                out.append(self.model(**inputs).pooler_output.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return np.concatenate(out)


class EmbeddingCollector:
    """A `write_tracks` hook: embeds box and masked jersey crops every `stride` frames.

    Masks only exist while SAM is running, so the masked crops must be made here.
    """

    def __init__(self, clip, embedder: Embedder, stride: int = 5):
        self.frames = {f.index: f for f in clip.frames}
        self.embedder = embedder
        self.stride = stride
        self.frame: list[int] = []
        self.obj_id: list[int] = []
        self.box_crops: list[Image.Image] = []
        self.mask_crops: list[Image.Image] = []

    def __call__(self, frame_index: int, out: dict) -> None:
        if (frame_index - 1) % self.stride:
            return
        image = np.asarray(Image.open(self.frames[frame_index].path).convert("RGB"))
        for obj_id, mask in zip(out["out_obj_ids"], out["out_binary_masks"]):
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
            box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            self.frame.append(frame_index)
            self.obj_id.append(int(obj_id))
            self.box_crops.append(jersey_crop(image, box))
            self.mask_crops.append(jersey_crop(image, box, mask))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frame=np.array(self.frame, dtype=np.int32),
            obj_id=np.array(self.obj_id, dtype=np.int32),
            box=self.embedder(self.box_crops).astype(np.float16),
            mask=self.embedder(self.mask_crops).astype(np.float16),
        )


def cluster_teams(embeddings: np.ndarray, track_ids: np.ndarray, seed: int = 0) -> dict[int, int]:
    """Cluster crops into two teams, then give each track its majority cluster (0 or 1)."""
    import umap
    from sklearn.cluster import KMeans

    n_neighbors = max(2, min(15, len(embeddings) - 1))
    reduced = umap.UMAP(n_components=3, n_neighbors=n_neighbors, random_state=seed).fit_transform(embeddings)
    labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(reduced)
    votes: dict[int, Counter] = {}
    for track, label in zip(track_ids, labels):
        votes.setdefault(int(track), Counter())[int(label)] += 1
    return {track: counter.most_common(1)[0][0] for track, counter in votes.items()}


def name_sides(track_cluster: dict[int, int], track_mean_x: dict[int, float]) -> dict[int, str]:
    """Name clusters as GSR does: the team whose players stand further left (lower x, in metres)
    defends the left half. Returns "left"/"right" per track."""
    xs = {0: [], 1: []}
    for track, cluster in track_cluster.items():
        if track in track_mean_x:
            xs[cluster].append(track_mean_x[track])
    mean0 = np.mean(xs[0]) if xs[0] else 0.0
    mean1 = np.mean(xs[1]) if xs[1] else 0.0
    left = 0 if mean0 <= mean1 else 1
    return {track: ("left" if cluster == left else "right") for track, cluster in track_cluster.items()}
