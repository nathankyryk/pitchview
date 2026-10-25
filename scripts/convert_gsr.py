"""Convert SoccerNet GSR labels to COCO format for RF-DETR training.

Usage:
    uv run python scripts/convert_gsr.py
    uv run python scripts/convert_gsr.py --splits valid --stride 10
"""

import argparse
from pathlib import Path

from pitchview.coco import convert_split
from pitchview.gsr import DEFAULT_ROOT, SPLITS


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SoccerNet GSR labels to COCO for RF-DETR.")
    parser.add_argument("--src", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=Path("data/rfdetr_gsr"))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--stride", type=int, default=5, help="keep every Nth frame (neighbours are near-duplicates)")
    args = parser.parse_args()

    for split in args.splits:
        stats = convert_split(args.src, split, args.out / split, args.stride)
        print(f"{split}: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))


if __name__ == "__main__":
    main()
