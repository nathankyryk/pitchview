# PitchView

> Work in progress.

- **Perception:** fine-tuned RF-DETR detection, SAM 3.1 masks and multi-object tracking, team ID, pitch homography.
- **Game state:** every player's team, role and position in metres, frame by frame.
- **Evaluation:** all results are reported on the [SoccerNet Game State Reconstruction](https://github.com/SoccerNet/sn-gamestate) benchmark.

## Setup

```
uv sync
uv run python scripts/download_gsr.py   # ~28 GB: SoccerNet GSR train/valid/test
```
