# Court-Map Demo (investor)

Side-by-side video: **left** = live game camera, **right** = top-down CAD
court with a **red dot per player** (YOLO11m person detection → foot point
→ our calibrated homography → court cm → diagram). Dots move as players
move.

## Regenerate

```bash
D="/path/to/2026-05-05_..._FR.mp4"
~/miniconda3/envs/court_tracking/bin/python demo/make_courtmap_demo.py \
  --video "$D" --name FR_cal --t0 1800 --dur 300 --fps 15 \
  --conf 0.40 --roi-top 0.22 --out demo/courtmap_demo_FR_5min.mp4
```

- Frames are deinterlaced (`field=top,scale=1920:1080`) so pixel space
  matches the calibration (`demo/calibration/FR_cal_H.npy`).
- Source video is the local 2026-05-05 FR recording — chosen because
  `FR_cal` was solved on a frame **from that exact camera/session**, so
  the homography matches best. `--name FL_cal` + the FL video also works.
- `.mp4` is git-ignored (size); the rendered file is a delivered
  artifact, not committed.

## What it shows / honest limits

- Mapping is accurate **in the play area** (median 6 cm on FR's own
  calibration points; center/near/key 1–7 cm).
- It is a **single-camera** prototype. One planar homography on a
  non-pinhole action-cam degrades toward the far baseline corners, so
  detections there are filtered (`--roi-top`, on-court bounds, min box
  height) rather than shown wrong.
- Next step for precision: 4-camera court-space **fusion** (see
  `lib/dual_camera_fusion.py` / `cross_camera_merger.py`) so each
  camera contributes only its accurate region.
