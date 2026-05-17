#!/usr/bin/env python3
"""
Extract a single representative frame from a c2a354fe game video in S3,
without downloading the multi-GB file (presigned URL + ffmpeg HTTP range).

03_DEMO_BUILD_PLAN.md Step 1 helper. Proven working on FL/FR @1080p30.

Usage:
  python demo/extract_frame.py --angle FL --t 1400
  python demo/extract_frame.py --angle FR --t 900 --out demo/frames/x.jpg

Requires: aws CLI with creds, ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

GAME = "c2a354fe"
S3_BASE = (
    "s3://uball-videos-production/court-a/2026-03-19/"
    "c2a354fe-eb34-4980-af00/"
    "2026-03-19_c2a354fe-eb34-4980-af00"
)
VALID_ANGLES = ("FL", "FR", "NL", "NR")


def presign(s3_uri: str, expires: int = 3600) -> str:
    out = subprocess.run(
        ["aws", "s3", "presign", s3_uri, "--expires-in", str(expires)],
        check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def extract(angle: str, t_seconds: float, out_path: Path) -> Path:
    if angle not in VALID_ANGLES:
        raise SystemExit(f"angle must be one of {VALID_ANGLES}, got {angle!r}")
    url = presign(f"{S3_BASE}_{angle}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # input-seek (-ss before -i) = fast, uses HTTP range; one frame, high quality
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", str(t_seconds), "-i", url,
         "-frames:v", "1", "-q:v", "2", str(out_path)],
        check=True,
    )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit(f"ffmpeg produced no frame at t={t_seconds}s")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--angle", default="FL", choices=VALID_ANGLES)
    p.add_argument("--t", type=float, default=1400.0,
                   help="timestamp in seconds (game is ~3597s)")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    out = a.out or Path(f"demo/frames/{GAME}_{a.angle}_t{int(a.t)}.jpg")
    extract(a.angle, a.t, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
