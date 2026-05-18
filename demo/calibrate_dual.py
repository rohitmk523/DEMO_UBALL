#!/usr/bin/env python3
"""
Dual-camera calibration orchestrator (06_DUAL_CAMERA_FUSION.md Step 1, ×2).

Calibrates FL and NL into the SAME shared court space and writes a manifest
that DualCameraFusion consumes. Each camera is calibrated ONCE (cameras are
bolted in place) — reuse the manifest for every game from this rig forever.

Per-camera calibration is delegated to the proven Step-1 tool
(`demo/calibrate_homography.py`), so the interactive clicker / config modes,
the H cache and the sanity overlays are identical to single-camera Step 1.

Typical flow (calibration needs a clean, ideally player-free frame per camera):

  # 1. pull an empty-court frame for each fixed camera (once)
  python demo/extract_frame.py --angle FL --t <empty_t> --out demo/frames/FL_cal.jpg
  python demo/extract_frame.py --angle NL --t <empty_t> --out demo/frames/NL_cal.jpg

  # 2. click each camera's court landmarks (needs a local GUI)
  python demo/calibrate_homography.py --frame demo/frames/FL_cal.jpg --interactive
  python demo/calibrate_homography.py --frame demo/frames/NL_cal.jpg --interactive

  # 3. bind the two calibrations into one reusable manifest
  python demo/calibrate_dual.py --game c2a354fe \
      --fl demo/calibration/FL_cal_calibration.json \
      --nl demo/calibration/NL_cal_calibration.json

Output: demo/calibration/<game>_dual.json  (paths to both per-camera
calibrations + the shared court dims). Feed its `fl`/`nl` entries to
DualCameraFusion.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lib.calibration_integration import CalibrationIntegration  # noqa: E402
from lib import court  # noqa: E402

CALIB_DIR = Path("demo/calibration")


def _validate(tag: str, calib_json: Path) -> None:
    if not calib_json.exists():
        raise SystemExit(f"{tag}: calibration not found: {calib_json} "
                         f"— run demo/calibrate_homography.py for {tag} first")
    ci = CalibrationIntegration(str(calib_json))
    if ci.homography_matrix is None or ci.inverse_homography is None:
        raise SystemExit(f"{tag}: {calib_json} has no usable homography")
    print(f"  {tag}: OK ({calib_json})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True, help="e.g. c2a354fe")
    p.add_argument("--fl", type=Path, required=True,
                   help="FL *_calibration.json from calibrate_homography.py")
    p.add_argument("--nl", type=Path, required=True,
                   help="NL *_calibration.json from calibrate_homography.py")
    a = p.parse_args(argv)

    print(f"Validating dual calibration for game {a.game}:")
    _validate("FL", a.fl)
    _validate("NL", a.nl)

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    out = CALIB_DIR / f"{a.game}_dual.json"
    out.write_text(json.dumps({
        "game": a.game,
        "shared_court_space": {
            "system": "NBA full court, origin at a baseline corner",
            "length_cm": court.COURT_LENGTH_CM,
            "width_cm": court.COURT_WIDTH_CM,
        },
        "cameras": {
            "FL": {"id": 1, "calibration": str(a.fl)},
            "NL": {"id": 2, "calibration": str(a.nl)},
        },
        "note": "Calibrate-once: cameras are fixed; reuse for every game "
                "from this rig. Feed FL/NL calibration paths to "
                "lib.dual_camera_fusion.DualCameraFusion.",
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
