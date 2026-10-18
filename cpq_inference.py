#!/usr/bin/env python3
"""CPQ final inference: ICAO hard/soft + truncation + bio100(CR).

Usage:
  python cpq_inference.py image.jpg --cr 2.10
  python cpq_inference.py --batch list.txt --cr-csv scores.csv

CR-FIQA scalar is required for a non-zero CPQ product (pass --cr or CSV).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "icao-components"))
sys.path.insert(0, str(ROOT / "calibration"))

import biometric_axis_calibration as BIO  # noqa: E402
import icao_axis_calibration as ICAO  # noqa: E402
import _shared as S  # noqa: E402
from eyes_violation_detection import EyesViolationDetector  # noqa: E402
from gaze_violation_detection import GazeViolationDetector  # noqa: E402
from hat_violation_detection import HatViolationDetector  # noqa: E402
from neutrality_violation_detection import NeutralityViolationDetector  # noqa: E402
from pose_violation_detection import PoseViolationDetector  # noqa: E402
from sunglasses_violation_detection import SunglassesViolationDetector  # noqa: E402


def bio100(cr: float) -> float:
    tl = float(BIO.BIOMETRIC["TL"])
    th = float(BIO.BIOMETRIC["TH"])
    return float(np.clip((cr - tl) / (th - tl) * 100.0, 0.0, 100.0))


def soft_score(p: float | None, hard: float, onset_ratio: float) -> float:
    if p is None:
        return 1.0
    onset = onset_ratio * hard
    if p <= onset:
        return 1.0
    if p >= hard:
        return 0.0
    return float((hard - p) / (hard - onset))


def truncation_from_face(face, img_h: int, img_w: int, cfg: dict) -> dict:
    empty = {
        "margin_left": None,
        "margin_right": None,
        "margin_top": None,
        "margin_bottom": None,
        "margin_bottom_fh": None,
        "kps_edge_min": None,
        "truncated": False,
        "trunc_reason": "",
    }
    if face is None or img_h <= 1 or img_w <= 1:
        return empty
    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    x1 = max(0.0, min(float(img_w), x1))
    x2 = max(0.0, min(float(img_w), x2))
    y1 = max(0.0, min(float(img_h), y1))
    y2 = max(0.0, min(float(img_h), y2))
    face_h = max(1.0, y2 - y1)
    m_left = x1 / float(img_w)
    m_right = (float(img_w) - x2) / float(img_w)
    m_top = y1 / float(img_h)
    m_bottom = (float(img_h) - y2) / float(img_h)
    m_bottom_fh = (float(img_h) - y2) / face_h
    kps_edge_min = None
    kps = getattr(face, "kps", None)
    if kps is not None:
        min_d = min(
            min(float(kp[0]), float(kp[1]), float(img_w) - float(kp[0]), float(img_h) - float(kp[1]))
            for kp in kps
        )
        kps_edge_min = min_d / float(min(img_h, img_w))
    out = {
        "margin_left": m_left,
        "margin_right": m_right,
        "margin_top": m_top,
        "margin_bottom": m_bottom,
        "margin_bottom_fh": m_bottom_fh,
        "kps_edge_min": kps_edge_min,
        "truncated": False,
        "trunc_reason": "",
    }
    if not cfg.get("enabled", True):
        return out
    sides = set(cfg.get("sides", ["left", "right", "top", "bottom"]))
    thr_img = cfg.get("margin_img", {})
    side_vals = {"left": m_left, "right": m_right, "top": m_top, "bottom": m_bottom}
    reasons = []
    for side in sides:
        thr = float(thr_img.get(side, 0.01))
        if side_vals[side] < thr:
            reasons.append(f"bbox_{side}<{thr:g}")
    thr_fh = cfg.get("margin_face_bottom")
    if thr_fh is not None and "bottom" in sides and m_bottom_fh < float(thr_fh):
        reasons.append(f"bbox_bottom_fh<{float(thr_fh):g}")
    thr_kps = cfg.get("kps_edge")
    if thr_kps is not None and kps_edge_min is not None and kps_edge_min < float(thr_kps):
        reasons.append(f"kps_edge<{float(thr_kps):g}")
    if reasons:
        out["truncated"] = True
        out["trunc_reason"] = ",".join(reasons)
    return out


def decide(row: dict) -> dict:
    hard_cfg = ICAO.HARD
    soft_cfg = ICAO.SOFT
    onset = float(soft_cfg["onset_ratio"])
    soft_axes = set(soft_cfg.get("axes", ["smile", "pose"]))

    out = dict(row)
    out.setdefault("decision", "accept")
    out.setdefault("unable_to_assess", False)

    if out.get("face_ok") is False:
        out["hard"] = ICAO.MISSING["face"]
        out["decision"] = ICAO.MISSING["action"]
        out["unable_to_assess"] = True
        out["s_mouth"] = out["s_pose"] = out["s_gaze"] = 0.0
        out["cpq"] = 0.0
        return out

    if out.get("truncated"):
        out["hard"] = "trunc"
        out["decision"] = "reject_hard"
        out["s_mouth"] = out["s_pose"] = out["s_gaze"] = 0.0
        out["cpq"] = 0.0
        return out

    hard = ""
    p_hat = out.get("p_hat")
    p_gl = out.get("p_glasses")
    p_eyes = out.get("p_eyes")
    p_sm = out.get("p_smile")
    p_po = out.get("p_pose")
    p_ga = out.get("p_gaze")

    if p_hat is not None and p_hat >= hard_cfg["hat"]:
        hard = "hat"
    elif p_gl is not None and p_gl >= hard_cfg["sunglasses"]:
        hard = "sunglasses"
    elif p_eyes is not None and p_eyes >= hard_cfg["eyes"]:
        hard = "eyes_open"
    elif p_sm is not None and p_sm >= hard_cfg["smile"]:
        hard = "expr"
    elif p_po is not None and p_po >= hard_cfg["pose"]:
        hard = "pose"
    elif p_ga is not None and p_ga >= hard_cfg["gaze"]:
        hard = "gaze"
    out["hard"] = hard

    out["s_mouth"] = (
        soft_score(p_sm, hard_cfg["smile"], onset) if "smile" in soft_axes else 1.0
    )
    out["s_pose"] = (
        soft_score(p_po, hard_cfg["pose"], onset) if "pose" in soft_axes else 1.0
    )
    out["s_gaze"] = (
        soft_score(p_ga, hard_cfg["gaze"], onset) if "gaze" in soft_axes else 1.0
    )

    if hard:
        out["decision"] = "reject_hard"
        out["cpq"] = 0.0
    else:
        out["decision"] = "accept"
        bio = out.get("bio")
        if bio is not None:
            soft = 1.0
            if "smile" in soft_axes:
                soft *= out["s_mouth"]
            if "pose" in soft_axes:
                soft *= out["s_pose"]
            if "gaze" in soft_axes:
                soft *= out["s_gaze"]
            out["cpq"] = float(bio) * soft
        else:
            out["cpq"] = None
    return out


class CPQInference:
    def __init__(self, device: str | None = None):
        self.hat = HatViolationDetector(device=device)
        self.sun = SunglassesViolationDetector(device=device)
        self.smile = NeutralityViolationDetector(device=device)
        self.eyes = EyesViolationDetector(device=device)
        self.gaze = GazeViolationDetector(device=device)
        self.pose = PoseViolationDetector(device=device)

    def score(self, path: str | Path, cr: float | None = None) -> dict:
        path = Path(path)
        bgr = S.read_bgr(path)
        out = {
            "path": str(path),
            "p_hat": None,
            "p_glasses": None,
            "p_smile": None,
            "p_eyes": None,
            "p_gaze": None,
            "p_pose": None,
            "bio": None,
            "cr": cr,
            "anomaly_cr_gt_TH": False,
            "hard": "",
            "decision": "",
            "unable_to_assess": False,
            "cpq": None,
            "error": "",
            "face_ok": False,
        }
        if bgr is None:
            out["error"] = "read_fail"
            out["decision"] = ICAO.MISSING["action"]
            out["unable_to_assess"] = True
            out["cpq"] = 0.0
            return out

        face = S.largest_face(bgr)
        out["face_ok"] = face is not None
        h, w = bgr.shape[:2]
        out.update(truncation_from_face(face, h, w, ICAO.TRUNCATION))

        if face is not None:
            out.update({k: v for k, v in self.hat.score_bgr(bgr, face).items() if k.startswith("p_")})
            out.update({k: v for k, v in self.sun.score_bgr(bgr, face).items() if k.startswith("p_")})
            out.update({k: v for k, v in self.smile.score_bgr(bgr, face).items() if k.startswith("p_")})
            out.update({k: v for k, v in self.eyes.score_bgr(bgr, face).items() if k.startswith("p_")})
            out.update({k: v for k, v in self.gaze.score_bgr(bgr, face).items() if k.startswith("p_")})
            out.update({k: v for k, v in self.pose.score_bgr(bgr, face).items() if k.startswith("p_")})
        else:
            out["error"] = "no_face"

        if cr is not None:
            out["cr"] = float(cr)
            out["bio"] = bio100(float(cr))
            out["anomaly_cr_gt_TH"] = float(cr) > float(BIO.BIOMETRIC["TH"])

        return decide(out)

    def score_batch(
        self, paths: list[str | Path], cr_map: dict[str, float] | None = None
    ) -> list[dict]:
        cr_map = cr_map or {}
        rows = []
        for p in paths:
            key = str(Path(p))
            cr = cr_map.get(key)
            if cr is None:
                cr = cr_map.get(Path(p).name)
            rows.append(self.score(p, cr=cr))
        return rows


def _load_cr_csv(path: Path) -> dict[str, float]:
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            cr = r.get("cr") or r.get("score")
            if cr is None or cr == "":
                continue
            for key in (r.get("path"), r.get("relative_path"), r.get("filename"), Path(r.get("path") or "").name):
                if key:
                    out[str(key)] = float(cr)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", nargs="?", help="single image path")
    p.add_argument("--batch", type=Path, help="text file with one image path per line")
    p.add_argument("--cr", type=float, default=None, help="CR-FIQA scalar for single image")
    p.add_argument("--cr-csv", type=Path, help="CSV with path/filename + cr/score for batch")
    p.add_argument("--out-csv", type=Path, help="write batch results CSV")
    args = p.parse_args()

    eng = CPQInference()
    if args.batch:
        paths = [ln.strip() for ln in args.batch.read_text().splitlines() if ln.strip()]
        cr_map = _load_cr_csv(args.cr_csv) if args.cr_csv else {}
        rows = eng.score_batch(paths, cr_map)
        if args.out_csv:
            fields = sorted({k for r in rows for k in r})
            with args.out_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            print(f"wrote {args.out_csv} n={len(rows)}")
        else:
            for r in rows:
                print(f"{r['path']}: cpq={r.get('cpq')} hard={r.get('hard')!r} decision={r.get('decision')}")
        return 0

    if not args.image:
        p.error("provide image or --batch")
    row = eng.score(args.image, cr=args.cr)
    for k, v in row.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    return 0 if not row.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
