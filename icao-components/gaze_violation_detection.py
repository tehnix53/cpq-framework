#!/usr/bin/env python3
"""Gaze-in-camera violation probability (mean of L/R eye nets). Independent axis."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import _shared as S

CKPT = S.weights_dir() / "gaze_print_robust_pereye_s96.pt"


class GazeViolationDetector:
    def __init__(self, weights: Path = CKPT, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        blob = S.load_blob(weights, torch.device("cpu"))
        self.net = S.make_efficientnet()
        self.net.load_state_dict(blob["model"])
        self.net = self.net.to(self.device).eval()
        self.side = int(blob.get("side", 96))

    def score_bgr(self, bgr, face=None) -> dict:
        out = {"p_gaze": None, "face_ok": False, "error": ""}
        if bgr is None:
            out["error"] = "read_fail"
            return out
        face = face or S.largest_face(bgr)
        out["face_ok"] = face is not None
        if face is None:
            out["error"] = "no_face"
            return out
        (lx, ly), (rx, ry) = S.eye_pair(face)
        ied = float(((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5)
        if ied < 8:
            out["error"] = "ied_small"
            return out
        left = S.one_eye(bgr, lx, ly, ied, self.side)
        right = S.one_eye(bgr, rx, ry, ied, self.side)
        if left is None or right is None:
            out["error"] = "crop_fail"
            return out
        x = torch.stack([S.tensor_imagenet(left), S.tensor_imagenet(right)]).to(self.device)
        with torch.no_grad():
            p = torch.sigmoid(self.net(x)).flatten()
        out["p_gaze"] = float(p.mean().cpu())
        return out

    def score(self, path: str | Path) -> dict:
        bgr = S.read_bgr(path)
        row = self.score_bgr(bgr)
        row["path"] = str(path)
        return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image")
    p.add_argument("--weights", type=Path, default=CKPT)
    args = p.parse_args()
    det = GazeViolationDetector(weights=args.weights)
    row = det.score(args.image)
    print(row)
    return 0 if not row.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
