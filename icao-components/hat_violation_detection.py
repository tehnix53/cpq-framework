#!/usr/bin/env python3
"""Head-covering (hat) violation probability. Independent of other CPQ axes."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import _shared as S

CKPT = S.weights_dir() / "hat_print_robust.pt"


class HatViolationDetector:
    def __init__(self, weights: Path = CKPT, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        blob = S.load_blob(weights, torch.device("cpu"))
        self.net = S.make_fc_mobilenet()
        self.net.load_state_dict(blob["model"])
        self.net = self.net.to(self.device).eval()
        self.side = int(blob.get("side", 160))

    def score_bgr(self, bgr, face=None) -> dict:
        out = {"p_hat": None, "face_ok": False, "error": ""}
        if bgr is None:
            out["error"] = "read_fail"
            return out
        face = face or S.largest_face(bgr)
        out["face_ok"] = face is not None
        if face is None:
            out["error"] = "no_face"
            return out
        crop = S.region_crop("hat", bgr, face, self.side)
        if crop is None:
            out["error"] = "crop_fail"
            return out
        x = S.tensor_imagenet(crop).unsqueeze(0).to(self.device)
        out["p_hat"] = S.sigmoid_logit(self.net, x)
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
    det = HatViolationDetector(weights=args.weights)
    row = det.score(args.image)
    print(row)
    return 0 if not row.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
