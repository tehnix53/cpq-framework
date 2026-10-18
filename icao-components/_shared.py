"""Shared face geometry, crops, and model helpers for ICAO components."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from insightface.app import FaceAnalysis
from torchvision.models import efficientnet_b0, mobilenet_v3_small

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

_DET = None


def weights_dir() -> Path:
    return WEIGHTS


def get_detector(ctx_id: int = -1) -> FaceAnalysis:
    global _DET
    if _DET is None:
        app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        _DET = app
    return _DET


def read_bgr(path: str | Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def largest_face(bgr: np.ndarray):
    faces = get_detector().get(bgr)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def eye_pair(face) -> tuple[tuple[float, float], tuple[float, float]]:
    kps = face.kps
    return (
        (float(kps[0][0]), float(kps[0][1])),
        (float(kps[1][0]), float(kps[1][1])),
    )


def tensor_imagenet(crop_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def square_crop(bgr, cx, cy, side: float, out: int) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    s = max(2, int(round(side)))
    x0, y0 = cx - side / 2, cy - side / 2
    canvas = np.full((s, s, 3), 114, np.uint8)
    src_x0, src_y0 = int(math.floor(x0)), int(math.floor(y0))
    src_x1, src_y1 = int(math.ceil(x0 + side)), int(math.ceil(y0 + side))
    dst_x0, dst_y0 = max(0, -src_x0), max(0, -src_y0)
    src_x0, src_y0 = max(0, src_x0), max(0, src_y0)
    src_x1, src_y1 = min(w, src_x1), min(h, src_y1)
    copy_h = min(src_y1 - src_y0, s - dst_y0)
    copy_w = min(src_x1 - src_x0, s - dst_x0)
    if copy_h <= 0 or copy_w <= 0:
        return None
    patch = bgr[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
    canvas[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = patch
    interp = cv2.INTER_AREA if s >= out else cv2.INTER_LINEAR
    return cv2.resize(canvas, (out, out), interpolation=interp)


def region_crop(task: str, bgr, face, out_side: int = 160) -> np.ndarray | None:
    box = [float(v) for v in face.bbox]
    eyes = eye_pair(face)
    x1, y1, x2, y2 = box
    fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    (lx, ly), (rx, ry) = eyes
    eye_y = 0.5 * (ly + ry)
    ied = max(8.0, float(np.hypot(rx - lx, ry - ly)))
    mid_x = 0.5 * (lx + rx)
    if task == "hat":
        top = y1 - 0.55 * fh
        bot = eye_y - 0.04 * fh
        side = max(bot - top, 1.05 * fw)
        return square_crop(bgr, cx, bot - side / 2, side, out_side)
    side = 2.35 * ied
    return square_crop(bgr, mid_x, eye_y - 0.22 * ied, side, out_side)


def mouth_crop(bgr, face, out_side: int = 96) -> np.ndarray | None:
    """Same geometry as train_expression_mouth.mouth_crop_side."""
    kps = face.kps
    lx, ly = float(kps[0][0]), float(kps[0][1])
    rx, ry = float(kps[1][0]), float(kps[1][1])
    nx, ny = float(kps[2][0]), float(kps[2][1])
    mlx, mly = float(kps[3][0]), float(kps[3][1])
    mrx, mry = float(kps[4][0]), float(kps[4][1])
    ied = max(8.0, math.hypot(rx - lx, ry - ly))
    mw = max(4.0, math.hypot(mrx - mlx, mry - mly))
    mx, my = 0.5 * (mlx + mrx), 0.5 * (mly + mry)
    top = ny - 0.20 * ied
    bot = my + 0.95 * ied
    side = max(bot - top, 2.20 * mw, 1.15 * ied)
    return square_crop(bgr, mx, 0.5 * (top + bot), side, out_side)


def one_eye(bgr, cx, cy, ied, side: int, above: float = 0.10, side_ied: float = 0.72):
    crop_side = side_ied * ied
    cy = cy - above * crop_side
    return square_crop(bgr, cx, cy, crop_side, side)


def eye_strip(bgr, lx, ly, rx, ry, in_h: int = 64, in_w: int = 192):
    ied = max(8.0, math.hypot(rx - lx, ry - ly))
    cx, cy = 0.5 * (lx + rx), 0.5 * (ly + ry)
    w, h = 2.96 * ied, 0.94 * ied
    x0, y0 = cx - w / 2, cy - 0.30 * ied
    img_h, img_w = bgr.shape[:2]
    ch, cw = max(2, int(round(h))), max(2, int(round(w)))
    canvas = np.full((ch, cw, 3), 114, np.uint8)
    src_x0, src_y0 = int(math.floor(x0)), int(math.floor(y0))
    src_x1, src_y1 = int(math.ceil(x0 + w)), int(math.ceil(y0 + h))
    dst_x0, dst_y0 = max(0, -src_x0), max(0, -src_y0)
    src_x0, src_y0 = max(0, src_x0), max(0, src_y0)
    src_x1, src_y1 = min(img_w, src_x1), min(img_h, src_y1)
    copy_h = min(src_y1 - src_y0, ch - dst_y0)
    copy_w = min(src_x1 - src_x0, cw - dst_x0)
    if copy_h <= 0 or copy_w <= 0:
        return None
    patch = bgr[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
    canvas[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = patch
    interp = cv2.INTER_AREA if (ch >= in_h and cw >= in_w) else cv2.INTER_LINEAR
    return cv2.resize(canvas, (in_w, in_h), interpolation=interp)


class FcMobileNet(nn.Module):
    """Print-robust hat/sunglasses head: forward → (logits, emb), .fc not .classifier."""

    def __init__(self):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        in_f = base.classifier[0].in_features
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_f, 1)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        emb = torch.flatten(x, 1)
        return self.fc(self.dropout(emb)), emb


def make_fc_mobilenet() -> nn.Module:
    return FcMobileNet()


def make_eyes_mobilenet() -> nn.Module:
    """Kaggle MN64 closed-eye: Linear→Hardswish→Dropout→Linear(1)."""
    m = mobilenet_v3_small(weights=None)
    in_f = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_f, in_f),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_f, 1),
    )
    return m


def make_efficientnet() -> nn.Module:
    m = efficientnet_b0(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_f, 1))
    return m


def load_blob(path: Path, device: torch.device) -> dict:
    return torch.load(str(path), map_location=device, weights_only=False)


def sigmoid_logit(model: nn.Module, x: torch.Tensor) -> float:
    with torch.no_grad():
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        return float(torch.sigmoid(out).reshape(-1)[0].cpu())
