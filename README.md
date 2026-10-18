# CPQ (v9 final)

Portrait compliance score: ICAO hard gates + soft smile/pose × biometric quality from an external CR-FIQA scalar.

## Layout

```
cpq_main/
  cpq_inference.py          # single / batch entry point
  weights/                  # CPQ v9 checkpoints (6)
  icao-components/          # hat, sunglasses, smile, eyes, gaze, pose
  calibration/              # hard/soft/trunc + bio100(TL/TH)
  requirements.txt
```

## Install

```bash
pip install -r requirements.txt
# InsightFace downloads buffalo_l on first run (~280 MB)
```

## Usage

```bash
# single image — CR scalar required for a non-zero CPQ
python cpq_inference.py photo.jpg --cr 2.10

# batch: one path per line + CSV with path/filename and cr/score
python cpq_inference.py --batch list.txt --cr-csv scores.csv --out-csv out.csv
```

## Score

1. **Truncation** (bbox / kps margins) → hard reject, `cpq=0`
2. **Hard axes** (any ≥ thr) → `reject_hard`, `cpq=0`  
   order: hat → sunglasses → eyes → smile → pose → gaze
3. Else **accept**:  
   `cpq = bio100(CR) × soft(smile) × soft(pose)`  
   soft = 1 below `0.5×hard`, linear to 0 at hard thr  
   `bio100 = clip((CR − 1.0) / (2.33 − 1.0) × 100, 0, 100)`  
   `CR > 2.33` → clipped to 100, flagged `anomaly_cr_gt_TH`

CR-FIQA is **not** inside this package — pass `--cr` / `--cr-csv`.

## Weights (v9 only)

| axis | file |
|------|------|
| hat | `hat_print_robust.pt` |
| sunglasses | `sunglasses_print_robust.pt` |
| smile | `smile_mac_expression_mouth.pt` |
| pose | `pose_print_robust.pt` |
| gaze | `gaze_print_robust_pereye_s96.pt` |
| eyes | `eyes_kaggle_print_robust_mn64_e14.pt` |

Thresholds live in `calibration/icao_axis_calibration.py` and `calibration/biometric_axis_calibration.py`.
