"""Biometric quality (CR-FIQA) → bio100 mapping.

CR scalar is provided externally. Values above TH are clipped to 100 and are
candidates for digital/synthetic origin anomaly (not “better than 100”).
"""

# print / scan document ceiling (TL=1.0, TH=2.33)
BIOMETRIC = {
    "TL": 1.0,
    "TH": 2.33,
    "formula": "bio100 = clip((cr - TL) / (TH - TL) * 100, 0, 100)",
    "cr_gt_TH": "anomaly_candidate_digital_or_oversized_studio",
}
