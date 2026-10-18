"""ICAO presentation hard/soft thresholds and truncation (CPQ final stack).

Hard thr for hat/sunglasses/smile/pose/gaze: curated flagrant+border APCER≤5%.
Eyes: neural closed-eye score (v9 MN64); thr from checkpoint digital EER
(flagrant eyes used MediaPipe EAR historically — not used here).
Soft product terms only on smile + pose (onset_ratio × hard → linear 1→0).
"""

HARD = {
    "hat": 0.92539305,
    "sunglasses": 0.47877000000000003,
    "eyes": 0.629683256149292,  # p_eyes = 1 - min(compliant_L,R); high = closed
    "smile": 0.44678825,  # neutrality / expression
    "pose": 0.95812905,
    "gaze": 0.5531061,
}

# high_is_attack for each hard axis
HIGH_IS_ATTACK = {
    "hat": True,
    "sunglasses": True,
    "eyes": True,
    "smile": True,
    "pose": True,
    "gaze": True,
}

SOFT = {
    "onset_ratio": 0.5,
    "axes": ["smile", "pose"],
}

TRUNCATION = {
    "enabled": True,
    "margin_img": {
        "left": 0.01,
        "right": 0.01,
        "top": 0.01,
        "bottom": 0.02,
    },
    "margin_face_bottom": 0.06,
    "kps_edge": 0.01,
    "sides": ["left", "right", "top", "bottom"],
}

MISSING = {
    "face": "unable_to_assess",
    "iris": "unable_to_assess",  # kept for API parity; neural stack uses face kps
    "action": "reject_quality",
}
