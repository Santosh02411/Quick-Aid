"""Automatic photo-quality checks, run BEFORE any AI model sees the image.

A blurry, dark or tiny photo produces confident-sounding nonsense from any model, so we
measure the basics ourselves first:
  * resolution          - is there enough detail at all?
  * sharpness           - variance of the Laplacian (a standard focus measure)
  * exposure            - too dark / too bright / glare
  * contrast            - flat, featureless images

The result is used three ways: an unusable photo gets a "please retake" response without
calling the model at all; a weak photo lowers the reliability rating and caps confidence;
and the measurements are shown to the user so they understand WHY.
"""

from typing import Dict, List

import numpy as np
from PIL import Image

MAX_ANALYSIS_SIDE = 512

# thresholds (tuned on 512px-wide copies of the image)
MIN_SIDE_UNUSABLE = 80
MIN_SIDE_LOW = 240
BLUR_VERY = 4.0      # Laplacian variance below this: very blurry / featureless
BLUR_SLIGHT = 12.0   # below this: slightly soft
# DARK_UNUSABLE / DARK_LOW: an absolute brightness threshold is a structurally imperfect proxy for "not enough light
# reached the camera" because different skin tones reflect different amounts of light even under IDENTICAL lighting -
# the system sees only one photo, with no baseline to compare against, so it cannot distinguish "well-lit dark skin"
# from "dim room" by brightness alone. Lowered from the original 25/55 so a NATURALLY dark-skinned subject in good
# light has real headroom before either check fires (measured: at the original thresholds, the darkest realistic
# tone tested had essentially zero headroom - any dimming at all crossed DARK_LOW; at these thresholds it needs
# ~32% dimming, versus ~80%+ needed for a light-skinned subject to cross the same threshold). That remaining gap is
# real and not fully closable by threshold-tuning alone - see tests/test_image_quality_fairness.py, which measures
# and documents the residual disparity rather than claiming it is eliminated.
DARK_UNUSABLE = 20.0
DARK_LOW = 38.0
BRIGHT_UNUSABLE = 240.0
GLARE_FRACTION = 0.30
# Contrast measured RELATIVE to the image's own brightness (see relative_contrast in assess()), not raw std: a photo
# of naturally darker skin has a lower average pixel value than one of lighter skin under the SAME light, so a fixed
# absolute std threshold flagged "low contrast" for darker-skinned subjects' photos far more readily even when the
# lighting and actual detail were identical. This measure is invariant to uniform dimming regardless of base tone
# (checked directly), so it no longer compounds the darkness disparity above on top of itself.
FLAT_CONTRAST_UNUSABLE = 3.5
LOW_CONTRAST = 10.5

ADVICE = {
    'too_small': 'Use the camera at full size (not a thumbnail or a cropped screenshot).',
    'low_resolution': 'Move a little closer so the area fills most of the frame.',
    'very_blurry': 'Hold the phone steady, tap the screen to focus on the area, and retake it.',
    'slightly_blurry': 'Tap the screen to focus on the area and keep your hand steady.',
    'too_dark': 'Retake in bright, even light (daylight near a window works well).',
    'dark': 'Add more light, for example by facing a window or turning on a lamp.',
    'overexposed': 'Move away from direct light or the flash to avoid glare and washed-out colours.',
    'glare': 'Tilt the camera slightly to avoid reflections and glare on the skin.',
    'blank': 'The image looks blank or uniform. Make sure the affected area is in the frame.',
    'low_contrast': 'The image looks flat. Try better lighting and a plain background.',
}


def _gray(arr: np.ndarray) -> np.ndarray:
    a = arr[..., :3].astype(np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


def assess(image: Image.Image) -> Dict:
    """Return {'rating', 'score', 'unusable', 'issues', 'advice', 'metrics'}."""
    rgb = np.asarray(image.convert('RGB'))
    h, w = rgb.shape[:2]
    step = max(1, max(h, w) // MAX_ANALYSIS_SIDE)
    gray = _gray(rgb[::step, ::step])

    mean, std = float(gray.mean()), float(gray.std())
    # Contrast relative to the image's own brightness, not raw std: a photo of naturally darker skin has a lower
    # average pixel value than one of lighter skin under THE SAME light, so a fixed absolute std threshold flags
    # darker-skinned subjects' photos as "low contrast" far more readily even when the lighting and detail are
    # identical (checked directly: under identical moderate dimming, raw std fired "low contrast" for medium/tan/
    # brown/dark skin tones but not light ones, even though every tone lost the same proportion of its range).
    # Dividing by brightness makes this scale with the subject's own tone instead of penalising darker skin twice.
    relative_contrast = std / max(mean, 1.0) * 128
    sharp = laplacian_variance(gray)
    bright_frac = float((gray > 245).mean())

    unusable = False
    poor: List[str] = []      # problems serious enough to make the photo poor
    fair: List[str] = []      # problems that weaken it
    keys: List[str] = []

    def add(bucket, key, text):
        bucket.append(text)
        keys.append(key)

    side = min(h, w)
    if side < MIN_SIDE_UNUSABLE:
        unusable = True
        add(poor, 'too_small', f'The photo is very small ({w}×{h} px).')
    elif side < MIN_SIDE_LOW:
        add(fair, 'low_resolution', f'Low resolution ({w}×{h} px).')

    if relative_contrast < FLAT_CONTRAST_UNUSABLE:
        unusable = True
        add(poor, 'blank', 'The image is blank or almost one flat colour.')
    else:
        if sharp < BLUR_VERY:
            add(poor, 'very_blurry', 'The photo is very blurry or has no visible detail.')
        elif sharp < BLUR_SLIGHT:
            add(fair, 'slightly_blurry', 'The photo is slightly soft or out of focus.')
        if relative_contrast < LOW_CONTRAST:
            add(fair, 'low_contrast', 'The image has very low contrast.')

    if mean < DARK_UNUSABLE:
        unusable = True
        add(poor, 'too_dark', 'The photo is far too dark to see anything.')
    elif mean < DARK_LOW:
        add(fair, 'dark', 'The photo is quite dark.')
    if mean > BRIGHT_UNUSABLE:
        unusable = True
        add(poor, 'overexposed', 'The photo is washed out by too much light.')
    elif bright_frac > GLARE_FRACTION:
        add(fair, 'glare', 'Part of the photo is overexposed or has glare.')

    rating = 'poor' if (unusable or poor) else 'fair' if fair else 'good'
    score = max(0, 100 - 45 * len(poor) - 15 * len(fair))
    if unusable:
        score = min(score, 15)

    seen, advice = set(), []
    for k in keys:
        if k in ADVICE and ADVICE[k] not in seen:
            seen.add(ADVICE[k])
            advice.append(ADVICE[k])

    return {
        'rating': rating,
        'score': int(score),
        'unusable': unusable,
        'issues': poor + fair,
        'advice': advice,
        'metrics': {'width': int(w), 'height': int(h), 'brightness': round(mean, 1), 'contrast': round(std, 1), 'relative_contrast': round(relative_contrast, 1),
                    'sharpness': round(sharp, 1), 'glare_fraction': round(bright_frac, 3)},
    }


def worst(a: str, b: str) -> str:
    order = {'good': 0, 'fair': 1, 'poor': 2}
    return a if order.get(a, 1) >= order.get(b, 1) else b
