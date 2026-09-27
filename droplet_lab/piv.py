"""Single-pass, linear FFT cross-correlation PIV with subpixel peaks.

No displacement interpolation is hidden in returned results. Rejected vectors
remain NaN; confidence is a heuristic peak ratio, not a calibrated uncertainty.
"""

from dataclasses import dataclass
import numpy as np
from scipy.signal import fftconvolve


@dataclass
class Field:
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    snr: np.ndarray
    eligible: np.ndarray
    valid: np.ndarray


def subpixel(values, index):
    if index == 0 or index == len(values)-1:
        return 0.
    left, peak, right = values[index-1:index+2]
    denom = left - 2*peak + right
    return float(np.clip(0.5*(left-right)/denom, -.5, .5)) if abs(denom) > 1e-12 else 0.


def estimate(a, b, mask=None, window=32, step=16, max_shift=7, min_snr=1.15):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Two finite, equally sized grayscale images are required")
    if window < 8 or window > min(a.shape) or step < 1 or not 1 <= max_shift < window//2:
        raise ValueError("invalid PIV window, step or maximum displacement")
    mask = np.ones_like(a, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != a.shape:
        raise ValueError("mask shape must match images")
    ys, xs = np.arange(0, a.shape[0]-window+1, step), np.arange(0, a.shape[1]-window+1, step)
    x, y = np.meshgrid(xs+(window-1)/2, ys+(window-1)/2)
    u, v = np.full(x.shape, np.nan), np.full(x.shape, np.nan)
    snr, eligible = np.zeros(x.shape), np.zeros(x.shape, dtype=bool)
    taper = np.outer(np.hanning(window), np.hanning(window))
    for j, top in enumerate(ys):
        for i, left in enumerate(xs):
            sl = np.s_[top:top+window, left:left+window]
            if mask[sl].mean() < .85:
                continue
            eligible[j,i] = True
            aa, bb = a[sl], b[sl]
            if min(aa.std(), bb.std()) < 1e-6:
                continue
            aa, bb = (aa-aa.mean())*taper, (bb-bb.mean())*taper
            corr = fftconvolve(bb, aa[::-1, ::-1], mode="full")
            c = window-1
            search = corr[c-max_shift:c+max_shift+1, c-max_shift:c+max_shift+1]
            py, px = np.unravel_index(np.argmax(search), search.shape)
            # A peak at search bounds may indicate an unresolvable displacement.
            if py in (0, 2*max_shift) or px in (0, 2*max_shift):
                continue
            second = search.copy()
            second[max(0,py-2):py+3, max(0,px-2):px+3] = -np.inf
            snr[j,i] = search[py,px]/max(second.max(), 1e-12)
            if snr[j,i] < min_snr:
                continue
            u[j,i] = px-max_shift + subpixel(search[py,:], px)
            v[j,i] = py-max_shift + subpixel(search[:,px], py)
    valid = np.isfinite(u) & np.isfinite(v)
    return Field(x, y, u, v, snr, eligible, valid)


def physical_velocity(field, microns_per_pixel, dt_seconds):
    if microns_per_pixel <= 0 or dt_seconds <= 0:
        raise ValueError("calibration and frame interval must be positive")
    scale = microns_per_pixel / dt_seconds
    return field.u*scale, field.v*scale
