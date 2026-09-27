"""Interpretable features calculated only from measured PIV vectors."""

import numpy as np

FEATURE_NAMES = ["speed_mean", "speed_std", "speed_p90", "radial_mean", "radial_std",
                 "tangential_abs_mean", "tangential_std", "radial_speed_ratio", "valid_fraction"]


def extract(field, center):
    ok = field.valid
    if ok.sum() < 5:
        raise ValueError("Too few valid PIV vectors (need at least five)")
    dx, dy = field.x[ok]-center[0], field.y[ok]-center[1]
    r = np.maximum(np.hypot(dx,dy), 1e-9)
    u, v = field.u[ok], field.v[ok]
    speed = np.hypot(u,v)
    radial, tangential = (u*dx+v*dy)/r, (-u*dy+v*dx)/r
    return np.array([speed.mean(), speed.std(), np.quantile(speed,.9), radial.mean(),
                     radial.std(), np.abs(tangential).mean(), tangential.std(),
                     radial.mean()/max(speed.mean(),1e-9), ok.sum()/max(field.eligible.sum(),1)])
