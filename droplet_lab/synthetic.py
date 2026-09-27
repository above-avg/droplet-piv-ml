"""Kinematic surrogates, NOT a Navier-Stokes/evaporation solver.

Camera coordinates: x right, y down. Velocity unit: pixels per frame pair.
Each pair is freshly seeded; pairs in a run share physical nuisance parameters.
"""

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

REGIMES = ("radial", "circulation", "mixed")


@dataclass(frozen=True)
class Config:
    size: int = 128
    particles: int = 1500
    noise: float = 0.025
    particle_sigma: float = 0.9
    amplitude: float = 2.0
    age: float = 0.4
    regime: str = "radial"
    seed: int = 42
    direction: int = 1

    def __post_init__(self):
        if self.size < 64 or self.particles < 1:
            raise ValueError("size must be >=64 and particles must be positive")
        if self.regime not in REGIMES or not 0 <= self.age <= 1:
            raise ValueError("unknown regime or age outside [0,1]")
        if self.noise < 0 or self.particle_sigma <= 0 or self.amplitude < 0:
            raise ValueError("noise/amplitude must be nonnegative; sigma must be positive")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")


def geometry(size):
    y, x = np.mgrid[:size, :size]
    radius = 0.43 * size
    xn, yn = (x - (size - 1) / 2) / radius, (y - (size - 1) / 2) / radius
    return xn, yn, xn**2 + yn**2 < 1


def velocity(cfg):
    x, y, mask = geometry(cfg.size)
    r2 = x*x + y*y
    # Bounded radial surrogate: stronger outward motion at later normalized age.
    ur, vr = x * (0.7 + 0.5*r2), y * (0.7 + 0.5*r2)
    # Smooth planar swirl surrogate; NOT an axisymmetric meridional vortex.
    uc, vc = -y * (1 - 0.4*r2), x * (1 - 0.4*r2)
    radial, swirl = {"radial": (1., 0.), "circulation": (0., 1.), "mixed": (0.65, 0.75)}[cfg.regime]
    scale = cfg.amplitude * (0.7 + 0.6*cfg.age)
    u = scale * (radial*ur + cfg.direction*swirl*uc)
    v = scale * (radial*vr + cfg.direction*swirl*vc)
    return np.where(mask, u, 0), np.where(mask, v, 0), mask


def render(points, brightness, cfg):
    im = np.zeros((cfg.size, cfg.size), dtype=float)
    # Bilinear splatting retains subpixel particle positions before optical blur.
    xx, yy = points[:, 0], points[:, 1]
    ix, iy = np.floor(xx).astype(int), np.floor(yy).astype(int)
    for dx, dy, weight in [(0,0,(1-(xx-ix))*(1-(yy-iy))),
                           (1,0,(xx-ix)*(1-(yy-iy))),
                           (0,1,(1-(xx-ix))*(yy-iy)), (1,1,(xx-ix)*(yy-iy))]:
        valid = (ix+dx >= 0) & (ix+dx < cfg.size) & (iy+dy >= 0) & (iy+dy < cfg.size)
        np.add.at(im, (iy[valid]+dy, ix[valid]+dx), brightness[valid]*weight[valid])
    return gaussian_filter(im, cfg.particle_sigma)


def generate_pair(cfg=Config()):
    rng = np.random.default_rng(cfg.seed)
    u, v, mask = velocity(cfg)
    points = rng.uniform(0, cfg.size-1, (cfg.particles*2, 2))
    center, radius = (cfg.size-1)/2, cfg.size*0.43
    points = points[np.sum((points-center)**2, axis=1) < radius**2][:cfg.particles]
    brightness = rng.uniform(0.6, 1.4, len(points))
    coords = [points[:, 1], points[:, 0]]
    displacement = np.column_stack([map_coordinates(u, coords, order=1), map_coordinates(v, coords, order=1)])
    a, b = render(points, brightness, cfg), render(points+displacement, brightness, cfg)
    norm = max(a.max(), b.max(), 1e-12)
    a, b = a/norm, b/norm
    a = np.clip(a + rng.normal(0, cfg.noise, a.shape), 0, 1)*mask
    b = np.clip(b + rng.normal(0, cfg.noise, b.shape), 0, 1)*mask
    return a, b, u, v, mask
