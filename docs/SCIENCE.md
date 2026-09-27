# From evaporation physics to observable particle motion

## Physical motivation

A pinned contact line can drive outward liquid replenishment during evaporation and transport suspended material toward the edge. Surface-tension gradients can drive Marangoni stresses; density gradients can drive buoyancy. Flow depends on substrate temperature, composition, geometry, humidity and contact-line behavior. A visible vortex alone does not identify which mechanism generated it.

[Deegan et al. (1997)](https://www.nature.com/articles/39827) describe the capillary-flow origin of ring stains; [Diddens et al.](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/competing-marangoni-and-rayleigh-convection-in-evaporating-binary-droplets/67AEC6D85935B52BB15E85CE6D48180B) examine competing convection in binary droplets. These references motivate the study; their governing equations are not solved here.

## Kinematic model and image formation

The circular camera-plane footprint has radius R = 0.43 N and center (N-1)/2. Let X=(x-xc)/R, Y=(y-yc)/R and q=X²+Y². Inside q<1:

```
radial field       = (X, Y) (0.7 + 0.5 q)
circulation field  = (-Y, X) (1 - 0.4 q)
displacement      = A (0.7 + 0.6 age) [a radial + s b circulation]
```

The weights (a,b) are (1,0), (0,1), and (0.65,0.75). The circulation sign s is randomly ±1. Particles undergo one forward advection step using this displacement, then bilinear splatting, Gaussian blur, shared intensity normalization and independent additive Gaussian noise clipped into [0,1]. The footprint masks the final images.

This is a **kinematic camera-plane surrogate**. It does not conserve liquid volume, solve pressure/temperature/concentration, evolve a free surface or contact angle, model refraction, or reproduce 3D droplet circulation. The radial field has planar divergence. A real planar slice may also have nonzero planar divergence because of out-of-plane flow, but that does not make this generator a physical solution. The age variable changes amplitude only: it is not simulated evaporation time. Different pairs are freshly seeded, not a temporally continuous video. The finite candidate sampler makes particle count a target rather than an exact guarantee.

## Measurement equations

For each interrogation window, subtract the mean and apply a Hann taper. Compute linear cross-correlation using an FFT convolution of image B with reversed image A. The offset of the maximum gives displacement; a three-point parabolic fit estimates fractional pixels. Limit the search to ±7 pixels by default and reject boundary peaks. Compare the main peak with the strongest peak outside a 5×5 exclusion region. The default minimum peak ratio is 1.15.

Windows need 85% mask occupancy; blank windows and low-confidence peaks yield NaN. There is no hidden inpainting or vector smoothing. The peak ratio is a **quality heuristic**, not a confidence interval on velocity. Larger motion gradients, blur or particle loss can bias the displacement even when the peak is strong. For reference, see [OpenPIV basics](https://openpiv.readthedocs.io/en/stable/src/piv_basics.html) and [OpenPIV validation](https://openpiv.readthedocs.io/en/stable/src/generated/openpiv.validation.sig2noise_val.html).

```
u_physical = displacement_x × micrometres_per_pixel / frame_interval_seconds
v_physical = displacement_y × micrometres_per_pixel / frame_interval_seconds
endpoint error = sqrt((u_est-u_true)² + (v_est-v_true)²)
```

Ground truth is sampled at each window center. PIV measures a spatially weighted window motion; comparison to center velocity is an approximation, particularly near boundaries and strong gradients. Mean endpoint error excludes rejected vectors, so always report coverage alongside it.

At each grid point define er=(dx/r,dy/r) and et=(-dy/r,dx/r). Project measured velocity onto these bases to obtain radial and tangential components. The features aggregate these projections and speed. Positive tangential motion is clockwise on the displayed image because y increases downward.

## Classification and evaluation scope

The random forest maps measured feature vectors to generator labels. A majority classifier quantifies the class-balance baseline. The generator explicitly encodes three distinct patterns, so high classification accuracy reflects separability within this model family. The current benchmark does not establish an advantage over a radial/tangential ratio rule. Feature importance is impurity-based and can be biased; it does not establish causal relevance.

All frames for one run stay on the same side of the split. A run-level bootstrap describes variation among held-out synthetic runs, not uncertainty in physical modeling or laboratory transfer. If every held-out run is correct, the bootstrap interval can collapse to [1,1]; this is not proof of perfect generalization.

Before real mechanism inference, obtain experimental labels supported by controlled conditions and independent observations. Fit preprocessing on training runs, reserve independent conditions for validation, and study image quality, tracer response, depth of field, calibration and out-of-plane loss.
