# Experimental data and validation protocol

The image-pair CLI estimates calibrated velocities from grayscale images. Experimental classification requires a separately validated dataset; the included classifier is evaluated on synthetic patterns only.

## Acquisition metadata

`data/experiment_metadata_template.csv` defines fields for run identity, calibration, frame interval, composition, volume, substrate, ambient conditions, illumination and optics. Raw images should retain their original intensities and be stored with checksums. Acquisition conditions and label provenance belong in metadata, separate from measured flow features.

An independent run corresponds to a distinct droplet acquisition. Images from the same run must remain in one partition. Acquisition-day or condition holdouts provide additional checks on generalization when sufficient data are available.

## Calibration and image quality

1. Calibrate length scale using the acquisition optics; verify frame timing and exposure duration.
2. Estimate the background and define the droplet mask. Check focus, saturation, particle visibility and substrate reflections.
3. Validate displacement recovery using known image translations and an independent physical reference where available.
4. Select interrogation windows, displacement limits and rejection thresholds using calibration or training data. Record all settings before evaluating held-out runs.
5. Retain the original vector estimates and rejection masks. Report accepted coverage, displacement distributions and calibration uncertainty alongside velocities.

## Analysis conventions

Coordinates are x right and y down. The experimental CSV reports velocity in micrometres/second; diagnostic arrows represent pixels per image pair. Rejected velocities remain NaN. The correlation peak ratio is a quality heuristic and has not been calibrated as a velocity uncertainty estimate.

The current interface processes a single pair of image planes. Camera calibration, distortion correction, image registration and automatic segmentation are external preprocessing steps. The implementation does not estimate out-of-plane velocity or compensate for particle loss through the measurement plane.

## Experimental model validation

Classification labels require documented independent evidence. Flow geometry alone does not identify thermal, solutal or buoyancy-driven mechanisms. Preprocessing and model selection must use training runs, with final evaluation on untouched runs or conditions.

Report image provenance, independent run counts, acquisition conditions, calibration, PIV settings, rejection rules, vector coverage, split definitions, baselines and tuning procedures. Measurement error and classification error address different failure modes and should be evaluated separately.
