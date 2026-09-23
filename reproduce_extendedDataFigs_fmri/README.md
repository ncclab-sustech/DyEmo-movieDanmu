# Extended Data Figs — fMRI emotion / PCA encoding analysis

Reproduction package for the fMRI encoding analyses of six emotion dimensions
and three PCA components. Two notebooks, one per figure.

## Data

- `data/cache_signals_nowhiten_smooth_fwhm6_unit.npy` — BOLD time series,
  `3595 × 43783` voxels, float32.
- `data/preds_smoothed_danmu.npy` — six emotion time series.
- `data/design_danmu_pca3.npy` — three PCA components (Polarity, Complexity,
  Intensity).
- `data/epi_gm_mask.nii.gz` — gray-matter mask; `data/fsaverage5/` — surface
  meshes for plotting; `data/provenance/` — audit metadata for cross-checks.

Both notebooks find the package root automatically via `data/`, so no absolute
paths are needed. Outputs go to `results/`.

## Extended Data Fig. 1 — `reproduce_extendedFig1_joint_6d_encoding.ipynb`

Builds a six-column design matrix from the emotion series (per-column
transform / smoothing / lag), fits a joint 6-emotion OLS model at every voxel,
and tests significance with 2,000 circular-shift permutations followed by
whole-brain BH-FDR (`q < 0.05`). Significant voxels are projected onto
fsaverage5 to render the surface figure. Pure NumPy/CPU, finishes in a few
minutes; each run writes to `results/<run_id>/`.

## Extended Data Fig. 2 — `reproduce_extendedFig2_separate_lag.ipynb`

For each of the six emotions (and each of the three PCA components) scans 13
candidate lags (−12 s to +12 s), selects the lag with the most significant
voxels, and validates the selection with 2,000 raw-first permutations that
replay the full lag scan, plus Holm correction across dimensions. Produces the
per-dimension R² / optimal-lag maps and the final six-emotion and PCA surface
figures. PyTorch; CUDA GPU recommended for a full rerun (saved outputs from a
completed run are included).
