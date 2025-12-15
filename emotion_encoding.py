from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets, image, plotting, surface
from nilearn.maskers import NiftiMasker
from scipy import stats
from scipy.ndimage import label


def mean_lag_smooth_center(raw_signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Replicate MATLAB mean_lag_smooth_center: lag, moving average, recenter."""
    arr = np.asarray(raw_signal, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    mean_signal = arr.mean(axis=1)
    lag = np.concatenate(([0.0], mean_signal[:-1]))
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(lag, kernel, mode="valid")
    centered = smoothed - smoothed.mean()
    return centered.astype(np.float32)


def load_behavioral_design(raw_mat_path: Path, window: int = 5) -> np.ndarray:
    """Return a 6-column emotion design matrix aligned to fMRI (3595 x 6)."""
    with h5py.File(raw_mat_path, "r") as f:
        subs = [f[key][()].T.astype(np.float32) for key in sorted(f.keys())]
    raw = np.stack(subs, axis=2)  # (3599, 6, 12)
    processed = [
        mean_lag_smooth_center(raw[:, dim_index, :], window=window)
        for dim_index in range(raw.shape[1])
    ]
    design = np.column_stack(processed)
    return design


def load_danmu_design(npy_path: Path, window: int = 5) -> np.ndarray:
    """Return a 6-column Danmu-based design matrix aligned to fMRI (3595 x 6)."""
    raw = np.load(str(npy_path)).astype(np.float32)  # (timepoints, 6)
    processed = [
        mean_lag_smooth_center(raw[:, dim_index], window=window)
        for dim_index in range(raw.shape[1])
    ]
    design = np.column_stack(processed)
    return design


def zscore_columns(matrix: np.ndarray) -> np.ndarray:
    """Z-score columns while guarding against zero-variance."""
    matrix = matrix.astype(np.float32)
    matrix -= matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return matrix / std


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction for a 1D array."""
    p = np.asarray(p_values, dtype=np.float64)
    nan_mask = ~np.isfinite(p)
    clean = p[~nan_mask]
    n = clean.size
    if n == 0:
        result = np.ones_like(p, dtype=np.float32)
        result[nan_mask] = 1.0
        return result
    order = np.argsort(clean)
    ranked = clean[order]
    q = np.empty_like(ranked)
    cumulative = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        cumulative = min(cumulative, ranked[idx] * n / rank)
        q[idx] = cumulative
    q_values = np.ones_like(p, dtype=np.float64)
    q_values[~nan_mask] = q[np.argsort(order)]
    q_values = np.clip(q_values, 0.0, 1.0)
    q_values[np.isnan(q_values)] = 1.0
    return q_values.astype(np.float32)


def make_progress_updater(total: int, label: str, enabled: bool):
    """Return a callable that prints ASCII progress when enabled."""
    if not enabled or total <= 0:
        return lambda *_: None

    bar_width = 40
    start_time = time.time()

    def update(current: int) -> None:
        pct = (current + 1) / total
        filled = int(bar_width * pct)
        bar = "#" * filled + "." * (bar_width - filled)
        elapsed = time.time() - start_time
        remaining = (elapsed / max(pct, 1e-6)) * (1 - pct)
        sys.stdout.write(
            f"\r[{label}] |{bar}| {pct * 100:6.2f}% ({current + 1}/{total}) "
            f"ETA {remaining:6.1f}s"
        )
        sys.stdout.flush()
        if current + 1 == total:
            sys.stdout.write("\n")

    return update


def iaaft_surrogate(signal: np.ndarray, max_iter: int = 100, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate an IAAFT surrogate that preserves amplitude distribution and power spectrum."""
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(signal, dtype=np.float64).copy()
    if x.ndim != 1:
        raise ValueError("IAAFT expects a 1D array.")
    sorted_vals = np.sort(x)
    target_fft = np.fft.rfft(x)
    target_amp = np.abs(target_fft)

    # initial random shuffle maintains histogram
    surrogate = rng.permutation(x)
    prev = surrogate.copy()

    for _ in range(max_iter):
        # enforce power spectrum
        s_fft = np.fft.rfft(surrogate)
        phases = np.exp(1j * np.angle(s_fft))
        surrogate = np.fft.irfft(target_amp * phases, n=x.size)
        # enforce amplitude distribution via rank ordering
        order = np.argsort(surrogate)
        surrogate[order] = sorted_vals
        # stopping criterion
        if np.allclose(surrogate, prev, rtol=1e-6, atol=1e-6):
            break
        prev = surrogate.copy()
    return surrogate.astype(np.float32)


def run_iaaft_permutation_test(
    raw_design: np.ndarray,
    signals: np.ndarray,
    observed_r2: np.ndarray,
    n_perm: int,
    max_iter: int = 100,
    seed: int | None = None,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute permutation p/q values using IAAFT surrogates."""
    rng = np.random.default_rng(seed)
    n_timepoints, n_dims = raw_design.shape
    n_voxels = observed_r2.size

    counts = np.ones(n_voxels, dtype=np.int64)  # add-one smoothing
    progress = make_progress_updater(n_perm, "IAAFT permutation", show_progress)
    for idx in range(n_perm):
        surrogate_cols = [
            iaaft_surrogate(raw_design[:, dim], max_iter=max_iter, rng=rng)
            for dim in range(n_dims)
        ]
        surrogate_design = np.column_stack(surrogate_cols).reshape(n_timepoints, n_dims)
        surrogate_design = zscore_columns(surrogate_design)
        perm_r2 = fit_encoding_model(surrogate_design, signals)["r2"]
        counts += perm_r2 >= observed_r2
        progress(idx)
    p_perm = counts.astype(np.float32) / float(n_perm + 1)
    q_perm = benjamini_hochberg(p_perm)
    return p_perm.astype(np.float32), q_perm.astype(np.float32)


def run_circular_shift_permutation_test(
    raw_design: np.ndarray,
    signals: np.ndarray,
    observed_r2: np.ndarray,
    n_perm: int,
    min_shift: int = 30,
    seed: int | None = None,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Permutation test via global circular shifts of the design matrix."""
    rng = np.random.default_rng(seed)
    n_timepoints = raw_design.shape[0]
    counts = np.ones(observed_r2.size, dtype=np.int64)
    progress = make_progress_updater(n_perm, "Shift permutation", show_progress)
    for idx in range(n_perm):
        while True:
            shift = int(rng.integers(1, n_timepoints))
            if abs(shift) >= min_shift:
                if rng.random() < 0.5:
                    shift = -shift
                break
        surrogate_design = np.roll(raw_design, shift=shift, axis=0)
        surrogate_design = zscore_columns(surrogate_design)
        perm_r2 = fit_encoding_model(surrogate_design, signals)["r2"]
        counts += perm_r2 >= observed_r2
        progress(idx)
    p_perm = counts.astype(np.float32) / float(n_perm + 1)
    q_perm = benjamini_hochberg(p_perm)
    return p_perm.astype(np.float32), q_perm.astype(np.float32)


def bootstrap_r2_standard_error(
    raw_design: np.ndarray,
    signals: np.ndarray,
    n_bootstrap: int,
    seed: int | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Estimate voxel-wise R² standard error via bootstrap resampling."""
    rng = np.random.default_rng(seed)
    n_timepoints = raw_design.shape[0]
    n_voxels = signals.shape[1]
    mean = np.zeros(n_voxels, dtype=np.float64)
    m2 = np.zeros(n_voxels, dtype=np.float64)

    progress = make_progress_updater(n_bootstrap, "Bootstrap", show_progress)
    for idx in range(1, n_bootstrap + 1):
        sample_idx = rng.integers(0, n_timepoints, size=n_timepoints)
        design_sample = zscore_columns(raw_design[sample_idx, :])
        signal_sample = signals[sample_idx, :]
        r2 = fit_encoding_model(design_sample, signal_sample)["r2"].astype(np.float64)

        delta = r2 - mean
        mean += delta / idx
        delta2 = r2 - mean
        m2 += delta * delta2
        progress(idx - 1)

    denom = max(1, n_bootstrap - 1)
    se = np.sqrt(np.maximum(m2 / denom, 0.0)).astype(np.float32)
    return se


def fit_encoding_model(design: np.ndarray, signals: np.ndarray) -> dict[str, np.ndarray]:
    """OLS encoding model with implicit intercept (demeaned X/Y) across all voxels."""
    X = design.astype(np.float32)
    Y = signals.astype(np.float32)

    X -= X.mean(axis=0, keepdims=True)
    Y -= Y.mean(axis=0, keepdims=True)

    xtx = X.T @ X
    xtx_inv = np.linalg.inv(xtx)
    xty = X.T @ Y
    betas = xtx_inv @ xty

    y_hat = X @ betas
    residuals = Y - y_hat
    ss_res = np.sum(residuals * residuals, axis=0)
    ss_tot = np.sum(Y * Y, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - (ss_res / ss_tot)
    r2 = np.nan_to_num(r2, nan=0.0, posinf=0.0, neginf=0.0)

    dof_model = X.shape[1]
    dof_resid = X.shape[0] - dof_model - 1
    r2_clip = np.clip(r2, 0.0, 0.999999)
    f_stat = (r2_clip / dof_model) / ((1.0 - r2_clip) / dof_resid)
    p_values = 1.0 - stats.f.cdf(f_stat, dof_model, dof_resid)
    p_values = np.where(r2 > 0, p_values, 1.0)
    q_values = benjamini_hochberg(p_values)

    return {
        "betas": betas.T.astype(np.float32),
        "r2": r2.astype(np.float32),
        "f_stat": f_stat.astype(np.float32),
        "p_values": p_values.astype(np.float32),
        "q_values": q_values.astype(np.float32),
        "ss_res": ss_res.astype(np.float32),
        "ss_tot": ss_tot.astype(np.float32),
    }


def save_nifti_outputs(
    masker: NiftiMasker,
    fmri_img: nib.Nifti1Image,
    results: dict[str, np.ndarray],
    output_prefix: Path,
) -> dict[str, str]:
    """Persist beta and statistical maps to disk."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    coeffs_img = masker.inverse_transform(results["betas"].T)
    coeffs_img.set_sform(fmri_img.affine)
    coeffs_path = Path(f"{output_prefix}_coefficients.nii.gz")
    nib.save(coeffs_img, str(coeffs_path))

    stats_array = np.vstack(
        [
            results["r2"],
            1.0 - results["p_values"],
            1.0 - results["q_values"],
        ]
    )
    stats_img = masker.inverse_transform(stats_array)
    stats_img.set_sform(fmri_img.affine)
    stats_path = Path(f"{output_prefix}_stats.nii.gz")
    nib.save(stats_img, str(stats_path))

    q_map_img = masker.inverse_transform(results["q_values"])
    q_map_img.set_sform(fmri_img.affine)
    q_path = Path(f"{output_prefix}_qvals.nii.gz")
    nib.save(q_map_img, str(q_path))

    return {
        "coefficients_map": str(coeffs_path),
        "stats_map": str(stats_path),
        "q_map": str(q_path),
    }


def summarise_clusters(
    masker: NiftiMasker,
    q_values: np.ndarray,
    r2_values: np.ndarray,
    atlas_cache: Path,
    alpha: float = 0.01,
    r2_threshold: float = 0.01,
    max_clusters: int | None = 25,
) -> list[dict[str, object]]:
    """Summarise significant clusters using the Harvard-Oxford cortical atlas."""
    mask_img = masker.mask_img_
    mask_data = mask_img.get_fdata().astype(bool)
    sig_flat = (
        (q_values < alpha)
        & (r2_values >= r2_threshold)
        & np.isfinite(q_values)
        & np.isfinite(r2_values)
    )
    if sig_flat.sum() == 0:
        return []

    sig_map = np.zeros(mask_data.shape, dtype=bool)
    sig_map[mask_data] = sig_flat

    r2_map = np.zeros(mask_data.shape, dtype=np.float32)
    r2_map[mask_data] = r2_values

    labeled_map, n_clusters = label(sig_map)
    if n_clusters == 0:
        return []

    atlas = datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm", data_dir=str(atlas_cache)
    )
    atlas_img = (
        atlas.maps
        if isinstance(atlas.maps, nib.Nifti1Image)
        else nib.load(atlas.maps)
    )
    atlas_resampled = image.resample_to_img(
        atlas_img, mask_img, interpolation="nearest"
    )
    atlas_data = atlas_resampled.get_fdata()
    atlas_labels = atlas.labels

    summaries: list[dict[str, object]] = []
    affine = mask_img.affine

    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labeled_map == cluster_id
        voxel_indices = np.where(cluster_mask)
        if voxel_indices[0].size == 0:
            continue
        cluster_r2 = r2_map[voxel_indices]
        peak_idx = np.argmax(cluster_r2)
        peak_voxel = (
            int(voxel_indices[0][peak_idx]),
            int(voxel_indices[1][peak_idx]),
            int(voxel_indices[2][peak_idx]),
        )
        peak_coord = nib.affines.apply_affine(affine, peak_voxel)
        atlas_value = int(np.round(atlas_data[peak_voxel]))
        label_name = (
            atlas_labels[atlas_value]
            if 0 <= atlas_value < len(atlas_labels)
            else "Unknown"
        )

        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "voxel_count": int(cluster_r2.size),
                "peak_r2": float(cluster_r2[peak_idx]),
                "mean_r2": float(cluster_r2.mean()),
                "peak_voxel_index": peak_voxel,
                "peak_coord_mni": [float(x) for x in peak_coord],
                "atlas_label": str(label_name),
            }
        )

    summaries.sort(key=lambda item: item["peak_r2"], reverse=True)
    if max_clusters is not None:
        summaries = summaries[:max_clusters]
    return summaries


def plot_surface_maps(
    nifti_img: nib.Nifti1Image,
    output_dir: Path,
    prefix: str,
    mesh: str = "fsaverage5",
    threshold: float | None = None,
) -> dict[str, str]:
    """Project significant R² onto fsaverage surfaces and render four views."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fsavg = datasets.fetch_surf_fsaverage(mesh)

    tex_kwargs = dict(radius=5.0, interpolation="nearest")
    left_tex = surface.vol_to_surf(nifti_img, fsavg.pial_left, **tex_kwargs)
    right_tex = surface.vol_to_surf(nifti_img, fsavg.pial_right, **tex_kwargs)

    left_tex = np.asarray(left_tex, dtype=float)
    right_tex = np.asarray(right_tex, dtype=float)
    if threshold is not None:
        left_tex[left_tex < threshold] = np.nan
        right_tex[right_tex < threshold] = np.nan
    left_tex[~np.isfinite(left_tex)] = np.nan
    right_tex[~np.isfinite(right_tex)] = np.nan
    if not np.isfinite(left_tex).any():
        left_tex = np.zeros_like(left_tex)
    if not np.isfinite(right_tex).any():
        right_tex = np.zeros_like(right_tex)

    left_path = output_dir / f"{prefix}_left_texture.npy"
    right_path = output_dir / f"{prefix}_right_texture.npy"
    np.save(left_path, left_tex)
    np.save(right_path, right_tex)

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    view_specs = [
        ("left", "lateral", left_tex),
        ("left", "medial", left_tex),
        ("right", "lateral", right_tex),
        ("right", "medial", right_tex),
    ]
    for idx, (hemi, view, texture) in enumerate(view_specs, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        plotting.plot_surf_stat_map(
            surf_mesh=getattr(fsavg, f"infl_{hemi}"),
            stat_map=texture,
            hemi=hemi,
            view=view,
            colorbar=True if idx in (1, 3) else False,
            cmap="inferno",
            bg_map=getattr(fsavg, f"sulc_{hemi}"),
            bg_on_data=True,
            figure=fig,
            axes=ax,
            title=f"{hemi.capitalize()} {view}",
        )
        ax.set_axis_off()

    fig_path = output_dir / f"{prefix}_fsaverage.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "left_texture": str(left_path),
        "right_texture": str(right_path),
        "figure": str(fig_path),
    }

def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.atlas_cache.mkdir(parents=True, exist_ok=True)

    fmri_img = nib.load(str(args.fmri))
    masker = NiftiMasker(mask_img=str(args.mask), standardize=False, dtype="float32")
    signals = masker.fit_transform(fmri_img)

    ratings_path = args.ratings
    if ratings_path.suffix.lower() == ".npy":
        if args.design_preprocessed:
            design_raw = np.load(ratings_path).astype(np.float32)
        else:
            design_raw = load_danmu_design(ratings_path, window=args.window)
    else:
        design_raw = load_behavioral_design(ratings_path, window=args.window)
    design = zscore_columns(design_raw)
    results = fit_encoding_model(design, signals)

    significance_method = "parametric"
    permutation_meta: dict[str, object] | None = None
    if args.n_perm and args.n_perm > 0:
        if args.perm_method == "iaaft":
            p_perm, q_perm = run_iaaft_permutation_test(
                raw_design=design_raw,
                signals=signals,
                observed_r2=results["r2"],
                n_perm=args.n_perm,
                max_iter=args.iaaft_iterations,
                seed=args.perm_seed,
                show_progress=args.show_progress,
            )
        else:
            p_perm, q_perm = run_circular_shift_permutation_test(
                raw_design=design_raw,
                signals=signals,
                observed_r2=results["r2"],
                n_perm=args.n_perm,
                min_shift=args.min_shift,
                seed=args.perm_seed,
                show_progress=args.show_progress,
            )
        results["p_values_parametric"] = results["p_values"]
        results["q_values_parametric"] = results["q_values"]
        results["p_values"] = p_perm
        results["q_values"] = q_perm
        significance_method = f"{args.perm_method}_permutation"
        permutation_meta = {
            "n_perm": int(args.n_perm),
            "iaaft_iterations": int(args.iaaft_iterations),
            "min_shift": int(args.min_shift),
            "seed": None if args.perm_seed is None else int(args.perm_seed),
        }

    prefix = args.output_dir / "human_encoding"
    map_paths = save_nifti_outputs(masker, fmri_img, results, prefix)

    sig_mask = (results["q_values"] < args.alpha) & np.isfinite(results["r2"])
    sig_r2 = np.zeros_like(results["r2"], dtype=np.float32)
    sig_r2[sig_mask] = results["r2"][sig_mask]
    sig_r2_img = masker.inverse_transform(sig_r2)
    sig_r2_img.set_sform(fmri_img.affine)
    sig_r2_path = args.output_dir / "human_encoding_r2_sig.nii.gz"
    nib.save(sig_r2_img, str(sig_r2_path))

    cluster_summary = summarise_clusters(
        masker,
        q_values=results["q_values"],
        r2_values=results["r2"],
        atlas_cache=args.atlas_cache,
        alpha=args.alpha,
        r2_threshold=args.r2_threshold,
        max_clusters=args.max_clusters,
    )

    surf_payload = plot_surface_maps(
        sig_r2_img,
        output_dir=args.output_dir,
        prefix="human_encoding_r2_sig",
        mesh=args.fsaverage_mesh,
        threshold=args.surface_threshold,
    )

    r2_se = None
    if args.n_bootstrap and args.n_bootstrap > 0:
        r2_se = bootstrap_r2_standard_error(
            raw_design=design_raw,
            signals=signals,
            n_bootstrap=args.n_bootstrap,
            seed=args.bootstrap_seed,
            show_progress=args.show_progress,
        )

    summary = {
        "maps": map_paths,
        "significant_r2_map": str(sig_r2_path),
        "cluster_summary": cluster_summary,
        "alpha": args.alpha,
        "r2_threshold": args.r2_threshold,
        "surface_plot": surf_payload,
        "significance_method": significance_method,
    }
    if permutation_meta is not None:
        summary["permutation"] = permutation_meta
    if r2_se is not None:
        se_img = masker.inverse_transform(r2_se)
        se_img.set_sform(fmri_img.affine)
        se_path = args.output_dir / "human_encoding_r2_se.nii.gz"
        nib.save(se_img, str(se_path))
        summary["r2_standard_error_map"] = str(se_path)

    json_path = args.output_dir / "human_encoding_summary.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emotion encoding analysis with fsaverage surface visualisation."
    )
    parser.add_argument(
        "--fmri",
        default=Path("osfstorage-archive/fMRI_data/average_epi_nonlin_unsmooth.nii.gz"),
        type=Path,
        help="4D fMRI timeseries in MNI space.",
    )
    parser.add_argument(
        "--mask",
        default=Path("osfstorage-archive/fMRI_data/epi_gm_mask.nii.gz"),
        type=Path,
        help="Gray-matter mask NIfTI.",
    )
    parser.add_argument(
        "--ratings",
        default=Path("osfstorage-archive/behavioral_ratings/raw_ratings.mat"),
        type=Path,
        help="Six-dimensional emotion ratings (MAT file).",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("analysis/results_human"),
        type=Path,
        help="Output directory for maps and figures.",
    )
    parser.add_argument(
        "--atlas-cache",
        default=Path("analysis/atlas_cache"),
        type=Path,
        help="Cache directory for nilearn atlas downloads.",
    )
    parser.add_argument(
        "--alpha",
        default=0.01,
        type=float,
        help="Benjamini-Hochberg FDR threshold.",
    )
    parser.add_argument(
        "--r2-threshold",
        dest="r2_threshold",
        default=0.01,
        type=float,
        help="Minimum R² when summarising clusters.",
    )
    parser.add_argument(
        "--surface-threshold",
        dest="surface_threshold",
        default=0.01,
        type=float,
        help="Minimum R² displayed on the surface plots.",
    )
    parser.add_argument(
        "--window",
        default=5,
        type=int,
        help="Moving-average window for ratings (default 5 TRs).",
    )
    parser.add_argument(
        "--max-clusters",
        dest="max_clusters",
        default=25,
        type=int,
        help="Maximum number of clusters to report.",
    )
    parser.add_argument(
        "--fsaverage-mesh",
        dest="fsaverage_mesh",
        default="fsaverage5",
        choices=["fsaverage5", "fsaverage6", "fsaverage7"],
        help="fsaverage mesh resolution for plotting.",
    )
    parser.add_argument(
        "--n-perm",
        dest="n_perm",
        type=int,
        default=0,
        help="Number of IAAFT surrogates for permutation testing (0 disables).",
    )
    parser.add_argument(
        "--perm-method",
        dest="perm_method",
        choices=["iaaft", "shift"],
        default="iaaft",
        help="Permutation strategy: IAAFT surrogate or circular shift.",
    )
    parser.add_argument(
        "--iaaft-iterations",
        dest="iaaft_iterations",
        type=int,
        default=100,
        help="Maximum iterations for each IAAFT surrogate.",
    )
    parser.add_argument(
        "--min-shift",
        dest="min_shift",
        type=int,
        default=30,
        help="Minimum lag (in TRs) when drawing circular shifts.",
    )
    parser.add_argument(
        "--perm-seed",
        dest="perm_seed",
        type=int,
        default=None,
        help="Random seed for permutation surrogates.",
    )
    parser.add_argument(
        "--n-bootstrap",
        dest="n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for R² standard error (0 disables).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        dest="bootstrap_seed",
        type=int,
        default=None,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Print textual progress bars during permutation/bootstrap loops.",
    )
    parser.add_argument(
        "--design-preprocessed",
        action="store_true",
        help="Treat --ratings .npy as already lagged/aligned (skip extra smoothing).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
