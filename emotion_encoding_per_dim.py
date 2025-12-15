from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker

from run_emotion_encoding_pipeline import (
    load_behavioral_design,
    load_danmu_design,
    zscore_columns,
    fit_encoding_model,
    save_nifti_outputs,
    summarise_clusters,
    plot_surface_maps,
    run_iaaft_permutation_test,
    run_circular_shift_permutation_test,
)

DEFAULT_EMOTION_NAMES = [
    "Happiness",
    "Surprise",
    "Fear",
    "Sadness",
    "Anger",
    "Disgust",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Univariate emotion encoding for each Danmu dimension."
    )
    parser.add_argument("--fmri", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atlas-cache", type=Path, default=Path("analysis/atlas_cache"))
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--r2-threshold", type=float, default=0.0)
    parser.add_argument("--surface-threshold", type=float, default=0.0)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument(
        "--fsaverage-mesh",
        choices=["fsaverage5", "fsaverage6", "fsaverage7"],
        default="fsaverage5",
    )
    parser.add_argument(
        "--emotion-names",
        nargs="*",
        default=DEFAULT_EMOTION_NAMES,
        help="Optional custom labels for dimensions (provide 6 names).",
    )
    parser.add_argument(
        "--dims",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of dimension indices to process (0-based).",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=0,
        help="Number of IAAFT surrogates for permutation testing (0 disables).",
    )
    parser.add_argument(
        "--iaaft-iterations",
        type=int,
        default=100,
        help="Maximum iterations for each IAAFT surrogate.",
    )
    parser.add_argument(
        "--perm-method",
        choices=["iaaft", "shift"],
        default="iaaft",
        help="Permutation method: IAAFT (default) or circular shift.",
    )
    parser.add_argument(
        "--min-shift",
        type=int,
        default=30,
        help="Minimum absolute shift (TRs) for circular permutation.",
    )
    parser.add_argument(
        "--perm-seed",
        type=int,
        default=None,
        help="Random seed for permutation surrogates.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display permutation progress bars.",
    )
    parser.add_argument(
        "--design-preprocessed",
        action="store_true",
        help="Treat --ratings .npy as already lagged/smoothed (skip extra preprocessing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.atlas_cache.mkdir(parents=True, exist_ok=True)

    fmri_img = nib.load(str(args.fmri))
    masker = NiftiMasker(mask_img=str(args.mask), standardize=False, dtype="float32")
    signals = masker.fit_transform(fmri_img)

    if args.ratings.suffix.lower() == ".npy":
        if args.design_preprocessed:
            design_full = np.load(str(args.ratings)).astype(np.float32)
        else:
            design_full = load_danmu_design(args.ratings, window=args.window)
    else:
        design_full = load_behavioral_design(args.ratings, window=args.window)
    dimensions = args.dims if args.dims is not None else list(range(design_full.shape[1]))
    names = args.emotion_names
    if len(names) != design_full.shape[1]:
        print("[WARN] Emotion names length mismatch; falling back to generic labels.")
        names = [f"Dim{idx+1}" for idx in range(design_full.shape[1])]

    summary_payload: list[dict[str, object]] = []
    total_dims = len(dimensions)
    for idx_dim, dim in enumerate(dimensions, start=1):
        if args.show_progress:
            print(f"[PROGRESS] Dimension {idx_dim}/{total_dims}")
        label = names[dim] if dim < len(names) else f"Dim{dim+1}"
        print(f"[INFO] Processing dimension {dim + 1}: {label}")
        design_raw_dim = design_full[:, [dim]]
        design_dim = zscore_columns(design_raw_dim)

        results = fit_encoding_model(design_dim, signals)

        permutation_meta: dict[str, object] | None = None
        if args.n_perm and args.n_perm > 0:
            if args.perm_method == "iaaft":
                print(
                    f"[INFO] Running IAAFT permutation ({args.n_perm} surrogates) for {label}"
                )
                p_perm, q_perm = run_iaaft_permutation_test(
                    raw_design=design_raw_dim,
                    signals=signals,
                    observed_r2=results["r2"],
                    n_perm=args.n_perm,
                    max_iter=args.iaaft_iterations,
                    seed=args.perm_seed,
                    show_progress=args.show_progress,
                )
            else:
                print(
                    f"[INFO] Running circular-shift permutation ({args.n_perm} surrogates, min_shift={args.min_shift}) for {label}"
                )
                p_perm, q_perm = run_circular_shift_permutation_test(
                    raw_design=design_raw_dim,
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
            permutation_meta = {
                "method": args.perm_method,
                "n_perm": int(args.n_perm),
                "iaaft_iterations": int(args.iaaft_iterations),
                "seed": None if args.perm_seed is None else int(args.perm_seed),
                "min_shift": int(args.min_shift),
            }

        prefix = args.output_dir / f"emotion_{dim + 1:02d}_{label.replace(' ', '_')}"
        map_paths = save_nifti_outputs(masker, fmri_img, results, prefix)

        sig_mask = (
            (results["q_values"] < args.alpha)
            & (results["r2"] >= args.r2_threshold)
            & np.isfinite(results["r2"])
        )
        sig_values = np.where(sig_mask, results["r2"], np.nan).astype(np.float32)
        sig_img = masker.inverse_transform(sig_values)
        sig_img.set_sform(fmri_img.affine)
        sig_path = prefix.with_name(prefix.name + "_r2_sig.nii.gz")
        nib.save(sig_img, str(sig_path))

        cluster_summary = summarise_clusters(
            masker,
            q_values=results["q_values"],
            r2_values=results["r2"],
            atlas_cache=args.atlas_cache,
            alpha=args.alpha,
            r2_threshold=args.r2_threshold,
            max_clusters=25,
        )

        surf_payload = plot_surface_maps(
            sig_img,
            output_dir=args.output_dir,
            prefix=prefix.name + "_surface",
            mesh=args.fsaverage_mesh,
            threshold=args.surface_threshold,
        )

        entry = {
            "dimension": int(dim),
            "emotion": label,
            "maps": map_paths,
            "significant_r2_map": str(sig_path),
            "cluster_summary": cluster_summary,
            "surface_plot": surf_payload,
        }
        if permutation_meta is not None:
            entry["permutation"] = permutation_meta
        summary_payload.append(entry)

    summary_path = args.output_dir / "emotion_univariate_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"[DONE] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
