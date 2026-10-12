from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Literal, Sequence

import polars as pl

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction
    from pathlib import Path

from pathlib import Path
import math
import numpy.typing as npt
import numpy as np
import anndata as ad
from pycisTopic.diff_features import (
    calculate_per_region_mean_and_dispersion_on_normalized_imputed_acc,
    find_highly_variable_regions,
    find_diff_accessible_regions,
    get_marker_regions_for_contrast,
)


def read_regions_from_bed(bed_path: str | Path | os.PathLike) -> list[str]:
    """
    Read a BED file and return a list of region names in 'chr:start-end' format.

    Only the first 3 columns (chrom, start, end) are used; any extra columns
    are ignored. Blank lines and lines starting with '#' or 'track' are skipped.

    Parameters
    ----------
    bed_path:
        Path to the BED file.

    Returns
    -------
    List of region names in 'chr:start-end' format.
    """
    region_names: list[str] = []
    with open(bed_path, "r") as bed_file:
        for line in bed_file:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(
                    f"BED file '{bed_path}' has a line with fewer than 3 columns: {line!r}"
                )
            chrom, start, end = fields[0], fields[1], fields[2]
            region_names.append(f"{chrom}:{start}-{end}")
    return region_names


def write_regions_to_bed(region_names: list[str], bed_path: str | Path | os.PathLike) -> None:
    """
    Write a list of region names in 'chr:start-end' format to a 3-column BED file.
    """
    with open(bed_path, "w") as bed_file:
        for region in region_names:
            chrom, rest = region.split(":")
            start, end = rest.split("-")
            bed_file.write(f"{chrom}\t{start}\t{end}\n")


def save_dars_contrast_to_bed(dars_df: pl.DataFrame, output_path: str | Path | os.PathLike) -> None:
    """
    Write a single contrast's DARs Polars DataFrame to a 3-column BED file.

    The DataFrame is expected to have a 'RegionNames' column with entries in
    'chr:start-end' format (the output of `get_marker_regions_for_contrast`).
    """
    bed = (
        dars_df
        .with_columns([
            pl.col("RegionNames").str.split_exact(":", 1).alias("tmp1"),
        ])
        .with_columns([
            pl.col("tmp1").struct.field("field_0").alias("chrom"),
            pl.col("tmp1").struct.field("field_1").alias("tmp2"),
        ])
        .with_columns([
            pl.col("tmp2").str.split_exact("-", 1).alias("tmp3"),
        ])
        .with_columns([
            pl.col("tmp3").struct.field("field_0").cast(pl.Int64).alias("start"),
            pl.col("tmp3").struct.field("field_1").cast(pl.Int64).alias("end"),
        ])
        .select(["chrom", "start", "end"])
    )
    bed.write_csv(output_path, separator="\t", include_header=False)


def filter_hv_regions_for_contrast(
    region_topic: npt.NDArray[np.float32],
    cell_topic: npt.NDArray[np.float32],
    region_names: list[str],
    cell_names: list[str],
    highly_variable_regions: list[str],
    selected_foreground_cells: list[str],
    selected_background_cells: list[str],
    scale_factor1: int = 10**6,
    regions_chunk_size: int = 20_000,
) -> list[str]:
    """
    Drop HV regions that are never accessible in a contrast's FG+BG cell subset.

    A region can be highly variable globally yet still end up with zero scaled-imputed
    accessibility across all cells selected for a particular contrast (e.g. small or
    skewed cell-type contrasts). `get_marker_regions_for_contrast` raises a ValueError
    if any such region is present in its `highly_variable_regions` input, so this helper
    pre-filters the list per contrast.

    The filtering logic mirrors the internal check in `get_marker_regions_for_contrast`:
    compute imputed accessibility for HV regions x (FG + BG) cells, scale by
    `scale_factor1`, keep only the integer part, and retain regions with at least one
    non-zero cell.

    Parameters
    ----------
    region_topic, cell_topic, region_names, cell_names
        Same as for `get_marker_regions_for_contrast`.
    highly_variable_regions
        Candidate HV regions to filter.
    selected_foreground_cells, selected_background_cells
        Cell barcodes for this contrast.
    scale_factor1, regions_chunk_size
        Must match the values later passed to `get_marker_regions_for_contrast`.

    Returns
    -------
    Subset of `highly_variable_regions` that have non-zero accessibility in at least
    one cell of (foreground + background).
    """
    cell_name_to_idx = {c: i for i, c in enumerate(cell_names)}
    region_name_to_idx = {r: i for i, r in enumerate(region_names)}

    selected_cells = list(selected_foreground_cells) + list(selected_background_cells)
    cell_indices = np.fromiter(
        (cell_name_to_idx[c] for c in selected_cells),
        dtype=np.intp,
        count=len(selected_cells),
    )
    hv_indices = np.fromiter(
        (region_name_to_idx[r] for r in highly_variable_regions),
        dtype=np.intp,
        count=len(highly_variable_regions),
    )

    # Subset to HV regions (row subset) and FG+BG cells (column subset).
    region_topic_hv = region_topic[hv_indices]
    cell_topic_subset = cell_topic[:, cell_indices]

    n_hv = region_topic_hv.shape[0]
    keep_mask = np.zeros(n_hv, dtype=bool)

    # Process in chunks to control peak RAM (mirrors chunking inside diff_features).
    for start in range(0, n_hv, regions_chunk_size):
        end = min(start + regions_chunk_size, n_hv)
        imputed_chunk = np.matmul(region_topic_hv[start:end], cell_topic_subset)
        imputed_chunk *= np.float32(scale_factor1)
        np.floor(imputed_chunk, out=imputed_chunk)
        keep_mask[start:end] = (imputed_chunk != 0).any(axis=1)

    return [hv_region for hv_region, keep in zip(highly_variable_regions, keep_mask) if keep]


def get_contrast_barcodes_lists(
    cell_data_tsv: str | Path | os.PathLike,
    barcode_column: str,
    annotation_column: str,
    contrasts_tsv: str | Path | os.PathLike | None = None,
) -> dict[str, tuple[list[str], list[str]]]:
    """
    Prepares foreground and background cell barcode lists for DARs computation.

    Parameters
    ----------
    cell_data_tsv:
        Path to TSV file with cell annotations.
    barcode_column:
        Column name containing cell barcodes.
    annotation_column:
        Column name containing cell annotations.
    contrasts_tsv:
        Optional path to TSV file specifying contrasts. Must have columns
        'foreground' and 'background', each with a comma-separated list of
        annotations from `annotation_column`.
        - If 'background' is empty, all cells NOT in the foreground are used.
        - If None, each unique annotation is compared against all other cells (1-vs-all).

    Returns
    -------
    Dictionary with foreground and background cell barcode lists.
    E.g., {"contrast_1": ([foreground_barcodes], [background_barcodes]), ...}
    """

    # Read cell data
    df = pl.read_csv(cell_data_tsv, separator="\t")

    # Initialize contrast barcodes dictionary
    contrast_barcodes: dict[str, dict[str, list[str]]] = {}

    if contrasts_tsv is not None:
        # Read contrasts from file
        df_contrasts = pl.read_csv(contrasts_tsv, separator="\t")
        if "foreground" not in df_contrasts.columns or "background" not in df_contrasts.columns:
            raise ValueError("Contrasts TSV must have 'foreground' and 'background' columns.")

        # Read contrasts from file and prepare barcode lists. Multiple annotations can be
        # comma-separated. If background is empty, all other cells are used as background.
        for row in df_contrasts.iter_rows(named=True):
            fg_raw = row["foreground"] or ""
            bg_raw = row["background"] or ""
            fg_annotations = [ann.strip() for ann in fg_raw.split(",") if ann.strip()]
            bg_annotations = [ann.strip() for ann in bg_raw.split(",") if ann.strip()]
            if not fg_annotations:
                raise ValueError("Each contrast row must have at least one foreground annotation.")

            contrast_name = (
                f"fg_{'_'.join(fg_annotations)}_vs_bg_"
                f"{'_'.join(bg_annotations) if bg_annotations else 'others'}"
            )
            contrast_barcodes[contrast_name] = {
                "foreground": [],
                "background": [],
            }
            for fg_annotation in fg_annotations:
                if fg_annotation not in df[annotation_column].unique().to_list():
                    raise ValueError(f"Foreground annotation '{fg_annotation}' not found in cell data.")
                contrast_barcodes[contrast_name]["foreground"].extend(
                    df.filter(pl.col(annotation_column) == fg_annotation)[barcode_column].to_list()
                )
            if bg_annotations:
                for bg_annotation in bg_annotations:
                    if bg_annotation not in df[annotation_column].unique().to_list():
                        raise ValueError(f"Background annotation '{bg_annotation}' not found in cell data.")
                    contrast_barcodes[contrast_name]["background"].extend(
                        df.filter(pl.col(annotation_column) == bg_annotation)[barcode_column].to_list()
                    )
            else:
                # If background is empty, all other cells are used as background
                contrast_barcodes[contrast_name]["background"] = (
                    df.filter(~pl.col(annotation_column).is_in(fg_annotations))[barcode_column].to_list()
                )
    else:
        # 1-vs-all: each unique annotation in annotation_column against all other cells
        unique_annotations = df[annotation_column].unique().to_list()
        for annotation in unique_annotations:
            contrast_name = f"fg_{annotation}_vs_bg_others"
            contrast_barcodes[contrast_name] = {
                "foreground": df.filter(pl.col(annotation_column) == annotation)[barcode_column].to_list(),
                "background": df.filter(pl.col(annotation_column) != annotation)[barcode_column].to_list(),
            }

    # Format contrast_barcodes to have tuple of lists, since that's what find_diff_accessible_regions expects
    contrast_barcodes_formatted = {c: (lists['foreground'], lists['background']) for c, lists in contrast_barcodes.items()}
    return contrast_barcodes_formatted


def run_compute_dars(args):

    region_topic_h5ad: str = args.region_topic_h5ad
    cell_topic_h5ad: str = args.cell_topic_h5ad
    cell_data_tsv: str = args.cell_data
    barcode_column: str = args.barcode_column
    annotation_column: str = args.annotation_column
    contrasts_tsv: str | None = args.contrasts_tsv
    regions_subset_bed: str | None = args.regions_subset
    hv_regions_bed: str | None = args.hv_regions_bed
    hv_plot: str | None = args.hv_plot
    output_dir: str = args.output_dir

    # DAR / HV computation parameters
    adjusted_pvalue_threshold: float = args.adjusted_pvalue_threshold
    log2_fold_change_threshold: float = args.log2_fold_change_threshold
    scale_factor1: int = args.scale_factor1
    regions_chunk_size: int = args.regions_chunk_size

    # Sanity-check incompatible options
    if hv_regions_bed is not None and regions_subset_bed is not None:
        print(
            "WARNING: Both --hv-regions-bed and --regions-subset were provided. "
            "--regions-subset will be ignored; the regions in --hv-regions-bed will be "
            "used directly as highly variable regions.",
            file=sys.stderr,
        )
    if hv_regions_bed is not None and hv_plot is not None:
        print(
            "WARNING: --hv-plot was provided but --hv-regions-bed is also set, so the "
            "highly variable region computation is skipped and no plot will be generated.",
            file=sys.stderr,
        )

    print("Loading region-topic and cell-topic distributions...")

    region_topic_adata = ad.read_h5ad(region_topic_h5ad)
    cell_topic_adata = ad.read_h5ad(cell_topic_h5ad)
    region_topic = region_topic_adata.X.astype(np.float32)
    cell_topic = cell_topic_adata.X.astype(np.float32).T
    cell_names = cell_topic_adata.obs_names.tolist()
    all_region_names = region_topic_adata.obs_names.tolist()

    if hv_regions_bed is not None:
        # Path A: user provided precomputed highly variable regions -> skip HV computation.
        # We still pass the full region_topic matrix to find_diff_accessible_regions;
        # it will internally subset to the highly variable regions via their names.
        print(f"Loading highly variable regions from BED file: {hv_regions_bed}")
        hv_region_names = read_regions_from_bed(hv_regions_bed)

        # Keep only HV regions that are actually present in the region-topic matrix
        all_region_names_set = set(all_region_names)
        missing = [r for r in hv_region_names if r not in all_region_names_set]
        if missing:
            print(
                f"WARNING: {len(missing)} of {len(hv_region_names)} HV regions from the BED "
                f"are not in the region-topic matrix; they will be dropped. "
                f"First few missing: {missing[:5]}",
                file=sys.stderr,
            )
            hv_region_names = [r for r in hv_region_names if r in all_region_names_set]
        if not hv_region_names:
            raise ValueError(
                "No HV regions from --hv-regions-bed overlap with the region-topic matrix."
            )
        print(f"Using {len(hv_region_names)} highly variable regions from BED.")
        region_names = all_region_names
    else:
        # Path B: no HV regions provided -> optionally subset regions, then compute HV regions.
        if regions_subset_bed is not None:
            print("Subsetting region-topic matrix to provided regions...")
            subset_region_names = read_regions_from_bed(regions_subset_bed)
            region_name_to_index = {name: idx for idx, name in enumerate(all_region_names)}
            subset_indices = [
                region_name_to_index[name]
                for name in subset_region_names
                if name in region_name_to_index
            ]
            region_topic = region_topic[subset_indices, :]
            region_names = [all_region_names[idx] for idx in subset_indices]
            print(f"Number of regions after subsetting: {len(region_names)}")
        else:
            region_names = all_region_names

        print("Calculating per-region mean and dispersion on normalized imputed accessibility...")
        (
            region_names_to_keep,
            per_region_means_on_normalized_imputed_acc,
            per_region_dispersions_on_normalized_imputed_acc,
        ) = calculate_per_region_mean_and_dispersion_on_normalized_imputed_acc(
            region_topic=region_topic,
            cell_topic=cell_topic,
            region_names=region_names,
            scale_factor1=scale_factor1,
            scale_factor2=10**4,
            regions_chunk_size=regions_chunk_size,
        )

        print("Finding highly variable regions...")
        # plot: None/False -> no plot; str -> save to that path
        hv_region_names = find_highly_variable_regions(
            regions=region_names_to_keep,
            per_region_means_on_normalized_imputed_acc=per_region_means_on_normalized_imputed_acc,
            per_region_dispersions_on_normalized_imputed_acc=per_region_dispersions_on_normalized_imputed_acc,
            min_disp=0.05,
            min_mean=0.0125,
            max_disp=np.inf,
            max_mean=3,
            n_bins=20,
            n_top_features=None,
            plot=hv_plot if hv_plot is not None else False,
        )
        if hv_plot is not None:
            print(f"Saved mean-vs-dispersion plot to: {hv_plot}")
        print(f"Number of highly variable regions: {len(hv_region_names)}")

    print("Preparing contrast cell barcode lists...")
    contrast_barcodes = get_contrast_barcodes_lists(
        cell_data_tsv=cell_data_tsv,
        barcode_column=barcode_column,
        annotation_column=annotation_column,
        contrasts_tsv=contrasts_tsv,
    )

    for contrast, (fg_cells, bg_cells) in contrast_barcodes.items():
        print(f"Contrast: {contrast}")
        print(f"  Foreground cells: {len(fg_cells)}")
        print(f"  Background cells: {len(bg_cells)}")

    print("Computing differentially accessible regions (DARs)...")
    # Create output directory up front so we can write each contrast as it completes.
    os.makedirs(output_dir, exist_ok=True)

    # Loop per contrast so we can (a) pre-filter highly variable regions that are never
    # accessible in a particular contrast's FG+BG cell subset, and (b) write each
    # contrast's BED as soon as it finishes. This way, a crash mid-loop preserves all
    # already-completed contrasts, and re-running picks up where it left off.
    n_total = len(contrast_barcodes)
    n_saved = 0
    n_skipped = 0
    n_empty = 0
    for contrast_name, (fg_cells, bg_cells) in contrast_barcodes.items():
        output_path = os.path.join(output_dir, f"DARs_{contrast_name}.bed")

        # Resume: if this contrast's BED already exists, skip it.
        if os.path.exists(output_path):
            print(
                f"[{contrast_name}] Output BED already exists at {output_path}; "
                f"skipping. Delete the file to recompute."
            )
            n_skipped += 1
            continue

        filtered_hv = filter_hv_regions_for_contrast(
            region_topic=region_topic,
            cell_topic=cell_topic,
            region_names=region_names,
            cell_names=cell_names,
            highly_variable_regions=hv_region_names,
            selected_foreground_cells=fg_cells,
            selected_background_cells=bg_cells,
            scale_factor1=scale_factor1,
            regions_chunk_size=regions_chunk_size,
        )
        n_dropped = len(hv_region_names) - len(filtered_hv)
        if n_dropped > 0:
            print(
                f"[{contrast_name}] Dropped {n_dropped} of {len(hv_region_names)} HV "
                f"regions that are never accessible in the {len(fg_cells)} foreground + "
                f"{len(bg_cells)} background cells of this contrast."
            )
        if not filtered_hv:
            print(
                f"[{contrast_name}] WARNING: no HV regions remain after filtering; "
                f"skipping this contrast.",
                file=sys.stderr,
            )
            n_empty += 1
            continue

        dars_df = get_marker_regions_for_contrast(
            region_topic=region_topic,
            cell_topic=cell_topic,
            region_names=region_names,
            cell_names=cell_names,
            highly_variable_regions=filtered_hv,
            contrast_name=contrast_name,
            selected_foreground_cells=fg_cells,
            selected_background_cells=bg_cells,
            scale_factor1=scale_factor1,
            regions_chunk_size=regions_chunk_size,
            adjusted_pvalue_threshold=adjusted_pvalue_threshold,
            log2_fold_change_threshold=log2_fold_change_threshold,
        )
        save_dars_contrast_to_bed(dars_df, output_path)
        print(f"[{contrast_name}] Saved {dars_df.height} DARs to {output_path}")
        n_saved += 1

    print(
        f"\nDARs computation completed. "
        f"{n_saved} newly computed, {n_skipped} skipped (already existed), "
        f"{n_empty} empty (no HV regions left after filtering), "
        f"{n_total} total contrasts."
    )


def run_compute_hv_regions(args):
    output_bed: str = args.output_bed
    plot_path: str | None = args.plot

    print("Loading region-topic and cell-topic distributions...")

    region_topic_h5ad: str = args.region_topic_h5ad
    cell_topic_h5ad: str = args.cell_topic_h5ad

    region_topic_adata = ad.read_h5ad(region_topic_h5ad)
    cell_topic_adata = ad.read_h5ad(cell_topic_h5ad)
    region_topic = region_topic_adata.X.astype(np.float32)
    cell_topic = cell_topic_adata.X.astype(np.float32).T
    region_names = region_topic_adata.obs_names.tolist()

    print("Calculating per-region mean and dispersion on normalized imputed accessibility...")
    print("region_topic shape:", region_topic.shape)
    print("cell_topic shape:", cell_topic.shape)

    (
        region_names_to_keep,
        per_region_means_on_normalized_imputed_acc,
        per_region_dispersions_on_normalized_imputed_acc,
    ) = calculate_per_region_mean_and_dispersion_on_normalized_imputed_acc(
        region_topic=region_topic,
        cell_topic=cell_topic,
        region_names=region_names,
        scale_factor1=10**6,
        scale_factor2=10**4,
        regions_chunk_size=20_000,
    )

    print("Finding highly variable regions...")
    # plot: None/False -> no plot; str -> save to that path
    hv_region_names = find_highly_variable_regions(
        regions=region_names_to_keep,
        per_region_means_on_normalized_imputed_acc=per_region_means_on_normalized_imputed_acc,
        per_region_dispersions_on_normalized_imputed_acc=per_region_dispersions_on_normalized_imputed_acc,
        min_disp=0.05,
        min_mean=0.0125,
        max_disp=np.inf,
        max_mean=3,
        n_bins=20,
        n_top_features=None,
        plot=plot_path if plot_path is not None else False,
    )

    print(f"Number of highly variable regions selected: {len(hv_region_names)}")
    if plot_path is not None:
        print(f"Saved mean-vs-dispersion plot to: {plot_path}")

    print(f"Saving highly variable regions to bed file: {output_bed}")
    write_regions_to_bed(hv_region_names, output_bed)

    print("Highly variable regions computation completed.")


_CONTRASTS_TSV_HELP = (
    "Path to TSV file with contrasts. Must have two columns: 'foreground' and "
    "'background', each a comma-separated list of annotations from --annotation-column. "
    "If 'background' is empty, all cells not in the foreground are used. "
    "If this option is omitted, each unique annotation is compared against all "
    "other cells (1-vs-all). "
    "Example TSV content:\\n"
    "foreground\\tbackground\\n"
    "L2/3 IT\\tL5 IT A,L5 IT B,L6 IT\\n"
    "Oligo\\t\\n"
    "Vip,Sst,Pvalb,Lamp5\\tL2/3 IT,L5 IT A,L5 IT B"
)

_PLOT_HELP = (
    "Optional path to save the mean-vs-dispersion plot (e.g., 'hv_plot.png'). "
    "Any extension supported by matplotlib works (.png, .pdf, .svg). "
    "If omitted, no plot is generated."
)


def add_parser_dars(subparsers: _SubParsersAction[ArgumentParser]):
    """Creates an ArgumentParser to read the options for this script."""
    parser_dars = subparsers.add_parser(
        "dars",
        help="Compute differentially accessible regions (DARs).",
    )

    subparser_dars = parser_dars.add_subparsers(
        title="DARs",
        dest="dars",
        help="List of DARs subcommands.",
    )
    subparser_dars.required = True

    # ---------- compute_dars ----------
    parser_dars_compute_dars = subparser_dars.add_parser(
        "compute_dars",
        help="Compute differentially accessible regions (DARs) between two groups of cells.",
    )
    parser_dars_compute_dars.set_defaults(func=run_compute_dars)

    parser_dars_compute_dars.add_argument(
        "--region-topic-h5ad",
        required=True,
        help="Path to region-topic distribution H5AD file. Output of topic modeling.",
    )
    parser_dars_compute_dars.add_argument(
        "--cell-topic-h5ad",
        required=True,
        help="Path to cell-topic distribution H5AD file. Output of topic modeling.",
    )

    # Region filtering options (mutually informative)
    parser_dars_compute_dars.add_argument(
        "--regions-subset",
        default=None,
        help="Path to BED file with regions of interest. Region-topic matrix will be "
        "subsetted to these regions BEFORE highly variable region computation. "
        "Ignored if --hv-regions-bed is provided. If omitted, all regions are used.",
    )
    parser_dars_compute_dars.add_argument(
        "--hv-regions-bed",
        default=None,
        help="Path to BED file with pre-computed highly variable regions (e.g. output "
        "of `compute_hv_regions`). If provided, the HV computation step is SKIPPED "
        "and these regions are used directly for DAR computation. "
        "Takes precedence over --regions-subset.",
    )
    parser_dars_compute_dars.add_argument(
        "--hv-plot",
        default=None,
        help=(
            _PLOT_HELP
            + " Only takes effect if HV regions are computed internally (i.e. "
            "--hv-regions-bed is not provided)."
        ),
    )

    # Cell annotation / contrast options
    parser_dars_compute_dars.add_argument(
        "--cell-data",
        required=True,
        help="Path to TSV file with cell annotations (must include barcode and annotation columns).",
    )
    parser_dars_compute_dars.add_argument(
        "--barcode-column",
        default="cell_barcode",
        help="Column containing cell barcodes. Default: cell_barcode",
    )
    parser_dars_compute_dars.add_argument(
        "--annotation-column",
        default="cell_type",
        help="Column containing cell annotations. Default: cell_type",
    )
    parser_dars_compute_dars.add_argument(
        "--contrasts-tsv",
        required=False,
        default=None,
        help=_CONTRASTS_TSV_HELP,
    )

    parser_dars_compute_dars.add_argument(
        "--output-dir",
        required=True,
        help="Path to output directory, where all bed files for each contrast will be saved.",
    )

    # DAR / HV computation parameters
    parser_dars_compute_dars.add_argument(
        "--adjusted-pvalue-threshold",
        type=float,
        default=0.05,
        help="Maximum adjusted p-value (Benjamini-Hochberg) for a region to be called "
        "differentially accessible. Default: 0.05",
    )
    parser_dars_compute_dars.add_argument(
        "--log2-fold-change-threshold",
        type=float,
        default=math.log2(1.5),
        help="Minimum absolute log2 fold change for a region to be called differentially "
        "accessible. Default: log2(1.5) (approximately 0.585)",
    )
    parser_dars_compute_dars.add_argument(
        "--scale-factor1",
        type=int,
        default=10**6,
        help="Scale factor applied to imputed accessibility to create a 'count' matrix "
        "(removes noise by rounding small values to zero). Also used during internal HV "
        "computation when --hv-regions-bed is not provided. Default: 1000000",
    )
    parser_dars_compute_dars.add_argument(
        "--regions-chunk-size",
        type=int,
        default=20_000,
        help="Number of regions processed per chunk. Larger values use more RAM but may "
        "be faster. Also used during internal HV computation when --hv-regions-bed is not "
        "provided. Default: 20000",
    )

    # ---------- compute_hv_regions ----------
    parser_dars_compute_hv_regions = subparser_dars.add_parser(
        "compute_hv_regions",
        help="Compute highly variable regions and save into a BED file. "
        "The output can then be passed to `compute_dars --hv-regions-bed` to skip "
        "re-computing HV regions.",
    )
    parser_dars_compute_hv_regions.set_defaults(func=run_compute_hv_regions)

    parser_dars_compute_hv_regions.add_argument(
        "--region-topic-h5ad",
        required=True,
        help="Path to region-topic distribution H5AD file. Output of topic modeling.",
    )
    parser_dars_compute_hv_regions.add_argument(
        "--cell-topic-h5ad",
        required=True,
        help="Path to cell-topic distribution H5AD file. Output of topic modeling.",
    )
    parser_dars_compute_hv_regions.add_argument(
        "--output-bed",
        required=True,
        help="Path to output BED file.",
    )
    parser_dars_compute_hv_regions.add_argument(
        "--plot",
        default=None,
        help=_PLOT_HELP,
    )