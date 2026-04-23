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
)


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
        Optional path to TSV file specifying contrasts. Should have columns:
        'foreground' and 'background', with each containing a comma-separated list of annotations from annotation_column.
        'background' can be empty, in which case all other cells are used as background.
        If None, DARs for each cell type from annotation_column is computed against all others.
    
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

        # Read contrasts from file and prepare barcode lists. If multiple annotations are provided, they are comma-separated.
        # If background is empty, all other cells are used. If annotations are not found, raise error.
        for row in df_contrasts.iter_rows(named=True):
            fg_annotations = [ann.strip() for ann in row["foreground"].split(",")]
            bg_annotations = [ann.strip() for ann in row["background"].split(",")]
            contrast_name = f"fg_{'_'.join(fg_annotations)}_vs_bg_{'_'.join(bg_annotations)}"
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
        # Prepare contrasts for each unique annotation in annotation_column against all other cells
        unique_annotations = df[annotation_column].unique().to_list()
        for annotation in unique_annotations:
            contrast_name = f"fg_{annotation}_vs_bg_others"
            contrast_barcodes[contrast_name] = {
                "foreground": df.filter(pl.col(annotation_column) == annotation)[barcode_column].to_list(),
                "background": df.filter(pl.col(annotation_column) != annotation)[barcode_column].to_list(),
            }
    
    # Format contrast_barcodes to have tuple of lists, since that's what compute_dars expects
    contrast_barcodes_formatted = {c: (lists['foreground'], lists['background']) for c, lists in contrast_barcodes.items()}
    return contrast_barcodes_formatted


def compute_dars(
    region_topic: npt.NDArray[np.float32],
    topic_cell: npt.NDArray[np.float32],
    region_names: list[str],
    cell_names: list[str],
    contrasts: dict[str, tuple[list[str], list[str]]],
):

    # We will need to call three functions from diff_features module
    # 1. calculate_per_region_mean_and_dispersion_on_normalized_imputed_acc

    (
        region_names_to_keep,
        per_region_means_on_normalized_imputed_acc,
        per_region_dispersions_on_normalized_imputed_acc,
    ) = calculate_per_region_mean_and_dispersion_on_normalized_imputed_acc(
        region_topic=region_topic,
        cell_topic=topic_cell,
        region_names=region_names,
        scale_factor1 = 10**6,
        scale_factor2 = 10**4,
        regions_chunk_size=20_000,
    )

    # 3. find_diff_accessible_regions
    dars_dict = find_diff_accessible_regions(
        region_topic=region_topic,
        cell_topic=topic_cell,
        region_names=region_names,
        cell_names=cell_names,
        # highly_variable_regions=region_names,
        contrasts=contrasts,
        scale_factor1=10**6,
        regions_chunk_size=20_000,
        adjusted_pvalue_threshold=0.05,
        log2_fold_change_threshold=math.log2(1.5),
        )
    
    return dars_dict


def run_compute_dars(args):
    # TODO: add a logger

    region_topic_h5ad: str = args.region_topic_h5ad
    cell_topic_h5ad: str = args.cell_topic_h5ad
    cell_data_tsv: str = args.cell_data
    barcode_column: str = args.barcode_column
    annotation_column: str = args.annotation_column
    contrasts_tsv: str | None = args.contrasts_tsv
    regions_subset_bed: str | None = args.regions_subset
    output_dir: str = args.output_dir

    print("Loading region-topic and cell-topic distributions...")
    
    region_topic_adata = ad.read_h5ad(region_topic_h5ad)
    cell_topic_adata = ad.read_h5ad(cell_topic_h5ad)
    region_topic = region_topic_adata.X.astype(np.float32)
    topic_cell = cell_topic_adata.X.astype(np.float32).T
    cell_names = cell_topic_adata.obs_names.tolist()

    # TODO: will need to use genomic ranges for subsetting, it is already implemented in the pycisTopic_v3 branch (rebase and fetch, I guess?)
    if regions_subset_bed is not None:
        print("Subsetting to provided regions...")
        with open(regions_subset_bed, "r") as bed_file:
            subset_region_names = [line.strip().split("\t")[0] + ":" + line.strip().split("\t")[1] + "-" + line.strip().split("\t")[2] for line in bed_file]
        region_name_to_index = {name: idx for idx, name in enumerate(region_topic_adata.obs_names.tolist())}
        subset_indices = [region_name_to_index[name] for name in subset_region_names if name in region_name_to_index]
        region_topic = region_topic[subset_indices, :]
        region_names = [region_topic_adata.obs_names[idx] for idx in subset_indices]
        print(f"Number of regions after subsetting: {len(region_names)}")
    else:
        region_names = region_topic_adata.obs_names.tolist()

    

    print("Preparing contrast cell barcode lists...")

    contrast_barcodes = get_contrast_barcodes_lists(
        cell_data_tsv=cell_data_tsv,
        barcode_column=barcode_column,
        annotation_column=annotation_column,
        contrasts_tsv=contrasts_tsv
    )

    for contrast, barcodes in contrast_barcodes.items():
        print(f"Contrast: {contrast}")
        print(f"  Foreground cells: {len(barcodes[0])}")
        print(f"  Background cells: {len(barcodes[1])}")
    
    print("Computing differentially accessible regions (DARs)...")

    dars_dict = compute_dars(
        region_topic=region_topic,
        topic_cell=topic_cell,
        contrasts=contrast_barcodes,
        region_names=region_names,
        cell_names=cell_names
    )

    print("Saving DARs results...")
    save_dars_results(dars_dict, output_dir)

    print("DARs computation completed.")


def save_dars_results(dars_dict: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for contrast, dars_df in dars_dict.items():
        # Split "chr:start-end"
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

        out = os.path.join(output_dir, f"DARs_{contrast}.bed")
        bed.write_csv(out, separator="\t", has_header=False)
        print(f"Saved BED3: {out}")

def run_compute_hv_regions(args):
    output_bed: str = args.output_bed

    print("Loading region-topic and cell-topic distributions...")

    # For HV regions computation, we need region-topic and cell-topic distributions
    # Here we assume they are provided as arguments, but you can modify as needed
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
        plot="",  # Specify path to save plot if needed
    )

    print(f"Number of highly variable regions selected: {len(hv_region_names)}")

    print(f"Saving highly variable regions to bed file: {output_bed}")

    with open(output_bed, "w") as hv_bed_file:
        for region in hv_region_names:
            chrom, rest = region.split(":")
            start, end = rest.split("-")
            hv_bed_file.write(f"{chrom}\t{start}\t{end}\n")

    print("Highly variable regions computation completed.")



def add_parser_dars(subparsers: _SubParsersAction[ArgumentParser]):
    # TODO: include parameters for DARs computation (e.g., thresholds, chunk sizes, etc.)

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

    # Compute DARs parsers
    parser_dars_compute_dars = subparser_dars.add_parser(
        "compute_dars",
        help="Compute differentially accessible regions (DARs) between two groups of cells.",
    )

    parser_dars_compute_dars.set_defaults(func=run_compute_dars)

    # region_topic and cell_topic related parsers
    parser_dars_compute_dars.add_argument(
        "--region-topic-h5ad",
        required=True,
        help="Path to region-topic distribution H5AD file. This is an output of topic modeling.",
    )

    parser_dars_compute_dars.add_argument(
        "--cell-topic-h5ad",
        required=True,
        help="Path to cell-topic distribution H5AD file. This is an output of topic modeling.",
    )

    parser_dars_compute_dars.add_argument(
        "--regions-subset",
        help="Path to bed file with regions of interest for DARs computation. "
        "These will be overlapped with regions in the region-topic H5AD file. "
        "If omitted, all regions are used.",
    )

    # Prepare contrast cells related parsers
    parser_dars_compute_dars.add_argument(
        "--cell-data",
        required=True,
        help="Path to TSV file with cell annotations (must include barcode and annotation columns)."
    )

    parser_dars_compute_dars.add_argument(
        "--barcode-column",
        default="cell_barcode",
        help="Column containing cell barcodes. Default: cell_barcode"
    )

    parser_dars_compute_dars.add_argument(
        "--annotation-column",
        default="cell_type",
        help="Column containing cell annotations. Default: cell_type"
    )

    parser_dars_compute_dars.add_argument(
        "--contrasts-tsv",
        required=False,
        default=None,
        help="Path to TSV file with contrasts (foreground and background annotations). "
        "If omitted, each annotation is compared against all other cells."
    )

    # Output DARs bed files related parser
    parser_dars_compute_dars.add_argument(
        "--output-dir",
        required=True,
        help="Path to output directory, where all bed files for each contrast will be saved."
    )

    # Highly variable regions related parser
    parser_dars_compute_hv_regions = subparser_dars.add_parser(
        "compute_hv_regions",
        help="Compute highly variable regions and save into a bed file. "
        "This can be provided as input to compute_dars command.",
    )

    parser_dars_compute_hv_regions.set_defaults(func=run_compute_hv_regions)

    parser_dars_compute_hv_regions.add_argument(
        "--region-topic-h5ad",
        required=True,
        help="Path to region-topic distribution H5AD file. This is an output of topic modeling.",
    )

    parser_dars_compute_hv_regions.add_argument(
        "--cell-topic-h5ad",
        required=True,
        help="Path to cell-topic distribution H5AD file. This is an output of topic modeling.",
    )

    parser_dars_compute_hv_regions.add_argument(
        "--output-bed",
        required=True,
        help="Path to output bed file."
    )
    

    
