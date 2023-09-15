from __future__ import annotations

import argparse

import polars as pl

# Enable Polars global string cache so all categoricals are created with the same
# string cache.
pl.enable_string_cache(True)


def qc(
    fragments_tsv_filename: str,
    regions_bed_filename: str,
    tss_annotation_bed_filename: str,
    output_prefix: str,
    tss_flank_window: int = 1000,
    tss_smoothing_rolling_window: int = 10,
    tss_minimum_signal_window: int = 100,
    tss_window: int = 50,
    tss_min_norm: int = 0.2,
    use_genomic_ranges: bool = True,
    min_fragments_per_cb: int = 50,
    collapse_duplicates: bool = True,
) -> None:
    """
    Compute quality check statistics from fragments file.

    Parameters
    ----------
    fragments_tsv_filename
        Fragments TSV filename which contains scATAC fragments.
    regions_bed_filename
        Consensus peaks / SCREEN regions BED file.
        Used to calculate amount of fragments in peaks.
    tss_annotation_bed_filename
        TSS annotation BED file.
        Used to calculate distance of fragments to TSS positions.
    output_prefix
        Output prefix to use for QC statistics parquet output files.
    tss_flank_window
        Flanking window around the TSS.
        Used for intersecting fragments with TSS positions and keeping cut sites.
        Default: ``1000`` (+/- 1000 bp).
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    tss_smoothing_rolling_window
        Rolling window used to smooth the cut sites signal.
        Default: ``10``.
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    tss_minimum_signal_window
        Average signal in the tails of the flanking window around the TSS:
           - ``[-flank_window, -flank_window + minimum_signal_window + 1]``
           - ``[flank_window - minimum_signal_window + 1, flank_window]``
        is used to normalize the TSS enrichment.
        Default: ``100`` (average signal in ``[-1000, -901]``, ``[901, 1000]``
        around TSS if `flank_window=1000`).
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    tss_window
        Window around the TSS used to count fragments in the TSS when calculating
        the TSS enrichment per cell barcode.
        Default: ``50`` (+/- 50 bp).
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    tss_min_norm
        Minimum normalization score.
        If the average minimum signal value is below this value, this number is used
        to normalize the TSS signal. This approach penalizes cells with fewer reads.
        Default: ``0.2``
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    use_genomic_ranges
        Use genomic ranges implementation for calculating intersections, instead of
        using pyranges.
    min_fragments_per_cb
        Minimum number of fragments needed per cell barcode to keep the fragments
        for those cell barcodes.
    collapse_duplicates
        Collapse duplicate fragments (same chromosomal positions and linked to the same
        cell barcode).

    Returns
    -------
    None
    """
    from pycisTopic.fragments import read_bed_to_polars_df, read_fragments_to_polars_df
    from pycisTopic.gene_annotation import read_tss_annotation_from_bed
    from pycisTopic.qc import compute_qc_stats

    tss_annotation_bed_df_pl = read_tss_annotation_from_bed(
        tss_annotation_bed_filename=tss_annotation_bed_filename
    )

    regions_df_pl = read_bed_to_polars_df(
        bed_filename=regions_bed_filename,
        min_column_count=3,
    )

    fragments_df_pl = read_fragments_to_polars_df(
        fragments_tsv_filename,
        engine="pyarrow",
    )

    (
        fragments_stats_per_cb_df_pl,
        insert_size_dist_df_pl,
        tss_norm_matrix_sample,
        tss_norm_matrix_per_cb,
    ) = compute_qc_stats(
        fragments_df_pl=fragments_df_pl,
        regions_df_pl=regions_df_pl,
        tss_annotation=tss_annotation_bed_df_pl,
        tss_flank_window=tss_flank_window,
        tss_smoothing_rolling_window=tss_smoothing_rolling_window,
        tss_minimum_signal_window=tss_minimum_signal_window,
        tss_window=tss_window,
        tss_min_norm=tss_min_norm,
        use_genomic_ranges=use_genomic_ranges,
        min_fragments_per_cb=min_fragments_per_cb,
        collapse_duplicates=collapse_duplicates,
    )

    fragments_stats_per_cb_df_pl.write_parquet(
        f"{output_prefix}.fragments_stats_per_cb.parquet",
        compression="zstd",
        use_pyarrow=True,
    )

    insert_size_dist_df_pl.write_parquet(
        f"{output_prefix}.fragments_insert_size_dist.parquet",
        compression="zstd",
        use_pyarrow=True,
    )

    tss_norm_matrix_sample.write_parquet(
        f"{output_prefix}.tss_norm_matrix_sample.parquet",
        compression="zstd",
        use_pyarrow=True,
    )

    tss_norm_matrix_per_cb.write_parquet(
        f"{output_prefix}.tss_norm_matrix_per_cb.parquet",
        compression="zstd",
        use_pyarrow=True,
    )


def run_qc(args):
    qc(
        fragments_tsv_filename=args.fragments_tsv_filename,
        regions_bed_filename=args.regions_bed_filename,
        tss_annotation_bed_filename=args.tss_annotation_bed_filename,
        output_prefix=args.output_prefix,
        tss_flank_window=args.tss_flank_window,
        tss_smoothing_rolling_window=args.tss_smoothing_rolling_window,
        tss_minimum_signal_window=args.tss_minimum_signal_window,
        tss_window=args.tss_window,
        tss_min_norm=args.tss_min_norm,
        use_genomic_ranges=args.use_genomic_ranges,
        min_fragments_per_cb=args.min_fragments_per_cb,
        collapse_duplicates=args.collapse_duplicates,
    )


def add_parser_qc(subparsers):
    parser_qc = subparsers.add_parser(
        "qc",
        help="Run QC statistics on fragment file.",
    )
    parser_qc.set_defaults(func=run_qc)

    parser_qc.add_argument(
        "-f",
        "--fragments",
        dest="fragments_tsv_filename",
        action="store",
        type=str,
        required=True,
        help="Fragments TSV filename which contains scATAC fragments.",
    )

    parser_qc.add_argument(
        "-r",
        "--regions",
        dest="regions_bed_filename",
        action="store",
        type=str,
        required=True,
        help="""
            Consensus peaks / SCREEN regions BED file. Used to calculate amount of
            fragments in peaks.
            """,
    )

    parser_qc.add_argument(
        "-t",
        "--tss",
        dest="tss_annotation_bed_filename",
        action="store",
        type=str,
        required=True,
        help="""
            TSS annotation BED file. Used to calculate distance of fragments to TSS
            positions.
            """,
    )

    parser_qc.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        action="store",
        type=str,
        required=True,
        help="Output prefix to use for QC statistics parquet output files.",
    )

    parser_qc.add_argument(
        "--tss_flank_window",
        dest="tss_flank_window",
        action="store",
        type=int,
        required=False,
        default=1000,
        help="Flanking window around the TSS. "
        "Used for intersecting fragments with TSS positions and keeping cut sites."
        "Default: 1000 (+/- 1000 bp).",
    )

    parser_qc.add_argument(
        "--tss_smoothing_rolling_window",
        dest="tss_smoothing_rolling_window",
        action="store",
        type=int,
        required=False,
        default=10,
        help="Rolling window used to smooth the cut sites signal. Default: 10.",
    )

    parser_qc.add_argument(
        "--tss_minimum_signal_window",
        dest="tss_minimum_signal_window",
        action="store",
        type=int,
        required=False,
        default=100,
        help="""
            Average signal in the tails of the flanking window around the TSS
            ([-flank_window, -flank_window + minimum_signal_window + 1],
            [flank_window - minimum_signal_window + 1, flank_window])
            is used to normalize the TSS enrichment.
            Default: 100 (average signal in [-1000, -901], [901, 1000] around TSS,
            if flank_window=1000).
            """,
    )

    parser_qc.add_argument(
        "--tss_window",
        dest="tss_window",
        action="store",
        type=int,
        required=False,
        default=50,
        help="""
            Window around the TSS used to count fragments in the TSS when calculating
            the TSS enrichment per cell barcode.
            Default: 50 (+/- 50 bp).
            """,
    )

    parser_qc.add_argument(
        "--tss_min_norm",
        dest="tss_min_norm",
        action="store",
        type=float,
        required=False,
        default=0.2,
        help="""
            Minimum normalization score.
            If the average minimum signal value is below this value, this number is used
            to normalize the TSS signal. This approach penalizes cells with fewer reads.
            Default: 0.2
            """,
    )

    parser_qc.add_argument(
        "--use-pyranges",
        dest="use_genomic_ranges",
        action="store_false",
        required=False,
        help="""
            Use pyranges instead of genomic ranges implementation for calculating
            intersections.
            """,
    )

    parser_qc.add_argument(
        "--min_fragments_per_cb",
        dest="min_fragments_per_cb",
        action="store",
        type=int,
        required=False,
        default=50,
        help="""
            Minimum number of fragments needed per cell barcode to keep the fragments
            for those cell barcodes.
            Default: 50.
            """,
    )

    parser_qc.add_argument(
        "--dont-collapse_duplicates",
        dest="collapse_duplicates",
        action="store_false",
        required=False,
        help="""
            Don't collapse duplicate fragments (same chromosomal positions and linked to
            the same cell barcode).
            Default: collapse duplicates.
            """,
    )


def main():
    parser = argparse.ArgumentParser(description="pycisTopic CLI.")

    subparsers = parser.add_subparsers(
        title="Commands",
        description="List of available commands for pycisTopic CLI.",
        dest="command",
        help="Command description.",
    )
    subparsers.required = True

    add_parser_qc(subparsers)

    args = parser.parse_args()
    args.func(args)

    if args.command == "qc":
        qc(
            fragments_tsv_filename=args.fragments_tsv_filename,
            regions_bed_filename=args.regions_bed_filename,
            tss_annotation_bed_filename=args.tss_annotation_bed_filename,
            output_prefix=args.output_prefix,
            tss_flank_window=args.tss_flank_window,
            tss_smoothing_rolling_window=args.tss_smoothing_rolling_window,
            tss_minimum_signal_window=args.tss_minimum_signal_window,
            tss_window=args.tss_window,
            tss_min_norm=args.tss_min_norm,
            use_genomic_ranges=args.use_genomic_ranges,
            min_fragments_per_cb=args.min_fragments_per_cb,
            collapse_duplicates=args.collapse_duplicates,
        )


if __name__ == "__main__":
    main()
