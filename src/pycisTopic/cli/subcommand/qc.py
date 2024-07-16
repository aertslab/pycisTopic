from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction
    from pathlib import Path

# Enable Polars global string cache so all categoricals are created with the same
# string cache.
pl.enable_string_cache()


def qc(
    fragments_tsv_filename: str | Path,
    regions_bed_filename: str | Path,
    tss_annotation_bed_filename: str | Path,
    output_prefix: str,
    tss_flank_window: int = 2000,
    tss_smoothing_rolling_window: int = 10,
    tss_minimum_signal_window: int = 100,
    tss_window: int = 50,
    tss_min_norm: float = 0.2,
    use_genomic_ranges: bool = True,
    min_fragments_per_cb: int = 10,
    collapse_duplicates: bool = True,
    no_threads: int = 8,
    engine: str | Literal["polars"] | Literal["pyarrow"] = "pyarrow",
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
        Default: ``2000`` (+/- 2000 bp).
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
        Default: ``100`` (average signal in ``[-2000, -1901]``, ``[1901, 2000]``
        around TSS if `flank_window=2000`).
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
    no_threads
        Number of threads to use when calculating kernel-density estimate (KDE) to get
        probability density function (PDF) values for log10 unique fragments in peaks
        vs TSS enrichment, fractions of fragments in peaks and duplication ratio.
        Default: ``8``
    engine
        Use Polars or pyarrow to read BED and fragment files (default: `pyarrow`).

    Returns
    -------
    None

    """
    import logging

    from pycisTopic.fragments import read_bed_to_polars_df, read_fragments_to_polars_df
    from pycisTopic.gene_annotation import read_tss_annotation_from_bed
    from pycisTopic.qc import compute_qc_stats, get_otsu_threshold

    # Remove trailing dot(s) from the output prefix.
    output_prefix = output_prefix.rstrip(".")

    class RelativeSeconds(logging.Formatter):
        def format(self, record):
            record.relativeCreated = record.relativeCreated // 1000
            return super().format(record)

    formatter = RelativeSeconds(
        "%(asctime)s %(relativeCreated)ds - %(levelname)s - %(name)s:%(funcName)s - %(message)s"
    )
    logging.basicConfig(
        # format=formatter,
        filename=f"{output_prefix}.pycistopic_qc.log",
        encoding="utf-8",
        level=logging.INFO,
    )
    logging.root.handlers[0].setFormatter(formatter)

    logger = logging.getLogger(__name__)

    logger.info(f'Loading TSS annotation from "{tss_annotation_bed_filename}".')
    tss_annotation_bed_df_pl = read_tss_annotation_from_bed(
        tss_annotation_bed_filename=tss_annotation_bed_filename
    )

    logger.info(f'Loading regions BED file from "{regions_bed_filename}".')
    regions_df_pl = read_bed_to_polars_df(
        bed_filename=regions_bed_filename,
        min_column_count=3,
        engine=engine,
    )

    logger.info(f'Loading fragments TSV file from "{fragments_tsv_filename}".')
    fragments_df_pl = read_fragments_to_polars_df(
        fragments_tsv_filename,
        engine=engine,
    )

    logger.info("Computing QC stats.")
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
        no_threads=no_threads,
    )

    logger.info(f'Writing "{output_prefix}.fragments_stats_per_cb.parquet".')
    fragments_stats_per_cb_df_pl.write_parquet(
        f"{output_prefix}.fragments_stats_per_cb.parquet",
        compression="zstd",
        use_pyarrow=True,
    )

    logger.info(f'Writing "{output_prefix}.fragments_insert_size_dist.parquet".')
    insert_size_dist_df_pl.write_parquet(
        f"{output_prefix}.fragments_insert_size_dist.parquet",
        compression="zstd",
        use_pyarrow=True,
    )

    logger.info(f'Writing "{output_prefix}.tss_norm_matrix_sample.parquet".')
    tss_norm_matrix_sample.write_parquet(
        f"{output_prefix}.tss_norm_matrix_sample.parquet",
        compression="zstd",
        use_pyarrow=True,
    )

    logger.info(f'Writing "{output_prefix}.tss_norm_matrix_per_cb.parquet".')
    tss_norm_matrix_per_cb.write_parquet(
        f"{output_prefix}.tss_norm_matrix_per_cb.parquet",
        compression="zstd",
        use_pyarrow=True,
    )

    logger.info("Calculating Otsu thresholds.")
    (
        unique_fragments_in_peaks_count_otsu_threshold,
        tss_enrichment_otsu_threshold,
        fragments_stats_per_cb_for_otsu_threshold_df_pl,
    ) = get_otsu_threshold(
        fragments_stats_per_cb_df_pl=fragments_stats_per_cb_df_pl,
        min_otsu_fragments=100,
        min_otsu_tss=1.0,
    )

    logger.info(
        f'Writing "{output_prefix}.fragments_stats_per_cb_for_otsu_thresholds.parquet".'
    )
    fragments_stats_per_cb_for_otsu_threshold_df_pl.write_parquet(
        f"{output_prefix}.fragments_stats_per_cb_for_otsu_thresholds.parquet",
        compression="zstd",
        use_pyarrow=True,
    )
    logger.info(
        f'Writing "{output_prefix}.fragments_stats_per_cb_for_otsu_thresholds.tsv".'
    )
    fragments_stats_per_cb_for_otsu_threshold_df_pl.write_csv(
        f"{output_prefix}.fragments_stats_per_cb_for_otsu_thresholds.tsv",
        separator="\t",
        include_header=True,
    )

    logger.info(f'Writing "{output_prefix}.cbs_for_otsu_thresholds.tsv".')
    fragments_stats_per_cb_for_otsu_threshold_df_pl.select(pl.col("CB")).write_csv(
        f"{output_prefix}.cbs_for_otsu_thresholds.tsv",
        separator="\t",
        include_header=False,
    )

    logger.info(f'Writing "{output_prefix}.otsu_thresholds.tsv".')
    with open(f"{output_prefix}.otsu_thresholds.tsv", "w") as fh:
        print(
            "unique_fragments_in_peaks_count_otsu_threshold\ttss_enrichment_otsu_threshold\n"
            f"{unique_fragments_in_peaks_count_otsu_threshold}\t{tss_enrichment_otsu_threshold}",
            file=fh,
        )
    logger.info("pycisTopic QC finished.")


def run_qc_run(args):
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
        no_threads=args.threads,
        engine=args.engine,
    )


def add_parser_qc(subparsers: _SubParsersAction[ArgumentParser]):
    parser_qc = subparsers.add_parser(
        "qc",
        help="Run QC statistics on fragment file.",
    )
    subparser_qc = parser_qc.add_subparsers(
        title="QC",
        dest="qc",
        help="List of QC subcommands.",
    )
    subparser_qc.required = True

    parser_qc_run = subparser_qc.add_parser(
        "run",
        help="Run QC statistics on fragment file.",
    )
    parser_qc_run.set_defaults(func=run_qc_run)

    parser_qc_run.add_argument(
        "-f",
        "--fragments",
        dest="fragments_tsv_filename",
        action="store",
        type=str,
        required=True,
        help="Fragments TSV filename which contains scATAC fragments.",
    )

    parser_qc_run.add_argument(
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

    parser_qc_run.add_argument(
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

    parser_qc_run.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        action="store",
        type=str,
        required=True,
        help="Output prefix to use for QC statistics parquet output files.",
    )

    parser_qc_run.add_argument(
        "--threads",
        dest="threads",
        action="store",
        type=int,
        required=False,
        default=8,
        help="Number of threads to use when calculating kernel-density estimate (KDE) "
        "to get probability density function (PDF) values for log10 unique fragments "
        "in peaks vs TSS enrichment, fractions of fragments in peaks and duplication "
        "ratio. "
        "Default: 8.",
    )

    parser_qc_run.add_argument(
        "-e",
        "--engine",
        dest="engine",
        action="store",
        type=str,
        choices=["polars", "pyarrow"],
        required=False,
        default="pyarrow",
        help="Use Polars or pyarrow to read BED and fragment files. Default: pyarrow.",
    )

    group_qc_run_tss = parser_qc_run.add_argument_group(
        "TSS profile", "TSS profile statistics calculation settings."
    )
    group_qc_run_tss.add_argument(
        "--tss_flank_window",
        dest="tss_flank_window",
        action="store",
        type=int,
        required=False,
        default=2000,
        help="Flanking window around the TSS. "
        "Used for intersecting fragments with TSS positions and keeping cut sites."
        "Default: 2000 (+/- 2000 bp).",
    )

    group_qc_run_tss.add_argument(
        "--tss_smoothing_rolling_window",
        dest="tss_smoothing_rolling_window",
        action="store",
        type=int,
        required=False,
        default=10,
        help="Rolling window used to smooth the cut sites signal. Default: 10.",
    )

    group_qc_run_tss.add_argument(
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
            Default: 100 (average signal in [-2000, -1901], [1901, 2000] around TSS,
            if flank_window=2000).
            """,
    )

    group_qc_run_tss.add_argument(
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

    group_qc_run_tss.add_argument(
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

    group_qc_run_tss.add_argument(
        "--use-pyranges",
        dest="use_genomic_ranges",
        action="store_false",
        required=False,
        help="""
            Use pyranges instead of genomic ranges implementation for calculating
            intersections.
            """,
    )

    parser_qc_run.add_argument(
        "--min_fragments_per_cb",
        dest="min_fragments_per_cb",
        action="store",
        type=int,
        required=False,
        default=10,
        help="""
            Minimum number of fragments needed per cell barcode to keep the fragments
            for those cell barcodes.
            Default: 10.
            """,
    )

    parser_qc_run.add_argument(
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
