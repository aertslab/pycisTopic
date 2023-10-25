from __future__ import annotations

import polars as pl

from pycisTopic.fragments import (
    get_fragments_in_peaks,
    get_fragments_per_cb,
    get_insert_size_distribution,
)
from pycisTopic.tss_profile import get_tss_profile

# Enable Polars global string cache so all categoricals are created with the same
# string cache.
pl.enable_string_cache()


def compute_qc_stats(
    fragments_df_pl: pl.DataFrame,
    regions_df_pl: pl.DataFrame,
    tss_annotation: pl.DataFrame,
    tss_flank_window: int = 2000,
    tss_smoothing_rolling_window: int = 10,
    tss_minimum_signal_window: int = 100,
    tss_window: int = 50,
    tss_min_norm: int = 0.2,
    use_genomic_ranges: bool = True,
    min_fragments_per_cb: int = 10,
    collapse_duplicates: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Compute quality check statistics from Polars DataFrame with fragments.

    Parameters
    ----------
    fragments_df_pl
        Polars DataFrame with fragments.
        fragments_df_pl
        Polars DataFrame with fragments (filtered by cell barcodes of interest).
        See :func:`pycisTopic.fragments.filter_fragments_by_cb`.
    regions_df_pl
        Polars DataFrame with peak regions (consensus peaks or SCREEN regions).
        See :func:`pycisTopic.fragments.read_bed_to_polars_df` for a way to read a BED
        file with peak regions.
    tss_annotation
        TSS annotation Polars DataFrame with at least the following columns:
        ``["Chromosome", "Start", "Strand"]``.
        The "Start" column is 0-based like a BED file.
        See :func:`pycisTopic.gene_annotation.read_tss_annotation_from_bed`,
        :func:`pycisTopic.gene_annotation.get_tss_annotation_from_ensembl` and
        :func:`pycisTopic.gene_annotation.change_chromosome_source_in_bed` for ways
        to get TSS annotation from Ensembl BioMart.
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
        around TSS if ``flank_window=2000``).
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
    Tuple with:
      - Polars DataFrame with fragments statistics per cell barcode.
      - Polars DataFrame with insert size distribution of fragments.
      - Polars DataFrame with TSS normalization matrix for the whole sample.
      - Polars DataFrame with TSS normalization matrix per cell barcode.

    See Also
    --------
    pycisTopic.fragments.filter_fragments_by_cb
    pycisTopic.fragments.get_insert_size_distribution
    pycisTopic.fragments.get_fragments_in_peaks
    pycisTopic.fragments.read_bed_to_polars_df
    pycisTopic.fragments.read_fragments_to_polars_df
    pycisTopic.gene_annotation.read_tss_annotation_from_bed
    pycisTopic.tss_profile.get_tss_profile

    Examples
    --------
    >>> from pycisTopic.fragments import read_bed_to_polars_df
    >>> from pycisTopic.fragments import read_fragments_to_polars_df
    >>> from pycisTopic.gene_annotation import read_tss_annotation_from_bed

    1. Read gzipped fragments BED file to a Polars DataFrame.

    >>> fragments_df_pl = read_fragments_to_polars_df(
    ...     fragments_bed_filename="fragments.tsv.gz",
    ... )

    2. Read BED file with consensus peaks or SCREEN regions (get first 3 columns only)
       which will be used for counting number of fragments in peaks.

    >>> regions_df_pl = read_bed_to_polars_df(
    ...     bed_filename=screen_regions_bed_filename,
    ...     min_column_count=3,
    ... )

    3. Read TSS annotation from a file.
       See :func:`pycisTopic.gene_annotation.read_tss_annotation_from_bed` for more
       info.

    >>> tss_annotation_bed_df_pl = read_tss_annotation_from_bed(
    ...     tss_annotation_bed_filename="hg38.tss.bed",
    ... )

    4. Compute QC statistics.

    >>> (
    ...     fragments_stats_per_cb_df_pl,
    ...     insert_size_dist_df_pl,
    ...     tss_norm_matrix_sample,
    ...     tss_norm_matrix_per_cb,
    ... ) = compute_qc_stats(
    ...     fragments_df_pl=fragments_cb_filtered_df_pl,
    ...     regions_df_pl=regions_df_pl,
    ...     tss_annotation=tss_annotation_bed_df_pl,
    ...     tss_flank_window=2000,
    ...     tss_smoothing_rolling_window=10,
    ...     tss_minimum_signal_window=100,
    ...     tss_window=50,
    ...     tss_min_norm=0.2,
    ...     use_genomic_ranges=True,
    ...     min_fragments_per_cb=10,
    ...     collapse_duplicates=True,
    ... )

    """
    # Define correct column to get, based on the setting of `collapse_duplicates`.
    fragments_count_column = (
        "unique_fragments_count" if collapse_duplicates else "total_fragments_count"
    )
    fragments_in_peaks_count_column = (
        "unique_fragments_in_peaks_count"
        if collapse_duplicates
        else "total_fragments_in_peaks_count"
    )

    # Get Polars DataFrame with basic fragments statistics per cell barcode.
    fragments_stats_per_cb_df_pl = get_fragments_per_cb(
        fragments_df_pl=fragments_df_pl,
        min_fragments_per_cb=min_fragments_per_cb,
        collapse_duplicates=collapse_duplicates,
    )

    # Get Polars DataFrame with total fragment counts and unique fragment counts
    # per region.
    fragments_in_peaks_df_pl = get_fragments_in_peaks(
        fragments_df_pl=fragments_df_pl,
        regions_df_pl=regions_df_pl,
    )

    # Add fragment counts per region to fragments statistics per cell barcode.
    fragments_stats_per_cb_df_pl = (
        fragments_stats_per_cb_df_pl.lazy()
        .join(
            fragments_in_peaks_df_pl.lazy(),
            how="left",
            on="CB",
        )
        .with_columns(
            pl.col("total_fragments_in_peaks_count").fill_null(0),
            pl.col("unique_fragments_in_peaks_count").fill_null(0),
        )
        .with_columns(
            (
                pl.col(fragments_in_peaks_count_column) / pl.col(fragments_count_column)
            ).alias("fraction_of_fragments_in_peaks")
        )
        .select(
            pl.col("CB"),
            pl.col("barcode_rank"),
            pl.col("total_fragments_count"),
            (pl.col("total_fragments_count") + 1)
            .log10()
            .alias("log_total_fragments_count"),
            pl.col("unique_fragments_count"),
            (pl.col("unique_fragments_count") + 1)
            .log10()
            .alias("log_unique_fragments_count"),
            pl.col("total_fragments_in_peaks_count"),
            (pl.col("total_fragments_in_peaks_count") + 1)
            .log10()
            .alias("log_total_fragments_in_peaks_count"),
            pl.col("unique_fragments_in_peaks_count"),
            (pl.col("unique_fragments_in_peaks_count") + 1)
            .log10()
            .alias("log_unique_fragments_in_peaks_count"),
            pl.col("fraction_of_fragments_in_peaks"),
            pl.col("duplication_count"),
            pl.col("duplication_ratio"),
        )
    )

    # Get insert size distribution of fragments.
    insert_size_dist_df_pl = get_insert_size_distribution(
        fragments_df_pl=fragments_df_pl,
    )

    # Get TSS profile for fragments.
    (
        tss_enrichment_per_cb,
        tss_norm_matrix_sample,
        tss_norm_matrix_per_cb,
    ) = get_tss_profile(
        fragments_df_pl=fragments_df_pl,
        tss_annotation=tss_annotation,
        flank_window=tss_flank_window,
        smoothing_rolling_window=tss_smoothing_rolling_window,
        minimum_signal_window=tss_minimum_signal_window,
        tss_window=tss_window,
        min_norm=tss_min_norm,
        use_genomic_ranges=use_genomic_ranges,
    )

    # Add TSS enrichment to fragments statistics per cell barcode.
    fragments_stats_per_cb_df_pl = (
        fragments_stats_per_cb_df_pl.join(
            tss_enrichment_per_cb.lazy(),
            how="left",
            on="CB",
        )
        .with_columns(
            pl.col("tss_enrichment"),
        )
        .collect()
    )

    return (
        fragments_stats_per_cb_df_pl,
        insert_size_dist_df_pl,
        tss_norm_matrix_sample,
        tss_norm_matrix_per_cb,
    )
