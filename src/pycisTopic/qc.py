from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from pycisTopic.fragments import (
    get_fragments_in_peaks,
    get_fragments_per_cb,
    get_insert_size_distribution,
)
from pycisTopic.topic_binarization import threshold_otsu
from pycisTopic.tss_profile import get_tss_profile
from scipy.stats import gaussian_kde

if TYPE_CHECKING:
    import numpy.typing as npt

# Enable Polars global string cache so all categoricals are created with the same
# string cache.
pl.enable_string_cache()


def get_barcodes_passing_qc_for_sample(
    sample_id: str,
    pycistopic_qc_output_dir: str | Path,
    unique_fragments_threshold: int | None = None,
    tss_enrichment_threshold: float | None = None,
    frip_threshold: float | None = None,
    use_automatic_thresholds: bool = True,
) -> tuple[list[str], dict[str, float]]:
    """
    Get barcodes passing quality control (QC) for a sample.

    Parameters
    ----------
    sample_id
        Sample ID.
    pycistopic_qc_output_dir
        Directory with output from ``pycistopic qc``.
    unique_fragments_threshold
        Threshold for number of unique fragments in peaks.
        If not defined, and use_automatic_thresholds is False,
        the threshold will be set to 0.
    tss_enrichment_threshold
        Threshold for TSS enrichment score.
        If not defined, and use_automatic_thresholds is False,
        the threshold will be set to 0.0.
    frip_threshold
        Threshold for fraction of reads in peaks (FRiP).
        If not defined the threshold will be set to 0.0.
    use_automatic_thresholds
        Use automatic thresholds for unique fragments in peaks and TSS enrichment score
        as calculated by Otsu's method. If False, the thresholds will be set to 0 if not
        defined.

    Returns
    -------
    Tuple with:
        - List with cell barcodes passing QC.
        - Dictionary with thresholds used for QC.

    Raises
    ------
    FileNotFoundError
        If the file with fragments statistics per cell barcode does not exist.

    """
    print(f"{sample_id}:")

    if use_automatic_thresholds:
        otsu_thresholds_tsv_filename = os.path.join(
            pycistopic_qc_output_dir,
            f"{sample_id}.otsu_thresholds.tsv",
        )

        # Check whether files exist.
        if not os.path.exists(otsu_thresholds_tsv_filename):
            Warning(f'File "{otsu_thresholds_tsv_filename}" does not exist.')
        else:
            # Read automatic thresholds.
            (
                otsu_unique_fragments_threshold,
                otsu_tss_enrichment_threshold,
            ) = pl.read_csv(
                otsu_thresholds_tsv_filename,
                separator="\t",
                columns=[
                    "unique_fragments_in_peaks_count_otsu_threshold",
                    "tss_enrichment_otsu_threshold",
                ],
            ).row(0)

            print(f"{sample_id}:")

            if unique_fragments_threshold is None:
                print(
                    f"\tUsing automatic threshold for unique fragments: {otsu_unique_fragments_threshold}"
                )
                unique_fragments_threshold = otsu_unique_fragments_threshold
            else:
                print(
                    f"\tUsing user-defined threshold for unique fragments: {unique_fragments_threshold}"
                )

            if tss_enrichment_threshold is None:
                print(
                    f"\tUsing automatic threshold for TSS enrichment: {otsu_tss_enrichment_threshold}"
                )
                tss_enrichment_threshold = otsu_tss_enrichment_threshold
            else:
                print(
                    f"\tUsing user-defined threshold for TSS enrichment: {tss_enrichment_threshold}"
                )

    if (
        unique_fragments_threshold is None
        or tss_enrichment_threshold is None
        or frip_threshold is None
    ):
        if not use_automatic_thresholds:
            print(f"{sample_id}:")

        # Set thresholds to 0 if not defined.
        if unique_fragments_threshold is None:
            print("\tNo threshold for unique fragments defined, setting to 0.")
            unique_fragments_threshold = 0

        if tss_enrichment_threshold is None:
            print("\tNo threshold for TSS enrichment defined, setting to 0.0.")
            tss_enrichment_threshold = 0.0

        if frip_threshold is None:
            print("\tNo threshold for FRiP defined, setting to 0.0.")
            frip_threshold = 0.0

    # fragments_stats_per_cb_df_pl
    fragments_stats_per_cb_filename = os.path.join(
        pycistopic_qc_output_dir,
        f"{sample_id}.fragments_stats_per_cb.parquet",
    )

    # Check whether files exist.
    if not os.path.exists(fragments_stats_per_cb_filename):
        raise FileNotFoundError(
            f'File "{fragments_stats_per_cb_filename}" does not exist.'
        )

    # Get barcodes passing filters.
    barcodes_passing_filters = (
        pl.scan_parquet(fragments_stats_per_cb_filename)
        .filter(
            (pl.col("unique_fragments_in_peaks_count") > unique_fragments_threshold)
            & (pl.col("tss_enrichment") > tss_enrichment_threshold)
            & (pl.col("fraction_of_fragments_in_peaks") > frip_threshold)
        )
        .select("CB")
        .collect()
        .to_series()
        .to_list()
    )

    return barcodes_passing_filters, {
        "unique_fragments_threshold": unique_fragments_threshold,
        "tss_enrichment_threshold": tss_enrichment_threshold,
        "frip_threshold": frip_threshold,
    }


def compute_kde(
    training_data: npt.ArrayLike,
    test_data: npt.ArrayLike,
    no_threads: int = 8,
) -> npt.NDArray[np.float64]:
    """
    Compute kernel-density estimate (KDE) using Gaussian kernels.

    This function calculates the KDE in parallel and gives the same result as:

    >>> from scipy.stats import gaussian_kde
    >>> gaussian_kde(training_data)(test_data)

    Parameters
    ----------
    training_data
        2D numpy array with training data to train the KDE.
    test_data
        2D numpy array with test data for which to evaluate the estimated probability
        density function (PDF).
    no_threads
        Number of threads to use in parallelization of KDE function.

    Returns
    -------
    1D numpy array with probability density function (PDF) values for points in
    test_data.

    """
    training_data = np.asarray(training_data, dtype=np.float64)
    test_data = np.asarray(test_data, dtype=np.float64)

    # Avoid very rare cases where second column of training_data has the same
    # value everywhere. This can happen in some cases for duplication ratio as
    # it can be 0.0% when fragment counts for each fragment are 1.
    #
    # This will result in the following error:
    #     LinAlgError: The data appears to lie in a lower-dimensional subspace of
    #     the space in which it is expressed. This has resulted in a singular data
    #     covariance matrix, which cannot be treated using the algorithms implemented
    #     in `gaussian_kde`. Consider performing principle component analysis /
    #     dimensionality reduction and using `gaussian_kde` with the transformed data.
    if np.var(training_data[1]) == 0.0:
        # Add small value to first element to avoid all of them to be equal.
        if training_data[1][0] == 0.0:
            training_data[1][0] = 0.000000000001
        else:
            # In even rarer case that the value is not 0.0, change the value proportionally.
            training_data[1][0] = training_data[1][0] * 1.000000000001

    # Convert 2D numpy array test data to complex number array so numpy considers both
    # columns at the same time in further operations.
    test_data_all = np.empty(test_data.shape[1], dtype=np.complex128)
    test_data_all.real = test_data[0]
    test_data_all.imag = test_data[1]

    # Get unique values for test data considering whole rows of the original test_data.
    # The KDE calculation only needs to be done for unique values in the test data.
    test_data_unique = np.unique(test_data_all)

    # Get index position locations in test_data_unique for each value in test_data_all.
    test_data_original_order_idx = np.searchsorted(test_data_unique, test_data_all)

    # Split the array of unique test data values in one array for each thread.
    test_data_unique_split_arrays = np.array_split(
        np.vstack([test_data_unique.real, test_data_unique.imag]),
        no_threads,
        axis=1,
    )

    def compute_kde_part(
        test_data_unique_split_array: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Compute kernel-density estimate (KDE) using Gaussian kernels for a subsection of the test_data.

        Parameters
        ----------
        test_data_unique_split_array
            2D numpy array with a part of the test data for which to evaluate the
            estimated probability density function (PDF).

        Returns
        -------
        1D numpy array with probability density function (PDF) values for points in
        test_data.

        """
        return gaussian_kde(training_data)(test_data_unique_split_array)

    pdf_results = []

    # Calculate KDE for each subsection of the test dataStart the thread pool.
    with ThreadPoolExecutor(no_threads) as executor:
        # Execute tasks concurrently and process results in order.
        for pdf_result in executor.map(compute_kde_part, test_data_unique_split_arrays):
            # Store partial PDF result.
            pdf_results.append(pdf_result)

    # Get PDF values from the partial PDF results calculated on unique values in the
    # test data and construct a full PDF values array that matches the order of the
    # original test data.
    pdf_values = np.concatenate(pdf_results)[test_data_original_order_idx]

    return pdf_values


def get_otsu_threshold(
    fragments_stats_per_cb_df_pl: pl.DataFrame,
    min_otsu_fragments: int = 100,
    min_otsu_tss: float = 1.0,
):
    """
    Get Otsu thresholds for number of unique fragments in peaks and TSS enrichment score.

    Parameters
    ----------
    fragments_stats_per_cb_df_pl
        Polars DataFrame with fragments statistics per cell barcode as generated by
        :func:`pycisTopic.qc.compute_qc_stats`.
    min_otsu_fragments
        When calculating Otsu threshold for number of unique fragments in peaks per CB,
        only consider those CBs which have at least this number of fragments.
    min_otsu_tss
        When calculating Otsu threshold for TSS enrichment score per CB, only consider
        those CBs which have at least this TSS value.

    Returns
    -------
    Tuple with:
      - Otsu threshold for number of unique fragments in peaks.
      - Otsu threshold for TSS enrichment.
      - Polars DataFrame with fragments statistics per cell barcode for cell barcodes
        that passed both Otsu thresholds.

    Examples
    --------
    Only keep fragments stats for CBs that pass both Otsu thresholds.
    >>> (
    ...     unique_fragments_in_peaks_count_otsu_threshold,
    ...     tss_enrichment_otsu_threshold,
    ...     fragments_stats_per_cb_for_otsu_threshold_df_pl,
    ... ) = get_otsu_threshold(
    ...     fragments_stats_per_cb_df_pl=fragments_stats_per_cb_df_pl,
    ...     min_otsu_fragments=100,
    ...     min_otsu_tss=1.0,
    ... )

    """
    # Get number (in log10) of unique fragments in peaks per CB and TSS enrichment
    # score per CB.
    log10_unique_fragments_in_peaks_count, tss_enrichment = (
        fragments_stats_per_cb_df_pl.select(
            "log10_unique_fragments_in_peaks_count", "tss_enrichment"
        )
        .to_numpy()
        .T
    )

    # Get Otsu threshold for number of unique fragments in peaks per CB.
    unique_fragments_in_peaks_count_otsu_threshold = 10 ** threshold_otsu(
        log10_unique_fragments_in_peaks_count[
            log10_unique_fragments_in_peaks_count >= math.log10(min_otsu_fragments)
        ],
        nbins=5000,
    )

    # Get Otsu threshold for TSS enrichment score per CB.
    tss_enrichment_otsu_threshold = threshold_otsu(
        tss_enrichment[tss_enrichment >= min_otsu_tss],
        nbins=5000,
    )

    # Only keep fragments stats for CBs that pass both Otsu thresholds.
    fragments_stats_per_cb_for_otsu_threshold_df_pl = (
        fragments_stats_per_cb_df_pl.filter(
            (
                pl.col("unique_fragments_in_peaks_count")
                > unique_fragments_in_peaks_count_otsu_threshold
            )
            & (pl.col("tss_enrichment") > tss_enrichment_otsu_threshold)
        )
    )

    return (
        unique_fragments_in_peaks_count_otsu_threshold,
        tss_enrichment_otsu_threshold,
        fragments_stats_per_cb_for_otsu_threshold_df_pl,
    )


def compute_qc_stats(
    fragments_df_pl: pl.DataFrame,
    regions_df_pl: pl.DataFrame,
    tss_annotation: pl.DataFrame,
    tss_flank_window: int = 2000,
    tss_smoothing_rolling_window: int = 10,
    tss_minimum_signal_window: int = 100,
    tss_window: int = 50,
    tss_min_norm: float = 0.2,
    use_genomic_ranges: bool = True,
    min_fragments_per_cb: int = 10,
    collapse_duplicates: bool = True,
    no_threads: int = 8,
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
    no_threads
        Number of threads to use when calculating kernel-density estimate (KDE) to get
        probability density function (PDF) values for log10 unique fragments in peaks
        vs TSS enrichment, fractions of fragments in peaks and duplication ratio.
        Default: ``8``

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
    ...     no_threads=8,
    ... )

    """
    logger = logging.getLogger(__name__)

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
    logger.info("Get basic fragments statistics per cell barcode.")
    fragments_stats_per_cb_df_pl = get_fragments_per_cb(
        fragments_df_pl=fragments_df_pl,
        min_fragments_per_cb=min_fragments_per_cb,
        collapse_duplicates=collapse_duplicates,
    )

    # Get Polars DataFrame with total fragment counts and unique fragment counts
    # per region.
    logger.info("Get total fragment counts and unique fragment counts per region.")
    fragments_in_peaks_df_pl = get_fragments_in_peaks(
        fragments_df_pl=fragments_df_pl,
        regions_df_pl=regions_df_pl,
    )

    # Add fragment counts per region to fragments statistics per cell barcode.
    logger.info(
        "Add fragment counts per region to fragments statistics per cell barcode."
    )
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
            .alias("log10_total_fragments_count"),
            pl.col("unique_fragments_count"),
            (pl.col("unique_fragments_count") + 1)
            .log10()
            .alias("log10_unique_fragments_count"),
            pl.col("total_fragments_in_peaks_count"),
            (pl.col("total_fragments_in_peaks_count") + 1)
            .log10()
            .alias("log10_total_fragments_in_peaks_count"),
            pl.col("unique_fragments_in_peaks_count"),
            (pl.col("unique_fragments_in_peaks_count") + 1)
            .log10()
            .alias("log10_unique_fragments_in_peaks_count"),
            pl.col("fraction_of_fragments_in_peaks"),
            pl.col("duplication_count"),
            pl.col("duplication_ratio"),
            pl.col("nucleosome_signal"),
        )
    )

    # Get insert size distribution of fragments.
    logger.info("Get insert size distribution of fragments.")
    insert_size_dist_df_pl = get_insert_size_distribution(
        fragments_df_pl=fragments_df_pl,
    )

    # Get TSS profile for fragments.
    logger.info("Get TSS profile for fragments.")
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
    logger.info("Add TSS enrichment to fragments statistics per cell barcode.")
    fragments_stats_per_cb_df_pl = (
        fragments_stats_per_cb_df_pl.join(
            tss_enrichment_per_cb.lazy(),
            how="left",
            on="CB",
        )
        .with_columns(
            pl.col("tss_enrichment").fill_null(0.0),
        )
        .collect()
    )

    # Extract certain columns as numpy arrays as they are needed for calculating KDE.
    (
        log10_unique_fragments_in_peaks_count,
        tss_enrichment,
        fraction_of_fragments_in_peaks,
        duplication_ratio,
    ) = (
        fragments_stats_per_cb_df_pl.select(
            [
                pl.col("log10_unique_fragments_in_peaks_count"),
                pl.col("tss_enrichment"),
                pl.col("fraction_of_fragments_in_peaks"),
                pl.col("duplication_ratio"),
            ]
        )
        .to_numpy()
        .T
    )

    # Construct 2D numpy matrices for usage with compute_kde.
    kde_data_for_tss_enrichment = np.vstack(
        [log10_unique_fragments_in_peaks_count, tss_enrichment]
    )
    kde_data_for_fraction_of_fragments_in_peaks = np.vstack(
        [log10_unique_fragments_in_peaks_count, fraction_of_fragments_in_peaks]
    )
    kde_data_for_duplication_ratio = np.vstack(
        [log10_unique_fragments_in_peaks_count, duplication_ratio]
    )

    # Calculate KDE for log10 unique fragments in peaks vs TSS enrichment,
    # fractions of fragments in peaks and duplication ratio.
    logger.info("Calculate KDE for log10 unique fragments in peaks vs TSS enrichment.")
    pdf_values_for_tss_enrichment = compute_kde(
        training_data=kde_data_for_tss_enrichment,
        test_data=kde_data_for_tss_enrichment,
        no_threads=no_threads,
    )
    logger.info(
        "Calculate KDE for log10 unique fragments in peaks vs fractions of fragments "
        "in peaks."
    )
    pdf_values_for_fraction_of_fragments_in_peaks = compute_kde(
        training_data=kde_data_for_fraction_of_fragments_in_peaks,
        test_data=kde_data_for_fraction_of_fragments_in_peaks,
        no_threads=no_threads,
    )
    logger.info(
        "Calculate KDE for log10 unique fragments in peaks vs duplication ratio."
    )
    pdf_values_for_duplication_ratio = compute_kde(
        training_data=kde_data_for_duplication_ratio,
        test_data=kde_data_for_duplication_ratio,
        no_threads=no_threads,
    )

    # Add probability density function (PDF) values for log10 unique fragments in peaks
    # vs TSS enrichment, fractions of fragments in peaks and duplication ratio to
    # fragments statistics per cell barcode.
    logger.info(
        "Add probability density function (PDF) values to fragments statistics per "
        "cell barcode."
    )
    fragments_stats_per_cb_df_pl = fragments_stats_per_cb_df_pl.hstack(
        pl.DataFrame(
            {
                "pdf_values_for_tss_enrichment": pdf_values_for_tss_enrichment,
                "pdf_values_for_fraction_of_fragments_in_peaks": pdf_values_for_fraction_of_fragments_in_peaks,
                "pdf_values_for_duplication_ratio": pdf_values_for_duplication_ratio,
            }
        )
    )

    return (
        fragments_stats_per_cb_df_pl,
        insert_size_dist_df_pl,
        tss_norm_matrix_sample,
        tss_norm_matrix_per_cb,
    )
