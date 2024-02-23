from __future__ import annotations

import polars as pl

from pycisTopic.fragments import create_pyranges_from_polars_df
from pycisTopic.genomic_ranges import intersection as gr_intersection

# Enable Polars global string cache so all categoricals are created with the same
# string cache.
pl.enable_string_cache()


def get_tss_profile(
    fragments_df_pl: pl.DataFrame,
    tss_annotation: pl.DataFrame,
    flank_window: int = 2000,
    smoothing_rolling_window: int = 10,
    minimum_signal_window: int = 100,
    tss_window: int = 50,
    min_norm: float = 0.2,
    use_genomic_ranges: bool = True,
):
    """
    Get TSS profile for Polars DataFrame with fragments filtered by cell barcodes.

    Parameters
    ----------
    fragments_df_pl
        Polars DataFrame with fragments (filtered by cell barcodes of interest).
        See :func:`pycisTopic.fragments.filter_fragments_by_cb`.
    tss_annotation
        TSS annotation Polars DataFrame with at least the following columns:
        ``["Chromosome", "Start", "Strand"]``.
        The "Start" column is 0-based like a BED file.
        See :func:`pycisTopic.gene_annotation.get_tss_annotation_from_ensembl` and
        :func:`pycisTopic.gene_annotation.change_chromosome_source_in_bed` for ways
        to get TSS annotation from Ensembl BioMart.
    flank_window
        Flanking window around the TSS.
        Used for intersecting fragments with TSS positions and keeping cut sites.
        Default: ``2000`` (+/- 2000 bp).
    smoothing_rolling_window
        Rolling window used to smooth the cut sites signal.
        Default: 10.
    minimum_signal_window
        Average signal in the tails of the flanking window around the TSS:
           - ``[-flank_window, -flank_window + minimum_signal_window + 1]``
           - ``[flank_window - minimum_signal_window + 1, flank_window]``
        is used to normalize the TSS enrichment.
        Default: ``100`` (average signal in ``[-2000, -1901]``, ``[1901, 2000]``
        around TSS if ``flank_window=2000``).
    tss_window
        Window around the TSS used to count fragments in the TSS when calculating
        the TSS enrichment per cell barcode.
        Default: ``50`` (+/- 50 bp).
    min_norm
        Minimum normalization score.
        If the average minimum signal value is below this value, this number is used
        to normalize the TSS signal. This approach penalizes cells with fewer reads.
        Default: ``0.2``
    use_genomic_ranges
        Use genomic ranges implementation for calculating intersections, instead of
        using pyranges.

    Returns
    -------
    tss_enrichment_per_cb, tss_norm_matrix_sample, tss_norm_matrix_per_cb

    See Also
    --------
    pycisTopic.fragments.filter_fragments_by_cb
    pycisTopic.gene_annotation.change_chromosome_source_in_bed
    pycisTopic.gene_annotation.get_tss_annotation_from_ensembl

    Examples
    --------
    Get TSS annotation for requested transcript types from Ensembl BioMart.

    >>> ensembl_tss_annotation_bed_df_pl = get_tss_annotation_from_ensembl(
    ...     biomart_name="hsapiens_gene_ensembl"
    )

    Get TSS profile for Polars DataFrame with fragments filtered by cell barcodes.

    >>> get_tss_profile(
    ...     fragments_df_pl=fragments_cb_filtered_df_pl,
    ...     tss_annotation=ensembl_tss_annotation_bed_df_pl,
    ...     flank_window=2000,
    ...     smoothing_rolling_window=10,
    ...     minimum_signal_window=100,
    ...     tss_window=50,
    ...     min_norm=0.2,
    ... )

    """
    # Extend TSS position with flanking window and only keep minimal necessary columns
    # needed to find the overlap with fragments.
    tss_annotation_with_flanking_window_df_pl = (
        tss_annotation.select(
            # Only keep needed columns for faster Genomics Ranges / PyRanges join.
            pl.col("Chromosome").cast(pl.Categorical),
            pl.col("Start"),
            pl.col("Strand"),
        )
        # Filter out TSS annotations without strand info
        # (in case that would ever happen).
        .filter(pl.col("Strand") != ".")
        # Create [-flank_window, flank_window] around TSS
        # (size: flank_window * 2 + 1) and set them as "Start" and "End" column.
        .with_columns(
            (pl.col("Start") - flank_window).alias("Start"),
            (pl.col("Start") + flank_window + 1).alias("End"),
        )
    )

    # Get all chromosome names which are found in the TSS annotation.
    annotation_chromosomes: pl.Series = (
        tss_annotation_with_flanking_window_df_pl.clone()
        .select(pl.col("Chromosome").unique())
        .get_column("Chromosome")
    )

    # Filter out fragments which are located on chromosomes not found in the TSS
    # annotation and remove unneeded columns.
    filtered_fragments_df_pl = (
        fragments_df_pl.clone()
        .filter(
            # Filter out fragments which are located on chromosomes not found in
            # the TSS annotation.
            pl.col("Chromosome").is_in(annotation_chromosomes)
        )
        .select(
            # Only keep chromosomal position and CB.
            pl.col(["Chromosome", "Start", "End", "Name"])
        )
    )

    # Get overlap between fragments and TSS positions
    # with [-flank_window, flank_window].
    overlap_with_tss_df_pl = (
        # Use genomic_ranges to calculate the intersection.
        gr_intersection(
            regions1_df_pl=filtered_fragments_df_pl,
            regions2_df_pl=tss_annotation_with_flanking_window_df_pl,
            regions1_info=True,
            regions2_info=True,
            regions1_coord=True,
            regions2_coord=True,
            regions1_suffix="_fragment",
            # Add "_tss_flank" suffix for joined output that comes from the TSS
            # annotation BED file.
            regions2_suffix="_tss_flank",
        ).rename(
            {
                "Strand_tss_flank": "Strand",
            }
        )
        if use_genomic_ranges
        else
        # Use pyranges to calculate the intersection.
        pl.from_pandas(
            (
                # Create PyRanges object from filtered fragments Polars DataFrame.
                create_pyranges_from_polars_df(filtered_fragments_df_pl).join(
                    # Create PyRanges object from TSS annotation Polars DataFrame
                    # extended with flanking window.
                    create_pyranges_from_polars_df(
                        tss_annotation_with_flanking_window_df_pl
                    ),
                    # Add "_tss_flank" suffix for joined output that comes from the TSS
                    # annotation BED file.
                    suffix="_tss_flank",
                    apply_strand_suffix=False,
                )
            ).df
        ).rename({"Start": "Start_fragment", "End": "End_fragment"})
    )

    if overlap_with_tss_df_pl.shape == (0, 0):
        raise ValueError(
            "No overlap found between fragments and TSS annotation. Make sure both use the same chromosome name source."
        )

    # Get cut sites (start/end fragments) relative to the TSS position taking into
    # account the strand info of the TSS.
    #   - `-x`: cut site x bp upstream of TSS.
    #   - `0`: cut site at TSS.
    #   - `x`: cut site x bp downstream of TSS.
    cut_sites_tss_start_end = (
        overlap_with_tss_df_pl.clone()
        .with_columns(
            # Fragment start, fragment end and TSS position in 0-based coordinates.
            pl.col("Start_fragment").alias("fragment_start"),
            (pl.col("End_fragment") - 1).alias("fragment_end"),
            (pl.col("Start_tss_flank") + flank_window).alias("tss"),
            pl.col("Strand"),
        )
        .with_columns(
            pl.when(pl.col("Strand") == "-")
            .then(
                # TSS is for a gene on the negative strand.
                pl.struct(
                    [
                        # Calculate relative start and end of cut site to the TSS.
                        (pl.col("tss") - pl.col("fragment_end")).alias("rel_start"),
                        (pl.col("tss") - pl.col("fragment_start")).alias("rel_end"),
                    ]
                )
            )
            .otherwise(
                # TSS is for a gene on the positive strand.
                pl.struct(
                    [
                        # Calculate relative start and end of cut site to the TSS.
                        (pl.col("fragment_start") - pl.col("tss")).alias("rel_start"),
                        (pl.col("fragment_end") - pl.col("tss")).alias("rel_end"),
                    ]
                )
            )
            .alias("rel_start_end")
        )
        # Unnest "rel_start_end" struct column (creates: "rel_start" and "rel_end").
        .unnest("rel_start_end")
        .select(
            pl.col("rel_start"),
            pl.col("rel_end"),
            pl.col("Name").alias("CB"),
        )
    )

    # Get TSS matrix:
    #   - columns: cut site positions relative to TSS (TSS = 0).
    #   - rows: CBs
    #   - values: number of times a cut site position was found for a certain CB.
    tss_matrix_tmp = (
        # Get all cut site positions which fall in [-flank_window, flank_window]
        # (size: flank_window * 2 + 1):
        #   - Some fragments will have both cut sites in this interval.
        #   - Some fragments have only one cut site in this interval (start or end).
        pl.concat(
            [
                pl.DataFrame(
                    [
                        # Create [-flank_window, flank_window] range for all possible
                        # cut site positions.
                        pl.arange(
                            start=-flank_window,
                            end=flank_window + 1,
                            step=1,
                            eager=True,
                            dtype=pl.Int32,
                        ).alias("position_from_tss"),
                        # Create a CB column with all "no_CB" values.
                        # Needed temporarily so during the pivot operation all cut site
                        # position values ([-flank_window, flank_window] range) are
                        # kept even if there are no cut sites for certain positions.
                        pl.Series(
                            "CB",
                            ["no_CB"] * (flank_window * 2 + 1),
                            dtype=pl.Categorical,
                        ),
                    ]
                ),
                # Get all cut sites for the relative start that pass the filter.
                cut_sites_tss_start_end.filter(
                    pl.col("rel_start").abs() <= flank_window
                ).select(
                    # Get cut site position and associated CB.
                    pl.col("rel_start").alias("position_from_tss"),
                    pl.col("CB"),
                ),
                # Get all cut sites for the relative end that pass the filter.
                cut_sites_tss_start_end.filter(
                    pl.col("rel_end").abs() <= flank_window
                ).select(
                    # Get cut site position and associated CB.
                    pl.col("rel_end").alias("position_from_tss"),
                    pl.col("CB"),
                ),
            ],
        )
        # Count number of cut sites with the same CB and same position per CB.
        .pivot(
            values="position_from_tss",
            index="CB",
            columns="position_from_tss",
            aggregate_function="count",
        )
        # Remove "no_CB" cell barcode (was only needed for the pivot).
        .filter(pl.col("CB") != "no_CB").with_columns(
            # Fill in 0, for non-observed values in the pivot table.
            pl.col(pl.UInt32)
            .cast(pl.Int32)
            .fill_null(0),
        )
    )

    # Get TSS matrix:
    #   - columns: CBs
    #   - rows: cut site positions relative to TSS.
    #   - values: number of times a cut site position was found for a certain CB.
    tss_matrix = (
        tss_matrix_tmp.clone()
        # Remove "CB" column, so numeric values of the TSS matrix can be transposed.
        .drop("CB")
        # Transpose TSS matrix:
        #    - columns: CBs
        #    - rows: cut site positions relative to TSS.
        .transpose(
            # Keep cut sites position values (will be dtype=pl.Utf8) as the first
            # column.
            include_header=True,
            header_name="position_from_tss",
            # Add old "CB" column as column names.
            column_names=tss_matrix_tmp.get_column("CB"),
        ).with_columns(
            # Convert "position_from_tss" column from pl.Utf8 to pl.Int32.
            pl.col("position_from_tss").cast(pl.Int32)
        )
    )

    # Remove raw non-transposed TSS matrix.
    del tss_matrix_tmp

    # Smooth TSS matrix per CB by a rolling window.
    tss_smoothed_matrix_per_cb = tss_matrix.with_columns(
        pl.col("position_from_tss"),
        pl.all()
        .exclude("position_from_tss")
        .rolling_mean(window_size=smoothing_rolling_window, min_periods=0),
    )

    # Remove raw TSS matrix.
    del tss_matrix

    # Normalize smoothed TSS matrix.
    # Get normalized sample TSS enrichment per position from the per CB
    # smoothed TSS matrix.
    tss_norm_matrix_sample = tss_smoothed_matrix_per_cb.select(
        pl.col("position_from_tss"),
        # Get total number of cut sites per position over all CBs.
        pl.sum_horizontal(pl.all().exclude("position_from_tss")).alias(
            "smoothed_per_pos_sum"
        ),
    ).select(
        pl.col("position_from_tss"),
        # Normalize total number of cut sites per position over all CBs.
        (
            pl.col("smoothed_per_pos_sum")
            / (
                # Calculate background value from start and end over
                # minimum_signal_window length.
                (
                    (
                        pl.col("smoothed_per_pos_sum")
                        .head(minimum_signal_window)
                        .mean()
                        + pl.col("smoothed_per_pos_sum")
                        .tail(minimum_signal_window)
                        .mean()
                    )
                    / 2
                )
                # Or use min_norm.
                .append(min_norm)
                # Take highest value.
                .max()
            )
        ).alias("normalized_tss_enrichment"),
    )

    # Get normalized TSS matrix per CB for each cut site position.
    tss_norm_matrix_per_cb = tss_smoothed_matrix_per_cb.with_columns(
        [
            (
                pl.col(CB)
                / (
                    # Calculate background value from start and end over
                    # minimum_signal_window length.
                    (
                        (
                            pl.col(CB).head(minimum_signal_window).mean()
                            + pl.col(CB).tail(minimum_signal_window).mean()
                        )
                        / 2
                    )
                    # Or use min_norm.
                    .append(min_norm)
                    # Take highest value.
                    .max()
                )
            ).alias(CB)
            for CB in tss_smoothed_matrix_per_cb.columns[1:]
        ]
    )

    # Calculate TSS enrichment per CB.
    tss_enrichment_per_cb = (
        tss_norm_matrix_per_cb.clone()
        .filter(
            pl.col("position_from_tss").is_between(
                lower_bound=-tss_window,
                upper_bound=tss_window,
                closed="both",
            )
        )
        .drop("position_from_tss")
        .select(pl.all().mean())
        .transpose(include_header=True, header_name="CB")
        .with_columns(pl.col("CB").cast(pl.Categorical))
        .rename({"column_0": "tss_enrichment"})
    )

    return tss_enrichment_per_cb, tss_norm_matrix_sample, tss_norm_matrix_per_cb
