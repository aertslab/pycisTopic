from __future__ import annotations

import polars as pl

from fragments import create_pyranges_from_polars_df


def get_tss_profile(
    fragments_df_pl: pl.DataFrame,
    tss_annotation: pl.DataFrame,
    flank_window=1000,
    window_size=10,
    minimum_signal_window=100,
    tss_window=50,
    min_norm=0.2,
):
    """
    Get TSS profile.

    Parameters
    ----------
    fragments_df_pl: pl.DataFrame,
    tss_annotation: pl.DataFrame,
    flank_window=1000,
    window_size=10,
    minimum_signal_window=100,
    tss_window=50,
    min_norm=0.2

    Returns
    -------

    Examples
    --------

    """

    # Get chromosome names which are found in the TSS annotation.
    annotation_chromosomes: pl.Series = (
        tss_annotation.clone()
        .select(pl.col("Chromosome").unique())
        .get_column("Chromosome")
    )

    # Filter out fragments which are located on chromosomes not found in the TSS annotation and remove unneeded columns.
    filtered_fragments_df_pl = (
        fragments_df_pl.clone()
        .filter(
            # Filter out fragments which are located on chromosomes not found in the TSS annotation.
            pl.col("Chromosome").is_in(annotation_chromosomes)
        )
        .select(
            # Only keep chromosomal position and CB.
            pl.col(["Chromosome", "Start", "End", "Name"])
        )
    )

    # Get overlap between fragments and TSS positions with [-flank_window, flank_window].
    overlap_with_tss_df_pl = pl.from_pandas(
        (
            # Create PyRanges object from filtered fragments file and overlap with TSS annotation BED file.
            create_pyranges_from_polars_df(filtered_fragments_df_pl).join(
                # Create PyRanges object from TSS annotation BED file after extending TSS position with flanking window.
                create_pyranges_from_polars_df(
                    (
                        tss_annotation.select(
                            # Only keep needed columns for faster PyRanges join.
                            pl.col(["Chromosome", "Start", "Strand"])
                        )
                        # Filter out TSS annotations without strand info (in case that would ever happen).
                        .filter(pl.col("Strand") != ".")
                        # Create [-flank_window, flank_window] around TSS and set them as "Start" and "End" column.
                        .with_columns(
                            [
                                (pl.col("Start") - flank_window).alias("Start"),
                                (pl.col("Start") + flank_window).alias("End"),
                            ]
                        )
                    )
                ),
                # Add "_tss_flank" suffix for joined output that comes from the TSS annotation BED file.
                suffix="_tss_flank",
                apply_strand_suffix=False,
            )
        ).df
    )

    if overlap_with_tss_df_pl.shape == (0, 0):
        raise ValueError(
            "No overlap found between fragments and TSS annotation. Make sure both use the same chromosome name source."
        )

    # Get cut sites (start/end fragments) relative to the TSS position minus flank_window value, so:
    #   - 0: cut site `flank_window` bp upstream of the TSS.
    #   - `flank_window`: cut site at the TSS.
    #   - `flank_window` * 2: cut site `flank_window` bp downstream of the TSS.
    cut_sites_tss_start_end = (
        overlap_with_tss_df_pl.clone()
        .with_columns(
            [
                pl.when(pl.col("Strand") == "-")
                .then(
                    # TSS is for a gene on the negative strand.
                    pl.struct(
                        [
                            # Calculate relative start and end of cut site to the (TSS - flank_window = End_tss_flank).
                            (pl.col("End_tss_flank") - pl.col("End")).alias(
                                "rel_start"
                            ),
                            (pl.col("End_tss_flank") - pl.col("Start")).alias(
                                "rel_end"
                            ),
                        ]
                    ).alias("rel_start_end")
                )
                .otherwise(
                    # TSS is for a gene on the positive strand.
                    pl.struct(
                        [
                            # Calculate relative start and end of cut site to the (TSS - flank_window = Start_tss_flank).
                            (pl.col("Start") - pl.col("Start_tss_flank")).alias(
                                "rel_start"
                            ),
                            (pl.col("End") - pl.col("Start_tss_flank")).alias(
                                "rel_end"
                            ),
                        ]
                    ).alias("rel_start_end")
                )
            ]
        )
        # Unnest "rel_start_end" struct column (creates: "rel_start" and "rel_end").
        .unnest("rel_start_end")
        .select(
            [
                pl.col("rel_start"),
                pl.col("rel_end"),
                pl.col("Name").alias("CB"),
            ]
        )
    )

    # Get TSS matrix:
    #   - columns: cut site positions relative to (TSS - flank_window).
    #   - rows: CBs
    #   - values: number of times a cut site position was found for a certain CB.
    tss_matrix = (
        # Get all cut sites positions for which have a value between 0 (included) and (flank_window * 2) (included):
        #   - Some fragments will have both cut sites in this interval.
        #   - Some fragments have only one cut site in this interval (start or end).
        pl.concat(
            [
                pl.DataFrame(
                    [
                        pl.Series(
                            "Position",
                            pl.arange(
                                low=0, high=flank_window * 2 + 1, step=1, eager=True
                            ),
                        ).cast(pl.Int32),
                        pl.Series(
                            "CB",
                            ["no_CB"] * (flank_window * 2 + 1),
                            dtype=pl.Categorical,
                        ),
                    ]
                ),
                # Get all cut sites for the relative start that pass the filter.
                cut_sites_tss_start_end.filter(
                    (pl.col("rel_start") >= 0)
                    & (pl.col("rel_start") <= flank_window * 2)
                ).select(
                    [
                        # Get cut site position and associated CB.
                        pl.col("rel_start").alias("Position"),
                        pl.col("CB"),
                    ]
                ),
                # Get all cut sites for the relative end that pass the filter.
                cut_sites_tss_start_end.filter(
                    (pl.col("rel_end") >= 0) & (pl.col("rel_end") <= flank_window * 2)
                ).select(
                    [
                        # Get cut site position and associated CB.
                        pl.col("rel_end").alias("Position"),
                        pl.col("CB"),
                    ]
                ),
            ],
        )
        # Count number of cut sites with the same CB and same position per CB.
        .pivot(
            values="Position",
            index="CB",
            columns="Position",
            aggregate_fn="count",
        )
        .filter(pl.col("CB") != "no_CB")
        .with_columns(
            [
                # Fill in 0, for non-observed values in the pivot table.
                pl.col(pl.UInt32)
                .cast(pl.Int32)
                .fill_null(0),
            ]
        )
    )

    # Get TSS matrix:
    #   - columns: CBs
    #   - rows: cut site positions relative to (TSS - flank_window).
    #   - values: number of times a cut site position was found for a certain CB.
    tss_matrix = (
        tss_matrix.clone()
        # Remove "CB" column, so numeric values of the TSS matrix can be transposed.
        .drop("CB")
        # Transpose TSS matrix:
        #    - columns: CBs
        #    - rows: cut site positions relative to (TSS - flank_window).
        .transpose(
            include_header=True,
            header_name="Position",
            column_names=tss_matrix.get_column("CB"),
        ).with_column(
            # Convert "Position" column from pl.Utf8 to pl.Int32.
            pl.col("Position").cast(pl.Int32)
        )
    )

    # Smooth TSS matrix per CB by a rolling window.
    tss_smoothed_matrix_per_cb = tss_matrix.with_columns(
        [
            pl.col("Position"),
            pl.all()
            .exclude("Position")
            .rolling_mean(window_size=window_size, min_periods=0),
        ]
    )

    # Remove raw TSS matrix.
    del tss_matrix

    # Normalize smoothed TSS matrix.
    # Get normalized sample TSS enrichment per position from the per CB smoothed TSS matrix.
    tss_norm_matrix_sample = tss_smoothed_matrix_per_cb.select(
        [
            pl.col("Position"),
            # Get total number of cut sites per position over all CBs.
            pl.sum(pl.all().exclude("Position")).alias("smoothed_per_pos_sum"),
        ]
    ).select(
        [
            pl.col("Position"),
            # Normalize total number of cut sites per position over all CBs.
            (
                pl.col("smoothed_per_pos_sum")
                / (
                    # Calculate background value from start and end over minimum_signal_window length.
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
            ).alias("norm"),
        ]
    )

    # Get normalized TSS matrix per CB for each cut site position.
    tss_norm_matrix_per_cb = tss_smoothed_matrix_per_cb.with_columns(
        [
            (
                pl.col(CB)
                / (
                    # Calculate background value from start and end over minimum_signal_window length.
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
            pl.col("Position").is_between(
                start=flank_window - 1 - tss_window,
                end=flank_window - 1 + tss_window,
                include_bounds=[True, True],
            )
        )
        .drop("Position")
        .select(pl.all().mean())
        .transpose(include_header=True, header_name="CB")
        .rename({"column_0": "tss_enrichment"})
    )

    return tss_enrichment_per_cb, tss_norm_matrix_sample, tss_norm_matrix_per_cb
