from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl
from ncls import NCLS

if TYPE_CHECKING:
    import numpy as np

# Intersection/overlap code is based on:
#   https://github.com/biocore-ntnu/pyranges/blob/master/pyranges/methods/intersection.py


def _get_start_end_and_indexes_for_chrom(
    regions_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    chrom: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get start, end and index positions from per chromosome Polars dataframe.

    Parameters
    ----------
    regions_per_chrom_dfs_pl
        Dictionary of region dataframes partitioned by chromosome.
    chrom
        Chromosome name.

    Returns
    -------
    (starts, ends, indexes)
        Tuple of numpy arrays with starts, ends and index positions
        for the requested chromosome.

    """
    starts, ends, indexes = list(
        regions_per_chrom_dfs_pl[chrom]
        .with_row_count()
        .select(
            [
                pl.col("Start").cast(pl.Int64),
                pl.col("End").cast(pl.Int64),
                pl.col("row_nr").cast(pl.Int64),
            ]
        )
        .to_numpy()
        .T
    )

    return starts, ends, indexes


def _intersect_per_chrom(
    regions1_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    regions2_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    chrom: str,
    how: Literal["all", "containment", "first", "last"] | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get intersection between two region sets per chromosome.

    Get intersection between regions from first set and second set of regions for a
    chromosome and return index positions for those overlaps in the first and
    second set of regions.

    Parameters
    ----------
    regions1_per_chrom_dfs_pl
        Dictionary of region Polars dataframes partitioned by chromosome for first set
        of regions.
    regions2_per_chrom_dfs_pl
        Dictionary of region Polars dataframes partitioned by chromosome for second set
        of regions.
    chrom
        Chromosome name.
    how
        What intervals to report:
          - ``"all"`` (``None``): all overlaps with second set or regions.
          - ``"containment"``: only overlaps where region of first set is contained
            within region of second set.
          - ``"first"``: first overlap with second set of regions.
          - ``"last"``: last overlap with second set of regions.
          - ``"outer"``: all regions for first and all regions of second (outer join).
            If no overlap was found for a region, the other region set will contain
            ``None`` for that entry.
          - ``"left"``: all first set of regions and overlap with second set of regions
            (left join).
            If no overlap was found for a region in the first set, the second region
            set will contain None for that entry.
          - ``"right"``: all second set of regions and overlap with first set of regions
            (right join).
            If no overlap was found for a region in the second set, the first region
            set will contain ``None`` for that entry.

    Returns
    -------
    (regions1_indexes, regions2_indexes)
        Tuple of indexes for regions from Polars Dataframe 1 and indexes for regions
        from Polars Dataframe 2 that have an overlap.

    """
    starts2, ends2, indexes2 = _get_start_end_and_indexes_for_chrom(
        regions2_per_chrom_dfs_pl,
        chrom=chrom,
    )

    oncls = NCLS(starts2, ends2, indexes2)

    indexes2_length = len(indexes2)

    del starts2, ends2, indexes2

    starts1, ends1, indexes1 = _get_start_end_and_indexes_for_chrom(
        regions1_per_chrom_dfs_pl,
        chrom=chrom,
    )

    if not how or how is None or how == "all":
        regions1_indexes, regions2_indexes = oncls.all_overlaps_both(
            starts1, ends1, indexes1
        )
    elif how == "containment":
        regions1_indexes, regions2_indexes = oncls.all_containments_both(
            starts1, ends1, indexes1
        )
    elif how == "first":
        regions1_indexes, regions2_indexes = oncls.first_overlap_both(
            starts1, ends1, indexes1
        )
    elif how == "last":
        regions1_indexes, regions2_indexes = oncls.last_overlap_both(
            starts1, ends1, indexes1
        )
    elif how in {"outer", "left", "right"}:
        regions1_indexes, regions2_indexes = oncls.all_overlaps_both(
            starts1, ends1, indexes1
        )

        indexes1_length = len(indexes1)

        del starts1, ends1, indexes1, oncls

        regions1_indexes = pl.Series("idx", regions1_indexes, dtype=pl.get_index_type())
        regions2_indexes = pl.Series("idx", regions2_indexes, dtype=pl.get_index_type())

        regions1_all_indexes = pl.arange(
            0, indexes1_length, dtype=pl.get_index_type(), eager=True
        ).alias("idx")
        regions2_all_indexes = pl.arange(
            0, indexes2_length, dtype=pl.get_index_type(), eager=True
        ).alias("idx")

        regions1_missing_indexes = (
            regions1_all_indexes.to_frame()
            .join(
                regions1_indexes.to_frame(),
                on="idx",
                how="anti",
            )
            .to_series()
        )

        regions2_missing_indexes = (
            regions2_all_indexes.to_frame()
            .join(
                regions2_indexes.to_frame(),
                on="idx",
                how="anti",
            )
            .to_series()
        )

        regions1_none_indexes = pl.repeat(
            None, regions2_missing_indexes.len(), name="idx", eager=True
        ).cast(pl.get_index_type())
        regions2_none_indexes = pl.repeat(
            None, regions1_missing_indexes.len(), name="idx", eager=True
        ).cast(pl.get_index_type())

        if how == "outer":
            regions1_indexes = pl.concat(
                [
                    regions1_indexes,
                    regions1_missing_indexes,
                    regions1_none_indexes,
                ]
            )
            regions2_indexes = pl.concat(
                [
                    regions2_indexes,
                    regions2_none_indexes,
                    regions2_missing_indexes,
                ]
            )
        elif how == "left":
            regions1_indexes = pl.concat([regions1_indexes, regions1_missing_indexes])
            regions2_indexes = pl.concat([regions2_indexes, regions2_none_indexes])
        elif how == "right":
            regions1_indexes = pl.concat([regions1_indexes, regions1_none_indexes])
            regions2_indexes = pl.concat([regions2_indexes, regions2_missing_indexes])

        return regions1_indexes, regions2_indexes

    del starts1, ends1, indexes1, oncls

    regions1_indexes = pl.Series("idx", regions1_indexes, dtype=pl.get_index_type())
    regions2_indexes = pl.Series("idx", regions2_indexes, dtype=pl.get_index_type())

    return regions1_indexes, regions2_indexes


def _overlap_per_chrom(
    regions1_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    regions2_per_chrom_dfs_pl: dict[str, pl.DataFrame],
    chrom: str,
    how: Literal["all", "containment", "first"] | str | None = "first",
) -> np.ndarray:
    """
    Get overlap between two region sets per chromosome.

    Get overlap between regions from first set and second set of regions for a
    chromosome and return index positions for those overlaps in the first set
    of regions.

    Parameters
    ----------
    regions1_per_chrom_dfs_pl
        Dictionary of region Polars dataframes partitioned by chromosome for first set
        of regions.
    regions2_per_chrom_dfs_pl
        Dictionary of region Polars dataframes partitioned by chromosome for second set
        of regions.
    chrom
        Chromosome name.
    how
        What intervals to report:
          - ``"all"`` (``None``): all overlaps with second set or regions.
          - ``"containment"``: only overlaps where region of first set is contained
            within region of second set.
          - ``"first"``: first overlap with second set of regions.

    Returns
    -------
    regions1_indexes
        Indexes for regions from Polars Dataframe 1 that had an overlap.

    """
    starts2, ends2, indexes2 = _get_start_end_and_indexes_for_chrom(
        regions2_per_chrom_dfs_pl,
        chrom=chrom,
    )

    oncls = NCLS(starts2, ends2, indexes2)

    del starts2, ends2, indexes2

    starts1, ends1, indexes1 = _get_start_end_and_indexes_for_chrom(
        regions1_per_chrom_dfs_pl,
        chrom=chrom,
    )

    if not how or how is None or how == "all":
        regions1_indexes = oncls.all_overlaps_self(starts1, ends1, indexes1)
    elif how == "containment":
        regions1_indexes, _ = oncls.all_containments_both(starts1, ends1, indexes1)
    elif how == "first":
        regions1_indexes = oncls.has_overlaps(starts1, ends1, indexes1)

    del starts1, ends1, indexes1

    return regions1_indexes


def _filter_intersection_output_columns(
    df: pl.DataFrame | pl.LazyFrame,
    regions1_info: bool,
    regions2_info: bool,
    regions1_coord: bool,
    regions2_coord: bool,
    regions1_suffix: str,
    regions2_suffix: str,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Filter intersection output columns.

    Parameters
    ----------
    df
        Polars DataFrame or LazyFrame with intersection results.
    regions1_info
        Add non-coordinate columns from first set of regions to output of intersection.
    regions2_info
        Add non-coordinate columns from first set of regions to output of intersection.
    regions1_coord
        Add coordinates from first set of regions to output of intersection.
    regions2_coord
        Add coordinates from second set of regions to output of intersection.
    regions1_suffix
        Suffix added to coordinate columns of first set of regions.
    regions2_suffix
        Suffix added to coordinate and info columns of second set of regions.

    Returns
    -------
    Polars LazyFrame with intersection results with only the requested columns.

    """
    # Get coordinate column names for first set of regions.
    regions1_coord_columns = [
        f"Chromosome{regions1_suffix}",
        f"Start{regions1_suffix}",
        f"End{regions1_suffix}",
    ]

    # Get coordinate column names for second set of regions.
    regions2_coord_columns = [
        f"Chromosome{regions2_suffix}",
        f"Start{regions2_suffix}",
        f"End{regions2_suffix}",
    ]

    # Get info column names for first set of regions
    # (all columns except coordinate columns).
    regions1_suffix_length = len(regions1_suffix)
    regions1_info_columns = [
        # Remove region1 suffix from column names.
        pl.col(column_name).alias(column_name[:-regions1_suffix_length])
        for column_name in df.columns
        if (
            column_name.endswith(regions1_suffix)
            and column_name not in regions1_coord_columns
        )
    ]

    # Get info column names for second set of regions
    # (all columns except coordinate columns).
    regions2_suffix_length = len(regions2_suffix)
    regions2_info_columns = [
        # Remove region2 suffix from column names if no region1 info will be displayed.
        pl.col(column_name)
        if regions1_info
        else pl.col(column_name).alias(column_name[:-regions2_suffix_length])
        for column_name in df.columns
        if (
            column_name.endswith(regions2_suffix)
            and column_name not in regions2_coord_columns
        )
    ]

    select_columns = [pl.col(["Chromosome", "Start", "End"])]
    if regions1_coord:
        select_columns.append(pl.col(regions1_coord_columns))
    if regions2_coord:
        select_columns.append(pl.col(regions2_coord_columns))

    if regions1_info:
        select_columns.extend(regions1_info_columns)
    if regions2_info:
        select_columns.extend(regions2_info_columns)

    return df.select(select_columns)


def intersection(
    regions1_df_pl: pl.DataFrame,
    regions2_df_pl: pl.DataFrame,
    how: Literal["all", "containment", "first", "last"] | str | None = None,
    regions1_info: bool = True,
    regions2_info: bool = False,
    regions1_coord: bool = False,
    regions2_coord: bool = False,
    regions1_suffix: str = "@1",
    regions2_suffix: str = "@2",
) -> pl.DataFrame:
    """
    Get overlapping subintervals between first set and second set of regions.

    Parameters
    ----------
    regions1_df_pl
        Polars DataFrame containing BED entries for first set of regions.
    regions2_df_pl
        Polars DataFrame containing BED entries for second set of regions.
    how
        What intervals to report:
          - ``"all"`` (``None``): all overlaps with second set or regions.
          - ``"containment"``: only overlaps where region of first set is contained
            within region of second set.
          - ``"first"``: first overlap with second set of regions.
          - ``"last"``: last overlap with second set of regions.
          - ``"outer"``: all regions for first and all regions of second (outer join).
            If no overlap was found for a region, the other region set will contain
            ``None`` for that entry.
          - ``"left"``: all first set of regions and overlap with second set of regions
            (left join).
            If no overlap was found for a region in the first set, the second region
            set will contain None for that entry.
          - ``"right"``: all second set of regions and overlap with first set of regions
            (right join).
            If no overlap was found for a region in the second set, the first region
            set will contain ``None`` for that entry.
    regions1_info
        Add non-coordinate columns from first set of regions to output of intersection.
    regions2_info
        Add non-coordinate columns from first set of regions to output of intersection.
    regions1_coord
        Add coordinates from first set of regions to output of intersection.
    regions2_coord
        Add coordinates from second set of regions to output of intersection.
    regions1_suffix
        Suffix added to coordinate columns of first set of regions.
    regions2_suffix
        Suffix added to coordinate and info columns of second set of regions.

    strandedness
        Note: Not implemented yet.
        {``None``, ``"same"``, ``"opposite"``, ``False``}, default ``None``, i.e. auto
        Whether to compare PyRanges on the same strand, the opposite or ignore strand
        information. The default, ``None``, means use ``"same"`` if both PyRanges are
        stranded, otherwise ignore the strand information.

    Returns
    -------
    intersection_df_pl
        Polars Dataframe containing BED entries with the intersection.

    Examples
    --------
    >>> regions1_df_pl = pl.from_dict(
    ...     {
    ...         "Chromosome": ["chr1"] * 3,
    ...         "Start": [1, 4, 10],
    ...         "End": [3, 9, 11],
    ...         "ID": ["a", "b", "c"],
    ...     }
    ... )
    >>> regions1_df_pl
    shape: (3, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 1     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 10    ┆ 11  ┆ c   │
    └────────────┴───────┴─────┴─────┘

    >>> regions2_df_pl = pl.from_dict(
    ...     {
    ...         "Chromosome": ["chr1"] * 3,
    ...         "Start": [2, 2, 9],
    ...         "End": [3, 9, 10],
    ...         "Name": ["reg1", "reg2", "reg3"]
    ...     }
    ... )
    >>> regions2_df_pl
    shape: (3, 4)
    ┌────────────┬───────┬─────┬──────┐
    │ Chromosome ┆ Start ┆ End ┆ Name │
    │ ---        ┆ ---   ┆ --- ┆ ---  │
    │ str        ┆ i64   ┆ i64 ┆ str  │
    ╞════════════╪═══════╪═════╪══════╡
    │ chr1       ┆ 2     ┆ 3   ┆ reg1 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ chr1       ┆ 2     ┆ 9   ┆ reg2 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ chr1       ┆ 9     ┆ 10  ┆ reg3 │
    └────────────┴───────┴─────┴──────┘

    >>> intersection(regions1_df_pl, regions2_df_pl)
    shape: (3, 3)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 2     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 2     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    └────────────┴───────┴─────┴─────┘

    >>> intersection(regions1_df_pl, regions2_df_pl, how="first")
    shape: (2, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 2     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    └────────────┴───────┴─────┴─────┘

    >>> intersection(
    ...     regions1_df_pl,
    ...     regions2_df_pl,
    ...     how="containment",
    ...     regions1_info=False,
    ...     regions2_info=True,
    ... )
    shape: (1, 4)
    ┌────────────┬───────┬─────┬──────┐
    │ Chromosome ┆ Start ┆ End ┆ Name │
    │ ---        ┆ ---   ┆ --- ┆ ---  │
    │ str        ┆ i64   ┆ i64 ┆ str  │
    ╞════════════╪═══════╪═════╪══════╡
    │ chr1       ┆ 4     ┆ 9   ┆ reg2 │
    └────────────┴───────┴─────┴──────┘

    >>> intersection(
    ...     regions1_df_pl,
    ...     regions2_df_pl,
    ...     regions1_coord=True,
    ...     regions2_coord=True,
    ... )
    shape: (3, 10)
    ┌────────────┬───────┬─────┬──────────────┬─────────┬───────┬──────────────┬─────────┬───────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ Chromosome@1 ┆ Start@1 ┆ End@1 ┆ Chromosome@2 ┆ Start@2 ┆ End@2 ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ ---          ┆ ---     ┆ ---   ┆ ---          ┆ ---     ┆ ---   ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str          ┆ i64     ┆ i64   ┆ str          ┆ i64     ┆ i64   ┆ str │
    ╞════════════╪═══════╪═════╪══════════════╪═════════╪═══════╪══════════════╪═════════╪═══════╪═════╡
    │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 1       ┆ 3     ┆ chr1         ┆ 2       ┆ 9     ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 1       ┆ 3     ┆ chr1         ┆ 2       ┆ 3     ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ chr1         ┆ 4       ┆ 9     ┆ chr1         ┆ 2       ┆ 9     ┆ b   │
    └────────────┴───────┴─────┴──────────────┴─────────┴───────┴──────────────┴─────────┴───────┴─────┘

    >>> intersection(
    ...     regions1_df_pl,
    ...     regions2_df_pl,
    ...     regions1_info=False,
    ...     regions_info=True,
    ...     regions2_coord=True,
    ... )
    shape: (3, 7)
    ┌────────────┬───────┬─────┬──────────────┬─────────┬───────┬──────┐
    │ Chromosome ┆ Start ┆ End ┆ Chromosome@2 ┆ Start@2 ┆ End@2 ┆ Name │
    │ ---        ┆ ---   ┆ --- ┆ ---          ┆ ---     ┆ ---   ┆ ---  │
    │ str        ┆ i64   ┆ i64 ┆ str          ┆ i64     ┆ i64   ┆ str  │
    ╞════════════╪═══════╪═════╪══════════════╪═════════╪═══════╪══════╡
    │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 2       ┆ 9     ┆ reg2 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 2       ┆ 3     ┆ reg1 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ chr1         ┆ 2       ┆ 9     ┆ reg2 │
    └────────────┴───────┴─────┴──────────────┴─────────┴───────┴──────┘

    """
    # TODO: chrom, stranded partitioning
    regions1_per_chrom_dfs_pl = regions1_df_pl.partition_by(
        "Chromosome", as_dict=True, maintain_order=True
    )

    regions2_per_chrom_dfs_pl = regions2_df_pl.partition_by(
        "Chromosome", as_dict=True, maintain_order=True
    )

    intersection_chrom_dfs_pl = {}

    for chrom in list(regions1_per_chrom_dfs_pl.keys()):
        if chrom in list(regions2_per_chrom_dfs_pl.keys()):
            # Find intersection between regions form first and second per chromosome
            # dataframe and return index positions in both dataframes for those
            # intersections.
            regions1_indexes, regions2_indexes = _intersect_per_chrom(
                regions1_per_chrom_dfs_pl=regions1_per_chrom_dfs_pl,
                regions2_per_chrom_dfs_pl=regions2_per_chrom_dfs_pl,
                chrom=chrom,
                how=how,
            )

            # Skip empty intersections.
            if regions1_indexes.shape[0] == 0:
                continue

            # Get all regions from first and second per chromosome dataframe for the
            # index positions calculated above.
            intersection_chrom_df_pl = (
                regions1_per_chrom_dfs_pl.pop(chrom)[regions1_indexes]
                .select(pl.all().suffix(regions1_suffix))
                .hstack(
                    (regions2_per_chrom_dfs_pl.pop(chrom)[regions2_indexes]).select(
                        pl.all().suffix(regions2_suffix)
                    )
                )
            )

            # Calculate intersection start and end coordinates and return the columns
            # of interest.
            intersection_chrom_ldf_pl = (
                intersection_chrom_df_pl.lazy()
                .with_columns(
                    [
                        # Chromosome name for intersection.
                        pl.coalesce(
                            pl.col(f"Chromosome{regions1_suffix}"),
                            pl.col(f"Chromosome{regions2_suffix}"),
                        ).alias("Chromosome"),
                        # Calculate start coordinate for intersection.
                        pl.when(
                            pl.col(f"Start{regions1_suffix}")
                            > pl.col(f"Start{regions2_suffix}")
                        )
                        .then(pl.col(f"Start{regions1_suffix}"))
                        .otherwise(
                            pl.coalesce(
                                pl.col(f"Start{regions2_suffix}"),
                                pl.col(f"Start{regions1_suffix}"),
                            )
                        )
                        .alias("Start"),
                        # Calculate end coordinate for intersection.
                        pl.when(
                            pl.col(f"End{regions1_suffix}")
                            < pl.col(f"End{regions2_suffix}")
                        )
                        .then(pl.col(f"End{regions1_suffix}"))
                        .otherwise(
                            pl.coalesce(
                                pl.col(f"End{regions2_suffix}"),
                                pl.col(f"End{regions1_suffix}"),
                            )
                        )
                        .alias("End"),
                    ]
                )
                .pipe(
                    function=_filter_intersection_output_columns,
                    regions1_info=regions1_info,
                    regions2_info=regions2_info,
                    regions1_coord=regions1_coord,
                    regions2_coord=regions2_coord,
                    regions1_suffix=regions1_suffix,
                    regions2_suffix=regions2_suffix,
                )
            )

            intersection_chrom_dfs_pl[chrom] = intersection_chrom_ldf_pl.collect()

    # Combine per chromosome dataframes to a full one.
    intersection_df_pl = pl.concat(
        list(intersection_chrom_dfs_pl.values()), rechunk=False
    )

    return intersection_df_pl


def overlap(
    regions1_df_pl: pl.DataFrame,
    regions2_df_pl: pl.DataFrame,
    how: Literal["all", "containment", "first"] | str | None = "first",
    invert: bool = False,
) -> pl.DataFrame:
    """
    Get overlap between two region sets.

    Get overlap between first set and second set of regions and return interval of
    first set of regions.

    Parameters
    ----------
    regions1_df_pl
        Polars DataFrame containing BED entries for first set of regions.
    regions2_df_pl
        Polars DataFrame containing BED entries for second set of regions.
    how
        What overlaps to report:
          - ``"all"`` (``None``): all overlaps with second set or regions.
          - ``"containment"``: only overlaps where region of first set is contained
            within region of second set.
          - ``"first"``: first overlap with second set of regions.
    invert
        Whether to return the intervals without overlaps.

    strandedness
        Note: Not implemented yet.
        {``None``, ``"same"``, ``"opposite"``, ``False``}, default ``None``, i.e. auto
        Whether to compare PyRanges on the same strand, the opposite or ignore strand
        information. The default, ``None``, means use ``"same"`` if both PyRanges are
        stranded, otherwise ignore the strand information.

    Returns
    -------
    overlap_df_pl
        Polars Dataframe containing BED entries with the overlap.

    Examples
    --------
    >>> regions1_df_pl = pl.from_dict(
    ...     {
    ...         "Chromosome": ["chr1"] * 3,
    ...         "Start": [1, 4, 10],
    ...         "End": [3, 9, 11],
    ...         "ID": ["a", "b", "c"],
    ...     }
    ... )
    >>> regions1_df_pl
    shape: (3, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 1     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 10    ┆ 11  ┆ c   │
    └────────────┴───────┴─────┴─────┘

    >>> regions2_df_pl = pl.from_dict(
    ...     {
    ...         "Chromosome": ["chr1"] * 3,
    ...         "Start": [2, 2, 9],
    ...         "End": [3, 9, 10],
    ...         "Name": ["reg1", "reg2", "reg3"]
    ...     }
    ... )
    >>> regions2_df_pl
    shape: (3, 4)
    ┌────────────┬───────┬─────┬──────┐
    │ Chromosome ┆ Start ┆ End ┆ Name │
    │ ---        ┆ ---   ┆ --- ┆ ---  │
    │ str        ┆ i64   ┆ i64 ┆ str  │
    ╞════════════╪═══════╪═════╪══════╡
    │ chr1       ┆ 2     ┆ 3   ┆ reg1 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ chr1       ┆ 2     ┆ 9   ┆ reg2 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
    │ chr1       ┆ 9     ┆ 10  ┆ reg3 │
    └────────────┴───────┴─────┴──────┘

    >>> overlap(regions1_df_pl, regions2_df_pl, how="first")
    shape: (2, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 1     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    └────────────┴───────┴─────┴─────┘

    >>> overlap(regions1_df_pl, regions2_df_pl, how="all")
    shape: (3, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 1     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 1     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    └────────────┴───────┴─────┴─────┘

    >>> overlap(regions1_df_pl, regions2_df_pl, how="containment")
    shape: (1, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 4     ┆ 9   ┆ b   │
    └────────────┴───────┴─────┴─────┘

    >>> overlap(regions1_df_pl, regions2_df_pl, how="containment", invert=True)
    shape: (2, 4)
    ┌────────────┬───────┬─────┬─────┐
    │ Chromosome ┆ Start ┆ End ┆ ID  │
    │ ---        ┆ ---   ┆ --- ┆ --- │
    │ str        ┆ i64   ┆ i64 ┆ str │
    ╞════════════╪═══════╪═════╪═════╡
    │ chr1       ┆ 1     ┆ 3   ┆ a   │
    ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
    │ chr1       ┆ 10    ┆ 11  ┆ c   │
    └────────────┴───────┴─────┴─────┘

    """
    # TODO: chrom, stranded partitioning
    regions1_per_chrom_dfs_pl = regions1_df_pl.partition_by(
        "Chromosome", as_dict=True, maintain_order=True
    )

    regions2_per_chrom_dfs_pl = regions2_df_pl.partition_by(
        "Chromosome", as_dict=True, maintain_order=True
    )

    overlap_chrom_dfs_pl = {}

    for chrom in list(regions1_per_chrom_dfs_pl.keys()):
        if chrom in set(regions2_per_chrom_dfs_pl.keys()):
            # Find overlap between regions from first and second per chromosome
            # dataframe and return index positions in first dataframe for those
            # overlaps.
            regions1_indexes = _overlap_per_chrom(
                regions1_per_chrom_dfs_pl=regions1_per_chrom_dfs_pl,
                regions2_per_chrom_dfs_pl=regions2_per_chrom_dfs_pl,
                chrom=chrom,
                how=how,
            )

            # Skip empty intersections.
            if regions1_indexes.shape[0] == 0:
                continue

            overlap_chrom_dfs_pl[chrom] = (
                # Get inverse selection of regions from first dataframe for the
                # overlap.
                regions1_per_chrom_dfs_pl.pop(chrom)
                .with_row_count()
                .filter(
                    ~pl.col("row_nr").is_in(
                        pl.Series("regions1_indexes", regions1_indexes)
                    )
                )
                .drop("row_nr")
                if invert
                # Get selection of regions from first dataframe for the overlap.
                else regions1_per_chrom_dfs_pl.pop(chrom)[regions1_indexes]
            )

    # Combine per chromosome dataframes to a full one.
    overlap_df_pl = pl.concat(list(overlap_chrom_dfs_pl.values()), rechunk=False)

    return overlap_df_pl
