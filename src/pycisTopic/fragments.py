from __future__ import annotations

import gzip
import os
from typing import TYPE_CHECKING, Literal, Sequence, overload

import polars as pl
import scipy as sp

from pycisTopic.categoricals import PycisTopicCategoricals
from pycisTopic.genomic_ranges import intersection as gr_intersection
from pycisTopic.genomic_ranges import overlap as gr_overlap

if TYPE_CHECKING:
    from pathlib import Path


def normalise_filepath(path: str | Path, check_not_directory: bool = True) -> str:
    """Create a string path, expanding the home directory if present."""
    path = os.path.expanduser(path)
    if check_not_directory and os.path.exists(path) and os.path.isdir(path):
        raise IsADirectoryError(f"Expected a file path; {path!r} is a directory")
    return path


def cbs_to_cbs_series_pl(
    cbs: Sequence[str] | pl.Series,
) -> pl.Series:
    """
    Convert cell barcodes to a ``PycisTopicCategoricals.CB`` Series.

    Parameters
    ----------
    cbs
        List or Polars Series with cell barcodes.

    Returns
    -------
    `PycisTopicCategoricals.CB`` Series with cell barcodes.

    Examples
    --------
    Convert list of cell barcodes to a ``PycisTopicCategoricals.CB`` Series.

    >>> cbs_list = ["GGACATAAGGGCCACT-1", "ACCTTCATCTTTGAGA-1"]
    >>> cbs_series_pl = cbs_to_cbs_series_pl(cbs_list)

    Convert Polars Utf8/Categorical Series with cell barcodes to a
    ``PycisTopicCategoricals.CB`` Series.

    >>> cbs_series_pl_utf8 = pl.Series(
    ...     "CB",
    ...     ["GGACATAAGGGCCACT-1", "ACCTTCATCTTTGAGA-1"],
    ...     dtype=pl.Utf8,
    ... )
    >>> cbs_series_pl = cbs_to_cbs_series_pl(cbs_series_pl_utf8)
    >>> cbs_series_pl_cat = pl.Series(
    ...     "CB",
    ...     ["GGACATAAGGGCCACT-1", "ACCTTCATCTTTGAGA-1"],
    ...     dtype=pl.Categorical,
    ... )
    >>> cbs_series_pl = cbs_to_cbs_series_pl(cbs_series_pl_cat)

    """
    if isinstance(cbs, Sequence):
        if isinstance(cbs[0], str):
            cbs_series_pl = pl.Series(
                "CB", cbs, dtype=pl.Categorical(PycisTopicCategoricals.CB)
            )
        else:
            raise ValueError(
                "Unsupported type for cell barcodes. First element of cell barcodes is not a string."
            )
    elif isinstance(cbs, pl.Series):
        if isinstance(cbs.dtype, pl.Utf8):
            cbs_series_pl = cbs.cast(pl.Categorical(PycisTopicCategoricals.CB)).rename(
                "CB"
            )
        elif isinstance(cbs.dtype, pl.Categorical):
            if cbs.dtype.categories == PycisTopicCategoricals.CB:
                cbs_series_pl = cbs.rename("CB")
            else:
                cbs_series_pl = cbs.cast(
                    pl.Categorical(PycisTopicCategoricals.CB)
                ).rename("CB")
    else:
        raise ValueError("Unsupported type for cell barcodes.")

    return cbs_series_pl


def region_ids_to_bed_df_pl(region_ids: Sequence[str] | pl.Series) -> pl.DataFrame:
    """
    Convert region IDs to a BED Polars DataFrame.

    Parameters
    ----------
    region_ids
        List of region IDs in the format ``Chromosome:Start-End``.

    Returns
    -------
    Polars DataFrame with BED entries (``Chromosome``, ``Start``, ``End``
    and ``RegionID``).

    See Also
    --------
    pycisTopic.fragments.add_region_ids_to_bed_df_pl

    Examples
    --------
    Convert list of region IDs to a BED Polars DataFrame.

    >>> region_ids = ["chr1:1000-2000", "chr2:1500-2500"]
    >>> bed_df_pl = region_ids_to_bed_df_pl(region_ids)

    >>> region_ids = pl.Series(["chr1:1000-2000", "chr2:1500-2500"])
    >>> bed_df_pl = region_ids_to_bed_df_pl(region_ids)

    """
    bed_df_pl = (
        pl.DataFrame(
            region_ids,
            schema={"RegionID": pl.Categorical(PycisTopicCategoricals.REGION_ID)},
        )
        .lazy()
        .select(
            pl.col("RegionID")
            .cast(pl.Utf8)
            .str.extract_groups(
                r"""^(?<Chromosome>[^:]+):(?<Start>[0-9]+)-(?<End>[0-9]+)$"""
            )
            .alias("ChromStartEnd"),
            pl.col("RegionID"),
        )
        .unnest("ChromStartEnd")
        .with_columns(
            pl.col("Chromosome").cast(
                pl.Categorical(PycisTopicCategoricals.CHROMOSOME)
            ),
            pl.col("Start").cast(pl.Int32),
            pl.col("End").cast(pl.Int32),
        )
        .collect()
    )

    return bed_df_pl


@overload
def add_region_ids_to_bed_df_pl(
    bed_df_pl: pl.DataFrame,
) -> pl.DataFrame: ...


@overload
def add_region_ids_to_bed_df_pl(
    bed_df_pl: pl.LazyFrame,
) -> pl.LazyFrame: ...


def add_region_ids_to_bed_df_pl(
    bed_df_pl: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Add ``RegionID`` column to BED Polars DataFrame.

    Parameters
    ----------
    bed_df_pl
        Polars DataFrame with BED entries with at least the following columns:
        ``Chromosome``, ``Start`` and ``End``.

    Returns
    -------
    Polars DataFrame with original BED entries and ``RegionID`` column.

    See Also
    --------
    pycisTopic.fragments.read_bed_to_polars_df
    pycisTopic.fragments.region_ids_to_bed_df_pl

    Examples
    --------
    Read BED file to Polars DataFrame and add ``RegionID`` column.

    >>> bed_df_pl = read_bed_to_polars_df("test.bed")
    >>> bed_df_pl = add_region_ids_to_bed_df_pl(bed_df_pl)

    Read BED file to Polars DataFrame and add ``RegionID`` column using the ``pipe``
    method.

    >>> bed_df_pl = read_bed_to_polars_df("test.bed").pipe(add_region_ids_to_bed_df_pl)

    """
    bed_df_pl = bed_df_pl.with_columns(
        (
            pl.col("Chromosome").cast(pl.Utf8)
            + ":"
            + pl.col("Start").cast(pl.Utf8)
            + "-"
            + pl.col("End").cast(pl.Utf8)
        )
        .cast(pl.Categorical(PycisTopicCategoricals.REGION_ID))
        .alias("RegionID")
    )

    return bed_df_pl


def read_bed_to_polars_df(
    bed_filename: str,
    bed_parser_engine: str
    | Literal["polars_lazy", "polars", "pyarrow"] = "polars_lazy",
    min_column_count: int = 3,
) -> pl.DataFrame:
    """
    Read (gzipped) BED file to a Polars DataFrame.

    Parameters
    ----------
    bed_filename
        BED filename.
    bed_parser_engine
        BED parsing engine to use to read the (gzipped) BED file.

        Options:
          - ``polars_lazy`` (fastest, low memory usage): Use Polars lazy API (``pl.scan_csv``).
          - ``polars`` (slightly slower, highest memory usage): Use Polars eager API (``pl.read_csv``).
          - ``pyarrow`` (slowest, high memory usage): Use pyarrow CSV reader (``pa.csv.read_csv``).
    min_column_count
        Minimum number of required columns needed in BED file.

    Returns
    -------
    Polars DataFrame with BED entries.

    See Also
    --------
    pycisTopic.fragments.add_region_ids_to_bed_df_pl
    pycisTopic.fragments.read_fragments_to_polars_df

    Examples
    --------
    Read BED file to Polars DataFrame with Polars lazy API engine.

    >>> bed_df_pl = read_bed_to_polars_df("test.bed", bed_parser_engine="polars_lazy")

    Read BED file to Polars DataFrame with Polars lazy API engine and require that the
    BED file has at least 4 columns.

    >>> bed_with_at_least_4_columns_df_pl = read_bed_to_polars_df(
    ...     "test.bed",
    ...     bed_parser_engine="polars_lazy",
    ...     min_column_count=4,
    ... )

    """
    bed_column_names = (
        "Chromosome",
        "Start",
        "End",
        "Name",
        "Score",
        "Strand",
        "ThickStart",
        "ThickEnd",
        "ItemRGB",
        "BlockCount",
        "BlockSizes",
        "BlockStarts",
    )

    bed_filename = normalise_filepath(bed_filename)

    # Set the correct open function, depending upon if the fragments BED file is gzip
    # compressed or not.
    open_fn = gzip.open if bed_filename.endswith(".gz") else open

    skip_rows = 0
    column_count = 0
    with open_fn(bed_filename, "rt") as bed_fh:
        for line in bed_fh:
            # Remove newlines and spaces.
            line = line.strip()

            if not line or line.startswith("#"):
                # Count number of empty lines and lines which start with a comment
                # before the actual data.
                skip_rows += 1
            else:
                # Get number of columns from the first real BED entry.
                column_count = len(line.split("\t"))

                # Stop reading the BED file.
                break

    if column_count < min_column_count:
        raise ValueError(
            f"BED file needs to have at least {min_column_count} columns. "
            f'"{bed_filename}" contains only {column_count} columns.'
        )

    if bed_parser_engine == "polars_lazy":
        # Read BED file with Polars.
        bed_df_pl = pl.scan_csv(
            bed_filename,
            has_header=False,
            separator="\t",
            comment_prefix="#",
            with_column_names=lambda cols: bed_column_names[:column_count],
            schema_overrides={
                bed_column: dtype
                for bed_column, dtype in {
                    "Chromosome": pl.Categorical(PycisTopicCategoricals.CHROMOSOME),
                    "Start": pl.Int32,
                    "End": pl.Int32,
                    "Name": pl.Categorical(PycisTopicCategoricals.NAME),
                    "Strand": pl.Categorical(PycisTopicCategoricals.STRAND),
                }.items()
                if bed_column in bed_column_names[:column_count]
            },
        ).collect()
    elif bed_parser_engine == "polars":
        # Read BED file with Polars.
        bed_df_pl = pl.read_csv(
            bed_filename,
            has_header=False,
            separator="\t",
            comment_prefix="#",
            use_pyarrow=False,
            new_columns=bed_column_names[:column_count],
            schema_overrides={
                bed_column: dtype
                for bed_column, dtype in {
                    "Chromosome": pl.Categorical(PycisTopicCategoricals.CHROMOSOME),
                    "Start": pl.Int32,
                    "End": pl.Int32,
                    "Name": pl.Categorical(PycisTopicCategoricals.NAME),
                    "Strand": pl.Categorical(PycisTopicCategoricals.STRAND),
                }.items()
                if bed_column in bed_column_names[:column_count]
            },
        )
    elif bed_parser_engine == "pyarrow":
        import pyarrow as pa  # type: ignore[import]
        import pyarrow.csv  # type: ignore[import]

        # Read BED file with pyarrow.
        bed_df_pl = pl.from_arrow(
            pa.csv.read_csv(
                bed_filename,
                read_options=pa.csv.ReadOptions(
                    use_threads=True,
                    skip_rows=skip_rows,
                    column_names=bed_column_names[:column_count],
                ),
                parse_options=pa.csv.ParseOptions(
                    delimiter="\t",
                    quote_char=False,
                    escape_char=False,
                    newlines_in_values=False,
                ),
                convert_options=pa.csv.ConvertOptions(
                    column_types={
                        "Chromosome": pa.dictionary(pa.int32(), pa.large_string()),
                        "Start": pa.int32(),
                        "End": pa.int32(),
                        "Name": pa.dictionary(pa.int32(), pa.large_string()),
                        "Strand": pa.dictionary(pa.int32(), pa.large_string()),
                    },
                ),
            ),
            schema_overrides={
                bed_column: dtype
                for bed_column, dtype in {
                    "Chromosome": pl.Categorical(PycisTopicCategoricals.CHROMOSOME),
                    "Start": pl.Int32,
                    "End": pl.Int32,
                    "Name": pl.Categorical(PycisTopicCategoricals.NAME),
                    "Strand": pl.Categorical(PycisTopicCategoricals.STRAND),
                }.items()
                if bed_column in bed_column_names[:column_count]
            },
            rechunk=False,
        )
    else:
        raise ValueError(
            f'Unsupported bed_parser_engine value "{bed_parser_engine}" (allowed: ["polars_lazy", "polars", "pyarrow"]).'
        )

    return bed_df_pl


def read_fragments_to_polars_df(
    fragments_bed_filename: str,
    bed_parser_engine: str
    | Literal["polars_lazy", "polars", "pyarrow"] = "polars_lazy",
    sample_id: str | None = None,
    cb_end_to_remove: str | None = "-1",
    cb_sample_separator: str | None = "___",
) -> pl.DataFrame:
    """
    Read fragments BED file to a Polars DataFrame.

    If fragments don't have a ``CB_count`` column, a ``CB_count`` column is created by
    counting the number of fragments with the same chromosome, start, end and CB.

    Parameters
    ----------
    fragments_bed_filename
        Fragments BED filename.
    bed_parser_engine
        BED parsing engine to use to read the (gzipped) fragments file.

        Options:
          - ``polars_lazy`` (fastest, low memory usage): Use Polars lazy API (``pl.scan_csv``).
          - ``polars`` (slightly slower, highest memory usage): Use Polars eager API (``pl.read_csv``).
          - ``pyarrow`` (slowest, high memory usage): Use pyarrow CSV reader (``pa.csv.read_csv``).
    sample_id
        Optional sample ID to append after cell barcode after removing `cb_end_to_remove`
        and appending `cb_sample_separator`.
    cb_end_to_remove
        Remove this string from the end of the cell barcode if `sample_id` is specified.
    cb_sample_separator
        Add this string to the cell barcode if `sample_id` is specified, after removing
        `cb_end_to_remove` and before appending `sample_id`.

    Returns
    -------
    Polars DataFrame with fragments.

    See Also
    --------
    pycisTopic.fragments.read_bed_to_polars_df

    Examples
    --------
    Read gzipped fragments BED file to a Polars DataFrame.

    >>> fragments_df_pl = read_fragments_to_polars_df(
    ...     fragments_bed_filename="fragments.tsv.gz",
    ... )

    Read uncompressed fragments BED file to a Polars DataFrame.

    >>> fragments_df_pl = read_fragments_to_polars_df(
    ...     fragments_bed_filename="fragments.tsv",
    ... )

    Read gzipped fragments BED file with Polars lazy API engine to a Polars DataFrame
    and add sample ID to cell barcode names after removing `cb_end_to_remove` string
    from cell barcode and appending `cb_sample_separator` to the cell barcode.

    >>> fragments_df_pl = read_fragments_to_polars_df(
    ...     fragments_bed_filename="fragments.tsv.gz",
    ...     bed_parser_engine="polars_lazy",
    ...     sample_id="sample1",
    ...     cb_end_to_remove="-1",
    ...     cb_sample_separator="___",
    ... )

    """
    fragments_df_pl = (
        read_bed_to_polars_df(
            bed_filename=fragments_bed_filename,
            bed_parser_engine=bed_parser_engine,
            min_column_count=4,
        )
        .lazy()
        .rename({"Name": "CB", "Score": "CB_count"})
    )

    # If no "CB_count" is provided or "CB_count" column is ".", generate a
    # CB_count column with the number of fragments which have the same
    # chromosome, start, end and CB.
    if fragments_df_pl.collect_schema().get("CB_count") in (None, pl.Utf8):
        fragments_df_pl = fragments_df_pl.group_by(
            ["Chromosome", "Start", "End", "CB"]
        ).agg(pl.len().cast(pl.Int32()).alias("CB_count"))
    else:
        fragments_df_pl = fragments_df_pl.with_columns(
            pl.col("CB_count").cast(pl.Int32())
        )

    # Modify cell barcode if sample ID is specified or an empty string.
    if sample_id or sample_id == "":
        separator_and_sample_id = (
            f"{cb_sample_separator + sample_id}" if cb_sample_separator else sample_id
        )

        if not cb_end_to_remove:
            # Append separator and sample ID to cell barcode.
            fragments_df_pl = fragments_df_pl.with_columns(
                (pl.col("CB").cast(pl.Utf8) + pl.lit(separator_and_sample_id)).cast(
                    pl.Categorical(PycisTopicCategoricals.CB)
                )
            )
        else:
            fraction_of_CBs_with_end_to_remove = (
                fragments_df_pl.select(
                    pl.col("CB").unique(),
                )
                .select(
                    pl.col("CB")
                    .cat.ends_with(cb_end_to_remove)
                    .sum()
                    .alias("CBs_with_end_to_remove"),
                    pl.col("CB").count().alias("CB_count"),
                )
                .select(
                    (pl.col("CBs_with_end_to_remove") / pl.col("CB_count")).alias(
                        "fraction_of_CBs_with_end_to_remove"
                    ),
                )
                .collect()
                .to_series()[0]
            )

            if fraction_of_CBs_with_end_to_remove != 1.0:
                print(
                    f'Warning: Not all cell barcodes in fragments file "{fragments_bed_filename}" end with '
                    f'"{cb_end_to_remove}". Percentage of cell barcodes with '
                    f'"{cb_end_to_remove}" at the end: '
                    f"{(fraction_of_CBs_with_end_to_remove * 100):.2f}%."
                )

            # Remove `cb_end_to_remove` from the end of the cell barcode before adding
            # separator and sample ID to cell barcode.
            fragments_df_pl = fragments_df_pl.with_columns(
                (
                    pl.col("CB").cast(pl.Utf8).str.strip_suffix(cb_end_to_remove)
                    + pl.lit(separator_and_sample_id)
                )
                .cast(pl.Categorical(PycisTopicCategoricals.CB))
                .alias("CB")
            )

    fragments_df_pl = fragments_df_pl.collect()

    return fragments_df_pl


def read_barcodes_file_to_polars_series(
    barcodes_tsv_filename: str,
    sample_id: str | None = None,
    cb_end_to_remove: str | None = "-1",
    cb_sample_separator: str | None = "___",
) -> pl.Series:
    """
    Read barcode TSV file to a Polars Series.

    Parameters
    ----------
    barcodes_tsv_filename
        TSV file with CBs.
    sample_id
        Optional sample ID to append after cell barcode after removing `cb_end_to_remove`
        and appending `cb_sample_separator`.
    cb_end_to_remove
        Remove this string from the end of the cell barcode if `sample_id` is specified.
    cb_sample_separator
        Add this string to the cell barcode if `sample_id` is specified, after removing
        `cb_end_to_remove` and before appending `sample_id`.

    Returns
    -------
    Polars Series with CBs.

    See Also
    --------
    pycisTopic.fragments.filter_fragments_by_cb

    Examples
    --------
    Read gzipped barcodes TSV file to a Polars Series.

    >>> cbs = read_barcodes_file_to_polars_series(
    ...     barcodes_tsv_filename="barcodes.tsv.gz",
    ... )

    Read uncompressed barcodes TSV file to a Polars Series.

    >>> cbs = read_barcodes_file_to_polars_series(
    ...     barcodes_tsv_filename="barcodes.tsv",
    ... )

    Read gzipped barcodes TSV file to a Polars Series and add sample ID to cell
    barcode names after removing `cb_end_to_remove` string from cell barcode and
    appending `cb_sample_separator` to the cell barcode.

    >>> cbs = read_barcodes_file_to_polars_series(
    ...     barcodes_tsv_filename="barcodes.tsv",
    ...     sample_id="sample1",
    ...     cb_end_to_remove="-1",
    ...     cb_sample_separator="___",
    ... )

    """
    cbs = (
        pl.read_csv(
            barcodes_tsv_filename,
            has_header=False,
            separator="\t",
            columns=[0],
            new_columns=["CB"],
            schema={"CB": pl.Categorical(PycisTopicCategoricals.CB)},
        )
        .filter(pl.col("CB").is_not_null())
        .unique(maintain_order=True)
    )

    # Modify cell barcode if sample ID is specified or an empty string.
    if sample_id or sample_id == "":
        separator_and_sample_id = (
            f"{cb_sample_separator + sample_id}" if cb_sample_separator else sample_id
        )

        if not cb_end_to_remove:
            # Append separator and sample ID to cell barcode.
            cbs = cbs.with_columns(
                (pl.col("CB").cast(pl.Utf8) + pl.lit(separator_and_sample_id))
                .cast(pl.Categorical(PycisTopicCategoricals.CB))
                .alias("CB")
            )
        else:
            # Check fraction of cell barcodes which have the `cb_end_to_remove` string at the end.
            fraction_of_CBs_with_end_to_remove = (
                cbs.select(
                    pl.col("CB")
                    .cat.ends_with(cb_end_to_remove)
                    .sum()
                    .alias("CBs_with_end_to_remove"),
                    pl.col("CB").count().alias("CB_count"),
                )
                .select(
                    (pl.col("CBs_with_end_to_remove") / pl.col("CB_count")).alias(
                        "fraction_of_CBs_with_end_to_remove"
                    ),
                )
                .to_series()[0]
            )

            if fraction_of_CBs_with_end_to_remove != 1.0:
                print(
                    f'Warning: Not all cell barcodes in "{barcodes_tsv_filename}" end with '
                    f'"{cb_end_to_remove}". Percentage of cell barcodes with '
                    f'"{cb_end_to_remove}" at the end: '
                    f"{(fraction_of_CBs_with_end_to_remove * 100):.2f}%."
                )

            # Remove `cb_end_to_remove` from the end of the cell barcode before adding
            # separator and sample ID to cell barcode.
            cbs = cbs.with_columns(
                (
                    pl.col("CB").cast(pl.Utf8).str.strip_suffix(cb_end_to_remove)
                    + pl.lit(separator_and_sample_id)
                )
                .cast(pl.Categorical(PycisTopicCategoricals.CB))
                .alias("CB")
            )

    return cbs.to_series()


def get_fragments_per_cb(
    fragments_df_pl: pl.DataFrame,
    min_fragments_per_cb: int = 10,
    collapse_duplicates: bool | None = True,
) -> pl.DataFrame:
    """
    Get number of fragments and duplication ratio per cell barcode.

    Parameters
    ----------
    fragments_df_pl:
        Polars DataFrame with fragments.
        See :func:`pycisTopic.fragments.read_fragments_to_polars_df`.
    min_fragments_per_cb:
        Minimum number of fragments needed per cell barcode to keep the fragments
        for those cell barcodes.
    collapse_duplicates:
        Collapse duplicate fragments (same chromosomal positions and linked to the
        same cell barcode).

    Returns
    -------
    Polars DataFrame with number of fragments, duplication ratio and nucleosome signal per cell barcode.

    See Also
    --------
    pycisTopic.fragments.read_fragments_to_polars_df

    Examples
    --------
    Read gzipped fragments BED file to a Polars DataFrame.

    >>> fragments_df_pl = read_fragments_to_polars_df(
    ...    fragments_bed_filename="fragments.tsv.gz",
    ... )

    Get number of fragments and duplication ratio per cell barcode
    (which have 10 fragments or more after collapsing duplicates).

    >>> fragments_stats_per_cb_df_pl = get_fragments_per_cb(
    ...     fragments_df_pl=fragments_df_pl,
    ...     min_fragments_per_cb=10,
    ...     collapse_duplicates=True,
    ... )

    """
    fragments_count_column = (
        "unique_fragments_count" if collapse_duplicates else "total_fragments_count"
    )

    fragments_stats_per_cb_df_pl = (
        fragments_df_pl.lazy()
        .with_columns((pl.col("End") - pl.col("Start")).alias("fragment_length"))
        .with_columns(
            pl.col("fragment_length").lt(147).alias("nucleosome_free"),
            pl.col("fragment_length").is_between(147, 294).alias("mononucleosome"),
        )
        .group_by("CB", maintain_order=True)
        .agg(
            [
                pl.col("CB_count").sum().cast(pl.UInt32).alias("total_fragments_count"),
                pl.len().cast(pl.UInt32).alias("unique_fragments_count"),
                (
                    pl.col("mononucleosome").sum() / pl.col("nucleosome_free").sum()
                ).alias("nucleosome_signal"),
            ]
        )
        .filter(pl.col(fragments_count_column) > min_fragments_per_cb)
        .sort(fragments_count_column, descending=True)
        .with_row_index(
            name="barcode_rank",
            offset=1,
        )
        .with_columns(pl.col("barcode_rank").cast(pl.UInt32))
        .with_columns(
            (pl.col("total_fragments_count") - pl.col("unique_fragments_count")).alias(
                "duplication_count"
            )
        )
        .with_columns(
            (pl.col("duplication_count") / pl.col("total_fragments_count")).alias(
                "duplication_ratio"
            )
        )
        .select(
            pl.selectors.all() - pl.selectors.by_name("nucleosome_signal"),
            pl.selectors.by_name("nucleosome_signal"),
        )
        .collect()
    )

    return fragments_stats_per_cb_df_pl


def get_cbs_passing_filter(
    fragments_stats_per_cb_df_pl: pl.DataFrame,
    cbs: pl.Series | Sequence | None = None,
    min_fragments_per_cb: int | None = None,
    keep_top_x_cbs: int | None = None,
    collapse_duplicates: bool | None = True,
) -> (pl.Series, pl.DataFrame):
    """
    Get cell barcodes passing the filter.

    Parameters
    ----------
    fragments_stats_per_cb_df_pl
        Polars DataFrame with number of fragments and duplication ratio per cell
        barcode. See :func:`pycisTopic.fragments.get_fragments_per_cb`.
    cbs
        Cell barcodes to keep. If specified, ``min_fragments_per_cb`` and ``min_cbs``
        are ignored.
    min_fragments_per_cb
        Minimum number of fragments needed per cell barcode to keep the cell barcode.
        Only used if ``cbs`` is ``None``, ``min_cbs`` will be ignored.
    keep_top_x_cbs
        Keep the x most abundant cell barcodes based on the number of fragments.
        Only used if ``cbs`` is ``None`` and ``min_fragments_per_cb`` is ``None``.
    collapse_duplicates
        Collapse duplicate fragments (same chromosomal positions and linked to the same
        cell barcode).

    Returns
    -------
    (Cell barcodes passing the filter,
     fragments_stats_per_cb_df_pl filtered by the cell barcodes passing the filter)

    See Also
    --------
    pycisTopic.fragments.filter_fragments_by_cb
    pycisTopic.fragments.get_fragments_per_cb

    Examples
    --------
    Read gzipped fragments BED file to a Polars DataFrame.

    >>> fragments_df_pl = read_fragments_to_polars_df(
    ...     fragments_bed_filename="fragments.tsv.gz",
    ... )

    Get number of fragments and duplication ratio per cell barcode
    (which have 10 fragments or more after collapsing duplicates).

    >>> fragments_stats_per_cb_df_pl = get_fragments_per_cb(
    ...     fragments_df_pl=fragments_df_pl,
    ...     min_fragments_per_cb=10,
    ...     collapse_duplicates=True,
    ... )

    Keep only cell barcodes which have 1000 or more fragments.

    >>> cbs_selected, fragments_stats_per_cb_filtered_df_pl = get_cbs_passing_filter(
    ...     fragments_stats_per_cb_df_pl=fragments_stats_per_cb_df_pl,
    ...     min_fragments_per_cb=1000,
    ...     collapse_duplicates=True,
    ... )

    Keep only the 4000 most abundant cell barcodes based on the number of fragments
    after collapsing duplicates.

    >>> cbs_selected, fragments_stats_per_cb_filtered_df_pl = get_cbs_passing_filter(
    ...     fragments_stats_per_cb_df_pl=fragments_stats_per_cb_df_pl,
    ...     keep_top_x_cbs=4000,
    ...     collapse_duplicates=True,
    ... )

    """
    fragments_count_column = (
        "unique_fragments_count" if collapse_duplicates else "total_fragments_count"
    )

    if cbs:
        cbs_series_pl = cbs_to_cbs_series_pl(cbs)

        fragments_stats_per_cb_filtered_df_pl = fragments_stats_per_cb_df_pl.join(
            other=cbs_series_pl.to_frame(),
            on="CB",
            how="inner",
        )
    elif isinstance(min_fragments_per_cb, int):
        fragments_stats_per_cb_filtered_df_pl = (
            fragments_stats_per_cb_df_pl.lazy()
            .filter(pl.col(fragments_count_column) >= min_fragments_per_cb)
            .collect()
        )
    elif isinstance(keep_top_x_cbs, int):
        fragments_stats_per_cb_filtered_df_pl = (
            fragments_stats_per_cb_df_pl.lazy()
            .sort(fragments_count_column, descending=True)
            .head(keep_top_x_cbs)
            .collect()
        )
    else:
        raise ValueError(
            "Provide a minimal number of barcodes or a minimal number of fragments to select CBs."
        )

    cbs_selected = fragments_stats_per_cb_filtered_df_pl.get_column("CB")

    return cbs_selected, fragments_stats_per_cb_filtered_df_pl


def filter_fragments_by_cb(
    fragments_df_pl: pl.DataFrame,
    cbs: pl.Series | Sequence,
) -> pl.DataFrame:
    """
    Filter fragments by cell barcodes.

    Parameters
    ----------
    fragments_df_pl
        Polars DataFrame with fragments.
    cbs
        List/Polars Series with Cell barcodes.
        See :func:`pycisTopic.fragments.get_cbs_passing_filter` for a way to get a
        filtered list of cell barcodes (``selected_cbs`` variable).

    Returns
    -------
    Polars DataFrame with fragments for the requested cell barcodes.

    See Also
    --------
    pycisTopic.fragments.get_cbs_passing_filter
    pycisTopic.fragments.read_barcodes_file_to_polars_series

    Examples
    --------
    Read gzipped fragments BED file to a Polars DataFrame.

    >>> fragments_df_pl = read_fragments_to_polars_df(
    ...    fragments_bed_filename="fragments.tsv.gz",
    ... )

    List of cell barcodes for which to retain fragments.

    >>> cbs = ["GGACATAAGGGCCACT-1", "ACCTTCATCTTTGAGA-1"]

    Polars DataFrame with fragments for the requested cell barcodes.

    >>> fragments_cb_filtered_df_pl = filter_fragments_by_cb(
    ...     fragments_df_pl=fragments_df_pl,
    ...     cbs=cbs,
    ... )

    List of cell barcodes for which to retain fragments.

    >>> cbs = ["GGACATAAGGGCCACT-1", "ACCTTCATCTTTGAGA-1"]

    Polars DataFrame with fragments for the requested cell barcodes.

    >>> fragments_cb_filtered_df_pl = filter_fragments_by_cb(
    ...     fragments_df_pl=fragments_df_pl,
    ...     cbs=cbs,
    ... )

    List of cell barcodes as a Polars categorical Series for which to retain fragments.

    >>> cbs = pl.Series(
    ...     "CB",
    ...     ["GGACATAAGGGCCACT-1", "ACCTTCATCTTTGAGA-1"],
    ...     dtype=pl.Categorical(PycisTopicCategoricals.CB),
    ... )

    Read list of cell barcodes from a file.

    >>> cbs = read_barcodes_file_to_polars_series("barcodes.tsv")

    Polars DataFrame with fragments for the requested cell barcodes.

    >>> fragments_cb_filtered_df_pl = filter_fragments_by_cb(
    ...     fragments_df_pl=fragments_df_pl,
    ...     cbs=cbs,
    ... )

    """
    cbs_series_pl = cbs_to_cbs_series_pl(cbs)

    fragments_cb_filtered_df_pl = fragments_df_pl.join(
        other=cbs_series_pl.to_frame(),
        on="CB",
        how="inner",
    )

    return fragments_cb_filtered_df_pl


def get_insert_size_distribution(
    fragments_df_pl: pl.DataFrame,
) -> pl.DataFrame:
    """
    Get insert size distribution of fragments.

    Parameters
    ----------
    fragments_df_pl
        Polars DataFrame with fragments.

    Returns
    -------
    Polars DataFrame with fragment counts and fragment ratios for each found insert
    size.

    See Also
    --------
    pycisTopic.fragments.filter_fragments_by_cb

    Examples
    --------
    As input get a Polars DataFrame with fragments for the cell barcodes of interest.
    See `pycisTopic.fragments.filter_fragments_by_cb`

    >>> fragments_cb_filtered_df_pl = filter_fragments_by_cb(
    ...     fragments_df_pl=fragments_df_pl,
    ...     cbs=cbs,
    ... )

    Polars DataFrame with insert size distribution of fragments.

    >>> insert_size_dist_df_pl = get_insert_size_distribution(
    ...     fragments_df_pl=fragments_cb_filtered_df_pl,
    ... )

    """
    insert_size_distribution_df_pl = (
        fragments_df_pl.lazy()
        .with_columns(
            (pl.col("End") - pl.col("Start")).abs().alias("insert_size"),
        )
        .group_by("insert_size")
        .agg([pl.len().cast(pl.UInt32).alias("fragments_count")])
        .sort("insert_size", descending=True)
        .with_columns(
            (pl.col("fragments_count") / pl.col("fragments_count").sum()).alias(
                "fragments_ratio"
            ),
        )
        .collect()
    )

    return insert_size_distribution_df_pl


def get_fragments_in_peaks(
    fragments_df_pl: pl.DataFrame,
    regions_df_pl: pl.DataFrame,
    intersection_engine: str | Literal["ncls", "ruranges"] = "ncls",
) -> pl.DataFrame:
    """
    Get number of total and unique fragments in peaks.

    Parameters
    ----------
    fragments_df_pl
        Polars DataFrame with fragments.
    regions_df_pl
        Polars DataFrame with peak regions (consensus peaks or SCREEN regions).
        See :func:`pycisTopic.fragments.read_bed_to_polars_df` for a way to read a BED
        file with peak regsions.
    intersection_engine
        Engine to use to calculate intersections/overlaps between fragments and regions.
        Options: ``ncls`` or ``ruranges`` (faster).
        Default: ``ncls``.

    Returns
    -------
    Polars DataFrame with total fragment counts and unique fragment counts per region.

    See Also
    --------
    pycisTopic.fragments.filter_fragments_by_cb

    Examples
    --------
    As input get a Polars DataFrame with fragments for the cell barcodes of interest.
    See `pycisTopic.fragments.filter_fragments_by_cb`

    >>> fragments_cb_filtered_df_pl = filter_fragments_by_cb(
    ...     fragments_df_pl=fragments_df_pl,
    ...     cbs=cbs,
    ... )

    Read BED file with consensus peaks or SCREEN regions (get first 3 columns only).

    >>> regions_df_pl = read_bed_to_polars_df(
    ...     bed_filename=screen_regions_bed_filename,
    ...     min_column_count=3,
    ... )

    Polars DataFrame with number of total and unique fragments in peaks.

    >>> fragments_in_peaks_df_pl = get_fragments_in_peaks(
    ...     fragments_df_pl=fragments_cb_filtered_df_pl,
    ...     regions_df_pl=regions_df_pl,
    ...     intersection_engine="ncls",
    ... )

    """
    fragments_in_peaks_df_pl = (
        # Get all fragments that overlap with at least one region.
        gr_intersection(
            regions1_df_pl=fragments_df_pl,
            regions2_df_pl=regions_df_pl,
            how="first",
            regions1_info=True,
            regions2_info=False,
            regions1_coord=True,
            regions2_coord=False,
            regions1_suffix="@1",
            regions2_suffix="@2",
            engine=intersection_engine,
        )
        # Get all fragment file related columns.
        .select(
            pl.col("Chromosome"),
            pl.col("Start@1").alias("Start"),
            pl.col("End@1").alias("End"),
            pl.col("CB"),
            pl.col("CB_count"),
        )
        .group_by("CB", maintain_order=True)
        .agg(
            [
                pl.col("CB_count")
                .sum()
                .cast(pl.UInt32)
                .alias("total_fragments_in_peaks_count"),
                pl.len().cast(pl.UInt32).alias("unique_fragments_in_peaks_count"),
            ]
        )
    )

    return fragments_in_peaks_df_pl


def create_fragment_matrix_from_fragments(
    fragments_bed_filename: str | Path,
    regions_bed_filename: str | Path,
    barcodes_tsv_filename: str | Path,
    blacklist_bed_filename: str | Path | None = None,
    sample_id: str | None = None,
    cb_end_to_remove: str | None = "-1",
    cb_sample_separator: str | None = "___",
    bed_parser_engine: str
    | Literal["polars_lazy", "polars", "pyarrow"] = "polars_lazy",
    intersection_engine: str | Literal["ncls", "ruranges"] = "ncls",
):
    """
    Create fragments matrix from a fragment file and BED file with regions.

    Parameters
    ----------
    fragments_bed_filename
        Fragments BED filename.
    regions_bed_filename
        Consensus peaks / SCREEN regions BED file for which to make the fragments matrix per cell barcode.
    barcodes_tsv_filename
        TSV file with selected cell barcodes after pycisTopic QC filtering.
    blacklist_bed_filename
        BED file with blacklisted regions (Amemiya et al., 2019).
    sample_id
        Optional sample ID to append after cell barcode after removing `cb_end_to_remove`
        and appending `cb_sample_separator`.
    cb_end_to_remove
        Remove this string from the end of the cell barcode if `sample_id` is specified.
    cb_sample_separator
        Add this string to the cell barcode if `sample_id` is specified, after removing
        `cb_end_to_remove` and before appending `sample_id`.
    bed_parser_engine
        BED parsing bed_parser_engine to use to read (gzipped) BED/fragment files.

        Options:
          - ``polars_lazy`` (fastest, low memory usage): Use Polars lazy API (``pl.scan_csv``).
          - ``polars`` (slightly slower, highest memory usage): Use Polars eager API (``pl.read_csv``).
          - ``pyarrow`` (slowest, high memory usage): Use pyarrow CSV reader (``pa.csv.read_csv``).
    intersection_engine
        Engine to use to calculate intersections/overlaps between fragments and regions.
        Options: ``ncls`` or ``ruranges`` (faster).

    Returns
    -------
    (
        counts_fragments_matrix,
        cbs,
        region_ids,
    )

    References
    ----------
    Amemiya, H. M., Kundaje, A., & Boyle, A. P. (2019). The ENCODE blacklist: identification of problematic regions of the genome. Scientific reports, 9(1), 1-5.

    """
    # Create logger
    # level = logging.INFO
    # log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    # handlers = [logging.StreamHandler(stream=sys.stdout)]
    # logging.basicConfig(level=level, format=log_format, handlers=handlers)
    # log = logging.getLogger("cisTopic")

    # Read file with cell barcodes as a Polars Series and add sample ID to cell barcodes.
    cbs = read_barcodes_file_to_polars_series(
        barcodes_tsv_filename=barcodes_tsv_filename,
        sample_id=sample_id,
        cb_end_to_remove=cb_end_to_remove,
        cb_sample_separator=cb_sample_separator,
    )

    # Read fragments file to Polars Dataframe and add sample ID to cell barcodes.
    fragments_df_pl = read_fragments_to_polars_df(
        fragments_bed_filename=fragments_bed_filename,
        bed_parser_engine=bed_parser_engine,
        sample_id=sample_id,
        cb_end_to_remove=cb_end_to_remove,
        cb_sample_separator=cb_sample_separator,
    )

    # Only keep fragments with the requested cell barcode.
    fragments_cb_filtered_df_pl = filter_fragments_by_cb(
        fragments_df_pl=fragments_df_pl,
        cbs=cbs,
    )

    del fragments_df_pl

    # Read regions BED file as a Polars Dataframe.
    regions_df_pl = (
        read_bed_to_polars_df(
            bed_filename=regions_bed_filename,
            bed_parser_engine=bed_parser_engine,
            min_column_count=3,
        )
        .lazy()
        .pipe(add_region_ids_to_bed_df_pl)
        .select(
            pl.col("Chromosome"),
            pl.col("Start"),
            pl.col("End"),
            pl.col("RegionID"),
        )
        .sort(by=["Chromosome", "Start", "End", "RegionID"])
        .collect()
    )

    if blacklist_bed_filename:
        # Read BED file with blacklisted regions .
        blacklist_df_pl = (
            read_bed_to_polars_df(
                bed_filename=blacklist_bed_filename,
                bed_parser_engine=bed_parser_engine,
                min_column_count=3,
            )
            .lazy()
            .select(
                pl.col("Chromosome"),
                pl.col("Start"),
                pl.col("End"),
            )
            .sort(by=["Chromosome", "Start", "End"])
            .collect()
        )

        # Filter out regions that overlap with blacklisted regions.
        regions_df_pl = (
            regions_df_pl.lazy()
            .join(
                # Get all regionIDs that overlap with blacklisted regions.
                gr_overlap(
                    regions1_df_pl=regions_df_pl,
                    regions2_df_pl=blacklist_df_pl,
                    how="first",
                    engine=intersection_engine,
                )
                .lazy()
                .select(
                    pl.col("RegionID"),
                ),
                on="RegionID",
                how="anti",
                maintain_order="left",
            )
            .select(
                pl.col("Chromosome"),
                pl.col("Start"),
                pl.col("End"),
                pl.col("RegionID"),
            )
            .collect()
        )

    # Get accessibility (binary and counts) for each region ID and cell barcode.
    region_cb_df_pl = (
        gr_intersection(
            regions1_df_pl=regions_df_pl,
            regions2_df_pl=fragments_cb_filtered_df_pl,
            # how: Literal["all", "containment", "first", "last"] | str | None = None,
            how="all",
            regions1_info=True,
            regions2_info=True,
            regions1_coord=False,
            regions2_coord=False,
            regions1_suffix="@1",
            regions2_suffix="@2",
            engine=intersection_engine,
        )
        .rename({"CB@2": "CB"})
        .lazy()
        .group_by(["RegionID", "CB"])
        .agg(
            # Get accessibility in binary form.
            pl.lit(1).cast(pl.Int8).alias("accessible_binary"),
            # Get accessibility in count form.
            pl.len().cast(pl.UInt32).alias("accessible_count"),
        )
        .join(
            regions_df_pl.lazy()
            .select(pl.col("RegionID"))
            .with_row_index("region_idx"),
            on="RegionID",
            how="left",
        )
        .join(
            cbs.to_frame().lazy().with_row_index("CB_idx"),
            on="CB",
            how="left",
        )
        .collect()
    )

    # Construct binary accessibility matrix as a sparse matrix
    # (regions as rows and cells as columns).
    counts_fragments_matrix = sp.sparse.csr_matrix(
        (
            # All data points are 1:
            #   - same as: region_cb_df_pl.get_column("accessible_binary").to_numpy()
            #   - for count matrix: region_cb_df_pl.get_column("accessible_count").to_numpy()
            # np.ones(region_cb_df_pl.shape[0], dtype=np.int8),
            region_cb_df_pl.get_column("accessible_count").to_numpy(),
            (
                # Row indices:
                region_cb_df_pl.get_column("region_idx").to_numpy(),
                # Column indices:
                region_cb_df_pl.get_column("CB_idx").to_numpy(),
            ),
        ),
        # Specify shape of the sparse matrix to avoid potential issues if the last
        # (few) rows or (few) columns are empty as this will cause the CB list
        # and regions list to be greater than the dimensions of the sparse matrix.
        shape=(regions_df_pl.height, cbs.len()),
    )

    return (
        counts_fragments_matrix,
        cbs.to_list(),
        regions_df_pl.get_column("RegionID").to_list(),
    )
