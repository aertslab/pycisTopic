from __future__ import annotations

import gzip
from operator import itemgetter
from typing import Literal, Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv
import pyranges as pr

from pycisTopic.utils import format_path


def read_fragments_to_pyranges(
    fragments_bed_filename: str,
    engine: Union[
        str, Literal["polars"], Literal["pyarrow"], Literal["pandas"]
    ] = "polars",
) -> pr.PyRanges:
    """
    Read fragments BED file to PyRanges object.

    Parameters
    ----------
    fragments_bed_filename: Fragments BED filename.
    use_polars: Use Polars instead of Pandas for reading the fragments BED file.

    Returns
    -------
    PyRanges object of fragments.
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

    fragments_bed_filename = format_path(fragments_bed_filename)

    # Set the correct open function, depending upon if the fragments BED file is gzip compressed or not.
    open_fn = gzip.open if fragments_bed_filename.endswith(".gz") else open

    skip_rows = 0
    nbr_columns = 0
    with open_fn(fragments_bed_filename, "rt") as fragments_bed_fh:
        for line in fragments_bed_fh:
            # Remove newlines and spaces.
            line = line.strip()

            if not line or line.startswith("#"):
                # Count number of empty lines and lines which start with a comment before the actual data.
                skip_rows += 1
            else:
                # Get number of columns from the first real BED entry.
                nbr_columns = len(line.split("\t"))

                # Stop reading the BED file.
                break

    if nbr_columns < 4:
        raise ValueError(
            f'Fragments BED file needs to have at least 4 columns. "{fragments_bed_filename}" contains only '
            f"{nbr_columns} columns."
        )

    if not engine:
        engine = "pandas"

    if engine == "polars":
        # Read fragments BED file with Polars.
        df = pl.read_csv(
            fragments_bed_filename,
            has_header=False,
            skip_rows=skip_rows,
            sep="\t",
            use_pyarrow=False,
            new_columns=bed_column_names[:nbr_columns],
            dtypes={
                "Chromosome": pl.Categorical,
                "Start": pl.Int32,
                "End": pl.Int32,
                "Name": pl.Categorical,
                "Strand": pl.Categorical,
            },
        ).to_pandas()
    elif engine == "pyarrow":
        # Read fragments BED file with pyarrow.
        df = pa.csv.read_csv(
            fragments_bed_filename,
            read_options=pa.csv.ReadOptions(
                use_threads=True,
                skip_rows=skip_rows,
                column_names=bed_column_names[:nbr_columns],
            ),
            parse_options=pa.csv.ParseOptions(
                delimiter="\t",
                quote_char=False,
                escape_char=False,
                newlines_in_values=False,
            ),
            convert_options=pa.csv.ConvertOptions(
                column_types={
                    "Chromosome": pa.dictionary(pa.int32(), pa.string()),
                    "Start": pa.int32(),
                    "End": pa.int32(),
                    "Name": pa.dictionary(pa.int32(), pa.string()),
                    "Strand": pa.dictionary(pa.int32(), pa.string()),
                },
            ),
        ).to_pandas()
    else:
        # Read fragments BED file with Pandas.
        df = pd.read_table(
            fragments_bed_filename,
            sep="\t",
            skiprows=skip_rows,
            header=None,
            names=bed_column_names[:nbr_columns],
            doublequote=False,
            engine="c",
            dtype={
                "Chromosome": "category",
                "Start": np.int32,
                "End": np.int32,
                "Name": "category",
                "Strand": "category",
            },
        )

    # Convert Pandas DataFrame to PyRanges DataFrame.
    # This will convert "Chromosome" and "Strand" columns to pd.Categorical.
    return pr.PyRanges(df)


def read_bed_to_polars_df(
    bed_filename: str,
    engine: str | Literal["polars"] | Literal["pyarrow"] = "pyarrow",
    min_column_count: int = 3,
) -> pl.DataFrame:
    """
    Read BED file to a Polars DataFrame.

    Parameters
    ----------
    bed_filename: BED filename.
    engine: Use Polars or pyarrow to read the BED file (default: pyarrow).
    min_column_count: Minimum number of required columns needed in BED file.

    Returns
    -------
    Polars DataFrame with BED entries.

    Examples
    --------
    >>> bed_df_pl = read_bed_to_polars_df("test.bed", engine="pyarrow")
    >>> bed_with_at_least_4_columns_df_pl = read_bed_to_polars_df("test.bed", engine="pyarrow", min_column_count=4)
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

    bed_filename = format_path(bed_filename)

    # Set the correct open function, depending upon if the fragments BED file is gzip compressed or not.
    open_fn = gzip.open if bed_filename.endswith(".gz") else open

    skip_rows = 0
    column_count = 0
    with open_fn(bed_filename, "rt") as bed_fh:
        for line in bed_fh:
            # Remove newlines and spaces.
            line = line.strip()

            if not line or line.startswith("#"):
                # Count number of empty lines and lines which start with a comment before the actual data.
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

    # Set global string cache so categorical columns from multiple Polars DataFrames can be joined later, if necessary.
    pl.Config.set_global_string_cache()

    if engine == "polars":
        # Read BED file with Polars.
        bed_df_pl = pl.read_csv(
            bed_filename,
            has_header=False,
            skip_rows=skip_rows,
            sep="\t",
            use_pyarrow=False,
            new_columns=bed_column_names[:column_count],
            dtypes={
                "Chromosome": pl.Categorical,
                "Start": pl.Int32,
                "End": pl.Int32,
                "Name": pl.Categorical,
                "Strand": pl.Categorical,
            },
        )
    elif engine == "pyarrow":
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
            )
        )
    else:
        raise ValueError(
            f'Unsupported engine value "{engine}" (allowed: ["polars", "pyarrow"]).'
        )

    return bed_df_pl


def read_fragments_to_polars_df(
    fragments_bed_filename: str,
    engine: str | Literal["polars"] | Literal["pyarrow"] = "pyarrow",
) -> pl.DataFrame:
    """
    Read fragments BED file to a Polars DataFrame.

    If fragments don't have a Score column, a Score columns is created by counting
    the number of fragments with the same chromosome, start, end and CB.

    Parameters
    ----------
    fragments_bed_filename: Fragments BED filename.
    engine: Use Polars or pyarrow to read the fragments BED file (default: pyarrow).

    Returns
    -------
    Polars DataFrame with fragments.
    """

    fragments_df_pl = read_bed_to_polars_df(
        bed_filename=fragments_bed_filename,
        engine=engine,
        min_column_count=4,
    )

    # If no score is provided or score column is ".", generate a score column with the number of fragments which have
    # the same chromosome, start, end and CB.
    if (
        "Score" not in fragments_df_pl.columns
        or fragments_df_pl.schema["Score"] == pl.Utf8
    ):
        fragments_df_pl = fragments_df_pl.groupby(
            ["Chromosome", "Start", "End", "Name"]
        ).agg(pl.count().cast(pl.Int32()).alias("Score"))
    else:
        fragments_df_pl = fragments_df_pl.with_column(pl.col("Score").cast(pl.Int32()))

    return fragments_df_pl


def create_pyranges_from_polars_df(bed_df_pl: pl.DataFrame) -> pr.PyRanges:
    """
    Create PyRanges DataFrame from Polars DataFrame.

    Parameters
    ----------
    bed_df_pl: Polars DataFrame containing BED entries.

    Returns
    -------
    PyRanges DataFrame.
    """

    # Calling the PyRanges init function with a Pandas DataFrame causes too much overhead as it will create categorical
    # columns for Chromosome and Strand columns, even if they are already categorical. It will also create a Pandas
    # DataFrame per chromosome-strand (stranded) combination or a Pandas DataFrame per chromosome (unstranded).
    # So instead, create the PyRanges object manually with the use of Polars and pyarrow.

    # Create empty PyRanges object, which will be populated later.
    df_pr = pr.PyRanges()

    # Check if there is a "Strand" column with only "+" and/or "-"
    is_stranded = (
        set(bed_df_pl.get_column("Strand").unique().to_list()).issubset({"+", "-"})
        if "Strand" in bed_df_pl
        else False
    )

    # Create PyArrow schema for Polars DataFrame, where categorical columns are cast from
    # pa.dictionary(pa.uint32(), pa.large_string()) to pa.dictionary(pa.int32(), pa.large_string())
    # as for the later conversion to a Pandas DataFrame, only the latter is supported by pyarrow.
    pa_schema_fixed_categoricals_list = []
    for pa_field in bed_df_pl.head(1).to_arrow().schema:
        if pa_field.type == pa.dictionary(pa.uint32(), pa.large_string()):
            # ArrowTypeError: Converting unsigned dictionary indices to Pandas not yet supported, index type: uint32
            pa_schema_fixed_categoricals_list.append(
                pa.field(pa_field.name, pa.dictionary(pa.int32(), pa.large_string()))
            )
        else:
            pa_schema_fixed_categoricals_list.append(
                pa.field(pa_field.name, pa_field.type)
            )

    # Add entry for index as last column.
    pa_schema_fixed_categoricals_list.append(pa.field("__index_level_0__", pa.int64()))

    # Create pyarrow schema so categorical columns in chromosome-strand Polars DataFrames or chromosome Polars
    # DataFrames can be cast to a pyarrow supported dictionary type, which can be converted to a Pandas categorical.
    pa_schema_fixed_categoricals = pa.schema(pa_schema_fixed_categoricals_list)

    # Add (row) index column to Polars DataFrame with BED entries so original row indexes of BED entries can be tracked
    # by PyRanges (not sure if pyranges uses those index values or not).
    bed_with_idx_df_pl = (
        bed_df_pl
        # Add index column and cast it from UInt32 to Int64
        .with_row_count("__index_level_0__").with_column(
            pl.col("__index_level_0__").cast(pl.Int64)
        )
        # Put index column as last column.
        .select(pl.col(pa_schema_fixed_categoricals.names))
    )

    def create_per_chrom_or_chrom_strand_df_pd(
        per_chrom_or_chrom_strand_bed_df_pl: pl.DataFrame,
    ) -> pd.DataFrame:
        """
        Create per chromosome (unstranded) or per chromosome-strand (stranded) Pandas DataFrame for PyRanges from
        equivalent Polars DataFrame.

        Parameters
        ----------
        per_chrom_or_chrom_strand_bed_df_pl:
            Polars DataFrame partitioned by chromosome (unstranded) or chromosome-strand (stranded).

        Returns
        -------
        Pandas DataFrame partitioned by chromosome (unstranded) or chromosome-strand (stranded).
        """

        # Convert per chromosome (unstranded) or per chromosome-strand (stranded) Polars DataFrame with BED entries to
        # a pyarrow table and change categoricals dictionary type to Pandas compatible categorical type and convert to
        # a Pandas DataFrame.
        per_chrom_or_chrom_strand_bed_df_pd = (
            per_chrom_or_chrom_strand_bed_df_pl.to_arrow()
            .cast(pa_schema_fixed_categoricals)
            .to_pandas()
        )

        # Set Pandas index inplace and remove index name.
        per_chrom_or_chrom_strand_bed_df_pd.set_index("__index_level_0__", inplace=True)
        per_chrom_or_chrom_strand_bed_df_pd.index.name = None

        return per_chrom_or_chrom_strand_bed_df_pd

    if is_stranded:
        # Populate empty PyRanges object directly with per chromosome and strand Pandas DataFrames (stranded).
        df_pr.__dict__["dfs"] = {
            chrom_strand: create_per_chrom_or_chrom_strand_df_pd(
                per_chrom_or_chrom_strand_bed_df_pl
            )
            for chrom_strand, per_chrom_or_chrom_strand_bed_df_pl in sorted(
                # Partition Polars DataFrame with BED entries per chromosome-strand (stranded).
                bed_with_idx_df_pl.partition_by(
                    groups=["Chromosome", "Strand"], maintain_order=False, as_dict=True
                ).items(),
                key=itemgetter(0),
            )
        }
    else:
        # Populate empty PyRanges object directly with per chromosome Pandas DataFrames (unstranded).
        df_pr.__dict__["dfs"] = {
            chrom: create_per_chrom_or_chrom_strand_df_pd(per_chrom_bed_df_pl)
            for chrom, per_chrom_bed_df_pl in sorted(
                # Partition Polars DataFrame with BED entries per chromosome (unstranded).
                bed_with_idx_df_pl.partition_by(
                    groups=["Chromosome"], maintain_order=False, as_dict=True
                ).items(),
                key=itemgetter(0),
            )
        }

    df_pr.__dict__["features"] = pr.genomicfeatures.GenomicFeaturesMethods
    df_pr.__dict__["statistics"] = pr.statistics.StatisticsMethods

    return df_pr
