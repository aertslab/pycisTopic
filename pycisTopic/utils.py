from __future__ import annotations

import gc
import gzip
import logging
import math
import os
import re
import sys
from pathlib import Path
from operator import itemgetter
from typing import Literal, Sequence, Union

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv
import pyranges as pr
from PIL import Image
from scipy import sparse

from pycisTopic.lda_models import CistopicLDAModel


def format_path(path: str | Path) -> str:
    """
    Create a string path, expanding the home directory if present.
    """

    return os.path.expanduser(path)


def coord_to_region_names(df_pl: pl.DataFrame) -> list[str]:
    """
    Convert Polars DataFrame with fragments to region names.

    Parameters
    ----------
    df_pl

    Returns
    -------

    List of region names.
    """

    df_pl.select(
        [
            (
                pl.col("Chromosome").cast(pl.Utf8)
                + ":"
                + pl.col("Start").cast(pl.Utf8)
                + "-"
                + pl.col("End").cast(pl.Utf8)
            ).alias("RegionIDs")
        ]
    ).get_column("RegionIDs").to_list()


def region_names_to_coordinates(region_names: Sequence[str]) -> pd.DataFrame:
    """
    Create Pandas DataFrame with region IDs to coordinates mapping.

    Parameters
    ----------
    region_names: List of region names in "chrom:start-end" format.

    Returns
    -------

    Pandas DataFrame with region IDs to coordinates mapping.
    """

    region_df = (
        pl.DataFrame(
            data=region_names,
            columns=["RegionIDs"],
        )
        .with_columns(
            pl.col("RegionIDs")
            # Replace first ":" with "-".
            .str.replace(":", "-")
            # Split on "-" to generate 3 parts: "Chromosome", "Start" and "End"
            .str.split_exact("-", 2)
            # Give sensible names to each splitted part.
            .struct.rename_fields(
                ["Chromosome", "Start", "End"],
            ).alias("RegionIDsFields")
        )
        # Unpack "RegionIDsFields" struct column and create Chromosome", "Start" and "End" columns.
        .unnest("RegionIDsFields")
        .with_column(
            # Convert "Start" and "End" string columns to int32 columns.
            pl.col(["Start", "End"]).cast(pl.Int32)
        )
        # Convert to Pandas.
        .to_pandas()
    )

    # Set RegionIDs as index.
    region_df.set_index("RegionIDs", inplace=True)

    return region_df


def get_position_index(
    query_list: Sequence[str], target_list: Sequence[str]
) -> Sequence[int]:
    d = {k: idx for idx, k in enumerate(target_list)}
    index = [d[k] for k in query_list]
    return index


def subset_list(target_list: Sequence[str], index_list: Sequence[int]) -> Sequence[str]:
    return list(map(target_list.__getitem__, index_list))


def non_zero_rows(matrix: Union[sparse.csr_matrix, np.ndarray]):
    if isinstance(matrix, sparse.csr_matrix):
        # Remove all explicit zeros in sparse matrix.
        matrix.eliminate_zeros()
        # Get number of non zeros per row and get indices for each row which is
        # not completely zero.
        return np.nonzero(matrix.getnnz(axis=1))[0]
    else:
        # For non sparse matrices.
        return np.nonzero(np.count_nonzero(matrix, axis=1))[0]


def loglikelihood(nzw, ndz, alpha, eta):
    D = ndz.shape[0]
    n_topics = ndz.shape[1]
    vocab_size = nzw.shape[1]

    const_prior = (n_topics * math.lgamma(alpha) - math.lgamma(alpha * n_topics)) * D
    const_ll = (
        vocab_size * math.lgamma(eta) - math.lgamma(eta * vocab_size)
    ) * n_topics

    # calculate log p(w|z)
    topic_ll = 0
    for k in range(n_topics):
        sum = eta * vocab_size
        for w in range(vocab_size):
            if nzw[k, w] > 0:
                topic_ll = math.lgamma(nzw[k, w] + eta)
                sum += nzw[k, w]
        topic_ll -= math.lgamma(sum)

    # calculate log p(z)
    doc_ll = 0
    for d in range(D):
        sum = alpha * n_topics
        for k in range(n_topics):
            if ndz[d, k] > 0:
                doc_ll = math.lgamma(ndz[d, k] + alpha)
                sum += ndz[d, k]
        doc_ll -= math.lgamma(sum)

    ll = doc_ll - const_prior + topic_ll - const_ll
    return ll


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten()  # all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative
    array = np.sort(array)  # values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements
    return (np.sum((2 * index - n - 1) * array)) / (
        n * np.sum(array)
    )  # Gini coefficient


def regions_overlap(target, query):
    # Read input
    if isinstance(target, str):
        target_pr = pr.read_bed(target)
    if isinstance(target, list):
        target_pr = pr.PyRanges(region_names_to_coordinates(target))
    if isinstance(target, pr.PyRanges):
        target_pr = target
    # Read input
    if isinstance(query, str):
        query_pr = pr.read_bed(query)
    if isinstance(query, list):
        query_pr = pr.PyRanges(region_names_to_coordinates(query))
    if isinstance(query, pr.PyRanges):
        query_pr = query

    target_pr = target_pr.overlap(query_pr)
    selected_regions = (
        target_pr.Chromosome.astype(str)
        + ":"
        + target_pr.Start.astype(str)
        + "-"
        + target_pr.End.astype(str)
    ).to_list()
    return selected_regions


def load_cisTopic_model(path_to_cisTopic_model_matrices):
    metrics = None
    coherence = None
    marg_topic = None
    topic_ass = None
    cell_topic = pd.read_feather(path_to_cisTopic_model_matrices + "cell_topic.feather")
    cell_topic.index = ["Topic" + str(x) for x in range(1, cell_topic.shape[0] + 1)]
    topic_region = pd.read_feather(
        path_to_cisTopic_model_matrices + "topic_region.feather"
    )
    topic_region.index = ["Topic" + str(x) for x in range(1, topic_region.shape[0] + 1)]
    topic_region = topic_region.T
    parameters = None
    model = CistopicLDAModel(
        metrics, coherence, marg_topic, topic_ass, cell_topic, topic_region, parameters
    )
    return model


def prepare_tag_cells(cell_names, split_pattern="___"):
    if split_pattern == "-":
        new_cell_names = [
            re.findall(r"^[ACGT]*-[0-9]+-", x)[0].rstrip("-")
            if len(re.findall(r"^[ACGT]*-[0-9]+-", x)) != 0
            else x
            for x in cell_names
        ]
        new_cell_names = [
            re.findall(r"^\w*-[0-9]*", new_cell_names[i])[0].rstrip("-")
            if (len(re.findall(r"^\w*-[0-9]*", new_cell_names[i])) != 0)
            & (new_cell_names[i] == cell_names[i])
            else new_cell_names[i]
            for i in range(len(new_cell_names))
        ]
    else:
        new_cell_names = [x.split(split_pattern)[0] for x in cell_names]

    return new_cell_names


def multiplot_from_generator(
    g, num_columns, n_plots, figsize=None, plot=True, save=None
):
    if save is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)
    # call 'next(g)' to get past the first 'yield'
    next(g)
    # default to 15-inch rows, with square subplots
    if figsize is None:
        if num_columns == 1:
            figsize = (5, 5)
        else:
            num_rows = int(np.ceil(n_plots / num_columns))
            figsize = (6.4 * num_columns, 4.8 * num_rows)

    if num_columns > 1:
        fig = plt.figure(figsize=figsize)
        num_rows = int(np.ceil(n_plots / num_columns))
    plot = 0
    try:
        while True:
            # call plt.figure once per row
            if num_columns == 1:
                fig = plt.figure(figsize=figsize)
                ax = plt.subplot(1, 1, 1)
                if save is not None:
                    pdf.savefig(fig, bbox_inches="tight")
            if num_columns > 1:
                ax = plt.subplot(num_rows, num_columns, plot + 1)
                ax.autoscale(enable=True)
                plot = plot + 1
            next(g)
    except StopIteration:
        if num_columns == 1:
            if save is not None:
                pdf.savefig(fig, bbox_inches="tight")
        pass
    if num_columns > 1:
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if save is not None:
            pdf.savefig(fig, bbox_inches="tight")
    if save is not None:
        pdf.close()
    if not plot:
        plt.close()


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight", format="png", dpi=500, transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    return img


def collapse_duplicates(df):
    """
    Collapse duplicates from fragments df
    """
    a = df.values
    sidx = np.lexsort(a[:, :4].T)
    b = a[sidx, :4]
    m = np.concatenate(([True], (b[1:] != b[:-1]).any(1), [True]))
    out_ar = np.column_stack((b[m[:-1], :4], np.diff(np.flatnonzero(m) + 1)))
    return pd.DataFrame(out_ar, columns=["Chromosome", "Start", "End", "Name", "Score"])


def get_tss_matrix(fragments, flank_window, tss_space_annotation):
    """
    Get TSS matrix
    """
    overlap_with_TSS = fragments.join(tss_space_annotation, nb_cpu=1).df
    if len(overlap_with_TSS) == 0:
        return

    overlap_with_TSS["Strand"] = overlap_with_TSS["Strand"].astype(np.int32)
    overlap_with_TSS["start_pos"] = -(
        np.int32(overlap_with_TSS["Start_b"].values)
        + np.int32(flank_window)
        - np.int32(overlap_with_TSS["Start"].values)
    ) * np.int32(overlap_with_TSS["Strand"].values)
    overlap_with_TSS["end_pos"] = -(
        np.int32(overlap_with_TSS["Start_b"].values)
        + np.int32(flank_window)
        - np.int32(overlap_with_TSS["End"].values)
    ) * np.int32(overlap_with_TSS["Strand"].values)
    # We split them to also keep the start position of reads whose start is
    # in the space and their end not and viceversa
    overlap_with_TSS_start = overlap_with_TSS[
        (overlap_with_TSS["start_pos"].values <= flank_window)
        & (overlap_with_TSS["start_pos"].values >= -flank_window)
    ]
    overlap_with_TSS_end = overlap_with_TSS[
        (overlap_with_TSS["end_pos"].values <= flank_window)
        & (overlap_with_TSS["end_pos"].values >= -flank_window)
    ]
    overlap_with_TSS_start["rel_start_pos"] = (
        overlap_with_TSS_start["start_pos"].values + flank_window
    )
    overlap_with_TSS_end["rel_end_pos"] = (
        overlap_with_TSS_end["end_pos"].values + flank_window
    )
    cut_sites_TSS = pd.concat(
        [
            overlap_with_TSS_start[["Name", "rel_start_pos"]].rename(
                columns={"Name": "Barcode", "rel_start_pos": "Position"}
            ),
            overlap_with_TSS_end[["Name", "rel_end_pos"]].rename(
                columns={"Name": "Barcode", "rel_end_pos": "Position"}
            ),
        ],
        axis=0,
    )

    cut_sites_TSS["Barcode"] = cut_sites_TSS["Barcode"].astype("category")
    cut_sites_TSS["Position"] = cut_sites_TSS["Position"].astype("category")
    TSS_matrix = (
        cut_sites_TSS.groupby(["Position", "Barcode"], observed=True, sort=False)
        .size()
        .unstack(level="Position", fill_value=0)
        .astype(np.int32)
    )
    del cut_sites_TSS
    gc.collect()

    return TSS_matrix


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
            f'BED file needs to have at least {min_column_count} columns. '
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
        raise ValueError(f'Unsupported engine value "{engine}" (allowed: ["polars", "pyarrow"]).')

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
    if "Score" not in fragments_df_pl.columns or fragments_df_pl.schema["Score"] == pl.Utf8:
        fragments_df_pl = fragments_df_pl.groupby(["Chromosome", "Start", "End", "Name"]).agg(
            pl.count().cast(pl.Int32()).alias("Score")
        )
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
    pa_schema_fixed_categoricals_list.append(
        pa.field("__index_level_0__", pa.int64())
    )

    # Create pyarrow schema so categorical columns in chromosome-strand Polars DataFrames or chromosome Polars
    # DataFrames can be cast to a pyarrow supported dictionary type, which can be converted to a Pandas categorical.
    pa_schema_fixed_categoricals = pa.schema(pa_schema_fixed_categoricals_list)

    # Add (row) index column to Polars DataFrame with BED entries so original row indexes of BED entries can be tracked
    # by PyRanges (not sure if pyranges uses those index values or not).
    bed_with_idx_df_pl = (
        bed_df_pl
        # Add index column and cast it from UInt32 to Int64
        .with_row_count("__index_level_0__")
        .with_column(pl.col("__index_level_0__").cast(pl.Int64))
        # Put index column as last column.
        .select(pl.col(pa_schema_fixed_categoricals.names))
    )

    def create_per_chrom_or_chrom_strand_df_pd(per_chrom_or_chrom_strand_bed_df_pl: pl.DataFrame) -> pd.DataFrame:
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
            per_chrom_or_chrom_strand_bed_df_pl
            .to_arrow()
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
            chrom_strand: create_per_chrom_or_chrom_strand_df_pd(per_chrom_or_chrom_strand_bed_df_pl)
            for chrom_strand, per_chrom_or_chrom_strand_bed_df_pl in sorted(
                # Partition Polars DataFrame with BED entries per chromosome-strand (stranded).
                bed_with_idx_df_pl.partition_by(
                    groups=["Chromosome", "Strand"], maintain_order=False, as_dict=True
                ).items(),
                key=itemgetter(0)
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
                key=itemgetter(0)
            )
        }

    df_pr.__dict__["features"] = pr.genomicfeatures.GenomicFeaturesMethods
    df_pr.__dict__["statistics"] = pr.statistics.StatisticsMethods

    return df_pr
