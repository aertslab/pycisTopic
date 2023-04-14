from __future__ import annotations

import gc
import logging
import math
import os
import re
from pathlib import Path
from typing import Literal, Sequence, Union

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyranges as pr
from PIL import Image
from scipy import sparse

from pycisTopic.lda_models import CistopicLDAModel


def normalise_filepath(path: str | Path, check_not_directory: bool = True) -> str:
    """
    Create a string path, expanding the home directory if present.

    """
    path = os.path.expanduser(path)
    if check_not_directory and os.path.exists(path) and os.path.isdir(path):
        raise IsADirectoryError(f"Expected a file path; {path!r} is a directory")
    return path


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
        .with_columns(
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
