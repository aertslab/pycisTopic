import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def plot_barcode_rank(
    fragments_stats_per_cb_df: pl.DataFrame,
    ax: plt.Axes | None = None,
    **matplotlib_plot_kwargs
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    ax.plot(
        fragments_stats_per_cb_df.get_column("barcode_rank"),
        fragments_stats_per_cb_df.get_column("unique_fragments_count"),
        **matplotlib_plot_kwargs,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Barcode Rank", fontsize=10)
    ax.set_ylabel("Number of fragments", fontsize=10)
    return ax

def plot_insert_size_distribution(
    fragments_insert_size_dist_df: pl.DataFrame,
    ax: plt.Axes | None = None,
    insert_size_distriubtion_xlim: tuple[int, int] = (0, 1000),
    **matplotlib_plot_kwargs,
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    ax.plot(
        fragments_insert_size_dist_df.get_column("insert_size"),
        fragments_insert_size_dist_df.get_column("fragments_ratio"),
        **matplotlib_plot_kwargs,
    )
    ax.set_xlabel("Fragment size", fontsize=10)
    ax.set_ylabel("Fragments ratio", fontsize=10)
    ax.set_xlim(*insert_size_distriubtion_xlim)
    return ax

def plot_tss_enrichment(
        tss_norm_matrix_sample_df,
        ax: plt.Axes | None = None,
        **matplotlib_plot_kwargs
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    ax.plot(
        tss_norm_matrix_sample_df.get_column("position_from_tss"),
        tss_norm_matrix_sample_df.get_column("normalized_tss_enrichment"),
        **matplotlib_plot_kwargs,
    )
    ax.set_xlabel("Position from TSS", fontsize=10)
    ax.set_ylabel("Normalized enrichment", fontsize=10)
    ax.set_xlim(
        tss_norm_matrix_sample_df.select(
            pl.col("position_from_tss").min().alias("min_position"),
            pl.col("position_from_tss").max().alias("max_position"),
        ).row(0)
    )
    return ax


def plot_sample_stats(
    sample_id: str,
    pycistopic_qc_output_dir: str | Path,
    save: str | Path | None = None,
    insert_size_distriubtion_xlim: tuple[int, int] = (0, 1000),
    sample_alias: str | None = None,
) -> plt.Figure:
    # check if files exist
    for file_name in [
        f"{sample_id}.fragments_insert_size_dist.parquet",
        f"{sample_id}.fragments_stats_per_cb.parquet",
        f"{sample_id}.tss_norm_matrix_sample.parquet",
    ]:
        if not os.path.isfile(os.path.join(pycistopic_qc_output_dir, file_name)):
            raise FileNotFoundError(
                f"Could not find {file_name} in {pycistopic_qc_output_dir}"
            )

    fragments_insert_size_dist_df = pl.read_parquet(
        os.path.join(
            pycistopic_qc_output_dir, f"{sample_id}.fragments_insert_size_dist.parquet"
        )
    )

    fragments_stats_per_cb_df = pl.read_parquet(
        os.path.join(
            pycistopic_qc_output_dir, f"{sample_id}.fragments_stats_per_cb.parquet"
        )
    )

    tss_norm_matrix_sample_df = pl.read_parquet(
        os.path.join(
            pycistopic_qc_output_dir, f"{sample_id}.tss_norm_matrix_sample.parquet"
        )
    )

    ncols = 3
    nrows = 1
    figsize = (6.4 * ncols, 4.8 * nrows)

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize,
        layout = "constrained")

    # Set centered sample title for 3 combined plots.
    fig.suptitle(sample_id if sample_alias is None else sample_alias)

    # Plot barcode rank plot on the left.
    plot_barcode_rank(
        fragments_stats_per_cb_df,
        ax = axs[0]
    )

    # Plot insert size distribution plot in the center.
    plot_insert_size_distribution(
        fragments_insert_size_dist_df,
        ax = axs[1],
        insert_size_distriubtion_xlim = insert_size_distriubtion_xlim
    )

    # Plot TSS enrichment plot on the right.
    plot_tss_enrichment(
        tss_norm_matrix_sample_df,
        ax = axs[2]
    )

    if save:
        fig.savefig(save)
    else:
        fig.show()

    return fig

def _plot_fragment_stats(
    fragments_stats_per_cb_df: pl.DataFrame,
    ax: plt.Axes,
    x_var: str,
    y_var: str,
    c_var: str,
    **matplotlib_plot_kwargs
) -> plt.Axes:
    fragments_stats_per_cb_df = fragments_stats_per_cb_df.sort(
        by = c_var, descending = False
    )
    ax.scatter(
        x = fragments_stats_per_cb_df.get_column(x_var).to_numpy(),
        y = fragments_stats_per_cb_df.get_column(y_var).to_numpy(),
        c = fragments_stats_per_cb_df.get_column(c_var).to_numpy(),
        **matplotlib_plot_kwargs
    )
    return ax

def plot_barcode_stats(
    sample_id: str,
    pycistopic_qc_output_dir: str | Path,
    unique_fragments_threshold: int | None = None,
    tss_enrichment_threshold: float | None = None,
    frip_threshold: float | None = None,
    duplication_ratio_threshold: float | None = None,
    detailed_title: bool = True,
    bc_passing_filters: list[str] | None = None,
    sample_alias: str | None = None,
    save: str | Path | None = None,
) -> plt.Figure:
    # check if files exist
    if not os.path.isfile(os.path.join(pycistopic_qc_output_dir, f"{sample_id}.fragments_stats_per_cb.parquet")):
        raise FileNotFoundError(
            f"Could not find {sample_id}.fragments_stats_per_cb.parquet in {pycistopic_qc_output_dir}")

    fragments_stats_per_cb_df = pl.read_parquet(
        os.path.join(
            pycistopic_qc_output_dir, f"{sample_id}.fragments_stats_per_cb.parquet"
        )
    )

    if detailed_title and bc_passing_filters is  None:
        Warning("bc_passing_filters is None, no detailed title will be shown")

    ncols = 3
    nrows = 1
    figsize = (4 * ncols, 4 * nrows)
    fig, axs = plt.subplots(
        figsize = figsize, nrows = nrows, ncols = ncols,
        sharex = True,
        layout = "constrained")

    # Plot TSS enrichment vs unique number of fragments on the left.
    axs[0] = _plot_fragment_stats(
        fragments_stats_per_cb_df,
        ax = axs[0],
        x_var = "unique_fragments_in_peaks_count",
        y_var = "tss_enrichment",
        c_var = "pdf_values_for_tss_enrichment",
        s = 10,
        edgecolors = None,
        marker = "+",
        cmap = "viridis"
    )
    axs[0].set_ylabel("TSS enrichment")

    # Plot FRIP vs unique number of fragments in the center.
    axs[1] = _plot_fragment_stats(
        fragments_stats_per_cb_df,
        ax = axs[1],
        x_var = "unique_fragments_in_peaks_count",
        y_var = "fraction_of_fragments_in_peaks",
        c_var = "pdf_values_for_fraction_of_fragments_in_peaks",
        s = 10,
        edgecolors = None,
        marker = "+",
        cmap = "viridis"
    )
    axs[1].set_ylabel("Fraction of fragments in peaks")

    # Plot duplication ratio vs unique number of fragments on the right.
    axs[2] = _plot_fragment_stats(
        fragments_stats_per_cb_df,
        ax = axs[2],
        x_var = "unique_fragments_in_peaks_count",
        y_var = "duplication_ratio",
        c_var = "pdf_values_for_duplication_ratio",
        s = 10,
        edgecolors = None,
        marker = "+",
        cmap = "viridis"
    )
    axs[2].set_ylabel("Duplication ratio")

    # plot thresholds
    if unique_fragments_threshold is not None:
        for ax in axs:
            ax.axvline(x = unique_fragments_threshold, color = "r", linestyle = "--")

    if tss_enrichment_threshold is not None:
        axs[0].axhline(y = tss_enrichment_threshold, color = "r", linestyle = "--")

    if frip_threshold is not None:
        axs[1].axhline(y = frip_threshold, color = "r", linestyle = "--")

    if duplication_ratio_threshold is not None:
        axs[2].axhline(y = duplication_ratio_threshold, color = "r", linestyle = "--")

    # Set x-axis to log scale and plot x-axis label.
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("Number of (unique) fragments in regions")

    if detailed_title and bc_passing_filters is not None:
        (
            median_no_fragments,
            median_tss_enrichment,
            fraction_of_fragments_in_peaks,
        ) = (
            fragments_stats_per_cb_df.filter(pl.col("CB").is_in(bc_passing_filters))
            .select(
                pl.col("unique_fragments_in_peaks_count").median(),
                pl.col("tss_enrichment").median(),
                pl.col("fraction_of_fragments_in_peaks").median(),
            )
            .row(0)
        )
        title = f"{sample_id}\n" if sample_alias is None else f"{sample_alias}\n"
        title += f"Kept {len(bc_passing_filters)} cells after filtering\n"
        title += f"Median Unique Fragments: {median_no_fragments:.0f}\n"
        title += f"Median TSS Enrichment: {median_tss_enrichment:.2f}\n"
        title += f"Median FRIP: {fraction_of_fragments_in_peaks:.2f}\n"
        if (unique_fragments_threshold is not None) \
            or (tss_enrichment_threshold is not None) \
            or (frip_threshold is not None) \
            or (duplication_ratio_threshold is not None):
            title += "Thresholds:\n"
        if unique_fragments_threshold is not None:
            title += f"\tUnique fragments: {unique_fragments_threshold:.2f}\n"
        if tss_enrichment_threshold is not None:
            title += f"\tTSS enrichment: {tss_enrichment_threshold:.2f}\n"
        if frip_threshold is not None:
            title += f"\tFRIP: {frip_threshold:.2f}\n"
        if duplication_ratio_threshold is not None:
            title += f"\tDuplication rate: {duplication_ratio_threshold:.2f}\n"
    else:
        title = sample_id if sample_alias is None else sample_alias

    fig.suptitle(title)

    if save:
        fig.savefig(save)
    else:
        fig.show()

    return fig
