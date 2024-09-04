from __future__ import annotations

import logging
import random
import sys
from typing import TYPE_CHECKING

import harmonypy as hm
import igraph as ig
import leidenalg as la
import matplotlib.backends.backend_pdf
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import umap
from adjustText import adjust_text
from igraph import intersection
from pycisTopic.utils import subset_list
from scipy import sparse
from sklearn.neighbors import kneighbors_graph

if TYPE_CHECKING:
    from pycisTopic.cistopic_class import CistopicObject
    from pycisTopic.diff_features import CistopicImputedFeatures


def find_clusters(
    cistopic_obj: CistopicObject,
    target: str = "cell",
    k: int = 10,
    res: list[float] = [0.6],
    seed: int = 555,
    scale: bool = False,
    prefix: str = "",
    selected_topics: list[int] | None = None,
    selected_features: list[str] | None = None,
    harmony: bool = False,
    rna_components: pd.DataFrame | None = None,
    use_umap_integration: bool = False,
    rna_weight: float = 0.5,
    split_pattern: str = "___",
):
    """
    Performing leiden cell or region clustering and add results to cisTopic object's metadata.

    Parameters
    ----------
    cistopic_obj: `class::CistopicObject`
            A cisTopic object with a model in `class::CistopicObject.selected_model`.
    target: str
            Whether cells ('cell') or regions ('region') should be clustered. Default: 'cell'
    k: int
            Number of neighbours in the k-neighbours graph. Default: 10
    res: list[float]
            Resolution parameter for the leiden algorithm step. Default: [0.6]
    seed: int
            Seed parameter for the leiden algorithm step. Default: 555
    scale: bool
            Whether to scale the cell-topic or topic-regions contributions prior to the clustering. Default: False
    prefix: str
            Prefix to add to the clustering name when adding it to the correspondent metadata attribute. Default: ''
    selected_topics: list[int], optional
            A list with selected topics to be used for clustering. Default: None (use all topics)
    selected_features: list[str], optional
            A list with selected features (cells or regions) to cluster. This is recommended when working with regions (e.g. selecting
            regions in binarized topics), as working with all regions can be time consuming. Default: None (use all features)
    harmony: bool
            If target is 'cell', whether to use harmony processed topic contributions. Default: False.
    rna_components: pd.DataFrame, optional
            A pandas dataframe containing RNA dimensionality reduction (e.g. PCA) components. If provided, both layers (atac and rna)
            will be considered for clustering.
    use_umap_integration: bool
            Whether to use a weighted UMAP representation for the clustering or directly integrating the two graphs. Default: True
    rna_weight: float
            Weight of the RNA layer on the clustering (only applicable when clustering via UMAP). Default: 0.5 (same weight)
    split_pattern: str
            Pattern to split cell barcode from sample id. Default: '___'.

    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    log.info("Finding neighbours")
    model = cistopic_obj.selected_model

    if target == "cell":
        data_mat = model.cell_topic_harmony if harmony else model.cell_topic
        data_names = cistopic_obj.cell_names

    if target == "region":
        data_mat = model.topic_region.T
        data_names = cistopic_obj.region_names

    if selected_topics is not None:
        data_mat = data_mat.loc[["Topic" + str(x) for x in selected_topics]]
    if selected_features is not None:
        data_mat = data_mat[selected_features]
        data_names = selected_features

    if scale:
        data_mat = pd.DataFrame(
            sklearn.preprocessing.StandardScaler().fit_transform(data_mat),
            index=data_mat.index.to_list(),
            columns=data_mat.columns,
        )
    data_mat = data_mat.T

    if rna_components is not None:
        atac_topics, rna_components, data_names = input_check(data_mat, rna_components)
    if use_umap_integration:
        intersect, data_mat = weighted_integration(
            atac_topics, rna_components, data_names, rna_weight
        )

    if rna_components is None or use_umap_integration:
        A = kneighbors_graph(data_mat, k)
        sources, targets = A.nonzero()
        G = ig.Graph(directed=True)
        G.add_vertices(A.shape[0])
        edges = list(zip(sources, targets))
        G.add_edges(edges)
    elif rna_components is not None and not use_umap_integration:
        A = kneighbors_graph(atac_topics, k)
        sources, targets = A.nonzero()
        G1 = ig.Graph(directed=False)
        G1.add_vertices(A.shape[0])
        edges = list(zip(sources, targets))
        G1.add_edges(edges)
        A = kneighbors_graph(rna_components, k)
        sources, targets = A.nonzero()
        G2 = ig.Graph(directed=False)
        G2.add_vertices(A.shape[0])
        edges = list(zip(sources, targets))
        G2.add_edges(edges)
        G = intersection([G1, G2], keep_all_vertices=False)
        log.info("Finding clusters")
    for C in res:
        partition = la.find_partition(
            G, la.RBConfigurationVertexPartition, resolution_parameter=C, seed=seed
        )
        cluster = pd.DataFrame(
            partition.membership,
            index=data_names,
            columns=[prefix + "leiden_" + str(k) + "_" + str(C)],
        ).astype(str)
        if target == "cell":
            cistopic_obj.add_cell_data(cluster, split_pattern=split_pattern)
        if target == "region":
            cistopic_obj.add_region_data(cluster)


def run_umap(
    cistopic_obj: CistopicObject,
    target: str = "cell",
    scale: bool = False,
    reduction_name: str = "UMAP",
    random_state: int = 555,
    selected_topics: list[int] | None = None,
    selected_features: list[str] | None = None,
    harmony: bool = False,
    rna_components: pd.DataFrame | None = None,
    rna_weight: float = 0.5,
    **kwargs,
):
    """
    Run UMAP and add it to the dimensionality reduction dictionary.

    Parameters
    ----------
    cistopic_obj: `class::CistopicObject`
            A cisTopic object with a model in `class::CistopicObject.selected_model`.
    target: str
            Whether cells ('cell') or regions ('region') should be used. Default: 'cell'
    scale: bool
            Whether to scale the cell-topic or topic-regions contributions prior to the dimensionality reduction. Default: False
    reduction_name: str
            Reduction name to use as key in the dimensionality reduction dictionary. Default: 'UMAP'
    random_state: int
            Seed parameter for running UMAP. Default: 555
    selected_topics: list, optional
            A list with selected topics to be used for clustering. Default: None (use all topics)
    selected_features: list, optional
            A list with selected features (cells or regions) to cluster. This is recommended when working with regions (e.g. selecting
            regions in binarized topics), as working with all regions can be time consuming. Default: None (use all features)
    harmony: bool
            If target is 'cell', whether to use harmony processed topic contributions. Default: False.
    rna_components: pd.DataFrame, optional
            A pandas dataframe containing RNA dimensionality reduction (e.g. PCA) components. If provided, both layers (atac and rna)
            will be considered for clustering.
    rna_weight: float
            Weight of the RNA layer on the clustering (only applicable when clustering via UMAP). Default: 0.5 (same weight)
    **kwargs
            Parameters to pass to umap.UMAP.

    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    model = cistopic_obj.selected_model

    if target == "cell":
        data_mat = model.cell_topic_harmony if harmony else model.cell_topic
        data_names = cistopic_obj.cell_names

    if target == "region":
        data_mat = model.topic_region.T
        data_names = cistopic_obj.region_names

    if selected_topics is not None:
        data_mat = data_mat.loc[["Topic" + str(x) for x in selected_topics]]
    if selected_features is not None:
        data_mat = data_mat[selected_features]
        data_names = selected_features

    if scale:
        data_mat = pd.DataFrame(
            sklearn.preprocessing.StandardScaler().fit_transform(data_mat),
            index=data_mat.index.to_list(),
            columns=data_mat.columns,
        )

    data_mat = data_mat.T

    log.info("Running UMAP")
    if rna_components is None:
        reducer = umap.UMAP(random_state=random_state, **kwargs)
        embedding = reducer.fit_transform(data_mat)
    else:
        atac_topics, rna_components, data_names = input_check(data_mat, rna_components)
        intersect, embedding = weighted_integration(
            atac_topics, rna_components, data_names, rna_weight, **kwargs
        )
    dr = pd.DataFrame(embedding, index=data_names, columns=["UMAP_1", "UMAP_2"])
    if target == "cell":
        cistopic_obj.projections["cell"][reduction_name] = dr
    if target == "region":
        cistopic_obj.projections["region"][reduction_name] = dr


def run_tsne(
    cistopic_obj: CistopicObject,
    target: str = "cell",
    scale: bool = False,
    reduction_name: str = "tSNE",
    random_state: int = 555,
    perplexity: int = 30,
    selected_topics: list[int] | None = None,
    selected_features: list[str] | None = None,
    harmony: bool = False,
    rna_components: pd.DataFrame | None = None,
    rna_weight: float = 0.5,
    **kwargs,
):
    """
    Run tSNE and add it to the dimensionality reduction dictionary. If FItSNE is installed it will be used, otherwise sklearn TSNE implementation will be used.

    Parameters
    ----------
    cistopic_obj: `class::CistopicObject`
            A cisTopic object with a model in `class::CistopicObject.selected_model`.
    target: str
            Whether cells ('cell') or regions ('region') should be used. Default: 'cell'
    scale: bool
            Whether to scale the cell-topic or topic-regions contributions prior to the dimensionality reduction. Default: False
    reduction_name: str
            Reduction name to use as key in the dimensionality reduction dictionary. Default: 'tSNE'
    random_state: int
            Seed parameter for running tSNE. Default: 555
    perplexity: int
            Perplexity parameter for FitSNE. Default: 30
    selected_topics: list[int], optional
            A list with selected topics to be used for clustering. Default: None (use all topics)
    selected_features: list[str], optional
            A list with selected features (cells or regions) to cluster. This is recommended when working with regions (e.g. selecting
            regions in binarized topics), as working with all regions can be time consuming. Default: None (use all features)
    harmony: bool
            If target is 'cell', whether to use harmony processed topic contributions. Default: False
    rna_components: pd.DataFrame, optional
            A pandas dataframe containing RNA dimensionality reduction (e.g. PCA) components. If provided, both layers (atac and rna)
            will be considered for clustering.
    rna_weight: float
            Weight of the RNA layer on the clustering (only applicable when clustering via UMAP). Default: 0.5 (same weight)
    **kwargs
            Parameters to pass to fitsne.FItSNE or sklearn.manifold.TSNE.

    Reference
    ---------

    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    model = cistopic_obj.selected_model

    if target == "cell":
        data_mat = model.cell_topic_harmony if harmony else model.cell_topic
        data_names = cistopic_obj.cell_names

    if target == "region":
        data_mat = model.topic_region.T
        data_names = cistopic_obj.region_names

    if selected_topics is not None:
        data_mat = data_mat.loc[["Topic" + str(x) for x in selected_topics]]
    if selected_features is not None:
        data_mat = data_mat[selected_features]
        data_names = selected_features

    if scale:
        data_mat = pd.DataFrame(
            sklearn.preprocessing.StandardScaler().fit_transform(data_mat),
            index=data_mat.index.to_list(),
            columns=data_mat.columns,
        )

    data_mat = data_mat.T

    if rna_components is not None:
        atac_topics, rna_components, data_names = input_check(data_mat, rna_components)
        intersect, data_mat = weighted_integration(
            atac_topics, rna_components, data_names, rna_weight
        )

    try:
        import fitsne

        log.info("Running FItSNE")
        embedding = fitsne.FItSNE(
            np.ascontiguousarray(data_mat.to_numpy()),
            rand_seed=random_state,
            perplexity=perplexity,
            **kwargs,
        )
    except BaseException:
        log.info("Running TSNE")
        embedding = sklearn.manifold.TSNE(
            n_components=2, random_state=random_state
        ).fit_transform(data_mat.to_numpy(), **kwargs)
    dr = pd.DataFrame(embedding, index=data_names, columns=["tSNE_1", "tSNE_2"])

    if target == "cell":
        cistopic_obj.projections["cell"][reduction_name] = dr
    if target == "region":
        cistopic_obj.projections["region"][reduction_name] = dr


def plot_metadata(
    cistopic_obj: CistopicObject,
    reduction_name: str,
    variables: list[str],
    target: str = "cell",
    remove_nan: bool = True,
    show_label: bool = True,
    show_legend: bool = False,
    cmap: str | matplotlib.cm = cm.viridis,
    dot_size: int = 10,
    text_size: int = 10,
    alpha: float = 1.0,
    seed: int = 555,
    color_dictionary: dict[str, dict[str, str]] | None = None,
    figsize: tuple[float, float] = (6.4, 4.8),
    num_columns: int = 1,
    selected_features: list[str] | None = None,
    save: str | None = None,
):
    """
    Plot categorical and continuous metadata into dimensionality reduction.

    Parameters
    ----------
    cistopic_obj: `class::CistopicObject`
            A cisTopic object with dimensionality reductions in `class::CistopicObject.projections`.
    reduction_name: str
            Name of the dimensionality reduction to use
    variables: list[str]
            List of variables to plot. They should be included in `class::CistopicObject.cell_data` and `class::CistopicObject.region_data`, depending on which
            target is specified.
    target: str
            Whether cells ('cell') or regions ('region') should be used. Default: 'cell'
    remove_nan: bool
            Whether to remove data points for which the variable value is 'nan'. Default: True
    show_label: bool
            For categorical variables, whether to show the label in the plot. Default: True
    show_legend: bool
            For categorical variables, whether to show the legend next to the plot. Default: False
    cmap: str or 'matplotlib.cm'
            For continuous variables, color map to use for the legend color bar. Default: cm.viridis
    dot_size: int
            Dot size in the plot. Default: 10
    text_size: int
            For categorical variables and if show_label is True, size of the labels in the plot. Default: 10
    alpha: float
            Transparency value for the dots in the plot. Default: 1
    seed: int
            Random seed used to select random colors. Default: 555
    color_dictionary: dict, optional
            A dictionary containing an entry per variable, whose values are dictionaries with variable levels as keys and corresponding colors as values.
            Default: None
    figsize: tuple[float, float], optional
            Size of the figure. If num_columns is 1, this is the size for each figure; if num_columns is above 1, this is the overall size of the figure (if keeping
            default, it will be the size of each subplot in the figure). Default: (6.4, 4.8)
    num_columns: int
            For multiplot figures, indicates the number of columns (the number of rows will be automatically determined based on the number of plots). Default: 1
    selected_features: list,[str] optional
            A list with selected features (cells or regions) to plot. This is recommended when working with regions (e.g. selecting
            regions in binarized topics), as working with all regions can be time consuming. Default: None (use all features)
    save: str, optional
            Path to save plot. Default: None.

    """
    if target == "cell":
        data_mat = cistopic_obj.cell_data
    if target == "region":
        data_mat = cistopic_obj.region_data

    embedding = cistopic_obj.projections[target][reduction_name]

    if selected_features is not None:
        data_mat = data_mat.loc[selected_features]
        embedding = embedding.loc[selected_features]

    data_mat = data_mat.loc[embedding.index.to_list()]
    pdf = None
    if (save is not None) and (num_columns == 1):
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)

    if num_columns > 1:
        num_rows = int(np.ceil(len(variables) / num_columns))
        if figsize == (6.4, 4.8):
            figsize = (6.4 * num_columns, 4.8 * num_rows)
        i = 1

    fig = plt.figure(figsize=figsize)

    for var in variables:
        var_data = data_mat.copy().loc[:, var].dropna().to_list()
        if isinstance(var_data[0], str):
            if remove_nan and (data_mat[var].isnull().sum() > 0):
                var_data = data_mat.copy().loc[:, var].dropna().to_list()
                emb_nan = embedding.loc[
                    data_mat.copy().loc[:, var].dropna().index.tolist()
                ]
                label_pd = pd.concat(
                    [emb_nan, data_mat.loc[:, [var]].dropna()], axis=1, sort=False
                )
            else:
                var_data = (
                    data_mat.copy().astype(str).fillna("NA").loc[:, var].to_list()
                )
                label_pd = pd.concat(
                    [embedding, data_mat.astype(str).fillna("NA").loc[:, [var]]],
                    axis=1,
                    sort=False,
                )

            if color_dictionary is None:
                color_dictionary = {}
            categories = set(var_data)

            if var in color_dictionary:
                color_dict = color_dictionary[var]
            else:
                random.seed(seed)
                color = [
                    mcolors.to_rgb("#" + "%06x" % random.randint(0, 0xFFFFFF))
                    for i in range(len(categories))
                ]
                color_dict = dict(zip(categories, color))

            if num_columns > 1:
                plt.subplot(num_rows, num_columns, i)
                i = i + 1

            if remove_nan and (data_mat[var].isnull().sum() > 0):
                plt.scatter(
                    emb_nan.iloc[:, 0],
                    emb_nan.iloc[:, 1],
                    c=data_mat.loc[:, var].dropna().apply(lambda x: color_dict[x]),
                    s=dot_size,
                    alpha=alpha,
                )
                plt.xlabel(emb_nan.columns[0])
                plt.ylabel(emb_nan.columns[1])
            else:
                plt.scatter(
                    embedding.iloc[:, 0],
                    embedding.iloc[:, 1],
                    c=data_mat.astype(str)
                    .fillna("NA")
                    .loc[:, var]
                    .apply(lambda x: color_dict[x]),
                    s=dot_size,
                    alpha=alpha,
                )
                plt.xlabel(embedding.columns[0])
                plt.ylabel(embedding.columns[1])

            if show_label:
                label_pos = label_pd.groupby(var).agg(
                    {label_pd.columns[0]: np.mean, label_pd.columns[1]: np.mean}
                )
                texts = []
                for label in label_pos.index.tolist():
                    texts.append(
                        plt.text(
                            label_pos.loc[label][0],
                            label_pos.loc[label][1],
                            label,
                            horizontalalignment="center",
                            verticalalignment="center",
                            size=text_size,
                            weight="bold",
                            color=color_dict[label],
                            path_effects=[
                                PathEffects.withStroke(linewidth=3, foreground="w")
                            ],
                        )
                    )
                adjust_text(texts)

            plt.title(var)
            patchList = []
            for key in color_dict:
                data_key = mpatches.Patch(color=color_dict[key], label=key)
                patchList.append(data_key)
            if show_legend:
                plt.legend(
                    handles=patchList, bbox_to_anchor=(1.04, 1), loc="upper left"
                )

            if num_columns == 1:
                if save is not None:
                    pdf.savefig(fig, bbox_inches="tight")
                plt.show()
        else:
            var_data = data_mat.copy().loc[:, var].to_list()
            o = np.argsort(var_data)
            if num_columns > 1:
                plt.subplot(num_rows, num_columns, i)
                i = i + 1
            plt.scatter(
                embedding.iloc[o, 0],
                embedding.iloc[o, 1],
                c=subset_list(var_data, o),
                cmap=cmap,
                s=dot_size,
                alpha=alpha,
            )
            plt.xlabel(embedding.columns[0])
            plt.ylabel(embedding.columns[1])
            plt.title(var)
            # setup the colorbar
            normalize = mcolors.Normalize(
                vmin=np.array(var_data).min(), vmax=np.array(var_data).max()
            )
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
            scalarmappaple.set_array(var_data)
            plt.colorbar(scalarmappaple)
            if num_columns == 1:
                if save is not None:
                    pdf.savefig(fig, bbox_inches="tight")
                plt.show()

    if num_columns > 1:
        plt.tight_layout()
        if save is not None:
            fig.savefig(save, bbox_inches="tight")
        plt.show()
    if (save is not None) and (num_columns == 1):
        pdf = pdf.close()


def plot_topic(
    cistopic_obj: CistopicObject,
    reduction_name: str,
    target: str = "cell",
    cmap: str | matplotlib.cm = cm.viridis,
    dot_size: int = 10,
    alpha: float = 1.0,
    scale: bool = False,
    selected_topics: list[int] | None = None,
    selected_features: str | None = None,
    harmony: bool = False,
    figsize: tuple[float, float] = (6.4, 4.8),
    num_columns: int = 1,
    save: str | None = None,
):
    """
    Plot topic distributions into dimensionality reduction.

    Parameters
    ----------
    cistopic_obj: `class::CistopicObject`
            A cisTopic object with dimensionality reductions in `class::CistopicObject.projections`.
    reduction_name: str
            Name of the dimensionality reduction to use
    target: str
            Whether cells ('cell') or regions ('region') should be used. Default: 'cell'
    cmap: str or 'matplotlib.cm'
            For continuous variables, color map to use for the legend color bar. Default: cm.viridis
    dot_size: int
            Dot size in the plot. Default: 10
    alpha: float
            Transparency value for the dots in the plot. Default: 1
    scale: bool
            Whether to scale the cell-topic or topic-regions contributions prior to plotting. Default: False
    selected_topics: list[int], optional
            A list with selected topics to be used for plotting. Default: None (use all topics)
    selected_features: list[str], optional
            A list with selected features (cells or regions) to plot. This is recommended when working with regions (e.g. selecting
            regions in binarized topics), as working with all regions can be time consuming. Default: None (use all features)
    harmony: bool
            If target is 'cell', whether to use harmony processed topic contributions. Default: False
    figsize: tuple[float, float], optional
            Size of the figure. If num_columns is 1, this is the size for each figure; if num_columns is above 1, this is the overall size of the figure (if keeping
            default, it will be the size of each subplot in the figure). Default: (6.4, 4.8)
    num_columns: int
            For multiplot figures, indicates the number of columns (the number of rows will be automatically determined based on the number of plots). Default: 1
    save: str, optional
            Path to save plot. Default: None.

    """
    embedding = cistopic_obj.projections[target][reduction_name]
    model = cistopic_obj.selected_model

    if target == "cell":
        if harmony:
            data_mat = model.cell_topic_harmony
            prefix = "harmony_"
        else:
            data_mat = model.cell_topic
    elif target == "region":
        data_mat = model.topic_region.T

    if selected_features is not None:
        data_mat = data_mat.loc[selected_features]
        embedding = embedding.loc[selected_features]

    data_mat = data_mat.loc[:, embedding.index.to_list()]

    if selected_topics is not None:
        data_mat = data_mat.loc[["Topic" + str(x) for x in selected_topics],]

    if scale:
        data_mat = pd.DataFrame(
            sklearn.preprocessing.StandardScaler().fit_transform(data_mat),
            index=data_mat.index.to_list(),
            columns=data_mat.columns,
        )
    data_mat = data_mat.T

    if selected_topics is None:
        topic = data_mat.columns.to_list()
    else:
        topic = ["Topic" + str(t) for t in selected_topics]

    if (save is not None) and (num_columns == 1):
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)

    if num_columns > 1:
        num_rows = int(np.ceil(len(topic) / num_columns))
        if figsize == (6.4, 4.8):
            figsize = (6.4 * num_columns, 4.8 * num_rows)
        i = 1

    fig = plt.figure(figsize=figsize)

    for var in topic:
        var_data = data_mat.loc[:, var]
        var_data = var_data.sort_values()
        embedding_plot = embedding.loc[var_data.index.tolist()]
        o = np.argsort(var_data)
        if num_columns > 1:
            plt.subplot(num_rows, num_columns, i)
            i = i + 1
        if not scale:
            plt.scatter(
                embedding_plot.iloc[o, 0],
                embedding_plot.iloc[o, 1],
                c=subset_list(var_data, o),
                cmap=cmap,
                s=dot_size,
                alpha=alpha,
                vmin=0,
                vmax=max(var_data),
            )
            normalize = mcolors.Normalize(vmin=0, vmax=np.array(var_data).max())
        else:
            plt.scatter(
                embedding_plot.iloc[o, 0],
                embedding_plot.iloc[o, 1],
                c=subset_list(var_data, o),
                cmap=cmap,
                s=dot_size,
                alpha=alpha,
            )
            normalize = mcolors.Normalize(
                vmin=np.array(var_data).min(), vmax=np.array(var_data).max()
            )
        plt.xlabel(embedding_plot.columns[0])
        plt.ylabel(embedding_plot.columns[1])
        plt.title(var)
        # setup the colorbar
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        scalarmappaple.set_array(var_data)
        plt.colorbar(scalarmappaple)
        if num_columns == 1:
            if save is not None:
                pdf.savefig(fig, bbox_inches="tight")
            plt.show()

    if num_columns > 1:
        plt.tight_layout()
        if save is not None:
            fig.savefig(save, bbox_inches="tight")
        plt.show()

    if (save is not None) and (num_columns == 1):
        pdf.close()


def plot_imputed_features(
    cistopic_obj: CistopicObject,
    reduction_name: str,
    imputed_data: CistopicImputedFeatures,
    features: list[str],
    scale: bool = False,
    cmap: str | matplotlib.cm = cm.viridis,
    dot_size: int = 10,
    alpha: float = 1.0,
    selected_cells: list[str] | None = None,
    figsize: tuple[float, float] = (6.4, 4.8),
    num_columns: int = 1,
    save: str | None = None,
):
    """
    Plot imputed features into dimensionality reduction.

    Parameters
    ----------
    cistopic_obj: `class::CistopicObject`
            A cisTopic object with dimensionality reductions in `class::CistopicObject.dr`.
    reduction_name: str
            Name of the dimensionality reduction to use
    imputed_data: `class::cisTopicImputedFeatures`
            A `class::cisTopicImputedFeatures` object derived from the input cisTopic object.
    features: list[str]
            Names of the features to plot.
    scale: bool
            Whether to scale the imputed features prior to plotting. Default: False
    cmap: str or 'matplotlib.cm'
            For continuous variables, color map to use for the legend color bar. Default: cm.viridis
    dot_size: int
            Dot size in the plot. Default: 10
    alpha: float
            Transparency value for the dots in the plot. Default: 1
    selected_cells: list[str], optional
            A list with selected cells to plot. Default: None (use all cells)
    figsize: tuple[float, float], optional
            Size of the figure. If num_columns is 1, this is the size for each figure; if num_columns is above 1, this is the overall size of the figure (if keeping
            default, it will be the size of each subplot in the figure). Default: (6.4, 4.8)
    num_columns: int
            For multiplot figures, indicates the number of columns (the number of rows will be automatically determined based on the number of plots). Default: 1
    save: str, optional
            Path to save plot. Default: None.

    """
    pdf = None
    if (save is not None) and (num_columns == 1):
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)

    if num_columns > 1:
        num_rows = int(np.ceil(len(features) / num_columns))
        if figsize == (6.4, 4.8):
            figsize = (6.4 * num_columns, 4.8 * num_rows)
        i = 1

    fig = plt.figure(figsize=figsize)

    for feature in features:
        embedding = cistopic_obj.projections["cell"][reduction_name]
        if selected_cells is not None:
            embedding = embedding.loc[selected_cells]
        feature_data = imputed_data.subset(
            cells=embedding.index.tolist(), features=[feature], copy=True
        ).mtx
        if scale:
            try:
                feature_data = sklearn.preprocessing.scale(
                    feature_data.todense(), axis=1
                )
            except BaseException:
                feature_data = sklearn.preprocessing.scale(feature_data, axis=1)
        if isinstance(feature_data, sparse.csr_matrix):
            color_data = pd.DataFrame(
                feature_data.transpose().todense(), index=embedding.index.tolist()
            )
        else:
            color_data = pd.DataFrame(
                feature_data.transpose(), index=embedding.index.tolist()
            )
        color_data = color_data.sort_values(by=0)
        embedding = embedding.loc[color_data.index.tolist()]
        var_data = color_data.iloc[:, 0].to_list()
        o = np.argsort(var_data)
        if num_columns > 1:
            plt.subplot(num_rows, num_columns, i)
            i = i + 1
        plt.scatter(
            embedding.iloc[:, 0],
            embedding.iloc[:, 1],
            c=subset_list(var_data, o),
            s=dot_size,
            alpha=alpha,
        )
        plt.xlabel(embedding.columns[0])
        plt.ylabel(embedding.columns[1])
        plt.title(feature)
        # setup the colorbar
        normalize = mcolors.Normalize(
            vmin=np.array(color_data).min(), vmax=np.array(color_data).max()
        )
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        scalarmappaple.set_array(color_data)
        plt.colorbar(scalarmappaple)
        if num_columns == 1:
            if save is not None:
                pdf.savefig(fig, bbox_inches="tight")
            plt.show()

    if num_columns > 1:
        plt.tight_layout()
        if save is not None:
            fig.savefig(save, bbox_inches="tight")
        plt.show()

    if (save is not None) and (num_columns == 1):
        pdf = pdf.close()


def cell_topic_heatmap(
    cistopic_obj: CistopicObject,
    variables: list[str] | None = None,
    remove_nan: bool = True,
    scale: bool = False,
    cluster_topics: bool = False,
    color_dictionary: dict[str, dict[str, str]] | None = None,
    seed: int = 555,
    legend_loc_x: float = 1.2,
    legend_loc_y: float = -0.5,
    legend_dist_y: float = -1,
    figsize: tuple[float, float] = (6.4, 4.8),
    selected_topics: list[int] | None = None,
    selected_cells: list[str] | None = None,
    harmony: bool = False,
    save: str | None = None,
):
    """
    Plot heatmap with cell-topic distributions.

    Parameters
    ----------
    cistopic_obj: `class::CistopicObject`
            A cisTopic object with a model in `class::CistopicObject.selected_model`.
    variables: list[str]
            List of variables to plot. They should be included in `class::CistopicObject.cell_data` and `class::CistopicObject.region_data`, depending on which
            target is specified.
    remove_nan: bool
            Whether to remove data points for which the variable value is 'nan'. Default: True
    scale: bool
            Whether to scale the cell-topic or topic-regions contributions prior to plotting. Default: False
    cluster_topics: bool
            Whether to cluster rows in the heatmap. Otherwise, they will be ordered based on the maximum values over the ordered cells. Default: False
    color_dictionary: dict, optional
            A dictionary containing an entry per variable, whose values are dictionaries with variable levels as keys and corresponding colors as values.
            Default: None
    seed: int
            Random seed used to select random colors. Default: 555
    legend_loc_x: float
            X location for legend. Default: 1.2
    legend_loc_y: float
            Y location for legend. Default: -0.5
    legend_dist_y: float
            Y distance between legends. Default: -1
    figsize: tuple[float, float]
            Size of the figure. Default: (6.4, 4.8)
    selected_topics: list[int], optional
            A list with selected topics to be used for plotting. Default: None (use all topics)
    selected_cells: list[str], optional
            A list with selected cells to plot. Default: None (use all cells)
    harmony: bool
            If target is 'cell', whether to use harmony processed topic contributions. Default: False
    save: str, optional
            Path to save plot. Default: None.

    """
    model = cistopic_obj.selected_model
    cell_topic = model.cell_topic_harmony if harmony else model.cell_topic
    cell_data = cistopic_obj.cell_data

    if selected_topics is not None:
        cell_topic = cell_topic.loc[["Topic" + str(x) for x in selected_topics],]
    if selected_cells is not None:
        cell_topic = cell_topic.loc[:, selected_cells]
        cell_data = cell_data.loc[selected_cells]

    if scale:
        cell_topic = pd.DataFrame(
            sklearn.preprocessing.StandardScaler().fit_transform(cell_topic),
            index=cell_topic.index.to_list(),
            columns=cell_topic.columns,
        )

    if remove_nan and (sum(cell_data[variables].isnull().sum()) > 0):
        cell_data = cell_data[variables].dropna()
        cell_topic = cell_topic.loc[:, cell_data.index.tolist()]

    cell_topic = cell_topic.transpose()

    var = variables[0]
    var_data = cell_data.loc[:, var].sort_values()
    cell_topic = cell_topic.loc[var_data.index.to_list()]
    df = pd.concat([cell_topic, var_data], axis=1, sort=False)
    topic_order = df.groupby(var).mean().idxmax().sort_values().index.to_list()
    cell_topic = cell_topic.loc[:, topic_order].T
    # Check if color_dictionary exists
    if color_dictionary == None:
        for var in variables:
            c = colormaps['tab20']
            color_dictionary = {var: {str(k):c(k) for k in sorted(np.unique(cistopic_obj.cell_data[var]).astype(int))}}
    # Color dict
    col_colors = {}
    if variables is not None:
        for var in variables:
            var_data = cell_data.loc[:, var].sort_values()
            categories = set(var_data)

            if var not in color_dictionary:
                random.seed(seed)
                color = [
                    mcolors.to_rgb("#" + "%06x" % random.randint(0, 0xFFFFFF))
                    for i in range(len(categories))
                ]
                color_dict = dict(zip(categories, color))
                color_dictionary[var] = color_dict
            col_colors[var] = var_data.map(color_dictionary[var])
        col_colors = pd.concat(
            [col_colors[var] for var in variables], axis=1, sort=False
        )

        g = sns.clustermap(
            cell_topic,
            row_cluster=cluster_topics,
            col_cluster=False,
            col_colors=col_colors,
            cmap=cm.viridis,
            xticklabels=False,
            figsize=figsize,
        )

        cbar = g.cax
        cbar.set_position([legend_loc_x, 0.55, 0.05, 0.2])
        g.ax_col_dendrogram.set_visible(False)
        g.ax_row_dendrogram.set_visible(False)

        pos = legend_loc_y
        for key in color_dictionary:
            patchList = []
            for subkey in color_dictionary[key]:
                data_key = mpatches.Patch(
                    color=color_dictionary[key][subkey], label=subkey
                )
                patchList.append(data_key)
            legend = plt.legend(
                handles=patchList,
                bbox_to_anchor=(legend_loc_x, pos),
                loc="center",
                title=key,
            )
            ax = plt.gca().add_artist(legend)
            pos += legend_dist_y
    else:
        g = sns.clustermap(
            cell_topic,
            row_cluster=cluster_topics,
            col_cluster=True,
            cmap=cm.viridis,
            xticklabels=False,
            figsize=figsize,
        )

    if save is not None:
        g.savefig(save, bbox_inches="tight")
    plt.show()


def harmony(
    cistopic_obj: CistopicObject,
    vars_use: list[str],
    scale: bool = True,
    random_state: int = 555,
    **kwargs,
):
    """
    Apply harmony batch effect correction (Korsunsky et al, 2019) over cell-topic distribution.

    Parameters
    ----------
    cistopic_obj: `class::CistopicObject`
            A cisTopic object with a model in `class::CistopicObject.selected_model`.
    vars_use: list[str]
            List of variables to correct batch effect with.
    scale: bool
            Whether to scale probability matrix prior to correction. Default: True
    random_state: int
            Random seed used to use with harmony. Default: 555
    **kwargs
            Parameters to pass to harmonypy.run_harmony.

    References
    ----------
    Korsunsky, I., Millard, N., Fan, J., Slowikowski, K., Zhang, F., Wei, K., ... & Raychaudhuri, S. (2019). Fast, sensitive and accurate integration of
    single-cell data with Harmony. Nature methods, 16(12), 1289-1296.

    """
    cell_data = cistopic_obj.cell_data
    model = cistopic_obj.selected_model
    cell_topic = model.cell_topic
    if scale:
        cell_topic = pd.DataFrame(
            sklearn.preprocessing.StandardScaler().fit_transform(cell_topic),
            index=cell_topic.index.to_list(),
            columns=cell_topic.columns,
        )
    cell_topic = cell_topic.transpose().to_numpy()
    ho = hm.run_harmony(
        cell_topic, cell_data, vars_use, random_state=random_state, **kwargs
    )
    cell_topic_harmony = pd.DataFrame(
        ho.Z_corr,
        index=model.cell_topic.index.to_list(),
        columns=model.cell_topic.columns,
    )
    cistopic_obj.selected_model.cell_topic_harmony = cell_topic_harmony


# Helper functions for integration
def input_check(atac_topics: pd.DataFrame, rna_pca: pd.DataFrame):
    """
    A function to select cells present in both the RNA and the ATAC layers
    """
    # Cell names
    atac_cell_names = atac_topics.index.tolist()
    rna_cell_names = rna_pca.index.tolist()
    # Common cells
    common_cells = list(set(atac_cell_names).intersection(set(rna_cell_names)))
    atac_topics = atac_topics.loc[common_cells]
    rna_pca = rna_pca.loc[common_cells]
    return atac_topics, rna_pca, common_cells


def weighted_integration(
    atac_topics: pd.DataFrame,
    rna_pca: pd.DataFrame,
    common_cells: list[str],
    weight=0.5,
    **kwargs,
):
    """
    A function for weighted integration via UMAP
    """
    # Fit
    fit1 = umap.UMAP(random_state=123, **kwargs).fit(atac_topics)
    fit2 = umap.UMAP(random_state=123, **kwargs).fit(rna_pca)
    # Intersection
    intersection = umap.umap_.general_simplicial_set_intersection(
        fit1.graph_, fit2.graph_, weight=weight
    )
    # Embedding
    intersection = umap.umap_.reset_local_connectivity(intersection)
    weighted_comp = umap.umap_.simplicial_set_embedding(
        fit1._raw_data,
        intersection,
        fit1.n_components,
        fit1.learning_rate,
        fit1._a,
        fit1._b,
        fit1.repulsion_strength,
        fit1.negative_sample_rate,
        1000,
        "random",
        np.random.RandomState(123),
        fit1.metric,
        fit1._metric_kwds,
        False,
        {},
        False,
    )
    return intersection, weighted_comp[0]
