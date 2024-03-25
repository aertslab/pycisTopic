from __future__ import annotations

import json
import logging
import os
import sys
from collections import OrderedDict
from itertools import chain, islice, repeat
from multiprocessing import cpu_count
from operator import attrgetter
from typing import TYPE_CHECKING

import loompy as lp
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from ctxcore.genesig import Regulon
from loomxpy.loomxpy import SCopeLoom
from pyscenic.aucell import aucell
from pyscenic.binarization import binarize
from sklearn.feature_extraction.text import CountVectorizer

if TYPE_CHECKING:
    from pycisTopic.diff_features import CistopicImputedFeatures
    from pycisTopic.cistopic_class import CistopicObject


def export_gene_activity_to_loom(
    gene_activity_matrix: CistopicImputedFeatures | pd.DataFrame,
    cistopic_obj: CistopicObject,
    out_fname: str,
    regulons: list[Regulon] | None = None,
    selected_genes: list[str] | None = None,
    selected_cells: list[str] | None = None,
    auc_mtx: pd.DataFrame | None = None,
    auc_thresholds: pd.DataFrame | None = None,
    cluster_annotation: list[str] | None = None,
    cluster_markers: dict[str, dict[str, pd.DataFrame]] | None = None,
    tree_structure: tuple = (),
    title: str | None = None,
    nomenclature: str = "Unknown",
    split_pattern="___",
    num_workers: int = 1,
    **kwargs,
):
    """
    Create SCope [Davie et al, 2018] compatible loom files for gene activity exploration

    Parameters
    ---------
    gene_activity_matrix: class::CistopicImputedFeatures or class::pd.DataFrame
        A cisTopic imputed features object containing imputed gene activity as values. Alternatively, a pandas data frame with genes as
        columns, cells as rows and gene activity per gene as values.
    cistopic_obj: class::CisTopicObject
        The cisTopic object from which gene activity values have been derived. It must include cell meta data (including specified cluster
        annotation columns).
    regulons: list
        A list of regulons as derived from pySCENIC (Van de Sande et al., 2020).
    out_fname: str
        Path to output file.
    selected_genes: list, optional
        A list specifying which genes should be included in the loom file. Default: None
    selected_cells: list, optional
        A list specifying which cells should be included in the loom file. Default: None
    auc_mtx: pd.DataFrame, optional
        A regulon AUC matrix for the regulons as derived from pySCENIC (Van de Sande et al., 2020). If not provided it will be inferred.
    auc_thresholds: pd.DataFrame, optional
        A AUC thresholds for the regulons as derived from pySCENIC (Van de Sande et al., 2020). If not provided it will be inferred.
    cluster_annotation: list, optional
        A list indicating which information in `cistopic_obj.cell_data` should be used as clusters. The specified names must be included as columns
        in `cistopic_obj.cell_data`. Default: None.
    cluster_markers: dict, optional
        A dictionary including an entry per cluster annotation (which should match with the names in `cluster_annotation`) including a dictionary
        per cluster with a pandas data frame with marker regions as rows and logFC and adjusted p-values as columns (the output of
        `find_diff_features`). Default: None.
    tree_structure: sequence, optional
        A sequence of strings that defines the category tree structure. Needs to be a sequence of strings with three elements. Default: ()
    title: str, optional
        The title for this loom file. If None than the basename of the filename is used as the title. Default: None
    nomenclature: str, optional
        The name of the genome. Default: 'Unknown'
    **kwargs
        Additional parameters for pyscenic.export.export2loom


    References
    -----------
    Davie, K., Janssens, J., Koldere, D., De Waegeneer, M., Pech, U., Kreft, Ł., ... & Aerts, S. (2018). A single-cell transcriptome atlas of the
    aging Drosophila brain. Cell, 174(4), 982-998.

    Van de Sande, B., Flerin, C., Davie, K., De Waegeneer, M., Hulselmans, G., Aibar, S., ... & Aerts, S. (2020). A scalable SCENIC
    workflow for single-cell gene regulatory network analysis. Nature Protocols, 15(7), 2247-2276.
    """
    # Create logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    # Feature names
    if selected_genes is not None or selected_cells is not None:
        if not isinstance(gene_activity_matrix, pd.DataFrame):
            if selected_genes is not None:
                selected_genes = list(
                    set(selected_genes).intersection(accessibility_matrix.feature_names)
                )
            else:
                gene_activity_matrix = gene_activity_matrix.subset(
                    cells=selected_cells,
                    features=selected_genes,
                    copy=True,
                    split_pattern=split_pattern,
                )
        else:
            if selected_genes is not None:
                gene_activity_matrix = gene_activity_matrix.loc[:, selected_genes]
            if selected_cells is not None:
                gene_activity_matrix = gene_activity_matrix.loc[selected_cells]

    if not isinstance(gene_activity_matrix, pd.DataFrame):
        gene_names = gene_activity_matrix.feature_names
        cell_names = gene_activity_matrix.cell_names
    else:
        gene_names = gene_activity_matrix.columns.tolist()
        cell_names = gene_activity_matrix.index.tolist()

    # Extract cell data information and gene names
    cell_data = cistopic_obj.cell_data.loc[cell_names]

    # Check ups
    if cluster_annotation is not None:
        for annotation in cluster_annotation:
            if annotation not in cell_data:
                log.error(
                    "The cluster annotation",
                    annotation,
                    " is not included in cistopic_obj.cell_data",
                )

    # Prepare data for minimum loom file
    # Matrix
    if not isinstance(gene_activity_matrix, pd.DataFrame):
        if isinstance(gene_activity_matrix.mtx, sparse.csr_matrix):
            ex_mtx = pd.DataFrame.sparse.from_spmatrix(
                gene_activity_matrix.mtx.T, index=cell_names, columns=gene_names
            ).sparse.to_dense()
        else:
            ex_mtx = pd.DataFrame(
                gene_activity_matrix.mtx.T, index=cell_names, columns=gene_names
            )
    else:
        ex_mtx = gene_activity_matrix
    # Cell-topic values
    cell_topic = cistopic_obj.selected_model.cell_topic[cell_names].T
    ex_mtx = pd.concat([ex_mtx, cell_topic], axis=1)

    # Cell annotations and metrics
    metrics = []
    annotations = []
    for var in cell_data:
        if isinstance(cell_data[var][0], np.bool_):
            annotations.append(cell_data[var])
        else:
            try:
                metrics.append(cell_data[var].astype("float64"))
                cell_data[var] = cell_data[var].astype("float64")
            except BaseException:
                if len(set(cell_data[var])) < 255:
                    annotations.append(cell_data[var].astype("str"))
                    cell_data[var] = cell_data[var].astype("str")
    metrics = pd.concat(metrics, axis=1).fillna(0)
    annotations = pd.concat(annotations, axis=1)
    # Embeddings. Cell embeddings in this case
    embeddings = {}
    for x in cistopic_obj.projections["cell"].keys():
        emb_cell_names = list(
            set(cistopic_obj.projections["cell"][x].index.tolist()).intersection(
                set(cell_names)
            )
        )
        if len(emb_cell_names) == len(cell_names):
            emb_cell_names_mask = cistopic_obj.projections["cell"][x].index.isin(
                emb_cell_names
            )
            embeddings[x] = cistopic_obj.projections["cell"][x].loc[emb_cell_names_mask]

    # Create minimal loom
    log.info("Creating minimal loom")
    export_minimal_loom_gene(
        ex_mtx=ex_mtx,
        embeddings=embeddings,
        out_fname=out_fname,
        regulons=regulons,
        cell_annotations=None,
        tree_structure=tree_structure,
        title=title,
        nomenclature=nomenclature,
        auc_mtx=auc_mtx,
        auc_thresholds=auc_thresholds,
        num_workers=num_workers,
    )

    # Add annotations
    log.info("Adding annotations")
    path_to_loom = out_fname
    loom = SCopeLoom.read_loom(path_to_loom)
    if len(metrics):
        add_metrics(loom, metrics)
    if len(annotations):
        add_annotation(loom, annotations)

    # Add clusterings
    if cluster_annotation is not None:
        log.info("Adding clusterings")
        add_clusterings(loom, pd.DataFrame(cell_data[cluster_annotation]))
    # Add markers
    if cluster_markers is not None:
        log.info("Adding markers")
        annotation_in_markers = [
            x for x in cluster_annotation if x in cluster_markers.keys()
        ]
        annotation_not_in_markers = [
            x for x in cluster_annotation if x not in cluster_markers.keys()
        ]
        for x in annotation_not_in_markers:
            log.info(x, "is not in the cluster markers dictionary")
        cluster_markers = {
            k: v for k, v in cluster_markers.items() if k in annotation_in_markers
        }
        # Keep genes in data
        for y in cluster_markers:
            cluster_markers[y] = {
                x: cluster_markers[y][x][cluster_markers[y][x].index.isin(gene_names)]
                for x in cluster_markers[y].keys()
            }
        add_markers(loom, cluster_markers)

    log.info("Exporting")
    loom.export(out_fname)


def export_minimal_loom_gene(
    ex_mtx: pd.DataFrame,
    embeddings: dict[str, pd.DataFrame],
    out_fname: str,
    regulons: list[Regulon] | None = None,
    cell_annotations: dict[str, str] | None = None,
    tree_structure: tuple = (),
    title: str | None = None,
    nomenclature: str = "Unknown",
    num_workers: int = cpu_count(),
    auc_mtx=None,
    auc_thresholds=None,
    compress: bool = False,
):
    """
    Create a loom file for a single cell experiment to be used in SCope.
    :param ex_mtx: The expression matrix (n_cells x n_genes).
    :param regulons: A list of Regulons.
    :param cell_annotations: A dictionary that maps a cell ID to its corresponding cell type annotation.
    :param out_fname: The name of the file to create.
    :param tree_structure: A sequence of strings that defines the category tree structure. Needs to be a sequence of strings with three elements.
    :param title: The title for this loom file. If None than the basename of the filename is used as the title.
    :param nomenclature: The name of the genome.
    :param num_workers: The number of cores to use for AUCell regulon enrichment.
    :param embeddings: A dictionary that maps the name of an embedding to its representation as a pandas DataFrame with two columns: the first
    column is the first component of the projection for each cell followed by the second. The first mapping is the default embedding (use `collections.OrderedDict` to enforce this).
    :param compress: compress metadata (only when using SCope).
    """
    # Information on the general loom file format: http://linnarssonlab.org/loompy/format/index.html
    # Information on the SCope specific alterations: https://github.com/aertslab/SCope/wiki/Data-Format

    if cell_annotations is None:
        cell_annotations = dict(zip(ex_mtx.index, ["-"] * ex_mtx.shape[0]))

    # Calculate regulon enrichment per cell using AUCell.
    if auc_mtx is None:
        if regulons is not None:
            auc_mtx = aucell(ex_mtx, regulons, num_workers=num_workers)
            auc_mtx = auc_mtx.loc[ex_mtx.index]

    # Binarize matrix for AUC thresholds.
    if auc_thresholds is None:
        if auc_mtx is not None:
            _, auc_thresholds = binarize(auc_mtx, num_workers=num_workers)

    # Create an embedding based on tSNE.
    id2name = OrderedDict()
    embeddings_X = pd.DataFrame(index=ex_mtx.index)
    embeddings_Y = pd.DataFrame(index=ex_mtx.index)
    for idx, (name, df_embedding) in enumerate(embeddings.items()):
        if len(df_embedding.columns) != 2:
            raise Exception("The embedding should have two columns.")

        embedding_id = idx - 1  # Default embedding must have id == -1 for SCope.
        id2name[embedding_id] = name

        embedding = df_embedding.copy()
        embedding.columns = ["_X", "_Y"]
        embeddings_X = pd.merge(
            embeddings_X,
            embedding["_X"].to_frame().rename(columns={"_X": str(embedding_id)}),
            left_index=True,
            right_index=True,
        )
        embeddings_Y = pd.merge(
            embeddings_Y,
            embedding["_Y"].to_frame().rename(columns={"_Y": str(embedding_id)}),
            left_index=True,
            right_index=True,
        )
    embeddings_X = embeddings_X.loc[ex_mtx.index]
    embeddings_Y = embeddings_Y.loc[ex_mtx.index]
    # Encode genes in regulons as "binary" membership matrix.
    if regulons is not None:
        regulons_x = {
            regulons[x].name: " ".join(list(regulons[x].genes))
            for x in range(len(regulons))
        }
        cv = CountVectorizer(lowercase=False)
        regulon_assignment = cv.fit_transform(regulons_x.values())
        regulon_assignment = pd.DataFrame(
            regulon_assignment.todense(),
            columns=cv.get_feature_names(),
            index=regulons_x.keys(),
        )
        regulon_assignment = regulon_assignment.reindex(
            columns=ex_mtx.columns, fill_value=0
        ).T

    # Encode cell type clusters.
    # The name of the column should match the identifier of the clustering.
    name2idx = dict(map(reversed, enumerate(sorted(set(cell_annotations.values())))))
    clusterings = (
        pd.DataFrame(data=ex_mtx.index, index=ex_mtx.index, columns=["0"])
        .replace(cell_annotations)
        .replace(name2idx)
    )

    # Create meta-data structure.
    def create_structure_array(df):
        # Create a numpy structured array
        return np.array(
            [tuple(row) for row in df.values],
            dtype=np.dtype(list(zip(df.columns, df.dtypes))),
        )

    default_embedding = pd.DataFrame(
        [embeddings_X.iloc[:, 0], embeddings_Y.iloc[:, 0]],
        columns=ex_mtx.index,
        index=["_X", "_Y"],
    ).T
    if auc_mtx is None:
        column_attrs = {
            "CellID": ex_mtx.index.values.astype("str"),
            "Embedding": create_structure_array(default_embedding),
            "Clusterings": create_structure_array(clusterings),
            "ClusterID": clusterings.values,
            "Embeddings_X": create_structure_array(embeddings_X),
            "Embeddings_Y": create_structure_array(embeddings_Y),
        }
    else:
        column_attrs = {
            "CellID": ex_mtx.index.values.astype("str"),
            "Embedding": create_structure_array(default_embedding),
            "RegulonsAUC": create_structure_array(auc_mtx),
            "Clusterings": create_structure_array(clusterings),
            "ClusterID": clusterings.values,
            "Embeddings_X": create_structure_array(embeddings_X),
            "Embeddings_Y": create_structure_array(embeddings_Y),
        }
    if regulons is None:
        row_attrs = {
            "Gene": ex_mtx.columns.values.astype("str"),
        }
    else:
        row_attrs = {
            "Gene": ex_mtx.columns.values.astype("str"),
            "Regulons": create_structure_array(regulon_assignment),
        }

    def fetch_logo(context):
        for elem in context:
            if elem.endswith(".png"):
                return elem
        return ""

    if regulons is not None and auc_thresholds is not None:
        name2logo = {reg.name: fetch_logo(reg.context) for reg in regulons}
        regulon_thresholds = [
            {
                "regulon": name,
                "defaultThresholdValue": (
                    threshold if isinstance(threshold, float) else threshold[0]
                ),
                "defaultThresholdName": "gaussian_mixture_split",
                "allThresholds": {
                    "gaussian_mixture_split": (
                        threshold if isinstance(threshold, float) else threshold[0]
                    )
                },
                "motifData": name2logo.get(name, ""),
            }
            for name, threshold in auc_thresholds.iteritems()
        ]

        general_attrs = {
            "title": os.path.splitext(os.path.basename(out_fname))[0]
            if title is None
            else title,
            "MetaData": json.dumps(
                {
                    "embeddings": [
                        {"id": identifier, "name": name}
                        for identifier, name in id2name.items()
                    ],
                    "annotations": [{"name": "", "values": []}],
                    "clusterings": [
                        {
                            "id": 0,
                            "group": "celltype",
                            "name": "Cell Type",
                            "clusters": [
                                {"id": idx, "description": name}
                                for name, idx in name2idx.items()
                            ],
                        }
                    ],
                    "regulonThresholds": regulon_thresholds,
                }
            ),
            "Genome": nomenclature,
        }
    else:
        general_attrs = {
            "title": os.path.splitext(os.path.basename(out_fname))[0]
            if title is None
            else title,
            "MetaData": json.dumps(
                {
                    "embeddings": [
                        {"id": identifier, "name": name}
                        for identifier, name in id2name.items()
                    ],
                    "annotations": [{"name": "", "values": []}],
                    "clusterings": [
                        {
                            "id": 0,
                            "group": "celltype",
                            "name": "Cell Type",
                            "clusters": [
                                {"id": idx, "description": name}
                                for name, idx in name2idx.items()
                            ],
                        }
                    ],
                }
            ),
            "Genome": nomenclature,
        }

    # Add tree structure.
    # All three levels need to be supplied
    assert len(tree_structure) <= 3, ""
    general_attrs.update(
        ("SCopeTreeL{}".format(idx + 1), category)
        for idx, category in enumerate(
            list(islice(chain(tree_structure, repeat("")), 3))
        )
    )

    # Compress MetaData global attribute
    if compress:
        general_attrs["MetaData"] = compress_encode(value=general_attrs["MetaData"])

    # Create loom file for use with the SCope tool.
    # The loom file format opted for rows as genes to facilitate growth along the column axis (i.e add more cells)
    # PySCENIC chose a different orientation because of limitation set by the feather format: selectively reading
    # information from disk can only be achieved via column selection. For the ranking databases this is of utmost
    # importance.
    lp.create(
        filename=out_fname,
        layers=ex_mtx.T.values,
        row_attrs=row_attrs,
        col_attrs=column_attrs,
        file_attrs=general_attrs,
    )


def export_region_accessibility_to_loom(
    accessibility_matrix: CistopicImputedFeatures | pd.DataFrame,
    cistopic_obj: CistopicObject,
    binarized_topic_region: dict[str, pd.DataFrame],
    binarized_cell_topic: dict[str, pd.DataFrame],
    out_fname: str,
    selected_regions: list[str] | None = None,
    selected_cells: list[str] | None = None,
    cluster_annotation: list[str]  | None = None,
    cluster_markers: dict[str, dict[str, pd.DataFrame]] | None = None,
    tree_structure: tuple = (),
    title: str | None = None,
    nomenclature: str = "Unknown",
    split_pattern: str = "___",
    **kwargs,
):
    """
    Create SCope [Davie et al, 2018] compatible loom files for accessibility data exploration

    Parameters
    ---------
    accessibility_matrix: class::CistopicImputedFeatures or class::pd.DataFrame
        A cisTopic imputed features object containing imputed accessibility as values. Alternatively, a pandas data frame with regions as
        columns, cells as rows and accessibility per regions as values.
    cistopic_obj: class::CisTopicObject
        The cisTopic object from which accessibility values have been derived. It must include cell meta data (including specified cluster
        annotation columns) and the topic model from which accessibility has been imputed.
    binarized_topic_region: dictionary
        A dictionary containing topics as keys and class::pd.DataFrame with regions in topics as index and their topic contribution as values.
        This is the output of `binarize_topics()` using `target='region'`.
    binarized_cell_topic: dictionary
        A dictionary containing topics as keys and class::pd.DataFrame with cells in topics as index and their topic contribution as values.
        This is the output of `binarize_topics()` using `target='cell'`.
    out_fname: str
        Path to output file.
    selected_regions: list, optional
        A list specifying which regions should be included in the loom file. This is useful when working with very large data sets (e.g.
        one can select only regions in topics as DARs to reduce the file size). Default: None
    selected_cells: list, optional
        A list specifying which cells should be included in the loom file. Default: None
    cluster_annotation: list, optional
        A list indicating which information in `cistopic_obj.cell_data` should be used as clusters. The specified names must be included as columns
        in `cistopic_obj.cell_data`. Default: None.
    cluster_markers: dict, optional
        A dictionary including an entry per cluster annotation (which should match with the names in `cluster_annotation`) including a dictionary
        per cluster with a pandas data frame with marker regions as rows and logFC and adjusted p-values as columns (the output of
        `find_diff_features`). Default: None.
    tree_structure: sequence, optional
        A sequence of strings that defines the category tree structure. Needs to be a sequence of strings with three elements. Default: ()
    title: str, optional
        The title for this loom file. If None than the basename of the filename is used as the title. Default: None
    nomenclature: str, optional
        The name of the genome. Default: 'Unknown'
    **kwargs
        Additional parameters for pyscenic.export.export2loom


    References
    -----------
    Davie, K., Janssens, J., Koldere, D., De Waegeneer, M., Pech, U., Kreft, Ł., ... & Aerts, S. (2018). A single-cell transcriptome atlas of the
    aging Drosophila brain. Cell, 174(4), 982-998.
    """

    # Create logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    # Feature names
    if selected_regions is not None or selected_cells is not None:
        if not isinstance(accessibility_matrix, pd.DataFrame):
            if selected_regions is not None:
                selected_regions = list(
                    set(selected_regions).intersection(
                        accessibility_matrix.feature_names
                    )
                )
            else:
                accessibility_matrix = accessibility_matrix.subset(
                    cells=selected_cells,
                    features=selected_regions,
                    copy=True,
                    split_pattern=split_pattern,
                )
        else:
            if selected_regions is not None:
                selected_regions = list(
                    set(selected_regions).intersection(accessibility_matrix.columns)
                )
                accessibility_matrix = accessibility_matrix.loc[:, selected_regions]
            if selected_cells is not None:
                accessibility_matrix = accessibility_matrix.loc[selected_cells]

    # Prepare data for minimum loom file
    # Create input matrix
    cell_names = accessibility_matrix.cell_names
    cell_topic = cistopic_obj.selected_model.cell_topic.loc[:, cell_names]
    ex_mtx = sparse.vstack(
        [accessibility_matrix.mtx, sparse.csr_matrix(cell_topic.values)], format="csr"
    )
    feature_names = accessibility_matrix.feature_names + cell_topic.index.tolist()

    # Extract cell data information and region names
    cell_data = cistopic_obj.cell_data.loc[cell_names]

    # Check ups
    if not set(binarized_topic_region.keys()) == set(binarized_cell_topic.keys()):
        log.error(
            "Keys in binarized_topic_region and binarized_cell_topic do not agree."
        )
    if cluster_annotation is not None:
        for annotation in cluster_annotation:
            if annotation not in cell_data:
                log.error(
                    "The cluster annotation",
                    annotation,
                    " is not included in cistopic_obj.cell_data",
                )

    # Format regulons
    binarized_topic_region = {
        x: binarized_topic_region[x][
            binarized_topic_region[x].index.isin(feature_names)
        ]
        for x in binarized_topic_region.keys()
    }
    regulon_mat = cistopic_obj.selected_model.topic_region.copy()
    for col_idx in regulon_mat:
        thr = regulon_mat.loc[
            binarized_topic_region[col_idx][-1:].index, col_idx
        ].values[0]
        regulon_mat.loc[:, col_idx] = np.where(
            regulon_mat.loc[:, col_idx].values > thr, 1, 0
        )
    regulon_mat = regulon_mat.loc[accessibility_matrix.feature_names, :]
    extra = pd.DataFrame(0, index=regulon_mat.columns, columns=regulon_mat.columns)
    regulon_mat = pd.concat([regulon_mat, extra], axis=0)
    # Cell annotations and metrics
    metrics = []
    annotations = []
    for var in cell_data:
        if isinstance(cell_data[var][0], np.bool_):
            annotations.append(cell_data[var])
        else:
            try:
                metrics.append(cell_data[var].astype("float64"))
                cell_data[var] = cell_data[var].astype("float64")
            except BaseException:
                annotations.append(cell_data[var].astype("str"))
                cell_data[var] = cell_data[var].astype("str")
    metrics = pd.concat(metrics, axis=1).fillna(0)
    annotations = pd.concat(annotations, axis=1)
    # Auc thresholds
    # Keep only cells in data
    binarized_cell_topic = {
        x: binarized_cell_topic[x][binarized_cell_topic[x].index.isin(cell_names)]
        for x in binarized_cell_topic.keys()
    }
    cell_topic = cell_topic.T
    topic_thresholds = pd.Series(
        [
            cell_topic.sort_values(x, ascending=False)[x][len(binarized_cell_topic[x])]
            for x in binarized_cell_topic.keys()
        ],
        index=binarized_cell_topic.keys(),
    )
    # Embeddings. Cell embeddings in this case
    embeddings = {}
    for x in cistopic_obj.projections["cell"].keys():
        emb_cell_names = list(
            set(cistopic_obj.projections["cell"][x].index.tolist()).intersection(
                set(cell_names)
            )
        )
        if len(emb_cell_names) == len(cell_names):
            emb_cell_names_mask = cistopic_obj.projections["cell"][x].index.isin(
                emb_cell_names
            )
            embeddings[x] = cistopic_obj.projections["cell"][x].loc[emb_cell_names_mask]

    # Create minimal loom
    log.info("Creating minimal loom")
    export_minimal_loom_region(
        ex_mtx=ex_mtx,
        cell_names=cell_names,
        feature_names=feature_names,
        out_fname=out_fname,
        regulons=regulon_mat,
        cell_annotations=None,
        tree_structure=tree_structure,
        title=title,
        nomenclature=nomenclature,
        embeddings=embeddings,
        auc_mtx=cell_topic,
        auc_thresholds=topic_thresholds,
    )

    # Add annotations
    log.info("Adding annotations")
    path_to_loom = out_fname
    loom = SCopeLoom.read_loom(path_to_loom)
    if len(metrics):
        add_metrics(loom, metrics)
    if len(annotations):
        add_annotation(loom, annotations)

    # Add clusterings
    if cluster_annotation is not None:
        log.info("Adding clusterings")
        add_clusterings(loom, pd.DataFrame(cell_data[cluster_annotation]))

    # Add markers
    if cluster_markers is not None:
        log.info("Adding markers")
        annotation_in_markers = [
            x for x in cluster_annotation if x in cluster_markers.keys()
        ]
        annotation_not_in_markers = [
            x for x in cluster_annotation if x not in cluster_markers.keys()
        ]

        for x in annotation_not_in_markers:
            log.info(x, "is not in the cluster markers dictionary")
        cluster_markers = {
            k: v for k, v in cluster_markers.items() if k in annotation_in_markers
        }
        # Keeep regions in data
        for y in cluster_markers:
            cluster_markers[y] = {
                x: cluster_markers[y][x][
                    cluster_markers[y][x].index.isin(feature_names)
                ]
                for x in cluster_markers[y].keys()
            }
        add_markers(loom, cluster_markers)

    log.info("Exporting")
    loom.export(out_fname)


def export_minimal_loom_region(
    ex_mtx: sparse.csr_matrix,
    cell_names: list[str],
    feature_names: list[str],
    out_fname: str,
    regulons: pd.DataFrame | None = None,
    cell_annotations: dict[str, str] | None = None,
    tree_structure: tuple = (),
    title: str | None = None,
    nomenclature: str = "Unknown",
    num_workers: int = cpu_count(),
    embeddings: dict[str, pd.DataFrame] = {},
    auc_mtx=None,
    auc_thresholds=None,
    compress: bool = False,
):

    # Information on the general loom file format: http://linnarssonlab.org/loompy/format/index.html
    # Information on the SCope specific alterations: https://github.com/aertslab/SCope/wiki/Data-Format

    if cell_annotations is None:
        cell_annotations = dict(zip(cell_names, ["-"] * ex_mtx.shape[0]))

    # Create an embedding based on tSNE.
    # Name of columns should be "_X" and "_Y".
    if len(embeddings) == 0:
        embeddings = {
            "tSNE (default)": pd.DataFrame(
                data=TSNE().fit_transform(auc_mtx),
                index=cell_names,
                columns=["_X", "_Y"],
            )
        }  # (n_cells, 2)

    id2name = OrderedDict()
    embeddings_X = pd.DataFrame(index=cell_names)
    embeddings_Y = pd.DataFrame(index=cell_names)
    for idx, (name, df_embedding) in enumerate(embeddings.items()):
        if len(df_embedding.columns) != 2:
            raise Exception("The embedding should have two columns.")

        embedding_id = idx - 1  # Default embedding must have id == -1 for SCope.
        id2name[embedding_id] = name

        embedding = df_embedding.copy()
        embedding.columns = ["_X", "_Y"]
        embeddings_X = pd.merge(
            embeddings_X,
            embedding["_X"].to_frame().rename(columns={"_X": str(embedding_id)}),
            left_index=True,
            right_index=True,
        )
        embeddings_Y = pd.merge(
            embeddings_Y,
            embedding["_Y"].to_frame().rename(columns={"_Y": str(embedding_id)}),
            left_index=True,
            right_index=True,
        )

    # Encode cell type clusters.
    # The name of the column should match the identifier of the clustering.
    name2idx = dict(map(reversed, enumerate(sorted(set(cell_annotations.values())))))
    clusterings = (
        pd.DataFrame(data=cell_names, index=cell_names, columns=["0"])
        .replace(cell_annotations)
        .replace(name2idx)
    )

    # Create meta-data structure.
    def create_structure_array(df):
        # Create a numpy structured array
        return np.array(
            [tuple(row) for row in df.values],
            dtype=np.dtype(list(zip(df.columns, df.dtypes))),
        )

    default_embedding = next(iter(embeddings.values())).copy()
    default_embedding.columns = ["_X", "_Y"]
    column_attrs = {
        "CellID": np.array(cell_names),
        "Embedding": create_structure_array(default_embedding),
        "RegulonsAUC": create_structure_array(auc_mtx),
        "Clusterings": create_structure_array(clusterings),
        "ClusterID": clusterings.values,
        "Embeddings_X": create_structure_array(embeddings_X),
        "Embeddings_Y": create_structure_array(embeddings_Y),
    }
    row_attrs = {
        "Gene": np.array(feature_names),
        "Regulons": create_structure_array(regulons),
    }

    def fetch_logo(context):
        for elem in context:
            if elem.endswith(".png"):
                return elem
        return ""

    regulon_thresholds = [
        {
            "regulon": name,
            "defaultThresholdValue": (
                threshold if isinstance(threshold, float) else threshold[0]
            ),
            "defaultThresholdName": "gaussian_mixture_split",
            "allThresholds": {
                "gaussian_mixture_split": (
                    threshold if isinstance(threshold, float) else threshold[0]
                )
            },
            "motifData": "",
        }
        for name, threshold in auc_thresholds.iteritems()
    ]

    general_attrs = {
        "title": os.path.splitext(os.path.basename(out_fname))[0]
        if title is None
        else title,
        "MetaData": json.dumps(
            {
                "embeddings": [
                    {"id": identifier, "name": name}
                    for identifier, name in id2name.items()
                ],
                "annotations": [{"name": "", "values": []}],
                "clusterings": [
                    {
                        "id": 0,
                        "group": "celltype",
                        "name": "Cell Type",
                        "clusters": [
                            {"id": idx, "description": name}
                            for name, idx in name2idx.items()
                        ],
                    }
                ],
                "regulonThresholds": regulon_thresholds,
            }
        ),
        "Genome": nomenclature,
    }

    # Add tree structure.
    # All three levels need to be supplied
    assert len(tree_structure) <= 3, ""
    general_attrs.update(
        ("SCopeTreeL{}".format(idx + 1), category)
        for idx, category in enumerate(
            list(islice(chain(tree_structure, repeat("")), 3))
        )
    )

    # Compress MetaData global attribute
    if compress:
        general_attrs["MetaData"] = compress_encode(value=general_attrs["MetaData"])

    # Create loom file for use with the SCope tool.
    lp.create(
        filename=out_fname,
        layers=ex_mtx,
        row_attrs=row_attrs,
        col_attrs=column_attrs,
        file_attrs=general_attrs,
    )


def get_metadata(loom):
    """
    A helper function to get metadata
    """
    annot_metadata = loom.get_meta_data()["annotations"]
    annot_mt_column_names = [
        annot_metadata[x]["name"] for x in range(len(annot_metadata))
    ]
    annot_mt = pd.concat(
        [
            pd.DataFrame(loom.col_attrs[annot_mt_column_names[x]])
            for x in range(len(annot_mt_column_names))
        ],
        axis=1,
    )
    annot_mt.columns = [
        annot_mt_column_names[x] for x in range(len(annot_mt_column_names))
    ]
    annot_mt.index = loom.get_cell_ids().tolist()
    return annot_mt


def add_metrics(loom, metrics: pd.DataFrame):
    """
    A helper function to add metrics
    """
    md_metrics = []
    for metric in metrics:
        md_metrics.append({"name": metric})
        loom.col_attrs[metric] = np.array(metrics[metric])
    loom.global_attrs["MetaData"].update({"metrics": md_metrics})


def add_annotation(loom, annots: pd.DataFrame):
    """
    A helper function to add annotations
    """
    md_annot = []
    for annot in annots:
        vals = list(annots[annot])
        uniq_vals = np.unique(vals)
        md_annot.append(
            {"name": annot, "values": list(map(lambda x: str(x), uniq_vals.tolist()))}
        )
        loom.col_attrs[annot] = np.array(annots[annot])
    loom.global_attrs["MetaData"].update({"annotations": md_annot})


def add_clusterings(loom: SCopeLoom, cluster_data: pd.DataFrame):
    """
    A helper function to add clusters
    """
    col_attrs = loom.col_attrs

    attrs_metadata = {}
    attrs_metadata["clusterings"] = []
    clusterings = pd.DataFrame(index=cluster_data.index.tolist())
    j = 0

    for cluster_name in cluster_data.columns:
        clustering_id = j
        clustering_algorithm = cluster_name

        clustering_resolution = cluster_name
        cluster_marker_method = "Wilcoxon"

        num_clusters = len(np.unique(cluster_data[cluster_name]))
        cluster_2_number = {
            np.unique(cluster_data[cluster_name])[i]: i for i in range(num_clusters)
        }

        # Data
        clusterings[str(clustering_id)] = [
            cluster_2_number[x] for x in cluster_data[cluster_name].tolist()
        ]

        # Metadata
        attrs_metadata["clusterings"] = attrs_metadata["clusterings"] + [
            {
                "id": clustering_id,
                "group": clustering_algorithm,
                "name": clustering_algorithm,
                "clusters": [],
                "clusterMarkerMetrics": [
                    {
                        "accessor": "avg_logFC",
                        "name": "Avg. logFC",
                        "description": f"Average log fold change from {cluster_marker_method.capitalize()} test",
                    },
                    {
                        "accessor": "pval",
                        "name": "Adjusted P-Value",
                        "description": f"Adjusted P-Value from {cluster_marker_method.capitalize()} test",
                    },
                ],
            }
        ]

        for i in range(0, num_clusters):
            cluster = {}
            cluster["id"] = i
            cluster["description"] = np.unique(cluster_data[cluster_name])[i]
            attrs_metadata["clusterings"][j]["clusters"].append(cluster)

        j += 1

    # Update column attribute Dict
    col_attrs_clusterings = {
        # Pick the first one as default clustering (this is purely
        # arbitrary)
        "ClusterID": clusterings["0"].values,
        "Clusterings": df_to_named_matrix(clusterings),
    }

    col_attrs = {**col_attrs, **col_attrs_clusterings}
    loom.col_attrs = col_attrs
    loom.global_attrs["MetaData"].update({"clusterings": attrs_metadata["clusterings"]})


def add_markers(loom: SCopeLoom, markers_dict: dict[str, dict[str, pd.DataFrame]]):
    """
    A helper function to add markers to clusterings
    """
    attrs_metadata = loom.global_attrs["MetaData"]
    row_attrs = loom.row_attrs
    for cluster_name in markers_dict:
        idx = [
            i
            for i in range(len(attrs_metadata["clusterings"]))
            if attrs_metadata["clusterings"][i]["name"] == cluster_name
        ][0]
        clustering_id = attrs_metadata["clusterings"][idx]["id"]
        num_clusters = len(attrs_metadata["clusterings"][idx]["clusters"])
        cluster_description = [
            attrs_metadata["clusterings"][idx]["clusters"][x]["description"]
            for x in range(num_clusters)
        ]

        # Initialize
        cluster_markers = pd.DataFrame(
            index=loom.get_genes(), columns=[str(x) for x in np.arange(num_clusters)]
        ).fillna(0, inplace=False)
        cluster_markers_avg_logfc = pd.DataFrame(
            index=loom.get_genes(), columns=[str(x) for x in np.arange(num_clusters)]
        ).fillna(0, inplace=False)
        cluster_markers_pval = pd.DataFrame(
            index=loom.get_genes(), columns=[str(x) for x in np.arange(num_clusters)]
        ).fillna(0, inplace=False)

        # Populate
        for i in range(0, num_clusters):
            try:
                gene_names = markers_dict[cluster_name][
                    cluster_description[i]
                ].index.tolist()
                pvals_adj = markers_dict[cluster_name][cluster_description[i]][
                    "Adjusted_pval"
                ]
                logfoldchanges = markers_dict[cluster_name][cluster_description[i]][
                    "Log2FC"
                ]
                i = str(i)
                num_genes = len(gene_names)

                # Replace
                cluster_markers.loc[gene_names, i] = 1
                cluster_markers_avg_logfc.loc[gene_names, i] = logfoldchanges
                cluster_markers_pval.loc[gene_names, i] = pvals_adj
            except BaseException:
                print("No markers for ", cluster_description[i])

        # Update row attribute Dict
        row_attrs_cluster_markers = {
            f"ClusterMarkers_{str(idx)}": df_to_named_matrix(
                cluster_markers.astype(np.int8)
            ),
            f"ClusterMarkers_{str(idx)}_avg_logFC": df_to_named_matrix(
                cluster_markers_avg_logfc.astype(np.float32)
            ),
            f"ClusterMarkers_{str(idx)}_pval": df_to_named_matrix(
                cluster_markers_pval.astype(np.float32)
            ),
        }
        row_attrs = {**row_attrs, **row_attrs_cluster_markers}
        loom.row_attrs = row_attrs


def get_regulons(loom):
    """
    A helper function to get regulons
    """
    regulon_dict = pd.DataFrame(
        loom.get_regulons(), index=loom.row_attrs["Gene"]
    ).to_dict()
    for t in regulon_dict:
        regulon_dict[t] = {x: y for x, y in regulon_dict[t].items() if y != 0}
    motif_data = {
        x["regulon"]: x["motifData"]
        for x in loom.global_attrs["MetaData"]["regulonThresholds"]
    }
    regulon_list = [
        Regulon(
            name=x,
            gene2weight=regulon_dict[x],
            transcription_factor=x.split("_")[0],
            gene2occurrence=[],
            context=frozenset(list(motif_data[x])),
        )
        for x in regulon_dict.keys()
    ]
    return regulon_list


def df_to_named_matrix(df: pd.DataFrame):
    """
    A helper function to create metadata structure.
    """
    return np.array(
        [tuple(row) for row in df.values],
        dtype=np.dtype(list(zip(df.columns, df.dtypes))),
    )
