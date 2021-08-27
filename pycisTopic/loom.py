import json
import logging
import loompy
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import sys
from loomxpy.loomxpy import SCopeLoom
from pyscenic import export
from pyscenic.genesig import Regulon
from typing import Dict, List, Mapping, Optional, Sequence, Union


def export_gene_activity_to_loom(gene_activity_matrix: Union['CistopicImputedFeatures', pd.DataFrame],
                                 cistopic_obj: 'CistopicObject',
                                 regulons: List[Regulon],
                                 out_fname: str,
                                 selected_genes: Optional[List[str]] = None,
                                 selected_cells: Optional[List[str]] = None,
                                 auc_mtx: Optional[pd.DataFrame] = None,
                                 auc_thresholds: Optional[pd.DataFrame] = None,
                                 cluster_annotation: List[str] = None,
                                 cluster_markers: Dict[str, Dict[str, pd.DataFrame]] = None,
                                 tree_structure: Sequence[str] = (),
                                 title: str = None,
                                 nomenclature: str = "Unknown",
                                 **kwargs):
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
    log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    # Feature names
    if selected_genes is not None or selected_cells is not None:
        if not isinstance(gene_activity_matrix, pd.DataFrame):
            gene_activity_matrix = gene_activity_matrix.subset(
                cells=selected_cells, features=selected_genes, copy=True)
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
                    'The cluster annotation',
                    annotation,
                    ' is not included in cistopic_obj.cell_data')

    # Prepare data for minimum loom file
    # Matrix
    if not isinstance(gene_activity_matrix, pd.DataFrame):
        if isinstance(gene_activity_matrix.mtx, sparse.csr_matrix):
            ex_mtx = pd.DataFrame.sparse.from_spmatrix(
                gene_activity_matrix.mtx.T,
                index=cell_names,
                columns=gene_names).sparse.to_dense()
        else:
            ex_mtx = pd.DataFrame(
                gene_activity_matrix.mtx.T,
                index=cell_names,
                columns=gene_names)
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
                metrics.append(cell_data[var].astype('float64'))
                cell_data[var] = cell_data[var].astype('float64')
            except BaseException:
                annotations.append(cell_data[var].astype('str'))
                cell_data[var] = cell_data[var].astype('str')
    metrics = pd.concat(metrics, axis=1).fillna(0)
    annotations = pd.concat(annotations, axis=1)
    # Embeddings. Cell embeddings in this case
    embeddings = {x: cistopic_obj.projections['cell'][x].loc[list(set(cistopic_obj.projections['cell'][x].index.tolist()).intersection(set(cell_names)))]
                  for x in cistopic_obj.projections['cell'].keys()}

    # Create minimal loom
    log.info('Creating minimal loom')
    export.export2loom(ex_mtx=ex_mtx,
                       regulons=regulons,
                       out_fname=out_fname,
                       cell_annotations=None,
                       tree_structure=tree_structure,
                       title=title,
                       nomenclature=nomenclature,
                       embeddings=embeddings,
                       auc_mtx=auc_mtx,
                       auc_thresholds=auc_thresholds)

    # Add annotations
    log.info('Adding annotations')
    path_to_loom = out_fname
    loom = SCopeLoom.read_loom(path_to_loom)
    if len(metrics):
        add_metrics(loom, metrics)
    if len(annotations):
        add_annotation(loom, annotations)

    # Add clusterings
    if cluster_annotation is not None:
        log.info('Adding clusterings')
        add_clusterings(loom, pd.DataFrame(cell_data[cluster_annotation]))
    # Add markers
    if cluster_markers is not None:
        log.info('Adding markers')
        annotation_in_markers = [
            x for x in cluster_annotation if x in cluster_markers.keys()]
        annotation_not_in_markers = [
            x for x in cluster_annotation if x not in cluster_markers.keys()]
        for x in annotation_not_in_markers:
            log.info(x, 'is not in the cluster markers dictionary')
        cluster_markers = {
            k: v
            for k, v in cluster_markers.items()
            if k in annotation_in_markers
        }
        # Keep genes in data
        for y in cluster_markers:
            cluster_markers[y] = {
                x: cluster_markers[y][x][cluster_markers[y][x].index.isin(gene_names)]
                for x in cluster_markers[y].keys()
            }
        add_markers(loom, cluster_markers)

    log.info('Exporting')
    loom.export(out_fname)


def export_region_accessibility_to_loom(accessibility_matrix: Union['CistopicImputedFeatures', pd.DataFrame],
                                        cistopic_obj: 'CistopicObject',
                                        binarized_topic_region: Dict[str, pd.DataFrame],
                                        binarized_cell_topic: Dict[str, pd.DataFrame],
                                        out_fname: str,
                                        selected_regions: List[str] = None,
                                        selected_cells: List[str] = None,
                                        cluster_annotation: List[str] = None,
                                        cluster_markers: Dict[str, Dict[str, pd.DataFrame]] = None,
                                        tree_structure: Sequence[str] = (),
                                        title: str = None,
                                        nomenclature: str = "Unknown",
                                        **kwargs):
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
    log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    # Feature names
    if selected_regions is not None or selected_cells is not None:
        if not isinstance(accessibility_matrix, pd.DataFrame):
            accessibility_matrix = accessibility_matrix.subset(
                cells=selected_cells, features=selected_regions, copy=True)
        else:
            if selected_regions is not None:
                accessibility_matrix = accessibility_matrix.loc[:, selected_regions]
            if selected_cells is not None:
                accessibility_matrix = accessibility_matrix.loc[selected_cells]

    if not isinstance(accessibility_matrix, pd.DataFrame):
        region_names = accessibility_matrix.feature_names
        cell_names = accessibility_matrix.cell_names
    else:
        region_names = accessibility_matrix.columns.tolist()
        cell_names = accessibility_matrix.index.tolist()

    # Extract cell data information and region names
    cell_data = cistopic_obj.cell_data.loc[cell_names]

    # Check ups
    if not set(binarized_topic_region.keys()) == set(binarized_cell_topic.keys()):
        log.error('Keys in binarized_topic_region and binarized_cell_topic do not agree.')
    if cluster_annotation is not None:
        for annotation in cluster_annotation:
            if annotation not in cell_data:
                log.error(
                    'The cluster annotation',
                    annotation,
                    ' is not included in cistopic_obj.cell_data'
                )

    # Prepare data for minimum loom file
    # Matrix
    if not isinstance(accessibility_matrix, pd.DataFrame):
        if isinstance(accessibility_matrix.mtx, sparse.csr_matrix):
            ex_mtx = pd.DataFrame.sparse.from_spmatrix(
                accessibility_matrix.mtx.T,
                index=cell_names,
                columns=region_names
            ).sparse.to_dense()
        else:
            ex_mtx = pd.DataFrame(
                accessibility_matrix.mtx.T,
                index=cell_names,
                columns=region_names
            )
    else:
        ex_mtx = accessibility_matrix
    # Cell-topic values
    cell_topic = cistopic_obj.selected_model.cell_topic[cell_names].T
    ex_mtx = pd.concat([ex_mtx, cell_topic], axis=1)
    # Topics
    # Keep only regions in data
    binarized_topic_region = {
        x: binarized_topic_region[x][binarized_topic_region[x].index.isin(region_names)]
        for x in binarized_topic_region.keys()
    }
    topics = [
        Regulon(
            name=x,
            gene2weight=binarized_topic_region[x].to_dict()[x],
            transcription_factor=x,
            gene2occurrence=[]
        )
        for x in binarized_topic_region.keys()
    ]
    # Cell annotations and metrics
    metrics = []
    annotations = []
    for var in cell_data:
        if isinstance(cell_data[var][0], np.bool_):
            annotations.append(cell_data[var])
        else:
            try:
                metrics.append(cell_data[var].astype('float64'))
                cell_data[var] = cell_data[var].astype('float64')
            except BaseException:
                annotations.append(cell_data[var].astype('str'))
                cell_data[var] = cell_data[var].astype('str')
    metrics = pd.concat(metrics, axis=1).fillna(0)
    annotations = pd.concat(annotations, axis=1)
    # Auc thresholds
    # Keep only cells in data
    binarized_cell_topic = {
        x: binarized_cell_topic[x][binarized_cell_topic[x].index.isin(cell_names)]
        for x in cell_topic.keys()
    }
    topic_thresholds = pd.Series(
        [cell_topic.sort_values(x, ascending=False)[x][len(binarized_cell_topic[x])]
         for x in binarized_cell_topic.keys()
         ],
        index=binarized_cell_topic.keys()
    )
    # Embeddings. Cell embeddings in this case
    embeddings = {x: cistopic_obj.projections['cell'][x].loc[list(set(cistopic_obj.projections['cell'][x].index.tolist()).intersection(set(cell_names)))]
                  for x in cistopic_obj.projections['cell'].keys()}

    # Create minimal loom
    log.info('Creating minimal loom')
    export.export2loom(ex_mtx=ex_mtx,
                       regulons=topics,
                       out_fname=out_fname,
                       cell_annotations=None,
                       tree_structure=tree_structure,
                       title=title,
                       nomenclature=nomenclature,
                       embeddings=embeddings,
                       auc_mtx=cell_topic,
                       auc_thresholds=topic_thresholds)

    # Add annotations
    log.info('Adding annotations')
    path_to_loom = out_fname
    loom = SCopeLoom.read_loom(path_to_loom)
    if len(metrics):
        add_metrics(loom, metrics)
    if len(annotations):
        add_annotation(loom, annotations)

    # Add clusterings
    if cluster_annotation is not None:
        log.info('Adding clusterings')
        add_clusterings(loom, pd.DataFrame(cell_data[cluster_annotation]))

    # Add markers
    if cluster_markers is not None:
        log.info('Adding markers')
        annotation_in_markers = [
            x for x in cluster_annotation if x in cluster_markers.keys()
        ]
        annotation_not_in_markers = [
            x for x in cluster_annotation if x not in cluster_markers.keys()]

        for x in annotation_not_in_markers:
            log.info(x, 'is not in the cluster markers dictionary')
        cluster_markers = {
            k: v
            for k, v in cluster_markers.items()
            if k in annotation_in_markers
        }
        # Keeep regions in data
        for y in cluster_markers:
            cluster_markers[y] = {
                x: cluster_markers[y][x][cluster_markers[y][x].index.isin(region_names)]
                for x in cluster_markers[y].keys()
            }
        add_markers(loom, cluster_markers)

    log.info('Exporting')
    loom.export(out_fname)


def get_metadata(loom):
    """
    A helper function to get metadata
    """
    annot_metadata = loom.get_meta_data()['annotations']
    annot_mt_column_names = [annot_metadata[x]['name']
                             for x in range(len(annot_metadata))]
    annot_mt = pd.concat([pd.DataFrame(loom.col_attrs[annot_mt_column_names[x]])
                          for x in range(len(annot_mt_column_names))], axis=1)
    annot_mt.columns = [annot_mt_column_names[x]
                        for x in range(len(annot_mt_column_names))]
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
    loom.global_attrs["MetaData"].update({'metrics': md_metrics})


def add_annotation(loom, annots: pd.DataFrame):
    """
    A helper function to add annotations
    """
    md_annot = []
    for annot in annots:
        vals = list(annots[annot])
        uniq_vals = np.unique(vals)
        md_annot.append({
            "name": annot,
            "values": list(map(lambda x: str(x), uniq_vals.tolist()))
        })
        loom.col_attrs[annot] = np.array(annots[annot])
    loom.global_attrs["MetaData"].update({'annotations': md_annot})


def add_clusterings(loom: SCopeLoom,
                    cluster_data: pd.DataFrame):
    """
    A helper function to add clusters
    """
    col_attrs = loom.col_attrs

    attrs_metadata = {}
    attrs_metadata["clusterings"] = []
    j = 0

    for cluster_name in cluster_data.columns:

        clusterings = pd.DataFrame(index=cluster_data.index.tolist())

        clustering_id = j
        clustering_algorithm = cluster_name

        clustering_resolution = cluster_name
        cluster_marker_method = cluster_name

        num_clusters = len(np.unique(cluster_data[cluster_name]))
        cluster_2_number = {
            np.unique(cluster_data[cluster_name])[i]: i
            for i in range(num_clusters)
        }

        # Data
        clusterings[str(clustering_id)] = [cluster_2_number[x]
                                           for x in cluster_data[cluster_name].tolist()]

        # Metadata
        attrs_metadata["clusterings"] = attrs_metadata["clusterings"] + [{
            "id": clustering_id,
            "group": clustering_algorithm,
            "name": clustering_algorithm,
            "clusters": [],
            "clusterMarkerMetrics": [
                {
                    "accessor": "avg_logFC",
                    "name": "Avg. logFC",
                    "description": f"Average log fold change from {cluster_marker_method.capitalize()} test"
                }, {
                    "accessor": "pval",
                    "name": "Adjusted P-Value",
                    "description": f"Adjusted P-Value from {cluster_marker_method.capitalize()} test"
                }
            ]
        }]

        for i in range(0, num_clusters):
            cluster = {}
            cluster['id'] = i
            cluster['description'] = np.unique(cluster_data[cluster_name])[i]
            attrs_metadata['clusterings'][j]['clusters'].append(cluster)

        j += 1

        # Update column attribute Dict
        clusterings.columns = [str(clustering_id)]
        col_attrs_clusterings = {
            # Pick the first one as default clustering (this is purely
            # arbitrary)
            "ClusterID": clusterings[str(clustering_id)].values,
            "Clusterings": df_to_named_matrix(clusterings)
        }

    col_attrs = {**col_attrs, **col_attrs_clusterings}
    loom.col_attrs = col_attrs
    loom.global_attrs["MetaData"].update(
        {'clusterings': attrs_metadata["clusterings"]}
    )


def add_markers(loom: SCopeLoom,
                markers_dict: Dict[str, Dict[str, pd.DataFrame]]):
    """
    A helper function to add markers to clusterings
    """
    attrs_metadata = loom.global_attrs["MetaData"]
    row_attrs = loom.row_attrs
    for cluster_name in markers_dict:
        idx = [i for i in range(len(attrs_metadata['clusterings']))
               if attrs_metadata['clusterings'][i]["name"] == cluster_name][0]
        clustering_id = attrs_metadata['clusterings'][idx]["id"]
        num_clusters = len(attrs_metadata['clusterings'][idx]["clusters"])
        cluster_description = [
            attrs_metadata['clusterings'][idx]["clusters"][x]['description']
            for x in range(num_clusters)
        ]

        # Initialize
        cluster_markers = pd.DataFrame(
            index=loom.get_genes(),
            columns=[str(x) for x in np.arange(num_clusters)]
        ).fillna(0, inplace=False)
        cluster_markers_avg_logfc = pd.DataFrame(
            index=loom.get_genes(),
            columns=[str(x) for x in np.arange(num_clusters)]
        ).fillna(0, inplace=False)
        cluster_markers_pval = pd.DataFrame(
            index=loom.get_genes(),
            columns=[str(x) for x in np.arange(num_clusters)]
        ).fillna(0, inplace=False)

        # Populate
        for i in range(0, num_clusters):
            try:
                gene_names = markers_dict[cluster_name][cluster_description[i]].index.tolist()
                pvals_adj = markers_dict[cluster_name][cluster_description[i]]['Adjusted_pval']
                logfoldchanges = markers_dict[cluster_name][cluster_description[i]]['Log2FC']
                i = str(i)
                num_genes = len(gene_names)

                # Replace
                cluster_markers.loc[gene_names, i] = np.int(1)
                cluster_markers_avg_logfc.loc[gene_names, i] = logfoldchanges
                cluster_markers_pval.loc[gene_names, i] = pvals_adj
            except BaseException:
                print('No markers for ', cluster_description[i])

        # Update row attribute Dict
        row_attrs_cluster_markers = {
            f"ClusterMarkers_{str(idx)}": df_to_named_matrix(
                cluster_markers.astype(np.int8)),
            f"ClusterMarkers_{str(idx)}_avg_logFC": df_to_named_matrix(cluster_markers_avg_logfc.astype(np.float32)),
            f"ClusterMarkers_{str(idx)}_pval": df_to_named_matrix(cluster_markers_pval.astype(np.float32))
        }
        row_attrs = {**row_attrs, **row_attrs_cluster_markers}
        loom.row_attrs = row_attrs


def get_regulons(loom):
    """
    A helper function to get regulons
    """
    regulon_dict = pd.DataFrame(
        loom.get_regulons(),
        index=loom.row_attrs['Gene']
    ).to_dict()
    for t in regulon_dict:
        regulon_dict[t] = {x: y for x, y in regulon_dict[t].items() if y != 0}
    motif_data = {
        x['regulon']: x['motifData']
        for x in loom.global_attrs['MetaData']['regulonThresholds']
    }
    regulon_list = [
        Regulon(
            name=x,
            gene2weight=regulon_dict[x],
            transcription_factor=x.split('_')[0],
            gene2occurrence=[],
            context=frozenset(
                list(
                    motif_data[x]))
        )
        for x in regulon_dict.keys()
    ]
    return regulon_list


def df_to_named_matrix(df: pd.DataFrame):
    """
    A helper function to create metadata structure.
    """
    return np.array([tuple(row) for row in df.values],
                    dtype=np.dtype(list(zip(df.columns, df.dtypes))))
