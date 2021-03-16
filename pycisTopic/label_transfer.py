from anndata import AnnData
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import pandas as pd
import numpy as np
import scanorama
import logging
import sys

def label_transfer(ref_anndata: AnnData,
                  query_anndata: AnnData,
                  labels_to_transfer: List[str],
                  variable_genes: Optional[bool] = True,
                  methods: Optional[List[str]] = ['ingest', 'harmony', 'bbknn', 'scanorama', 'cca'],
                  pca_ncomps: Optional[List[int]] = [50, 50],
                  n_neighbours: Optional[List[int]] = [10,10],
                  bbknn_components: Optional[int] = 30,
                  cca_components: Optional[int] = 30,
                  return_label_weights: Optional[bool] = False):
    """
    Transferring labels from reference to query
    
    Parameters
    ---------
    ref_anndata: AnnData
        An AnnData object containing the reference data set (typically, scRNA-seq data)
    query_anndata: AnnData
        An AnnData object containing the query data set, with features matching with the reference data set 
        (typically, gene activities derived from scATAC-seq)
    labels_to_transfer: List
        Labels to transfer. They must be included in `ref_anndata.obs`.
    variable_genes: bool, optional
        Whether variable genes matching between the two data set should be used (True) or otherwise, all matching 
        genes (False). Default: True
    methods: List, optional
        Methods to be used for label transferring. These include: 'ingest' [from scanpy], 'harmony' [Korsunsky et al,
        2019], 'bbknn' [Polański et al, 2020], 'scanorama' [Hie et al, 2019] and 'cca'. Except for ingest, these
        methods return a common coembedding and labels are inferred using the distances between query and refenrence
        cells as weights.
    pca_ncomps: List, optional
        Number of principal components to use for reference and query, respectively. Default: [50,50]
    n_neighbours: List, optional
        Number of neighbours to use for reference and query, respectively. Default: [10,10]
    bbknn_components: int, optional
        Number of components to use for the umap for bbknn integration. Default: 30
    cca_components: int, optional
        Number of components to use for cca. Default: 30
    return_label_weights: bool, optional
        Whether to returnthe label scores per variable (as a dictionary, except for ingest). Default: False
    
    Return
    ------
    Dict, Dict
        A dictionary containing a data frame with mapped labels per method and optionally, a dictionary with data frames
        with the label scores per method and variable.
        
    References
    -----------  
    Korsunsky, I., Millard, N., Fan, J., Slowikowski, K., Zhang, F., Wei, K., ... & Raychaudhuri, S. (2019). Fast, 
    sensitive and accurate integration of single-cell data with Harmony. Nature methods, 16(12), 1289-1296.
    
    Polański, K., Young, M. D., Miao, Z., Meyer, K. B., Teichmann, S. A., & Park, J. E. (2020). BBKNN: fast batch 
    alignment of single cell transcriptomes. Bioinformatics, 36(3), 964-965.
    
    Hie, B., Bryson, B., & Berger, B. (2019). Efficient integration of heterogeneous single-cell transcriptomes
    using Scanorama. Nature biotechnology, 37(6), 685-691.
    """

    # Create cisTopic logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    # Process rna data
    log.info('Normalizing RNA data')
    ref_anndata = ref_anndata.copy()
    sc.pp.normalize_total(ref_anndata, target_sum=1e4)
    sc.pp.log1p(ref_anndata)
    sc.pp.highly_variable_genes(ref_anndata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    if variable_genes == True:
        ref_anndata = ref_anndata[:, ref_anndata.var.highly_variable]
    sc.pp.scale(ref_anndata, max_value=10)
    sc.tl.pca(ref_anndata, svd_solver='arpack', n_comps=pca_ncomps[0])
    sc.pp.neighbors(ref_anndata, use_rep="X_pca", n_neighbors=n_neighbours[0])
    sc.tl.umap(ref_anndata)
    # Process atac data 
    log.info('Normalizing ATAC data')
    query_anndata = query_anndata.copy()
    sc.pp.normalize_total(query_anndata, target_sum=1e4)
    sc.pp.log1p(query_anndata)
    sc.pp.highly_variable_genes(query_anndata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    if variable_genes == True:
        query_anndata = query_anndata[:, query_anndata.var.highly_variable]
    sc.pp.scale(query_anndata, max_value=10)
    sc.tl.pca(query_anndata, svd_solver='arpack', n_comps=pca_ncomps[1])
    sc.pp.neighbors(query_anndata, n_neighbors = n_neighbours[1])
    # Select overlapping variable features
    var_names = ref_anndata.var_names.intersection(set(query_anndata.var_names))
    ref_anndata = ref_anndata[:, var_names]
    query_anndata = query_anndata[:, var_names]
    # Concatenate object
    adata_concat = ref_anndata.concatenate(query_anndata, batch_categories=['RNA', 'ATAC'])
    # Run methods
    transfer_dict = {}
    label_weight_dict = {}
    if 'ingest' in methods:
        log.info('Running integration with ingest')
        for var in labels_to_transfer:
            sc.tl.ingest(query_anndata, ref_anndata, obs=var)
            query_anndata.obs.loc[:,var] = query_anndata.obs.loc[:,var].cat.remove_unused_categories()
        transfer_data = pd.DataFrame(query_anndata.obs.loc[:,labels_to_transfer])
        transfer_data.columns = 'ingest_' + transfer_data.columns
        transfer_dict['ingest']= transfer_data
    if 'harmony' in methods:
        log.info('Running integration with harmony')
        sc.tl.pca(adata_concat, svd_solver='arpack')
        sc.external.pp.harmony_integrate(adata_concat, ['batch'])
        distances = 1 - cosine_distances(adata_concat[adata_concat.obs.batch == "RNA"].obsm["X_pca_harmony"], 
                                         adata_concat[adata_concat.obs.batch == "ATAC"].obsm["X_pca_harmony"])
        harmony_transfer_list=[]
        for var in labels_to_transfer:
            class_prob = label_transfer_coembedded(distances, ref_anndata.obs.loc[:,var])
            cp_df = pd.DataFrame(class_prob, columns=np.sort(ref_anndata.obs.loc[:,var].unique()))
            cp_df.index = query_anndata.obs.index
            cp_df = pd.DataFrame(StandardScaler().fit_transform(cp_df), index=cp_df.index.to_list(), columns=cp_df.columns)
            label_weight_dict['harmony'] = {var: cp_df}
            assigned_label = pd.DataFrame(cp_df.idxmax(axis=1), columns=['harmony_'+var])
            harmony_transfer_list.append(assigned_label)
        transfer_dict['harmony'] = pd.concat(harmony_transfer_list, axis=1, sort=False)
    if 'bbknn' in methods:
        log.info('Running integration with bbknn')
        sc.external.pp.bbknn(adata_concat, batch_key='batch')
        sc.tl.umap(adata_concat, n_components=bbknn_components)
        distances = 1 - cosine_distances(adata_concat[adata_concat.obs.batch == "RNA"].obsm["X_umap"], 
                                         adata_concat[adata_concat.obs.batch == "ATAC"].obsm["X_umap"])
        bbknn_transfer_list=[]
        for var in labels_to_transfer:
            class_prob = label_transfer_coembedded(distances, ref_anndata.obs.loc[:,var])
            cp_df = pd.DataFrame(class_prob, columns=np.sort(ref_anndata.obs.loc[:,var].unique()))
            cp_df.index = query_anndata.obs.index
            cp_df = pd.DataFrame(StandardScaler().fit_transform(cp_df), index=cp_df.index.to_list(), columns=cp_df.columns)
            label_weight_dict['bbknn'] = {var: cp_df}
            assigned_label = pd.DataFrame(cp_df.idxmax(axis=1), columns=['bbknn_'+var])
            bbknn_transfer_list.append(assigned_label)
        transfer_dict['bbknn'] = pd.concat(bbknn_transfer_list, axis=1, sort=False)
    if 'scanorama' in methods:
        log.info('Running integration with scanorama')
        integrated = scanorama.correct_scanpy([ref_anndata, query_anndata], return_dimred=True)
        embedding = np.concatenate([x.obsm['X_scanorama'] for x in integrated], axis=0) 
        adata_concat.obsm["scanorama_embedding"] = embedding
        distances = 1 - cosine_distances(adata_concat[adata_concat.obs.batch == "RNA"].obsm["scanorama_embedding"], 
                                         adata_concat[adata_concat.obs.batch == "ATAC"].obsm["scanorama_embedding"])
        scanorama_transfer_list=[]
        for var in labels_to_transfer:
            class_prob = label_transfer_coembedded(distances, ref_anndata.obs.loc[:,var])
            cp_df = pd.DataFrame(class_prob, columns=np.sort(ref_anndata.obs.loc[:,var].unique()))
            cp_df.index = query_anndata.obs.index
            cp_df = pd.DataFrame(StandardScaler().fit_transform(cp_df), index=cp_df.index.to_list(), columns=cp_df.columns)
            label_weight_dict['scanorama'] = {var: cp_df}
            assigned_label = pd.DataFrame(cp_df.idxmax(axis=1), columns=['scanorama_'+var])
            scanorama_transfer_list.append(assigned_label)
        transfer_dict['scanorama'] = pd.concat(scanorama_transfer_list, axis=1, sort=False)
    if 'cca' in methods:
        log.info('Running integration with cca')
        X_sc = ref_anndata.X.T
        Y_sc = query_anndata.X.T
        X = CCA(n_components=cca_components).fit(X_sc, Y_sc)
        embedding = np.concatenate([X.x_loadings_, X.y_loadings_], axis=0)
        adata_concat.obsm["cca_embedding"] = embedding
        distances = 1 - cosine_distances(adata_concat[adata_concat.obs.batch == "RNA"].obsm["cca_embedding"], 
                                         adata_concat[adata_concat.obs.batch == "ATAC"].obsm["cca_embedding"])
        cca_transfer_list=[]
        for var in labels_to_transfer:
            class_prob = label_transfer_coembedded(distances, ref_anndata.obs.loc[:,var])
            cp_df = pd.DataFrame(class_prob, columns=np.sort(ref_anndata.obs.loc[:,var].unique()))
            cp_df.index = query_anndata.obs.index
            cp_df = pd.DataFrame(StandardScaler().fit_transform(cp_df), index=cp_df.index.to_list(), columns=cp_df.columns)
            label_weight_dict['cca'] = {var: cp_df}
            assigned_label = pd.DataFrame(cp_df.idxmax(axis=1), columns=['cca_'+var])
            cca_transfer_list.append(assigned_label)
        transfer_dict['cca'] = pd.concat(cca_transfer_list, axis=1, sort=False)
    return transfer_dict

        
def label_transfer_coembedded(dist, labels):
    """
    A helper function to propagate labels in a common space
    """
    lab = pd.get_dummies(labels).to_numpy().T
    class_prob = lab @ dist
    norm = np.linalg.norm(class_prob, 2, axis=0)
    class_prob = class_prob / norm
    class_prob = (class_prob.T - class_prob.min(1)) / class_prob.ptp(1)
    return class_prob
