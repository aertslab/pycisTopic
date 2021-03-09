from sklearn.metrics.pairwise import cosine_distances
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import pandas as pd
import numpy as np
import scanorama
import logging
import sys

def label_transfer(rna_anndata,
                  atac_anndata,
                  vars_to_transfer,
                  variable_genes=True,
                  methods=['ingest', 'harmony', 'bbknn', 'scanorama', 'cca'],
                  pca_ncomps = [50, 50],
                  n_neighbours = [10,10],
                  bbknn_components = 30,
                  cca_components = 30):
    # Create cisTopic logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    # Process rna data
    log.info('Normalizing RNA data')
    sc.pp.normalize_total(rna_anndata, target_sum=1e4)
    sc.pp.log1p(rna_anndata)
    sc.pp.highly_variable_genes(rna_anndata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    if variable_genes == True:
        rna_anndata = rna_anndata[:, rna_anndata.var.highly_variable]
    sc.pp.scale(rna_anndata, max_value=10)
    sc.tl.pca(rna_anndata, svd_solver='arpack', n_comps=pca_ncomps[0])
    sc.pp.neighbors(rna_anndata, use_rep="X_pca", n_neighbors=n_neighbours[0])
    sc.tl.umap(rna_anndata)
    # Process atac data 
    log.info('Normalizing ATAC data')
    sc.pp.normalize_total(atac_anndata, target_sum=1e4)
    sc.pp.log1p(atac_anndata)
    sc.pp.highly_variable_genes(atac_anndata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    if variable_genes == True:
        atac_anndata = atac_anndata[:, atac_anndata.var.highly_variable]
    sc.pp.scale(atac_anndata, max_value=10)
    sc.tl.pca(atac_anndata, svd_solver='arpack', n_comps=pca_ncomps[1])
    sc.pp.neighbors(atac_anndata, n_neighbors = n_neighbours[1])
    # Select overlapping variable features
    rna_anndata.var_names = rna_anndata.var.iloc[:,0]
    atac_anndata.var_names = atac_anndata.var.iloc[:,0]
    var_names = rna_anndata.var_names.intersection(set(atac_anndata.var.iloc[:,0]))
    rna_anndata = rna_anndata[:, var_names]
    atac_anndata = atac_anndata[:, var_names]
    # Concatenate object
    adata_concat = rna_anndata.concatenate(atac_anndata, batch_categories=['RNA', 'ATAC'])
    # Run methods
    transfer_dict = {}
    if 'ingest' in methods:
        log.info('Running integration with ingest')
        for var in vars_to_transfer:
            sc.tl.ingest(atac_anndata, rna_anndata, obs=var)
            atac_anndata.obs.loc[:,var] = atac_anndata.obs.loc[:,var].cat.remove_unused_categories()
        transfer_data = pd.DataFrame(atac_anndata.obs.loc[:,vars_to_transfer])
        transfer_data.columns = 'ingest_' + transfer_data.columns
        transfer_dict['ingest']= transfer_data
    if 'harmony' in methods:
        log.info('Running integration with harmony')
        sc.tl.pca(adata_concat, svd_solver='arpack')
        sc.external.pp.harmony_integrate(adata_concat, ['batch'])
        distances = 1 - cosine_distances(adata_concat[adata_concat.obs.batch == "RNA"].obsm["X_pca_harmony"], 
                                         adata_concat[adata_concat.obs.batch == "ATAC"].obsm["X_pca_harmony"])
        harmony_transfer_list=[]
        for var in vars_to_transfer:
            class_prob = label_transfer_coembedded(distances, rna_anndata.obs.loc[:,var])
            cp_df = pd.DataFrame(class_prob, columns=np.sort(rna_anndata.obs.loc[:,var].unique()))
            cp_df.index = atac_anndata.obs.index
            cp_df = pd.DataFrame(StandardScaler().fit_transform(cp_df), index=cp_df.index.to_list(), columns=cp_df.columns)
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
        for var in vars_to_transfer:
            class_prob = label_transfer_coembedded(distances, rna_anndata.obs.loc[:,var])
            cp_df = pd.DataFrame(class_prob, columns=np.sort(rna_anndata.obs.loc[:,var].unique()))
            cp_df.index = atac_anndata.obs.index
            cp_df = pd.DataFrame(StandardScaler().fit_transform(cp_df), index=cp_df.index.to_list(), columns=cp_df.columns)
            assigned_label = pd.DataFrame(cp_df.idxmax(axis=1), columns=['bbknn_'+var])
            bbknn_transfer_list.append(assigned_label)
        transfer_dict['bbknn'] = pd.concat(bbknn_transfer_list, axis=1, sort=False)
    if 'scanorama' in methods:
        log.info('Running integration with scanorama')
        integrated = scanorama.correct_scanpy([rna_anndata, atac_anndata], return_dimred=True)
        embedding = np.concatenate([x.obsm['X_scanorama'] for x in integrated], axis=0) 
        adata_concat.obsm["scanorama_embedding"] = embedding
        distances = 1 - cosine_distances(adata_concat[adata_concat.obs.batch == "RNA"].obsm["scanorama_embedding"], 
                                         adata_concat[adata_concat.obs.batch == "ATAC"].obsm["scanorama_embedding"])
        scanorama_transfer_list=[]
        for var in vars_to_transfer:
            class_prob = label_transfer_coembedded(distances, rna_anndata.obs.loc[:,var])
            cp_df = pd.DataFrame(class_prob, columns=np.sort(rna_anndata.obs.loc[:,var].unique()))
            cp_df.index = atac_anndata.obs.index
            cp_df = pd.DataFrame(StandardScaler().fit_transform(cp_df), index=cp_df.index.to_list(), columns=cp_df.columns)
            assigned_label = pd.DataFrame(cp_df.idxmax(axis=1), columns=['scanorama_'+var])
            scanorama_transfer_list.append(assigned_label)
        transfer_dict['scanorama'] = pd.concat(scanorama_transfer_list, axis=1, sort=False)
    if 'cca' in methods:
        log.info('Running integration with cca')
        X_sc = rna_anndata.X.T
        Y_sc = atac_anndata.X.T
        X = CCA(n_components=cca_components).fit(X_sc, Y_sc)
        embedding = np.concatenate([X.x_loadings_, X.y_loadings_], axis=0)
        adata_concat.obsm["cca_embedding"] = embedding
        distances = 1 - cosine_distances(adata_concat[adata_concat.obs.batch == "RNA"].obsm["cca_embedding"], 
                                         adata_concat[adata_concat.obs.batch == "ATAC"].obsm["cca_embedding"])
        cca_transfer_list=[]
        for var in vars_to_transfer:
            class_prob = label_transfer_coembedded(distances, rna_anndata.obs.loc[:,var])
            cp_df = pd.DataFrame(class_prob, columns=np.sort(rna_anndata.obs.loc[:,var].unique()))
            cp_df.index = atac_anndata.obs.index
            cp_df = pd.DataFrame(StandardScaler().fit_transform(cp_df), index=cp_df.index.to_list(), columns=cp_df.columns)
            assigned_label = pd.DataFrame(cp_df.idxmax(axis=1), columns=['cca_'+var])
            cca_transfer_list.append(assigned_label)
        transfer_dict['cca'] = pd.concat(cca_transfer_list, axis=1, sort=False)
    return transfer_dict

        
def label_transfer_coembedded(dist, labels):
    lab = pd.get_dummies(labels).to_numpy().T
    class_prob = lab @ dist
    norm = np.linalg.norm(class_prob, 2, axis=0)
    class_prob = class_prob / norm
    class_prob = (class_prob.T - class_prob.min(1)) / class_prob.ptp(1)
    return class_prob
