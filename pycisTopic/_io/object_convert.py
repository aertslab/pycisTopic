from mudata import MuData, AnnData
from pycisTopic.cistopic_class import CistopicObject
from pycisTopic.lda_models import CistopicLDAModel
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Tuple

def lda_model_object_to_mudata(
    lda_model_obj: CistopicLDAModel) -> MuData:
    """
    Converts an lda model object to MuData

    Parameters
    ----------
        lda_model_obj: CistopicLDAModel An LDA model object
    
    Returns
    -------
        Mudata object with topic model

    Examples
    --------
        >>> lda_model_object_to_mudata(cistopic_obj.selected_model)
            MuData object with n_obs × n_vars = 22 × 339284
            obs: 'Log10_Assignments', 'Assignments', 'Regions_in_binarized_topic', 'Coherence', 'Marginal_topic_dist', 'Gini_index'
            2 modalities
                cell_topic:	22 x 4792
                region_topic:	22 x 334492
    """
    #construct obs
    obs = lda_model_obj.topic_qc_metrics.copy()
    
    mudata_constructor = {}
    adata_cell_topic = AnnData(
        X = lda_model_obj.cell_topic, dtype = np.float64,
        obs = pd.DataFrame(index = lda_model_obj.cell_topic.index), 
        var = pd.DataFrame(index = lda_model_obj.cell_topic.columns))
    mudata_constructor['cell_topic'] = adata_cell_topic
    
    if type(lda_model_obj.cell_topic_harmony) == pd.DataFrame:
        adata_cell_topic_harmony = AnnData(
            X = lda_model_obj.cell_topic_harmony, dtype = np.float64,
            obs = pd.DataFrame(index = lda_model_obj.cell_topic_harmony.index),
            var = pd.DataFrame(index = lda_model_obj.cell_topic_harmony))
        mudata_constructor['cell_topic_harmony'] = adata_cell_topic_harmony

    adata_region_topic = AnnData(
        X = lda_model_obj.topic_region.T, dtype = np.float64,
        obs = pd.DataFrame(index = lda_model_obj.topic_region.columns),
        var = pd.DataFrame(index = lda_model_obj.topic_region.index))
    mudata_constructor['region_topic'] = adata_region_topic

    #construct uns:
    uns = OrderedDict()
    uns['metrics'] = lda_model_obj.metrics.loc['Metric'].to_dict()
    uns['parameters'] = lda_model_obj.parameters.T.loc['Parameter'].to_dict()

    return MuData(mudata_constructor, obs = obs, uns = uns)


def cistopic_object_to_mudata(
    cistopic_obj: CistopicObject) -> Tuple(MuData, MuData):
    """
    Converts a cistopic object into mudata,

    Parameters
    ----------
        cistopic_obj: CistopicObject A cistopic object
    
    Returns
    -------
        A tuple of a mudata for accessibility and a mudata for topics.
    
    Examples
    --------
        >>> mudata_accessibility, mudata_lda_model = cistopic_object_to_mudata(cistopic_obj)
        >>> mudata_accessibility
            MuData object with n_obs × n_vars = 4792 × 668984
            obs:	'TSS_enrichment', 'Log_total_nr_frag', 'cisTopic_log_nr_acc', 'cisTopic_nr_acc', 'barcode', 'cisTopic_nr_frag', 'cisTopic_log_nr_frag', 'Total_nr_frag', 'FRIP', 'Total_nr_frag_in_regions', 'Unique_nr_frag', 'Dupl_nr_frag', 'Dupl_rate', 'Unique_nr_frag_in_regions', 'Log_unique_nr_frag', 'BARCODE', 'NUM.SNPS', 'NUM.READS', 'DROPLET.TYPE', 'BEST.GUESS', 'BEST.LLK', 'NEXT.GUESS', 'NEXT.LLK', 'DIFF.LLK.BEST.NEXT', 'BEST.POSTERIOR', 'SNG.POSTERIOR', 'SNG.BEST.GUESS', 'SNG.BEST.LLK', 'SNG.NEXT.GUESS', 'SNG.NEXT.LLK', 'SNG.ONLY.POSTERIOR', 'DBL.BEST.GUESS', 'DBL.BEST.LLK', 'DIFF.LLK.SNG.DBL', 'sample_id', 'line', 'state'
            var:	'Chromosome', 'Start', 'End', 'Width', 'cisTopic_nr_frag', 'cisTopic_log_nr_frag', 'cisTopic_nr_acc', 'cisTopic_log_nr_acc', 'is_promoter'
            obsm:	'UMAP'
            2 modalities
                fragment_counts:	4792 x 334492
                binary_fragment_counts:	4792 x 334492
        >>> mudata_lda_model
            MuData object with n_obs × n_vars = 22 × 339284
            obs:	'Log10_Assignments', 'Assignments', 'Regions_in_binarized_topic', 'Coherence', 'Marginal_topic_dist', 'Gini_index'
            2 modalities
                cell_topic:	22 x 4792
                region_topic:	22 x 334492
    """
    #construct var field
    var = cistopic_obj.region_data.loc[cistopic_obj.region_names].infer_objects()

    #construct obs field
    obs = cistopic_obj.cell_data.loc[cistopic_obj.cell_names].infer_objects()

    #construct obsm
    obsm = None
    if 'cell' in cistopic_obj.projections.keys():
        obsm = {
            projection: cistopic_obj.projections['cell'][projection].to_numpy() 
            for projection in cistopic_obj.projections['cell'].keys()}
    
    #contruct varm
    varm = None
    if 'region' in cistopic_obj.projections.keys():
        varm = {
            projection: cistopic_obj.projections['region'][projection].to_numpy() 
            for projection in cistopic_obj.projections['region'].keys()}
    
    #construct uns:
    uns = OrderedDict()
    uns['project'] = cistopic_obj.project
    uns['path_to_fragments'] = cistopic_obj.path_to_fragments

    mudata_constructor = {}

    #construct fragment matrix AnnData
    adata_fragment_counts = AnnData(
        X = cistopic_obj.fragment_matrix.T, dtype = np.int32,
        var = pd.DataFrame(index = var.index), obs = pd.DataFrame(index = obs.index))
    mudata_constructor['fragment_counts'] = adata_fragment_counts

    #construct binary matrix AnnData
    adata_binary_fragment_counts = AnnData(
        X = cistopic_obj.binary_matrix.T, dtype = np.int32, #more efficient to store as boolean instead?
        var = pd.DataFrame(index = var.index), obs = pd.DataFrame(index = obs.index))
    mudata_constructor['binary_fragment_counts'] = adata_binary_fragment_counts

    if type(cistopic_obj.selected_model) == CistopicLDAModel:
        mudata_lda_model = lda_model_object_to_mudata(cistopic_obj.selected_model)
    
    mudata_accessibility = MuData(mudata_constructor, obs = obs, var = var, obsm = obsm, varm = varm)

    return mudata_accessibility, mudata_lda_model
    
