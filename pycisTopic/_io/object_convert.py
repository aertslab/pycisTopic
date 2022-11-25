from mudata import MuData, AnnData
from pycisTopic.cistopic_class import CistopicObject
from pycisTopic.lda_models import CistopicLDAModel
import numpy as np
import pandas as pd
from collections import OrderedDict

def lda_model_object_to_mudata(
    lda_model_obj: CistopicLDAModel) -> MuData:
    """
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
    cistopic_obj: CistopicObject) -> MuData:
    """
    project: <class 'str'>
    path_to_fragments: <class 'dict'>
    selected_model: <class 'pycisTopic.lda_models.CistopicLDAModel'>
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
    
