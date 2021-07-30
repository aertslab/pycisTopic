from ctxcore.genesig import GeneSignature
from pyscenic.aucell import aucell
from typing import Dict, Union


from .utils import *
from .diff_features import *

def signature_enrichment(input: Union[pd.DataFrame, 'CistopicImputedFeatures'],
                        region_sets: Dict[str, pr.PyRanges],
                        auc_threshold: float = 0.05,
                        normalize: bool = False,
                        seed = None,
                        n_cpu: int = 1):
    """
    Get enrichment of a region signature in cells or topics using AUCell (Van de Sande et al., 2020)
    
    Parameters
    ---------
    input: pd.DataFrame or CistopicImputedFeatures
        A dataframe with regions as columns and topics/cells and rows; or a CistopicImputedFeatures object
    region_sets: Dictionary of pr.PyRanges
        A dictionary containing signatures as pr.PyRanges
    auc_threshold: float
        The fraction of the ranked genome to take into account for the calculation of the Area Under the recovery Curve. Default: 0.05
    normalize: bool
        Normalize the AUC values to a maximum of 1.0 per regulon. Default: False
    num_workers: int
        The number of cores to use. Default: 1
        
    Return
    ---------
        A pd.DataFrame containing signatures as columns, cells/topics as rows and AUC scores as values
        
    References
    ---------
    Van de Sande, B., Flerin, C., Davie, K., De Waegeneer, M., Hulselmans, G., Aibar, S., ... & Aerts, S. (2020). A scalable SCENIC workflow for single-cell gene 
    regulatory network analysis. Nature Protocols, 15(7), 2247-2276.
    """
    # Format input
    if isinstance(input, CistopicImputedFeatures):
        input =  pd.DataFrame.sparse.from_spmatrix(input.mtx.transpose(), columns = input.feature_names, index = input.cell_names)
    # Take regions in input
    regions_in_input = pr.PyRanges(region_names_to_coordinates(input.columns.tolist()))
    # Get signatures
    signatures = [region_set_to_signature(region_sets[key], regions_in_input, key) for key in region_sets.keys()]
    # Run aucell
    auc_sig = aucell(input, signatures,
                     num_workers = n_cpu,
                     auc_threshold = auc_threshold,
                     noweights = False,
                     normalize = normalize,
                     seed = seed)
    auc_sig.columns.names = [None]
    return auc_sig


def region_set_to_signature(query_region_set: pr.PyRanges,
                           target_region_set: pr.PyRanges,
                           name: str):
    """
    A helper function to intersect query regions with the input data set regions 
    
    Parameters
    ---------
    query_region_set: pr.PyRanges
        Pyranges with regions to query
    target_region_set: pr.PyRanges
        Pyranges with target regions
    name: str
        Name for the signature
    
    Return
    ---------
        A GeneSignature object to use with AUCell
    """
    query_in_target = query_region_set.join(target_region_set)
    query_in_target = query_in_target.df[['Chromosome', 'Start_b', 'End_b']]
    query_in_target.columns = ['Chromosome', 'Start', 'End']
    query_in_target = coord_to_region_names(pr.PyRanges(query_in_target))
    weights = np.ones(len(query_in_target))
    signature = GeneSignature(
                    name = name,
                    gene2weight = dict(zip(query_in_target, weights)))
    return signature

