from typing import Dict, Union

from ctxcore.genesig import GeneSignature
from pyscenic.aucell import aucell4r

from .diff_features import *
from .utils import *


def signature_enrichment(
    rankings: "CistopicImputedFeatures",
    signatures: Union[Dict[str, pr.PyRanges], Dict[str, List]],
    enrichment_type: str = "region",
    auc_threshold: float = 0.05,
    normalize: bool = False,
    n_cpu: int = 1,
):
    """
    Get enrichment of a region signature in cells or topics using AUCell (Van de Sande et al., 2020)

    Parameters
    ---------
    rankings: CistopicImputedFeatures
        A CistopicImputedFeatures object with ranking values
    signatures: Dictionary of pr.PyRanges (for regions) or list (for genes)
        A dictionary containing region signatures as pr.PyRanges or gene names as list
    enrichment_type: str
        Whether features are genes or regions
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
    # Compute rankings if needed and format input
    rankings = pd.DataFrame(
        rankings.mtx.transpose(),
        columns=rankings.feature_names,
        index=rankings.cell_names,
    )
    # Take regions in input
    if enrichment_type == "region":
        regions_in_input = pr.PyRanges(
            region_names_to_coordinates(rankings.columns.tolist())
        )
        # Get signatures
        signatures = [
            region_set_to_signature(signatures[key], regions_in_input, key)
            for key in signatures.keys()
        ]
    if enrichment_type == "gene":
        # Get signatures
        signatures = [
            gene_set_to_signature(signatures[key], key) for key in signatures.keys()
        ]
    # Run aucell
    auc_sig = aucell4r(
        df_rnk=rankings,
        signatures=signatures,
        auc_threshold=auc_threshold,
        noweights=False,
        normalize=normalize,
        num_workers=n_cpu,
    )
    auc_sig.columns.names = [None]
    return auc_sig


def region_set_to_signature(
    query_region_set: pr.PyRanges, target_region_set: pr.PyRanges, name: str
):
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
    query_in_target = query_in_target.df[["Chromosome", "Start_b", "End_b"]]
    query_in_target.columns = ["Chromosome", "Start", "End"]
    query_in_target = coord_to_region_names(pr.PyRanges(query_in_target))
    weights = np.ones(len(query_in_target))
    signature = GeneSignature(
        name=name, gene2weight=dict(zip(query_in_target, weights))
    )
    return signature


def gene_set_to_signature(gene_set: List, name: str):
    """
    A helper function to generat gene signatures

    Parameters
    ---------
    gene_set: pr.PyRanges
        List of genes
    name: str
        Name for the signature

    Return
    ---------
        A GeneSignature object to use with AUCell
    """
    weights = np.ones(len(gene_set))
    signature = GeneSignature(name=name, gene2weight=dict(zip(gene_set, weights)))
    return signature
