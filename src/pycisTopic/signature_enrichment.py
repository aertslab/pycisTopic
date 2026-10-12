from __future__ import annotations

from multiprocessing import cpu_count
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore
import polars as pl
from ctxcore.aucell import aucell4r  # type: ignore
from ctxcore.genesig import GeneSignature  # type: ignore

from pycisTopic.genomic_ranges import intersection
from pycisTopic.imputed_accessibility import (
    impute_accessibility_chunked,
    rank_imputed_accessibility,
)

if TYPE_CHECKING:
    import logging


def _polars_granges_to_region_names(
    granges: pl.DataFrame,
) -> list[str]:
    return granges.with_columns(
        region_names = (
            pl.concat_str(
                [
                    pl.col("Chromosome"),
                    pl.concat_str(
                        [
                            pl.col("Start"),
                            pl.col("End")
                        ],
                        separator="-"
                    )
                ],
                separator=":"
            )
        )
    )["region_names"].to_list()

def _region_names_to_signature(
    region_names: list[str],
    name: str,
) -> GeneSignature:
    weights = np.ones(len(region_names))
    return GeneSignature(
        name=name, gene2weight=dict(zip(region_names, weights))
    )

def signature_enrichment(
    region_topic: npt.NDArray[np.float32],
    cell_topic: npt.NDArray[np.float32],
    region_topic_granges: pl.DataFrame,
    signatures: dict[str, pl.DataFrame],
    chunk_size: int,
    normalize: bool,
    min_frac_consensus: float,
    min_frac_signature: float,
    seed: int,
    auc_threshold: float = 0.05,
    log: logging.Logger | None = None,
) -> npt.NDArray[np.float32]:
    """
    Calculate enrichment of region signatures in cells using AUCell (Van de Sande et al., 2020).

    Parameters
    ----------
    region_topic
        Region topic matrix (regions x topics).
    cell_topic
        Cell topic matrix (topic x cells).
    region_topic_granges
        Polars Dataframe with genomic ranges corresponding to region topic.
    signatures
        Dictionary of genomic ranges signatures (polars DataFrames).
    chunk_size
        The number of cells to process at once.
    normalize: bool
        Normalize the AUC values to a maximum of 1.0 per regulon.
    min_frac_consensus
        Minimal fractional overlap of signature and consensus peak
        relative to consensus peak.
    min_frac_signature
        Minimal fractional overlap of signature and consensus peak
        relative to signature.
    seed
        Seed used to randomly resolve tied values in imputed accessibility
        for generatin the ranking.
    auc_threshold: float
        The fraction of the ranked genome to take into account for the calculation
        of the Area Under the recovery Curve. Default: 0.05.
    log
        Optional logger.

    Returns
    -------
    A Numpy array with auc values across cells (cell x signature)

    """
    if region_topic_granges.shape[0] != region_topic.shape[0]:
        raise ValueError(
            f"Length of the region names ({region_topic_granges.shape[0]}) "
            f"does not match the shape of region_topic {region_topic.shape}"
        )
    region_names = _polars_granges_to_region_names(region_topic_granges)

    # Put signatures in coordinate frame of region topic by performing
    # intersect and retaining regions of the region topic granges that pass
    # the overlap thresholds (min_frac_consensus and min_frac_signatures).
    gr_signatures_consensus: dict[str, pl.DataFrame] = {}
    for signature, sign_granges in signatures.items():
        gr_signatures_consensus[signature] = intersection(
            regions1_df_pl=region_topic_granges,
            regions2_df_pl=sign_granges,
            regions1_coord=True,
            add_overlap_size=True,
            regions1_suffix="@1"
        ).filter(
            (pl.col("fr1_inter") >= min_frac_consensus) &
            (pl.col("fr2_inter") >= min_frac_signature)
        ).select(
            pl.col("Chromosome@1"),
            pl.col("Start@1"),
            pl.col("End@1")
        ).rename(
            {
                "Chromosome@1": "Chromosome",
                "Start@1": "Start",
                "End@1": "End"
            }
        )

    # Convert granges signatures to gene signatures
    gs_signatures_consensus: list[GeneSignature] = [
        _region_names_to_signature(
            region_names=_polars_granges_to_region_names(granges),
            name=signature
        )
        for signature, granges in gr_signatures_consensus.items()
    ]

    # initialize aucell values
    n_cells = cell_topic.shape[1]
    n_signatures = len(signatures)
    aucell_values: npt.NDArray[np.float32] = np.empty(
        (n_cells, n_signatures), dtype=np.float32
    )
    for (cell_start, cell_end), imputed_acc_chunk in impute_accessibility_chunked(
        region_topic=region_topic,
        cell_topic=cell_topic,
        chunk_size=chunk_size,
        chunk_along="cell",
        log=log,
        return_start_end=True,
    ):
        if log is not None:
            log.info("Generating ranking.")
        ranking_chunk = rank_imputed_accessibility(
            imputed_accessibility=imputed_acc_chunk,
            seed=seed
        )
        if log is not None:
            log.info("Calculating AUCs.")
        aucell_values[cell_start: cell_end] = aucell4r(
            df_rnk=pd.DataFrame(
                ranking_chunk.T,
                columns=region_names
            ),
            signatures=gs_signatures_consensus, # type: ignore
            auc_threshold=auc_threshold,
            noweights=False,
            normalize=False,
            num_workers=min(chunk_size, cpu_count()),
        ).to_numpy()

    return aucell_values / aucell_values.max(axis=0) if normalize else aucell_values
