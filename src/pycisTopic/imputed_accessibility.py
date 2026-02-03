import logging
from typing import Iterator, Literal

import numpy as np
import numpy.typing as npt
import polars as pl


def _alocate_chunk_array(
    n_cells: int,
    n_regions: int,
    chunk_size: int,
    chunk_along: Literal["cell", "region"],
    log: logging.Logger | None
) -> npt.NDArray[np.float32]:
    if chunk_along == "region":
        if log is not None:
            log.info(
                f"Allocate {(chunk_size * n_cells * 4 / 1024**3):.3f} GiB of RAM for "
                f"calculating (partial) imputed accessibility per cell for ({n_cells}) cells "
                f"for chunk of {chunk_size} regions."
            )
        return np.empty(
            (chunk_size, n_cells),
            dtype=np.float32
        )
    else:
        if log is not None:
            log.info(
                f"Allocate {(chunk_size * n_regions * 4 / 1024**3):.3f} GiB of RAM for "
                f"calculating (partial) imputed accessibility per region for ({n_regions}) regions "
                f"for chunk of {chunk_size} cells."
            )
        return np.empty(
            (n_regions, chunk_size),
            dtype=np.float32
        )

def impute_accessibility_chunked(
    region_topic: npt.NDArray[np.float32],
    cell_topic: npt.NDArray[np.float32],
    chunk_size: int,
    chunk_along: Literal["cell", "region"],
    log: logging.Logger | None = None,
    return_start_end: bool = False,
) -> Iterator[npt.NDArray[np.float32]] | Iterator[tuple[tuple[int, int], npt.NDArray[np.float32]]]:
    """
    Impute accessibility in chunks.

    Parameters
    ----------
    region_topic
        Region topic matrix (regions x topics).
    cell_topic
        Cell topic matrix (topic x cells).
    chunk_size
        The size of the chunks.
    chunk_along
        Whether to chunk along cells or regions.
    log
        Optional logging.Logger
    return_start_end
        Whether to return the start and end index of the chunk.

    Yields
    ------
    Numpy array with imputed accessibility of chunk.

    """
    if region_topic.shape[1] != cell_topic.shape[0]:
        raise ValueError(
            "region- and cell-topic dimensions do not match"
            f" region_topic: {region_topic.shape[1]}"
            f" cell_topic: {cell_topic.shape[0]}."
            "Maybe you have to transpose region_topic or cell_topic"
        )

    if chunk_along not in ["cell", "region"]:
        raise ValueError(f"chunk along should be either 'cell' or 'region', not {chunk_along}!")

    if log is not None:
        log.info(f"Calculating imputed accessibility in chunks across {chunk_along} ...")

    n_regions = region_topic.shape[0]
    n_cells = cell_topic.shape[1]

    imputed_acc_chunk: npt.NDArray[np.float32] = _alocate_chunk_array(
        n_cells=n_cells,
        n_regions=n_regions,
        chunk_size=chunk_size,
        chunk_along=chunk_along,
        log=log
    )

    n_total: int = n_regions if chunk_along == "region" else n_cells

    for chunk_start in range(0, n_total, chunk_size):
        chunk_end = chunk_start + chunk_size

        if log is not None:
            log.info(
                "Calculate partial imputed accessibility "
                f"{chunk_start}-{chunk_end} (out of {n_total})."
            )

        # get current chunk of regions or cells
        if chunk_along == "region":
            _cell_topic = cell_topic
            _region_topic = region_topic[
                chunk_start: chunk_end
            ]
            current_chunk_size = _region_topic.shape[0]
        else:
            _cell_topic = cell_topic[:, chunk_start: chunk_end]
            _region_topic = region_topic
            current_chunk_size = _cell_topic.shape[1]

        if current_chunk_size < chunk_size:
            del imputed_acc_chunk
            imputed_acc_chunk = _alocate_chunk_array(
                n_cells=n_cells,
                n_regions=n_regions,
                chunk_size=current_chunk_size,
                chunk_along=chunk_along,
                log=log
            )

        np.matmul(_region_topic, _cell_topic, out=imputed_acc_chunk)
        if return_start_end:
            yield (chunk_start, chunk_end), imputed_acc_chunk
        else:
            yield imputed_acc_chunk

def rank_imputed_accessibility(
   imputed_accessibility: npt.NDArray[np.float32],
   seed: int = 123,
   method: Literal["numpy", "polars"] = "polars",
) -> npt.NDArray[np.int32]:
    """
    Generate rankings per cell based on the imputed accessibility scores per region.

    Parameters
    ----------
    imputed_accessibility
        Numpy array of imputed accessibility (regions x cells)
    seed
        Random seed to ensure reproducibility of the rankings when there are ties
    method
        Method to use for the ranking implementation.
        Options are "numpy" (same rankings as with older versions of pycisTopic)
        or "polars" (fastest). Default: "polars".

    Return
    ------
        Numpy array
        Containing ranking values rather than scores (regions x cells).

    """
    if method != "numpy" and method != "polars":
        raise ValueError(
            f'Invalid method ("{method}") for ranking implementation. Use "numpy" or "polars".'
        )

    # Initialize random number generator, for handling ties.
    rng = np.random.default_rng(seed=seed)

    # Function to make rankings per array.
    def rank_scores_and_assign_random_ranking_in_range_for_ties_with_numpy(
        scores_with_ties_for_motif_or_track_numpy: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.int32]:
        #
        # Create random permutation so tied scores will have a different ranking each time.
        random_permutations_to_break_ties_numpy = rng.permutation(
            scores_with_ties_for_motif_or_track_numpy.shape[0]
        )
        ranking_with_broken_ties_for_motif_or_track_numpy = np.empty(
            scores_with_ties_for_motif_or_track_numpy.shape[0],
            dtype=np.int32,
        )
        ranking_with_broken_ties_for_motif_or_track_numpy[
            random_permutations_to_break_ties_numpy[
                (-scores_with_ties_for_motif_or_track_numpy)[
                    random_permutations_to_break_ties_numpy
                ].argsort()
            ]
        ] = np.arange(
            scores_with_ties_for_motif_or_track_numpy.shape[0],
            dtype=np.int32,
        )

        return ranking_with_broken_ties_for_motif_or_track_numpy

    def rank_scores_and_assign_random_ranking_in_range_for_ties_with_polars(
        scores_with_ties_for_motif_or_track_numpy: npt.NDArray[np.float32],
        seed: int,
    ) -> npt.NDArray[np.int32]:
        # Rank scores and assign a random ranking in range for regions/genes with
        # the same score.
        #   - Convert numpy array to Polars Series  .
        #   - Replace NaN values with the minimum value of the dtype, so NaNs are
        #     ranked last.
        #   - Use the `rank` method from Polars to assign ranks, using the "random"
        #     method to break ties so that regions/genes with the same score get a
        #     random ranking in the range of their scores instead of depending on
        #     the order in which they appear in the input array.
        #   - Subtract 1 from the ranks to make them zero-based.
        return (
            pl.Series(scores_with_ties_for_motif_or_track_numpy)
            .fill_nan(float(np.finfo(scores_with_ties_for_motif_or_track_numpy.dtype).min))
            .rank(method="random", descending=True, seed=seed)
            - 1
        ).to_numpy()

    n_regions, n_cells = imputed_accessibility.shape
    # Create zeroed imputed object rankings database.
    rankings = np.zeros((n_regions, n_cells), dtype=np.int32)

    # Rank all scores per motif/track and assign a random ranking in range for regions/genes with the same score.
    if method == "numpy":
        for cell_idx in range(n_cells):
            rankings[:, cell_idx] = (
                rank_scores_and_assign_random_ranking_in_range_for_ties_with_numpy(
                    imputed_accessibility[:, cell_idx]
                )
            )
    else:
        for cell_idx in range(n_cells):
            rankings[:, cell_idx] = (
                rank_scores_and_assign_random_ranking_in_range_for_ties_with_polars(
                    imputed_accessibility[:, cell_idx],
                    seed=rng.integers(2**32 - 1),
                )
            )

    return rankings
