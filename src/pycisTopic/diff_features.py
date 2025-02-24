from __future__ import annotations

import logging
import math
import sys
from typing import TYPE_CHECKING, Self

import matplotlib
import matplotlib.pyplot as plt
import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import scipy.sparse as sparse

if TYPE_CHECKING:
    from pycisTopic.cistopic_class import CistopicObject

from pycisTopic.utils import (
    get_nonzero_row_indices,
    get_position_index,
    non_zero_rows,
    prepare_tag_cells,
    subset_list,
)


class CistopicImputedFeatures:
    """
    cisTopic imputation data class.

    :class:`CistopicImputedFeatures` contains the cell by features matrices (stored at :attr:`mtx`, with features being eithere regions or genes ),
    cell names :attr:`cell_names` and feature names :attr:`feature_names`.

    Attributes
    ----------
    mtx: sparse.csr_matrix
        A matrix containing imputed values.
    cell_names: list
        A list containing cell names.
    feature_names: list
        A list containing feature names.
    project: str
        Name of the cisTopic imputation project.
    """

    def __init__(
        self,
        imputed_acc: sparse.csr_matrix,
        feature_names: list[str],
        cell_names: list[str],
        project: str,
    ):
        self.mtx = imputed_acc
        self.feature_names = feature_names
        self.cell_names = cell_names
        self.project = project

    def __str__(self):
        descr = f"CistopicImputedFeatures from project {self.project} with nCells × nFeatures = {len(self.cell_names)} × {len(self.feature_names)}"
        return descr

    def subset(
        self,
        cells: list[str] | None = None,
        features: list[str] | None = None,
        copy: bool = False,
        split_pattern: str = "___",
    ):
        """
        Subset cells and/or regions from :class:`CistopicImputedFeatures`.

        Parameters
        ----------
        cells: list, optional
            A list containing the names of the cells to keep.
        features: list, optional
            A list containing the names of the features to keep.
        copy: bool, optional
            Whether changes should be done on the input :class:`CistopicObject` or a new object should be returned
        split_pattern: str
            Pattern to split cell barcode from sample id. Default: ___

        Return
        ------
        CistopicImputedFeatures
            A :class:`CistopicImputedFeatures` containing the selected cells and/or features.

        """

        # Create cisTopic logger
        level = logging.INFO
        log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level=level, format=log_format, handlers=handlers)
        log = logging.getLogger("cisTopic")

        mtx = self.mtx
        cell_names = self.cell_names
        feature_names = self.feature_names

        if cells is not None:
            try:
                cells_index = get_position_index(cells, self.cell_names)
            except BaseException:
                try:
                    cells_index = get_position_index(
                        cells, prepare_tag_cells(self.cell_names, split_pattern)
                    )
                except BaseException:
                    log.error(
                        "None of the given cells is contained in this cisTopic object!"
                    )
            mtx = mtx[:, cells_index]
            cell_names = subset_list(cell_names, cells_index)

        if features is not None:
            features_index = get_position_index(features, feature_names)
            mtx = mtx[features_index, :]
            feature_names = subset_list(feature_names, features_index)

        features_index = non_zero_rows(mtx)
        mtx = mtx[features_index, :]
        feature_names = subset_list(feature_names, features_index)

        if copy is True:
            return CistopicImputedFeatures(mtx, feature_names, cell_names, self.project)
        else:
            self.mtx = mtx
            self.cell_names = cell_names
            self.feature_names = feature_names

    def merge(
        self,
        cistopic_imputed_features_list: list[Self],
        project: str = "cisTopic_impute_merge",
        copy: bool = False,
    ):
        """
        Merge a list of :class:`CistopicImputedFeatures` to the input :class:`CistopicImputedFeatures`. Reference coordinates (for regions) must be the same between the objects.

        Parameters
        ----------
        cistopic_imputed_features_list: list
            A list containing one or more :class:`CistopicImputedFeatures` to merge.
        project: str, optional
            Name of the cisTopic imputation project.
        copy: bool, optional
            Whether changes should be done on the input :class:`CistopicObject` or a new object should be returned
        Return
        ------
        CistopicImputedFeatures
            A combined :class:`CistopicImputedFeatures`.

        """
        # Create cisTopic logger
        level = logging.INFO
        log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level=level, format=log_format, handlers=handlers)
        log = logging.getLogger("cisTopic")

        cistopic_imputed_features_list.insert(0, self)
        mtx_list = [x.mtx for x in cistopic_imputed_features_list]
        feature_names_list = [x.feature_names for x in cistopic_imputed_features_list]
        cell_names_list = [x.cell_names for x in cistopic_imputed_features_list]

        mtx = mtx_list[0]
        feature_names = feature_names_list[0]
        cell_names = cell_names_list[0]

        for i in range(1, len(feature_names_list)):
            feature_names_to_add = feature_names_list[i]
            mtx_to_add = mtx_list[i]
            cell_names_to_add = cell_names_list[i]
            cell_names = cell_names + cell_names_to_add

            common_features = list(set(feature_names) & set(feature_names_to_add))
            diff_features = list(set(feature_names) ^ set(feature_names_to_add))

            common_index_fm = get_position_index(common_features, feature_names)
            common_index_fm_to_add = get_position_index(
                common_features, feature_names_to_add
            )
            if sparse.issparse(mtx):
                mtx_common = sparse.hstack(
                    [
                        mtx[common_index_fm,],
                        mtx_to_add[common_index_fm_to_add,],
                    ],
                    format="csr",
                )
            else:
                mtx_common = np.hstack(
                    [
                        mtx[common_index_fm,],
                        mtx_to_add[common_index_fm_to_add,],
                    ]
                )
            if len(diff_features) > 0:
                diff_features_1 = list(
                    np.setdiff1d(feature_names, feature_names_to_add)
                )
                diff_index_fm_1 = get_position_index(diff_features_1, feature_names)
                if sparse.issparse(mtx):
                    mtx_diff_1 = sparse.hstack(
                        [
                            mtx[diff_index_fm_1,],
                            np.zeros((len(diff_features_1), mtx_to_add.shape[1])),
                        ],
                        format="csr",
                    )
                else:
                    mtx_diff_1 = np.hstack(
                        [
                            mtx[diff_index_fm_1,],
                            np.zeros((len(diff_features_1), mtx_to_add.shape[1])),
                        ]
                    )

                diff_features_2 = list(
                    np.setdiff1d(feature_names_to_add, feature_names)
                )
                diff_index_fm_2 = get_position_index(
                    diff_features_2, feature_names_to_add
                )
                if sparse.issparse(mtx):
                    mtx_diff_2 = sparse.hstack(
                        [
                            np.zeros((len(diff_features_2), mtx.shape[1])),
                            mtx_to_add[diff_index_fm_2,],
                        ],
                        format="csr",
                    )
                    mtx = sparse.vstack(
                        [mtx_common, mtx_diff_1, mtx_diff_2], format="csr"
                    )
                else:
                    mtx_diff_2 = np.hstack(
                        [
                            np.zeros((len(diff_features_2), mtx.shape[1])),
                            mtx_to_add[diff_index_fm_2,],
                        ]
                    )
                    mtx = np.vstack([mtx_common, mtx_diff_1, mtx_diff_2])

                feature_names = common_features + diff_features_1 + diff_features_2
            else:
                mtx = mtx_common
                feature_names = common_features

        if copy is True:
            return CistopicImputedFeatures(mtx, feature_names, cell_names, project)
        else:
            self.mtx = mtx
            self.cell_names = cell_names
            self.feature_names = feature_names
            self.project = project

    def make_rankings(self, seed=123):
        """
        A function to generate rankings per cell based on the imputed accessibility scores per region.

        Parameters
        ----------
        seed: int, optional
            Random seed to ensure reproducibility of the rankings when there are ties
        Return
        ------
           CistopicImputedFeatures
            A :class:`CistopicImputedFeatures` containing with ranking values rather than scores.

        """
        # Initialize random number generator, for handling ties
        rng = np.random.default_rng(seed=seed)

        # Function to make rankings per array
        def rank_scores_and_assign_random_ranking_in_range_for_ties(
            scores_with_ties_for_motif_or_track_numpy: np.ndarray,
        ) -> np.ndarray:
            #
            # Create random permutation so tied scores will have a different ranking each time.
            random_permutations_to_break_ties_numpy = rng.permutation(
                scores_with_ties_for_motif_or_track_numpy.shape[0]
            )
            ranking_with_broken_ties_for_motif_or_track_numpy = (
                random_permutations_to_break_ties_numpy[
                    (-scores_with_ties_for_motif_or_track_numpy)[
                        random_permutations_to_break_ties_numpy
                    ].argsort()
                ]
                .argsort()
                .astype(imputed_acc_obj_ranking_db_dtype)
            )

            return ranking_with_broken_ties_for_motif_or_track_numpy

        # Create zeroed imputed object rankings database.
        imputed_acc_ranking = CistopicImputedFeatures(
            np.zeros((len(self.feature_names), len(self.cell_names)), dtype=np.int32),
            self.feature_names,
            self.cell_names,
            self.project,
        )

        # Get dtype of the scores
        imputed_acc_obj_ranking_db_dtype = "uint32"

        # Convert to csc
        if sparse.issparse(self.mtx):
            mtx = self.mtx.tocsc()
        else:
            mtx = self.mtx

        # Rank all scores per motif/track and assign a random ranking in range for regions/genes with the same score.
        for col_idx in range(len(imputed_acc_ranking.cell_names)):
            imputed_acc_ranking.mtx[:, col_idx] = (
                rank_scores_and_assign_random_ranking_in_range_for_ties(
                    mtx[:, col_idx].toarray().flatten()
                )
            )

        return imputed_acc_ranking


@numba.njit(parallel=False, error_model="numpy")
def calculate_partial_imputed_acc_sums_per_cell_for_requested_regions(
    imputed_acc_chunk: npt.NDArray[np.float32],
    region_idx_to_keep_chunk: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    """
    Calculate (partial) sum of imputed accessibility for the whole (partial) cell column for each cell.

    Calculate (partial) sum of imputed accessibility for the whole (partial) cell
    column for each cell using all regions for which the whole row is not completely
    zero (taken from `region_idx_to_keep_chunk`)
    (regions which are never accessible in any cell).

    Returns
    -------
    per_cell_imputed_acc_sums_partial

    """
    # To get the sum of imputed accessibility for the whole (partial) cell
    # column, the values for each imputed_acc_chunk need to be summed together.
    # To avoid problems with precision in summing a lot of values, summing is done
    # with float64 values.
    #
    # This function is a memory optimized version of:
    #
    #     np.sum(
    #         imputed_acc_chunk[region_idx_to_keep_chunk],
    #         axis=0,
    #         dtype=np.float64,
    #     )
    #
    # The above code triggers a big temporarily memory allocation, when the input
    # matrix is not contiguous in memory (which happens due subsetting
    # `imputed_acc_chunk` with specific region indexes).

    n_cells = imputed_acc_chunk.shape[1]

    # Preallocate array for whole (partial) imputed accessibility per cell for each
    # cell.
    per_cell_imputed_acc_sums_partial = np.zeros((n_cells,), dtype=np.float64)

    # Get each region index of regions to keep and retrieve imputed accessibility
    # per region and compute (partial) sum of imputed accessibility for the whole
    # (partial) cell for each cell. Use float64 to avoid problems with precision
    # when summing a lot of values.
    for region_idx in region_idx_to_keep_chunk:
        per_cell_imputed_acc_sums_partial += imputed_acc_chunk[region_idx].astype(
            np.float64
        )
    return per_cell_imputed_acc_sums_partial


@numba.njit(parallel=True, error_model="numpy")
def calculate_per_region_mean_and_dispersion_on_normalized_imputed_acc_chunk(
    normalized_imputed_acc_chunk: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate mean and dispersion on normalized imputed accessibility for each region.

    Parameters
    ----------
    normalized_imputed_acc_chunk
        Normalized imputed accessibility matrix (regions_chunk_size x n_cells).

    Returns
    -------
    Mean and dispersion of normalized imputed accessibility per region for all regions:
    (per_region_means_on_normalized_imputed_acc_chunk,
     per_region_dispersions_on_normalized_imputed_acc_chunk)

    """
    # Memory and speed optimized version of:
    #
    #     per_region_means_on_normalized_imputed_acc_chunk = normalized_imputed_acc_chunk.mean(
    #         axis=1, dtype=np.float64
    #     )
    #     per_region_variances_on_normalized_imputed_acc_chunk = normalized_imputed_acc_chunk.var(
    #         axis=1, dtype=np.float64
    #     )
    #     per_region_means_on_normalized_imputed_acc_chunk[
    #         per_region_means_on_normalized_imputed_acc_chunk == 0
    #     ] = 1e-12
    #     per_region_dispersions_on_normalized_imputed_acc_chunk = (
    #         per_region_variances_on_normalized_imputed_acc_chunk
    #         / per_region_means_on_normalized_imputed_acc_chunk
    #     )
    #     per_region_dispersions_on_normalized_imputed_acc_chunk[
    #         per_region_dispersions_on_normalized_imputed_acc_chunk == 0
    #     ] = np.nan
    #     np.log(
    #         per_region_dispersions_on_normalized_imputed_acc_chunk,
    #         out=per_region_dispersions_on_normalized_imputed_acc_chunk,
    #     )
    #     return (
    #         per_region_means_on_normalized_imputed_acc_chunk,
    #         per_region_dispersions_on_normalized_imputed_acc_chunk,
    #     )

    n_regions_in_chunk = normalized_imputed_acc_chunk.shape[0]
    n_cells = normalized_imputed_acc_chunk.shape[1]

    # Preallocate arrays for mean and dispersion of normalized imputed accessibility
    # per region for each region in the current chunk.
    per_region_means_on_normalized_imputed_acc_chunk = np.empty(
        (n_regions_in_chunk),
        dtype=np.float64,
    )
    per_region_dispersions_on_normalized_imputed_acc_chunk = np.empty(
        (n_regions_in_chunk),
        dtype=np.float64,
    )

    for region_idx in numba.prange(n_regions_in_chunk):
        # Get normalized imputed accessibility for current region.
        normalized_impute_acc_for_region = normalized_imputed_acc_chunk[
            region_idx
        ].astype(np.float64)

        # Calculate mean of normalized imputed accessibility for current region.
        mean = np.float64(0.0)
        for cell_idx in range(n_cells):
            mean += normalized_impute_acc_for_region[cell_idx]
        mean /= n_cells

        # Calculate variance of normalized imputed accessibility for current region.
        variance = np.float64(0.0)
        for cell_idx in range(n_cells):
            variance += (normalized_impute_acc_for_region[cell_idx] - mean) ** 2
        variance /= n_cells

        # Calculate dispersion of normalized imputed accessibility for current region.
        mean = mean if mean != 0.0 else 1e-12
        dispersion = variance / mean
        dispersion = np.log(dispersion) if dispersion != 0.0 else np.nan

        per_region_means_on_normalized_imputed_acc_chunk[region_idx] = mean
        per_region_dispersions_on_normalized_imputed_acc_chunk[region_idx] = dispersion
    return (
        per_region_means_on_normalized_imputed_acc_chunk,
        per_region_dispersions_on_normalized_imputed_acc_chunk,
    )


def calculate_per_region_mean_and_dispersion_on_normalized_imputed_acc(
    region_topic: npt.NDArray[np.float32],
    cell_topic: npt.NDArray[np.float32],
    region_names: list[str],
    scale_factor1: int = 10**6,
    scale_factor2: int = 10**4,
    regions_chunk_size: int = 20000,
) -> tuple[list[str], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate per region mean and dispersion on normalized imputed accessibility in chunks of `regions_chunk_size`.

    High level overview of the function:
      - Calculate imputed accessibility: `region_topic @ cell_topic`.
      - Scale imputed accessibility is scaled by `scale_factor1` to create a "count"
        matrix.
      - Only keep integer part of the scaled imputed accessibility.
      - Remove all regions for which scaled imputed accessibility was 0 in all cells.
      - Calculate total imputed accessibility per cell by summing the imputed
        accessibility for each region.
      - Calculate normalized imputed accessibility for regions for which scaled imputed
        accessibility was not 0 in all cells, by dividing the scaled imputed
        accessibility by the total imputed accessibility per cell and multiplying by
        scale_factor2 and taking the log(x + 1) of the result.
      - Calculate mean and dispersion of normalized imputed accessibility per region.

    Parameters
    ----------
    region_topic
        Region topic matrix (regions x topics).
    cell_topic
        Cell topic matrix (topic x cells).
    region_names
        List of all region names.
    scale_factor1
        Multiply imputed accessiblitity by this scale factor to create a "count" matrix.
        This will remove noise by putting very small values to zero.
        Default: 10**6.
    scale_factor2
        Scale factor used to normalize scaled imputed accessibility, similar to
        RNA-seq normalization.
        Divide scaled (`scale_factor1`) imputed accessibility by total imputed
        accessibility per cell, multiply by `scale_factor2`, add 1 and take logarithm.
        Default: 10**4.
    regions_chunk_size
        Regions chunk size used (number of regions for which imputed accessibility is
        calculated at the same time).

    Returns
    -------
    Numpy array with imputed accessibility for each region and a list of region
    names (some regions for which all row values were 0 are filtered out).
    (imputed_acc, region_names_to_keep)

    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    output_regions_chunk_end = 0
    n_regions = region_topic.shape[0]
    n_cells = cell_topic.shape[1]
    # n_topics = region_topic.shape[1]

    region_idx_to_keep = np.zeros((n_regions,), dtype=np.intp)

    impute_acc_per_cell_sum = np.zeros((n_cells,), dtype=np.float64)

    log.info("Calculate total imputed accessibility per cell.")
    region_topic = np.asarray(region_topic, dtype=np.float32)
    cell_topic = np.asarray(cell_topic, dtype=np.float32)

    log.info(
        f"Allocate {(regions_chunk_size * n_cells * 4 / 1024 ** 3):.3f} GiB of RAM for "
        f"calculating (partial) imputed accessibility per cell for ({n_cells}) cells "
        f"for chunk of {regions_chunk_size} regions."
    )
    # Preallocate imputed accessibility chunk array (regions_chunk_size x n_cells) so
    # it can be reused in each loop iteration (except for the last one as that one will
    # likely be smaller).
    imputed_acc_chunk = np.empty((regions_chunk_size, n_cells), dtype=np.float32)

    # Calculate total imputed accessibility per cell.
    for input_regions_chunk_start in range(0, n_regions, regions_chunk_size):
        input_regions_chunk_end = input_regions_chunk_start + regions_chunk_size

        # Set correct output chunk start position.
        output_regions_chunk_start = output_regions_chunk_end

        log.info(
            "Calculate (partial) imputed accessibility per cell for regions "
            f"{input_regions_chunk_start}-{input_regions_chunk_end} (out of {n_regions})."
        )

        # Get the current chunk of regions.
        topic_region_chunk = region_topic[
            input_regions_chunk_start : input_regions_chunk_start + regions_chunk_size
        ]
        current_regions_chunk_size = topic_region_chunk.shape[0]

        if current_regions_chunk_size < regions_chunk_size:
            log.info(
                f"Allocate {(current_regions_chunk_size * n_cells * 4 / 1024 ** 3):.3f} "
                "GiB of RAM for calculating (partial) imputed accessibility per cell "
                f"for ({n_cells}) cells for chunk of {current_regions_chunk_size} "
                "regions."
            )
            # Reallocate imputed_acc_chunk to the correct size.
            del imputed_acc_chunk
            imputed_acc_chunk = np.empty(
                (current_regions_chunk_size, n_cells), dtype=np.float32
            )

        log.info(
            "  - Calculate imputed accessibility for the current chunk of regions."
        )
        # Calculate imputed accessibility for the current chunk of regions.
        np.matmul(topic_region_chunk, cell_topic, out=imputed_acc_chunk)

        log.info('  - Scale imputed accessibility matrix chunk ("count" matrix).')
        # Scale imputed accessibility matrix chunk ("count" matrix).
        imputed_acc_chunk *= np.float32(scale_factor1)

        log.info("  - Only keep integer part.")
        # Only keep integer part.
        # This will convert very small values (< 1.0) to zero and removes noise.
        np.floor(imputed_acc_chunk, out=imputed_acc_chunk)

        log.info("  - Get non-zero regions.")
        # Get all region index positions of the matrix for which the whole row is not
        # completely zero (regions which are never accessible in any cell).
        region_idx_to_keep_chunk = get_nonzero_row_indices(imputed_acc_chunk)

        # Set correct output chunk end position by taking into account
        # that rows with all zeros will be filtered out.
        output_regions_chunk_end = output_regions_chunk_start + len(
            region_idx_to_keep_chunk
        )

        # Get all region indexes that need to be kept from this chunk and
        # assign them to the correct positions in the full region_idx_to_keep array.
        region_idx_to_keep[output_regions_chunk_start:output_regions_chunk_end] = (
            region_idx_to_keep_chunk + input_regions_chunk_start
        )

        log.info(
            "  - Calculate (partial) sum of imputed accessibility for the whole (partial) cell column."
        )
        # Calculate (partial) sum of imputed accessibility for the whole (partial) cell
        # column for each cell using all regions for which the whole row is not
        # completely zero (regions which are never accessible in any cell).
        # To avoid problems with precision in summing multiple chunks later, float64 are
        # used.
        cells_imputed_acc_sums_partial = (
            calculate_partial_imputed_acc_sums_per_cell_for_requested_regions(
                imputed_acc_chunk, region_idx_to_keep_chunk
            )
        )

        impute_acc_per_cell_sum += cells_imputed_acc_sums_partial

    del imputed_acc_chunk

    n_regions_to_keep = output_regions_chunk_end

    log.info(f"Keeping {n_regions_to_keep} of {n_regions} (non_zero) regions.")

    # Only retain that part of region_idx_to_keep that was actually
    # filled in.
    region_idx_to_keep = region_idx_to_keep[:output_regions_chunk_end]

    # Subset region_topic to regions we want to keep.
    region_topic = region_topic[region_idx_to_keep]

    # Get all region names that need to be kept.
    region_names_to_keep = subset_list(
        region_names,
        region_idx_to_keep,
    )

    # Preallocate arrays for mean and dispersion of normalized imputed accessibility per
    # region.
    per_region_means_on_normalized_imputed_acc = np.empty(
        (n_regions_to_keep,),
        dtype=np.float64,
    )
    per_region_dispersions_on_normalized_imputed_acc = np.empty(
        (n_regions_to_keep,),
        dtype=np.float64,
    )

    log.info(
        f"Scale total imputed accessibility per cell by dividing by {scale_factor2}."
    )
    # Scale total imputed accessibility per cell by dividing by scale_factor2.
    # The sum was calculated in float64 to avoid problems with precision, but after
    # scaling it can be converted back to float32 to avoid converting
    # normalized_imputed_acc_chunk in each for loop iteration to float64, causing a
    # big allocation (if there are a lot of cells).
    impute_acc_per_cell_sum_scaled = (impute_acc_per_cell_sum / scale_factor2).astype(
        np.float32
    )

    log.info(
        f"Allocate {(regions_chunk_size * n_cells * 4 / 1024 ** 3):.3f} GiB of RAM "
        "for calculating normalized imputed accessibility per region for chunk of "
        f"{regions_chunk_size} regions."
    )
    # Preallocate normalized imputed accessibility chunk array
    # (regions_chunk_size x n_cells) so it can be reused in each loop iteration
    # (except for the last one as that one will likely be smaller).
    normalized_imputed_acc_chunk = np.empty(
        (regions_chunk_size, n_cells),
        dtype=np.float32,
    )

    output_regions_chunk_end = 0

    log.info(
        "Calculate mean and dispersion of normalized imputed accessibility per region."
    )

    # Calculate mean and dispersion of normalized imputed accessibility per region.
    for input_regions_chunk_start in range(0, n_regions_to_keep, regions_chunk_size):
        input_regions_chunk_end = input_regions_chunk_start + regions_chunk_size

        # Set correct output chunk start position.
        output_regions_chunk_start = output_regions_chunk_end

        log.info(
            "Calculate mean and dispersion of normalized imputed accessibility for regions "
            f"{input_regions_chunk_start}-{input_regions_chunk_end} (out of {n_regions_to_keep})."
        )

        topic_region_chunk = region_topic[
            input_regions_chunk_start : input_regions_chunk_start + regions_chunk_size
        ]
        current_regions_chunk_size = topic_region_chunk.shape[0]

        if current_regions_chunk_size < regions_chunk_size:
            log.info(
                f"Allocate {(current_regions_chunk_size * n_cells * 4 / 1024 ** 3):.3f} "
                f"GiB of RAM for calculating normalized imputed accessibility per "
                f"region for chunk of {current_regions_chunk_size} regions."
            )
            # Reallocate imputed_acc_chunk to the correct size.
            del normalized_imputed_acc_chunk
            normalized_imputed_acc_chunk = np.empty(
                (current_regions_chunk_size, n_cells),
                dtype=np.float32,
            )

        log.info(
            "  - Calculate imputed accessibility for the current chunk of regions."
        )
        # Calculate imputed accessibility for the current chunk of regions.
        np.matmul(topic_region_chunk, cell_topic, out=normalized_imputed_acc_chunk)

        log.info('  - Scale imputed accessibility matrix chunk ("count" matrix).')
        # Scale imputed accessibility matrix chunk ("count" matrix).
        normalized_imputed_acc_chunk *= np.float32(scale_factor1)

        log.info("  - Only keep integer part.")
        # Only keep integer part.
        # This will convert very small values (< 1.0) to zero and removes noise.
        np.floor(normalized_imputed_acc_chunk, out=normalized_imputed_acc_chunk)

        # Set correct output chunk end position by taking into account
        # that rows with all zeros will be filtered out.
        output_regions_chunk_end = output_regions_chunk_start + regions_chunk_size

        log.info(
            "  - Normalize imputed accessibility by dividing by the total imputed "
            f"accessibility per cell and multiply by {scale_factor2}."
        )
        # Normalize imputed accessibility by dividing by the total imputed
        # accessibility per cell and multiply by 10^4.
        normalized_imputed_acc_chunk /= impute_acc_per_cell_sum_scaled

        log.info("  - Add pseudocount of 1 and apply log normalization.")
        # Add pseudocount of 1 and apply log normalization.
        np.log1p(
            normalized_imputed_acc_chunk,
            out=normalized_imputed_acc_chunk,
        )

        log.info(
            "  - Calculate mean and dispersion of imputed accessibility per region."
        )
        (
            per_region_means_on_normalized_imputed_acc_chunk,
            per_region_dispersions_on_normalized_imputed_acc_chunk,
        ) = calculate_per_region_mean_and_dispersion_on_normalized_imputed_acc_chunk(
            normalized_imputed_acc_chunk
        )

        per_region_means_on_normalized_imputed_acc[
            output_regions_chunk_start:output_regions_chunk_end
        ] = per_region_means_on_normalized_imputed_acc_chunk
        per_region_dispersions_on_normalized_imputed_acc[
            output_regions_chunk_start:output_regions_chunk_end
        ] = per_region_dispersions_on_normalized_imputed_acc_chunk

    del normalized_imputed_acc_chunk

    log.info(
        "Finished Calculating mean and dispersion of imputed accessibility per region."
    )

    return (
        region_names_to_keep,
        per_region_means_on_normalized_imputed_acc,
        per_region_dispersions_on_normalized_imputed_acc,
    )


def impute_accessibility(
    cistopic_obj: CistopicObject,
    selected_cells: list[str] | None = None,
    selected_regions: list[str] | None = None,
    scale_factor: int = 10**6,
    chunk_size: int = 20000,
    project: str = "cisTopic_Impute",
):
    """
    Impute region accessibility.

    Parameters
    ----------
    cistopic_obj: `class::CistopicObject`
        A cisTopic object with a model in `class::CistopicObject.selected_model`.
    selected_cells: list, optional
        A list with selected cells to impute accessibility for. Default: None
    selected_regions: list, optional
        A list with selected regions to impute accessibility for. Default: None
    scale_factor: int, optional
        A number to multiply the imputed values for. This is useful to convert low
        probabilities to 0, making the matrix more sparse. Default: 10**6.
    chunk_size:
        Chunk size used (number of regions for which imputed accessibility is
        calculated at the same time).
    project: str, optional
        Name of the cisTopic imputation project. Default: ``cisTopic_impute``.

    Return
    ------
    CistopicImputedFeatures

    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    model = cistopic_obj.selected_model
    cell_names = cistopic_obj.cell_names
    cell_topic = model.cell_topic.loc[:, cell_names]
    region_names = cistopic_obj.region_names
    topic_region = model.topic_region.loc[region_names]
    if selected_cells is not None:
        cell_topic = cell_topic.loc[:, selected_cells]
        cell_names = selected_cells
    if selected_regions is not None:
        topic_region = topic_region.loc[selected_regions]
        region_names = selected_regions
    # Convert cell_topic and region_topic 2d arrays to np.float32 so
    # multiplying them uses 4 times less memory than with np.float64
    cell_topic = cell_topic.to_numpy().astype(np.float32)
    topic_region = topic_region.to_numpy().astype(np.float32)

    log.info("Imputing region accessibility")

    def calculate_imputed_accessibility(
        topic_region: np.ndarray,
        cell_topic: np.ndarray,
        region_names: list,
        scale_factor: int,
        chunk_size: int,
    ) -> tuple[np.ndarray, list]:
        """
        Calculate imputed accessibility in chunks of chunk_size.

        Parameters
        ----------
        topic_region:
            Topic region matrix (regions x topics).
        cell_topic:
            Cell topic matrix (topic x cells).
        region_names:
            List of all region names.
        scale_factor:
            A number to multiply the imputed values for. This is useful to convert
            low probabilities to 0, making the matrix more sparse. Default: 10**6.
        chunk_size:
            Chunk size used (number of regions for which imputed accessibility is
            calculated at the same time).

        Returns
        -------
        Numpy array with imputed accessibility for each region and a list of region
        names (some regions for which all row values were 0 are filtered out).
        (imputed_acc, region_names_to_keep)

        """
        output_chunk_end = 0
        region_names_to_keep = []

        # Create empty imputed accessibility matrix which will be filled in chunks.
        imputed_acc = np.empty(
            (topic_region.shape[0], cell_topic.shape[1]),
            dtype=(
                np.int32
                if isinstance(scale_factor, int) and scale_factor != 1
                else np.float32
            ),
        )

        for input_chunk_start in range(0, topic_region.shape[0], chunk_size):
            input_chunk_end = input_chunk_start + chunk_size

            # Set correct output chunk start position.
            output_chunk_start = output_chunk_end

            log.info(
                "Impute region accessibility for regions "
                f"{input_chunk_start}-{input_chunk_end}"
            )
            topic_region_chunk = topic_region[
                input_chunk_start : input_chunk_start + chunk_size
            ]
            imputed_acc_chunk = topic_region_chunk @ cell_topic

            if isinstance(scale_factor, int) and scale_factor != 1:
                # Scale imputed accessibility matrix chunk.
                imputed_acc_chunk *= np.float32(scale_factor)

                # Convert from float32 to int32.
                # This will convert very small values to zero.
                imputed_acc_chunk = imputed_acc_chunk.astype(np.int32)

            # Get all region index positions of the matrix for which
            # the whole row is not completely zero.
            region_idx_to_keep_chunk = non_zero_rows(imputed_acc_chunk)

            # Get all region names that need to be kept for this chunk
            region_names_to_keep.extend(
                subset_list(
                    region_names[input_chunk_start:input_chunk_end],
                    region_idx_to_keep_chunk,
                )
            )

            # Set correct output chunk end position by taking into account
            # that rows with all zeros will be filtered out.
            output_chunk_end = output_chunk_start + len(region_idx_to_keep_chunk)

            # Convert from float32 to int32 and fill in the values in the full
            # imputed accessibility matrix.
            imputed_acc[output_chunk_start:output_chunk_end, :] = imputed_acc_chunk[
                region_idx_to_keep_chunk
            ]

        # Only retain that part of the imputed accessibility matrix that was actually
        # filled in.
        imputed_acc = imputed_acc[0:output_chunk_end]

        return imputed_acc, region_names_to_keep

    # Fill `imputed_acc` matrix in chunks of 20000.
    imputed_acc, region_names_to_keep = calculate_imputed_accessibility(
        topic_region=topic_region,
        cell_topic=cell_topic,
        region_names=region_names,
        scale_factor=scale_factor,
        chunk_size=chunk_size,
    )

    imputed_acc_obj = CistopicImputedFeatures(
        imputed_acc, region_names_to_keep, cell_names, project
    )

    log.info("Done!")
    return imputed_acc_obj


def find_highly_variable_regions(
    regions: list[str],
    per_region_means_on_normalized_imputed_acc: npt.NDArray[np.float64],
    per_region_dispersions_on_normalized_imputed_acc: npt.NDArray[np.float64],
    min_disp: float = 0.05,
    min_mean: float = 0.0125,
    max_disp: float = float("inf"),
    max_mean: float = 3,
    n_bins: int = 20,
    n_top_features: int | None = None,
    plot: bool | str | None = True,
):
    """
    Find highly variable regions.

    Find highly variable regions by using output of
    `calculate_per_region_mean_and_dispersion_on_normalized_imputed_acc`
    as input:
      - `regions`
      - `per_region_means_on_normalized_imputed_acc`
      - `per_region_dispersions_on_normalized_imputed_acc`

    Parameters
    ----------
    regions
        List of region names.
    per_region_means_on_normalized_imputed_acc
        Mean of normalized imputed accessibility per region.
    per_region_dispersions_on_normalized_imputed_acc
        Dispersion of normalized imputed accessibility per region.
    min_disp
        Minimum dispersion value for a feature to be selected.
        Default: 0.05
    min_mean
        Minimum mean value for a feature to be selected.
        Default: 0.0125
    max_disp
        Maximum dispersion value for a feature to be selected.
        Default: float("inf")
    max_mean
        Maximum mean value for a feature to be selected.
        Default: 3
    n_bins
        Number of bins for binning the mean gene expression.
        Normalization is done with respect to each bin.
        Default: 20
    n_top_features
        Number of highly-variable regions to keep.
        If specified, dispersion and mean thresholds will be ignored.
        Default: None
    plot
        Whether to plot dispersion versus mean values.
        If `True`, plot will be shown.
        if `str`, plot will be saved to the specified path.
        If `False` or `None, plot will not be shown.
        Default: True

    Return
    ------
    List with highly variable regions.

    Examples
    --------
    Get mean and dispersion of normalized imputed accessibility per region.
    >>> (
    ...     region_names_to_keep,
    ...     per_region_means_on_normalized_imputed_acc,
    ...     per_region_dispersions_on_normalized_imputed_acc,
    ... ) = calculate_per_region_mean_and_dispersion_on_normalized_imputed_acc(
    ...     region_topic=region_topic,
    ...     cell_topic=cell_topic,
    ...     region_names=region_names,
    ...     scale_factor1 = 10**6,
    ...     scale_factor2 = 10**4,
    ...     regions_chunk_size=20000,
    ... )

    Find highly variable regions.
    >>> find_highly_variable_regions(
    ...    regions=region_names_to_keep,
    ...    per_region_means_on_normalized_imputed_acc=per_region_means_on_normalized_imputed_acc,
    ...    per_region_dispersions_on_normalized_imputed_acc=per_region_dispersions_on_normalized_imputed_acc,
    ...    min_disp = 0.05,
    ...    min_mean = 0.0125,
    ...    max_disp = float("inf"),
    ...    max_mean = 3,
    ...    n_bins = 20,
    ...    n_top_features = None,
    ...    plot = True,
    ... )

    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    df = pd.DataFrame()
    df["means"] = np.asarray(
        per_region_means_on_normalized_imputed_acc,
        dtype=np.float64,
    )
    df["dispersions"] = np.asarray(
        per_region_dispersions_on_normalized_imputed_acc,
        dtype=np.float64,
    )
    df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
    disp_grouped = df.groupby("mean_bin", observed=False)["dispersions"]
    disp_mean_bin = disp_grouped.mean()
    disp_std_bin = disp_grouped.std(ddof=1)
    # Retrieve those regions that have nan std, these are the ones where
    # only a single gene fell in the bin and implicitly set them to have
    # a normalized dispersion of 1.
    one_feature_per_bin = disp_std_bin.isnull()
    feature_indices = np.where(one_feature_per_bin[df["mean_bin"].values])[0].tolist()

    if len(feature_indices) > 0:
        log.debug(
            f"Feature indices {feature_indices} fell into a single bin: their "
            "normalized dispersion was set to 1.\n    "
            "Decreasing `n_bins` will likely avoid this effect."
        )

    disp_std_bin[one_feature_per_bin.values] = disp_mean_bin[
        one_feature_per_bin.values
    ].values
    disp_mean_bin[one_feature_per_bin.values] = 0
    # Normalize
    df["dispersions_norm"] = (
        df["dispersions"].values  # use values here as index differs
        - disp_mean_bin[df["mean_bin"].values].values
    ) / disp_std_bin[df["mean_bin"].values].values

    dispersion_norm = df["dispersions_norm"].values.astype("float32")

    if n_top_features is not None:
        dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
        dispersion_norm[::-1].sort()
        disp_cut_off = dispersion_norm[n_top_features - 1]
        feature_subset = np.nan_to_num(df["dispersions_norm"].values) >= disp_cut_off
        log.debug(
            f"the {n_top_features} top regions correspond to a "
            f"normalized dispersion cutoff of {disp_cut_off}"
        )
    else:
        dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
        feature_subset = np.logical_and.reduce(
            (
                per_region_means_on_normalized_imputed_acc > min_mean,
                per_region_means_on_normalized_imputed_acc < max_mean,
                dispersion_norm > min_disp,
                dispersion_norm < max_disp,
            )
        )

    df["highly_variable"] = feature_subset
    highly_variable_regions = [
        regions[i] for i in df[df.highly_variable].index.to_list()
    ]

    if plot:
        fig = plt.figure()
        matplotlib.rcParams["agg.path.chunksize"] = 10000
        plt.scatter(
            df["means"], df["dispersions_norm"], c=feature_subset, s=10, alpha=0.1
        )
        plt.xlabel("Mean measurement of regions")
        plt.ylabel("Normalized dispersion of the regions")
        if isinstance(plot, str):
            fig.savefig(plot)
        plt.show()

    log.info("Done!")
    return highly_variable_regions


def get_marker_regions_for_contrast(
    region_topic: npt.NDArray[np.float32],
    cell_topic: npt.NDArray[np.float32],
    region_names: list[str],
    cell_names: list[str],
    highly_variable_regions: list[str],
    contrast_name: str,
    selected_foreground_cells: list[str],
    selected_background_cells: list[str],
    scale_factor1: int = 10**6,
    regions_chunk_size: int = 20000,
    adjusted_pvalue_threshold: float = 0.05,
    log2_fold_change_threshold: float = math.log2(1.5),
) -> pl.DataFrame:
    """
    Get marker regions for a contrast.

    Get marker regions for a contrast of foreground and background cells by calculating
    the adjusted p-values and log2 fold change for the scaled imputed accessibility of
    foreground cells vs background cells for each highly variable region.

    High level overview of the function:
      - Subset region_topic and cell_topic to requested regions (highly variable regions)
        and cells (foreground + background).
      - Calculate imputed accessibility: `region_topic @ cell_topic`.
      - Scale imputed accessibility is scaled by `scale_factor1` to create a "count"
        matrix.
      - Only keep integer part of the scaled imputed accessibility.
      - Error out if there are regions for which scaled imputed accessibility was 0 in
        all cells as this indicates that those regions are not highly variable and thus
        that the user gave the wrong (unfiltered) regions.
      - Calculate adjusted Wilcoxon test p-values and log2 fold change for scaled
        imputed accessibility of foreground cells vs background cells for each highly
        variable region.

    Parameters
    ----------
    region_topic
        Region topic matrix (regions x topics).
    cell_topic
        Cell topic matrix (topic x cells).
    region_names
        List of all region names in the region topic matrix.
    cell_names
        List of all cell names in the cell topic matrix.
    highly_variable_regions
        List of highly variable regions. Only these regions will be used to calculate
        the adjusted p-values and log2 fold change for the foreground and background
        cells contrast.
    contrast_name
        Name of the contrast.
    selected_foreground_cells
        List of foreground cells for the contrast.
    selected_background_cells
        List of background cells for the contrast.
    scale_factor1
        Multiply imputed accessibility by this scale factor to create a "count" matrix.
        This will remove noise by putting very small values to zero.
        Default: `10**6`.
    regions_chunk_size
        Regions chunk size used (number of regions for which imputed accessibility is
        calculated at the same time).
        Default: `20000`.
    adjusted_pvalue_threshold
        Adjusted p-value threshold.
        Default: `0.05`.
    log2_fold_change_threshold
        Log2FC threshold.
        Default: `math.log2(1.5)`.

    Returns
    -------
    Polars DataFrame with highly variable regions, log2 fold change, adjusted p-values
    for contrast.

    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    # Get subset of region names index positions.
    selected_region_names_idx = get_position_index(
        highly_variable_regions, region_names
    )

    # Get subset of cell names index positions for foreground and background cells.
    selected_foreground_and_background_cell_names_idx = get_position_index(
        selected_foreground_cells + selected_background_cells,
        cell_names,
    )

    # Get number of foreground cells as it will be used later to divide the imputed
    # accessibility chunk back in a foreground and background part.
    n_foreground_cells = len(selected_foreground_cells)
    n_background_cells = len(selected_background_cells)

    # Subset region_topic and cell_topic to regions and cells we want to keep.
    region_topic_subset = np.asarray(region_topic, dtype=np.float32)[
        selected_region_names_idx, :
    ]
    cell_topic_subset = np.asarray(cell_topic, dtype=np.float32)[
        :, selected_foreground_and_background_cell_names_idx
    ]

    output_regions_chunk_end = 0
    n_regions = region_topic_subset.shape[0]
    n_cells = cell_topic_subset.shape[1]
    # n_topics = region_topic_subset.shape[1]

    adjusted_pvalues = np.zeros((n_regions,), dtype=np.float64)
    log2_fold_change = np.zeros((n_regions,), dtype=np.float64)

    log.info(
        "Calculate adjusted Wilcoxon test p-values and log2 fold change for imputed "
        "accessibility of foreground cells vs background cells for each highly "
        f'variable region for "{contrast_name}".'
    )

    log.info(
        f"Allocate {(regions_chunk_size * n_cells * 4 / 1024**3):.3f} GiB of RAM "
        f"for calculating imputed accessibility for {n_foreground_cells} foreground "
        f"and {n_background_cells} background cells for chunk of {regions_chunk_size} "
        f"regions."
    )
    # Preallocate imputed accessibility chunk array (regions_chunk_size x n_cells) so
    # it can be reused in each loop iteration (except for the last one as that one will
    # likely be smaller).
    imputed_acc_chunk = np.empty((regions_chunk_size, n_cells), dtype=np.float32)

    # Calculate total imputed accessibility per cell.
    for input_regions_chunk_start in range(0, n_regions, regions_chunk_size):
        input_regions_chunk_end = input_regions_chunk_start + regions_chunk_size

        # Set correct output chunk start position.
        output_regions_chunk_start = output_regions_chunk_end

        log.info(
            "Calculate adjusted Wilcoxon test p-values and log2 fold change for regions "
            f"{input_regions_chunk_start}-{input_regions_chunk_end} (out of {n_regions})."
        )

        # Get the current chunk of regions.
        topic_region_chunk = region_topic_subset[
            input_regions_chunk_start : input_regions_chunk_start + regions_chunk_size
        ]
        current_regions_chunk_size = topic_region_chunk.shape[0]

        if current_regions_chunk_size < regions_chunk_size:
            log.info(
                f"Allocate {(regions_chunk_size * n_cells * 4 / 1024**3):.3f} GiB of "
                f"RAM for calculating imputed accessibility for {n_foreground_cells} "
                f"foreground and {n_background_cells} background cells for chunk of "
                f"{current_regions_chunk_size} regions."
            )
            # Reallocate imputed_acc_chunk to the correct size.
            del imputed_acc_chunk
            imputed_acc_chunk = np.empty(
                (current_regions_chunk_size, n_cells), dtype=np.float32
            )

        log.info(
            "  - Calculate imputed accessibility for the current chunk of regions."
        )
        # Calculate imputed accessibility for the current chunk of regions.
        np.matmul(topic_region_chunk, cell_topic_subset, out=imputed_acc_chunk)

        log.info('  - Scale imputed accessibility matrix chunk ("count" matrix).')
        # Scale imputed accessibility matrix chunk ("count" matrix).
        imputed_acc_chunk *= np.float32(scale_factor1)

        log.info("  - Only keep integer part.")
        # Only keep integer part.
        # This will convert very small values (< 1.0) to zero and removes noise.
        np.floor(imputed_acc_chunk, out=imputed_acc_chunk)

        log.info("  - Get non-zero regions.")
        # Get all region index positions of the matrix for which the whole row is not
        # completely zero (regions which are never accessible in any cell).
        region_idx_to_keep_chunk = get_nonzero_row_indices(imputed_acc_chunk)

        if len(region_idx_to_keep_chunk) != current_regions_chunk_size:
            raise ValueError(
                '"highly_variable_regions" contains regions that are never accessible '
                "in any cell"
            )

        # Set correct output chunk end position by taking into account that rows with
        # all zeros will be filtered out if the above check would be removed.
        # In the current case the output chunk positions would be the same as the input.
        output_regions_chunk_end = output_regions_chunk_start + len(
            region_idx_to_keep_chunk
        )

        # Extract imputed accessibility for foreground and background cells.
        imputed_acc_chunk_foreground_cells = imputed_acc_chunk[:, :n_foreground_cells]
        imputed_acc_chunk_background_cells = imputed_acc_chunk[:, n_foreground_cells:]

        log.info("  - Calculate adjusted Wilcoxon test p-values.")
        # Calculate Wilcoxon test p-values for each region in the current chunk
        # and calculate adjusted p-values for them.
        wilcoxon_test_pvalues_chunk = ranksums_numba_multiple(
            imputed_acc_chunk_foreground_cells.astype(np.uint32),
            imputed_acc_chunk_background_cells.astype(np.uint32),
        )[1]
        adjusted_pvalues_chunk = p_adjust_bh(wilcoxon_test_pvalues_chunk)

        log.info("  - Calculate log2 fold change.")
        # Calculate log2 fold change for each region in the current chunk between
        # foreground and background cells.
        log2_fold_change_chunk = get_log2_fc(
            imputed_acc_chunk_foreground_cells, imputed_acc_chunk_background_cells
        )

        # Fill full adjusted p-values and log2 fold change arrays with the values calculated on the current chunk.
        adjusted_pvalues[output_regions_chunk_start:output_regions_chunk_end] = (
            adjusted_pvalues_chunk
        )
        log2_fold_change[output_regions_chunk_start:output_regions_chunk_end] = (
            log2_fold_change_chunk
        )

    del imputed_acc_chunk

    markers_df = (
        pl.DataFrame(
            {
                "RegionNames": highly_variable_regions,
                "Log2FC": log2_fold_change,
                "Adjusted_pval": adjusted_pvalues,
            }
        )
        .lazy()
        .with_columns(pl.lit(contrast_name).alias("Contrast"))
        .filter(pl.col("Adjusted_pval") <= adjusted_pvalue_threshold)
        .filter(pl.col("Log2FC") >= log2_fold_change_threshold)
        .sort(
            ["Log2FC", "Adjusted_pval"], descending=[True, False], maintain_order=True
        )
        .collect()
    )

    log.info(
        "Finished calculating adjusted Wilcoxon test p-values and log2 fold change "
        "for imputed accessibility of foreground cells vs background cells for each "
        f'highly variable region for "{contrast_name}".'
    )

    return markers_df


def find_diff_accessible_regions(
    region_topic: npt.NDArray[np.float32],
    cell_topic: npt.NDArray[np.float32],
    region_names: list[str],
    cell_names: list[str],
    highly_variable_regions: list[str],
    contrasts: dict[str, list[str], list[str]],
    scale_factor1: int = 10**6,
    regions_chunk_size: int = 20000,
    adjusted_pvalue_threshold: float = 0.05,
    log2_fold_change_threshold: float = math.log2(1.5),
) -> dict[str, pl.DataFrame]:
    """
    Find differential imputed regions for multiple contrasts.

    Get marker regions for each contrast of foreground and background cells by
    calculating the adjusted p-values and log2 fold change for the scaled imputed
    accessibility of foreground cells vs background cells for each highly variable
    region.

    High level overview of the function:
      - Subset region_topic and cell_topic to requested regions (highly variable regions)
        and cells (foreground + background).
      - Calculate imputed accessibility: `region_topic @ cell_topic`.
      - Scale imputed accessibility is scaled by `scale_factor1` to create a "count"
        matrix.
      - Only keep integer part of the scaled imputed accessibility.
      - Error out if there are regions for which scaled imputed accessibility was 0 in
        all cells as this indicates that those regions are not highly variable and thus
        that the user gave the wrong (unfiltered) regions.
      - Calculate adjusted Wilcoxon test p-values and log2 fold change for scaled
        imputed accessibility of foreground cells vs background cells for each highly
        variable region.

    Parameters
    ----------
    region_topic
        Region topic matrix (regions x topics).
    cell_topic
        Cell topic matrix (topic x cells).
    region_names
        List of all region names in the region topic matrix.
    cell_names
        List of all cell names in the cell topic matrix.
    highly_variable_regions
        List of highly variable regions. Only these regions will be used to calculate
        the adjusted p-values and log2 fold change for the foreground and background
        cells contrast.
    contrasts
        Dictionary with as keys the contrast names and as values a tuple with a list
        of selected foreground and a list of background cells for that contrast.
    scale_factor1
        Multiply imputed accessibility by this scale factor to create a "count" matrix.
        This will remove noise by putting very small values to zero.
        Default: `10**6`.
    regions_chunk_size
        Regions chunk size used (number of regions for which imputed accessibility is
        calculated at the same time).
        Default: `20000`.
    adjusted_pvalue_threshold
        Adjusted p-value threshold.
        Default: `0.05`.
    log2_fold_change_threshold
        Log2FC threshold.
        Default: `math.log2(1.5)`.

    Returns
    -------
    Dictionary with as keys the contrast names and as values a Polars DataFrame with
    marker regions, log2 fold change and adjusted p-values for that contrast.

    """
    markers_dict = {}

    for contrast_name, (
        selected_foreground_cells,
        selected_background_cells,
    ) in contrasts.items():
        markers_dict[contrast_name] = get_marker_regions_for_contrast(
            region_topic=region_topic,
            cell_topic=cell_topic,
            region_names=region_names,
            cell_names=cell_names,
            highly_variable_regions=highly_variable_regions,
            contrast_name=contrast_name,
            selected_foreground_cells=selected_foreground_cells,
            selected_background_cells=selected_background_cells,
            scale_factor1=scale_factor1,
            regions_chunk_size=regions_chunk_size,
            adjusted_pvalue_threshold=adjusted_pvalue_threshold,
            log2_fold_change_threshold=log2_fold_change_threshold,
        )

    return markers_dict


# TODO: Add these generic functions to another package
def p_adjust_bh(p: npt.NDArray):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


@numba.njit(parallel=True)
def subset_array_second_axis(arr: npt.NDArray, col_indices: npt.NDArray):
    """
    Subset array by second axis based on provided `col_indices`.

    Returns the same as `arr[:, col_indices]`, but is much faster
    when arr and col_indices are big.

    Parameters
    ----------
    arr
        2D-numpy array to subset by provided column indices.
    col_indices
        1D-numpy array (preferably with np.int64 as dtype) with column indices.

    """
    if arr.ndim != 2:
        raise ValueError("arr should be a 2D array")

    if col_indices.ndim != 1:
        raise ValueError("col_indices should be a 1D array")

    if np.max(col_indices) >= arr.shape[1]:
        raise IndexError(
            f"index {np.max(col_indices)} is out of bounds for axis 1 with size {arr.shape[1]}"
        )
    if np.min(col_indices) < -arr.shape[1]:
        raise IndexError(
            f"index {np.min(col_indices)} is out of bounds for axis 1 with size {arr.shape[1]}"
        )

    # Create empty subset array of correct dimensions and dtype.
    subset_arr = np.empty(
        (arr.shape[0], col_indices.shape[0]),
        dtype=arr.dtype,
    )

    for i in numba.prange(arr.shape[0]):
        # Get requested column values for each row.
        subset_arr[i, :] = arr[i, :][col_indices]

    return subset_arr


@numba.njit(parallel=True)
def mean_axis1(arr: npt.NDArray):
    """
    Calculate column wise mean of 2D-numpy matrix with numba, mimicking `np.mean(x, axis=1)`.

    Parameters
    ----------
    arr
        2D-numpy array to calculate the mean per column for.

    """
    mean_axis1_array = np.empty(arr.shape[0], dtype=np.float64)
    for i in numba.prange(arr.shape[0]):
        mean_axis1_array[i] = np.mean(arr[i, :])
    return mean_axis1_array


@numba.njit
def get_log2_fc(fg_mat: npt.NDArray, bg_mat: npt.NDArray):
    """
    Calculate log2 fold change between foreground and background matrix.

    Parameters
    ----------
    fg_mat
        2D-numpy foreground matrix.
    bg_mat
        2D-numpy background matrix.

    """
    if fg_mat.ndim != 2 or bg_mat.ndim != 2:
        raise ValueError("fg_mat and bg_mat should be 2D arrays")

    if fg_mat.shape[0] != bg_mat.shape[0]:
        raise ValueError(
            "Foreground matrix and background matrix have a different first dimension:"
            f" {fg_mat.shape[0]} vs {bg_mat.shape[0]}"
        )

    # Calculate log2 fold change between foreground and background matrix with numba in
    # a similar way as the following numpy code:
    #    np.log2(
    #        (np.mean(fg_mat, axis=1) + 10**-12) / (np.mean(bg_mat, axis=1) + 10**-12)
    #    )
    return np.log2((mean_axis1(fg_mat) + 10**-12) / (mean_axis1(bg_mat) + 10**-12))


@numba.jit(nopython=True)
def rankdata_average_numba(arr: npt.NDArray):
    """
    Assign ranks to data, dealing with ties by taking average of ranks that would have been assigned to all tied values.

    Ranks begin at 1.

    Algorithm based on `scipy.stats.ranksums` of scipy 1.11.x with the following
    parameters: `rankdata(a, method="average, axis=None, nan_policy="omit")`,
    but with the assumption that there are no `np.nan` values.

    https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/stats/_stats_py.py#L10123-L10267

    """
    sorter = np.argsort(arr, kind="quicksort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    arr = arr[sorter]
    obs = np.empty(arr.shape, dtype=np.intp)
    obs[0] = True
    obs[1:] = arr[1:] != arr[:-1]
    dense = obs.cumsum()[inv]
    non_zero = np.nonzero(obs)[0]
    count = np.empty(non_zero.shape[0] + 1)
    count[0:-1] = non_zero
    count[-1] = len(obs)
    result = 0.5 * (count[dense] + count[dense - 1] + 1)
    return result


@numba.jit(nopython=True)
def norm_sf(z: float):
    """Survival function (1 - `cdf`) at z of the given RV."""
    return (1.0 + math.erf(-z / math.sqrt(2.0))) / 2.0


@numba.jit(nopython=True)
def ranksums_numba(x: npt.NDArray, y: npt.NDArray):
    """
    Compute the Wilcoxon rank-sum statistic for two samples.

    The Wilcoxon rank-sum test tests the null hypothesis that two sets
    of measurements are drawn from the same distribution.  The alternative
    hypothesis is that values in one sample are more likely to be
    larger than the values in the other sample.

    This test should be used to compare two samples from continuous
    distributions.  It does not handle ties between measurements
    in x and y.

    Algorithm based on `scipy.stats.ranksums`.
    """
    n1 = len(x)
    n2 = len(y)
    alldata = np.concatenate((x, y))
    ranked = rankdata_average_numba(alldata)
    x = ranked[:n1]
    s = np.sum(x, axis=0)
    expected = n1 * (n1 + n2 + 1) / 2.0
    z = (s - expected) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    prob = 2 * norm_sf(abs(z))
    return z, prob


@numba.jit(nopython=True, parallel=True)
def ranksums_numba_multiple(X: npt.NDArray, Y: npt.NDArray):
    """
    Compute multiple Wilcoxon rank-sum statistics for two samples.

    For each row of X and Y Wilcoxon rank-sum statistics for two samples are computed.

    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y should be 2D arrays")
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("X and Y should have the same shape on dimension 0")
    ranksums_z = np.empty((n,), dtype=np.float64)
    ranksums_p = np.empty((n,), dtype=np.float64)
    for i in numba.prange(n):
        z, p = ranksums_numba(X[i], Y[i])
        ranksums_z[i] = z
        ranksums_p[i] = p
    return ranksums_z, ranksums_p
