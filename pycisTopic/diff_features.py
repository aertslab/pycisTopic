import logging
import sys
from typing import List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import scipy
import scipy.sparse as sparse
import sklearn
from numpy import array, count_nonzero
from scipy.stats import ranksums
from sklearn.preprocessing import normalize

from .cistopic_class import *
from .utils import *


class CistopicImputedFeatures:
    """
    cisTopic imputation data class.

    :class:`CistopicImputedFeatures` contains the cell by features matrices (stored at :attr:`mtx`, with features being eithere regions or genes ),
    cell names :attr:`cell_names` and feature names :attr:`feature_names`.

    Attributes
    ---------
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
        feature_names: List[str],
        cell_names: List[str],
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
        cells: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        copy: Optional[bool] = False,
        split_pattern: Optional[str] = "___",
    ):
        """
        Subset cells and/or regions from :class:`CistopicImputedFeatures`.

        Parameters
        ---------
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
        cistopic_imputed_features_list: List["CistopicImputedFeatures"],
        project: Optional[str] = "cisTopic_impute_merge",
        copy: Optional[bool] = False,
    ):
        """
        Merge a list of :class:`CistopicImputedFeatures` to the input :class:`CistopicImputedFeatures`. Reference coordinates (for regions) must be the same between the objects.

        Parameters
        ---------
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
                        mtx[
                            common_index_fm,
                        ],
                        mtx_to_add[
                            common_index_fm_to_add,
                        ],
                    ],
                    format="csr",
                )
            else:
                mtx_common = np.hstack(
                    [
                        mtx[
                            common_index_fm,
                        ],
                        mtx_to_add[
                            common_index_fm_to_add,
                        ],
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
                            mtx[
                                diff_index_fm_1,
                            ],
                            np.zeros((len(diff_features_1), mtx_to_add.shape[1])),
                        ],
                        format="csr",
                    )
                else:
                    mtx_diff_1 = np.hstack(
                        [
                            mtx[
                                diff_index_fm_1,
                            ],
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
                            mtx_to_add[
                                diff_index_fm_2,
                            ],
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
                            mtx_to_add[
                                diff_index_fm_2,
                            ],
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
        ---------
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
            np.zeros((len(self.feature_names), len(self.cell_names))),
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
            imputed_acc_ranking.mtx[
                :, col_idx
            ] = rank_scores_and_assign_random_ranking_in_range_for_ties(
                mtx[:, col_idx].toarray().flatten()
            )

        return imputed_acc_ranking


def impute_accessibility(
    cistopic_obj: "CistopicObject",
    selected_cells: Optional[List[str]] = None,
    selected_regions: Optional[List[str]] = None,
    scale_factor: Optional[int] = 10**6,
    sparsity_thr: Optional[float] = 0.67,
    project: Optional[str] = "cisTopic_Impute",
):
    """
    Impute region accessibility.

    Parameters
    ---------
    cistopic_obj: `class::CistopicObject`
        A cisTopic object with a model in `class::CistopicObject.selected_model`.
    selected_cells: list, optional
        A list with selected cells to impute accessibility for. Default: None
    selected_regions: list, optional
        A list with selected regions to impute accessibility for. Default: None
    scale_factor: int, optional
        A number to multiply the imputed values for. This is useful to convert low probabilities to 0, making the matrix more sparse. Default: 10**6.
    sparsity_thr: float, optional
        Minimum sparsity ratio in matrix to be converted to sparse matrix. Default: 0.67.
    project: str, optional
            Name of the cisTopic imputation project. Default: 'cisTopic_impute.'

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
    # Convert cell_topic and topic_region 2d arrays to np.float32 so
    # multiplying them uses 4 times less memory than with np.float64
    cell_topic = cell_topic.to_numpy().astype(np.float32)
    topic_region = topic_region.to_numpy().astype(np.float32)
    log.info("Imputing drop-outs")
    imputed_acc = topic_region @ cell_topic
    imputed_acc = imputed_acc.astype(np.float32)
    if isinstance(scale_factor, int):
        if scale_factor != 1:
            log.info("Scaling")
            # Only multiply non-zero data of sparse matrix.
            imputed_acc *= np.int32(scale_factor)
            imputed_acc = imputed_acc.astype(np.int32)
            log.info("Keep non zero rows")
            keep_regions_index = non_zero_rows(imputed_acc)
            imputed_acc = imputed_acc[
                keep_regions_index,
            ]
            region_names = subset_list(region_names, keep_regions_index)
    sparsity = 1.0 - (count_nonzero(imputed_acc) / float(imputed_acc.size))
    log.info(f"Imputed accessibility sparsity: {sparsity}")
    log.info("Create CistopicImputedFeatures object")
    imputed_acc_obj = CistopicImputedFeatures(
        imputed_acc, region_names, cell_names, project
    )
    if sparsity > sparsity_thr:
        log.info("Making matrix sparse")
        imputed_acc_obj.mtx = sparse.csr_matrix(imputed_acc_obj.mtx)
    log.info("Done!")
    return imputed_acc_obj


def normalize_scores(
    input_mat: Union[pd.DataFrame, "CistopicImputedFeatures"],
    scale_factor: Optional[int] = 10**4,
):
    """
    Log-normalize imputation data. Feature counts for each cell are divided by the total counts for that cell and multiplied by the scale_factor.

    Parameters
    ---------
    input_mat: pd.DataFrame or :class:`CistopicImputedFeatures`
        A dataframe with values to be normalize or cisTopic imputation data.
    scale_factor: int, optional
        Scale factor for cell-level normalization. Default: 10**4

    Return
    ------
    pd.DataFrame or CistopicImputedFeatures
        The output class will be the same as the used as input.
    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    log.info("Normalizing imputed data")
    if isinstance(input_mat, CistopicImputedFeatures):
        mtx = normalize(input_mat.mtx, norm="l1", axis=0)
        mtx *= scale_factor
        mtx = np.log1p(mtx)
        output = CistopicImputedFeatures(
            mtx, input_mat.feature_names, input_mat.cell_names, input_mat.project
        )
    elif isinstance(input_mat, pd.DataFrame):
        output = np.log1p(input_mat.values / input_mat.values.sum(0) * scale_factor)
        output = pd.DataFrame(
            output, index=input_mat.index.tolist(), columns=input_mat.columns
        )
    log.info("Done!")
    return output


def find_highly_variable_features(
    input_mat: Union[pd.DataFrame, "CistopicImputedFeatures"],
    min_disp: Optional[float] = 0.05,
    min_mean: Optional[float] = 0.0125,
    max_disp: Optional[float] = np.inf,
    max_mean: Optional[float] = 3,
    n_bins: Optional[int] = 20,
    n_top_features: Optional[int] = None,
    plot: Optional[bool] = True,
    save: Optional[str] = None,
):
    """
    Find highly variable features.

    Parameters
    ---------
    input_mat: pd.DataFrame or :class:`CistopicImputedFeatures`
        A dataframe with values to be normalize or cisTopic imputation data.
    min_disp: float, optional
        Minimum dispersion value for a feature to be selected. Default: 0.05
    min_mean: float, optional
        Minimum mean value for a feature to be selected. Default: 0.0125
    max_disp: float, optional
        Maximum dispersion value for a feature to be selected. Default: np.inf
    max_mean: float, optional
        Maximum mean value for a feature to be selected. Default: 3
    n_bins: int, optional
        Number of bins for binning the mean gene expression. Normalization is done with respect to each bin. Default: 20
    n_top_features: int, optional
        Number of highly-variable features to keep. If specifed, dispersion and mean thresholds will be ignored. Default: None
    plot: bool, optional
        Whether to plot dispersion versus mean values. Default: True.
    save: str, optional
        Path to save feature selection plot. Default: None

    Return
    ------
    List
        List with selected features.
    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    if isinstance(input_mat, pd.DataFrame):
        mat = input_mat.values
        features = input_mat.index.tolist()
    else:
        mat = input_mat.mtx
        features = input_mat.feature_names

    if sparse.issparse(mat):
        mean, var = sklearn.utils.sparsefuncs.mean_variance_axis(mat, axis=1)
    else:
        log.info("Calculating mean")
        mean = np.mean(mat, axis=1, dtype=np.float32)
        log.info("Calculating variance")
        var = np.var(mat, axis=1, dtype=np.float32)

    mean[mean == 0] = 1e-12
    dispersion = var / mean
    # Logarithmic dispersion as in Seurat
    dispersion[dispersion == 0] = np.nan
    dispersion = np.log(dispersion)
    df = pd.DataFrame()
    df["means"] = mean
    df["dispersions"] = dispersion
    df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
    disp_grouped = df.groupby("mean_bin")["dispersions"]
    disp_mean_bin = disp_grouped.mean()
    disp_std_bin = disp_grouped.std(ddof=1)
    # Retrieve those regions that have nan std, these are the ones where
    # only a single gene fell in the bin and implicitly set them to have
    # a normalized dispersion of 1
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
            f"the {n_top_features} top features correspond to a "
            f"normalized dispersion cutoff of {disp_cut_off}"
        )
    else:
        dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
        feature_subset = np.logical_and.reduce(
            (
                mean > min_mean,
                mean < max_mean,
                dispersion_norm > min_disp,
                dispersion_norm < max_disp,
            )
        )

    df["highly_variable"] = feature_subset
    var_features = [features[i] for i in df[df.highly_variable].index.to_list()]

    fig = plt.figure()
    if plot:
        matplotlib.rcParams["agg.path.chunksize"] = 10000
        plt.scatter(
            df["means"], df["dispersions_norm"], c=feature_subset, s=10, alpha=0.1
        )
        plt.xlabel("Mean measurement of features")
        plt.ylabel("Normalized dispersion of the features")
        if save is not None:
            fig.savefig(save)
        plt.show()

    log.info("Done!")
    return var_features


def find_diff_features(
    cistopic_obj: "CistopicObject",
    imputed_features_obj: "CistopicImputedFeatures",
    variable: str,
    var_features: Optional[List[str]] = None,
    contrasts: Optional[List[List[str]]] = None,
    adjpval_thr: Optional[float] = 0.05,
    log2fc_thr: Optional[float] = np.log2(1.5),
    n_cpu: Optional[int] = 1,
    split_pattern: Optional[str] = "___",
    **kwargs,
):
    """
    Find differential imputed features.

    Parameters
    ---------
    cistopic_obj: `class::CistopicObject`
        A cisTopic object including the cells in imputed_features_obj.
    imputed_features_obj: :class:`CistopicImputedFeatures`
        A cisTopic imputation data object.
    variable: str
        Name of the group variable to do comparison. It must be included in `class::CistopicObject.cell_data`
    var_features: list, optional
        A list of features to use (e.g. variable features from `find_highly_variable_features()`)
    contrast: List, optional
        A list including contrasts to make in the form of lists with foreground and background, e.g.
        [[['Group_1'], ['Group_2, 'Group_3']], []['Group_2'], ['Group_1, 'Group_3']], []['Group_1'], ['Group_2, 'Group_3']]].
        Default: None.
    adjpval_thr: float, optional
        Adjusted p-values threshold. Default: 0.05
    log2fc_thr: float, optional
        Log2FC threshold. Default: np.log2(1.5)
    n_cpu: int, optional
        Number of cores to use. Default: 1
    **kwargs
        Parameters to pass to ray.init()

    Return
    ------
    List
        List of `class::pd.DataFrame` per contrast with the selected features and logFC and adjusted p-values.
    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    selected_cells = list(
        set(cistopic_obj.cell_data.index.tolist())
        & set(imputed_features_obj.cell_names)
    )
    group_var = cistopic_obj.cell_data.loc[selected_cells, variable].dropna()
    if contrasts is None:
        levels = sorted(list(set(group_var.tolist())))
        contrasts = [
            [[x], levels[: levels.index(x)] + levels[levels.index(x) + 1 :]]
            for x in levels
        ]
        contrasts_names = levels
    else:
        contrasts_names = [
            "_".join(contrasts[i][0]) + "_VS_" + "_".join(contrasts[i][1])
            for i in range(len(contrasts))
        ]
    # Get barcodes in each class per contrats
    barcode_groups = [
        [
            group_var[group_var.isin(contrasts[x][0])].index.tolist(),
            group_var[group_var.isin(contrasts[x][1])].index.tolist(),
        ]
        for x in range(len(contrasts))
    ]
    # Subset imputed accessibility matrix
    subset_imputed_features_obj = imputed_features_obj.subset(
        cells=None, features=var_features, copy=True, split_pattern=split_pattern
    )
    # Convert to csc
    if sparse.issparse(subset_imputed_features_obj.mtx):
        mtx = subset_imputed_features_obj.mtx.tocsc()
    # Compute p-val and log2FC
    if n_cpu > 1:
        ray.init(num_cpus=n_cpu, **kwargs)
        markers_list = ray.get(
            [
                markers_ray.remote(
                    subset_imputed_features_obj,
                    barcode_groups[i],
                    contrasts_names[i],
                    adjpval_thr=adjpval_thr,
                    log2fc_thr=log2fc_thr,
                )
                for i in range(len(contrasts))
            ]
        )
        ray.shutdown()
    else:
        markers_list = [
            markers_one(
                subset_imputed_features_obj,
                barcode_groups[i],
                contrasts_names[i],
                adjpval_thr=adjpval_thr,
                log2fc_thr=log2fc_thr,
            )
            for i in range(len(contrasts))
        ]
    markers_dict = {
        contrasts_names[i]: markers_list[i] for i in range(len(markers_list))
    }
    return markers_dict


@ray.remote
def markers_ray(
    input_mat: Union[pd.DataFrame, "CistopicImputedFeatures"],
    barcode_group: List[List[str]],
    contrast_name: str,
    adjpval_thr: Optional[float] = 0.05,
    log2fc_thr: Optional[float] = 1,
):
    """
    Find differential imputed features.

    Parameters
    ---------
    input_mat: :class:`pd.DataFrame` or :class:`CistopicImputedFeatures`
        A data frame or a cisTopic imputation data object.
    barcode_group: List
        List of length 2, including foreground cells on the first slot and background on the second.
    contrast_name: str
        Name of the contrast
    adjpval_thr: float, optional
        Adjusted p-values threshold. Default: 0.05
    log2fc_thr: float, optional
        Log2FC threshold. Default: np.log2(1.5)

    Return
    ------
    List
        `class::pd.DataFrame` with the selected features and logFC and adjusted p-values.
    """
    return markers_one(input_mat, barcode_group, contrast_name, adjpval_thr, log2fc_thr)


def markers_one(
    input_mat: Union[pd.DataFrame, "CistopicImputedFeatures"],
    barcode_group: List[List[str]],
    contrast_name: str,
    adjpval_thr: Optional[float] = 0.05,
    log2fc_thr: Optional[float] = 1,
):
    """
    Find differential imputed features.

    Parameters
    ---------
    input_mat: :class:`pd.DataFrame` or :class:`CistopicImputedFeatures`
        A data frame or a cisTopic imputation data object.
    barcode_group: List
        List of length 2, including foreground cells on the first slot and background on the second.
    contrast_name: str
        Name of the contrast
    adjpval_thr: float, optional
        Adjusted p-values threshold. Default: 0.05
    log2fc_thr: float, optional
        Log2FC threshold. Default: np.log2(1.5)

    Return
    ------
    List
        `class::pd.DataFrame` with the selected features and logFC and adjusted p-values.
    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    if isinstance(input_mat, pd.DataFrame):
        mat = input_mat.values
        features = input_mat.index.tolist()
        samples = input_mat.columns
    else:
        mat = input_mat.mtx
        features = input_mat.feature_names
        samples = input_mat.cell_names

    fg_cells_index = get_position_index(barcode_group[0], samples)
    bg_cells_index = get_position_index(barcode_group[1], samples)
    log.info("Formatting data for " + contrast_name)
    if sparse.issparse(mat):
        fg_mat = mat[:, fg_cells_index].toarray()
        bg_mat = mat[:, bg_cells_index].toarray()
    else:
        fg_mat = mat[:, fg_cells_index]
        bg_mat = mat[:, bg_cells_index]

    log.info("Computing p-value for " + contrast_name)
    wilcox_test = [ranksums(fg_mat[x], y=bg_mat[x]) for x in range(mat.shape[0])]
    log.info("Computing log2FC for " + contrast_name)
    logFC = np.log2(
        (np.mean(fg_mat, axis=1) + 10**-12) / (np.mean(bg_mat, axis=1) + 10**-12)
    ).tolist()

    pvalue = [wilcox_test[x].pvalue for x in range(len(wilcox_test))]
    adj_pvalue = p_adjust_bh(pvalue)
    name = [contrast_name] * len(adj_pvalue)
    markers_dataframe = pd.DataFrame(
        [logFC, adj_pvalue, name],
        index=["Log2FC", "Adjusted_pval", "Contrast"],
        columns=features,
    ).transpose()
    markers_dataframe = markers_dataframe.loc[
        markers_dataframe["Adjusted_pval"] <= adjpval_thr
    ]
    markers_dataframe = markers_dataframe.loc[markers_dataframe["Log2FC"] >= log2fc_thr]
    markers_dataframe = markers_dataframe.sort_values(
        ["Log2FC", "Adjusted_pval"], ascending=[False, True]
    )
    log.info(contrast_name + " done!")
    return markers_dataframe


def p_adjust_bh(p: float):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]
