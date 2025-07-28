from __future__ import annotations

from functools import partial
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt

#from pyscenic import binarization


def smooth_topics_distributions(
    cell_or_region_topic_prob: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""
    Smooth topic-region distributions.

    Smooth topics distributions to penalize regions enriched across many topics.
    The formula applied is:

    .. math::
      \beta_{w, k} (\log\beta_{w,k} - 1 / K \sum_{k'} \log \beta_{w,k'})

    Parameters
    ----------
    cell_or_region_topic_prob
       Numpy array containing cell or region topic probabilities with topics along columns.

    Returns
    -------
    Smoothed topic-region dataframe.

    """

    def smooth_topic_distribution(
        x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Smooth topic-region distribution for a topic.

        Parameters
        ----------
        x
            A 1D numpy array with the topic-region distribution for a topic.

        Return
        ------
        Smoothed topic-region distribution for a topic.

        """
        return x * (np.log(x + 1e-45) - np.sum(np.log(x + 1e-45)) / x.shape[0])

    return np.apply_along_axis(smooth_topic_distribution, 1, cell_or_region_topic_prob)


def threshold_yen(array: npt.NDArray[np.float64], nbins: int = 100) -> float:
    """
    Apply Yen threshold on topic-region distributions [Yen et al., 1995].

    Parameters
    ----------
    array
        Array containing the region values for the topic to be binarized.
    nbins
        Number of bins to use in the binarization histogram.

    Returns
    -------
    Binarization threshold.

    Reference
    ---------
    Yen, J.C., Chang, F.J. and Chang, S., 1995. A new criterion for automatic
    multilevel thresholding. IEEE Transactions on Image Processing, 4(3), pp.370-378.

    """
    hist, bin_centers = histogram_and_bin_centers(array, nbins)
    # Calculate probability mass function.
    pmf = hist.astype(np.float32) / hist.sum()
    P1 = np.cumsum(pmf)  # Cumulative normalized histogram
    P1_sq = np.cumsum(pmf**2)
    # Get cumsum calculated from end of squared array
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    # P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid
    # '-inf' in crit. ImageJ Yen implementation replaces those values by zero.
    crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) * (P1[:-1] * (1.0 - P1[:-1])) ** 2)
    return bin_centers[crit.argmax()]


def threshold_otsu(array: npt.NDArray[np.float64], nbins: int = 100) -> float:
    """
    Apply Otsu threshold on topic-region distributions [Otsu, 1979].

    Parameters
    ----------
    array
        Array containing the region values for the topic to be binarized.
    nbins
        Number of bins to use in the binarization histogram.

    Returns
    -------
    Binarization threshold.

    Reference
    ---------
    Otsu, N., 1979. A threshold selection method from gray-level histograms.
    IEEE transactions on systems, man, and cybernetics, 9(1), pp.62-66.

    """
    hist, bin_centers = histogram_and_bin_centers(array, nbins)
    hist = hist.astype(float)
    # Class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # Class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


def cross_entropy(
    array: npt.NDArray[np.float64], threshold: float, nbins: int = 100
) -> float:
    """
    Calculate entropies for Li thresholding on topic-region distributions [Li & Lee, 1993].

    Parameters
    ----------
    array
        Array containing the region values for the topic to be binarized.
    threshold
        Distribution threshold to calculate entropy from.
    nbins
        Number of bins to use in the binarization histogram.

    Returns
    -------
    Entropy for the given threshold.

    Reference
    ---------
    Li, C.H. and Lee, C.K., 1993. Minimum cross entropy thresholding.
    Pattern recognition, 26(4), pp.617-625.

    """
    hist, bin_centers = histogram_and_bin_centers(array, nbins=nbins)
    t = np.flatnonzero(bin_centers > threshold)[0]
    m0a = np.sum(hist[:t])  # 0th moment, background
    m0b = np.sum(hist[t:])
    m1a = np.sum(hist[:t] * bin_centers[:t])  # 1st moment, background
    m1b = np.sum(hist[t:] * bin_centers[t:])
    mua = m1a / m0a  # mean value, background
    mub = m1b / m0b
    nu = -m1a * np.log(mua) - m1b * np.log(mub)
    return nu


def histogram_and_bin_centers(
    array: npt.NDArray[np.float64], nbins: int = 100
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Draw histogram from distribution and identify centers.

    Parameters
    ----------
    array
        Scores distribution.
    nbins
        Number of bins to use in the histogram.

    Returns
    -------
    Histogram values and bin centers.

    """
    array = array.ravel().flatten()
    hist, bin_edges = np.histogram(array, bins=nbins, range=None)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return hist, bin_centers


def threshold_li(array: npt.NDArray[np.float64], nbins) -> float:
    thresholds = np.arange(np.min(array) + 0.01, np.max(array) - 0.01, 0.01)
    entropies = [cross_entropy(array, t, nbins=nbins) for t in thresholds]
    thr = thresholds[np.argmin(entropies)]
    return thr


def threshold_aucell(array: npt.NDArray[np.float64]):
    # TODO: implement this function to avoid pyscenic dependency
    raise NotImplementedError("AUCell binarization is not implemented in this version of pycisTopic.")
    #_, thr = binarization.binarize(pd.DataFrame(array))
    #return float(thr)


def threshold_ntop(array: npt.NDArray[np.float64], ntop: int) -> float:
    return np.sort(array)[::-1][ntop]


def binarize_topics(
    cell_or_region_topic_prob: npt.NDArray[np.float64],
    cell_or_region_names: list[str],
    method: Literal["otsu", "ntop", "li", "yen", "aucell"] = "otsu",
    smooth_topics: bool = True,
    ntop: int | None = None,
    nbins: int = 100,
) -> tuple[list[list[str]], list[npt.NDArray[np.float64]], list[float]]:
    r"""
    Binarize topic distributions.

    Parameters
    ----------
    cell_or_region_topic_prob
        Numpy array containing cell or region topic probabilities with topics along columns.
    cell_or_region_names
        A list of str containing cell or region names (should be the same length as the number of rows in `cell_or_region_topic_prob`)
    method
        Method to use for topic binarization. Possible options are:
          - ``otsu`` [Otsu, 1979]
          - ``yen`` [Yen et al., 1995]
          - ``li`` [Li & Lee, 1993]
          - ``aucell`` [Van de Sande et al., 2020]
          - ``ntop`` [Taking the top n regions per topic]

    smooth_topics
        Whether to smooth topics distributions to penalize regions enriched across many
        topics. The following formula is applied:

        .. math::
          \beta_{w, k} (\log\beta_{w,k} - 1 / K \sum_{k'} \log \beta_{w,k'})
    ntop
        Number of top regions to select when using ``method="ntop"``.
    nbins
        Number of bins to use in the histogram used for ``otsu``, ``yen`` and
        ``li`` thresholding.
        Default: 100.

    Returns
    -------
    A list of string containing binarized cells or regions, an array of scores and a list of floats containing thresholds

    """
    # input validation
    if len(cell_or_region_names) != cell_or_region_topic_prob.shape[0]:
        raise ValueError(
            f"{len(cell_or_region_names)} cells or region names provided while `cell_or_region_topic_prob` only has {cell_or_region_topic_prob.shape[0]} rows."
        )

    if len(cell_or_region_names) != len(set(cell_or_region_names)):
        raise ValueError("`cell_or_region_names` contains duplicates.")

    if method == "ntop" and ntop is None:
        raise ValueError(
            "A value for ntop should be provided when using `ntop` as binarization method."
        )

    method_to_bin_func: dict[str, Callable[[npt.NDArray[np.float64]], float]] = {
        "otsu": partial(threshold_otsu, nbins=nbins),
        "yen": partial(threshold_yen, nbins=nbins),
        "li": partial(threshold_li, nbins=nbins),
        "aucell": threshold_aucell,
        "ntop": partial(threshold_ntop, ntop=ntop),  # type: ignore
    }

    bin_func = method_to_bin_func.get(method)

    if bin_func is None:
        raise ValueError(
            f'`method` should be one of "otsu", "ntop", "li", "yen", "aucell". Not {method}.'
        )

    # create index used for sorting
    cell_or_region_names_idx = {x: i for i, x in enumerate(cell_or_region_names)}

    if smooth_topics:
        cell_or_region_topic_prob = smooth_topics_distributions(
            cell_or_region_topic_prob
        )

    cell_or_region_names_per_topic: list[list[str]] = []
    scores_per_topic: list[npt.NDArray[np.float64]] = []
    thresholds: list[float] = []

    # iterate over topics
    for i in range(cell_or_region_topic_prob.shape[1]):
        # normalize between 0 and 1
        l_norm = (
            cell_or_region_topic_prob[:, i] - np.min(cell_or_region_topic_prob[:, i])
        ) / np.ptp(cell_or_region_topic_prob[:, i])
        # get threshold
        thr = bin_func(l_norm)
        # sort cell or region names based on l_norm, features with highest score first (reverse=True)
        cell_or_region_names_sorted = sorted(
            cell_or_region_names,
            key=lambda x: l_norm[cell_or_region_names_idx[x]],
            reverse=True,
        )
        # get cell or regions passing threshold
        l_norm_a_sort = np.argsort(l_norm)[::-1]
        l_norm_sorted = l_norm[l_norm_a_sort]
        cell_or_regions_passing_threshold = cell_or_region_names_sorted[
            0 : np.where(l_norm_sorted > thr)[0].max() + 1
        ]
        scores_passing_threshold = cell_or_region_topic_prob[l_norm_a_sort, i][
            0 : np.where(l_norm_sorted > thr)[0].max() + 1
        ]
        cell_or_region_names_per_topic.append(cell_or_regions_passing_threshold)
        scores_per_topic.append(scores_passing_threshold)
        thresholds.append(thr)

    return cell_or_region_names_per_topic, scores_per_topic, thresholds
