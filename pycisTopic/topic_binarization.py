import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyscenic import binarization
import sys

from typing import Optional
from typing import Dict, Tuple

from .cistopic_class import *


def binarize_topics(cistopic_obj: 'CistopicObject',
                    target: Optional[str] = 'region',
                    method: Optional[str] = 'otsu',
                    smooth_topics: Optional[bool] = False,
                    ntop: Optional[int] = 2000,
                    predifined_thr: Optional[Dict[str, float]] = {},
                    nbins: Optional[int] = 100,
                    plot: Optional[bool] = False,
                    figsize: Optional[Tuple[float, float]] = (6.4, 4.8),
                    num_columns: Optional[int] = 1,
                    save: Optional[str] = None):
    """
    Binarize topic distributions.

    Parameters
    ---------
    cistopic_obj: `class::CistopicObject`
        A cisTopic object with a model in `class::CistopicObject.selected_model`.
    target: str, optional
        Whether cell-topic ('cell') or region-topic ('region') distributions should be binarized. Default: 'region'
    method: str, optional
        Method to use for topic binarization. Possible options are: 'otsu' [Otsu, 1979], 'yen' [Yen et al., 1995], 'li'
        [Li & Lee, 1993], 'aucell' [Van de Sande et al., 2020] or 'ntop' [Taking the top n regions per topic]. Default: 'otsu'
    smooth_topics: bool, optional
        Whether to smooth topics distributions to penalize regions enriched across many topics. The formula applied is
        \\eqn{\beta_{w, k} (\\log\beta_{w,k} - 1 / K \\sum_{k'} \\log \beta_{w,k'})}
    ntop: int, optional
        Number of top regions to select when using method='ntop'. Default: 2000
    predefined_thr: Dict, optional
        A dictionary containing topics as keys and threshold as values. If a topic is not present, thresholds will be computed with the specified method.
        This can be used for manually adjusting thresholds when necessary. Default: None.
    nbins: int, optional
        Number of bins to use in the histogram used for otsu, yen and li thresholding. Default: 100
    plot: bool, optional
        Whether to plot region-topic distributions and their threshold. Default: False
    figsize: tuple, optional
        Size of the figure. If num_columns is 1, this is the size for each figure; if num_columns is above 1, this is the overall size of the figure (if keeping
        default, it will be the size of each subplot in the figure). Default: (6.4, 4.8)
    num_columns: int, optional
        For multiplot figures, indicates the number of columns (the number of rows will be automatically determined based on the number of plots). Default: 1
    save: str, optional
        Path to save plot. Default: None.

    Return
    ---------
    dict
        A dictionary containing a pd.DataFrame with the selected regions with region names as indexes and a topic score
        column.

    References
    ---------
    Otsu, N., 1979. A threshold selection method from gray-level histograms. IEEE transactions on systems, man, and
    cybernetics, 9(1), pp.62-66.
    Yen, J.C., Chang, F.J. and Chang, S., 1995. A new criterion for automatic multilevel thresholding. IEEE Transactions on
    Image Processing, 4(3), pp.370-378.
    Li, C.H. and Lee, C.K., 1993. Minimum cross entropy thresholding. Pattern recognition, 26(4), pp.617-625.
    Van de Sande, B., Flerin, C., Davie, K., De Waegeneer, M., Hulselmans, G., Aibar, S., Seurinck, R., Saelens, W., Cannoodt, R.,
    Rouchon, Q. and Verbeiren, T., 2020. A scalable SCENIC workflow for single-cell gene regulatory network analysis. Nature Protocols,
    15(7), pp.2247-2276.
    """
    # Create cisTopic logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    if target is 'region':
        topic_dist = cistopic_obj.selected_model.topic_region
    elif target is 'cell':
        topic_dist = cistopic_obj.selected_model.cell_topic.T

    if smooth_topics:
        topic_dist = smooth_topics(topic_dist)

    binarized_topics = {}
    pdf = None
    if (save is not None) & (num_columns == 1):
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)

    if num_columns > 1:
        num_rows = np.ceil(topic_dist.shape[1] / num_columns)
        if figsize == (6.4, 4.8):
            figsize = (6.4 * num_columns, 4.8 * num_rows)

    fig = plt.figure(figsize=figsize)
    j = 1
    for i in range(topic_dist.shape[1]):
        l = np.asarray(topic_dist.iloc[:, i])
        l_norm = (l - np.min(l)) / np.ptp(l)
        if 'Topic' + str(i + 1) in (list(predifined_thr.keys())):
            thr = predifined_thr['Topic' + str(i + 1)]
        elif method == 'otsu':
            thr = threshold_otsu(l_norm, nbins=nbins)
        elif method == 'yen':
            thr = threshold_yen(l_norm, nbins=nbins)
        elif method == 'li':
            thresholds = np.arange(
                np.min(l_norm) + 0.01, np.max(l_norm) - 0.01, 0.01)
            entropies = [cross_entropy(l_norm, t, nbins=nbins)
                         for t in thresholds]
            thr = thresholds[np.argmin(entropies)]
        elif method == 'aucell':
            df, thr = binarization.binarize(pd.DataFrame(l_norm))
            thr = float(thr)
        elif method == 'ntop':
            data = pd.DataFrame(l_norm).sort_values(0, ascending=False)
            thr = float(data.iloc[ntop, ])
        else:
            log.info(
                'Binarization method not found. Please choose: "otsu", "yen", "li" or "ntop".')

        if plot:
            if num_columns > 1:
                plt.subplot(num_rows, num_columns, j)
                j = j + 1
            plt.hist(l_norm, bins=nbins)
            plt.axvline(thr, color='tomato', linestyle='--')
            plt.xlabel('Standardized probability Topic ' + str(i + 1) +
                       '\n' + 'Selected:' + str(sum(l_norm > thr)), fontsize=10)
            if num_columns == 1:
                if save is not None:
                    pdf.savefig(fig, bbox_inches='tight')
                if plot:
                    plt.show()
        binarized_topics['Topic' + str(i + 1)] = pd.DataFrame(
            topic_dist.iloc[l_norm > thr, i]).sort_values('Topic' + str(i + 1), ascending=False)

    if target == 'region':
        cistopic_obj.selected_model.topic_ass['Regions_in_binarized_topic'] = [
            binarized_topics[x].shape[0] for x in binarized_topics.keys()]
    elif target == 'cell':
        cistopic_obj.selected_model.topic_ass['Cells_in_binarized_topic'] = [
            binarized_topics[x].shape[0] for x in binarized_topics.keys()]

    if num_columns > 1:
        plt.tight_layout()
        if save is not None:
            fig.savefig(save, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()

    if (save is not None) & (num_columns == 1):
        pdf.close()

    return binarized_topics


def smooth_topics(topic_region):
    """
    Smooth topic-region distributions.

    Parameters
    ---------
    topic_region: `class::pd.DataFrame`
            A pandas dataframe with topic-region distributions (with topics as columns and regions as rows)

    Return
    ---------
    pd.DataFrame
    """
    topic_region_np = np.apply_along_axis(norm, 1, topic_region.values)
    topic_region = pd.DataFrame(
        topic_region_np,
        index=topic_region.index.tolist(),
        columns=topic_region.columns)
    return topic_region


def norm(x):
    """
    Smooth topic-region distributions.

    Parameters
    ---------
    x: `class::pd.Series`
            A pandas series with the topic-region distribution for a topic

    Return
    ---------
    numpy.array
    """
    return x * (np.log(x + 1e-05) - np.sum(np.log(x + 1e-05)) / len(x))


def threshold_yen(array: np.array,
                  nbins: Optional[int] = 100):
    """
    Apply Yen threshold on topic-region distributions [Yen et al., 1995].

    Parameters
    ---------
    array: `class::np.array`
            Array containing the region values for the topic to be binarized.
    nbins: int
            Number of bins to use in the binarization histogram

    Return
    ---------
    float
            Binarization threshold.

    Reference
    ---------
    Yen, J.C., Chang, F.J. and Chang, S., 1995. A new criterion for automatic multilevel thresholding. IEEE Transactions on
    Image Processing, 4(3), pp.370-378.
    """
    hist, bin_centers = histogram(array, nbins)
    # Calculate probability mass function
    pmf = hist.astype(np.float32) / hist.sum()
    P1 = np.cumsum(pmf)  # Cumulative normalized histogram
    P1_sq = np.cumsum(pmf ** 2)
    # Get cumsum calculated from end of squared array
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    # P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid
    # '-inf' in crit. ImageJ Yen implementation replaces those values by zero.
    crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) *
                  (P1[:-1] * (1.0 - P1[:-1])) ** 2)
    return bin_centers[crit.argmax()]


def threshold_otsu(array, nbins=100):
    """
    Apply Otsu threshold on topic-region distributions [Otsu, 1979].

    Parameters
    ---------
    array: `class::np.array`
            Array containing the region values for the topic to be binarized.
    nbins: int
            Number of bins to use in the binarization histogram

    Return
    ---------
    float
            Binarization threshold.

    Reference
    ---------
    Otsu, N., 1979. A threshold selection method from gray-level histograms. IEEE transactions on systems, man, and
    cybernetics, 9(1), pp.62-66.
    """
    hist, bin_centers = histogram(array, nbins)
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


def cross_entropy(array, threshold, nbins=100):
    """
    Calculate entropies for Li thresholding on topic-region distributions [Li & Lee, 1993].

    Parameters
    ---------
    array: `class::np.array`
            Array containing the region values for the topic to be binarized.
    threshold: float
            Distribution threshold to calculate entropy from.
    nbins: int
            Number of bins to use in the binarization histogram

    Return
    ---------
    float
            Entropy for the given threshold.

    Reference
    ---------
    Li, C.H. and Lee, C.K., 1993. Minimum cross entropy thresholding. Pattern recognition, 26(4), pp.617-625.
    """
    hist, bin_centers = histogram(array, nbins=nbins)
    t = np.flatnonzero(bin_centers > threshold)[0]
    m0a = np.sum(hist[:t])  # 0th moment, background
    m0b = np.sum(hist[t:])
    m1a = np.sum(hist[:t] * bin_centers[:t])  # 1st moment, background
    m1b = np.sum(hist[t:] * bin_centers[t:])
    mua = m1a / m0a  # mean value, background
    mub = m1b / m0b
    nu = -m1a * np.log(mua) - m1b * np.log(mub)
    return nu


def histogram(array, nbins=100):
    """
    Draw histogram from distribution and identify centers.

    Parameters
    ---------
    array: `class::np.array`
            Scores distribution
    nbins: int
            Number of bins to use in the histogram

    Return
    ---------
    float
            Histogram values and bin centers.
    """
    array = array.ravel().flatten()
    hist, bin_edges = np.histogram(array, bins=nbins, range=None)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    return hist, bin_centers
