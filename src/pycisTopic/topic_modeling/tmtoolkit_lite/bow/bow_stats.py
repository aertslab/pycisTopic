"""
Common statistics for bag-of-words (BoW) or sparse word representation models.

.. codeauthor:: Markus Konrad <post@mkonrad.net>
"""

import numpy as np
from scipy.sparse import issparse


def doc_frequencies(dtm, min_val=1, proportions=0):
    """
    For each term in the vocab of `dtm` (i.e. its columns), return how often it occurs at least `min_val` times per
    document.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :param min_val: threshold for counting occurrences
    :param proportions: one of :attr:`~tmtoolkit.types.Proportion`: ``NO (0)`` – return counts; ``YES (1)`` – return
                        proportions; ``LOG (2)`` – return log of proportions
    :return: NumPy array of size M (vocab size) indicating how often each term occurs at least `min_val` times.
    """
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    doc_freq = np.sum(dtm >= min_val, axis=0)

    if doc_freq.ndim != 1:
        doc_freq = doc_freq.A.flatten()

    if proportions == 1:
        return doc_freq / dtm.shape[0]
    elif proportions == 2:
        return np.log(doc_freq) - np.log(dtm.shape[0])
    else:
        return doc_freq


def codoc_frequencies(dtm, min_val=1, proportions=0):
    """
    Calculate the co-document frequency (aka word co-occurrence) matrix for a document-term matrix `dtm`, i.e. how often
    each pair of tokens occurs together at least `min_val` times in the same document. If `proportions` is True,
    return proportions scaled to the number of documents instead of absolute numbers.

    .. seealso:: See :func:`~tmtoolkit.utils.pairwise_max_table` for a convenient way to get the maximum token
                 cooccurrences in tabular form.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :param min_val: threshold for counting occurrences
    :param proportions: one of :attr:`~tmtoolkit.types.Proportion`: ``NO (0)`` – return counts; ``YES (1)`` – return
                        proportions; ``LOG (2)`` – convert input to dense matrix if necessary and return
                        *log(proportions + 1)*
    :return: co-document frequency (aka word co-occurrence) matrix with shape (vocab size, vocab size)
    """
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    if dtm.shape[1] < 2:
        raise ValueError('`dtm` must have at least two columns')

    if issparse(dtm) and dtm.format != 'csc':
        dtm = dtm.tocsc()

    bin_dtm = (dtm >= min_val).astype(int)

    cooc = bin_dtm.T @ bin_dtm

    if proportions == 1:
        return cooc / dtm.shape[0]
    elif proportions == 2:
        if issparse(cooc):
            cooc = cooc.todense()
        return np.log1p(cooc) - np.log(dtm.shape[0])
    else:  #  proportions == 0
        return cooc
