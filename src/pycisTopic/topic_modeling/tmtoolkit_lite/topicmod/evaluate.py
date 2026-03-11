"""
Metrics for topic model evaluation.

In order to run model evaluations in parallel use one of the modules :mod:`~tmtoolkit.topicmod.tm_gensim`,
:mod:`~tmtoolkit.topicmod.tm_lda` or :mod:`~tmtoolkit.topicmod.tm_sklearn`.

.. codeauthor:: Markus Konrad <post@mkonrad.net>
"""

import numpy as np
from scipy.sparse import issparse
from scipy.spatial.distance import pdist

from pycisTopic.topic_modeling.tmtoolkit_lite.bow.bow_stats import (
    codoc_frequencies,
    doc_frequencies,
)
from pycisTopic.topic_modeling.tmtoolkit_lite.topicmod.model_stats import (
    top_words_for_topics,
)

# %% Evaluation metrics


def metric_cao_juan_2009(topic_word_distrib):
    """
    Calculate metric as in [Cao2009]_ using topic-word distribution `topic_word_distrib`.

    :param topic_word_distrib: topic-word distribution; shape KxM, where K is number of topics, M is vocabulary size
    :return: calculated metric
    """
    # pdist will calculate the pair-wise cosine distance between all topics in the topic-word distribution
    # then calculate the mean of cosine similarity (1 - cosine_distance)
    cos_sim = 1 - pdist(topic_word_distrib, metric='cosine')
    return np.mean(cos_sim)
metric_cao_juan_2009.direction = 'minimize'


def metric_arun_2010(topic_word_distrib, doc_topic_distrib, doc_lengths):
    """
    Calculate metric as in [Arun2010]_ using topic-word distribution `topic_word_distrib`, document-topic
    distribution `doc_topic_distrib` and document lengths `doc_lengths`.

    .. note:: It will fail when num. of words in the vocabulary is less then the num. of topics (which is very unusual).

    .. warning:: There's no code available for the [Arun2010]_ paper. The code follows the procedures outlined in the
                 paper so that its results could be reproduced for the NIPS dataset. See the discussion at
                 https://github.com/nikita-moor/ldatuning/issues/7.

    :param topic_word_distrib: topic-word distribution; shape KxM, where K is number of topics, M is vocabulary size
    :param doc_topic_distrib: document-topic distribution; shape NxK, where N is the number of documents
    :param doc_lengths: array of length `N` with number of tokens per document
    :return: calculated metric
    """

    # CM1 – sing. value decomp. of topic-word distrib.
    cm1 = np.linalg.svd(topic_word_distrib, compute_uv=False)
    cm1 /= np.sum(cm1)     # normalize

    # CM2 – topics scaled by document lengths
    doc_lengths = np.asarray(doc_lengths).flatten()
    cm2 = doc_lengths @ doc_topic_distrib
    cm2 = -np.sort(-cm2)   # sort in desc. order (just like cm1 is already sorted in desc. order)
    cm2 /= np.sum(cm2)     # normalize

    # symmetric Kullback-Leibler divergence KL(cm1||cm2) + KL(cm2||cm1)
    # note: using log(x/y) instead of log(x) - log(y) here because values in cm vectors are not small
    return np.sum(cm1 * (np.log(cm1 / cm2))) + np.sum(cm2 * (np.log(cm2 / cm1)))
metric_arun_2010.direction = 'minimize'




def metric_coherence_mimno_2011(topic_word_distrib, dtm, top_n=20, eps=1, include_prob=False, normalize=False,
                                return_mean=False):
    """
    Calculate coherence metric according to [Mimno2011]_. You need to provide a topic word distribution as
    `topic_word_distrib` and a document-term-matrix `dtm` (can be sparse). `top_n` controls how many most probable
    words per topic are selected.

    If you set ``eps=1e-12`` and ``normalize=True``, this is equivalent to the "U_Mass" coherence metric as provided
    in the Gensim package and as wrapper function in :func:`~tmtoolkit.topicmod.evaluate.metric_coherence_gensim` with
    ``measure='u_mass'``.

    By default, it will return a NumPy array of coherence values per topic (same ordering as in `topic_word_distrib`).
    Set `return_mean` to True to return the mean of all topics instead.

    :param topic_word_distrib: topic-word distribution; shape KxM, where K is number of topics, M is vocabulary size
    :param dtm: document-term matrix of shape NxM with N documents and vocabulary size M
    :param top_n: number of most probable words selected per topic
    :param eps: smoothing constant epsilon
    :param include_prob: if True, include probabilities of top words per topic in the calculations
    :param normalize: if True, normalize coherence values
    :param return_mean: if True, return mean of all coherence values, otherwise array of coherence per topic
    :return: if `return_mean` is True, mean of all coherence values, otherwise array of length K with coherence per
             topic
    """
    n_topics, n_vocab = topic_word_distrib.shape

    if n_vocab != dtm.shape[1]:
        raise ValueError('shapes of provided `topic_word_distrib` and `dtm` do not match (vocab sizes differ)')

    if top_n > n_vocab:
        raise ValueError('`top_n=%d` is larger than the vocabulary size of %d words'
                         % (top_n, topic_word_distrib.shape[1]))

    if include_prob:
        top_words, top_prob = top_words_for_topics(topic_word_distrib, top_n, return_prob=True)   # V
    else:
        top_words = top_words_for_topics(topic_word_distrib, top_n, return_prob=False)            # V
        top_prob = None

    if issparse(dtm) and dtm.format != 'csc':
        dtm = dtm.tocsc()

    coh = []
    for t in range(n_topics):
        # calc. coherence for topic t
        v = top_words[t]     # V_t
        p = None if top_prob is None else top_prob[t]    # prob. of words in V_t
        top_dtm = dtm[:, v]  # occurrences for top words V_t; shape (n_docs, top_n)
        df = doc_frequencies(top_dtm)      # for D(v)
        codf = codoc_frequencies(top_dtm)  # for D(v, v')

        c_t = 0
        for m in range(1, top_n):
            for l in range(m):
                if p is None:   # include_prob is False: sum(log((D(v_m, v_l) + eps) / D(v_l)))
                    c_t += np.log((codf[m, l] + eps) / df[l])
                else:           # include_prob is True: sum(log(p_m * p_l * (D(v_m, v_l) + eps) / D(v_l)))
                    c_t += np.log(p[m] * p[l] * (codf[m, l] + eps) / df[l])

        coh.append(c_t)

    coh = np.array(coh)

    if normalize:
        coh *= 2 / (top_n * (top_n-1))

    if return_mean:
        return coh.mean()
    else:
        return coh


metric_coherence_mimno_2011.direction = "maximize"
