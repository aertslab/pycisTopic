"""
Common statistics and tools for topic models.
"""

import numpy as np

# %% Common statistics from topic-word or document-topic distribution


def marginal_topic_distrib(doc_topic_distrib, doc_lengths):
    """
    Return marginal topic distribution ``p(T)`` (topic proportions) given the document-topic distribution (theta)
    `doc_topic_distrib` and the document lengths `doc_lengths`. The latter can be calculated with
    :func:`~tmtoolkit.bow.bow_stats.doc_lengths`.

    :param doc_topic_distrib: document-topic distribution; shape NxK, where N is the number of documents, K is the
                              number of topics
    :param doc_lengths: array of size N (number of docs) with integers indicating the number of terms per document
    :return: array of size K (number of topics) with marginal topic distribution
    """
    unnorm = (doc_topic_distrib.T * doc_lengths).sum(axis=1)
    return unnorm / unnorm.sum()


def top_words_for_topics(topic_word_distrib, top_n=None, vocab=None, return_prob=False):
    """
    Generate sorted list of `top_n` words (or word indices) per topic in topic-word distribution `topic_word_distrib`.

    :param topic_word_distrib: topic-word distribution; shape KxM, where K is number of topics, M is vocabulary size
    :param top_n: number of top words (according to probability given topic) to select per topic; if None return full
                  sorted lists of words
    :param vocab: vocabulary array of length M; if None, return word indices instead of word strings
    :param return_prob: if True, also return sorted arrays of word probabilities given topic for each topic
    :return: list of length K consisting of sorted arrays of most probable words; arrays have length `top_n` or M
             (if `top_n` is None); if `return_prob` is True, another list of sorted arrays of word probabilities for
             each topic is returned
    """
    if not isinstance(topic_word_distrib, np.ndarray) or topic_word_distrib.ndim != 2:
        raise ValueError("`topic_word_distrib` must be a 2D NumPy array")

    if len(topic_word_distrib) == 0:
        raise ValueError("`topic_word_distrib` cannot be empty")

    if vocab is not None:
        if not isinstance(vocab, np.ndarray) or vocab.ndim != 1:
            raise ValueError("`vocab` must be a 1D NumPy array")

        if len(vocab) == 0:
            raise ValueError("`vocab` cannot be empty")

        if topic_word_distrib.shape[1] != len(vocab):
            raise ValueError(
                "shapes of provided `topic_word_distrib` and `vocab` do not match (vocab sizes differ)"
            )

    n_vocab = topic_word_distrib.shape[1]

    if top_n is None:
        top_n = n_vocab

    if top_n < 1:
        raise ValueError("`top_n` must be at least 1")
    elif top_n > n_vocab:
        raise ValueError("`top_n` cannot be larger than vocab size")

    topic_words = []
    topic_probs = []

    for topic in topic_word_distrib:
        sorter_arr = np.argsort(topic)
        sorter_slice = slice(None, -(top_n + 1), -1) if top_n < n_vocab else slice(None)

        if vocab is None:
            topic_words.append(sorter_arr[sorter_slice])
        else:
            topic_words.append(vocab[sorter_arr][sorter_slice])

        if return_prob:
            topic_probs.append(topic[sorter_arr[sorter_slice]])

    if return_prob:
        return topic_words, topic_probs
    else:
        return topic_words
