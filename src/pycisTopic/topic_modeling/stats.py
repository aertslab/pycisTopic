import json
import math
from itertools import chain

import numpy as np
import scipy

import pycisTopic.topic_modeling.tmtoolkit_lite as tmtoolkit_lite
from pycisTopic.topic_modeling.mallet_models import LDAMallet, LDAMalletFilenames


class JsonNumpyEncode(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def loglikelihood(nzw, ndz, alpha, eta):
    D = ndz.shape[0]
    n_topics = ndz.shape[1]
    vocab_size = nzw.shape[1]

    const_prior = (n_topics * math.lgamma(alpha) - math.lgamma(alpha * n_topics)) * D
    const_ll = (
        vocab_size * math.lgamma(eta) - math.lgamma(eta * vocab_size)
    ) * n_topics

    # calculate log p(w|z)
    topic_ll = 0
    for k in range(n_topics):
        sum = eta * vocab_size
        for w in range(vocab_size):
            if nzw[k, w] > 0:
                topic_ll = math.lgamma(nzw[k, w] + eta)
                sum += nzw[k, w]
        topic_ll -= math.lgamma(sum)

    # calculate log p(z)
    doc_ll = 0
    for d in range(D):
        sum = alpha * n_topics
        for k in range(n_topics):
            if ndz[d, k] > 0:
                doc_ll = math.lgamma(ndz[d, k] + alpha)
                sum += ndz[d, k]
        doc_ll -= math.lgamma(sum)

    ll = doc_ll - const_prior + topic_ll - const_ll
    return ll


def calculate_model_evaluation_stats(
    binary_accessibility_matrix: scipy.sparse.csr_matrix,
    output_prefix: str,
    n_topics: int,
    top_topics_coh: int = 5,
) -> None:
    """
    Calculate model evaluation statistics after running Mallet (McCallum, 2002) topic modeling.

    Parameters
    ----------
    binary_accessibility_matrix
        Binary accessibility sparse matrix with cells as columns, regions as rows,
         and 1 as value if a region is considered accessible in a cell (otherwise, 0).
    output_prefix
        Output prefix used for running topic modeling with Mallet.
    n_topics
        Number of topics used in the topic model created by Mallet.
        In combination with output_prefix, this allows to load the correct region
        topic counts and cell topic probabilties parquet files.
    top_topics_coh
        Number of topics to use to calculate the model coherence. For each model,
        the coherence will be calculated as the average of the top coherence values.
        Default: 5.

    Return
    ------
    None

    References
    ----------
    McCallum, A. K. (2002). Mallet: A machine learning for language toolkit. http://mallet.cs.umass.edu.

    """  # noqa: W505
    # Get distributions
    lda_mallet_filenames = LDAMalletFilenames(
        output_prefix=output_prefix, n_topics=n_topics
    )
    topic_word_distrib = LDAMallet.read_region_topic_counts_parquet_file_to_region_topic_probabilities(
        mallet_region_topic_counts_parquet_filename=lda_mallet_filenames.region_topic_counts_parquet_filename
    )
    doc_topic_distrib = LDAMallet.read_cell_topic_probabilities_parquet_file(
        mallet_cell_topic_probabilities_parquet_filename=lda_mallet_filenames.cell_topic_probabilities_parquet_filename
    )
    topic_word_counts = LDAMallet.read_region_topic_counts_parquet_file(
        mallet_region_topic_counts_parquet_filename=lda_mallet_filenames.region_topic_counts_parquet_filename
    )

    # Read used Mallet LDA parameters from JSON file.
    mallet_train_topics_parameters = LDAMallet.read_parameters_json_filename(
        lda_mallet_filenames.parameters_json_filename
    )

    if mallet_train_topics_parameters["n_topics"] != n_topics:
        raise ValueError(
            f"Number of topics does not match: {n_topics} vs {mallet_train_topics_parameters['n_topics']}."
        )

    alpha = mallet_train_topics_parameters["alpha"]
    alpha_by_topic = mallet_train_topics_parameters["alpha_by_topic"]
    eta = mallet_train_topics_parameters["eta"]
    eta_by_topic = mallet_train_topics_parameters["eta_by_topic"]

    ll_alpha = alpha / n_topics if alpha_by_topic else alpha
    ll_eta = eta / n_topics if eta_by_topic else eta

    # Model evaluation
    cell_cov = np.asarray(binary_accessibility_matrix.sum(axis=0)).astype(float)
    arun_2010 = tmtoolkit_lite.topicmod.evaluate.metric_arun_2010(
        topic_word_distrib=topic_word_distrib,
        doc_topic_distrib=doc_topic_distrib,
        doc_lengths=cell_cov,
    )
    cao_juan_2009 = tmtoolkit_lite.topicmod.evaluate.metric_cao_juan_2009(
        topic_word_distrib=topic_word_distrib,
    )
    mimno_2011 = tmtoolkit_lite.topicmod.evaluate.metric_coherence_mimno_2011(
        topic_word_distrib=topic_word_distrib,
        dtm=binary_accessibility_matrix.transpose(),
        top_n=20,
        eps=1e-12,
        normalize=True,
        return_mean=False,
    )

    doc_topic_counts = (doc_topic_distrib.T * (cell_cov)).T
    ll = loglikelihood(topic_word_counts, doc_topic_counts, ll_alpha, ll_eta)

    marg_topic = tmtoolkit_lite.topicmod.model_stats.marginal_topic_distrib(
        doc_topic_distrib=doc_topic_distrib,
        doc_lengths=cell_cov,
    )

    topic_ass = list(chain.from_iterable(topic_word_counts.sum(axis=1)[:, None]))

    metrics = {
        "Arun_2010": arun_2010,
        "Cao_Juan_2009": cao_juan_2009,
        "Mimno_2011": (
            np.mean(mimno_2011)
            if len(mimno_2011) <= top_topics_coh
            else np.mean(
                mimno_2011[
                    np.argpartition(mimno_2011, -top_topics_coh)[-top_topics_coh:]
                ]
            )
        ),
        "loglikelihood": ll,
        "coherence": mimno_2011,
        "marg_topic": marg_topic,
        "assignments": topic_ass,
    }

    with open(lda_mallet_filenames.model_stats_filename, "w") as fh:
        json.dump(
            metrics,
            fh,
            cls=JsonNumpyEncode,
        )
