from __future__ import annotations

import json
from itertools import chain

import numpy as np
import scipy
from scipy.special import gammaln

import pycisTopic.topic_modeling.tmtoolkit_lite as tmtoolkit_lite
from pycisTopic.topic_modeling.topic_models import TopicModelFilenames, load_topic_model_backend


class JsonNumpyEncode(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def loglikelihood(
    nzw: np.ndarray,
    ndz: np.ndarray,
    alpha: float | list[float] | np.ndarray,
    eta: float | list[float] | np.ndarray,
) -> float:
    """Calculate the LDA log likelihood for scalar or vector hyperparameters."""
    nzw = np.asarray(nzw, dtype=np.float64)
    ndz = np.asarray(ndz, dtype=np.float64)

    alpha_array = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if alpha_array.size == 1:
        alpha_array = np.repeat(alpha_array.item(), ndz.shape[1])
    elif alpha_array.size != ndz.shape[1]:
        raise ValueError(
            f"alpha length {alpha_array.size} does not match topic count {ndz.shape[1]}."
        )

    eta_array = np.asarray(eta, dtype=np.float64).reshape(-1)
    if eta_array.size == 1:
        eta_array = np.repeat(eta_array.item(), nzw.shape[1])
    elif eta_array.size != nzw.shape[1]:
        raise ValueError(
            f"eta length {eta_array.size} does not match vocab size {nzw.shape[1]}."
        )

    alpha_sum = float(alpha_array.sum())
    eta_sum = float(eta_array.sum())
    doc_lengths = ndz.sum(axis=1)
    topic_lengths = nzw.sum(axis=1)

    doc_ll = np.sum(
        gammaln(alpha_sum)
        - gammaln(doc_lengths + alpha_sum)
        + np.sum(gammaln(ndz + alpha_array) - gammaln(alpha_array), axis=1)
    )
    topic_ll = np.sum(
        gammaln(eta_sum)
        - gammaln(topic_lengths + eta_sum)
        + np.sum(gammaln(nzw + eta_array) - gammaln(eta_array), axis=1)
    )

    return float(doc_ll + topic_ll)


def calculate_model_evaluation_stats(
    binary_accessibility_matrix: scipy.sparse.csr_matrix,
    output_prefix: str,
    n_topics: int,
    top_topics_coh: int = 5,
) -> None:
    """
    Calculate topic model evaluation statistics from backend-agnostic v3 artifacts.

    Parameters
    ----------
    binary_accessibility_matrix
        Binary accessibility sparse matrix with cells as columns and regions as rows.
    output_prefix
        Output prefix used for running topic modeling.
    n_topics
        Number of topics used in the topic model.
    top_topics_coh
        Number of topics to use to calculate the model coherence.

    """
    filenames = TopicModelFilenames(output_prefix=output_prefix, n_topics=n_topics)
    backend_cls = load_topic_model_backend(output_prefix=output_prefix, n_topics=n_topics)

    topic_word_distrib = (
        backend_cls.read_region_topic_counts_parquet_file_to_region_topic_probabilities(
            region_topic_counts_parquet_filename=filenames.region_topic_counts_parquet_filename
        )
    )
    doc_topic_distrib = backend_cls.read_cell_topic_probabilities_parquet_file(
        cell_topic_probabilities_parquet_filename=filenames.cell_topic_probabilities_parquet_filename
    )
    topic_word_counts = backend_cls.read_region_topic_counts_parquet_file(
        region_topic_counts_parquet_filename=filenames.region_topic_counts_parquet_filename
    )
    parameters = backend_cls.read_parameters_json_filename(
        filenames.parameters_json_filename
    )

    if parameters["n_topics"] != n_topics:
        raise ValueError(
            f"Number of topics does not match: {n_topics} vs {parameters['n_topics']}."
        )

    ll_alpha, ll_eta = _resolve_loglikelihood_hyperparameters(
        parameters=parameters,
        n_topics=n_topics,
    )

    cell_cov = np.asarray(binary_accessibility_matrix.sum(axis=0)).astype(float).ravel()
    arun_2010 = tmtoolkit_lite.topicmod.evaluate.metric_arun_2010(
        topic_word_distrib=topic_word_distrib,
        doc_topic_distrib=doc_topic_distrib,
        doc_lengths=cell_cov,
    )
    cao_juan_2009 = tmtoolkit_lite.topicmod.evaluate.metric_cao_juan_2009(
        topic_word_distrib=topic_word_distrib,
    )
    coherence_top_n = max(1, min(20, topic_word_distrib.shape[1]))
    mimno_2011 = tmtoolkit_lite.topicmod.evaluate.metric_coherence_mimno_2011(
        topic_word_distrib=topic_word_distrib,
        dtm=binary_accessibility_matrix.transpose(),
        top_n=coherence_top_n,
        eps=1e-12,
        normalize=True,
        return_mean=False,
    )

    doc_topic_counts = (doc_topic_distrib.T * cell_cov).T
    ll = loglikelihood(topic_word_counts, doc_topic_counts, ll_alpha, ll_eta)

    marg_topic = tmtoolkit_lite.topicmod.model_stats.marginal_topic_distrib(
        doc_topic_distrib=doc_topic_distrib,
        doc_lengths=cell_cov,
    )
    topic_assignments = list(chain.from_iterable(topic_word_counts.sum(axis=1)[:, None]))

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
        "assignments": topic_assignments,
    }

    with open(filenames.model_stats_filename, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, cls=JsonNumpyEncode)


def _resolve_loglikelihood_hyperparameters(
    parameters: dict,
    n_topics: int,
) -> tuple[float | list[float], float | list[float]]:
    backend_name = str(parameters.get("backend", "mallet")).lower()

    if backend_name == "tomotopy":
        return parameters["alpha"], parameters["eta"]

    alpha = parameters["alpha"]
    eta = parameters["eta"]
    alpha_by_topic = parameters.get("alpha_by_topic", True)
    eta_by_topic = parameters.get("eta_by_topic", False)

    ll_alpha = alpha / n_topics if alpha_by_topic else alpha
    ll_eta = eta / n_topics if eta_by_topic else eta
    return ll_alpha, ll_eta
