from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from pycisTopic.topic_modeling.topic_models import TopicModelFilenames, load_topic_model_backend


def scale(values: np.ndarray) -> np.ndarray:
    min_value = values.min()
    max_value = values.max()
    if min_value == max_value:
        return np.ones_like(values, dtype=np.float64)
    return (values - min_value) / (max_value - min_value)


@dataclass
class TopicModelMetrics:
    Arun_2010: float
    Cao_Juan_2009: float
    Mimno_2011: float
    loglikelihood: float
    coherence: list[float]
    marg_topic: list[float]
    assignments: list[float]


def plot_stats(
    output_prefix: str,
    n_topics: list[int],
    plot_file_format: str = "png",
):
    """
    Plot topic modeling evaluation statistics.

    Parameters
    ----------
    output_prefix
        Output prefix used for running topic modeling with Mallet.
    n_topics
        List of number of topics to plot statistics for.
    plot_file_format
        The file format of the plot, default is png.

    Retrun
    ------
    None

    """
    metrics_per_topic: dict[int, TopicModelMetrics] = {}
    for n_topic in n_topics:
        load_topic_model_backend(output_prefix=output_prefix, n_topics=n_topic)
        filenames = TopicModelFilenames(output_prefix=output_prefix, n_topics=n_topic)
        with open(filenames.model_stats_filename, encoding="utf-8") as infile:
            metrics_per_topic[n_topic] = TopicModelMetrics(**json.load(infile))

    sorted_topics = sorted(n_topics)
    metrics = {
        "Inv_Arun_2010": scale(
            -np.asarray([metrics_per_topic[t].Arun_2010 for t in sorted_topics])
        ),
        "Inv_Cao_Juan_2009": scale(
            -np.asarray([metrics_per_topic[t].Cao_Juan_2009 for t in sorted_topics])
        ),
        "Mimno_2011": scale(
            np.asarray([metrics_per_topic[t].Mimno_2011 for t in sorted_topics])
        ),
        "Loglikelihood": scale(
            np.asarray([metrics_per_topic[t].loglikelihood for t in sorted_topics])
        ),
    }

    figure, axis = plt.subplots(figsize=(8, 8))
    for metric, values in metrics.items():
        axis.plot(
            sorted_topics,
            values,
            linestyle="--",
            marker="o",
            label=metric,
        )
    axis.grid(True)
    axis.set_axisbelow(True)
    axis.set_xlabel("Number of topics")
    axis.set_ylabel("Scaled metric")
    axis.legend()
    figure.tight_layout()
    figure.savefig(f"{output_prefix}.model_evaluation_stats.{plot_file_format}")
