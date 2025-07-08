import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from pycisTopic.topic_modeling.mallet_models import LDAMalletFilenames


def scale(X: np.ndarray) -> np.ndarray:
    return (X - X.min()) / (X.max() - X.min())

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
        lda_mallet_filenames = LDAMalletFilenames(
            output_prefix=output_prefix, n_topics=n_topic
        )
        with open(lda_mallet_filenames.model_stats_filename) as infile:
            metrics_per_topic[n_topic] = TopicModelMetrics(
                **json.load(infile)
            )

    metrics = {
        "Inv_Arun_2010": scale(
            -np.array([
                metrics_per_topic[t].Arun_2010
                for t in sorted(n_topics)
            ])
        ),
        "Inv_Cao_Juan_2009": scale(
            -np.array([
                metrics_per_topic[t].Cao_Juan_2009
                for t in sorted(n_topics)
            ])
        ),
        "Mimno_2011": scale(
            np.array([
                metrics_per_topic[t].Mimno_2011
                for t in sorted(n_topics)
            ])
        ),
        "Loglikelihood": scale(
            np.array([
                metrics_per_topic[t].loglikelihood
                for t in sorted(n_topics)
            ])
        )
    }

    fig, ax = plt.subplots(figsize = (8, 8))
    for metric, values in metrics.items():
        _ = ax.plot(
            sorted(n_topics),
            values,
            linestyle="--",
            marker="o",
            label=metric
        )
    ax.grid(True)
    ax.set_axisbelow(True)
    _ = ax.set_xlabel("Number of topics")
    _ = ax.set_ylabel("Scaled metric")
    _ = ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.model_evaluation_stats.{plot_file_format}")
