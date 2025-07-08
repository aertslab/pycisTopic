raise NotImplementedError("LDA modeling is not yet implemented in pycisTopic v3.")

import logging
import os
import pickle
import sys
import time
import warnings
from itertools import chain
from typing import Any

import lda
import numpy as np
import pandas as pd
import ray
import scipy
import tmtoolkit


class CistopicLDAModel:
    """
    cisTopic LDA model class.

    :class:`cistopicLdaModel` contains model quality metrics (model coherence (adaptation from Mimno et al., 2011), log-likelihood (Griffiths and Steyvers, 2004), density-based (Cao Juan et al., 2009) and divergence-based (Arun et al., 2010)), topic quality metrics (coherence, marginal distribution and total number of assignments), cell-topic and topic-region distribution, model parameters and model dimensions.

    Parameters
    ----------
    metrics: pd.DataFrame
        :class:`pd.DataFrame` containing model quality metrics, including model coherence (adaptation from Mimno et al., 2011), log-likelihood and density and divergence-based methods (Cao Juan et al., 2009; Arun et al., 2010).
    coherence: pd.DataFrame
        :class:`pd.DataFrame` containing the coherence of each topic (Mimno et al., 2011).
    marginal_distribution: pd.DataFrame
        :class:`pd.DataFrame` containing the marginal distribution for each topic. It can be interpreted as the importance of each topic for the whole corpus.
    topic_ass: pd.DataFrame
        :class:`pd.DataFrame` containing the total number of assignments per topic.
    cell_topic: pd.DataFrame
        :class:`pd.DataFrame` containing the topic cell distributions, with cells as columns, topics as rows and the probability of each topic in each cell as values.
    topic_region: pd.DataFrame
        :class:`pd.DataFrame` containing the topic cell distributions, with topics as columns, regions as rows and the probability of each region in each topic as values.
    parameters: pd.DataFrame
        :class:`pd.DataFrame` containing parameters used for the model.
    n_cells: int
        Number of cells in the model.
    n_regions: int
        Number of regions in the model.
    n_topic: int
        Number of topics in the model.

    References
    ----------
    Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A. (2011). Optimizing semantic coherence in topic models. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing (pp. 262-272).

    Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. Proceedings of the National academy of Sciences, 101(suppl 1), 5228-5235.

    Cao, J., Xia, T., Li, J., Zhang, Y., & Tang, S. (2009). A density-based method for adaptive LDA model selection. Neurocomputing, 72(7-9), 1775-1781.

    Arun, R., Suresh, V., Madhavan, C. V., & Murthy, M. N. (2010). On finding the natural number of topics with latent dirichlet allocation: Some observations. In Pacific-Asia conference on knowledge discovery and data mining (pp. 391-402). Springer, Berlin, Heidelberg.

    """

    def __init__(
        self,
        metrics: pd.DataFrame,
        coherence: pd.DataFrame,
        marg_topic: pd.DataFrame,
        topic_ass: pd.DataFrame,
        cell_topic: pd.DataFrame,
        topic_region: pd.DataFrame,
        parameters: pd.DataFrame,
    ):
        self.metrics = metrics
        self.coherence = coherence
        self.marg_topic = marg_topic
        self.topic_ass = topic_ass
        self.cell_topic = cell_topic
        self.cell_topic_harmony = []
        self.topic_region = topic_region
        self.parameters = parameters
        self.n_cells = cell_topic.shape[1]
        self.n_regions = topic_region.shape[0]
        self.n_topic = cell_topic.shape[0]

    def __str__(self):
        descr = f"CistopicLDAModel with {self.n_topic} topics and n_cells × n_regions = {self.n_cells} × {self.n_regions}"
        return descr


def run_cgs_models(
    cistopic_obj: Any,
    n_topics: list[int],
    n_cpu: int = 1,
    n_iter: int = 150,
    random_state: int = 555,
    alpha: float = 50,
    alpha_by_topic: bool = True,
    eta: float = 0.1,
    eta_by_topic: bool = False,
    top_topics_coh: int = 5,
    save_path: str | None = None,
    **kwargs,
):
    """
    Run Latent Dirichlet Allocation using Gibbs Sampling as described in Griffiths and Steyvers, 2004.

    Parameters
    ----------
    cistopic_obj: CistopicObject
        A :class:`CistopicObject`. Note that cells/regions have to be filtered before running any LDA model.
    n_topics: list of int
        A list containing the number of topics to use in each model.
    n_cpu: int, optional
        Number of cpus to use for modelling. In this function parallelization is done per model, that is, one model will run entirely in a unique cpu. We recommend to set the number of cpus as the number of models that will be inferred, so all models start at the same time.
    n_iter: int, optional
        Number of iterations for which the Gibbs sampler will be run. Default: 150.
    random_state: int, optional
        Random seed to initialize the models. Default: 555.
    alpha: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic proportions. Default: 50.
    alpha_by_topic: bool, optional
        Boolean indicating whether the scalar given in alpha has to be divided by the number of topics. Default: True
    eta: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic multinomials. Default: 0.1.
    eta_by_topic: bool, optional
        Boolean indicating whether the scalar given in beta has to be divided by the number of topics. Default: False
    top_topics_coh: int, optional
        Number of topics to use to calculate the model coherence. For each model, the coherence will be calculated as the average of the top coherence values. Default: 5.
    save_path: str, optional
        Path to save models as independent files as they are completed. This is recommended for large data sets. Default: None.

    Return
    ------
    list of :class:`CistopicLDAModel`
        A list with cisTopic LDA models.

    References
    ----------
    Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. Proceedings of the National academy of Sciences, 101(suppl 1), 5228-5235.

    """
    binary_accessibility_matrix = scipy.sparse.csr_matrix(
        cistopic_obj.binary_matrix.transpose()
    )
    region_names = cistopic_obj.region_names
    cell_names = cistopic_obj.cell_names
    ray.init(num_cpus=n_cpu, **kwargs)
    model_list = ray.get(
        [
            run_cgs_model.remote(
                binary_accessibility_matrix,
                n_topics=n_topic,
                cell_names=cell_names,
                region_names=region_names,
                n_iter=n_iter,
                random_state=random_state,
                alpha=alpha,
                alpha_by_topic=alpha_by_topic,
                eta=eta,
                eta_by_topic=eta_by_topic,
                top_topics_coh=top_topics_coh,
                save_path=save_path,
            )
            for n_topic in n_topics
        ]
    )
    ray.shutdown()
    return model_list


@ray.remote
def run_cgs_model(
    binary_accessibility_matrix: scipy.sparse.csr_matrix,
    n_topics: int,
    cell_names: list[str],
    region_names: list[str],
    n_iter: int = 150,
    random_state: int = 555,
    alpha: float = 50,
    alpha_by_topic: bool = True,
    eta: float = 0.1,
    eta_by_topic: bool = False,
    top_topics_coh: int = 5,
    save_path: str = None,
):
    """
    Run Latent Dirichlet Allocation per model using Gibbs Sampling as described in Griffiths and Steyvers, 2004.

    Parameters
    ----------
    binary_accessibility_matrix: sparse.csr_matrix
        Binary sparse matrix containing cells as columns, regions as rows, and 1 if a regions is considered accessible on a cell (otherwise, 0).
    n_topics: int
        Number of topics to use in the model.
    cell_barcodes: list of str
        List containing cell names as ordered in the binary matrix columns.
    region_ids: list of str
        List containing region names as ordered in the binary matrix rows.
    n_iter: int, optional
        Number of iterations for which the Gibbs sampler will be run. Default: 150.
    random_state: int, optional
        Random seed to initialize the models. Default: 555.
    alpha: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic proportions. Default: 50.
    alpha_by_topic: bool, optional
        Boolean indicating whether the scalar given in alpha has to be divided by the number of topics. Default: True
    eta: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic multinomials. Default: 0.1.
    eta_by_topic: bool, optional
        Boolean indicating whether the scalar given in beta has to be divided by the number of topics. Default: False
    top_topics_coh: int, optional
        Number of topics to use to calculate the model coherence. For each model, the coherence will be calculated as the average of the top coherence values. Default: 5.
    save_path: str, optional
        Path to save models as independent files as they are completed. This is recommended for large data sets. Default: None.

    Return
    ------
    CistopicLDAModel
        A cisTopic LDA model.

    References
    ----------
    Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. Proceedings of the National academy of Sciences, 101(suppl 1), 5228-5235.

    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    # Suppress lda logger
    lda_log = logging.getLogger("lda")
    lda_log.addHandler(logging.NullHandler())
    lda_log.propagate = False
    warnings.filterwarnings("ignore")

    lda_alpha = alpha / n_topics if alpha_by_topic else alpha
    lda_eta = eta / n_topics if eta_by_topic else eta

    model = lda.LDA(
        n_topics=n_topics,
        n_iter=n_iter,
        random_state=random_state,
        alpha=lda_alpha,
        eta=lda_eta,
        refresh=n_iter,
    )

    # Running model
    log.info(f"Running model with {n_topics} topics")
    start_time = time.time()
    model.fit(binary_accessibility_matrix)
    end_time = time.time() - start_time

    # Model evaluation
    arun_2010 = tmtoolkit.topicmod.evaluate.metric_arun_2010(
        model.topic_word_,
        model.doc_topic_,
        np.asarray(binary_accessibility_matrix.sum(axis=1)).astype(float),
    )
    cao_juan_2009 = tmtoolkit.topicmod.evaluate.metric_cao_juan_2009(model.topic_word_)
    mimno_2011 = tmtoolkit.topicmod.evaluate.metric_coherence_mimno_2011(
        model.topic_word_,
        dtm=binary_accessibility_matrix,
        top_n=20,
        eps=1e-12,
        normalize=True,
        return_mean=False,
    )
    ll = loglikelihood(model.nzw_, model.ndz_, lda_alpha, lda_eta)

    # Organize data
    if len(mimno_2011) <= top_topics_coh:
        metrics = pd.DataFrame(
            [arun_2010, cao_juan_2009, np.mean(mimno_2011), ll],
            index=["Arun_2010", "Cao_Juan_2009", "Mimno_2011", "loglikelihood"],
            columns=["Metric"],
        ).transpose()
    else:
        metrics = pd.DataFrame(
            [
                arun_2010,
                cao_juan_2009,
                np.mean(
                    mimno_2011[
                        np.argpartition(mimno_2011, -top_topics_coh)[-top_topics_coh:]
                    ]
                ),
                ll,
            ],
            index=["Arun_2010", "Cao_Juan_2009", "Mimno_2011", "loglikelihood"],
            columns=["Metric"],
        ).transpose()
    coherence = pd.DataFrame(
        [range(1, n_topics + 1), mimno_2011], index=["Topic", "Mimno_2011"]
    ).transpose()
    marg_topic = pd.DataFrame(
        [
            range(1, n_topics + 1),
            list(
                chain.from_iterable(
                    tmtoolkit.topicmod.model_stats.marginal_topic_distrib(
                        model.doc_topic_, binary_accessibility_matrix.sum(axis=1)
                    ).tolist()
                )
            ),
        ],
        index=["Topic", "Marg_Topic"],
    ).transpose()
    topic_ass = pd.DataFrame.from_records(
        [range(1, n_topics + 1), model.nz_], index=["Topic", "Assignments"]
    ).transpose()
    cell_topic = pd.DataFrame.from_records(
        model.doc_topic_,
        index=cell_names,
        columns=["Topic" + str(i) for i in range(1, n_topics + 1)],
    ).transpose()
    topic_region = pd.DataFrame.from_records(
        model.topic_word_,
        columns=region_names,
        index=["Topic" + str(i) for i in range(1, n_topics + 1)],
    ).transpose()
    parameters = pd.DataFrame(
        [
            "lda",
            n_topics,
            n_iter,
            random_state,
            alpha,
            alpha_by_topic,
            eta,
            eta_by_topic,
            top_topics_coh,
            end_time,
        ],
        index=[
            "package",
            "n_topics",
            "n_iter",
            "random_state",
            "alpha",
            "alpha_by_topic",
            "eta",
            "eta_by_topic",
            "top_topics_coh",
            "time",
        ],
        columns=["Parameter"],
    )
    # Create object
    model = CistopicLDAModel(
        metrics, coherence, marg_topic, topic_ass, cell_topic, topic_region, parameters
    )
    log.info(f"Model with {n_topics} topics done!")
    if isinstance(save_path, str):
        log.info(f"Saving model with {n_topics} topics at {save_path}")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(os.path.join(save_path, f"Topic{n_topics}.pkl"), "wb") as f:
            pickle.dump(model, f)
    return model
