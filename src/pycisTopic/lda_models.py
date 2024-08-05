from __future__ import annotations

import logging
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time
import warnings
from itertools import chain
from typing import TYPE_CHECKING

import lda
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import ray
import tmtoolkit
from gensim import matutils, utils
from gensim.models import basemodel
from pycisTopic.utils import loglikelihood, subset_list
from scipy import sparse

if TYPE_CHECKING:
    from pycisTopic.cistopic_class import CistopicObject


class CistopicLDAModel:
    """
    cisTopic LDA model class

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
    cistopic_obj: CistopicObject,
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
    binary_matrix = sparse.csr_matrix(cistopic_obj.binary_matrix.transpose())
    region_names = cistopic_obj.region_names
    cell_names = cistopic_obj.cell_names
    ray.init(num_cpus=n_cpu, **kwargs)
    model_list = ray.get(
        [
            run_cgs_model.remote(
                binary_matrix,
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
    binary_matrix: sparse.csr_matrix,
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
    binary_matrix: sparse.csr_matrix
        Binary sparse matrix containing cells as columns, regions as rows, and 1 if a regions is considered accessible on a cell (otherwise, 0).
    n_topics: int
        Number of topics to use in the model.
    cell_names: list of str
        List containing cell names as ordered in the binary matrix columns.
    region_names: list of str
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

    # Set models
    if alpha_by_topic and eta_by_topic:
        model = lda.LDA(
            n_topics=n_topics,
            n_iter=n_iter,
            random_state=random_state,
            alpha=alpha / n_topics,
            eta=eta / n_topics,
            refresh=n_iter,
        )
    elif alpha_by_topic and eta_by_topic is False:
        model = lda.LDA(
            n_topics=n_topics,
            n_iter=n_iter,
            random_state=random_state,
            alpha=alpha / n_topics,
            eta=eta,
            refresh=n_iter,
        )
    elif alpha_by_topic is False and eta_by_topic is True:
        model = lda.LDA(
            n_topics=n_topics,
            n_iter=n_iter,
            random_state=random_state,
            alpha=alpha,
            eta=eta / n_topics,
            refresh=n_iter,
        )
    else:
        model = lda.LDA(
            n_topics=n_topics,
            n_iter=n_iter,
            random_state=random_state,
            alpha=alpha,
            eta=eta,
            refresh=n_iter,
        )

    # Running model
    log.info(f"Running model with {n_topics} topics")
    start_time = time.time()
    model.fit(binary_matrix)
    end_time = time.time() - start_time

    # Model evaluation
    arun_2010 = tmtoolkit.topicmod.evaluate.metric_arun_2010(
        model.topic_word_,
        model.doc_topic_,
        np.asarray(binary_matrix.sum(axis=1)).astype(float),
    )
    cao_juan_2009 = tmtoolkit.topicmod.evaluate.metric_cao_juan_2009(model.topic_word_)
    mimno_2011 = tmtoolkit.topicmod.evaluate.metric_coherence_mimno_2011(
        model.topic_word_,
        dtm=binary_matrix,
        top_n=20,
        eps=1e-12,
        normalize=True,
        return_mean=False,
    )
    ll = loglikelihood(model.nzw_, model.ndz_, alpha, eta)

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
                        model.doc_topic_, binary_matrix.sum(axis=1)
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


class LDAMallet(utils.SaveLoad, basemodel.BaseTopicModel):
    """
    Wrapper class to run LDA models with Mallet. This class has been adapted from gensim (https://github.com/RaRe-Technologies/gensim/blob/27bbb7015dc6bbe02e00bb1853e7952ac13e7fe0/gensim/models/wrappers/ldamallet.py).

    Parameters
    ----------
    num_topics: int
        The number of topics to use in the model.
    corpus: iterable of iterable of (int, int), optional
        Collection of texts in BoW format. Default: None.
    alpha: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic proportions. Default: 50.
    id2word : :class:`gensim.utils.FakeDict`, optional
        Mapping between tokens ids and words from corpus, if not specified - will be inferred from `corpus`. Default: None.
    n_cpu : int, optional
        Number of threads that will be used for training. Default: 1.
    tmp_dir : str, optional
        tmp_dir for produced temporary files. Default: None.
    optimize_interval : int, optional
        Optimize hyperparameters every `optimize_interval` iterations (sometimes leads to Java exception 0 to switch off hyperparameter optimization). Default: 0.
    iterations : int, optional
        Number of training iterations. Default: 150.
    topic_threshold : float, optional
        Threshold of the probability above which we consider a topic. Default: 0.0.
    random_seed: int, optional
        Random seed to ensure consistent results, if 0 - use system clock. Default: 555.
    mallet_path: str
        Path to the mallet binary (e.g. /xxx/Mallet/bin/mallet). Default: "mallet".

    """

    @staticmethod
    def create_regions_topics_count_matrix(
        mallet_region_topics_count_filename: str,
    ) -> np.ndarray:
        """
        Create regions vs topics count matrix from Mallet region-topics count file.

        Parameters
        ----------
        mallet_region_topics_counts_filename
            Mallet region-topics count file.

        Returns
        -------
        Regions vs topics count matrix.

        """
        no_region_ids = -1
        no_topics = -1
        region_id_topic_counts = []

        with open(mallet_region_topics_count_filename, "r") as fh:
            # Column 0: order in which region ID idx was seen in the input corpus file.
            # Column 1: region ID idx
            # Column 3-n: "topic:count" pairs
            #
            # Mallet region-topics count file example:
            # ----------------------------------------
            #
            # 0 12 3:94 11:84 1:84 18:75 17:36 0:31 13:25 4:23 6:22 12:16 9:10 10:6 15:3 7:2 8:1
            # 1 28 8:368 15:267 3:267 17:255 0:245 10:227 16:216 19:201 7:92 18:85 1:58 14:52 9:31 6:17 13:6 2:3
            # 2 33 8:431 16:418 10:354 3:257 17:211 12:146 7:145 9:115 4:108 13:106 18:66 1:60 15:45 6:45 19:33 5:19 14:12 0:1
            # 3 35 7:284 18:230 15:199 10:191 16:164 0:114 4:112 19:107 12:104 13:68 3:49 9:35 1:28 11:25 5:20 17:17 6:11 14:2 8:1
            # 4 57 8:192 3:90 19:88 1:69 18:67 2:63 10:62 17:38 15:37 13:10 4:9 12:2 9:1
            for line in fh:
                columns = line.rstrip().split()
                # Get region ID index from second column.
                region_id_idx = int(columns[1])
                # Get topic index and counts from column 3 till the end by splitting: "topic:count" pairs.
                topics_counts = [
                    (int(topic), int(count))
                    for topic, count in [
                        topic_counts.split(":", 1) for topic_counts in columns[2:]
                    ]
                ]
                # Get topic indices.
                topics_idx = np.array([topic for topic, count in topics_counts])
                # Get counts.
                counts = np.array([count for topic, count in topics_counts])
                # Store region ID index, topics indices and counts till we know how many regions and topics we have.
                region_id_topic_counts.append((region_id_idx, topics_idx, counts))

                # Keep track of the highest seen region ID index and topic index (0-based).
                no_region_ids = max(region_id_idx, no_region_ids)
                no_topics = max(topics_idx.max(), no_topics)

        # Add 1 to region IDs and topics counts to account for start at 0.
        no_region_ids += 1
        no_topics += 1

        # Create regions topics count matrix and populate it.
        regions_topics_counts = np.zeros((no_topics, no_region_ids), dtype=np.int32)
        for region_idx, topics_idx, counts in region_id_topic_counts:
            regions_topics_counts[topics_idx, region_idx] = counts

        return regions_topics_counts

    def __init__(
        self,
        num_topics: int,
        corpus: list | None = None,
        alpha: float = 50,
        eta: float = 0.1,
        id2word: utils.FakeDict = None,
        n_cpu: int = 1,
        tmp_dir: str = None,
        optimize_interval: int = 0,
        iterations: int = 150,
        topic_threshold: float = 0.0,
        random_seed: int = 555,
        reuse_corpus: bool = False,
        mallet_path: str = "mallet",
    ):
        logger = logging.getLogger("LDAMalletWrapper")
        if id2word is None:
            logger.warning(
                "No id2word mapping provided; initializing from corpus, assuming identity"
            )
            self.num_terms = utils.get_max_id(corpus) + 1
        else:
            self.num_terms = id2word.num_terms

        if self.num_terms == 0:
            raise ValueError("Cannot compute LDA over an empty collection (no terms)")

        self.num_topics = num_topics
        self.topic_threshold = topic_threshold
        self.alpha = alpha
        self.eta = eta
        self.tmp_dir = tmp_dir if tmp_dir else tempfile.gettempdir()
        self.random_label = hex(random.randint(0, 0xFFFFFF))[2:]
        self.n_cpu = n_cpu
        self.optimize_interval = optimize_interval
        self.iterations = iterations
        self.random_seed = random_seed
        self.mallet_path = mallet_path
        if corpus is not None:
            self.train(corpus, reuse_corpus)

    def corpus_to_mallet(self, corpus, file_like):
        """
        Convert `corpus` to Mallet format and write it to `file_like` descriptor.

        Parameters
        ----------
        corpus
            iterable of iterable of (int, int)
            Collection of texts in BoW format.
        file_like
             Writable file-like object in text mode.

        Returns
        -------
        None.

        """
        # Iterate over each cell ("document").
        for doc_idx, doc in enumerate(corpus):
            # Get all accessible regions for the current cell.
            tokens = chain.from_iterable([str(token_id)] for token_id, _cnt in doc)

            file_like.write(f'{doc_idx}\t0\t{" ".join(tokens)}\n')

    def convert_input(self, corpus):
        """
        Convert corpus to Mallet format and save it to a temporary text file.

        Parameters
        ----------
        corpus
            iterable of iterable of (int, int)
            Collection of texts in BoW format.

        Returns
        -------
        None.

        """
        logger = logging.getLogger("LDAMalletWrapper")

        logger.info(f"Serializing temporary corpus to {self.fcorpustxt()}")

        with utils.open(self.fcorpustxt(), "wt") as fh:
            self.corpus_to_mallet(corpus, fh)

        cmd = [
            self.mallet_path,
            "import-file",
            "--preserve-case",
            "--keep-sequence",
            "--token-regex",
            "\\S+",
            "--input",
            self.fcorpustxt(),
            "--output",
            self.fcorpusmallet(),
        ]

        logger.info(
            f"Converting temporary corpus to MALLET format with: {' '.join(cmd)}"
        )
        try:
            subprocess.check_output(args=cmd, shell=False, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
            )

    def train(self, corpus, reuse_corpus):
        """
        Train Mallet LDA.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Corpus in BoW format
        reuse_corpus: bool, optional
            Whether to reuse the mallet corpus in the tmp directory. Default: False

        """
        logger = logging.getLogger("LDAMalletWrapper")
        if os.path.isfile(self.fcorpusmallet()) is False or reuse_corpus is False:
            self.convert_input(corpus)
        else:
            logger.info("MALLET corpus already exists, training model")

        cmd = [
            self.mallet_path,
            "train-topics",
            "--input",
            self.fcorpusmallet(),
            "--num-topics",
            str(self.num_topics),
            "--alpha",
            str(self.alpha),
            "--beta",
            str(self.eta),
            "--optimize-interval",
            str(self.optimize_interval),
            "--num-threads",
            str(self.n_cpu),
            "--output-state",
            self.fstate(),
            "--output-doc-topics",
            self.fdoctopics(),
            "--output-topic-keys",
            self.ftopickeys(),
            "--num-iterations",
            str(self.iterations),
            "--inferencer-filename",
            self.finferencer(),
            "--doc-topics-threshold",
            str(self.topic_threshold),
            "--random-seed",
            str(self.random_seed),
        ]

        start = time.time()
        logger.info(f"Training MALLET LDA with: {' '.join(cmd)}")
        try:
            subprocess.check_output(args=cmd, shell=False, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
            )
        self.word_topics = self.load_word_topics()
        self.time = time.time() - start

    def load_word_topics(self):
        """
        Load words X topics matrix from :meth:`gensim.models.wrappers.LDAMallet.LDAMallet.fstate` file.

        Returns
        -------
        np.ndarray
            Matrix words X topics.

        """
        logger = logging.getLogger("LDAMalletWrapper")
        logger.info("loading assigned topics from %s", self.fstate())
        word_topics = np.zeros((self.num_topics, self.num_terms), dtype=np.float64)

        with utils.open(self.fstate(), "rb") as fin:
            _ = next(fin)  # header
            self.alpha = np.fromiter(next(fin).split()[2:], dtype=float)
            assert (
                len(self.alpha) == self.num_topics
            ), "Mismatch between MALLET vs. requested topics"

        # Get occurrence of each found topic-region combination:
        #   - Get region (type) and topic column from Mallet state file.
        #   - Count occurrence of each topic-region combination.
        topic_region_occurrence_df_pl = (
            pl.read_csv(
                self.fstate(),
                separator=" ",
                has_header=False,
                skip_rows=3,
                columns=[4, 5],
                new_columns=["region", "topic"],
            )
            .lazy()
            .group_by(["topic", "region"])
            .agg(pl.len().cast(pl.UInt32).alias("occurrence"))
            .collect()
        )

        # Fill in word topics matrix values.
        word_topics[
            topic_region_occurrence_df_pl.get_column("topic"),
            topic_region_occurrence_df_pl.get_column("region"),
        ] = topic_region_occurrence_df_pl.get_column("occurrence")

        return word_topics

    def get_topics(self) -> np.ndarray:
        """
        Get the region-topic probability matrix learned during inference.

        Returns
        -------
        The probability for each region in each topic, shape (no_regions, no_topics).

        """
        regions_topics_counts = np.asarray(self.word_topics, np.float64)

        # Create regions topics frequency matrix by dividing all count values for topic
        # by total counts for that topic.
        regions_topics_frequency = (
            regions_topics_counts / regions_topics_counts.sum(axis=1)[:, None]
        ).astype(np.float32)

        return regions_topics_frequency

    def fcorpustxt(self):
        """
        Get path to corpus text file.

        Returns
        -------
        str
            Path to corpus text file.

        """
        return os.path.join(self.tmp_dir, "corpus.txt")

    def fcorpusmallet(self):
        """
        Get path to corpus.mallet file.

        Returns
        -------
        str
            Path to corpus.mallet file.

        """
        return os.path.join(self.tmp_dir, "corpus.mallet")

    def fstate(self):
        """
        Get path to temporary file.

        Returns
        -------
        str
            Path to file.

        """
        return os.path.join(self.tmp_dir, f"{self.random_label}_state.mallet.gz")

    def fdoctopics(self):
        """
        Get path to document topic text file.

        Returns
        -------
        str
            Path to document topic text file.

        """
        return os.path.join(self.tmp_dir, f"{self.random_label}_doctopics.txt")

    def finferencer(self):
        """
        Get path to inferencer.mallet file.

        Returns
        -------
        str
            Path to inferencer.mallet file.

        """
        return os.path.join(self.tmp_dir, f"{self.random_label}_inferencer.mallet")

    def ftopickeys(self):
        """
        Get path to topic keys text file.

        Returns
        -------
        str
            Path to topic keys text file.

        """
        return os.path.join(self.tmp_dir, f"{self.random_label}_topickeys.txt")


def run_cgs_models_mallet(
    cistopic_obj: CistopicObject,
    n_topics: list[int],
    n_cpu: int = 1,
    n_iter: int = 150,
    random_state: int = 555,
    alpha: float = 50.0,
    alpha_by_topic: bool = True,
    eta: float = 0.1,
    eta_by_topic: bool = False,
    top_topics_coh: int = 5,
    tmp_path: str = None,
    save_path: str = None,
    reuse_corpus: bool = False,
    mallet_path: str = "mallet",
):
    """
    Run Latent Dirichlet Allocation per model as implemented in Mallet (McCallum, 2002).

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
    tmp_path: str, optional
        Path to a temporary folder for Mallet. Default: None.
    save_path: str, optional
        Path to save models as independent files as they are completed. This is recommended for large data sets. Default: None.
    reuse_corpus: bool, optional
        Whether to reuse the mallet corpus in the tmp directory. Default: False
    mallet_path: str
        Path to Mallet binary (e.g. "/xxx/Mallet/bin/mallet"). Default: "mallet".

    Return
    ------
    list of :class:`CistopicLDAModel`
        A list with cisTopic LDA models.

    References
    ----------
    McCallum, A. K. (2002). Mallet: A machine learning for language toolkit. http://mallet.cs.umass.edu.

    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    binary_matrix = cistopic_obj.binary_matrix
    region_names = cistopic_obj.region_names
    cell_names = cistopic_obj.cell_names

    log.info("Formatting input to corpus")
    corpus = matutils.Sparse2Corpus(binary_matrix)
    id2word = utils.FakeDict(len(region_names))

    model_list = [
        run_cgs_model_mallet(
            binary_matrix=binary_matrix,
            corpus=corpus,
            id2word=id2word,
            n_topics=n_topic,
            cell_names=cell_names,
            region_names=region_names,
            n_cpu=n_cpu,
            n_iter=n_iter,
            random_state=random_state,
            alpha=alpha,
            alpha_by_topic=alpha_by_topic,
            eta=eta,
            eta_by_topic=eta_by_topic,
            top_topics_coh=top_topics_coh,
            tmp_path=tmp_path,
            save_path=save_path,
            reuse_corpus=reuse_corpus,
            mallet_path=mallet_path,
        )
        for n_topic in n_topics
    ]
    return model_list


def run_cgs_model_mallet(
    binary_matrix: sparse.csr_matrix,
    corpus: list,
    id2word: utils.FakeDict,
    n_topics: list[int],
    cell_names: list[str],
    region_names: list[str],
    n_cpu: int = 1,
    n_iter: int = 500,
    random_state: int = 555,
    alpha: float = 50,
    alpha_by_topic: bool = True,
    eta: float = 0.1,
    eta_by_topic: bool = False,
    top_topics_coh: int = 5,
    tmp_path: str = None,
    save_path: str = None,
    reuse_corpus: bool = False,
    mallet_path: str = "mallet",
):
    """
    Run Latent Dirichlet Allocation in a model as implemented in Mallet (McCallum, 2002).

    Parameters
    ----------
    binary_matrix: sparse.csr_matrix
        Binary sparse matrix containing cells as columns, regions as rows, and 1 if a regions is considered accessible on a cell (otherwise, 0).
    n_topics: list of int
        A list containing the number of topics to use in each model.
    cell_names: list of str
        List containing cell names as ordered in the binary matrix columns.
    region_names: list of str
        List containing region names as ordered in the binary matrix rows.
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
    tmp_path: str, optional
        Path to a temporary folder for Mallet. Default: None.
    save_path: str, optional
        Path to save models as independent files as they are completed. This is recommended for large data sets. Default: None.
    reuse_corpus: bool, optional
        Whether to reuse the mallet corpus in the tmp directory. Default: False
    mallet_path: str
        Path to Mallet binary (e.g. "/xxx/Mallet/bin/mallet"). Default: "mallet".

    Return
    ------
    CistopicLDAModel
        A cisTopic LDA model.

    References
    ----------
    McCallum, A. K. (2002). Mallet: A machine learning for language toolkit. http://mallet.cs.umass.edu.

    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    # Set models
    if not alpha_by_topic:
        alpha = alpha * n_topics
    if eta_by_topic:
        eta = eta / n_topics

    # Running model
    start = time.time()
    log.info(f"Running model with {n_topics} topics")
    model = LDAMallet(
        corpus=corpus,
        id2word=id2word,
        num_topics=n_topics,
        iterations=n_iter,
        alpha=alpha,
        eta=eta,
        n_cpu=n_cpu,
        tmp_dir=tmp_path,
        random_seed=random_state,
        reuse_corpus=reuse_corpus,
        mallet_path=mallet_path,
    )
    end_time = time.time() - start

    # Get distributions
    topic_word = model.get_topics()
    doc_topic = (
        pd.read_csv(model.fdoctopics(), header=None, sep="\t").iloc[:, 2:].to_numpy()
    )

    # Model evaluation
    cell_cov = np.asarray(binary_matrix.sum(axis=0)).astype(float)
    arun_2010 = tmtoolkit.topicmod.evaluate.metric_arun_2010(
        topic_word, doc_topic, cell_cov
    )
    cao_juan_2009 = tmtoolkit.topicmod.evaluate.metric_cao_juan_2009(topic_word)
    mimno_2011 = tmtoolkit.topicmod.evaluate.metric_coherence_mimno_2011(
        topic_word,
        dtm=binary_matrix.transpose(),
        top_n=20,
        eps=1e-12,
        normalize=True,
        return_mean=False,
    )
    topic_word_assig = model.word_topics
    doc_topic_assig = (doc_topic.T * (cell_cov)).T
    ll = loglikelihood(topic_word_assig, doc_topic_assig, alpha, eta)

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
            tmtoolkit.topicmod.model_stats.marginal_topic_distrib(doc_topic, cell_cov),
        ],
        index=["Topic", "Marg_Topic"],
    ).transpose()
    topic_ass = pd.DataFrame.from_records(
        [
            range(1, n_topics + 1),
            list(chain.from_iterable(model.word_topics.sum(axis=1)[:, None])),
        ],
        index=["Topic", "Assignments"],
    ).transpose()
    cell_topic = pd.DataFrame.from_records(
        doc_topic,
        index=cell_names,
        columns=["Topic" + str(i) for i in range(1, n_topics + 1)],
    ).transpose()
    topic_region = pd.DataFrame.from_records(
        topic_word,
        columns=region_names,
        index=["Topic" + str(i) for i in range(1, n_topics + 1)],
    ).transpose()
    parameters = pd.DataFrame(
        [
            "Mallet",
            n_topics,
            n_iter,
            random_state,
            alpha,
            alpha_by_topic,
            eta,
            top_topics_coh,
            end_time,
            model.time,
        ],
        index=[
            "package",
            "n_topics",
            "n_iter",
            "random_state",
            "alpha",
            "alpha_by_topic",
            "eta",
            "top_topics_coh",
            "full_time",
            "model_time",
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


def evaluate_models(
    models: list[CistopicLDAModel],
    select_model: int | None = None,
    return_model: bool = True,
    metrics: str = [
        "Minmo_2011",
        "loglikelihood",
        "Cao_Juan_2009",
        "Arun_2010",
    ],
    min_topics_coh: int = 5,
    plot: bool = True,
    figsize: tuple[float, float] = (6.4, 4.8),
    plot_metrics: bool = False,
    save: str | None = None,
):
    """
    Model selection based on model quality metrics (model coherence (adaptation from Mimno et al., 2011), log-likelihood (Griffiths and Steyvers, 2004), density-based (Cao Juan et al., 2009) and divergence-based (Arun et al., 2010)).

    Parameters
    ----------
    models: list of :class:`CistopicLDAModel`
        A list containing cisTopic LDA models, as returned from run_cgs_models or run_cgs_modelsMallet.
    selected_model: int, optional
        Integer indicating the number of topics of the selected model. If not provided, the best model will be selected automatically based on the model quality metrics. Default: None.
    return_model: bool, optional
        Whether to return the selected model as :class:`CistopicLDAModel`
    metrics: list of str
        Metrics to use for plotting and model selection:
            Minmo_2011: Uses the average model coherence as calculated by Mimno et al (2011). In order to reduce the impact of the number of topics, we calculate the average coherence based on the top selected average values. The better the model, the higher coherence.
            log-likelihood: Uses the log-likelihood in the last iteration as calculated by Griffiths and Steyvers (2004). The better the model, the higher the log-likelihood.
            Arun_2010: Uses a divergence-based metric as in Arun et al (2010) using the topic-region distribution, the cell-topic distribution and the cell coverage. The better the model, the lower the metric.
            Cao_Juan_2009: Uses a density-based metric as in Cao Juan et al (2009) using the topic-region distribution. The better the model, the lower the metric.
        Default: all metrics.
    min_topics_coh: int, optional
        Minimum number of topics on a topic to use its coherence for model selection. Default: 5.
    plot: bool, optional
        Whether to return plot to the console. Default: True.
    figsize: tuple, optional
                Size of the figure. Default: (6.4, 4.8)
    plot_metrics: bool, optional
        Whether to plot metrics independently. Default: False.
    save: str, optional
        Output file to save plot. Default: None.

    Return
    ------
    plot
        Plot with the combined metrics in which the best model should have high values for all metrics (Arun_2010 and Cao_Juan_2011 are inversed).

    References
    ----------
    Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A. (2011). Optimizing semantic coherence in topic models. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing (pp. 262-272).

    Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. Proceedings of the National academy of Sciences, 101(suppl 1), 5228-5235

    Cao, J., Xia, T., Li, J., Zhang, Y., & Tang, S. (2009). A density-based method for adaptive LDA model selection. Neurocomputing, 72(7-9), 1775-1781.

    Arun, R., Suresh, V., Madhavan, C. V., & Murthy, M. N. (2010). On finding the natural number of topics with latent dirichlet allocation: Some observations. In Pacific-Asia conference on knowledge discovery and data mining (pp. 391-402). Springer, Berlin, Heidelberg.

    """
    models = [models[i] for i in np.argsort([m.n_topic for m in models])]
    all_topics = sorted([models[x].n_topic for x in range(0, len(models))])
    metrics_dict = {}
    fig = plt.figure(figsize=figsize)
    if "Minmo_2011" in metrics:
        in_index = [
            i for i in range(len(all_topics)) if all_topics[i] >= min_topics_coh
        ]
    if "Arun_2010" in metrics:
        arun_2010 = [
            models[index].metrics.loc["Metric", "Arun_2010"]
            for index in range(0, len(all_topics))
        ]
        arun_2010_negative = [-x for x in arun_2010]
        arun_2010_rescale = (arun_2010_negative - min(arun_2010_negative)) / (
            max(arun_2010_negative) - min(arun_2010_negative)
        )
        if "Minmo_2011" in metrics:
            metrics_dict["Arun_2010"] = np.array(
                subset_list(arun_2010_rescale, in_index)
            )
        else:
            metrics_dict["Arun_2010"] = arun_2010_rescale
        plt.plot(
            all_topics,
            arun_2010_rescale,
            linestyle="--",
            marker="o",
            label="Inv_Arun_2010",
        )

    if "Cao_Juan_2009" in metrics:
        Cao_Juan_2009 = [
            models[index].metrics.loc["Metric", "Cao_Juan_2009"]
            for index in range(0, len(all_topics))
        ]
        Cao_Juan_2009_negative = [-x for x in Cao_Juan_2009]
        Cao_Juan_2009_rescale = (
            Cao_Juan_2009_negative - min(Cao_Juan_2009_negative)
        ) / (max(Cao_Juan_2009_negative) - min(Cao_Juan_2009_negative))
        if "Minmo_2011" in metrics:
            metrics_dict["Cao_Juan_2009"] = np.array(
                subset_list(Cao_Juan_2009_rescale, in_index)
            )
        else:
            metrics_dict["Cao_Juan_2009"] = Cao_Juan_2009_rescale
        plt.plot(
            all_topics,
            Cao_Juan_2009_rescale,
            linestyle="--",
            marker="o",
            label="Inv_Cao_Juan_2009",
        )

    if "Minmo_2011" in metrics:
        Mimno_2011 = [
            models[index].metrics.loc["Metric", "Mimno_2011"]
            for index in range(0, len(all_topics))
        ]
        Mimno_2011 = subset_list(Mimno_2011, in_index)
        Mimno_2011_all_topics = subset_list(all_topics, in_index)
        Mimno_2011_rescale = (Mimno_2011 - min(Mimno_2011)) / (
            max(Mimno_2011) - min(Mimno_2011)
        )
        metrics_dict["Minmo_2011"] = np.array(Mimno_2011_rescale)
        plt.plot(
            Mimno_2011_all_topics,
            Mimno_2011_rescale,
            linestyle="--",
            marker="o",
            label="Mimno_2011",
        )

    if "loglikelihood" in metrics:
        loglikelihood = [
            models[index].metrics.loc["Metric", "loglikelihood"]
            for index in range(0, len(all_topics))
        ]
        loglikelihood_rescale = (loglikelihood - min(loglikelihood)) / (
            max(loglikelihood) - min(loglikelihood)
        )
        if "Minmo_2011" in metrics:
            metrics_dict["loglikelihood"] = np.array(
                subset_list(loglikelihood_rescale, in_index)
            )
        else:
            metrics_dict["loglikelihood"] = loglikelihood_rescale
        plt.plot(
            all_topics,
            loglikelihood_rescale,
            linestyle="--",
            marker="o",
            label="Loglikelihood",
        )

    if select_model is None:
        combined_metric = sum(metrics_dict.values())
        if "Minmo_2011" in metrics:
            best_model = Mimno_2011_all_topics[
                combined_metric.tolist().index(max(combined_metric))
            ]
        else:
            best_model = all_topics[
                combined_metric.tolist().index(max(combined_metric))
            ]
    else:
        combined_metric = None
        best_model = select_model

    plt.axvline(best_model, linestyle="--", color="grey")
    plt.xlabel("Number of topics\nOptimal number of topics: " + str(best_model))
    plt.ylabel("Rescaled metric")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    if save is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)
        pdf.savefig(fig, bbox_inches="tight")
    if plot is True:
        plt.show()
    else:
        plt.close(fig)

    if plot_metrics:
        if "Arun_2010" in metrics:
            fig = plt.figure()
            plt.plot(all_topics, arun_2010, linestyle="--", marker="o")
            plt.axvline(best_model, linestyle="--", color="grey")
            plt.title("Arun_2010 - Minimize")
            if save is not None:
                pdf.savefig(fig)
            plt.show()

        if "Cao_Juan_2009" in metrics:
            fig = plt.figure()
            plt.plot(all_topics, Cao_Juan_2009, linestyle="--", marker="o")
            plt.axvline(best_model, linestyle="--", color="grey")
            plt.title("Cao_Juan_2009 - Minimize")
            if save is not None:
                pdf.savefig(fig)
            plt.show()
        if "Minmo_2011" in metrics:
            fig = plt.figure()
            plt.plot(Mimno_2011_all_topics, Mimno_2011, linestyle="--", marker="o")
            plt.axvline(best_model, linestyle="--", color="grey")
            plt.title("Mimno_2011 - Maximize")
            if save is not None:
                pdf.savefig(fig)
            plt.show()

        if "loglikelihood" in metrics:
            fig = plt.figure()
            plt.plot(all_topics, loglikelihood, linestyle="--", marker="o")
            plt.axvline(best_model, linestyle="--", color="grey")
            plt.title("Loglikelihood - Maximize")
            if save is not None:
                pdf.savefig(fig)
            plt.show()

    if save is not None:
        pdf.close()

    if return_model:
        return models[all_topics.index(best_model)]


def load_cisTopic_model(path_to_cisTopic_model_matrices):
    metrics = None
    coherence = None
    marg_topic = None
    topic_ass = None
    cell_topic = pd.read_feather(path_to_cisTopic_model_matrices + "cell_topic.feather")
    cell_topic.index = ["Topic" + str(x) for x in range(1, cell_topic.shape[0] + 1)]
    topic_region = pd.read_feather(
        path_to_cisTopic_model_matrices + "topic_region.feather"
    )
    topic_region.index = ["Topic" + str(x) for x in range(1, topic_region.shape[0] + 1)]
    topic_region = topic_region.T
    parameters = None
    model = CistopicLDAModel(
        metrics, coherence, marg_topic, topic_ass, cell_topic, topic_region, parameters
    )
    return model
