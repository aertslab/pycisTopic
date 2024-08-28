from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
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
import scipy
import tmtoolkit
from pycisTopic.utils import loglikelihood, subset_list

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
    binary_accessibility_matrix: sparse.csr_matrix,
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


class LDAMallet:
    """Class for running LDA models with Mallet."""

    @staticmethod
    def convert_binary_matrix_to_mallet_corpus_file(
        binary_accessibility_matrix: scipy.sparse.csr,
        mallet_corpus_filename: str,
        mallet_path: str = "mallet",
    ) -> None:
        """
        Convert binary matrix to Mallet serialized corpus file.

        Parameters
        ----------
        binary_accessibility_matrix
            Binary accessibility matrix (region IDs vs cell barcodes)
        mallet_corpus_filename
            Mallet serialized corpus filename
        mallet_path
            Path to Mallet binary.

        Returns
        -------
        None.

        """
        logger = logging.getLogger("LDAMallet")

        # Convert binary accessibility matrix to compressed sparse column matrix format
        # and eliminate zeros as we assume later that for each found index, the
        # associated value is 1.
        binary_accessibility_matrix_csc = binary_accessibility_matrix.tocsc()
        binary_accessibility_matrix_csc.eliminate_zeros()

        mallet_corpus_txt_filename = f"{mallet_corpus_filename}.txt"

        logger.info(
            f'Serializing binary accessibility matrix to Mallet text corpus to "{mallet_corpus_txt_filename}".'
        )

        if binary_accessibility_matrix_csc.shape[0] == 0:
            raise ValueError(
                "Binary accessibility matrix does not contain any cell barcodes."
            )

        if binary_accessibility_matrix_csc.shape[1] == 0:
            raise ValueError(
                "Binary accessibility matrix does not contain any regions."
            )

        with open(mallet_corpus_txt_filename, "w") as mallet_corpus_txt_fh:
            # Iterate over each column (cell barcode index) of the sparse binary
            # accessibility matrix in compressed sparse column matrix format and get
            # all index positions (region IDs indices) for that cell barcode index.
            for cell_barcode_idx, (indptr_start, indptr_end) in enumerate(
                zip(
                    binary_accessibility_matrix_csc.indptr,
                    binary_accessibility_matrix_csc.indptr[1:],
                )
            ):
                # Get all region ID indices (assume all have an associated value of 1)
                # for the current cell barcode index.
                region_ids_idx = binary_accessibility_matrix_csc.indices[
                    indptr_start:indptr_end
                ]

                # Write Mallet text corpus for the current cell barcode index:
                #   - column 1: cell barcode index.
                #   - column 2: document number (always 0).
                #   - column 3: region IDs indices accessible in the current cell barcode.
                mallet_corpus_txt_fh.write(
                    f'{cell_barcode_idx}\t0\t{" ".join([str(x) for x in region_ids_idx])}\n'
                )

        mallet_import_file_cmd = [
            mallet_path,
            "import-file",
            "--preserve-case",
            "--keep-sequence",
            "--token-regex",
            "\\S+",
            "--input",
            mallet_corpus_txt_filename,
            "--output",
            mallet_corpus_filename,
        ]

        logger.info(
            f"Converting Mallet text corpus to Mallet serialised corpus with: {' '.join(mallet_import_file_cmd)}"
        )

        try:
            subprocess.check_output(
                args=mallet_import_file_cmd, shell=False, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
            )

        # Remove Mallet text corpus as only Mallet serialised corpus file is needed.
        if os.path.exists(mallet_corpus_txt_filename):
            os.remove(mallet_corpus_txt_filename)

    @staticmethod
    def convert_cell_topic_probabilities_txt_to_parquet(
        mallet_cell_topic_probabilities_txt_filename: str,
        mallet_cell_topic_probabilities_parquet_filename: str,
    ) -> None:
        """
        Convert cell-topic probabilities from Mallet output to Parquet file.

        Parameters
        ----------
        mallet_cell_topic_probabilities_txt_filename
            Mallet cell-topic probabilities text file.
        mallet_cell_topic_probabilities_parquet_filename
            Parquet output file with cell-topic probabilities.

        Returns
        -------
        None

        """
        # Read cell-topic probabilities Mallet output file and extract for each cell
        # barcode the probability for the cell barcode to belong to a certain topic.
        #
        # Column 0: order in which cell barcode idx was seen in the input corpus file.
        # Column 1: cell barcode idx
        # Column 3-n: "topic probability" for each topic
        #
        # Mallet cell-topic probabilities file example:
        # ---------------------------------------------
        #
        # 0	0	0.06355276993175679	0.1908026307651073	0.06691338680081645	0.007391295383790694	0.07775807681999052	0.08091252087499742	0.08262375523163516	9.793208667505102E-4	0.007721171886275076	0.01605055357400573	0.014071294559099437	0.025307712924973712	0.020524503638950167	0.061903387419334883	0.07344906500628827	0.02866832979403336	0.03520400799950518	0.07608807702616333	0.047656845968290625	0.022421293528235367
        # 1	1	0.10109016579604815	0.0016579604814898933	0.033499886441062915	0.003792868498750852	0.06665909607086078	0.19216443334090394	0.023143311378605497	0.0011128775834658188	0.08719055189643425	0.00401998637292755	0.0030206677265500795	0.03617987735634794	0.02473313649784238	0.255984555984556	0.004383374971610266	0.037179196002725415	0.023143311378605497	0.06202589143765614	0.009379968203497615	0.02963888258005905
        # 2	2	0.08937104175357427	0.03120615116234973	0.11623971329970799	0.03952083886381736	0.034562364898175886	0.08415658538435283	0.03002104744207213	0.040440479350752775	0.02172532140012894	0.025119458455003983	0.01332530623080132	0.06196196291099397	0.07174617922560582	0.03189825173499185	0.05144772270469111	0.00540881337934696	0.08696291099397019	0.07489381470666313	0.04997819409154689	0.04001384201145284
        # 3	3	0.05694870514375401	0.003620603552828708	0.07264393236783906	0.11541342655347078	0.005546835984875508	0.025451237782692444	0.010790468716558465	0.377309695369908	0.03540343868160091	0.007580081329813798	0.023453663408717986	0.02869729614040094	0.08166868802168795	0.01703288863522865	0.006153242491260612	0.0172112434900478	0.06311978312049654	0.02124206320896055	0.012895056003424414	0.017817649996432903
        # 4	4	0.08079825190344497	0.002168049438355697	0.06058588548601864	0.002919184676841135	0.07448188739799926	0.12989518249172044	0.15225852709208235	0.008962409095564889	0.02753593499265936	0.001519341732391	0.011386527365222438	0.012376660179589606	0.015108061046809382	0.1424596264809314	0.015449486155211854	0.027740790057700842	0.068370377957595	0.1540339376557752	0.002168049438355697	0.00978182935573082
        cell_topic_probabilities_ldf = pl.scan_csv(
            mallet_cell_topic_probabilities_txt_filename,
            separator="\t",
            has_header=False,
            with_column_names=lambda cols: [
                f"topic_{idx - 1}" if idx > 1 else f"cell_idx{idx}"
                for idx, col in enumerate(cols)
            ],
        )
        # Get cell-topic probabilities as numpy matrix.
        cell_topic_probabilities = (
            cell_topic_probabilities_ldf.select(
                pl.col("^topic_[0-9]+$").cast(pl.Float32)
            )
            .collect()
            .to_numpy()
        )

        # Write cell-topic probabilities matrix to one column of a Parquet file.
        pl.Series(
            "cell_topic_probabilities", cell_topic_probabilities
        ).to_frame().write_parquet(
            f"{mallet_cell_topic_probabilities_parquet_filename}"
        )

    @staticmethod
    def read_cell_topic_probabilities_parquet_file(
        mallet_cell_topic_probabilities_parquet_filename: str,
    ) -> np.ndarray:
        """
        Read cell-topic probabilities Parquet file to cell-topic probabilities matrix.

        Parameters
        ----------
        mallet_cell_topic_probabilities_parquet_filename
             Mallet cell-topic probabilities Parquet filename.

        Returns
        -------
        Cell-topic probabilities matrix.

        """
        return (
            pl.read_parquet(mallet_cell_topic_probabilities_parquet_filename)
            .get_column("cell_topic_probabilities")
            .to_numpy()
        )

    @staticmethod
    def convert_region_topic_counts_txt_to_parquet(
        mallet_region_topic_counts_txt_filename: str,
        mallet_region_topic_counts_parquet_filename: str,
    ) -> None:
        """
        Convert region-topic counts from Mallet output to Parquet file.

        Parameters
        ----------
        mallet_region_topic_counts_txt_filename
            Mallet region-topic counts text file.
        mallet_region_topic_counts_parquet_filename
            Parquet output file with region-topic counts.

        Returns
        -------
        None

        """
        n_region_ids = -1
        n_topics = -1
        region_id_topic_counts = []

        with open(mallet_region_topic_counts_txt_filename) as fh:
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
                # Get topic index and counts from column 3 till the end by splitting
                # "topic:count" pairs.
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
                # Store region ID index, topics indices and counts till we know how many
                # regions and topics we have.
                region_id_topic_counts.append((region_id_idx, topics_idx, counts))

                # Keep track of the highest seen region ID index and topic index
                # (0-based).
                n_region_ids = max(region_id_idx, n_region_ids)
                n_topics = max(topics_idx.max(), n_topics)

        # Add 1 to region IDs and topics counts to account for start at 0.
        n_region_ids += 1
        n_topics += 1

        # Create region-topic counts matrix and populate it.
        regions_topic_counts = np.zeros((n_topics, n_region_ids), dtype=np.int32)
        for region_idx, topics_idx, counts in region_id_topic_counts:
            regions_topic_counts[topics_idx, region_idx] = counts

        # Write region-topic counts matrix to one column of a Parquet file.
        pl.Series("region_topic_counts", regions_topic_counts).to_frame().write_parquet(
            mallet_region_topic_counts_parquet_filename
        )

    @staticmethod
    def read_region_topic_counts_parquet_file(
        mallet_region_topic_counts_parquet_filename: str,
    ) -> np.ndarray:
        """
        Read region-topic counts Parquet file to region-topic counts matrix.

        Parameters
        ----------
        mallet_region_topic_counts_parquet_filename
             Mallet region-topic counts Parquet filename.

        Returns
        -------
        Region-topic counts matrix.

        """
        return (
            pl.read_parquet(mallet_region_topic_counts_parquet_filename)
            .get_column("region_topic_counts")
            .to_numpy()
        )

    @staticmethod
    def read_region_topic_counts_parquet_file_to_region_topic_probabilities(
        mallet_region_topic_counts_parquet_filename: str,
    ) -> np.ndarray:
        """
        Get the region-topic probabilities matrix learned during inference.

        Returns
        -------
        The probability for each region in each topic, shape (n_regions, n_topics).

        """
        region_topic_counts = np.asarray(
            LDAMallet.read_region_topic_counts_parquet_file(
                mallet_region_topic_counts_parquet_filename=mallet_region_topic_counts_parquet_filename,
            ),
            np.float64,
        )

        # Create region-topic probabilities matrix by dividing all count values for a
        # topic by total counts for that topic.
        region_topic_probabilities = (
            region_topic_counts / region_topic_counts.sum(axis=1)[:, None]
        ).astype(np.float32)

        return region_topic_probabilities

    @staticmethod
    def read_parameters_json_filename(parameters_json_filename: str) -> dict:
        """
        Read parameters from JSON file which gets written by `LDAMallet.run_mallet_topic_modeling`.

        Parameters
        ----------
        parameters_json_filename
            Parameters JSON filename created by `LDAMallet.run_mallet_topic_modeling`.

        Returns
        -------
        Dictionary with Mallet LDA parameters and settings.

        """
        with open(parameters_json_filename, "r") as fh:
            mallet_train_topics_parameters = json.load(fh)
        return mallet_train_topics_parameters

    @staticmethod
    def run_mallet_topic_modeling(
        mallet_corpus_filename: str,
        output_prefix: str,
        n_topics: int,
        alpha: float = 50,
        alpha_by_topic: bool = True,
        eta: float = 0.1,
        eta_by_topic: bool = False,
        n_cpu: int = 1,
        optimize_interval: int = 0,
        iterations: int = 150,
        topic_threshold: float = 0.0,
        random_seed: int = 555,
        mallet_path: str = "mallet",
    ):
        """
        Run Mallet LDA.

        Parameters
        ----------
        mallet_corpus_filename
            Mallet corpus file.
        output_prefix
            Output prefix.
        n_topics
            The number of topics to use in the model.
        alpha
            Scalar value indicating the symmetric Dirichlet hyperparameter for topic
            proportions. Default: 50.
        alpha_by_topic
            Boolean indicating whether the scalar given in alpha has to be divided by
            the number of topics. Default: True.
        eta
            Scalar value indicating the symmetric Dirichlet hyperparameter for topic
            multinomials. Default: 0.1.
        eta_by_topic
            Boolean indicating whether the scalar given in beta has to be divided by
            the number of topics. Default: False
        n_cpu
            Number of threads that will be used for training. Default: 1.
        optimize_interval
            Optimize hyperparameters every `optimize_interval` iterations (sometimes
            leads to Java exception, 0 to switch off hyperparameter optimization).
            Default: 0.
        iterations
            Number of training iterations. Default: 150.
        topic_threshold
            Threshold of the probability above which we consider a topic. Default: 0.0.
        random_seed
            Random seed to ensure consistent results, if 0 - use system clock.
            Default: 555.
        mallet_path
            Path to the mallet binary (e.g. /xxx/Mallet/bin/mallet). Default: "mallet".

        """
        logger = logging.getLogger("LDAMallet")

        # Mallet divides alpha value by default by the number of topics, so in case
        # alpha_by_topic=False, input alpha needs to be multiplied by n_topics.
        mallet_alpha = alpha if alpha_by_topic else alpha * n_topics

        mallet_beta = eta / n_topics if eta_by_topic else eta

        lda_mallet_filenames = LDAMalletFilenames(
            output_prefix=output_prefix, n_topics=n_topics
        )

        if not os.path.exists(mallet_corpus_filename):
            raise FileNotFoundError(
                f'Mallet corpus file "{mallet_corpus_filename}" does not exist.'
            )

        cmd = [
            mallet_path,
            "train-topics",
            "--input",
            mallet_corpus_filename,
            "--num-topics",
            str(n_topics),
            "--alpha",
            str(mallet_alpha),
            "--beta",
            str(mallet_beta),
            "--optimize-interval",
            str(optimize_interval),
            "--num-threads",
            str(n_cpu),
            "--num-iterations",
            str(iterations),
            "--word-topic-counts-file",
            lda_mallet_filenames.region_topic_counts_txt_filename,
            "--output-doc-topics",
            lda_mallet_filenames.cell_topic_probabilities_txt_filename,
            "--doc-topics-threshold",
            str(topic_threshold),
            "--random-seed",
            str(random_seed),
        ]

        start_time = time.time()
        logger.info(f"Train topics with Mallet LDA: {' '.join(cmd)}")
        try:
            subprocess.check_output(args=cmd, shell=False, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(  # noqa: B904
                f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
            )

        # Convert cell-topic probabilities text version to parquet.
        logger.info(
            f'Write cell-topic probabilities to "{lda_mallet_filenames.cell_topic_probabilities_parquet_filename}".'
        )
        LDAMallet.convert_cell_topic_probabilities_txt_to_parquet(
            mallet_cell_topic_probabilities_txt_filename=lda_mallet_filenames.cell_topic_probabilities_txt_filename,
            mallet_cell_topic_probabilities_parquet_filename=lda_mallet_filenames.cell_topic_probabilities_parquet_filename,
        )

        # Convert region-topic counts text version to parquet.
        logger.info(
            f'Write region-topic counts to "{lda_mallet_filenames.region_topic_counts_parquet_filename}".'
        )
        LDAMallet.convert_region_topic_counts_txt_to_parquet(
            mallet_region_topic_counts_txt_filename=lda_mallet_filenames.region_topic_counts_txt_filename,
            mallet_region_topic_counts_parquet_filename=lda_mallet_filenames.region_topic_counts_parquet_filename,
        )

        total_time = time.time() - start_time

        # Write JSON file with all used parameters.
        logger.info(
            f'Write JSON parameters file to "{lda_mallet_filenames.parameters_json_filename}".'
        )
        with open(lda_mallet_filenames.parameters_json_filename, "w") as fh:
            mallet_train_topics_parameters = {
                "mallet_corpus_filename": mallet_corpus_filename,
                "output_prefix": output_prefix,
                "n_topics": n_topics,
                "alpha": alpha,
                "alpha_by_topic": alpha_by_topic,
                "eta": eta,
                "eta_by_topic": eta_by_topic,
                "n_cpu": n_cpu,
                "optimize_interval": optimize_interval,
                "iterations": iterations,
                "random_seed": random_seed,
                "mallet_path": mallet_path,
                "time": total_time,
                "mallet_cmd": cmd,
            }
            json.dump(mallet_train_topics_parameters, fh)


class LDAMalletFilenames:
    """Class to generate output filenames when running functions of LDAMallet."""

    def __init__(self, output_prefix: str, n_topics: int):
        """
        Generate output filenames when running functions of LDAMallet.

        Parameters
        ----------
        output_prefix
            Output prefix.
        n_topics
            The number of topics used in the model.

        """
        self.output_prefix = output_prefix
        self.n_topics = n_topics

    @property
    def parameters_json_filename(self):
        return os.path.join(
            f"{self.output_prefix}.{self.n_topics}_topics.parameters.json"
        )

    @property
    def cell_topic_probabilities_txt_filename(self):
        return os.path.join(
            f"{self.output_prefix}.{self.n_topics}_topics.cell_topic_probabilities.txt"
        )

    @property
    def cell_topic_probabilities_parquet_filename(self):
        return os.path.join(
            f"{self.output_prefix}.{self.n_topics}_topics.cell_topic_probabilities.parquet"
        )

    @property
    def region_topic_counts_txt_filename(self):
        return f"{self.output_prefix}.{self.n_topics}_topics.region_topic_counts.txt"

    @property
    def region_topic_counts_parquet_filename(self):
        return (
            f"{self.output_prefix}.{self.n_topics}_topics.region_topic_counts.parquet"
        )

    @property
    def model_pickle_filename(self):
        return f"{self.output_prefix}.{self.n_topics}_topics.model.pkl"


def calculate_model_evaluation_stats(
    binary_accessibility_matrix: scipy.sparse.csr_matrix,
    cell_barcodes: list[str],
    region_ids: list[str],
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
    cell_barcodes
        List containing cell names as ordered in the binary matrix columns.
    region_ids
        List containing region names as ordered in the binary matrix rows.
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

    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

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
    n_iter = mallet_train_topics_parameters["iterations"]
    random_state = mallet_train_topics_parameters["random_seed"]
    mallet_time = mallet_train_topics_parameters["time"]

    ll_alpha = alpha / n_topics if alpha_by_topic else alpha
    ll_eta = eta / n_topics if eta_by_topic else eta

    # Model evaluation
    cell_cov = np.asarray(binary_accessibility_matrix.sum(axis=0)).astype(float)
    arun_2010 = tmtoolkit.topicmod.evaluate.metric_arun_2010(
        topic_word_distrib=topic_word_distrib,
        doc_topic_distrib=doc_topic_distrib,
        doc_lengths=cell_cov,
    )
    cao_juan_2009 = tmtoolkit.topicmod.evaluate.metric_cao_juan_2009(
        topic_word_distrib=topic_word_distrib
    )
    mimno_2011 = tmtoolkit.topicmod.evaluate.metric_coherence_mimno_2011(
        topic_word_distrib=topic_word_distrib,
        dtm=binary_accessibility_matrix.transpose(),
        top_n=20,
        eps=1e-12,
        normalize=True,
        return_mean=False,
    )

    doc_topic_counts = (doc_topic_distrib.T * (cell_cov)).T
    ll = loglikelihood(topic_word_counts, doc_topic_counts, ll_alpha, ll_eta)

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
            tmtoolkit.topicmod.model_stats.marginal_topic_distrib(
                doc_topic_distrib=doc_topic_distrib, doc_lengths=cell_cov
            ),
        ],
        index=["Topic", "Marg_Topic"],
    ).transpose()
    topic_ass = pd.DataFrame.from_records(
        [
            range(1, n_topics + 1),
            list(chain.from_iterable(topic_word_counts.sum(axis=1)[:, None])),
        ],
        index=["Topic", "Assignments"],
    ).transpose()
    cell_topic = pd.DataFrame.from_records(
        doc_topic_distrib,
        index=cell_barcodes,
        columns=["Topic" + str(i) for i in range(1, n_topics + 1)],
    ).transpose()
    topic_region = pd.DataFrame.from_records(
        topic_word_distrib,
        columns=region_ids,
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
            mallet_time,
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
            "model_time",
        ],
        columns=["Parameter"],
    )
    # Create object
    model = CistopicLDAModel(
        metrics, coherence, marg_topic, topic_ass, cell_topic, topic_region, parameters
    )

    log.info(
        f"Saving model with {n_topics} topics at {lda_mallet_filenames.model_pickle_filename}"
    )
    with open(lda_mallet_filenames.model_pickle_filename, "wb") as fh:
        pickle.dump(model, fh)


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
