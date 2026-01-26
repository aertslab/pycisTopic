import json
import logging
import os
import subprocess
import time

import numpy as np
import polars as pl
import scipy


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
        n_threads: int = 1,
        iterations: int = 150,
        optimize_interval: int = 0,
        optimize_burn_in: int = 50,
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
            the number of topics. Default: False.
        n_threads
            Number of threads that will be used for training. Default: 1.
        iterations
            Number of training iterations of Gibbs sampling. Default: 150.
        optimize_interval
            Optimize hyperparameters every `optimize_interval` iterations (sometimes
            leads to Java exception, 0 to switch off hyperparameter optimization).
            Only takes effect after running `optimize_burn_in` iterations.
            Default: 0.
        optimize_burn_in
            The number of iterations before hyperparameter optimization begins.
            Default: 50.
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
            "--optimize-burn-in",
            str(optimize_burn_in),
            "--num-threads",
            str(n_threads),
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
                "n_threads": n_threads,
                "iterations": iterations,
                "optimize_interval": optimize_interval,
                "optimize_burn_in": optimize_burn_in,
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
    def model_stats_filename(self):
        return f"{self.output_prefix}.{self.n_topics}_topics.stats.json"

    @property
    def anndata_cell_topic_filename(self):
        return f"{self.output_prefix}.{self.n_topics}_topics_cell_topic_adata.h5ad"

    @property
    def anndata_region_topic_filename(self):
        return f"{self.output_prefix}.{self.n_topics}_topics_region_topic_adata.h5ad"
