from __future__ import annotations

import json
import logging
import os
import subprocess
import time

import numpy as np
import polars as pl
import scipy

from pycisTopic.topic_modeling.topic_models import LDAModel, TopicModelFilenames


class LDAMallet(LDAModel):
    """Run LDA models with Mallet and write v3 artifacts."""

    backend = "mallet"

    @staticmethod
    def convert_binary_matrix_to_mallet_corpus_file(
        binary_accessibility_matrix: scipy.sparse.csr_matrix,
        mallet_corpus_filename: str,
        mallet_path: str = "mallet",
    ) -> None:
        """
        Convert a binary accessibility matrix to a serialized Mallet corpus file.

        Parameters
        ----------
        binary_accessibility_matrix
            Binary accessibility matrix (region IDs vs cell barcodes).
        mallet_corpus_filename
            Mallet serialized corpus filename.
        mallet_path
            Path to the Mallet binary.

        """
        logger = logging.getLogger("LDAMallet")

        matrix = binary_accessibility_matrix.tocsc()
        matrix.eliminate_zeros()

        if matrix.shape[0] == 0:
            raise ValueError("Binary accessibility matrix does not contain any regions.")
        if matrix.shape[1] == 0:
            raise ValueError(
                "Binary accessibility matrix does not contain any cell barcodes."
            )

        mallet_corpus_txt_filename = f"{mallet_corpus_filename}.txt"
        logger.info(
            'Serializing binary accessibility matrix to Mallet text corpus "%s".',
            mallet_corpus_txt_filename,
        )

        with open(mallet_corpus_txt_filename, "w", encoding="utf-8") as fh:
            buffered_lines: list[str] = []
            for cell_barcode_idx, (indptr_start, indptr_end) in enumerate(
                zip(matrix.indptr, matrix.indptr[1:])
            ):
                region_ids_idx = matrix.indices[indptr_start:indptr_end]
                buffered_lines.append(
                    f"{cell_barcode_idx}\t0\t{' '.join(map(str, region_ids_idx))}\n"
                )
                if len(buffered_lines) >= 1024:
                    fh.writelines(buffered_lines)
                    buffered_lines.clear()

            if buffered_lines:
                fh.writelines(buffered_lines)

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
            "Converting Mallet text corpus to Mallet serialized corpus with: %s",
            " ".join(mallet_import_file_cmd),
        )

        try:
            subprocess.check_output(
                args=mallet_import_file_cmd,
                shell=False,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"command '{exc.cmd}' return with error (code {exc.returncode}): {exc.output}"
            ) from exc

        if os.path.exists(mallet_corpus_txt_filename):
            os.remove(mallet_corpus_txt_filename)

    @staticmethod
    def convert_cell_topic_probabilities_txt_to_parquet(
        mallet_cell_topic_probabilities_txt_filename: str,
        mallet_cell_topic_probabilities_parquet_filename: str,
    ) -> None:
        """Convert Mallet cell-topic probabilities text output to Parquet."""
        cell_topic_probabilities = (
            pl.scan_csv(
                mallet_cell_topic_probabilities_txt_filename,
                separator="\t",
                has_header=False,
                with_column_names=lambda cols: [
                    f"topic_{idx - 1}" if idx > 1 else f"cell_idx{idx}"
                    for idx, _ in enumerate(cols)
                ],
            )
            .select(pl.col("^topic_[0-9]+$").cast(pl.Float32))
            .collect()
            .to_numpy()
        )

        pl.Series(
            "cell_topic_probabilities",
            cell_topic_probabilities,
        ).to_frame().write_parquet(mallet_cell_topic_probabilities_parquet_filename)

    @staticmethod
    def convert_region_topic_counts_txt_to_parquet(
        mallet_region_topic_counts_txt_filename: str,
        mallet_region_topic_counts_parquet_filename: str,
    ) -> None:
        """Convert Mallet region-topic counts text output to Parquet."""
        n_region_ids = -1
        n_topics = -1
        region_id_topic_counts: list[tuple[int, np.ndarray, np.ndarray]] = []

        with open(
            mallet_region_topic_counts_txt_filename, encoding="utf-8"
        ) as text_file:
            for line in text_file:
                columns = line.rstrip().split()
                region_id_idx = int(columns[1])
                topics_counts = [
                    (int(topic), int(count))
                    for topic, count in (
                        topic_count.split(":", 1) for topic_count in columns[2:]
                    )
                ]

                topics_idx = np.asarray(
                    [topic for topic, _ in topics_counts], dtype=np.int32
                )
                counts = np.asarray([count for _, count in topics_counts], dtype=np.int32)
                region_id_topic_counts.append((region_id_idx, topics_idx, counts))

                n_region_ids = max(region_id_idx, n_region_ids)
                if topics_idx.size > 0:
                    n_topics = max(int(topics_idx.max()), n_topics)

        n_region_ids += 1
        n_topics += 1

        region_topic_counts = np.zeros((n_topics, n_region_ids), dtype=np.int32)
        for region_idx, topics_idx, counts in region_id_topic_counts:
            region_topic_counts[topics_idx, region_idx] = counts

        pl.Series("region_topic_counts", region_topic_counts).to_frame().write_parquet(
            mallet_region_topic_counts_parquet_filename
        )

    @staticmethod
    def run_topic_modeling(
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
    ) -> None:
        """Run Mallet LDA and write the standard v3 artifacts."""
        logger = logging.getLogger("LDAMallet")

        if topic_threshold != 0.0:
            raise ValueError(
                "topic_threshold must be 0.0 because pycisTopic stores dense "
                "cell-topic probability tables and cannot parse Mallet sparse "
                "doc-topic output."
            )

        mallet_alpha = alpha if alpha_by_topic else alpha * n_topics
        mallet_beta = eta / n_topics if eta_by_topic else eta
        filenames = TopicModelFilenames(output_prefix=output_prefix, n_topics=n_topics)

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
            filenames.region_topic_counts_txt_filename,
            "--output-doc-topics",
            filenames.cell_topic_probabilities_txt_filename,
            "--doc-topics-threshold",
            str(topic_threshold),
            "--random-seed",
            str(random_seed),
        ]

        start_time = time.time()
        logger.info("Train topics with Mallet LDA: %s", " ".join(cmd))
        try:
            subprocess.check_output(args=cmd, shell=False, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"command '{exc.cmd}' return with error (code {exc.returncode}): {exc.output}"
            ) from exc

        logger.info(
            'Write cell-topic probabilities to "%s".',
            filenames.cell_topic_probabilities_parquet_filename,
        )
        LDAMallet.convert_cell_topic_probabilities_txt_to_parquet(
            mallet_cell_topic_probabilities_txt_filename=filenames.cell_topic_probabilities_txt_filename,
            mallet_cell_topic_probabilities_parquet_filename=filenames.cell_topic_probabilities_parquet_filename,
        )

        logger.info(
            'Write region-topic counts to "%s".',
            filenames.region_topic_counts_parquet_filename,
        )
        LDAMallet.convert_region_topic_counts_txt_to_parquet(
            mallet_region_topic_counts_txt_filename=filenames.region_topic_counts_txt_filename,
            mallet_region_topic_counts_parquet_filename=filenames.region_topic_counts_parquet_filename,
        )

        parameters = {
            "backend": LDAMallet.backend,
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
            "topic_threshold": topic_threshold,
            "random_seed": random_seed,
            "mallet_path": mallet_path,
            "time": time.time() - start_time,
            "mallet_cmd": cmd,
        }
        logger.info('Write JSON parameters file to "%s".', filenames.parameters_json_filename)
        with open(filenames.parameters_json_filename, "w", encoding="utf-8") as fh:
            json.dump(parameters, fh, indent=2)
