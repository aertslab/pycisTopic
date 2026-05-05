from __future__ import annotations

import json
import logging
import time

import numpy as np
import polars as pl
import tomotopy as tp
from scipy import sparse

from pycisTopic.topic_modeling.topic_models import LDAModel, TopicModelFilenames


class LDATomotopy(LDAModel):
    """Run LDA models with tomotopy and write v3 artifacts."""

    backend = "tomotopy"

    @staticmethod
    def run_topic_modeling(
        binary_accessibility_matrix: sparse.csr_matrix,
        cell_names: list[str],
        region_names: list[str],
        output_prefix: str,
        n_topics: int,
        alpha: float = 50.0,
        alpha_by_topic: bool = True,
        eta: float = 0.1,
        eta_by_topic: bool = False,
        n_threads: int = 1,
        iterations: int = 150,
        optimize_interval: int = 0,
        random_seed: int = 555,
    ) -> None:
        """Run tomotopy LDA and write the standard v3 artifacts."""
        logger = logging.getLogger("LDATomotopy")
        matrix = sparse.csr_matrix(binary_accessibility_matrix)

        if len(cell_names) != matrix.shape[1]:
            raise ValueError(
                "Number of cell names does not match the accessibility matrix columns."
            )
        if len(region_names) != matrix.shape[0]:
            raise ValueError(
                "Number of region names does not match the accessibility matrix rows."
            )

        documents, valid_cell_indices = _build_documents(
            binary_accessibility_matrix=matrix,
            region_names=region_names,
        )

        effective_alpha = alpha / n_topics if alpha_by_topic else alpha
        effective_eta = eta / n_topics if eta_by_topic else eta

        model = tp.LDAModel(
            k=n_topics,
            alpha=effective_alpha,
            eta=effective_eta,
            seed=random_seed,
            min_cf=0,
        )
        model.optim_interval = optimize_interval

        for document in documents:
            model.add_doc(words=document)

        start_time = time.time()
        logger.info(
            "Training tomotopy model for %s topics, %s iterations and %s threads.",
            n_topics,
            iterations,
            n_threads,
        )
        model.train(iterations=iterations, workers=n_threads, show_progress=False)
        elapsed = time.time() - start_time

        region_topic_counts, doc_topic_counts = _extract_exact_counts(
            model=model,
            region_names=region_names,
            cell_count=matrix.shape[1],
            valid_cell_indices=valid_cell_indices,
            n_topics=n_topics,
        )

        cell_topic_probabilities = _counts_to_probabilities(doc_topic_counts)
        filenames = TopicModelFilenames(output_prefix=output_prefix, n_topics=n_topics)

        pl.Series(
            "cell_topic_probabilities",
            cell_topic_probabilities.astype(np.float32),
        ).to_frame().write_parquet(filenames.cell_topic_probabilities_parquet_filename)
        pl.Series(
            "region_topic_counts",
            region_topic_counts.astype(np.int32),
        ).to_frame().write_parquet(filenames.region_topic_counts_parquet_filename)

        learned_alpha = np.asarray(model.alpha, dtype=np.float64).reshape(-1)
        parameters = {
            "backend": LDATomotopy.backend,
            "output_prefix": output_prefix,
            "n_topics": n_topics,
            "alpha": (
                learned_alpha.tolist()
                if learned_alpha.size > 1
                else float(learned_alpha[0])
            ),
            "alpha_input": alpha,
            "alpha_by_topic": alpha_by_topic,
            "eta": float(np.asarray(model.eta, dtype=np.float64).reshape(())),
            "eta_input": eta,
            "eta_by_topic": eta_by_topic,
            "n_threads": n_threads,
            "iterations": iterations,
            "optimize_interval": optimize_interval,
            "random_seed": random_seed,
            "time": elapsed,
            "tomotopy_version": tp.__version__,
        }
        with open(filenames.parameters_json_filename, "w", encoding="utf-8") as fh:
            json.dump(parameters, fh, indent=2)


def _build_documents(
    binary_accessibility_matrix: sparse.csr_matrix,
    region_names: list[str],
) -> tuple[list[list[str]], list[int]]:
    matrix = binary_accessibility_matrix.tocsc()
    matrix.eliminate_zeros()

    if matrix.shape[0] == 0:
        raise ValueError("Binary accessibility matrix does not contain any regions.")
    if matrix.shape[1] == 0:
        raise ValueError(
            "Binary accessibility matrix does not contain any cell barcodes."
        )

    documents: list[list[str]] = []
    valid_cell_indices: list[int] = []
    for cell_idx, (indptr_start, indptr_end) in enumerate(
        zip(matrix.indptr, matrix.indptr[1:])
    ):
        region_indices = matrix.indices[indptr_start:indptr_end]
        if region_indices.size == 0:
            continue
        documents.append([region_names[region_idx] for region_idx in region_indices])
        valid_cell_indices.append(cell_idx)

    if not documents:
        raise ValueError(
            "Binary accessibility matrix does not contain any accessible cells."
        )

    return documents, valid_cell_indices


def _extract_exact_counts(
    model: tp.LDAModel,
    region_names: list[str],
    cell_count: int,
    valid_cell_indices: list[int],
    n_topics: int,
) -> tuple[np.ndarray, np.ndarray]:
    region_name_to_index = {
        region_name: idx for idx, region_name in enumerate(region_names)
    }
    word_id_to_region_index = np.asarray(
        [region_name_to_index[token] for token in model.used_vocabs],
        dtype=np.int64,
    )

    region_topic_counts = np.zeros((n_topics, len(region_names)), dtype=np.int64)
    doc_topic_counts = np.zeros((cell_count, n_topics), dtype=np.int64)

    for doc_idx, doc in enumerate(model.docs):
        cell_idx = valid_cell_indices[doc_idx]
        word_ids = np.asarray(doc.words, dtype=np.int64)
        topic_ids = np.asarray(doc.topics, dtype=np.int64)
        region_indices = word_id_to_region_index[word_ids]

        np.add.at(region_topic_counts, (topic_ids, region_indices), 1)
        doc_topic_counts[cell_idx] = np.bincount(topic_ids, minlength=n_topics)

    return region_topic_counts, doc_topic_counts


def _counts_to_probabilities(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True).astype(np.float64)
    return np.divide(
        counts,
        totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=totals != 0,
    ).astype(np.float32)
