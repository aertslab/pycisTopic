from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import polars as pl


class TopicModelFilenames:
    """Generate artifact filenames for v3 topic modeling outputs."""

    def __init__(self, output_prefix: str, n_topics: int):
        self.output_prefix = output_prefix
        self.n_topics = n_topics

    @property
    def parameters_json_filename(self) -> str:
        return f"{self.output_prefix}.{self.n_topics}_topics.parameters.json"

    @property
    def cell_topic_probabilities_txt_filename(self) -> str:
        return (
            f"{self.output_prefix}.{self.n_topics}_topics.cell_topic_probabilities.txt"
        )

    @property
    def cell_topic_probabilities_parquet_filename(self) -> str:
        return (
            f"{self.output_prefix}.{self.n_topics}_topics.cell_topic_probabilities.parquet"
        )

    @property
    def region_topic_counts_txt_filename(self) -> str:
        return f"{self.output_prefix}.{self.n_topics}_topics.region_topic_counts.txt"

    @property
    def region_topic_counts_parquet_filename(self) -> str:
        return f"{self.output_prefix}.{self.n_topics}_topics.region_topic_counts.parquet"

    @property
    def model_stats_filename(self) -> str:
        return f"{self.output_prefix}.{self.n_topics}_topics.stats.json"

    @property
    def anndata_cell_topic_filename(self) -> str:
        return f"{self.output_prefix}.{self.n_topics}_topics_cell_topic_adata.h5ad"

    @property
    def anndata_region_topic_filename(self) -> str:
        return f"{self.output_prefix}.{self.n_topics}_topics_region_topic_adata.h5ad"


class LDAModel(ABC):
    """Abstract base class for v3 LDA backends."""

    backend: ClassVar[str]

    @staticmethod
    def read_parameters_json_filename(parameters_json_filename: str) -> dict:
        with open(parameters_json_filename, encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def read_matrix_parquet_file(
        parquet_filename: str,
        column_name: str,
    ) -> np.ndarray:
        matrix_column = pl.read_parquet(parquet_filename).get_column(column_name)
        values = matrix_column.to_numpy()

        if values.ndim == 2:
            return values
        if values.size == 0:
            return np.empty((0, 0), dtype=np.float32)

        return np.stack([np.asarray(row) for row in values], axis=0)

    @classmethod
    def read_cell_topic_probabilities_parquet_file(
        cls,
        cell_topic_probabilities_parquet_filename: str,
    ) -> np.ndarray:
        return cls.read_matrix_parquet_file(
            parquet_filename=cell_topic_probabilities_parquet_filename,
            column_name="cell_topic_probabilities",
        )

    @classmethod
    def read_region_topic_counts_parquet_file(
        cls,
        region_topic_counts_parquet_filename: str,
    ) -> np.ndarray:
        return cls.read_matrix_parquet_file(
            parquet_filename=region_topic_counts_parquet_filename,
            column_name="region_topic_counts",
        )

    @classmethod
    def read_region_topic_counts_parquet_file_to_region_topic_probabilities(
        cls,
        region_topic_counts_parquet_filename: str,
    ) -> np.ndarray:
        region_topic_counts = np.asarray(
            cls.read_region_topic_counts_parquet_file(
                region_topic_counts_parquet_filename=region_topic_counts_parquet_filename
            ),
            dtype=np.float64,
        )
        topic_totals = region_topic_counts.sum(axis=1, keepdims=True)
        return np.divide(
            region_topic_counts,
            topic_totals,
            out=np.zeros_like(region_topic_counts, dtype=np.float64),
            where=topic_totals != 0,
        ).astype(np.float32)

    @staticmethod
    @abstractmethod
    def run_topic_modeling(*args, **kwargs) -> None:
        """Run topic modeling and write the standard v3 artifacts."""


def load_topic_model_backend(output_prefix: str, n_topics: int) -> type[LDAModel]:
    """Resolve the backend used for a topic-model artifact bundle."""

    filenames = TopicModelFilenames(output_prefix=output_prefix, n_topics=n_topics)
    backend_name = "mallet"

    if os.path.exists(filenames.parameters_json_filename):
        parameters = LDAModel.read_parameters_json_filename(
            filenames.parameters_json_filename
        )
        backend_name = str(parameters.get("backend", "mallet")).lower()

    if backend_name == "mallet":
        from pycisTopic.topic_modeling.mallet_models import LDAMallet

        return LDAMallet

    if backend_name == "tomotopy":
        from pycisTopic.topic_modeling.tomotopy_models import LDATomotopy

        return LDATomotopy

    raise ValueError(
        f"Unsupported topic modeling backend {backend_name!r} in "
        f'"{filenames.parameters_json_filename}".'
    )
