from __future__ import annotations

import anndata
import pandas as pd

from pycisTopic.topic_modeling.topic_models import TopicModelFilenames, load_topic_model_backend


def create_anndata_from_topic_model(
    output_prefix: str,
    n_topics: int,
    cell_barcodes: list[str],
    region_ids: list[str],
) -> None:
    """Create AnnData objects from backend-agnostic v3 topic modeling artifacts."""
    filenames = TopicModelFilenames(output_prefix=output_prefix, n_topics=n_topics)
    backend_cls = load_topic_model_backend(output_prefix=output_prefix, n_topics=n_topics)

    print(f"Reading {backend_cls.backend} results ...")
    topic_word_distrib = (
        backend_cls.read_region_topic_counts_parquet_file_to_region_topic_probabilities(
            region_topic_counts_parquet_filename=filenames.region_topic_counts_parquet_filename
        )
    )
    doc_topic_distrib = backend_cls.read_cell_topic_probabilities_parquet_file(
        cell_topic_probabilities_parquet_filename=filenames.cell_topic_probabilities_parquet_filename
    )

    cell_topic = pd.DataFrame.from_records(
        doc_topic_distrib,
        index=cell_barcodes,
        columns=[f"Topic{i}" for i in range(1, n_topics + 1)],
    )
    region_topic = pd.DataFrame.from_records(
        topic_word_distrib,
        columns=region_ids,
        index=[f"Topic{i}" for i in range(1, n_topics + 1)],
    ).transpose()

    print("Generating cell topic AnnData object")
    adata_cell_topic = anndata.AnnData(X=cell_topic)
    print(f"Done, shape is: {adata_cell_topic.shape}")

    print("Generating region topic AnnData object")
    adata_region_topic = anndata.AnnData(X=region_topic)
    print(f"Done, shape is: {adata_region_topic.shape}")

    print(f"Writing to: {filenames.anndata_cell_topic_filename}")
    adata_cell_topic.write(filenames.anndata_cell_topic_filename)

    print(f"Writing to: {filenames.anndata_region_topic_filename}")
    adata_region_topic.write(filenames.anndata_region_topic_filename)
