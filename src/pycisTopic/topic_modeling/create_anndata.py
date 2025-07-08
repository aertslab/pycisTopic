import anndata
import pandas as pd

from pycisTopic.topic_modeling.mallet_models import LDAMallet, LDAMalletFilenames


def create_anndata_from_mallet(
    output_prefix: str,
    n_topics: int,
    cell_barcodes: list[str],
    region_ids: list[str]
):
    """
    Create an AnnData object from mallet topic modeling results.

    Parameters
    ----------
    output_prefix
        Output prefix used for running topic modeling with Mallet.
    n_topics
        Number of topics used in the topic model created by Mallet.
        In combination with output_prefix, this allows to load the correct region
        topic counts and cell topic probabilties parquet files.
    cell_barcodes
        List containing cell names as ordered in the binary matrix columns.
    region_ids
        List containing region names as ordered in the binary matrix rows.

    Return
    ------
    None

    """
    # Get distributions
    print("Reading Mallet results ...")
    lda_mallet_filenames = LDAMalletFilenames(
        output_prefix=output_prefix, n_topics=n_topics
    )
    topic_word_distrib = LDAMallet.read_region_topic_counts_parquet_file_to_region_topic_probabilities(
        mallet_region_topic_counts_parquet_filename=lda_mallet_filenames.region_topic_counts_parquet_filename
    )
    doc_topic_distrib = LDAMallet.read_cell_topic_probabilities_parquet_file(
        mallet_cell_topic_probabilities_parquet_filename=lda_mallet_filenames.cell_topic_probabilities_parquet_filename
    )

    cell_topic = pd.DataFrame.from_records(
        doc_topic_distrib,
        index=cell_barcodes,
        columns=["Topic" + str(i) for i in range(1, n_topics + 1)],
    )

    region_topic = pd.DataFrame.from_records(
        topic_word_distrib,
        columns=region_ids,
        index=["Topic" + str(i) for i in range(1, n_topics + 1)],
    ).transpose()

    print("Generating cell_topic AnnData object")
    adata_cell_topic = anndata.AnnData(
        X=cell_topic
    )
    print(f"Done, shape is: {adata_cell_topic.shape}")

    print("Generating region topic AnnData object")
    adata_region_topic = anndata.AnnData(
        X=region_topic
    )
    print(f"Done, shape is: {adata_region_topic.shape}")

    print(f"Writing to: {lda_mallet_filenames.anndata_cell_topic_filename}")
    adata_cell_topic.write(lda_mallet_filenames.anndata_cell_topic_filename)

    print(f"Writing to: {lda_mallet_filenames.anndata_region_topic_filename}")
    adata_region_topic.write(lda_mallet_filenames.anndata_region_topic_filename)
