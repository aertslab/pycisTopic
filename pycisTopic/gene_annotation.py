from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl


def get_all_gene_annotation_ensembl_biomart_dataset_names(
    biomart_host: str = "http://www.ensembl.org",
) -> pd.DataFrame:
    """
    Get all avaliable gene annotation Ensembl BioMart dataset names.

    Parameters
    ----------
    biomart_host:
        BioMart host URL to use.
          - Default: "http://www.ensembl.org"
          - Archived Ensembl BioMart URLs:
            https://www.ensembl.org/info/website/archives/index.html (List of currently available archives)

    Returns
    -------
    Pandas dataframe with all available gene annotation Ensembl BioMart datasets.

    Examples
    --------
    >>> biomart_latest_datasets = get_all_biomart_ensembl_dataset_names(
    ...    biomart_host="http://www.ensembl.org"
    ... )
    >>> biomart_jul2022_datasets = get_all_biomart_ensembl_dataset_names(
    ...     biomart_host="http://jul2022.archive.ensembl.org/"
    ... )

    """

    import pybiomart as pbm

    biomart_server = pbm.Server(host=biomart_host)
    biomart = biomart_server["ENSEMBL_MART_ENSEMBL"]

    biomart_datasets = biomart.list_datasets()

    return biomart_datasets


def get_biomart_dataset_name_for_species(
    biomart_datasets: pd.DataFrame, species: str
) -> pd.DataFrame:
    """
    Get Ensembl BioMart dataset names for species of interest.

    Parameters
    ----------
    biomart_datasets:
        All gene annotation Ensembl BioMart datasets (`get_all_gene_annotation_ensembl_biomart_dataset_names()`).
    species:
        Species name to search for.

    Returns
    -------
    Filtered list of Biomart Ensembl dataset names.

    Example
    -------
    >>> biomart_datasets = get_all_biomart_ensembl_dataset_names(biomart_host="http://www.ensembl.org")
    >>> get_biomart_dataset_name_for_species(biomart_datasets=biomart_datasets, species="mouse")

    """

    return biomart_datasets[
        biomart_datasets.display_name.str.lower().str.contains(species.lower())
        | biomart_datasets.name.str.lower().str.contains(species.lower())
    ]


def get_tss_annotation_from_ensembl(
    biomart_name: str,
    biomart_host: str = "http://www.ensembl.org",
    transcript_type: list[str] | None = ["protein_coding"],
) -> pl.DataFrame:
    """
    Get TSS annotation for requested transcript types from Ensembl BioMart.

    Parameters
    ----------
    biomart_name
        Ensembl BioMart ID of the dataset.
        See `get_biomart_dataset_name_for_species()` to get the biomart_name for species of interest, if needed.
        e.g.: "hsapiens_gene_ensembl", "mmusculus_gene_ensembl", "dmelanogaster_gene_ensembl", ...
    biomart_host
        BioMart host URL to use.
          - Default: "http://www.ensembl.org"
          - Archived Ensembl BioMart URLs:
            https://www.ensembl.org/info/website/archives/index.html (List of currently available archives)
    transcript_type
        Only keep list of specified transcript types (e.g.: ["protein_coding"]) or all (None).

    Returns
    -------
    Polars DataFrame with TSS positions in BED format.

    Examples
    --------
    >>> get_tss_annotation_from_ensembl(biomart_name="hsapiens_gene_ensembl")
    >>> get_tss_annotation_from_ensembl(biomart_name="hsapiens_gene_ensembl", biomart_host="http://jul2022.archive.ensembl.org/")

    """

    import pybiomart as pbm

    dataset = pbm.Dataset(name=biomart_name, host=biomart_host)

    ensembl_tss_annotation = dataset.query(
        attributes=[
            "chromosome_name",
            "transcription_start_site",
            "strand",
            "external_gene_name",
            "transcript_biotype",
        ],
        filters={"transcript_biotype": transcript_type} if transcript_type else None,
    )

    ensembl_tss_annotation_bed_df_pl = pl.from_pandas(ensembl_tss_annotation).select(
        [
            pl.col("Chromosome/scaffold name").alias("Chromosome"),
            # Start coordinate of TSS in BED format.
            (pl.col("Transcription start site (TSS)") - 1)
            .cast(pl.Int32)
            .alias("Start"),
            # End coordinate of TSS in BED format.
            pl.col("Transcription start site (TSS)").cast(pl.Int32).alias("End"),
            pl.col("Gene name").alias("Gene"),
            pl.lit(".").alias("Score"),
            # Convert 1, -1 and 0 to "+", "-" and "." respectively.
            (
                pl.when(pl.col("Strand") == 1)
                .then(pl.lit("+"))
                .otherwise(
                    (
                        pl.when(pl.col("Strand") == -1)
                        .then(pl.lit("-"))
                        .otherwise(".")
                    )
                )
                .alias("Strand")
            ),
            pl.col("Transcript type").alias("Transcript_type"),
        ]
    )

    return ensembl_tss_annotation_bed_df_pl


def get_chrom_alias_mapping(
    ucsc_assembly: str | None = None, chrom_alias_tsv_filename: str | Path | None = None
) -> pl.DataFrame:
    """
    Get chromosome alias mapping from UCSC genome browser for UCSC assembly to be able to map
    chromosome names between UCSC, Ensembl, GenBank and RefSeq chromosome names or read mapping
    from local file (chromosome_alias_tsv_file) instead.

    Parameters
    ----------
    ucsc_assembly:
        UCSC assembly names ("hg38", "mm10", "dm6", ...)
        If specified, retrieve data from UCSC.
    chrom_alias_tsv_filename:
        If specified and `ucsc_assembly` is not `None`, write the chromosome alias mapping to the specified file.
        If specified and `ucsc_assembly=None`, read chromosome alias mapping from specified file.

    Returns
    -------
    Polars Dataframe with chromosome alias mappings between UCSC, Ensembl, GenBank and RefSeq chromosome names.

    Examples
    --------
    Get chromosome aliases for different assemblies from UCSC:
    >>> chrom_alias_hg38_df_pl = get_chrom_alias_mapping(ucsc_assembly="hg38")
    >>> chrom_alias_mm10_df_pl = get_chrom_alias_mapping(ucsc_assembly="mm10")
    >>> chrom_alias_dm6_df_pl = get_chrom_alias_mapping(ucsc_assembly="dm6")

    Get chromosome aliases for hg38 and also write it to a TSV file:
    >>> chrom_alias_hg38_df_pl = get_chrom_alias_mapping(
    ...     ucsc_assembly="hg38",
    ...     chrom_alias_tsv_filename="chrom_alias_hg38_mapping.tsv"
    ... )

    Get chromosome aliases for hg38 from a previous written TSV file:
    >>> chrom_alias_hg38_from_file_df_pl = get_chrom_alias_mapping(
    ...    chrom_alias_tsv_filename="chrom_alias_hg38_mapping.tsv"
    ... )

    """

    if ucsc_assembly:
        # Get chromosome alias file from UCSC genome browser.
        chrom_alias_tsv_url_filename = f"https://hgdownload.soe.ucsc.edu/goldenPath/{ucsc_assembly}/bigZips/{ucsc_assembly}.chromAlias.txt"
        chrom_alias_df_pl = pl.read_csv(
            chrom_alias_tsv_url_filename, sep="\t", has_header=False, comment_char="#"
        )

        # Set column headers for Polars DataFrame.
        if chrom_alias_df_pl.width == 5:
            # hg38
            chrom_alias_df_pl.columns = [
                "ucsc",
                "assembly",
                "ensembl",
                "genbank",
                "refseq",
            ]
        elif chrom_alias_df_pl.width == 4:
            # mm10 and dm6.
            # Wrong header: "# sequenceName	alias names	UCSC database: mm10".
            chrom_alias_df_pl.columns = ["ucsc", "refseq", "genbank", "ensembl"]
        else:
            raise ValueError(
                f"Chromosome alias TSV file with unsupported number of columns ({chrom_alias_df_pl.width})."
            )

        if chrom_alias_tsv_filename and (
            isinstance(chrom_alias_tsv_filename, str)
            or isinstance(chrom_alias_tsv_filename, Path)
        ):
            chrom_alias_df_pl.write_csv(
                file=chrom_alias_tsv_filename, sep="\t", has_header=True
            )
    elif chrom_alias_tsv_filename and (
        isinstance(chrom_alias_tsv_filename, str)
        or isinstance(chrom_alias_tsv_filename, Path)
    ):
        chrom_alias_df_pl = pl.read_csv(
            chrom_alias_tsv_filename, sep="\t", has_header=True, comment_char="#"
        )
    else:
        raise ValueError(
            """Both "ucsc_assembly" and "chrom_alias_tsv_filename" can't be `None`."""
        )

    return chrom_alias_df_pl


def check_most_likely_chromosome_source_in_bed(
    chrom_alias_df_pl: pl.DataFrame, bed_df_pl: pl.DataFrame
) -> (str, pl.DataFrame):
    """
    Check which chromosome source (UCSC, Ensembl, GenBank and RefSeq) is
    the most likely in the provided Polars DataFrame with BED entries.

    Parameters
    ----------
    chrom_alias_df_pl
        Polars DataFrame with UCSC chromosome alias content. See `get_chrom_alias_mapping()`.
    bed_df_pl
        Polars DataFrame with BED entries. See `read_bed_to_polars_df()`.

    Returns
    -------

    Most likely chromosome source and Polars DataFrame with the ranking of all possible chromosome sources.

    Examples
    --------
    >>> chrom_alias_hg38_df_pl = get_chrom_alias_mapping(ucsc_assembly="hg38")
    >>> bed_df_pl = read_bed_to_polars_df("test.bed", engine="pyarrow")
    >>> best_chrom_source_name, chrom_source_stats_df_pl = check_most_likely_chromosome_source_in_bed(
    ...     chrom_alias_df_pl=chrom_alias_hg38_df_pl,
    ...     bed_df_pl=bed_df_pl,
    ... )
    >>> print(best_chrom_source_name, chrom_source_stats_df_pl)

    """

    # Get all unique chromosome names from BED file.
    chroms_from_bed = bed_df_pl.get_column("Chromosome").unique()

    # Check how often those chromosome names are found for each chromosome source (UCSC, Ensembl, GenBank and RefSeq).
    chrom_source_stats_df_pl = chrom_alias_df_pl.select(
        [
            pl.col(column_name).is_in(chroms_from_bed).sum()
            for column_name in chrom_alias_df_pl.columns
        ]
    )

    # Get the best chromosome source.
    best_chrom_source_name = chrom_source_stats_df_pl.transpose(
        include_header=True
    ).filter(pl.col("column_0") == pl.col("column_0").max())[0, 0]

    return best_chrom_source_name, chrom_source_stats_df_pl


def change_chromosome_source_in_bed(
    chrom_alias_df_pl: pl.DataFrame,
    bed_df_pl: pl.DataFrame,
    from_chrom_source_name: str,
    to_chrom_source_name: str,
) -> pl.DataFrame:
    """
    Change chromosome names from Polars DataFrame with BED entries from one chromosome source to another one.

    Parameters
    ----------
    chrom_alias_df_pl
        Polars DataFrame with UCSC chromosome alias content. See `get_chrom_alias_mapping()`.
    bed_df_pl
        Polars DataFrame with BED entries for which chromosome names need to be remapped
        from from_chrom_source_name to to_chrom_source_name. See `read_bed_to_polars_df()`.
    from_chrom_source_name
        Current chromosome source name for the input BED file: "ucsc", "ensembl", "genbank" or "refseq".
        Can be guessed with `check_most_likely_chromosome_source_in_bed()`.
    to_chrom_source_name
        Chromosome source name to which the output Polars DataFrame with BED entries should be mapped:
        "ucsc", "ensembl", "genbank" or "refseq"

    Returns
    -------
    Polars Dataframe with BED entries with changed chromosome names.

    Examples
    --------
    Get chromosome alias mapping for hg38.
    >>> chrom_alias_hg38_df_pl = get_chrom_alias_mapping(ucsc_assembly="hg38")

    Get gene annotation for hg38 from Ensembl BioMart.
    >>> hg38_gene_annotation_bed_df_pl = get_tss_annotation_from_ensembl(biomart_name="hsapiens_gene_ensembl")

    Change Ensembl chromosome names with UCSC chromosome names in gene annotation for hg38.
    >>> hg38_gene_annotation_ucsc_chroms_bed_df_pl = change_chromosome_source_in_bed(
    ...     chrom_alias_df_pl=chrom_alias_hg38_df_pl,
    ...     bed_df_pl=hg38_gene_annotation_bed_df_pl,
    ...     from_chrom_source_name="ensembl",
    ...     to_chrom_source_name="ucsc",
    ... )
    >>> hg38_gene_annotation_bed_df_pl

    """

    chrom_source_stats_df_pl = (
        bed_df_pl.join(
            chrom_alias_df_pl.select(
                [
                    pl.col(from_chrom_source_name),
                    pl.col(to_chrom_source_name),
                ]
            ),
            left_on="Chromosome",
            right_on=from_chrom_source_name,
            how="left",
        )
        .with_column(
            pl.when(pl.col(to_chrom_source_name).is_null())
            .then(pl.col("Chromosome"))
            .otherwise(pl.col(to_chrom_source_name))
            .cast(pl.Categorical)
            .alias("Chromosome")
        )
        .drop(to_chrom_source_name)
    )

    return chrom_source_stats_df_pl
