from __future__ import annotations

import io
import re
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import polars as pl
import requests

if TYPE_CHECKING:
    import pandas as pd

# Enable Polars global string cache so all categoricals are created with the same
# string cache.
pl.enable_string_cache(True)


def get_all_gene_annotation_ensembl_biomart_dataset_names(
    biomart_host: str = "http://www.ensembl.org",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Get all avaliable gene annotation Ensembl BioMart dataset names.

    Parameters
    ----------
    biomart_host
        BioMart host URL to use.
          - Default: ``http://www.ensembl.org``
          - Archived Ensembl BioMart URLs:
            https://www.ensembl.org/info/website/archives/index.html
            (List of currently available archives)
    use_cache
        Whether to cache requests to Ensembl BioMart server.

    Returns
    -------
    Pandas dataframe with all available gene annotation Ensembl BioMart datasets.

    See Also
    --------
    pycisTopic.gene_annotation.get_all_gene_annotation_ensembl_biomart_dataset_names

    Examples
    --------
    >>> biomart_latest_datasets = get_all_biomart_ensembl_dataset_names(
    ...    biomart_host="http://www.ensembl.org",
    ... )
    >>> biomart_jul2022_datasets = get_all_biomart_ensembl_dataset_names(
    ...     biomart_host="http://jul2022.archive.ensembl.org/",
    ... )

    """
    import pybiomart as pbm

    biomart_server = pbm.Server(host=biomart_host, use_cache=use_cache)
    biomart = biomart_server["ENSEMBL_MART_ENSEMBL"]

    biomart_datasets = biomart.list_datasets()

    return biomart_datasets


def get_biomart_dataset_name_for_species(
    biomart_datasets: pd.DataFrame,
    species: str,
) -> pd.DataFrame:
    """
    Get gene annotation Ensembl BioMart dataset names for species of interest.

    Parameters
    ----------
    biomart_datasets
        All gene annotation Ensembl BioMart datasets
        See :func:`pycisTopic.gene_annotation.get_all_gene_annotation_ensembl_biomart_dataset_names`.
    species
        Species name to search for.

    Returns
    -------
    Filtered list of gene annotation Ensembl BioMart dataset names.

    See Also
    --------
    pycisTopic.gene_annotation.get_all_gene_annotation_ensembl_biomart_dataset_names

    Example
    -------
    >>> biomart_datasets = get_all_biomart_ensembl_dataset_names(
    ...     biomart_host="http://www.ensembl.org",
    ... )
    >>> get_biomart_dataset_name_for_species(
    ...     biomart_datasets=biomart_datasets,
    ...     species="mouse",
    ... )

    """  # noqa: W505
    return biomart_datasets[
        biomart_datasets.display_name.str.lower().str.contains(species.lower())
        | biomart_datasets.name.str.lower().str.contains(species.lower())
    ]


def get_tss_annotation_from_ensembl(
    biomart_name: str,
    biomart_host: str = "http://www.ensembl.org",
    transcript_type: Sequence[str] | None = ["protein_coding"],
    use_cache: bool = True,
) -> pl.DataFrame:
    """
    Get TSS annotation for requested transcript types from Ensembl BioMart.

    Parameters
    ----------
    biomart_name
        Ensembl BioMart ID of the dataset.
        See :func:`pycisTopic.gene_annotation.get_biomart_dataset_name_for_species`
        to get the biomart_name for species of interest:
        e.g.:
        ``hsapiens_gene_ensembl``, ``mmusculus_gene_ensembl``,
        ``dmelanogaster_gene_ensembl``, ...
    biomart_host
        BioMart host URL to use.
          - Default: ``http://www.ensembl.org``
          - Archived Ensembl BioMart URLs:
            https://www.ensembl.org/info/website/archives/index.html
            (List of currently available archives)
    transcript_type
        Only keep list of specified transcript types (e.g.: ``["protein_coding"]``) or
        all (``None``).
    use_cache
        Whether to cache requests to Ensembl BioMart server.

    Returns
    -------
    Polars DataFrame with TSS positions in BED format.

    See Also
    --------
    pycisTopic.gene_annotation.get_biomart_dataset_name_for_species
    pycisTopic.gene_annotation.read_tss_annotation_from_bed
    pycisTopic.gene_annotation.write_tss_annotation_to_bed

    Examples
    --------
    >>> tss_annotation_bed_df_pl = get_tss_annotation_from_ensembl(
    ...     biomart_name="hsapiens_gene_ensembl"
    ... )
    >>> tss_annotation_jul2022_bed_df_pl = get_tss_annotation_from_ensembl(
    ...     biomart_name="hsapiens_gene_ensembl",
    ...     biomart_host="http://jul2022.archive.ensembl.org/",
    ... )

    """
    import pybiomart as pbm

    dataset = pbm.Dataset(name=biomart_name, host=biomart_host, use_cache=use_cache)

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
                    pl.when(pl.col("Strand") == -1).then(pl.lit("-")).otherwise(".")
                )
                .alias("Strand")
            ),
            pl.col("Transcript type").alias("Transcript_type"),
        ]
    )

    return ensembl_tss_annotation_bed_df_pl


def read_tss_annotation_from_bed(tss_annotation_bed_filename: str) -> pl.DataFrame:
    """
    Read TSS annotation BED file to Polars DataFrame.

    Read TSS annotation BED file created by
    :func:`pycisTopic.gene_annotation.get_tss_annotation_from_ensembl`
    and :func:`pycisTopic.gene_annotation.write_tss_annotation_to_bed`
    to Polars DataFrame with TSS positions in BED format.

    Parameters
    ----------
    tss_annotation_bed_filename
        TSS annotation BED file to read.
        TSS annotation BED files can be written with
        :func:`pycisTopic.gene_annotation.write_tss_annotation_to_bed`
        and will have the following header line:
            `# Chromosome Start End Gene Score Strand Transcript_type`
        Minimum required columns for :func:`pycisTopic.tss_profile.get_tss_profile`:
            `Chromosome, Start (0-based BED), Strand`

    See Also
    --------
    pycisTopic.gene_annotation.change_chromosome_source_in_bed
    pycisTopic.gene_annotation.get_tss_annotation_from_ensembl
    pycisTopic.gene_annotation.write_tss_annotation_to_bed

    Examples
    --------
    Get TSS annotation from Ensembl.

    >>> tss_annotation_bed_df_pl = get_tss_annotation_from_ensembl(
    ...     biomart_name="hsapiens_gene_ensembl"
    ... )

    If your fragments files use a different chromosome convention than
    the one used by Ensembl, take a look at
    :func:`pycisTopic.gene_annotation.change_chromosome_source_in_bed`
    to convert the Ensembl chromosome names to UCSC, Ensembl, GenBank
    or RefSeq chromosome names.

    Write TSS annotation to a file.

    >>> write_tss_annotation_to_bed(
    ...     tss_annotation_bed_df_pl=tss_annotation_bed_df_pl,
    ...     tss_annotation_bed_filename="hg38.tss.bed",
    ... )

    Read TSS annotation from a file.

    >>> tss_annotation_bed_df_pl = read_tss_annotation_from_bed(
    ...     tss_annotation_bed_filename="hg38.tss.bed"
    ... )

    Returns
    -------
    Polars DataFrame with TSS positions in BED format.

    """
    tss_annotation_bed_df_pl = pl.read_csv(
        tss_annotation_bed_filename,
        separator="\t",
        # Use 0-bytes as comment character so the header can start with "# Chromosome".
        comment_char="\0",
        dtypes={
            # Convert Chromosome, Start and End column to the correct datatypes.
            "Chromosome": pl.Categorical,
            "# Chromosome": pl.Categorical,
            "Start": pl.Int32,
            "End": pl.Int32,
        },
    ).rename({"# Chromosome": "Chromosome"})

    return tss_annotation_bed_df_pl


def write_tss_annotation_to_bed(
    tss_annotation_bed_df_pl, tss_annotation_bed_filename: str
) -> None:
    """
    Write TSS annotation Polars DataFrame to a BED file.

    Write TSS annotation Polars DataFrame with TSS positions in BED format.
    to a BED file.

    Parameters
    ----------
    tss_annotation_bed_df_pl
        TSS annotation Polars DataFrame with TSS positions in BED format
        created with
        :func:`pycisTopic.gene_annotation.get_tss_annotation_from_ensembl`.
    tss_annotation_bed_filename
        TSS annotation BED file to write to.
        TSS annotation BED files from
        :func:`pycisTopic.gene_annotation.get_tss_annotation_from_ensembl`
        will have the following header line:
            `# Chromosome Start End Gene Score Strand Transcript_type`
        Minimum required columns for :func:`pycisTopic.tss_profile.get_tss_profile`:
            `Chromosome, Start (0-based BED), Strand`

    See Also
    --------
    pycisTopic.gene_annotation.change_chromosome_source_in_bed
    pycisTopic.gene_annotation.get_tss_annotation_from_ensembl
    pycisTopic.gene_annotation.read_tss_annotation_from_bed

    Examples
    --------
    Get TSS annotation from Ensembl.

    >>> tss_annotation_bed_df_pl = get_tss_annotation_from_ensembl(
    ...     biomart_name="hsapiens_gene_ensembl"
    ... )

    If your fragments files use a different chromosome convention than
    the one used by Ensembl, take a look at
    :func:`pycisTopic.gene_annotation.change_chromosome_source_in_bed`
    to convert the Ensembl chromosome names to UCSC, Ensembl, GenBank
    or RefSeq chromosome names.

    Write TSS annotation to a file.

    >>> write_tss_annotation_to_bed(
    ...     tss_annotation_bed_df_pl=tss_annotation_bed_df_pl,
    ...     tss_annotation_bed_filename="hg38.tss.bed",
    ... )

    Read TSS annotation from a file.

    >>> tss_annotation_bed_df_pl = read_tss_annotation_from_bed(
    ...     tss_annotation_bed_filename="hg38.tss.bed"
    ... )

    Returns
    -------
    Polars DataFrame with TSS positions in BED format.

    """
    tss_annotation_bed_df_pl.rename({"Chromosome": "# Chromosome"}).write_csv(
        tss_annotation_bed_filename,
        separator="\t",
    )


def get_chrom_alias_mapping(
    ucsc_assembly: str | None = None, chrom_alias_tsv_filename: str | Path | None = None
) -> pl.DataFrame:
    """
    Get chromosome alias mapping from UCSC genome browser.

    Get chromosome alias mapping from UCSC genome browser for UCSC assembly to be able
    to map chromosome names between UCSC, Ensembl, GenBank and RefSeq chromosome names
    or read mapping from local file (chromosome_alias_tsv_file) instead.

    Parameters
    ----------
    ucsc_assembly:
        UCSC assembly names (``hg38``, ``mm10``, ``dm6``, ...)
        If specified, retrieve data from UCSC.
    chrom_alias_tsv_filename:
        If specified and ``ucsc_assembly`` is not ``None``, write the chromosome alias
        mapping to the specified file.
        If specified and ``ucsc_assembly=None``, read chromosome alias mapping from
        specified file.

    Returns
    -------
    Polars Dataframe with chromosome alias mappings between UCSC, Ensembl, GenBank and
    RefSeq chromosome names.

    See Also
    --------
    pycisTopic.gene_annotation.get_chrom_alias_mapping

    Examples
    --------
    Get chromosome aliases for different assemblies from UCSC:

    >>> chrom_alias_hg38_df_pl = get_chrom_alias_mapping(ucsc_assembly="hg38")
    >>> chrom_alias_mm10_df_pl = get_chrom_alias_mapping(ucsc_assembly="mm10")
    >>> chrom_alias_dm6_df_pl = get_chrom_alias_mapping(ucsc_assembly="dm6")

    Get chromosome aliases for hg38 and also write it to a TSV file:

    >>> chrom_alias_hg38_df_pl = get_chrom_alias_mapping(
    ...     ucsc_assembly="hg38",
    ...     chrom_alias_tsv_filename="chrom_alias_hg38_mapping.tsv",
    ... )

    Get chromosome aliases for hg38 from a previous written TSV file:

    >>> chrom_alias_hg38_from_file_df_pl = get_chrom_alias_mapping(
    ...    chrom_alias_tsv_filename="chrom_alias_hg38_mapping.tsv",
    ... )

    """
    if ucsc_assembly:
        chrom_alias_tsv_url_filename = f"https://hgdownload.soe.ucsc.edu/goldenPath/{ucsc_assembly}/bigZips/{ucsc_assembly}.chromAlias.txt"

        # Get chromosome alias file from UCSC genome browser.
        response = requests.get(chrom_alias_tsv_url_filename)

        if response.status_code == 200:
            # Define some regexes to recognise chromosome names for:
            #   - genbank ID:
            #       starts with 1 or 2 capital characters followed by 5 up to 8
            #       numbers, a dot and some more numbers.
            #   - refseq IDs:
            #       starts with "NC_", "NG_", "NT_" or "NW_" followed by 5 up to 9
            #       numbers, a dot and some more numbers.
            #   - UCSC:
            #       starts with "chr".
            genbank_id_regex = "^[A-Z]{1,2}[0-9]{5,8}\\.[0-9]+$"
            refseq_id_regex = "^(NC|NG|NT|NW)_[0-9]{5,9}\\.[0-9]+$"
            ucsc_regex = "^chr"

            # Create an in memory TSV file to which the ucsc, ensembl, refseq_id and
            # genbank_id chromosome columns will be written in the correct order.
            chrom_alias_tsv_string_io = io.StringIO()

            # Write header to in memory TSV file.
            print(
                "ucsc\tensembl\trefseq_id\tgenbank_id", file=chrom_alias_tsv_string_io
            )

            for line in response.content.decode("utf-8").split("\n"):
                if not line or line.startswith("#"):
                    continue

                columns = line.split("\t")

                ucsc = ""
                ensembl = ""
                refseq_id = ""
                genbank_id = ""

                # Find correct chromosome source column for UCSC, Ensembl, RefSeq
                # and Genbank per line of a UCSC chromosome alias file, instead of
                # for whole columns at once as UCSC sorts the chromosome identifiers
                # per line alphabetically instead of keeping it per source.
                for column in columns:
                    if re.match(ucsc_regex, column):
                        ucsc = column
                    elif re.match(genbank_id_regex, column):
                        genbank_id = column
                    elif re.match(refseq_id_regex, column):
                        refseq_id = column
                    else:
                        ensembl = column

                # Write columns in the correct order to the in memory TSV file.
                print(
                    "\t".join([ucsc, ensembl, refseq_id, genbank_id]),
                    file=chrom_alias_tsv_string_io,
                )

            # Rewind file descriptor to the start of the in memory TSV file.
            chrom_alias_tsv_string_io.seek(0)

            # Read in memory TSV file as a Polars dataframe.
            chrom_alias_df_pl = pl.read_csv(
                chrom_alias_tsv_string_io,
                separator="\t",
                has_header=True,
                comment_char="#",
                # Read all columns as strings.
                infer_schema_length=0,
            )
        else:
            raise ValueError(
                f'Failed to download the file "{chrom_alias_tsv_url_filename}". '
                f"Status code: {response.status_code}.\n"
                f"{response.text}"
            )

        if chrom_alias_tsv_filename and isinstance(
            chrom_alias_tsv_filename, (str, Path)
        ):
            chrom_alias_df_pl.write_csv(
                file=chrom_alias_tsv_filename, separator="\t", has_header=True
            )
    elif chrom_alias_tsv_filename and isinstance(chrom_alias_tsv_filename, (str, Path)):
        chrom_alias_df_pl = pl.read_csv(
            chrom_alias_tsv_filename, separator="\t", has_header=True, comment_char="#"
        )
    else:
        raise ValueError(
            """Both "ucsc_assembly" and "chrom_alias_tsv_filename" can't be `None`."""
        )

    return chrom_alias_df_pl


def find_most_likely_chromosome_source_in_bed(
    chrom_alias_df_pl: pl.DataFrame, bed_df_pl: pl.DataFrame
) -> (str, pl.DataFrame):
    """
    Find which chromosome source is the most likely in the provided BED file entries.

    Find which chromosome source (UCSC, Ensembl, GenBank and RefSeq) given as a
    ``chrom_alias_df_pl`` Polars DataFrame is the most likely in the provided Polars
    DataFrame with BED entries.

    Parameters
    ----------
    chrom_alias_df_pl
        Polars DataFrame with chromosome alias content.
        See :func:`pycisTopic.gene_annotation.get_chrom_alias_mapping`.
    bed_df_pl
        Polars DataFrame with BED entries.
        See :func:`pycisTopic.fragments.read_bed_to_polars_df`.

    Returns
    -------
    Tuple of most likely chromosome source and a Polars DataFrame with the ranking of
    all possible chromosome sources.

    See Also
    --------
    pycisTopic.fragments.read_bed_to_polars_df
    pycisTopic.gene_annotation.change_chromosome_source_in_bed
    pycisTopic.gene_annotation.get_chrom_alias_mapping

    Examples
    --------
    >>> chrom_alias_hg38_df_pl = get_chrom_alias_mapping(ucsc_assembly="hg38")
    >>> bed_df_pl = read_bed_to_polars_df("test.bed", engine="pyarrow")
    >>> best_chrom_source_name, chrom_source_stats_df_pl = find_most_likely_chromosome_source_in_bed(
    ...     chrom_alias_df_pl=chrom_alias_hg38_df_pl,
    ...     bed_df_pl=bed_df_pl,
    ... )
    >>> print(best_chrom_source_name, chrom_source_stats_df_pl)

    """  # noqa: W505
    # Get all unique chromosome names from BED file.
    chroms_from_bed = bed_df_pl.get_column("Chromosome").unique()

    # Find how often those chromosome names are found for each chromosome source
    # (UCSC, Ensembl, GenBank and RefSeq).
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
        Polars DataFrame with UCSC chromosome alias content.
        See :func:`pycisTopic.gene_annotation.get_chrom_alias_mapping`.
    bed_df_pl
        Polars DataFrame with BED entries for which chromosome names need to be
        remapped from ``from_chrom_source_name`` to ``to_chrom_source_name``.
        See :func:`pycisTopic.fragments.read_bed_to_polars_df` and
        :func:`pycisTopic.gene_annotation.read_tss_annotation_from_bed`
    from_chrom_source_name
        Current chromosome source name for the input BED file: ``ucsc``, ``ensembl``,
        ``genbank`` or ``refseq``.
        Can be guessed with
        :func:`pycisTopic.gene_annotation.find_most_likely_chromosome_source_in_bed`.
    to_chrom_source_name
        Chromosome source name to which the output Polars DataFrame with BED entries
        should be mapped:
        ``ucsc``, ``ensembl``, ``genbank`` or ``refseq``.

    Returns
    -------
    Polars Dataframe with BED entries with changed chromosome names.

    See Also
    --------
    pycisTopic.fragments.read_bed_to_polars_df
    pycisTopic.gene_annotation.find_most_likely_chromosome_source_in_bed
    pycisTopic.gene_annotation.get_chrom_alias_mapping
    pycisTopic.gene_annotation.read_tss_annotation_from_bed
    pycisTopic.gene_annotation.write_tss_annotation_to_bed

    Examples
    --------
    Get chromosome alias mapping for hg38.

    >>> chrom_alias_hg38_df_pl = get_chrom_alias_mapping(ucsc_assembly="hg38")

    Get gene annotation for hg38 from Ensembl BioMart.

    >>> hg38_tss_annotation_bed_df_pl = get_tss_annotation_from_ensembl(
    ...     biomart_name="hsapiens_gene_ensembl",
    ... )
    >>> hg38_tss_annotation_bed_df_pl

    Replace Ensembl chromosome names with UCSC chromosome names in gene annotation for hg38.

    >>> hg38_tss_annotation_ucsc_chroms_bed_df_pl = change_chromosome_source_in_bed(
    ...     chrom_alias_df_pl=chrom_alias_hg38_df_pl,
    ...     bed_df_pl=hg38_tss_annotation_bed_df_pl,
    ...     from_chrom_source_name="ensembl",
    ...     to_chrom_source_name="ucsc",
    ... )
    >>> hg38_tss_annotation_ucsc_chroms_bed_df_pl

    """  # noqa: W505
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
        .with_columns(
            pl.when(pl.col(to_chrom_source_name).is_null())
            .then(pl.col("Chromosome"))
            .otherwise(pl.col(to_chrom_source_name))
            .cast(pl.Categorical)
            .alias("Chromosome")
        )
        .drop(to_chrom_source_name)
    )

    return chrom_source_stats_df_pl
