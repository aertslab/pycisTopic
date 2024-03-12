from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Literal, Sequence

import polars as pl

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction
    from pathlib import Path

# Enable Polars global string cache so all categoricals are created with the same
# string cache.
pl.enable_string_cache()


def get_tss_annotation_bed_file(
    tss_annotation_bed_filename: str | Path,
    biomart_name: str,
    to_chrom_source_name: Literal["ucsc"]
    | Literal["refseq"]
    | Literal["genbank"]
    | None = None,
    chrom_sizes_and_alias_tsv_filename: str | Path | None = None,
    ncbi_accession_id: str | None = None,
    ucsc_assembly: str | None = None,
    biomart_host: str = "http://www.ensembl.org",
    transcript_type: Sequence[str] | None = ["protein_coding"],
    use_cache: bool = True,
):
    """
    Get TSS annotation from Ensembl BioMart and write result to a BED file.

    Get TSS annotation for requested transcript types from Ensembl BioMart and write
    result to a BED file and optionally remap chromosome names from Ensembl chromosome
    names to UCSC, RefSeq or GenBank chromosome names.

    Parameters
    ----------
    tss_annotation_bed_filename
        BED output file with requested Ensembl BioMart TSS annotation and with
        optionally remapped chromosome names from Ensembl chromosome names to UCSC,
        RefSeq or GenBank chromosome names.
    biomart_name
        Ensembl BioMart ID of the dataset.
        See :func:`pycisTopic.cli.pycistopic.get_species_gene_annotation_ensembl_biomart_dataset_names`
        to get the biomart_name for species of interest:
        e.g.:
        ``hsapiens_gene_ensembl``, ``mmusculus_gene_ensembl``,
        ``dmelanogaster_gene_ensembl``, ...
    to_chrom_source_name
        If defined, remap Ensembl chromosome names to UCSC ("ucsc"), RefSeq("refseq" or
        GenBank ("genbank") chromosome names.
    chrom_sizes_and_alias_tsv_filename
        If chromosome sizes and alias TSV file exist, read chromosme sizes and alias
        mapping from the file. If chromosome sizes and alias TSV file does not exist
        and ``ncbi_accession_id`` or``ucsc_assembly`` are defined, the chromosome
        sizes and alias mapping for that option will be written to the chromosome
        sizes and alias TSV file.
    ncbi_accession_id
        NCBI genome accession ID for which to retrieve NCBI sequence reports, which
        will be used to build chromosome sizes and alias mapping, which can be used to
        map Ensembl chromosome names (from TSS annotation) to UCSC, RefSeq or GenBank
        chromosome names.
        e.g.: "GCF_000001405.40", "GCF_000001215.4", "GCF_000001215.4", ...
    ucsc_assembly
        UCSC genome accession ID for which to retrieve chromosome sizes and alias
        mapping, which can be used to map Ensembl chromosome names (from TSS
        annotation) to UCSC, RefSeq or GenBank chromosome names.
        e.g.: "hg38", "mm10", "dm6", ...
    biomart_host
        BioMart host URL to use.
          - Default: ``http://www.ensembl.org``
          - Archived Ensembl BioMart URLs:
            https://www.ensembl.org/info/website/archives/index.html
            (List of currently available archives)
    transcript_type
        Only keep list of comma separated transcript types
        (e.g.: ``["protein_coding", "pseudogene"]``) or all (``None``).
    use_cache
        Whether to cache requests to Ensembl BioMart server.

    Returns
    -------
    None.

    See Also
    --------
    pycisTopic.cli.pycistopic.get_ncbi_assembly_accessions_for_species
    pycisTopic.cli.pycistopic.get_species_gene_annotation_ensembl_biomart_dataset_names
    pycisTopic.gene_annotation.change_chromosome_source_in_bed
    pycisTopic.gene_annotation.get_biomart_dataset_name_for_species
    pycisTopic.gene_annotation.get_chrom_sizes_and_alias_mapping_from_file
    pycisTopic.gene_annotation.get_chrom_sizes_and_alias_mapping_from_ncbi
    pycisTopic.gene_annotation.get_chrom_sizes_and_alias_mapping_from_ucsc
    pycisTopic.gene_annotation.get_tss_annotation_from_ensembl
    pycisTopic.gene_annotation.write_tss_annotation_to_bed

    Examples
    --------
    Get TSS annotation BED file for human from Ensembl BioMart.

    >>> get_tss_annotation_bed_file(
    ...    tss_annotation_bed_filename="hg38.ensembl.tss.bed",
    ...    biomart_name="hsapiens_gene_ensembl",
    ... )

    Get TSS annotation BED file for human from a specific version of Ensembl
    BioMart.

    >>> get_tss_annotation_bed_file(
    ...    tss_annotation_bed_filename="hg38.ensembl_jul20222.tss.bed",
    ...    biomart_name="hsapiens_gene_ensembl",
    ...    biomart_host="http://jul2022.archive.ensembl.org/",
    ... )

    Get TSS annotation BED file for human from Ensembl BioMart and remap Ensembl
    chromosome names to UCSC chromosome names. Chromosome sizes and alias mapping TSV
    file will be saved too as `hg38.chrom_sizes_and_alias.tsv`.

    >>> get_tss_annotation_bed_file(
    ...    tss_annotation_bed_filename="hg38.ucsc.tss.bed",
    ...    biomart_name="hsapiens_gene_ensembl",
    ...    to_chrom_source_name="ucsc",
    ...    ucsc_assembly="hg38",
    ... )

    Get TSS annotation BED file for human from Ensembl BioMart and remap Ensembl
    chromosome names to UCSC chromosome names and write chromosome alias mapping
    explicitly to `hg38.explicit.chrom_sizes_and_alias.tsv` (only if it does not exist
    yet, otherwise chromosome alias mapping will be loaded from this file instead).

    >>> get_tss_annotation_bed_file(
    ...    tss_annotation_bed_filename="hg38.ucsc.tss.bed",
    ...    biomart_name="hsapiens_gene_ensembl",
    ...    to_chrom_source_name="ucsc",
    ...    chrom_sizes_and_alias_tsv_filename="hg38.explicit.chrom_sizes_and_alias.tsv",
    ...    ucsc_assembly="hg38",
    ... )

    Get TSS annotation BED file for human from Ensembl BioMart and remap Ensembl
    chromosome names to UCSC chromosome names from an existing chromosome sizes and
    alias mapping TSV file.

    >>> get_tss_annotation_bed_file(
    ...    tss_annotation_bed_filename="hg38.ucsc.tss.bed",
    ...    biomart_name="hsapiens_gene_ensembl",
    ...    to_chrom_source_name="ucsc",
    ...    chrom_sizes_and_alias_tsv_filename="hg38.explicit.chrom_sizes_and_alias.tsv",
    ... )

    """  # noqa: W505
    import pycisTopic.gene_annotation as ga

    print(
        f"- Get TSS annotation from Ensembl BioMart with the following settings:\n"
        f'  - biomart_name: "{biomart_name}"\n'
        f'  - biomart_host: "{biomart_host}"\n'
        f"  - transcript_type: {transcript_type}\n"
        f"  - use_cache: {use_cache}",
        file=sys.stderr,
    )
    tss_annotation_bed_df_pl = ga.get_tss_annotation_from_ensembl(
        biomart_name=biomart_name,
        biomart_host=biomart_host,
        transcript_type=transcript_type,
        use_cache=use_cache,
    )

    if to_chrom_source_name and (
        chrom_sizes_and_alias_tsv_filename or ncbi_accession_id or ucsc_assembly
    ):
        if chrom_sizes_and_alias_tsv_filename and os.path.exists(
            chrom_sizes_and_alias_tsv_filename
        ):
            print(
                "- Loading chromosome sizes and alias mapping from "
                f'"{chrom_sizes_and_alias_tsv_filename}".',
                file=sys.stderr,
            )
            chrom_sizes_and_alias_df_pl = (
                ga.get_chrom_sizes_and_alias_mapping_from_file(
                    chrom_sizes_and_alias_tsv_filename
                )
            )
        elif ncbi_accession_id:
            print(
                "- Getting chromosome sizes and alias mapping for "
                f'"{ncbi_accession_id}" from NCBI.',
                file=sys.stderr,
            )
            chrom_sizes_and_alias_df_pl = ga.get_chrom_sizes_and_alias_mapping_from_ncbi(
                accession_id=ncbi_accession_id,
                chrom_sizes_and_alias_tsv_filename=chrom_sizes_and_alias_tsv_filename
                if chrom_sizes_and_alias_tsv_filename
                else os.path.join(
                    os.path.dirname(tss_annotation_bed_filename),
                    f"{ncbi_accession_id}.chrom_sizes_and_alias.tsv",
                ),
            )
        elif ucsc_assembly:
            print(
                f'- Getting chromosome sizes and alias mapping for "{ucsc_assembly}" '
                "from UCSC.",
                file=sys.stderr,
            )
            chrom_sizes_and_alias_df_pl = ga.get_chrom_sizes_and_alias_mapping_from_ucsc(
                ucsc_assembly=ucsc_assembly,
                chrom_sizes_and_alias_tsv_filename=chrom_sizes_and_alias_tsv_filename
                if chrom_sizes_and_alias_tsv_filename
                else os.path.join(
                    os.path.dirname(tss_annotation_bed_filename),
                    f"{ucsc_assembly}.chrom_sizes_and_alias.tsv",
                ),
            )
        else:
            raise ValueError(
                "Chromosome sizes and alias TSV file "
                f'"{chrom_sizes_and_alias_tsv_filename}" does not exist.'
            )

        print(
            f'- Update chromosome names in TSS annotation to "{to_chrom_source_name}" '
            "chromosome names.",
            file=sys.stderr,
        )
        # Replace Ensembl chromosome names with `to_chrom_source_name` chromosome
        # names.
        tss_annotation_bed_df_pl = ga.change_chromosome_source_in_bed(
            chrom_sizes_and_alias_df_pl=chrom_sizes_and_alias_df_pl,
            bed_df_pl=tss_annotation_bed_df_pl,
            from_chrom_source_name="ensembl",
            to_chrom_source_name=to_chrom_source_name,
        )

    print(
        f'- Writing TSS annotation BED file to "{tss_annotation_bed_filename}".',
        file=sys.stderr,
    )
    ga.write_tss_annotation_to_bed(
        tss_annotation_bed_df_pl=tss_annotation_bed_df_pl,
        tss_annotation_bed_filename=tss_annotation_bed_filename,
    )


def get_species_gene_annotation_ensembl_biomart_dataset_names(
    species: str | None,
    biomart_host: str = "http://www.ensembl.org",
    use_cache: bool = True,
) -> None:
    """
    Get all avaliable gene annotation Ensembl BioMart dataset names, optionally filtered by species.

    Parameters
    ----------
    species
        Filter list of all avaliable gene annotation Ensembl BioMart dataset names
        by species.
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
    Optionally filtered list of gene annotation Ensembl BioMart dataset names.

    See Also
    --------
    pycisTopic.cli.pycistopic.get_tss_annotation_bed_file
    pycisTopic.gene_annotation.get_all_gene_annotation_ensembl_biomart_dataset_names
    pycisTopic.gene_annotation.get_biomart_dataset_name_for_species

    Example
    -------
    Get full list of gene annotation Ensembl BioMart datasets from Ensembl BioMart.
    >>> get_species_gene_annotation_ensembl_biomart_dataset_names()

    Get filtered list of gene annotation Ensembl BioMart datasets from Ensembl BioMart.
    >>> get_species_gene_annotation_ensembl_biomart_dataset_names(
    ...     species="human",
    ... )

    Get filtered list of gene annotation Ensembl BioMart datasets from an archived
    Ensembl BioMart.
    >>> get_species_gene_annotation_ensembl_biomart_dataset_names(
    ...     species="mouse",
    ...     biomart_host="http://jul2022.archive.ensembl.org/",
    ... )

    """  # noqa: W505
    import pycisTopic.gene_annotation as ga

    biomart_datasets = ga.get_all_gene_annotation_ensembl_biomart_dataset_names(
        biomart_host=biomart_host,
        use_cache=use_cache,
    )

    if not species:
        biomart_datasets.to_csv(sys.stdout, sep="\t", header=False, index=False)
    else:
        biomart_datasets_for_species = ga.get_biomart_dataset_name_for_species(
            biomart_datasets=biomart_datasets,
            species=species,
        )

        biomart_datasets_for_species.to_csv(
            sys.stdout, sep="\t", header=False, index=False
        )


def get_ncbi_assembly_accessions_for_species(species: str) -> None:
    """
    Get NCBI assembly accession numbers and assembly names for a certain species.

    Parameters
    ----------
    species
         Species name (latin name) for which to look for NCBI assembly accession
         numbers.

    Returns
    -------
    None.

    See Also
    --------
    pycisTopic.cli.pycistopic.get_tss_annotation_bed_file
    pycisTopic.gene_annotation.get_ncbi_assembly_accessions_for_species

    Examples
    --------
    >>> get_ncbi_assembly_accessions_for_species(species="homo sapiens")

    """
    import pycisTopic.gene_annotation as ga

    print(ga.get_ncbi_assembly_accessions_for_species(species))


def get_chrom_sizes_and_alias_mapping_from_ncbi(
    accession_id: str,
    chrom_sizes_and_alias_tsv_filename: str | Path,
) -> None:
    """
    Get chromosome sizes and alias mapping from NCBI sequence reports.

    Get chromosome sizes and alias mapping from NCBI sequence reports to be able to map
    chromosome names between UCSC, Ensembl, GenBank and RefSeq chromosome names.

    Parameters
    ----------
    accession_id
        NCBI assembly accession ID.
    chrom_sizes_and_alias_tsv_filename
        Write chromosome sizes and alias mapping to the specified file.

    Returns
    -------
    None.

    See Also
    --------
    pycisTopic.cli.get_chrom_sizes_and_alias_mapping_from_ucsc
    pycisTopic.cli.get_ncbi_assembly_accessions_for_species
    pycisTopic.gene_annotation.get_chrom_sizes_and_alias_mapping_from_ncbi

    Examples
    --------
    Get chromosome sizes and alias mapping for different assemblies from NCBI.

    Assemby accession IDs for a species can be queries with
    `pycisTopic.cli.get_ncbi_assembly_accessions_for_species`

    Get chromosome sizes and alias mapping for Homo sapiens and write it to a TSV file:

    >>> chrom_sizes_and_alias_hg38_df_pl = get_chrom_sizes_and_alias_mapping_from_ncbi(
    ...     accession_id="GCF_000001405.40",
    ...     chrom_sizes_and_alias_tsv_filename="GCF_000001405.40.chrom_sizes_and_alias.tsv",
    ... )

    Get chromosome sizes and alias mapping for Drosophila melanogaster and write it to
    a TSV file:

    >>> chrom_sizes_and_alias_dm6_df_pl = get_chrom_sizes_and_alias_mapping_from_ncbi(
    ...     accession_id="GCF_000001215.4",
    ...     chrom_sizes_and_alias_tsv_filename="GCF_000001215.4.chrom_sizes_and_alias.tsv",
    ... )

    """
    import pycisTopic.gene_annotation as ga

    ga.get_chrom_sizes_and_alias_mapping_from_ncbi(
        accession_id=accession_id,
        chrom_sizes_and_alias_tsv_filename=chrom_sizes_and_alias_tsv_filename,
    )


def get_chrom_sizes_and_alias_mapping_from_ucsc(
    ucsc_assembly: str,
    chrom_sizes_and_alias_tsv_filename: str | Path,
) -> None:
    """
    Get chromosome sizes and alias mapping from UCSC genome browser.

    Get chromosome sizes and alias mapping from UCSC genome browser for UCSC assembly
    to be able to map chromosome names between UCSC, Ensembl, GenBank and RefSeq
    chromosome names.

    Parameters
    ----------
    ucsc_assembly:
        UCSC assembly names (``hg38``, ``mm10``, ``dm6``, ...).
    chrom_sizes_and_alias_tsv_filename:
        Write the chromosome sizes and alias mapping to the specified file.

    Returns
    -------
    None.

    See Also
    --------
    pycisTopic.cli.get_chrom_sizes_and_alias_mapping_from_ncbi
    pycisTopic.gene_annotation.get_chrom_sizes_and_alias_mapping_from_ucsc

    Examples
    --------
    Get chromosome sizes and alias mapping for different assemblies from UCSC.

    Get chromosome sizes and alias mapping for hg38 and also write it to a TSV file:

    >>> chrom_sizes_and_alias_hg38_df_pl = get_chrom_sizes_and_alias_mapping_from_ucsc(
    ...     ucsc_assembly="hg38",
    ...     chrom_sizes_and_alias_tsv_filename="hg38.chrom_sizes_and_alias.tsv",
    ... )

    Get chromosome sizes and alias mapping for dm6 and also write it to a TSV file:

    >>> chrom_sizes_and_alias_hg38_df_pl = get_chrom_sizes_and_alias_mapping_from_ucsc(
    ...     ucsc_assembly="dm6",
    ...     chrom_sizes_and_alias_tsv_filename="dm6.chrom_sizes_and_alias.tsv",
    ... )

    """
    import pycisTopic.gene_annotation as ga

    ga.get_chrom_sizes_and_alias_mapping_from_ucsc(
        ucsc_assembly=ucsc_assembly,
        chrom_sizes_and_alias_tsv_filename=chrom_sizes_and_alias_tsv_filename,
    )


def run_tss_get_tss_annotation(args):
    get_tss_annotation_bed_file(
        tss_annotation_bed_filename=args.tss_annotation_bed_filename,
        biomart_name=args.biomart_name,
        to_chrom_source_name=args.to_chrom_source_name,
        chrom_sizes_and_alias_tsv_filename=args.chrom_sizes_and_alias_tsv_filename,
        ncbi_accession_id=args.ncbi_accession_id,
        ucsc_assembly=args.ucsc_assembly,
        biomart_host=args.biomart_host,
        transcript_type=(
            args.transcript_type.rstrip(",").split(",")
            if args.transcript_type
            and args.transcript_type != ""
            and args.transcript_type != "all"
            else None
        ),
        use_cache=args.use_cache,
    )


def run_tss_gene_annotation_list(args):
    get_species_gene_annotation_ensembl_biomart_dataset_names(
        species=args.filter,
        biomart_host=args.biomart_host,
        use_cache=args.use_cache,
    )


def run_tss_get_ncbi_acc(args):
    get_ncbi_assembly_accessions_for_species(species=args.species)


def run_tss_get_ncbi_chrom_sizes_and_alias_mapping(args):
    get_chrom_sizes_and_alias_mapping_from_ncbi(
        accession_id=args.ncbi_accession_id,
        chrom_sizes_and_alias_tsv_filename=args.chrom_sizes_and_alias_tsv_filename,
    )


def run_tss_get_ucsc_chrom_sizes_and_alias_mapping(args):
    get_chrom_sizes_and_alias_mapping_from_ucsc(
        ucsc_assembly=args.ucsc_assembly,
        chrom_sizes_and_alias_tsv_filename=args.chrom_sizes_and_alias_tsv_filename,
    )


def add_parser_tss(subparsers: _SubParsersAction[ArgumentParser]):
    parser_tss = subparsers.add_parser(
        "tss",
        help="Get TSS gene annotation from Ensembl BioMart.",
    )

    subparser_tss = parser_tss.add_subparsers(
        title="TSS",
        dest="tss",
        help="Get TSS gene annotation from Ensembl BioMart.",
    )
    subparser_tss.required = True

    parser_tss_get_tss = subparser_tss.add_parser(
        "get_tss",
        help="Get TSS gene annotation from Ensembl BioMart.",
    )
    parser_tss_get_tss.set_defaults(func=run_tss_get_tss_annotation)

    parser_tss_get_tss.add_argument(
        "-o",
        "--output",
        dest="tss_annotation_bed_filename",
        action="store",
        type=str,
        required=True,
        help="BED output file with selected Ensembl BioMart TSS annotation and "
        "with optionally remapped chromosome names from Ensembl chromosome names to "
        "UCSC, RefSeq or GenBank chromosome names.",
    )

    group_tgt_biomart = parser_tss_get_tss.add_argument_group(
        "Ensembl BioMart", "Ensembl BioMart server settings."
    )
    group_tgt_biomart.add_argument(
        "-n",
        "--name",
        dest="biomart_name",
        action="store",
        type=str,
        required=True,
        help="Ensembl BioMart gene annotation name of the dataset. "
        "Run `pycistopic tss gene_annotation_list` to get a list of available gene "
        "annotation names. "
        'e.g.: "hsapiens_gene_ensembl", "mmusculus_gene_ensembl", '
        '"dmelanogaster_gene_ensembl", ...',
    )

    group_tgt_biomart.add_argument(
        "-t",
        "--transcript",
        dest="transcript_type",
        action="store",
        type=str,
        required=False,
        default="protein_coding",
        help="Only keep comma separated list of specified transcript types "
        '(e.g.: "protein_coding,pseudogene") or all ("" or "all": useful to see all possible transcript types). '
        'Default: "protein_coding".',
    )

    group_tgt_biomart.add_argument(
        "-s",
        "--server",
        dest="biomart_host",
        action="store",
        type=str,
        required=False,
        default="http://www.ensembl.org",
        help='BioMart host URL to use. Default: "http://www.ensembl.org". '
        "Archived Ensembl BioMart URLs: "
        "https://www.ensembl.org/info/website/archives/index.html "
        "(List of currently available archives).",
    )

    group_tgt_biomart.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        required=False,
        default=True,
        help="Disable caching of requests to Ensembl BioMart server.",
    )

    group_tgt_remap_chroms = parser_tss_get_tss.add_argument_group(
        "Remap chromosomes",
        "Remap Ensembl chromosome names in TSS file to UCSC, RefSeq or GenBank chromosome names.",
    )

    group_tgt_remap_chroms.add_argument(
        "--to-chrom-source",
        dest="to_chrom_source_name",
        action="store",
        type=str,
        required=False,
        help="Chromosome source name to which the Ensembl chromosome names should be "
        'mapped to: "ucsc", "refseq" or "genbank".',
    )

    group_tgt_remap_chroms.add_argument(
        "--chrom-sizes-alias",
        dest="chrom_sizes_and_alias_tsv_filename",
        action="store",
        type=str,
        required=False,
        help="Read/write chromosome sizes and alias TSV file with chromosome sizes and"
        "alias mappings, which can be used to map Ensembl chromosome names (from TSS "
        "annotation) to UCSC, RefSeq or GenBank chromosome names. Read from chromosome "
        'sizes and alias TSV file if "--ncbi" and "--ucsc" are not specified and write '
        "to chromosome sizes and alias TSV file one of them is.",
    )

    group_tgt_chrom_sizes_and_alias = (
        group_tgt_remap_chroms.add_mutually_exclusive_group()
    )

    group_tgt_chrom_sizes_and_alias.add_argument(
        "--ncbi",
        dest="ncbi_accession_id",
        action="store",
        type=str,
        required=False,
        help="NCBI genome accession ID for which to retrieve NCBI sequence reports, "
        "which will be used to build chromosome sizes and alias mapping, which can "
        "be used to map Ensembl chromosome names (from TSS annotation) to UCSC, "
        "RefSeq or GenBank chromosome names. Run `pycistopic tss get_ncbi_acc` to get "
        "all possible NCBI genome accession IDs for a species. "
        'e.g.: "GCF_000001405.40", "GCF_000001215.4", "GCF_000001215.4", ...',
    )

    group_tgt_chrom_sizes_and_alias.add_argument(
        "--ucsc",
        dest="ucsc_assembly",
        action="store",
        type=str,
        required=False,
        help="UCSC genome accession ID for which to retrieve chromosome sizes and "
        "alias mapping, which can be used to map Ensembl chromosome names (from TSS "
        "annotation) to UCSC, RefSeq or GenBank chromosome names. "
        'e.g.: "hg38", "mm10", "dm6", ...',
    )

    parser_tss_gene_annotation_list = subparser_tss.add_parser(
        "gene_annotation_list",
        help="Get list of all Ensembl BioMart gene annotation names.",
    )
    parser_tss_gene_annotation_list.set_defaults(func=run_tss_gene_annotation_list)

    parser_tss_gene_annotation_list.add_argument(
        "-f",
        "--filter",
        dest="filter",
        action="store",
        type=str,
        required=False,
        help="Only keep list of Ensembl BioMart gene annotation names that contain specified string.",
    )

    group_tgal_biomart = parser_tss_gene_annotation_list.add_argument_group(
        "Ensembl BioMart", "Ensembl BioMart server settings."
    )

    group_tgal_biomart.add_argument(
        "-s",
        "--server",
        dest="biomart_host",
        action="store",
        type=str,
        required=False,
        default="http://www.ensembl.org",
        help='BioMart host URL to use. Default: "http://www.ensembl.org". '
        "Archived Ensembl BioMart URLs: "
        "https://www.ensembl.org/info/website/archives/index.html "
        "(List of currently available archives).",
    )

    group_tgal_biomart.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        required=False,
        default=True,
        help="Disable caching of requests to Ensembl BioMart server.",
    )

    parser_tss_get_ncbi_acc = subparser_tss.add_parser(
        "get_ncbi_acc",
        help="Get NCBI assembly accession numbers and assembly names for a certain species.",
    )
    parser_tss_get_ncbi_acc.set_defaults(func=run_tss_get_ncbi_acc)

    parser_tss_get_ncbi_acc.add_argument(
        "-s",
        "--species",
        dest="species",
        action="store",
        type=str,
        required=True,
        help="Species name (latin name) for which to look for NCBI assembly accession "
        'numbers. e.g.: "homo sapiens".',
    )

    parser_tss_get_ncbi_chrom_sizes_and_alias_mapping = subparser_tss.add_parser(
        "get_ncbi_chrom_sizes_and_alias_mapping",
        help="Get chromosome sizes and alias mapping from NCBI sequence reports.",
    )
    parser_tss_get_ncbi_chrom_sizes_and_alias_mapping.set_defaults(
        func=run_tss_get_ncbi_chrom_sizes_and_alias_mapping
    )

    parser_tss_get_ncbi_chrom_sizes_and_alias_mapping.add_argument(
        "--ncbi",
        dest="ncbi_accession_id",
        action="store",
        type=str,
        required=False,
        help="NCBI genome accession ID for which to retrieve NCBI sequence reports, "
        "which will be used to build chromosome sizes and alias mappings, which can "
        "be used to map Ensembl chromosome names (from TSS annotation) to UCSC, "
        "RefSeq or GenBank chromosome names. Run `pycistopic tss get_ncbi_acc` to get "
        "all possible NCBI genome accession IDs for a species. "
        'e.g.: "GCF_000001405.40", "GCF_000001215.4", "GCF_000001215.4", ...',
    )

    parser_tss_get_ncbi_chrom_sizes_and_alias_mapping.add_argument(
        "--chrom-sizes-alias",
        dest="chrom_sizes_and_alias_tsv_filename",
        action="store",
        type=str,
        required=False,
        help="Write chromosome sizes and alias TSV file with chromosome sizes and "
        "alias mapping, which can be used to map Ensembl chromosome names (from TSS "
        "annotation) to UCSC, RefSeq or GenBank chromosome names.",
    )

    parser_tss_get_ucsc_chrom_sizes_and_alias_mapping = subparser_tss.add_parser(
        "get_ucsc_chrom_sizes_and_alias_mapping",
        help="Get chromosome sizes and alias mapping from UCSC.",
    )
    parser_tss_get_ucsc_chrom_sizes_and_alias_mapping.set_defaults(
        func=run_tss_get_ucsc_chrom_sizes_and_alias_mapping
    )

    parser_tss_get_ucsc_chrom_sizes_and_alias_mapping.add_argument(
        "--ucsc",
        dest="ucsc_assembly",
        action="store",
        type=str,
        required=False,
        help="UCSC genome accession ID for which to retrieve chromosome sizes and "
        "alias mapping, which can be used to map Ensembl chromosome names (from TSS "
        "annotation) to UCSC, RefSeq or GenBank chromosome names. "
        'e.g.: "hg38", "mm10", "dm6", ...',
    )

    parser_tss_get_ucsc_chrom_sizes_and_alias_mapping.add_argument(
        "--chrom-sizes-alias",
        dest="chrom_sizes_and_alias_tsv_filename",
        action="store",
        type=str,
        required=False,
        help="Write chromosome sizes and alias TSV file with chromosome sizes and "
        "alias mapping, which can be used to map Ensembl chromosome names (from TSS "
        "annotation) to UCSC, RefSeq or GenBank chromosome names.",
    )
