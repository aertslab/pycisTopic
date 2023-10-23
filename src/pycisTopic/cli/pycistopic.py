from __future__ import annotations

import argparse
import os
import sys
from typing import Literal, Sequence

import polars as pl

# Enable Polars global string cache so all categoricals are created with the same
# string cache.
pl.enable_string_cache(True)


def get_tss_annotation_bed_file(
    tss_annotation_bed_filename: str,
    biomart_name: str,
    to_chrom_source_name: Literal["ucsc"]
    | Literal["refseq"]
    | Literal["genbank"]
    | None = None,
    chrom_alias_tsv_filename: str | None = None,
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
    chrom_alias_tsv_filename
        If chromosome alias TSV file exist, read chromosme alias mapping from the file.
        If chromosome alias TSV file does not exist and ``ncbi_accession_id`` or
        ``ucsc_assembly`` are defined, the chromosome alias mapping for that option
        will be written to the chromosome alias TSV file.
    ncbi_accession_id
        NCBI genome accession ID for which to retrieve NCBI sequence reports, which
        will be used to build chromosome alias mappings, which can be used to map
        Ensembl chromosome names (from TSS annotation) to UCSC, RefSeq or GenBank
        chromosome names. e.g.: "GCF_000001405.40", "GCF_000001215.4",
        "GCF_000001215.4", ...
    ucsc_assembly
        UCSC genome accession ID for which to retrieve chromosome alias mappings,
        which can be used to map Ensembl chromosome names (from TSS annotation) to
        UCSC, RefSeq or GenBank chromosome names. e.g.: "hg38", "mm10", "dm6", ...
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
    pycisTopic.gene_annotation.get_chrom_alias_mapping_from_file
    pycisTopic.gene_annotation.get_chrom_alias_mapping_from_ncbi
    pycisTopic.gene_annotation.get_chrom_alias_mapping_from_ucsc
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
    chromosome names to UCSC chromosome names. Chromosome alias mapping TSV file
    will be saved too as `hg38.chrom_alias.tsv`.
    >>> get_tss_annotation_bed_file(
    ...    tss_annotation_bed_filename="hg38.ucsc.tss.bed",
    ...    biomart_name="hsapiens_gene_ensembl",
    ...    to_chrom_source_name="ucsc",
    ...    ucsc_assembly="hg38",
    ... )

    Get TSS annotation BED file for human from Ensembl BioMart and remap Ensembl
    chromosome names to UCSC chromosome names and write chromosome alias mapping
    explicitly to `hg38.explicit.chrom_alias.tsv` (only if it does not exist yet,
    otherwise chromosome alias mapping will be loaded from this file instead).
    >>> get_tss_annotation_bed_file(
    ...    tss_annotation_bed_filename="hg38.ucsc.tss.bed",
    ...    biomart_name="hsapiens_gene_ensembl",
    ...    to_chrom_source_name="ucsc",
    ...    chrom_alias_tsv_filename="hg38.explicit.chrom_alias.tsv",
    ...    ucsc_assembly="hg38",
    ... )

    Get TSS annotation BED file for human from Ensembl BioMart and remap Ensembl
    chromosome names to UCSC chromosome names from an existing chromosome alias
    TSV file.
    >>> get_tss_annotation_bed_file(
    ...    tss_annotation_bed_filename="hg38.ucsc.tss.bed",
    ...    biomart_name="hsapiens_gene_ensembl",
    ...    to_chrom_source_name="ucsc",
    ...    chrom_alias_tsv_filename="hg38.explicit.chrom_alias.tsv",
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
        chrom_alias_tsv_filename or ncbi_accession_id or ucsc_assembly
    ):
        if chrom_alias_tsv_filename and os.path.exists(chrom_alias_tsv_filename):
            print(
                f'- Loading chromosome alias mapping from "{chrom_alias_tsv_filename}".',
                file=sys.stderr,
            )
            chrom_alias_df_pl = ga.get_chrom_alias_mapping_from_file(
                chrom_alias_tsv_filename
            )
        elif ncbi_accession_id:
            print(
                f'- Getting chromosome alias mapping for "{ncbi_accession_id}" from '
                "NCBI.",
                file=sys.stderr,
            )
            chrom_alias_df_pl = ga.get_chrom_alias_mapping_from_ncbi(
                accession_id=ncbi_accession_id,
                chrom_alias_tsv_filename=chrom_alias_tsv_filename
                if chrom_alias_tsv_filename
                else os.path.join(
                    os.path.dirname(tss_annotation_bed_filename),
                    f"{ncbi_accession_id}.chrom_alias.tsv",
                ),
            )
        elif ucsc_assembly:
            print(
                f"- Getting chromosome alias mapping for {ucsc_assembly} from UCSC.",
                file=sys.stderr,
            )
            chrom_alias_df_pl = ga.get_chrom_alias_mapping_from_ucsc(
                ucsc_assembly=ucsc_assembly,
                chrom_alias_tsv_filename=chrom_alias_tsv_filename
                if chrom_alias_tsv_filename
                else os.path.join(
                    os.path.dirname(tss_annotation_bed_filename),
                    f"{ucsc_assembly}.chrom_alias.tsv",
                ),
            )
        else:
            raise ValueError(
                f'Chromosome alias TSV file "{chrom_alias_tsv_filename}" does not '
                "exist."
            )

        print(
            f'- Update chromosome names in TSS annotation to "{to_chrom_source_name}" '
            "chromosome names.",
            file=sys.stderr,
        )
        # Replace Ensembl chromosome names with `to_chrom_source_name` chromosome
        # names.
        tss_annotation_bed_df_pl = ga.change_chromosome_source_in_bed(
            chrom_alias_df_pl=chrom_alias_df_pl,
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


def qc(
    fragments_tsv_filename: str,
    regions_bed_filename: str,
    tss_annotation_bed_filename: str,
    output_prefix: str,
    tss_flank_window: int = 1000,
    tss_smoothing_rolling_window: int = 10,
    tss_minimum_signal_window: int = 100,
    tss_window: int = 50,
    tss_min_norm: int = 0.2,
    use_genomic_ranges: bool = True,
    min_fragments_per_cb: int = 50,
    collapse_duplicates: bool = True,
) -> None:
    """
    Compute quality check statistics from fragments file.

    Parameters
    ----------
    fragments_tsv_filename
        Fragments TSV filename which contains scATAC fragments.
    regions_bed_filename
        Consensus peaks / SCREEN regions BED file.
        Used to calculate amount of fragments in peaks.
    tss_annotation_bed_filename
        TSS annotation BED file.
        Used to calculate distance of fragments to TSS positions.
    output_prefix
        Output prefix to use for QC statistics parquet output files.
    tss_flank_window
        Flanking window around the TSS.
        Used for intersecting fragments with TSS positions and keeping cut sites.
        Default: ``1000`` (+/- 1000 bp).
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    tss_smoothing_rolling_window
        Rolling window used to smooth the cut sites signal.
        Default: ``10``.
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    tss_minimum_signal_window
        Average signal in the tails of the flanking window around the TSS:
           - ``[-flank_window, -flank_window + minimum_signal_window + 1]``
           - ``[flank_window - minimum_signal_window + 1, flank_window]``
        is used to normalize the TSS enrichment.
        Default: ``100`` (average signal in ``[-1000, -901]``, ``[901, 1000]``
        around TSS if `flank_window=1000`).
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    tss_window
        Window around the TSS used to count fragments in the TSS when calculating
        the TSS enrichment per cell barcode.
        Default: ``50`` (+/- 50 bp).
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    tss_min_norm
        Minimum normalization score.
        If the average minimum signal value is below this value, this number is used
        to normalize the TSS signal. This approach penalizes cells with fewer reads.
        Default: ``0.2``
        See :func:`pycisTopic.tss_profile.get_tss_profile`.
    use_genomic_ranges
        Use genomic ranges implementation for calculating intersections, instead of
        using pyranges.
    min_fragments_per_cb
        Minimum number of fragments needed per cell barcode to keep the fragments
        for those cell barcodes.
    collapse_duplicates
        Collapse duplicate fragments (same chromosomal positions and linked to the same
        cell barcode).

    Returns
    -------
    None

    """
    from pycisTopic.fragments import read_bed_to_polars_df, read_fragments_to_polars_df
    from pycisTopic.gene_annotation import read_tss_annotation_from_bed
    from pycisTopic.qc import compute_qc_stats

    tss_annotation_bed_df_pl = read_tss_annotation_from_bed(
        tss_annotation_bed_filename=tss_annotation_bed_filename
    )

    regions_df_pl = read_bed_to_polars_df(
        bed_filename=regions_bed_filename,
        min_column_count=3,
    )

    fragments_df_pl = read_fragments_to_polars_df(
        fragments_tsv_filename,
        engine="pyarrow",
    )

    (
        fragments_stats_per_cb_df_pl,
        insert_size_dist_df_pl,
        tss_norm_matrix_sample,
        tss_norm_matrix_per_cb,
    ) = compute_qc_stats(
        fragments_df_pl=fragments_df_pl,
        regions_df_pl=regions_df_pl,
        tss_annotation=tss_annotation_bed_df_pl,
        tss_flank_window=tss_flank_window,
        tss_smoothing_rolling_window=tss_smoothing_rolling_window,
        tss_minimum_signal_window=tss_minimum_signal_window,
        tss_window=tss_window,
        tss_min_norm=tss_min_norm,
        use_genomic_ranges=use_genomic_ranges,
        min_fragments_per_cb=min_fragments_per_cb,
        collapse_duplicates=collapse_duplicates,
    )

    fragments_stats_per_cb_df_pl.write_parquet(
        f"{output_prefix}.fragments_stats_per_cb.parquet",
        compression="zstd",
        use_pyarrow=True,
    )

    insert_size_dist_df_pl.write_parquet(
        f"{output_prefix}.fragments_insert_size_dist.parquet",
        compression="zstd",
        use_pyarrow=True,
    )

    tss_norm_matrix_sample.write_parquet(
        f"{output_prefix}.tss_norm_matrix_sample.parquet",
        compression="zstd",
        use_pyarrow=True,
    )

    tss_norm_matrix_per_cb.write_parquet(
        f"{output_prefix}.tss_norm_matrix_per_cb.parquet",
        compression="zstd",
        use_pyarrow=True,
    )


def run_tss_get_tss_annotation(args):
    get_tss_annotation_bed_file(
        tss_annotation_bed_filename=args.tss_annotation_bed_filename,
        biomart_name=args.biomart_name,
        to_chrom_source_name=args.to_chrom_source_name,
        chrom_alias_tsv_filename=args.chrom_alias_tsv_filename,
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


def run_qc(args):
    qc(
        fragments_tsv_filename=args.fragments_tsv_filename,
        regions_bed_filename=args.regions_bed_filename,
        tss_annotation_bed_filename=args.tss_annotation_bed_filename,
        output_prefix=args.output_prefix,
        tss_flank_window=args.tss_flank_window,
        tss_smoothing_rolling_window=args.tss_smoothing_rolling_window,
        tss_minimum_signal_window=args.tss_minimum_signal_window,
        tss_window=args.tss_window,
        tss_min_norm=args.tss_min_norm,
        use_genomic_ranges=args.use_genomic_ranges,
        min_fragments_per_cb=args.min_fragments_per_cb,
        collapse_duplicates=args.collapse_duplicates,
    )


def add_parser_tss(subparsers):
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
        "Ensembl BioMart", "Ensembl BioMart server"
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
        "--chrom-alias",
        dest="chrom_alias_tsv_filename",
        action="store",
        type=str,
        required=False,
        help="Read/write chromosome alias TSV file with chromosome alias mappings, which "
        "can be used to map Ensembl chromosome names (from TSS annotation) to UCSC, "
        "RefSeq or GenBank chromosome names. Read from chromosome alias TSV file if "
        '"--ncbi" and "--ucsc" are not specified and write to chromosome alias TSV '
        "file one of them is.",
    )

    group_tgt_chrom_alias = group_tgt_remap_chroms.add_mutually_exclusive_group()

    group_tgt_chrom_alias.add_argument(
        "--ncbi",
        dest="ncbi_accession_id",
        action="store",
        type=str,
        required=False,
        help="NCBI genome accession ID for which to retrieve NCBI sequence reports, "
        "which will be used to build chromosome alias mappings, which can be used to "
        "map Ensembl chromosome names (from TSS annotation) to UCSC, RefSeq or "
        "GenBank chromosome names. Run `pycistopic tss get_ncbi_acc` to get all "
        "possible NCBI genome accession IDs for a species. "
        'e.g.: "GCF_000001405.40", "GCF_000001215.4", "GCF_000001215.4", ...',
    )

    group_tgt_chrom_alias.add_argument(
        "--ucsc",
        dest="ucsc_assembly",
        action="store",
        type=str,
        required=False,
        help="UCSC genome accession ID for which to retrieve chromosome alias "
        "mappings, which can be used to map Ensembl chromosome names (from TSS "
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
        "Ensembl BioMart", "Ensembl BioMart server"
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


def add_parser_qc(subparsers):
    parser_qc = subparsers.add_parser(
        "qc",
        help="Run QC statistics on fragment file.",
    )
    parser_qc.set_defaults(func=run_qc)

    parser_qc.add_argument(
        "-f",
        "--fragments",
        dest="fragments_tsv_filename",
        action="store",
        type=str,
        required=True,
        help="Fragments TSV filename which contains scATAC fragments.",
    )

    parser_qc.add_argument(
        "-r",
        "--regions",
        dest="regions_bed_filename",
        action="store",
        type=str,
        required=True,
        help="""
            Consensus peaks / SCREEN regions BED file. Used to calculate amount of
            fragments in peaks.
            """,
    )

    parser_qc.add_argument(
        "-t",
        "--tss",
        dest="tss_annotation_bed_filename",
        action="store",
        type=str,
        required=True,
        help="""
            TSS annotation BED file. Used to calculate distance of fragments to TSS
            positions.
            """,
    )

    parser_qc.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        action="store",
        type=str,
        required=True,
        help="Output prefix to use for QC statistics parquet output files.",
    )

    parser_qc.add_argument(
        "--tss_flank_window",
        dest="tss_flank_window",
        action="store",
        type=int,
        required=False,
        default=1000,
        help="Flanking window around the TSS. "
        "Used for intersecting fragments with TSS positions and keeping cut sites."
        "Default: 1000 (+/- 1000 bp).",
    )

    parser_qc.add_argument(
        "--tss_smoothing_rolling_window",
        dest="tss_smoothing_rolling_window",
        action="store",
        type=int,
        required=False,
        default=10,
        help="Rolling window used to smooth the cut sites signal. Default: 10.",
    )

    parser_qc.add_argument(
        "--tss_minimum_signal_window",
        dest="tss_minimum_signal_window",
        action="store",
        type=int,
        required=False,
        default=100,
        help="""
            Average signal in the tails of the flanking window around the TSS
            ([-flank_window, -flank_window + minimum_signal_window + 1],
            [flank_window - minimum_signal_window + 1, flank_window])
            is used to normalize the TSS enrichment.
            Default: 100 (average signal in [-1000, -901], [901, 1000] around TSS,
            if flank_window=1000).
            """,
    )

    parser_qc.add_argument(
        "--tss_window",
        dest="tss_window",
        action="store",
        type=int,
        required=False,
        default=50,
        help="""
            Window around the TSS used to count fragments in the TSS when calculating
            the TSS enrichment per cell barcode.
            Default: 50 (+/- 50 bp).
            """,
    )

    parser_qc.add_argument(
        "--tss_min_norm",
        dest="tss_min_norm",
        action="store",
        type=float,
        required=False,
        default=0.2,
        help="""
            Minimum normalization score.
            If the average minimum signal value is below this value, this number is used
            to normalize the TSS signal. This approach penalizes cells with fewer reads.
            Default: 0.2
            """,
    )

    parser_qc.add_argument(
        "--use-pyranges",
        dest="use_genomic_ranges",
        action="store_false",
        required=False,
        help="""
            Use pyranges instead of genomic ranges implementation for calculating
            intersections.
            """,
    )

    parser_qc.add_argument(
        "--min_fragments_per_cb",
        dest="min_fragments_per_cb",
        action="store",
        type=int,
        required=False,
        default=50,
        help="""
            Minimum number of fragments needed per cell barcode to keep the fragments
            for those cell barcodes.
            Default: 50.
            """,
    )

    parser_qc.add_argument(
        "--dont-collapse_duplicates",
        dest="collapse_duplicates",
        action="store_false",
        required=False,
        help="""
            Don't collapse duplicate fragments (same chromosomal positions and linked to
            the same cell barcode).
            Default: collapse duplicates.
            """,
    )


def main():
    parser = argparse.ArgumentParser(description="pycisTopic CLI.")

    subparsers = parser.add_subparsers(
        title="Commands",
        description="List of available commands for pycisTopic CLI.",
        dest="command",
        help="Command description.",
    )
    subparsers.required = True

    add_parser_tss(subparsers)
    add_parser_qc(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
