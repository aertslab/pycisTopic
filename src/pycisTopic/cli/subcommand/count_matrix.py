from typing import TYPE_CHECKING

from pycisTopic.count_matrix import (
    create_count_matrix_from_fragment_files,
    subset_accessibility_matrix,
)

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction


def run_create_count_matrix_from_fragment_files(args):
    create_count_matrix_from_fragment_files(
        sample_to_fragments_file_mapping_tsv_filename=args.sample_to_fragments_file_mapping_tsv_filename,
        sample_to_cell_barcodes_file_mapping_tsv_filenames=args.sample_to_cell_barcodes_file_tsv_filenames,
        regions_bed_filename=args.regions_bed_filename,
        output_prefix=args.output_prefix,
        blacklist_bed_filename=args.blacklist_bed_filename,
        cb_end_to_remove=args.cb_end_to_remove,
        cb_sample_separator=args.cb_sample_separator,
        fragment_matrix_type=args.fragment_matrix_type,
        bed_parser_engine=args.bed_parser_engine,
        intersection_engine=args.intersection_engine,
    )


def run_subset_count_matrix(args):
    """Subset a count matrix by cell barcodes and/or regions."""
    # Load cell barcodes to keep if provided.
    cell_barcodes_to_keep = None
    if args.cell_barcodes_to_keep is not None:
        with open(args.cell_barcodes_to_keep, "r") as f:
            cell_barcodes_to_keep = [line.strip() for line in f]

    # Load region IDs to keep if provided.
    region_ids_to_keep = None
    if args.region_ids_to_keep is not None:
        with open(args.region_ids_to_keep, "r") as f:
            region_ids_to_keep = [line.strip() for line in f]

    subset_accessibility_matrix(
        binary_accessibility_matrix_filename=args.binary_accessibility_matrix_filename,
        output_prefix=args.output_prefix,
        cell_barcodes_to_keep=cell_barcodes_to_keep,
        region_ids_to_keep=region_ids_to_keep,
    )


def add_parser_count_matrix(subparsers: _SubParsersAction[ArgumentParser]):
    parser_count_matrix = subparsers.add_parser(
        "count_matrix",
        help="Generate or subset fragment count matrix.",
        description="Generate or subset fragment count matrix.",
    )

    count_matrix_subparsers = parser_count_matrix.add_subparsers(
        title="COUNT MATRIX SUBCOMMANDS",
        dest="count_matrix_subcommand",
        help="List of count_matrix subcommands",
        description="List of count_matrix subcommands",
        required=True,
    )

    parser_count_matrix_create = count_matrix_subparsers.add_parser(
        "create",
        help="Generate fragment count matrix from multiple fragment files.",
        description="Generate fragment count matrix from multiple fragment files.",
    )

    parser_count_matrix_create.add_argument(
        "-f",
        "--sample_fragment",
        dest="sample_to_fragments_file_mapping_tsv_filename",
        action="store",
        type=str,
        required=True,
        help="Path to TSV file mapping sample IDs to fragments files. "
        "Format: tab-separated with columns `sample` and `fragments_filename`.",
    )
    parser_count_matrix_create.add_argument(
        "-c",
        "--sample_barcodes",
        dest="sample_to_cell_barcodes_file_mapping_tsv_filenames",
        action="store",
        type=str,
        required=True,
        help="Path to TSV file mapping sample IDs to cell barcode files. "
        "Format: tab-separated with columns `sample` and `cell_barcodes_filename`. "
        "Only cell barcodes listed in these files are retained in the output.",
    )
    parser_count_matrix_create.add_argument(
        "-r",
        "--regions",
        dest="regions_bed_filename",
        action="store",
        type=str,
        required=True,
        help="Path to BED file containing regions to generate count matrix on.",
    )
    parser_count_matrix_create.add_argument(
        "-o",
        "--output_prefix",
        dest="output_prefix",
        action="store",
        type=str,
        required=True,
        help="""
        Output prefix for (binary) fragment count matrix file.
        Generates: `OUTPUT_PREFIX.cell_barcodes.tsv` (cell identifiers), `OUTPUT_PREFIX.region_ids.tsv` (genomic regions),
        and `OUTPUT_PREFIX.matrix.mtx` (sparse count matrix in Matrix Market format).
        """,
    )
    parser_count_matrix_create.add_argument(
        "-b",
        "--blacklist",
        dest="blacklist_bed_filename",
        type=str,
        action="store",
        required=False,
        help="Path to blacklist BED file.",
        default=None,
    )
    parser_count_matrix_create.add_argument(
        "-d",
        "--cb_end_remove",
        dest="cb_end_to_remove",
        required=False,
        type=str,
        action="store",
        help='Barcode suffix to remove (e.g., "-1"). Default: None.',
        default=None,
    )
    parser_count_matrix_create.add_argument(
        "-s",
        "--sample_sep",
        dest="cb_sample_separator",
        required=False,
        type=str,
        action="store",
        help='Separator to place between cell barcode and sample id (written to `out_cell_barcodes`). Default: "___".',
        default="___",
    )
    parser_count_matrix_create.add_argument(
        "-t",
        "--fragment_matrix_type",
        dest="fragment_matrix_type",
        action="store",
        type=str,
        choices=["binary", "count"],
        required=False,
        default="binary",
        help='Create "binary" or "count" fragment matrix. Default: "binary".',
    )
    parser_count_matrix_create.add_argument(
        "--bed_parser_engine",
        dest="bed_parser_engine",
        action="store",
        type=str,
        choices=["polars_lazy", "polars", "pyarrow"],
        required=False,
        default="polars_lazy",
        help="""
                BED parsing bed_parser_engine to use to read (gzipped) BED/fragment files. 
                Options: "polars_lazy" (fastest, low memory usage), "polars" (slightly slower, 
                highest memory usage) or "pyarrow" (slowest, high memory usage). Default: "polars_lazy".
                """,
    )
    parser_count_matrix_create.add_argument(
        "--intersection_engine",
        dest="intersection_engine",
        action="store",
        type=str,
        choices=["ncls", "ruranges"],
        required=False,
        default="ncls",
        help="""
                Engine to use to calculate intersections/overlaps between fragments and regions.
                Options: "ncls" or "ruranges" (faster).
                Default: "ncls".
                """,
    )
    parser_count_matrix_create.set_defaults(
        func=run_create_count_matrix_from_fragment_files
    )

    parser_count_matrix_subset = count_matrix_subparsers.add_parser(
        "subset",
        help="Subset an existing count matrix by cell barcodes and/or regions.",
        description="Subset an existing count matrix by cell barcodes and/or regions.",
    )

    parser_count_matrix_subset.add_argument(
        "-m",
        "--matrix",
        dest="binary_accessibility_matrix_filename",
        action="store",
        type=str,
        required=True,
        help="Path to fragment count matrix in Matrix Market format (`*.matrix.mtx`).",
    )
    parser_count_matrix_subset.add_argument(
        "-o",
        "--output_prefix",
        dest="output_prefix",
        action="store",
        type=str,
        required=True,
        help="""
        Output prefix for subsetted count matrix files.
        Generates: `OUTPUT_PREFIX.cell_barcodes.tsv`, `OUTPUT_PREFIX.region_ids.tsv`,
        and `OUTPUT_PREFIX.matrix.mtx`.
        """,
    )
    parser_count_matrix_subset.add_argument(
        "-c",
        "--cell_barcodes",
        dest="cell_barcodes_to_keep",
        action="store",
        type=str,
        required=False,
        help="Path to file with cell barcodes to keep (one per line). If not provided, all cell barcodes are retained.",
        default=None,
    )
    parser_count_matrix_subset.add_argument(
        "-r",
        "--regions",
        dest="region_ids_to_keep",
        action="store",
        type=str,
        required=False,
        help="Path to file with region IDs to keep (one per line). If not provided, all regions are retained.",
        default=None,
    )
    parser_count_matrix_subset.set_defaults(func=run_subset_count_matrix)
