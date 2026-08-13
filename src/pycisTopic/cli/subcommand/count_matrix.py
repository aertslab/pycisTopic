import polars as pl
from scipy import io as sp_io
from scipy import sparse

from pycisTopic.count_matrix import create_count_matrix_from_fragment_files


def run_create_count_matrix_from_fragment_files(args):
    create_count_matrix_from_fragment_files(
        sample_to_fragment_filenames=args.sample_to_fragment_filenames,
        sample_to_cell_barcodes_filenames=args.sample_to_cell_barcodes_filenames,
        region_ids_filename=args.region_ids_filename,
        output_prefix=args.output_prefix,
        blacklist=args.blacklist,
        cb_end_to_remove=args.cb_end_to_remove,
        cb_sample_separator=args.cb_sample_separator,
        fragment_matrix_type=args.fragment_matrix_type,
        bed_parser_engine=args.bed_parser_engine,
        intersection_engine=args.intersection_engine,
    )


def add_parser_count_matrix(subparsers):
    """Creates an ArgumentParser to read the options for this script."""
    parser_count_matrix = subparsers.add_parser(
        "count_matrix",
        help="Generate (binary) fragment count matrix for multiple samples.",
        description="Generate (binary) fragment count matrix for multiple samples.",
    )

    parser_count_matrix.add_argument(
        "-f",
        "--sample_fragment",
        dest="sample_to_fragment_filenames",
        action="store",
        type=str,
        required=True,
        help="Path to `sample_to_fragment_files` TSV file, containing mapping between sample "
        "ids and fragment files (tab-separated).",
    )
    parser_count_matrix.add_argument(
        "-c",
        "--sample_barcodes",
        dest="sample_to_cell_barcodes_filenames",
        action="store",
        type=str,
        required=True,
        help="Path to `sample_to_cell_barcode_filenames` TSV file, containing mapping between "
        "sample ids and cell barcode filenames to keep per sample id (tab-separated).",
    )
    parser_count_matrix.add_argument(
        "-r",
        "--regions",
        dest="region_ids_filename",
        action="store",
        type=str,
        required=True,
        help="Path to BED file containing regions to generate count matrix on.",
    )
    parser_count_matrix.add_argument(
        "-o",
        "--output_prefix",
        dest="output_prefix",
        action="store",
        type=str,
        required=True,
        help="""
        Output prefix for (binary) fragment count matrix file.
        Generates: `OUTPUT_PREFIX.cell_barcodes.tsv` (cell identifiers), `OUTPUT_PREFIX.region_names.tsv` (genomic regions),
        and `OUTPUT_PREFIX.matrix.mtx` (sparse count matrix in Matrix Market format).
        """,
    )
    parser_count_matrix.add_argument(
        "-b",
        "--blacklist",
        dest="blacklist",
        type=str,
        action="store",
        required=False,
        help="Path to blacklist BED file.",
        default=None,
    )
    parser_count_matrix.add_argument(
        "-d",
        "--cb_end_remove",
        dest="cb_end_to_remove",
        required=False,
        type=str,
        action="store",
        help='Barcode suffix to remove (e.g., "-1"). Default: None.',
        default=None,
    )
    parser_count_matrix.add_argument(
        "-s",
        "--sample_sep",
        dest="cb_sample_separator",
        required=False,
        type=str,
        action="store",
        help='Separator to place between cell barcode and sample id (written to `out_cell_barcodes`). Default: "___".',
        default="___",
    )
    parser_count_matrix.add_argument(
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
    parser_count_matrix.add_argument(
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
    parser_count_matrix.add_argument(
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
    parser_count_matrix.set_defaults(
        func=run_create_count_matrix_from_fragment_files,
    )
