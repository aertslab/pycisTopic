import polars as pl
from scipy import io as sp_io
from scipy import sparse

from pycisTopic.fragments import create_fragment_matrix_from_fragments

COL_NAME_SAMPLE = "sample"
COL_NAME_PATH_FRAGMENTS = "path_to_fragment_file"
COL_NAME_PATH_TO_CBS = "barcode"
FIELD_SEP = "\t"


def read_mapping(filename: str, key: str, value: str, separator: str) -> dict[str, str]:
    """
    Reads file containing key value pairs.

    Parameters
    ----------
    filename:
        Path to file.
    key:
        column name of key.
    value:
        column name of value.
    separator:
        field separator used for file.

    Returns
    -------
    Dictionary mapping keys to values

    """
    # check headers of definition files, by reading the first line
    with open(filename) as f:
        header = f.readline().strip().split(separator)
        if not (key in header and value in header):
            raise KeyError(f"{filename} must have columns {key} and {value}")
    # read sample to fragment file
    key_value: dict[str, str] = {}
    d_key_value = pl.read_csv(
        filename,
        separator=separator,
    ).to_dict()
    for k, v in zip(
        d_key_value[key],
        d_key_value[value],
    ):
        if k in key_value:
            raise ValueError(f"Duplicate {k} in {filename}")
        key_value[k] = v
    return key_value


def run_create_count_matrix(args):
    sample_to_fragment_file = read_mapping(
        filename=args.sample_to_fragment,
        key=COL_NAME_SAMPLE,
        value=COL_NAME_PATH_FRAGMENTS,
        separator=FIELD_SEP,
    )
    sample_to_barcode_file = read_mapping(
        filename=args.sample_to_cell_barcodes,
        key=COL_NAME_SAMPLE,
        value=COL_NAME_PATH_TO_CBS,
        separator=FIELD_SEP,
    )
    print("Fragment matrix will be generated for following samples: ")
    for sample in sample_to_fragment_file:
        print(f"\t{sample}")
        if sample not in sample_to_barcode_file:
            raise ValueError(
                f"{sample} was not present in {args.sample_to_cell_barcodes} aborting."
            )

    print("Reading fragments files and creating count matrix")
    fragment_matrices: list[sparse.csr_matrix] = []
    cell_names: list[str] = []
    region_names: list[str] = []
    for sample_id, path_to_fragments in sample_to_fragment_file.items():
        print(f"\t{sample_id}\t{path_to_fragments}")
        count_matrix, cbs, region_ids = create_fragment_matrix_from_fragments(
            fragments_bed_filename=path_to_fragments,
            regions_bed_filename=args.region_ids_filename,
            barcodes_tsv_filename=sample_to_barcode_file[sample_id],
            blacklist_bed_filename=args.blacklist,
            sample_id=sample_id,
            cb_end_to_remove=args.cb_end_to_remove,
            cb_sample_separator=args.cb_sample_separator,
        )
        print(f"Generated matrix with shape: {count_matrix.shape}")
        fragment_matrices.append(count_matrix)
        cell_names.extend(cbs)
        region_names = region_ids

    print("Merging fragment matrix and binarizing")
    binary_matrix_merged = sparse.hstack(
        fragment_matrices
    )

    binary_matrix_merged.data.fill(1)

    region_out = args.out_region_names
    cbs_out = args.out_cell_barcodes
    mat_out = args.out_matrix
    print("Writing:")
    print(f"\tBinary matrix: {mat_out}")
    print(f"\tregion names: {region_out}")
    print(f"\tcell names: {cbs_out}")
    sp_io.mmwrite(mat_out, binary_matrix_merged)
    with open(cbs_out, "w") as f:
        for cb in cell_names:
            _ = f.write(f"{cb}\n")
    with open(region_out, "w") as f:
        for region in region_names:
            _ = f.write(f"{region}\n")


def add_parser_count_matrix(subparsers):
    """Creates an ArgumentParser to read the options for this script."""
    parser_count_matrix = subparsers.add_parser(
        "count_matrix",
        help="Generate binary fragment count matrix for multiple samples.",
        description="Generate binary fragment count matrix for multiple samples.",
    )

    parser_count_matrix.add_argument(
        "-f",
        "--sample_fragment",
        dest="sample_to_fragment",
        action="store",
        type=str,
        required=True,
        help="Path to sample_to_fragment tsv file, containing mapping between sample ids and fragment files (tab-separated).",
    )
    parser_count_matrix.add_argument(
        "-c",
        "--sample_barcodes",
        dest="sample_to_cell_barcodes",
        action="store",
        type=str,
        required=True,
        help="Path to sample_to_cell_barcode tsv file, containing mapping between sample ids and cell barcodes to keep per sample id (tab-separated).",
    )
    parser_count_matrix.add_argument(
        "-r",
        "--regions",
        dest="region_ids_filename",
        action="store",
        type=str,
        required=True,
        help="Path to bed file containing regions to generate count matrix on.",
    )
    parser_count_matrix.add_argument(
        "--out_region_names",
        dest="out_region_names",
        action="store",
        type=str,
        required=True,
        help="Path to output region names .txt file.",
    )
    parser_count_matrix.add_argument(
        "--out_cell_barcodes",
        dest="out_cell_barcodes",
        action="store",
        type=str,
        required=True,
        help="Path to output cell barcodes names .txt file.",
    )
    parser_count_matrix.add_argument(
        "--out_matrix",
        dest="out_matrix",
        action="store",
        type=str,
        required=True,
        help="Path to output matrix .mtx file.",
    )
    parser_count_matrix.add_argument(
        "-b",
        "--blacklist",
        dest="blacklist",
        type=str,
        action="store",
        required=False,
        help="Path to blacklist bed file.",
        default=None,
    )
    parser_count_matrix.add_argument(
        "-d",
        "--cb_end_remove",
        dest="cb_end_to_remove",
        required=False,
        type=str,
        action="store",
        help="Barcode suffix to remove (e.g., -1)",
        default=None,
    )
    parser_count_matrix.add_argument(
        "-s",
        "--sample_sep",
        dest="cb_sample_separator",
        required=False,
        type=str,
        action="store",
        help="Separator to place between cell barcode and sample id.",
        default="___",
    )
    parser_count_matrix.set_defaults(
        func=run_create_count_matrix,
    )
