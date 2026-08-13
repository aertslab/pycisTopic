from pathlib import Path
from typing import Literal

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
    Dictionary mapping keys to values.

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


def create_count_matrix_from_fragment_files(
    sample_to_fragment_filenames: str,
    sample_to_cell_barcodes_filenames: str,
    regions_bed_filename: str,
    output_prefix: str,
    blacklist_bed_filename: str | None = None,
    cb_end_to_remove: str | None = None,
    cb_sample_separator: str = "___",
    fragment_matrix_type: str | Literal["binary", "count"] = "binary",
    bed_parser_engine: str
    | Literal["polars_lazy", "polars", "pyarrow"] = "polars_lazy",
    intersection_engine: str | Literal["ncls", "ruranges"] = "ncls",
) -> None:
    """
    Generate fragment count matrix from multiple fragment files.

    Parameters
    ----------
    sample_to_fragment_filenames
        Path to `sample_to_fragment_filenames` TSV file, containing mapping between
        sample IDs and fragment files (tab-separated).
    sample_to_cell_barcodes_filenames
        Path to `sample_to_cell_barcodes_filenames` TSV file, containing mapping between
        sample IDs and cell barcodes to keep per sample ID (tab-separated).
    regions_bed_filename
        Path to BED file containing regions to generate count matrix on.
    output_prefix
        Output prefix for fragment count matrix files.
        Generates: ``OUTPUT_PREFIX.cell_barcodes.tsv`` (cell identifiers),
        ``OUTPUT_PREFIX.region_ids.tsv`` (genomic regions),
        and ``OUTPUT_PREFIX.matrix.mtx`` (sparse count matrix in Matrix Market format).
    blacklist_bed_filename
        Path to blacklist BED file with blacklisted regions (Amemiya et al., 2019).
        Default: ``None``.
    cb_end_to_remove
        Remove this string from the end of the cell barcode (e.g., "-1").
        Default: ``None``.
    cb_sample_separator
        Separator to place between cell barcode and sample ID in the output.
        Default: ``"___"``.
    fragment_matrix_type
        Create "binary" or "count" fragment matrix.

        Options:
          - ``binary``: Binary matrix (presence/absence of fragments).
          - ``count``: Count matrix (number of fragments per region).

        Default: ``"binary"``.
    bed_parser_engine
        BED parsing engine to use to read (gzipped) BED/fragment files.

        Options:
          - ``polars_lazy`` (fastest, low memory usage): Use Polars lazy API (``pl.scan_csv``).
          - ``polars`` (slightly slower, highest memory usage): Use Polars eager API (``pl.read_csv``).
          - ``pyarrow`` (slowest, high memory usage): Use pyarrow CSV reader (``pa.csv.read_csv``).

        Default: ``"polars_lazy"``.
    intersection_engine
        Engine to use to calculate intersections/overlaps between fragments and regions.

        Options:
          - ``ncls``: NCLS-based intersection.
          - ``ruranges`` (faster): Rust-based interval tree.

        Default: ``"ncls"``.
    """
    sample_to_fragment_file = read_mapping(
        filename=sample_to_fragment_filenames,
        key=COL_NAME_SAMPLE,
        value=COL_NAME_PATH_FRAGMENTS,
        separator=FIELD_SEP,
    )
    sample_to_barcode_file = read_mapping(
        filename=sample_to_cell_barcodes_filenames,
        key=COL_NAME_SAMPLE,
        value=COL_NAME_PATH_TO_CBS,
        separator=FIELD_SEP,
    )
    print("Fragment matrix will be generated for following samples: ")
    for sample in sample_to_fragment_file:
        print(f"\t{sample}")
        if sample not in sample_to_barcode_file:
            raise ValueError(
                f"{sample} was not present in {sample_to_cell_barcodes_filenames} aborting."
            )

    print("Reading fragments files and creating fragment matrix ...")
    fragment_matrices: list[sparse.csr_matrix] = []
    cell_names: list[str] = []
    region_ids: list[str] = []
    for sample_id, path_to_fragments in sample_to_fragment_file.items():
        print(f"\t{sample_id}\t{path_to_fragments}")
        fragment_matrix, cbs, region_ids = create_fragment_matrix_from_fragments(
            fragments_bed_filename=path_to_fragments,
            regions_bed_filename=regions_bed_filename,
            barcodes_tsv_filename=sample_to_barcode_file[sample_id],
            blacklist_bed_filename=blacklist_bed_filename,
            sample_id=sample_id,
            cb_end_to_remove=cb_end_to_remove,
            cb_sample_separator=cb_sample_separator,
            fragment_matrix_type=fragment_matrix_type,
            bed_parser_engine=bed_parser_engine,
            intersection_engine=intersection_engine,
        )
        print(f"Generated fragment matrix with shape: {fragment_matrix.shape}")
        fragment_matrices.append(fragment_matrix)
        cell_names.extend(cbs)

    print("Merging fragment matrix...")
    fragment_matrix_merged = sparse.hstack(fragment_matrices)

    region_out = f"{output_prefix}.region_ids.tsv"
    cbs_out = f"{output_prefix}.cell_barcodes.tsv"
    mat_out = f"{output_prefix}.matrix.mtx"
    print("Writing:")
    print(f'  - fragment matrix: "{mat_out}"')
    sp_io.mmwrite(mat_out, fragment_matrix_merged)

    print(f'  - region names: "{region_out}"')
    with open(cbs_out, "w") as f:
        for cb in cell_names:
            _ = f.write(f"{cb}\n")

    print(f'  - cell names: "{cbs_out}"')
    with open(region_out, "w") as f:
        for region_id in region_ids:
            _ = f.write(f"{region_id}\n")


def check_accessibility_matrix_files(binary_accessibility_matrix_filename: str) -> bool:
    """
    Check if all required accessibility matrix files exist.

    Given a matrix.mtx filename, verifies that the corresponding cell barcodes TSV
    and region IDs TSV files also exist.

    Args:
        binary_accessibility_matrix_filename: Path to the matrix.mtx file
                                             (e.g., "/path/to/PREFIX.matrix.mtx")

    Returns:
        True if all three files exist, or raise FileNotFoundError if any of the
        required files are missing.

    """
    # Extract the prefix by removing ".matrix.mtx"
    if not binary_accessibility_matrix_filename.endswith(".matrix.mtx"):
        raise ValueError(
            f'Expected filename to end with ".matrix.mtx", got: "{binary_accessibility_matrix_filename}".'
        )

    prefix = binary_accessibility_matrix_filename[: -len(".matrix.mtx")]

    required_files = [
        binary_accessibility_matrix_filename,
        f"{prefix}.cell_barcodes.tsv",
        f"{prefix}.region_ids.tsv",
    ]

    missing_files = [f for f in required_files if not Path(f).is_file()]

    if missing_files:
        raise FileNotFoundError(
            "Missing accessibility matrix files:\n"
            + "\n".join(f'  - "{f}"' for f in missing_files)
        )

    return True
