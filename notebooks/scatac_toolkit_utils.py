# scatac_toolkit_utils.py
import argparse
from pathlib import Path
import pandas as pd

def map_sample_to_fragments(fragment_dir, output_tsv, suffix):
    """
    Generate the sample_to_fragments.tsv file (matching the sample names to the location of their fragment files)
    To be used in scatac_fragment_tools split command.
    """
    fragment_dir = Path(fragment_dir).resolve()
    output_tsv = Path(output_tsv).resolve()

    files = sorted(fragment_dir.glob(f"*{suffix}"))
    if not files:
        raise RuntimeError(f"No files matching '*{suffix}' in {fragment_dir}")

    pd.DataFrame(
        [(f.name.replace(suffix, ""), str(f)) for f in files],
        columns=["sample", "path_to_fragment_file"],
    ).to_csv(output_tsv, sep="\t", index=False)

    return output_tsv


from pathlib import Path
import pandas as pd

def map_cell_names_to_all_cells(cell_names_tsv, output_tsv):
    """
    Generate the all_cells_with_annotation.tsv file (matching the cells (identified by sample and barcode) to their consensus_cell_type annotaion)
    To be used in scatac_fragment_tools split command.
    """
    cell_names_tsv = Path(cell_names_tsv).resolve()
    output_tsv = Path(output_tsv).resolve()

    df = pd.read_csv(cell_names_tsv, sep=None, engine="python")

    required = {"barcode", "sample", "consensus_cell_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns {sorted(missing)} in {cell_names_tsv}. "
            f"Found columns: {list(df.columns)}"
        )

    # Drop empty / NA cell types
    ct = df["consensus_cell_type"].astype("string")
    df = df[ct.notna() & (ct.str.strip() != "")].copy()

    # Optional: drop empty / NA samples
    sm = df["sample"].astype("string")
    df = df[sm.notna() & (sm.str.strip() != "")].copy()

    out = df.rename(
        columns={"consensus_cell_type": "cell_type", "barcode": "cell_barcode"}
    )[["sample", "cell_type", "cell_barcode"]]

    out.to_csv(output_tsv, sep="\t", index=False)
    return output_tsv


from pathlib import Path
import pandas as pd


def filter_sample_to_fragments_by_annotations(
    sample_to_fragments_tsv,
    all_cells_with_annotation_tsv,
    output_tsv,
):
    """
    Filter sample_to_fragments.tsv so that it only contains samples
    that are present in all_cells_with_annotation.tsv.

    This is optional if you provided more samples than you have annotations for in the beginning. 
    """
    sample_to_fragments_tsv = Path(sample_to_fragments_tsv).resolve()
    all_cells_with_annotation_tsv = Path(all_cells_with_annotation_tsv).resolve()
    output_tsv = Path(output_tsv).resolve()

    frag = pd.read_csv(sample_to_fragments_tsv, sep="\t")
    anno = pd.read_csv(all_cells_with_annotation_tsv, sep="\t")

    if "sample" not in frag.columns:
        raise ValueError(f"'sample' column missing in {sample_to_fragments_tsv}")
    if "sample" not in anno.columns:
        raise ValueError(f"'sample' column missing in {all_cells_with_annotation_tsv}")

    anno_samples = set(anno["sample"].unique())

    frag_filt = frag[frag["sample"].isin(anno_samples)].copy()

    frag_filt.to_csv(output_tsv, sep="\t", index=False)

    print(f"Wrote: {output_tsv}")
    print(
        "Samples in filtered fragments:",
        sorted(frag_filt["sample"].unique())[:20],
    )
    print(
        "Samples in annotations:",
        sorted(anno_samples)[:20],
    )

    return output_tsv


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="scATAC toolkit utils")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_frag = sub.add_parser("map_sample_to_fragments")
    p_frag.add_argument("fragment_dir")
    p_frag.add_argument("output_tsv")
    p_frag.add_argument("--suffix", required=True)

    p_cells = sub.add_parser("map_cell_names_to_all_cells")
    p_cells.add_argument("cell_names_file")
    p_cells.add_argument("output_tsv")

    p_filter = sub.add_parser(
        "filter_sample_to_fragments",
        help="Filter sample_to_fragments.tsv to samples present in cell annotations",
    )
    p_filter.add_argument("sample_to_fragments")
    p_filter.add_argument("all_cells_with_annotation")
    p_filter.add_argument("output_tsv")

    args = parser.parse_args(argv)

    if args.cmd == "map_sample_to_fragments":
        map_sample_to_fragments(
            args.fragment_dir,
            args.output_tsv,
            args.suffix,
        )

    elif args.cmd == "map_cell_names_to_all_cells":
        map_cell_names_to_all_cells(
            args.cell_names_file,
            args.output_tsv,
        )

    elif args.cmd == "filter_sample_to_fragments":
        filter_sample_to_fragments_by_annotations(
            args.sample_to_fragments,
            args.all_cells_with_annotation,
            args.output_tsv,
        )