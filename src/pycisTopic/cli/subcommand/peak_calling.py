import os
import sys
from typing import TYPE_CHECKING, Literal, Sequence
from joblib import Parallel, delayed
import subprocess

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction
    from pathlib import Path

SUFFIX = ".fragments.tsv.gz"

def run_macs_callback(args):
    run_macs3(
        args.macs3_exe,
        args.path_to_pseudobulk_fragments,
        args.outdir,
        args.genome_size,
        args.q_value_threshold,
        args.shift,
        args.extsize,
        args.n_cores
        )

def run_macs3(
    macs3_exe: str,
    path_to_pseudobulk_fragments: str,
    outdir: str,
    genome_size: int,
    q_value_threshold: float,
    shift: int,
    extsize: int,
    n_cores: int
):
    """
    Run macs3.

    Parameter
    ---------
    macs3_exe: str
        path to macs3 executable
    path_to_pseudobulk_fragments: str
        path to directory containing pseudobulk fragment files
    genome_size: int
        --gsize parameter for macs3
    q_value_threshold: float
        --qvalue parameter for macs3
    shift: int
        --shift parameter for macs3
    extsize: int
        --extsize parameter for macs3
    n_cores: int
        Number of cores to use for parallel.
    """
    pseudobulk_fragment_files: dict[str, str] = {}
    for file in os.listdir(path_to_pseudobulk_fragments):
        if not file.endswith(SUFFIX):
            print(f"skipping: {file} it does not end with {SUFFIX}.")
            continue
        cell_type = file.replace(SUFFIX, "")
        pseudobulk_fragment_files[cell_type] = f"{path_to_pseudobulk_fragments}/{file}"

    Parallel(n_jobs=n_cores, prefer="threads")(delayed(subprocess.run)([
        macs3_exe,
        'callpeak',
        '--treatment', frag_file,
        '--name', cell_type,
        '--format', 'FRAG',
        '--outdir', outdir,
        '-g', str(genome_size),
        '-q', str(q_value_threshold),
        '--shift', str(shift),
        '--extsize', str(extsize),
        '--max-count','1',
        '--nolambda',
        '--nomodel',
        '--call-summits'       
        ]) for cell_type, frag_file in pseudobulk_fragment_files.items())


def add_parser_peak_calling(subparsers):
    """Peak calling with macs3"""
    parser_peak_calling = subparsers.add_parser(
        "callpeaks",
        help="Call peaks on .bed files using macs3"
    )

    parser_peak_calling.add_argument(
        "-e",
        "--exe",
        dest="macs3_exe",
        type=str,
        action="store",
        required=True,
        help="macs3 executable",
    )

    parser_peak_calling.add_argument(
        "--ncores",
        dest="n_cores",
        type=str,
        action="store",
        required=True,
        help="macs3 executable",
    )

    parser_peak_calling.add_argument(
        "--bulk_path",
        dest="path_to_pseudobulk_fragments",
        type=str,
        action="store",
        required=True,
        help="Path to directory containing fragment files",
    )

    # parser_peak_calling.add_argument(
    #     "-n",
    #     "--name",
    #     dest="name",
    #     action="store",
    #     type=str,
    #     required=True,
    #     help=""
    # )

    parser_peak_calling.add_argument(
        "--outdir",
        dest="outdir",
        action="store",
        type=str,
        required=True,
        help="Save output files into specified folder, created anew if necessary"
    )

    # parser_peak_calling.add_argument(
    #     "--format",
    #     dest="format",
    #     choices=[
    #         "ELAND",
    #         "BED",
    #         "ELANDMULTI",
    #         "ELANDEXPORT",
    #         "SAM",
    #         "BAM",
    #         "BOWTIE",
    #         "BAMPE",
    #         "BEDPE",
    #         "FRAG",
    #     ],
    #     action="store",
    #     type=str,
    #     help="Fornat of tag file; ELAND, BED, ELANDMULTI, ELANDEXPORT, SAM, BAM, BOWTIE, BAMPE, BEDPE, or FRAG. Default FRAG",
    #     default='FRAG'
    # )

    parser_peak_calling.add_argument(
        "-g",
        "--gsize",
        dest="genome_size",
        action="store",
        type=int,
        help="Mappable genome size for this species",
        default=1368780147
    )

    # parser_peak_calling.add_argument(
    #     "--max-count",
    #     dest="max_count",
    #     action="store",
    #     type=int,
    #     help="value 1 prevents counting fragments twice with --format FRAG, default 1",
    #     default=1
    # )

    parser_peak_calling.add_argument(
        "-q",
        "--qvalue",
        dest="q_value_threshold",
        action="store",
        type=float,
        help="The q-value (minimum FDR) cutoff to call significant regions. Default is 0.05.",
        default=0.05
    )

    # parser_peak_calling.add_argument(
    #     "--nomodel",
    #     dest="nomodel",
    #     action="store",
    #     type=bool,
    #     help="While on, MACS3 will bypass building the shifting model. Also use --extsize and --shift",
    #     default=True
    # )

    parser_peak_calling.add_argument(
        "--shift",
        dest="shift",
        action="store",
        type=int,
        help="Shift in bp here to adjust the alignment positions of reads.",
        default=73
    )

    parser_peak_calling.add_argument(
        "--extsize",
        dest="extsize",
        action="store",
        type=int,
        help="extend reads in 5->3 direction to fix-sized fragments.",
        default=146
    )

    # parser_peak_calling.add_argument(
    #     "--nolambda",
    #     dest="nolambda",
    #     action="store",
    #     type=bool,
    #     help="With this flag on, MACS3 will use the background lambda as local lambda.",
    #     default=True
    # )

    # parser_peak_calling.add_argument(
    #     "--call-summits",
    #     dest="summits",
    #     action="store",
    #     type=bool,
    #     help="Ddeconvolve subpeaks within each peak called.",
    #     default=True
    # )

    parser_peak_calling.set_defaults(
        func=run_macs_callback
    )