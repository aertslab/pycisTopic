from __future__ import annotations

import logging
import os
import subprocess
import sys

import joblib
import pandas as pd
import pyranges as pr
import ray
from pycisTopic.cistopic_class import CistopicObject
from scatac_fragment_tools.library.bigwig.fragments_to_bigwig import (
    fragments_to_bw,
    read_fragments_to_polars_df,
)
from scatac_fragment_tools.library.split.split_fragments_by_cell_type import (
    _santize_string_for_filename,
    split_fragment_files_by_cell_type,
)

# FIXME
from .utils import *


def _generate_bigwig(
        path_to_fragments: str,
        chromsizes: dict[str, int],
        normalize_bigwig: bool,
        bw_filename: str,
        log: logging.Logger):
    fragments_df = read_fragments_to_polars_df(path_to_fragments)
    fragments_to_bw(
        fragments_df = fragments_df,
        chrom_sizes = chromsizes,
        bw_filename = bw_filename,
        normalize = normalize_bigwig,
        scaling_factor = 1,
        cut_sites = False
    )
    log.info(f"{bw_filename} done!")

def export_pseudobulk(
    input_data: Union[CistopicObject, pd.DataFrame],
    variable: str,
    chromsizes: Union[pd.DataFrame, pr.PyRanges],
    bed_path: str,
    bigwig_path: str,
    path_to_fragments: dict[str, str] | None = None,
    sample_id_col: str = "sample_id",
    n_cpu: int = 1,
    normalize_bigwig: bool = True,
    split_pattern: str = "___",
    temp_dir: str = "/tmp"
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Create pseudobulks as bed and bigwig from single cell fragments file given a barcode annotation.

    Parameters
    ---------
    input_data: CistopicObject or pd.DataFrame
            A :class:`CistopicObject` containing the specified `variable` as a column in :class:`CistopicObject.cell_data` or a cell metadata
            :class:`pd.DataFrame` containing barcode as rows, containing the specified `variable` as a column (additional columns are
            possible) and a `sample_id` column. Index names must contain the BARCODE (e.g. ATGTCGTC-1), additional tags are possible separating with -
            (e.g. ATGCTGTGCG-1-Sample_1). The levels in the sample_id column must agree with the keys in the path_to_fragments dictionary.
            Alternatively, if the cell metadata contains a column named barcode it will be used instead of the index names.
    variable: str
            A character string indicating the column that will be used to create the different group pseudobulk. It must be included in
            the cell metadata provided as input_data.
    chromsizes: pd.DataFrame or pr.PyRanges
            A data frame or :class:`pr.PyRanges` containing size of each chromosome, containing 'Chromosome', 'Start' and 'End' columns.
    bed_path: str
            Path to folder where the fragments bed files per group will be saved. If None, files will not be generated.
    bigwig_path: str
            Path to folder where the bigwig files per group will be saved. If None, files will not be generated.
    path_to_fragments: str or dict, optional
            A dictionary of character strings, with sample name as names indicating the path to the fragments file/s from which pseudobulk profiles have to
            be created. If a :class:`CistopicObject` is provided as input it will be ignored, but if a cell metadata :class:`pd.DataFrame` is provided it
            is necessary to provide it. The keys of the dictionary need to match with the sample_id tag added to the index names of the input data frame.
    sample_id_col: str, optional
            Name of the column containing the sample name per barcode in the input :class:`CistopicObject.cell_data` or class:`pd.DataFrame`. Default: 'sample_id'.
    n_cpu: int, optional
            Number of cores to use. Default: 1.
    normalize_bigwig: bool, optional
            Whether bigwig files should be CPM normalized. Default: True.

    split_pattern: str, optional
            Pattern to split cell barcode from sample id. Default: '___'. Note, if `split_pattern` is not None, then `export_pseudobulk` will
            attempt to infer `sample_id` from the index of `input_data` and ignore `sample_id_col`.
    temp_dir: str
            Path to temporary directory. Default: '/tmp'.

    Return
    ------
    dict
            A dictionary containing the paths to the newly created bed fragments files per group a dictionary containing the paths to the
            newly created bigwig files per group.
    """
    # Create logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    # Get fragments file
    if isinstance(input_data, CistopicObject):
        path_to_fragments = input_data.path_to_fragments
        if path_to_fragments is None:
            log.error("No path_to_fragments in this cisTopic object.")

        cell_data = input_data.cell_data.copy()
    elif isinstance(input_data, pd.DataFrame):
        if path_to_fragments is None:
            log.error("Please, provide path_to_fragments.")
        cell_data = input_data.copy()
    # Check for sample_id column
    try:
        sample_ids = list(set(cell_data[sample_id_col]))
    except ValueError:
        print(
            'Please, include a sample identification column (e.g. "sample_id") in your cell metadata!'
        )

    # Check wether we have a path to fragments for each sample
    if not all([sample_id in path_to_fragments.keys() for sample_id in sample_ids]):
        raise ValueError("Please, provide a path to fragments for each sample in your cell metadata!")
    # Check for NaNs in variable column
    if cell_data[variable].isna().any():
        log.warning(
            f"NaNs detected in {variable} column. These will be converted to 'nan' string.")
    # Check for numerical values in variable column
    if not all([isinstance(x, str) for x in cell_data[variable].dropna()]):
        log.warning(
            f"Non-string values detected in {variable} column. These will be converted to strings.")
    # Convert variable column to string
    cell_data[variable] = cell_data[variable].astype(str)
    # make output folders, if they don't exists
    if not os.path.exists(bed_path):
        os.makedirs(bed_path)
    if not os.path.exists(bigwig_path):
        os.makedirs(bigwig_path)
    if "barcode" not in cell_data.columns:
        cell_data["barcode"] = prepare_tag_cells(
            cell_data.index.tolist(), split_pattern
        )
    sample_to_cell_type_to_barcodes = {}
    for sample in sample_ids:
        _sample_cell_data = cell_data.loc[cell_data[sample_id_col] == sample]
        _cell_type_to_cell_barcodes = _sample_cell_data \
            .groupby(variable, group_keys=False)["barcode"] \
            .apply(list) \
            .to_dict()
        sample_to_cell_type_to_barcodes[sample] = _cell_type_to_cell_barcodes
    if isinstance(chromsizes, pr.PyRanges):
        chromsizes_dict = chromsizes.df.set_index("Chromosome").to_dict()["End"]
    else:
        chromsizes_dict = chromsizes.set_index("Chromosome").to_dict()["End"]
    # For each sample, get fragments for each cell type

    log.info("Splitting fragments by cell type.")
    split_fragment_files_by_cell_type(
        sample_to_fragment_file = path_to_fragments,
        path_to_temp_folder = temp_dir,
        path_to_output_folder = bed_path,
        sample_to_cell_type_to_cell_barcodes = sample_to_cell_type_to_barcodes,
        chromsizes = chromsizes_dict,
        n_cpu = n_cpu,
        verbose = False,
        clear_temp_folder = True
    )

    bed_paths = {}
    for cell_type in cell_data[variable].unique():
        _bed_fname = os.path.join(
            bed_path,
            f"{_santize_string_for_filename(cell_type)}.fragments.tsv.gz")
        if os.path.exists(_bed_fname):
            bed_paths[cell_type] = _bed_fname
        else:
            log.warning(f"Missing fragments for {cell_type}!")

    log.info("generating bigwig files")
    joblib.Parallel(n_jobs=n_cpu)(
        joblib.delayed(_generate_bigwig)
        (
            path_to_fragments = bed_paths[cell_type],
            chromsizes = chromsizes_dict,
            normalize_bigwig = normalize_bigwig,
            bw_filename = os.path.join(bigwig_path, f"{_santize_string_for_filename(cell_type)}.bw"),
            log = log
        )
        for cell_type in bed_paths.keys()
    )
    bw_paths = {}
    for cell_type in cell_data[variable].unique():
        _bw_fname = os.path.join(
            bigwig_path,
            f"{_santize_string_for_filename(cell_type)}.bw")
        if os.path.exists(_bw_fname):
            bw_paths[cell_type] = _bw_fname
        else:
            log.warning(f"Missing bigwig for {cell_type}!")

    return bw_paths, bed_paths

def peak_calling(
    macs_path: str,
    bed_paths: dict,
    outdir: str,
    genome_size: str,
    n_cpu: int = 1,
    input_format: str = "BEDPE",
    shift: int = 73,
    ext_size: int = 146,
    keep_dup: str = "all",
    q_value: float = 0.05,
    nolambda: bool = True,
    skip_empty_peaks: bool = False,
    **kwargs
):
    """
    Performs pseudobulk peak calling with MACS2. It requires to have MACS2 installed (https://github.com/macs3-project/MACS).

    Parameters
    ---------
    macs_path: str
            Path to MACS binary (e.g. /xxx/MACS/xxx/bin/macs2).
    bed_paths: dict
            A dictionary containing group label as name and the path to their corresponding fragments bed file as value.
    outdir: str
            Path to the output directory.
    genome_size: str
            Effective genome size which is defined as the genome size which can be sequenced. Possible values: 'hs', 'mm', 'ce' and 'dm'.
    n_cpu: int, optional
            Number of cores to use. Default: 1.
    input_format: str, optional
            Format of tag file can be ELAND, BED, ELANDMULTI, ELANDEXPORT, SAM, BAM, BOWTIE, BAMPE, or BEDPE. Default is AUTO which will
            allow MACS to decide the format automatically. Default: 'BEDPE'.
    shift: int, optional
            To set an arbitrary shift in bp. For finding enriched cutting sites (such as in ATAC-seq) a shift of 73 bp is recommended.
            Default: 73.
    ext_size: int, optional
            To extend reads in 5'->3' direction to fix-sized fragment. For ATAC-seq data, a extension of 146 bp is recommended.
            Default: 146.
    keep_dup: str, optional
            Whether to keep duplicate tags at te exact same location. Default: 'all'.
    q_value: float, optional
            The q-value (minimum FDR) cutoff to call significant regions. Default: 0.05.
    **kwargs
            Additional parameters to pass to ray.init().

    Return
    ------
    dict
            A dictionary containing each group label as names and :class:`pr.PyRanges` with MACS2 narrow peaks as values.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if n_cpu > 1:
        ray.init(num_cpus=n_cpu, **kwargs)
        try:
            narrow_peaks = ray.get(
                [
                    macs_call_peak_ray.remote(
                        macs_path,
                        bed_paths[name],
                        name,
                        outdir,
                        genome_size,
                        input_format,
                        shift,
                        ext_size,
                        keep_dup,
                        q_value,
                        nolambda,
                        skip_empty_peaks

                    )
                    for name in list(bed_paths.keys())
                ]
            )
        except Exception as e:
            ray.shutdown()
            raise(e)
        ray.shutdown()
    else:
        narrow_peaks = [macs_call_peak(
                    macs_path,
                    bed_paths[name],
                    name,
                    outdir,
                    genome_size,
                    input_format,
                    shift,
                    ext_size,
                    keep_dup,
                    q_value,
                    nolambda,
                    skip_empty_peaks

                )
                for name in list(bed_paths.keys())
            ]
    narrow_peaks_dict = {
        list(bed_paths.keys())[i]: narrow_peaks[i].narrow_peak
        for i in range(len(narrow_peaks))
        if len(narrow_peaks[i].narrow_peak) > 0
    }
    return narrow_peaks_dict

def macs_call_peak(
    macs_path: str,
    bed_path: str,
    name: str,
    outdir: str,
    genome_size: str,
    input_format: str = "BEDPE",
    shift: int = 73,
    ext_size: int = 146,
    keep_dup: str = "all",
    q_value: int = 0.05,
    nolambda: bool = True,
    skip_empty_peaks: bool = False
):
    """
    Performs pseudobulk peak calling with MACS2 in a group. It requires to have MACS2 installed (https://github.com/macs3-project/MACS).

    Parameters
    ---------
    macs_path: str
            Path to MACS binary (e.g. /xxx/MACS/xxx/bin/macs2).
    bed_path: str
            Path to fragments file bed file.
    name: str
            Name of string of the group.
    outdir: str
            Path to the output directory.
    genome_size: str
            Effective genome size which is defined as the genome size which can be sequenced. Possible values: 'hs', 'mm', 'ce' and 'dm'.
    input_format: str, optional
            Format of tag file can be ELAND, BED, ELANDMULTI, ELANDEXPORT, SAM, BAM, BOWTIE, BAMPE, or BEDPE. Default is AUTO which will
            allow MACS to decide the format automatically. Default: 'BEDPE'.
    shift: int, optional
            To set an arbitrary shift in bp. For finding enriched cutting sites (such as in ATAC-seq) a shift of 73 bp is recommended.
            Default: 73.
    ext_size: int, optional
            To extend reads in 5'->3' direction to fix-sized fragment. For ATAC-seq data, a extension of 146 bp is recommended.
            Default: 146.
    keep_dup: str, optional
            Whether to keep duplicate tags at te exact same location. Default: 'all'.
    q_value: float, optional
            The q-value (minimum FDR) cutoff to call significant regions. Default: 0.05.
    nolambda: bool, optional
            Do not consider the local bias/lambda at peak candidate regions.

    Return
    ------
    dict
            A :class:`pr.PyRanges` with MACS2 narrow peaks as values.
    """
    # Create logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    MACS_peak_calling = MACSCallPeak(
        macs_path,
        bed_path,
        name,
        outdir,
        genome_size,
        input_format=input_format,
        shift=shift,
        ext_size=ext_size,
        keep_dup=keep_dup,
        q_value=q_value,
        nolambda=nolambda,
        skip_empty_peaks=skip_empty_peaks
    )
    log.info(f"{name} done!")
    return MACS_peak_calling

@ray.remote
def macs_call_peak_ray(
    macs_path: str,
    bed_path: str,
    name: str,
    outdir: str,
    genome_size: str,
    input_format: str = "BEDPE",
    shift: int = 73,
    ext_size: int = 146,
    keep_dup: str = "all",
    q_value: int = 0.05,
    nolambda: bool = True,
    skip_empty_peaks: bool = False
):
    """
    Performs pseudobulk peak calling with MACS2 in a group. It requires to have MACS2 installed (https://github.com/macs3-project/MACS).

    Parameters
    ---------
    macs_path: str
            Path to MACS binary (e.g. /xxx/MACS/xxx/bin/macs2).
    bed_path: str
            Path to fragments file bed file.
    name: str
            Name of string of the group.
    outdir: str
            Path to the output directory.
    genome_size: str
            Effective genome size which is defined as the genome size which can be sequenced. Possible values: 'hs', 'mm', 'ce' and 'dm'.
    input_format: str, optional
            Format of tag file can be ELAND, BED, ELANDMULTI, ELANDEXPORT, SAM, BAM, BOWTIE, BAMPE, or BEDPE. Default is AUTO which will
            allow MACS to decide the format automatically. Default: 'BEDPE'.
    shift: int, optional
            To set an arbitrary shift in bp. For finding enriched cutting sites (such as in ATAC-seq) a shift of 73 bp is recommended.
            Default: 73.
    ext_size: int, optional
            To extend reads in 5'->3' direction to fix-sized fragment. For ATAC-seq data, a extension of 146 bp is recommended.
            Default: 146.
    keep_dup: str, optional
            Whether to keep duplicate tags at te exact same location. Default: 'all'.
    q_value: float, optional
            The q-value (minimum FDR) cutoff to call significant regions. Default: 0.05.
    nolambda: bool, optional
            Do not consider the local bias/lambda at peak candidate regions.

    Return
    ------
    dict
            A :class:`pr.PyRanges` with MACS2 narrow peaks as values.
    """
    # Create logger
    level = logging.INFO
    log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger("cisTopic")

    MACS_peak_calling = MACSCallPeak(
        macs_path,
        bed_path,
        name,
        outdir,
        genome_size,
        input_format=input_format,
        shift=shift,
        ext_size=ext_size,
        keep_dup=keep_dup,
        q_value=q_value,
        nolambda=nolambda,
        skip_empty_peaks=skip_empty_peaks

    )
    log.info(name + " done!")
    return MACS_peak_calling


class MACSCallPeak:
    """
    Parameters
    ---------
    macs_path: str
            Path to MACS binary (e.g. /xxx/MACS/xxx/bin/macs2).
    bed_path: str
            Path to fragments file bed file.
    name: str
            Name of string of the group.
    outdir: str
            Path to the output directory.
    genome_size: str
            Effective genome size which is defined as the genome size which can be sequenced. Possible values: 'hs', 'mm', 'ce' and 'dm'.
    input_format: str, optional
            Format of tag file can be ELAND, BED, ELANDMULTI, ELANDEXPORT, SAM, BAM, BOWTIE, BAMPE, or BEDPE. Default is AUTO which will
            allow MACS to decide the format automatically. Default: 'BEDPE'.
    shift: int, optional
            To set an arbitrary shift in bp. For finding enriched cutting sites (such as in ATAC-seq) a shift of 73 bp is recommended.
            Default: 73.
    ext_size: int, optional
            To extend reads in 5'->3' direction to fix-sized fragment. For ATAC-seq data, a extension of 146 bp is recommended.
            Default: 146.
    keep_dup: str, optional
            Whether to keep duplicate tags at te exact same location. Default: 'all'.
    q_value: float, optional
            The q-value (minimum FDR) cutoff to call significant regions. Default: 0.05.
    nolambda: bool, optional
            Do not consider the local bias/lambda at peak candidate regions.
    """

    def __init__(
        self,
        macs_path: str,
        bed_path: str,
        name: str,
        outdir: str,
        genome_size: str,
        input_format: str = "BEDPE",
        shift: int = 73,
        ext_size: int = 146,
        keep_dup: str = "all",
        q_value: int = 0.05,
        nolambda: bool = True,
        skip_empty_peaks: bool = False,
    ):
        self.macs_path = macs_path
        self.treatment = bed_path
        self.name = str(name)
        self.outdir = outdir
        self.input_format = input_format
        self.gsize = genome_size
        self.shift = shift
        self.ext_size = ext_size
        self.keep_dup = keep_dup
        self.qvalue = q_value
        self.nolambda = nolambda
        self.skip_empty_peaks = skip_empty_peaks

        self.call_peak()

    def call_peak(self):
        """
        Run MACS2 peak calling.
        """
        # Create logger
        level = logging.INFO
        log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level=level, format=log_format, handlers=handlers)
        log = logging.getLogger("cisTopic")

        if self.nolambda is True:
            cmd = (
                self.macs_path
                + " callpeak --treatment %s --name %s  --outdir %s --format %s --gsize %s "
                "--qvalue %s --nomodel --shift %s --extsize %s --keep-dup %s --call-summits --nolambda"
            )
        else:
            cmd = (
                self.macs_path
                + " callpeak --treatment %s --name %s  --outdir %s --format %s --gsize %s "
                "--qvalue %s --nomodel --shift %s --extsize %s --keep-dup %s --call-summits"
            )

        cmd = cmd % (
            self.treatment,
            self.name,
            self.outdir,
            self.input_format,
            self.gsize,
            self.qvalue,
            self.shift,
            self.ext_size,
            self.keep_dup,
        )
        log.info(f"Calling peaks for {self.name} with {cmd}")
        try:
            subprocess.check_output(args=cmd, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {}".format(
                    e.cmd, e.returncode, e.output
                )
            )
        self.narrow_peak = self.load_narrow_peak(self.skip_empty_peaks)

    def load_narrow_peak(self, skip_empty_peaks: bool):
        """
        Load MACS2 narrow peak files as :class:`pr.PyRanges`.
        """
        # check if file is empty
        file_is_empty = False
        with open(os.path.join(self.outdir, f"{self.name}_peaks.narrowPeak")) as f:
            first_line = f.readline()
            if len(first_line) == 0:
                file_is_empty = True
        if file_is_empty and skip_empty_peaks:
            print(f"{self.name} has no peaks, skipping")
            return  pr.PyRanges()
        elif file_is_empty and not skip_empty_peaks:
            raise ValueError(f"{self.name} has no peaks, exiting. Set skip_empty_peaks to True to skip empty peaks.")
        narrow_peak = pd.read_csv(
            os.path.join(self.outdir, f"{self.name}_peaks.narrowPeak"),
            sep="\t",
            header=None,
        )
        narrow_peak.columns = [
            "Chromosome",
            "Start",
            "End",
            "Name",
            "Score",
            "Strand",
            "FC_summit",
            "-log10_pval",
            "-log10_qval",
            "Summit",
        ]
        narrow_peak_pr = pr.PyRanges(narrow_peak)
        return narrow_peak_pr
