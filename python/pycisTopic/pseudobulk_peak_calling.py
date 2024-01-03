import gc
import logging
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Union, Tuple, Set

import numpy as np
import pandas as pd
import pyBigWig
import pyranges as pr
import ray

from .cistopic_class import *
from .utils import *

import pysam
import joblib
from pyrle import Rle
from pyrle import PyRles
from pyrle.src.coverage import _coverage
from pyrle.methods import _to_ranges
import gzip
import shutil

def _get_fragments_for_cell_barcodes_single_contig(
        path_to_fragments: str,
        contig: str,
        cell_type_to_cell_barcodes: Dict[str, Set[str]])  -> Dict[str, List[Tuple[str, int, int, str, int]]]:
    cell_type_to_fragments = {
        cell_type: [] for cell_type in cell_type_to_cell_barcodes.keys()
    }
    tbx = pysam.TabixFile(path_to_fragments)
    for line in tbx.fetch(contig):
        chromosome, start, end, barcode, score = line.strip().split("\t")
        start = int(start)
        end = int(end)
        score = int(score)
        for cell_type in cell_type_to_cell_barcodes.keys():
            if barcode in cell_type_to_cell_barcodes[cell_type]:
                cell_type_to_fragments[cell_type].append(
                    (chromosome, start, end, barcode, score)
                )
    return cell_type_to_fragments

def _get_fragments_for_cell_barcodes(
    path_to_fragments: str,
    cell_type_to_cell_barcodes: Dict[str, List[str]],
    n_cores: int = 1) -> Dict[str, List[Tuple[str, int, int, str, int]]]:
    """
    Get fragments for cell barcodes.
    Parameters
    ----------
    path_to_fragments: str
        Path to fragments file.
    cell_type_to_cell_barcodes: dict
        A dictionary containing cell types as keys and a list of barcodes as values.
        It specifies which fragments should be extracted for each cell type.
    Returns
    -------
    dict
        A dictionary containing cell types as keys and a list of fragments as values.
    """
    tbx = pysam.TabixFile(path_to_fragments)
    contigs = tbx.contigs
    tbx.close()
    cell_type_to_fragments_per_contig = joblib.Parallel(n_jobs=n_cores)(
        joblib.delayed(_get_fragments_for_cell_barcodes_single_contig)(
            path_to_fragments, contig, cell_type_to_cell_barcodes
        )
        for contig in contigs
    )
    cell_type_to_fragments = {
        cell_type: [] for cell_type in cell_type_to_cell_barcodes.keys()
    }
    for i in range(len(cell_type_to_fragments_per_contig)):
        for cell_type in cell_type_to_cell_barcodes.keys():
            cell_type_to_fragments[cell_type].extend(
                cell_type_to_fragments_per_contig[i][cell_type]
            )
    tbx.close()
    return cell_type_to_fragments

def _fragments_to_run_length_encoding(
        fragments: List[Tuple[str, int, int, str, int]],
        contig: str,
        use_value_col: bool = False,
) -> Rle:
    """
    Convert fragments to run length encoding.

    Parameters
    ----------
    fragments: list
        A list of fragments of a single contig!.
    """
    if use_value_col:
        values = np.array([fragment[4] for fragment in fragments], dtype=np.float64)
    else:
        values = np.ones(len(fragments), dtype = np.float64)
    positions_and_values = []
    for fragment, score in zip(fragments, values):
        positions_and_values.append(
            (fragment[1], score)
        )
        positions_and_values.append(
            (fragment[2], -score)
        )
    positions_and_values.sort(key = lambda x: x[0])
    positons, values = zip(*positions_and_values)
    runs, values = _coverage(
        np.array(positons, dtype = np.int64),
        np.array(values, dtype = np.float64))
    return {contig: Rle(runs, values)}

def _pyrles_to_bigwig(
        rle: PyRles,
        path: str,
        chromosome_sizes: pr.PyRanges
    ) -> None:
    """
    Convert a PyRles object to a bigwig file.
    Parameters
    ----------
    gr: PyRles
        A PyRles object.
    path: str
        Path to the output bigwig file.
    """
    # based on pyranges out.py
    unique_chromosomes = rle.chromosomes
    size_df = chromosome_sizes.df
    chromosome_sizes = {k: v for k, v in zip(size_df.Chromosome, size_df.End)}
    header = [(c, int(chromosome_sizes[c])) for c in unique_chromosomes]
    bw = pyBigWig.open(path, "w")
    bw.addHeader(header)
    for chromosome in unique_chromosomes:
        starts, ends, values = _to_ranges(rle[chromosome])
        bw.addEntries(
            [chromosome for _ in starts],
            list(starts),
            ends=list(ends), 
            values=list(values))
    bw.close()

def export_pseudobulk(
    input_data: Union[CistopicObject, pd.DataFrame],
    variable: str,
    chromsizes: Union[pd.DataFrame, pr.PyRanges],
    bed_path: str,
    bigwig_path: str,
    path_to_fragments: Optional[Dict[str, str]] = None,
    sample_id_col: str = "sample_id",
    n_cpu: int = 1,
    normalize_bigwig: bool = True,
    remove_duplicates: bool = True,
    split_pattern: str = "___"
):
    """
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
        cell_data = input_data.cell_data
    elif isinstance(input_data, pd.DataFrame):
        if path_to_fragments is None:
            log.error("Please, provide path_to_fragments.")
        cell_data = input_data
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
    # make output folders, if they don't exists
    if not os.path.exists(bed_path):
        os.makedirs(bed_path)
    if not os.path.exists(bigwig_path):
        os.makedirs(bigwig_path)
    # For each sample, get fragments for each cell type
    cell_type_to_fragments_all_samples = {
        cell_type: [] for cell_type in cell_data[variable].unique()
    }
    for sample in sample_ids:
        print(
            f"Reading fragments for {sample}.\nfrom: {path_to_fragments[sample]}")
        _sample_cell_data = cell_data.loc[cell_data[sample_id_col] == sample]
        # format barcodes, if needed
        if "barcode" not in _sample_cell_data.columns:
            _sample_cell_data["barcode"] = prepare_tag_cells(
                _sample_cell_data.index.tolist(), split_pattern
            )
        _cell_type_to_cell_barcodes = _sample_cell_data \
            .groupby(variable, group_keys=False)["barcode"] \
            .apply(set) \
            .to_dict()
        _cell_type_to_fragments = _get_fragments_for_cell_barcodes(
            path_to_fragments[sample], _cell_type_to_cell_barcodes, n_cores=n_cpu
        )
        for cell_type in _cell_type_to_fragments.keys():
            cell_type_to_fragments_all_samples[cell_type].extend(
                _cell_type_to_fragments[cell_type]
            )
    bed_paths = {}
    bw_paths = {}
    log.info("Saving bed and BigWig files.")
    for cell_type in cell_type_to_fragments_all_samples.keys():
        # Define output paths
        _bigwig_fname = os.path.join(bigwig_path, f"{cell_type}.bw")
        _bed_fname = os.path.join(bed_path, f"{cell_type}.bed.gz")
        bw_paths[cell_type] = _bigwig_fname
        bed_paths[cell_type] = _bed_fname
        log.info(
            f"Saving {cell_type}.\n\tBigWig: {_bigwig_fname}\n\tBED: {_bed_fname}"
        )
        # sort fragments in place, by chromosome and 
        cell_type_to_fragments_all_samples[cell_type].sort(
            key=lambda x: (x[0], x[1])
        )
        # get indices for each contig
        contig_start_end_indices = {}
        prev_contig = None
        for i, fragment in enumerate(cell_type_to_fragments_all_samples[cell_type]):
            contig = fragment[0]
            if contig not in contig_start_end_indices:
                # this initialization with None ensures that the last contig is also includes
                # None, here means "until the end of the list"
                contig_start_end_indices[contig] = (i, None)
                if prev_contig is not None:
                    contig_start_end_indices[prev_contig] = (
                        contig_start_end_indices[prev_contig][0],
                        i,
                    )
                prev_contig = contig

        log.info(f"Generating bigwig for {cell_type}")
        contigs_in_chromsizes = set(chromsizes.Chromosome) & set(contig_start_end_indices.keys())

        RLE_per_contig = joblib.Parallel(n_jobs=n_cpu)(
            joblib.delayed(_fragments_to_run_length_encoding)
            (
                fragments = cell_type_to_fragments_all_samples[cell_type][
                    contig_start_end_indices[contig][0] : contig_start_end_indices[contig][1]
                ],
                contig = contig,
                use_value_col = not remove_duplicates
            )
            for contig in contigs_in_chromsizes
        )
        RLE_per_contig = {k: v for _dict in RLE_per_contig for k, v in _dict.items()}
        if normalize_bigwig:
            # TODO: add better normalization methods
            multiplier = 1e6 / len(cell_type_to_fragments_all_samples[cell_type])
            RLE_per_contig = {k: v * multiplier for k, v in RLE_per_contig.items()}
        pyrles_cell_type = PyRles(RLE_per_contig)
        log.info(f"Saving bigwig for {cell_type}")
        _pyrles_to_bigwig(pyrles_cell_type, _bigwig_fname, chromsizes)
        log.info(f"Saving bed for {cell_type}")
        with open(_bed_fname.rsplit(".", 1)[0], "wt") as f:
            for fragment in cell_type_to_fragments_all_samples[cell_type]:
                f.write("\t".join([str(x) for x in fragment]) + "\n")
        with open(_bed_fname.rsplit(".", 1)[0], "rb") as f_in:
            with gzip.open(_bed_fname, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(_bed_fname.rsplit(".", 1)[0])
    return bw_paths, bed_paths

def peak_calling(
    macs_path: str,
    bed_paths: Dict,
    outdir: str,
    genome_size: str,
    n_cpu: Optional[int] = 1,
    input_format: Optional[str] = "BEDPE",
    shift: Optional[int] = 73,
    ext_size: Optional[int] = 146,
    keep_dup: Optional[str] = "all",
    q_value: Optional[float] = 0.05,
    nolambda: Optional[bool] = True,
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
                )
                for name in list(bed_paths.keys())
            ]
        )
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
                )
                for name in list(bed_paths.keys())
            ]
    narrow_peaks_dict = {
        list(bed_paths.keys())[i]: narrow_peaks[i].narrow_peak
        for i in range(len(narrow_peaks))
    }
    return narrow_peaks_dict

def macs_call_peak(
    macs_path: str,
    bed_path: str,
    name: str,
    outdir: str,
    genome_size: str,
    input_format: Optional[str] = "BEDPE",
    shift: Optional[int] = 73,
    ext_size: Optional[int] = 146,
    keep_dup: Optional[str] = "all",
    q_value: Optional[int] = 0.05,
    nolambda: Optional[bool] = True,
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
    input_format: Optional[str] = "BEDPE",
    shift: Optional[int] = 73,
    ext_size: Optional[int] = 146,
    keep_dup: Optional[str] = "all",
    q_value: Optional[int] = 0.05,
    nolambda: Optional[bool] = True,
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
        input_format: Optional[str] = "BEDPE",
        shift: Optional[int] = 73,
        ext_size: Optional[int] = 146,
        keep_dup: Optional[str] = "all",
        q_value: Optional[int] = 0.05,
        nolambda: Optional[bool] = True,
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
        self.narrow_peak = self.load_narrow_peak()

    def load_narrow_peak(self):
        """
        Load MACS2 narrow peak files as :class:`pr.PyRanges`.
        """
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
