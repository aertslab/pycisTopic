import collections as cl
import gc
import pandas as pd
import pyranges as pr
import logging
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import ray
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.stats import norm
import sys
from typing import Optional, Union
from typing import List, Dict, Tuple
from .cistopic_class import *
from .utils import multiplot_from_generator, collapse_duplicates

pd.options.mode.chained_assignment = None
dtype = pd.SparseDtype(int, fill_value=0)
plt.ioff()


def barcode_rank_plot(fragments: Union[str, pd.DataFrame],
                      valid_bc: Optional[List[str]] = None,
                      n_cpu: Optional[int] = 1,
                      n_frag: Optional[int] = None,
                      n_bc: Optional[int] = None,
                      remove_duplicates: Optional[bool] = True,
                      plot: Optional[bool] = True,
                      color: Optional[List[str]] = None,
                      plot_data: Optional[pd.DataFrame] = None,
                      save: Optional[str] = None,
                      return_plot_data: Optional[bool] = False,
                      return_bc: Optional[bool] = False):
    """
    Generate a barcode rank plot and marks the selected barcodes

    Parameters
    ---------
    fragments: str or pd.DataFrame
            The path to the fragments file containing chromosome, start, end and assigned barcode for each read (e.g. from CellRanger ATAC (/outs/fragments.tsv.gz)) or a data frame
            containing 'Chromosome', 'Start', 'End', 'Name', and 'Score', which indicates the number of times that a fragments is found assigned to that barcode. The fragments data
            frame can be obtained using PyRanges:
                    import pyranges as pr
                    fragments = pr.read_bed(fragments_file, as_df=True))
    valid_bc: list, optional
            A list containing selected barcodes. This parameter is ignored if n_frag or n_bc are specified. Default: None.
    n_cpu: int, optional
            Number of cores to use. Default: 1.
    n_frag: int, optional
            Minimal number of fragments assigned to a barcode to be kept. Either n_frag or n_bc can be specified. Default: None.
    n_bc: int, optional
            Number of barcodes to select. Either n_frag or n_bc can be specified. Default: None.
    remove_duplicates: optional, bool
            Whether duplicated should not be considered. Default: True
    plot: bool, optional
            Whether to return the plot to the console. Default: True.
    color: list, optional
            List with line colors for the plot [color for selected barcodes, color for non-selected barcodes]. Default: None.
    plot_data: pd.DataFrame, optional
            Data frame containing precomputed plot data. This parameter is useful when adjusting parameters, as it skips the fragment counting steps. Default: None.
    save: str, optional
            Output file to save plot. Default: None
    return_plot_data: bool, optional
            Whether to return the plot data frame. Default: False.
    return_bc: bool, optional
            Whether to return the list of valid barcodes. Default: False.

    Return
    ------
    dict
            A dictionary containing a :class:`list` with the valid barcodes and/or a :class:`pd.DataFrame` with the knee plot data.
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    if isinstance(plot_data, pd.DataFrame):
        log.info('Using precomputed plot data')
        FPB_DF = plot_data
    else:
        if isinstance(fragments, str):
            fragments = pr.read_bed(fragments).df

        log.info('Counting fragments')
        fragments_per_barcode_dup = fragments.groupby(
            ["Name"], sort=False, observed=True).agg({"Score": np.sum}).rename_axis(None)
        fragments_per_barcode_dup.columns = ['Total_nr_frag']
        fragments_per_barcode_nodup = fragments.groupby(
            ["Name"], sort=False).size().to_frame(
            name='Unique_nr_frag').rename_axis(None)
        FPB_DF = pd.concat(
            [fragments_per_barcode_dup, fragments_per_barcode_nodup], axis=1)

    if not remove_duplicates:
        FPB_DF = FPB_DF.sort_values(by=['Total_nr_frag'], ascending=False)
        NF = FPB_DF.loc[:, 'Total_nr_frag']
    else:
        FPB_DF = FPB_DF.sort_values(by=['Unique_nr_frag'], ascending=False)
        NF = FPB_DF.loc[:, 'Unique_nr_frag']

    FPB_DF['Barcode_rank'] = range(1, len(FPB_DF) + 1)

    BR = FPB_DF.loc[:, 'Barcode_rank']

    if n_frag is None:
        if n_bc is None:
            if valid_bc is None:
                log.error(
                    "Please provide a list of valid barcodes, the minimal number of barcodes or the minimal number of fragments per barcode to select")
                return
            else:
                log.info('Marking valid barcodes contained')
                selected_bc = len(valid_bc)
        else:
            log.info(f'Marking top {n_bc} barcodes')
            selected_bc = n_bc
    else:
        if n_bc is None:
            log.info(f'Marking barcodes with more than {n_frag}')
            selected_bc = sum(NF > n_frag)
        else:
            log.error('Either the number of cells to select or the minimal number of fragments can be specified, but not both at the same time. Please, set n_frag=None or n_bc=None.')
            return

    if plot is True or save is not None:
        sel = np.ma.masked_where(BR >= selected_bc, BR)
        nosel = np.ma.masked_where(BR < selected_bc, BR)

        fig, ax = plt.subplots()
        if color is None:
            ax.plot(sel, NF, nosel, NF)
        else:
            ax.plot(sel, NF, color[0], nosel, NF, color[1])
        ax.legend((f'Selected BC: {selected_bc}',
               f'Non-selected BC: {len(BR-selected_bc)}'))
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Barcode Rank", fontsize=10)
        plt.ylabel("Number of fragments", fontsize=10)
        
        if save is not None:
            fig.savefig(save)
        if plot:
            plt.show()
        else:
            plt.close(fig)

    output = {}
    flag = False
    if return_bc:
        log.info('Returning plot data')
        output.update(
            {'valid_bc': FPB_DF.iloc[0:selected_bc, ].index.tolist()})
        flag = True
    if return_plot_data:
        log.info('Returning valid barcodes')
        output.update({'valid_bc_plot_data': FPB_DF})
        flag = True
    if flag:
        return output


def duplicate_rate(fragments: Union[str, pd.DataFrame],
                   valid_bc: Optional[List[str]] = None,
                   plot: Optional[bool] = True,
                   cmap: Optional[List[str]] = None,
                   plot_data: Optional[pd.DataFrame] = None,
                   save: Optional[str] = None,
                   return_plot_data: Optional[bool] = False):
    """
    Generate duplication rate plot.

    Parameters
    ---------
    fragments: str or pd.DataFrame
            The path to the fragments file containing chromosome, start, end and assigned barcode for each read (e.g. from CellRanger ATAC (/outs/fragments.tsv.gz)) or a data frame
            containing 'Chromosome', 'Start', 'End', 'Name', and 'Score', which indicates the number of times that a fragments is found assigned to that barcode. The fragments data
            frame can be obtained using PyRanges:
                    import pyranges as pr
                    fragments = pr.read_bed(fragments_file, as_df=True))
    valid_bc: list, optional
            A list containing selected barcodes. This parameter is ignored if n_frag or n_bc are specified. Default: None.
    plot: bool, optional
            Whether to return the plot to the console. Default: True.
    cmap: list, optional
            Color map to color the plot by density.
    plot_data: pd.DataFrame, optional
            Data frame containing precomputed plot data. This parameter is useful when adjusting parameters, as it skips the fragment counting steps. Default: None.
    save: str, optional
            Output file to save plot. Default: None
    return_plot_data: bool, optional
            Whether to return the plot data frame. Default: False.

    Return
    ------
    dict
            A dictionary containing a :class:`pd.DataFrame` with the duplication rate data.
    """

    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    if isinstance(plot_data, pd.DataFrame):
        log.info('Using precomputed plot data')
        fragments_per_barcode = plot_data
    else:
        if isinstance(fragments, str):
            log.info('Reading fragments file')
            fragments = pr.read_bed(fragments).df

        if valid_bc is not None:
            log.info('Using provided valid barcodes')
            fragments = fragments[fragments.Name.isin(set(valid_bc))]

        FPB_dup = fragments.groupby(["Name"], observed=True, sort=False).agg(
            {"Score": np.sum}).rename_axis(None)
        FPB_dup.columns = ['Total_nr_frag']
        FPB_nodup = fragments.groupby(
            ["Name"], sort=False, observed=True).size().to_frame(
            name='Unique_nr_frag').rename_axis(None)

        FPB = pd.concat([FPB_dup, FPB_nodup], axis=1)
        FPB['Dupl_nr_frag'] = FPB['Total_nr_frag'] - FPB['Unique_nr_frag']
        FPB['Dupl_rate'] = FPB['Dupl_nr_frag'] / FPB['Total_nr_frag']

    x = FPB['Unique_nr_frag']
    y = FPB['Dupl_rate']
    
    if plot is True or save is True:
        try:
            fig = plt.figure()
            xy = np.vstack([np.log(x), y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            plt.scatter(x, y, c=z, s=10, edgecolor=None, cmap=cmap)
        except BaseException:
            log.info('All fragments are unique')
            plt.scatter(x, y, s=10, edgecolor=None, cmap=cmap)
        plt.ylim(0, 1)
        plt.xscale("log")
        plt.xlabel("Number of (unique) fragments", fontsize=10)
        plt.ylabel("Duplication rate", fontsize=10)
        plt.colorbar().set_label('Density')

        if save is not None:
            fig.savefig(save)
        if plot:
            plt.show()
        else:
            plt.close(fig)

    output = {}
    if return_plot_data:
        log.info('Return plot data')
        output.update({'duplicate_rate_plot_data': FPB})
        return output


def insert_size_distribution(fragments: Union[str, pd.DataFrame],
                             valid_bc: Optional[List[str]] = None,
                             remove_duplicates: Optional[bool] = True,
                             plot: Optional[bool] = True,
                             plot_data: Optional[pd.DataFrame] = None,
                             color: Optional[str] = None,
                             save: Optional[str] = None,
                             return_plot_data: Optional[bool] = False,
                             xlim: Optional[List[float]] = None):
    """
    Plot the insert size distribution of the sample.

    Parameters
    ---------
    fragments: str or pd.DataFrame
            The path to the fragments file containing chromosome, start, end and assigned barcode for each read (e.g. from CellRanger ATAC (/outs/fragments.tsv.gz)) or a data frame
            containing 'Chromosome', 'Start', 'End', 'Name', and 'Score', which indicates the number of times that a fragments is found assigned to that barcode. The fragments data
            frame can be obtained using PyRanges:
                    import pyranges as pr
                    fragments = pr.read_bed(fragments_file, as_df=True))
    valid_bc: list, optional
            A list containing selected barcodes. Default: None
    remove_duplicates: optional, bool
            Whether duplicated should not be considered. Default: True
    plot: bool, optional
            Whether to return the plot to the console. Default: True.
    plot_data: pd.DataFrame, optional
            Data frame containing precomputed plot data. Default: None.
    color: str, optional
            Line color for the plot. Default: None.
    save: str, optional
            Output file to save plot. Default: None.
    return_plot_data: bool, optional
            Whether to return the plot data frame. Default: False.
    xlim: list, optional
            A list with two numbers that indicate the x axis limits. Default: None.

    Return
    ------
    dict
            A dictionary containing a :class:`pd.DataFrame` with the insert size plot data.
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    if isinstance(plot_data, pd.DataFrame):
        log.info('Using precomputed plot data')
        FPW_DF = plot_data
    else:
        if isinstance(fragments, str):
            log.info('Reading fragments file')
            fragments = pr.read_bed(fragments, as_df=True)

        if valid_bc is not None:
            log.info('Using provided valid barcodes')
            fragments = fragments[fragments.Name.isin(set(valid_bc))]

        log.info('Counting fragments')
        fragments['Width'] = abs(
            fragments['End'].values -
            fragments['Start'].values)
        if not remove_duplicates:
            FPW_DF = fragments.groupby(["Name"]).agg({"Score": np.sum}).rename_axis(
                None).reset_index().rename(columns={"index": "Width", "Score": "Nr_frag"})
        else:
            FPW_DF = fragments.groupby(
                ["Width"]).size().to_frame(
                name='Nr_frag').rename_axis(None).reset_index().rename(
                columns={
                    "index": "Width"})
        FPW_DF['Ratio_frag'] = (
            FPW_DF['Nr_frag'].values /
            np.sum(
                FPW_DF['Nr_frag']))
        FPW_DF = FPW_DF.sort_values(by=['Width'], ascending=False)

    if plot is True or save is not None:
        W = FPW_DF.loc[:, 'Width']
        pF = FPW_DF.loc[:, 'Ratio_frag']
        fig, ax = plt.subplots()
        ax.plot(W, pF, color=color)
        plt.xlabel("Fragment size", fontsize=10)
        plt.ylabel("Fragments ratio", fontsize=10)
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        if save is not None:
            fig.savefig(save)
        if plot:
            plt.show()
        else:
            plt.close(fig)

    output = {}
    if return_plot_data:
        log.info('Returning plot data')
        output.update({'fragment_size_plot_data': FPW_DF})
        return(output)


def profile_tss(fragments: Union[str, pd.DataFrame],
                annotation: Union[pd.DataFrame, pr.PyRanges],
                valid_bc: Optional[List[str]] = None,
                plot: Optional[bool] = True,
                plot_data: Optional[pd.DataFrame] = None,
                n_cpu: Optional[int] = 1,
                partition: Optional[int] = 5,
                flank_window: Optional[int] = 1000,
                tss_window: Optional[int] = 50,
                minimum_signal_window: Optional[int] = 100,
                rolling_window: Optional[int] = 10,
                min_norm: Optional[int] = 0.2,
                color: Optional[str] = None,
                save: Optional[str] = None,
                return_TSS_enrichment_per_barcode: Optional[bool] = False,
                return_TSS_coverage_matrix_per_barcode: Optional[bool] = False,
                return_plot_data: Optional[bool] = False):
    """
    Plot the Transcription Start Site (TSS) profile. It is computed as the summed accessibility signal (sample-level), or the number of cut sites per base (barcode-level), in a space around the full set of annotated TSSs and is normalized by the minimum signal in the window. This profile is helpful to assess the signal-to-noise ratio of the library, as it is well known that TSSs and the promoter regions around them have, on average, a high degree of chromatin accessibility compared to the intergenic and intronic regions of the genome.

    Parameters
    ---------
    fragments: str or pd.DataFrame
            The path to the fragments file containing chromosome, start, end and assigned barcode for each read (e.g. from CellRanger ATAC (/outs/fragments.tsv.gz)) or a data frame
            containing 'Chromosome', 'Start', 'End', 'Name', and 'Score', which indicates the number of times that a fragments is found assigned to that barcode. The fragments data
            frame can be obtained using PyRanges:
                    import pyranges as pr
                    fragments = pr.read_bed(fragments_file, as_df=True))
    annotation: pd.DataFrame or pyRanges
            A data frame or pyRanges containing transcription start sites for each gene, with 'Chromosome', 'Start' and 'Strand' as columns (additional columns will be ignored). This data frame can be easily obtained via pybiomart:
                    # Get TSS annotations
                    import pybiomart as pbm
                    # For mouse
                    dataset = pbm.Dataset(name='mmusculus_gene_ensembl',  host='http://www.ensembl.org')
                    # For human
                    dataset = pbm.Dataset(name='hsapiens_gene_ensembl',  host='http://www.ensembl.org')
                    # For fly
                    dataset = pbm.Dataset(name='dmelanogaster_gene_ensembl',  host='http://www.ensembl.org')
                    # Query TSS list and format
                    annot = dataset.query(attributes=['chromosome_name', 'transcription_start_site', 'strand', 'external_gene_name', 'transcript_biotype'])
                    filter = annot['Chromosome/scaffold name'].str.contains('CHR|GL|JH|MT')
                    annot = annot[~filter]
                    annot['Chromosome/scaffold name'] = annot['Chromosome/scaffold name'].str.replace(r'(\b\\S)', r'chr\1')
                    annot.columns=['Chromosome', 'Start', 'Strand', 'Gene', 'Transcript_type']
                    # Select TSSs of protein coding genes
                    annot = annot[annot.Transcript_type == 'protein_coding']
    valid_bc: list, optional
            A list containing selected barcodes. Default: None,
    plot: bool, optional
            Whether to return the plot to the console. Default: True.
    plot_data: pd.DataFrame, optional
            Data frame containing precomputed plot data. Default: None.
    flank_window: int, optional
            Flanking window around the TSS. Default: 1000 (+/- 1000 bp).
    tss_window: int, optional
            Window around the TSS used to count fragments in the TSS when calculating the TSS enrichment per barcode. Default: 50 (+/- 50 bp).
    minimum_signal_window: int, optional
            Tail window use to normalize the TSS enrichment. Default: 100 (average signal in the 100bp in the extremes of the TSS window).
    rolling_window: int, optional
            Rolling window used to smooth signal. Default: 10.
    min_norm: int, optional
            Minimum normalization score. If the average minimum signal value is below this value, this number is used to normalize the TSS signal. This approach penalizes cells with fewer reads.
    color: str, optional
            Line color for the plot. Default: None.
    save: str, optional
            Output file to save plot. Default: None.
    remove_duplicates: bool, optional
            Whether to remove duplicates. Default: True.
    return_TSS_enrichment_per_barcode: bool, optional
            Whether to return a data frame containing the normalized enrichment score on the TSS for each barcode. Default: False.
    return_TSS_coverage_matrix_per_barcode: bool, optional
            Whether to return a matrix containing the normalized enrichment in each position in the window for each barcode, with positions as columns and barcodes as rows. Default: False.
    return_plot_data: bool, optional
                    Whether to return the TSS profile plot data. Default: False.

    Return
    ------
    dict
    A dictionary containing a :class:`pd.DataFrame` with the normalized enrichment score on the TSS for each barcode, a :class:`pd.DataFrame` with the normalized enrichment scores in each position for each barcode and/or a :class:`pd.DataFrame` with the TSS profile plot data.
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    if isinstance(plot_data, pd.DataFrame):
        log.info('Using plot_data. TSS enrichment per barcode will not be computed')
        fig, ax = plt.subplots()
        ax.plot(plot_data.Position, plot_data.TSSEnrichment)
        plt.xlim(-space_TSS, space_TSS)
        plt.xlabel("Position from TSS", fontsize=10)
        plt.ylabel("Normalized enrichment", fontsize=10)
    else:
        if isinstance(fragments, str):
            log.info('Reading fragments file')
            fragments = pr.read_bed(fragments)
        else:
            if isinstance(fragments, pd.DataFrame):
                fragments = pr.PyRanges(fragments)

        if valid_bc is not None:
            log.info('Using provided valid barcodes')
            fragments = fragments[fragments.Name.isin(set(valid_bc))]
        else:
            valid_bc = list(set(fragments.Name.tolist()))

        log.info('Formatting annnotation')
        if isinstance(annotation, pr.PyRanges):
            annotation = annotation.df
        tss_space_annotation = annotation[['Chromosome', 'Start', 'Strand']]
        tss_space_annotation['End'] = tss_space_annotation['Start'] + flank_window
        tss_space_annotation['Start'] = tss_space_annotation['Start'] - flank_window
        tss_space_annotation = tss_space_annotation[[
            "Chromosome", "Start", "End", "Strand"]]
        tss_space_annotation = pr.PyRanges(tss_space_annotation)
        
        log.info('Creating coverage matrix')
        if partition > 1:
            barcode_list = np.array_split(valid_bc, partition)
            TSS_matrix = pd.concat([get_tss_matrix(fragments[fragments.Name.isin(set(barcode_list[x]))],
         flank_window, tss_space_annotation).fillna(0) for x in range(partition)])
        else:
            TSS_matrix = get_tss_matrix(fragments, flank_window, tss_space_annotation)
        log.info('Coverage matrix done') 
        if not TSS_matrix.columns.tolist() == list(range(2 * flank_window + 1)):
            missing_values = list(set(TSS_matrix.columns.tolist()).symmetric_difference(
                list(range(2 * flank_window + 1))))
            for x in missing_values:
                TSS_matrix[x] = 0
        
            TSS_matrix = TSS_matrix.reindex(sorted(TSS_matrix.columns), axis=1)
        
        if rolling_window is not None:
            TSS_matrix = TSS_matrix.rolling(
                window=rolling_window, min_periods=0, axis=1).mean()
        
        TSS_counts = TSS_matrix.values.sum(axis=0) 
        div = max((np.mean(TSS_counts[-minimum_signal_window:]) +
                  np.mean(TSS_counts[0:minimum_signal_window])) / 2, min_norm)    
        if plot is True or save is not None:
            fig, ax = plt.subplots()
            ax.plot(range(-flank_window - 1, flank_window),
                TSS_counts / div, color=color)
            plt.xlim(-flank_window, flank_window)
            plt.xlabel("Position from TSS", fontsize=10)
            plt.ylabel("Normalized enrichment", fontsize=10)
            if save is not None:
                fig.savefig(save)
            if plot:
                log.info('Plotting normalized sample TSS enrichment')
                plt.show()
            else:
                   plt.close(fig)

    output = {}
    flag = False
    if return_TSS_enrichment_per_barcode:
        TSS_enrich = TSS_matrix.apply(
            lambda x: x / max([((np.mean(x[-minimum_signal_window:]) + np.mean(x[0:minimum_signal_window])) / 2), min_norm]), axis=1)
        TSS_enrich = pd.DataFrame(TSS_enrich.iloc[:, range(
            flank_window - tss_window, flank_window + tss_window)].mean(axis=1))
        TSS_enrich.columns = ['TSS_enrichment']
        output.update({'TSS_enrichment': TSS_enrich})
        flag = True
    if return_TSS_coverage_matrix_per_barcode:
        log.info('Returning normalized TSS coverage matrix per barcode')
        TSS_mat = TSS_matrix.apply(
            lambda x: x / max([((np.mean(x[-minimum_signal_window:]) + np.mean(x[0:minimum_signal_window])) / 2), min_norm]), axis=1)
        output.update({'TSS_coverage_mat': TSS_mat})
        flag = True
    if return_plot_data:
        log.info('Returning normalized sample TSS enrichment data')
        output.update({'TSS_plot_data': pd.DataFrame(
            {'Position': range(-flank_window - 1, flank_window), 'TSS_enrichment': TSS_counts / div})})
        flag = True
    del TSS_matrix
    if flag:
        return output


def frip(fragments: Union[str, pd.DataFrame],
         path_to_regions: str,
         valid_bc: Optional[List[str]] = None,
         path_to_blacklist: Optional[str] = None,
         remove_duplicates: Optional[bool] = True,
         n_bins: Optional[int] = 100,
         color: Optional[str] = None,
         n_cpu: Optional[int] = 1,
         save: Optional[str] = None,
         plot: Optional[bool] = True,
         as_density: Optional[bool] = True,
         plot_data: Optional[pd.DataFrame] = None,
         return_plot_data: Optional[bool] = False):
    """
    Plot the Fraction of Reads In a given set of Peaks (FRIP). This metric is useful to assess the noise in cells.

    Parameters
    ---------
    fragments: str or pd.DataFrame
            The path to the fragments file containing chromosome, start, end and assigned barcode for each read (e.g. from CellRanger ATAC (/outs/fragments.tsv.gz)) or a data frame
            containing 'Chromosome', 'Start', 'End', 'Name', and 'Score', which indicates the number of times that a fragments is found assigned to that barcode. The fragments data
            frame can be obtained using PyRanges:
            import pyranges as pr
            fragments = pr.read_bed(fragments_file, as_df=True))
    path_to_regions: str, optional
            Path to the bed file with the defined regions in which fragments will be considered.
    valid_bc: list, optional
            A list containing selected barcodes. Default: None
    path_to_blacklist: str, optional
            Path to bed file containing blacklist regions (Amemiya et al., 2019). Default: None
    remove_duplicates: optional, bool
            Whether duplicated should not be considered. Default: True
    n_bins: int, optional
            Number of bins to use in the histogram. Default: 100
    color: str, optional
            Color of the histogram. Default: None (standard line color sequence)
    n_cpu: int, optional
            Number of cores to use. Default: 1.
    save: str, optional
            Output file to save plot. Default: None.
    plot: bool, optional
            Whether to return the plot to the console. Default: True.
    as_density: bool, optional
    Whether to plot density plot instead of histogram. Default: True.
    plot_data: pd.DataFrame, optional
            Data frame containing precomputed plot data. Default: None.
    return_plot_data: bool, optional
            Whether to return the plot data frame. Default: False.

    Return
    ------
    dict
            A dictionary containing a :class:`pd.DataFrame` with the Fraction of Reads in Peaks (FRIP) for each barcode.
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    path_to_fragments = None

    if isinstance(plot_data, pd.DataFrame):
        FPB_FPBIR_DF = plot_data
    else:
        if isinstance(fragments, str):
            log.info('Reading fragments file')
            fragments = pr.read_bed(fragments)
        else:
            if isinstance(fragments, pd.DataFrame):
                fragments = pr.PyRanges(fragments)

        if valid_bc is not None:
            log.info('Using provided valid barcodes')
            fragments = fragments[fragments.Name.isin(set(valid_bc))]

        regions = pr.read_bed(path_to_regions)
        regions = regions[['Chromosome', 'Start', 'End']
                          ].drop_duplicate_positions()

        if isinstance(path_to_blacklist, str):
            blacklist = pr.read_bed(path_to_blacklist)
            regions = regions.overlap(blacklist, invert=True)

        log.info('Counting fragments')
        fragments_per_barcode_dup = fragments.df.groupby(
            ["Name"]).agg({"Score": np.sum}).rename_axis(None)
        fragments_per_barcode_dup.columns = ['Total_nr_frag']
        fragments_per_barcode_nodup = fragments.df.groupby(
            ["Name"]).size().to_frame(
            name='Unique_nr_frag').rename_axis(None)
        FPB_DF = pd.concat(
            [fragments_per_barcode_dup, fragments_per_barcode_nodup], axis=1)

        log.info('Intersecting fragments with regions')
        fragments_in_regions = fragments.join(regions, nb_cpu=n_cpu)
        fragments_in_regions = fragments_in_regions[[
            'Chromosome', 'Start', 'End', 'Name', 'Score']].drop_duplicate_positions()
        fragments_in_regions_dup = fragments_in_regions.df.groupby(
            ["Name"]).agg({"Score": np.sum}).rename_axis(None)
        fragments_in_regions_dup.columns = ['Total_nr_frag_in_regions']
        fragments_in_regions_nodup = fragments_in_regions.df.groupby(
            ["Name"]).size().to_frame(
            name='Unique_nr_frag_in_regions').rename_axis(None)
        FPBIR_DF = pd.concat(
            [fragments_in_regions_dup, fragments_in_regions_nodup], axis=1)

        FPB_FPBIR_DF = pd.concat([FPB_DF, FPBIR_DF], axis=1, sort=False)

        if not remove_duplicates:
            FPB_FPBIR_DF['FRIP'] = (
                FPB_FPBIR_DF.Total_nr_frag_in_regions /
                FPB_FPBIR_DF.Total_nr_frag)
        else:
            FPB_FPBIR_DF['FRIP'] = (
                FPB_FPBIR_DF.Unique_nr_frag_in_regions /
                FPB_FPBIR_DF.Unique_nr_frag)
                
        if plot is True or save is not None:
            fig = plt.figure()
            if as_density:
                sns.distplot(
                    FPB_FPBIR_DF['FRIP'],
                    hist=False,
                    kde=True,
                    color=color)
                plt.ylabel('Density')
            else:
                plt.hist(FPB_FPBIR_DF['FRIP'], bins=n_bins, color=color)
                plt.ylabel('Frequency')
            plt.xlabel("FRIP", fontsize=10)
            if save is not None:
                plt.savefig(save)
            if plot:
                plt.show()
            else:
                plt.close(fig)

        output = {}
        if return_plot_data:
            output.update({'FRIP_plot_data': FPB_FPBIR_DF})
            return output


def metrics2data(metrics: Optional[Dict]):
    """
    Reformats metrics dictionary into a dictionary containing sample-level profiles and a data-frame containing barcode-level statistics (from FRIP and profile_tss).

    Parameters
    ---------
    metrics: dict
            Dictionary with keys 'barcode_rank_plot', 'insert_size_distribution', 'profile_tss' and/or 'FRIP', containing the results of the corresponding metric functions.

    Return
    ------
    dict
            A dictionary containing a :class:`pd.DataFrame` containing barcode-level statistics and a dictionary containing sample-level profiles.
    """

    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    # Add log total fragments
    metrics_list = []
    if 'duplicate_rate' in metrics:
        metrics['duplicate_rate']['duplicate_rate_plot_data']['Log_total_nr_frag'] = np.log10(
            metrics['duplicate_rate']['duplicate_rate_plot_data']['Total_nr_frag'])
        metrics['duplicate_rate']['duplicate_rate_plot_data']['Log_unique_nr_frag'] = np.log10(
            metrics['duplicate_rate']['duplicate_rate_plot_data']['Unique_nr_frag'])
        metrics_list.append(
            metrics['duplicate_rate']['duplicate_rate_plot_data'].iloc[:, [4, 5, 0, 1, 2, 3]])
    if 'frip' in metrics:
        if 'duplicate_rate' not in metrics:
            metrics['frip']['FRIP_plot_data']['Log_total_nr_frag'] = np.log10(
                metrics['frip']['FRIP_plot_data']['Total_nr_frag'])
            metrics['frip']['FRIP_plot_data']['Log_unique_nr_frag'] = np.log10(
                metrics['frip']['FRIP_plot_data']['Unique_nr_frag'])
            metrics_list.append(
                metrics['frip']['FRIP_plot_data'].iloc[:, [5, 6, 0, 1, 2, 3, 4]])
        else:
            metrics_list.append(
                metrics['frip']['FRIP_plot_data'].iloc[:, [2, 3, 4]])
    if 'profile_tss' in metrics:
        metrics_list.append(metrics['profile_tss']['TSS_enrichment'])

    if len(metrics_list) > 1:
        metadata_bc = pd.concat(metrics_list, axis=1, sort=False)
    elif len(metrics_list) == 1:
        metadata_bc = metrics_list[0]
    elif len(metrics_list) < 1:
        log.info('No barcode statistics have been computed')
        metadara_bc = []

    # Plot dict
    profile_dict = {}
    if 'barcode_rank_plot' in metrics:
        profile_dict['barcode_rank_plot'] = metrics['barcode_rank_plot']['valid_bc_plot_data']
        selected_bc = len(metrics['barcode_rank_plot']['valid_bc'])
        profile_dict['barcode_rank_plot']['Selected'] = np.ma.masked_where(
            profile_dict['barcode_rank_plot']['Barcode_rank'] <= selected_bc,
            profile_dict['barcode_rank_plot']['Barcode_rank']).mask
    if 'duplicate_rate' in metrics:
        profile_dict['duplicate_rate'] = metrics['duplicate_rate']['duplicate_rate_plot_data']
    if 'insert_size_distribution' in metrics:
        profile_dict['insert_size_distribution'] = metrics['insert_size_distribution']['fragment_size_plot_data']
    if 'profile_tss' in metrics:
        profile_dict['profile_tss'] = metrics['profile_tss']['TSS_plot_data']
        profile_dict['profile_tssPerBarcode'] = metrics['profile_tss']['TSS_coverage_mat']
    if 'frip' in metrics:
        profile_dict['frip'] = metrics['frip']['FRIP_plot_data']
    return metadata_bc, profile_dict


def compute_qc_stats(fragments_dict: Dict[str,
                                          Union[str,
                                                pd.DataFrame]],
                     tss_annotation: Union[pd.DataFrame,
                                           pr.PyRanges],
                     stats: Optional[List[str]] = ['barcode_rank_plot',
                                                   'duplicate_rate',
                                                   'insert_size_distribution',
                                                   'profile_tss',
                                                   'frip'],
                     label_list: Optional[List[str]] = None,
                     path_to_regions: Optional[Dict[str,
                                                    str]] = None,
                     n_cpu: Optional[int] = 1,
                     partition: Optional[int] = 1,
                     valid_bc: Optional[List[str]] = None,
                     n_frag: Optional[int] = None,
                     n_bc: Optional[int] = None,
                     tss_flank_window: Optional[int] = 1000,
                     tss_window: Optional[int] = 50,
                     tss_minimum_signal_window: Optional[int] = 100,
                     tss_rolling_window: Optional[int] = 10,
                     min_norm: Optional[int] = 0.2,
                     check_for_duplicates: Optional[bool] = True,
                     remove_duplicates: Optional[bool] = True,
                     **kwargs):
    """"
    Wrapper function to compute QC statistics on several samples. For detailed instructions, please see the independent functions.

    Parameters
    ---
    fragments_dict: dict
            Dictionary containing the path/s to the fragments file containing chromosome, start, end and assigned barcode for each read (e.g. from CellRanger ATAC (/outs/fragments.tsv.gz)) or data frames
            containing 'Chromosome', 'Start', 'End', 'Name', and 'Score', which indicates the number of times that a fragments is found assigned to that barcode.
    tss_annotation: pd.DataFrame or pr.PyRanges
            A data frame or pyRanges containing transcription start sites for each gene, with 'Chromosome', 'Start' and 'Strand' as columns (additional columns will be ignored).
    stats: list, optional
            A list with the statistics that have to be computed. Default: All ('barcode_rank_plot', 'duplicate_rate', 'insert_size_distribution', 'profile_tss', 'FRIP).
    label_list: list, optional
            A list containing the labels for each sample. By default, samples will be called 'Sample_number' (e.g.'Sample_1'). Default: None.
    path_to_regions: dict, optional
            A dictionary containing the regions to be used for each sample when calculating the rank plot and the Fraction of Reads In Peaks (FRIP).
    n_cpu: int, optional
            Number of cores to use. Default: 1.
    valid_bc: list, optional
            A list containing selected barcodes. This parameter is ignored if n_frag or n_bc are specified. Default: None.
    n_frag: int, optional
            Minimal number of fragments assigned to a barcode to be kept. Either n_frag or n_bc can be specified. Default: None.
    n_bc: int, optional
            Number of barcodes to select. Either n_frag or n_bc can be specified. Default: None.
    tss_window: int, optional
            Window around the TSS used to count fragments in the TSS when calculating the TSS enrichment per barcode. Default: 50 (+/- 50 bp).
    tss_flank_window: int, optional
            Flanking window around the TSS. Default: 1000 (+/- 1000 bp).
    tss_minimum_signal_window: int, optional
            Tail window use to normalize the TSS enrichment. Default: 100 (average signal in the 100bp in the extremes of the TSS window).
    tss_rolling_window: int, optional
            Rolling window used to smooth signal. Default: 10.
    min_norm: int, optional
            Minimum normalization score. If the average minimum signal value is below this value, this number is used to normalize the TSS signal. This approach penalizes cells with fewer reads.
    check_for_duplicates: bool, optional
            If no duplicate counts are provided per row in the fragments file, whether to collapse duplicates. Default: True.
    remove_duplicates: bool, optional
            Whether to remove duplicates. Default: True.
    **kwargs
            Additional parameters for ray.init.

    Return
    ---
    pd.DataFrame or list and list
            A list with the barcode statistics for all samples (or a combined data frame with a column 'Sample' indicating the sample of origin) and a list of dictionaries with the sample-level profiles for each sample.
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')
    # Convert dictionaries to list
    label_list = list(fragments_dict.keys())
    fragments_list = [fragments_dict[key] for key in fragments_dict.keys()]
    path_to_regions = [path_to_regions[key] for key in fragments_dict.keys()]

    if n_cpu > len(fragments_list):
        log.info('n_cpu is larger than the number of samples. Setting n_cpu to the number of samples')
        n_cpu = len(fragments_list)
        
    ray.init(num_cpus=n_cpu, **kwargs)
    qc_stats = ray.get(
        [
            compute_qc_stats_ray.remote(
                fragments_list[i],
                tss_annotation=tss_annotation,
                stats=stats,
                label=label_list[i],
                path_to_regions=path_to_regions[i],
                valid_bc=valid_bc,
                n_frag=n_frag,
                n_bc=n_bc,
                tss_flank_window=tss_flank_window,
                tss_window=tss_window,
                tss_minimum_signal_window=tss_minimum_signal_window,
                tss_rolling_window=tss_rolling_window,
                min_norm=min_norm,
                partition = partition,
                check_for_duplicates=check_for_duplicates,
                remove_duplicates=remove_duplicates) for i in range(
                    len(fragments_list))])
    ray.shutdown()
    metadata_dict = {key: x[key] for x in list(
        list(zip(*qc_stats))[0]) for key in x.keys()}
    profile_data_dict = {key: x[key] for x in list(
        list(zip(*qc_stats))[1]) for key in x.keys()}
    return metadata_dict, profile_data_dict


@ray.remote
def compute_qc_stats_ray(fragments,
                         tss_annotation: Union[pd.DataFrame,
                                               pr.PyRanges],
                         stats: Optional[List[str]] = ['barcode_rank_plot',
                                                       'duplicate_rate',
                                                       'insert_size_distribution',
                                                       'profile_tss',
                                                       'frip'],
                         label: Optional[str] = None,
                         path_to_regions: Optional[str] = None,
                         valid_bc: Optional[List[str]] = None,
                         n_frag: Optional[int] = None,
                         n_bc: Optional[int] = None,
                         tss_flank_window: Optional[int] = 1000,
                         tss_window: Optional[int] = 50,
                         tss_minimum_signal_window: Optional[int] = 100,
                         tss_rolling_window: Optional[int] = 10,
                         min_norm: Optional[int] = 0.2,
                         partition: Optional[int] = 1,
                         check_for_duplicates: Optional[bool] = True,
                         remove_duplicates: Optional[bool] = True):
    """"
    Wrapper function to compute QC statistics on several samples. For detailed instructions, please see the independent functions.

    Parameters
    ---
    fragments: str
            Path to fragments file.
    tss_annotation: pd.DataFrame or pr.PyRanges
            A data frame or pyRanges containing transcription start sites for each gene, with 'Chromosome', 'Start' and 'Strand' as columns (additional columns will be ignored).
    stats: list, optional
            A list with the statistics that have to be computed. Default: All ('barcode_rank_plot', 'duplicate_rate', 'insert_size_distribution', 'profile_tss', 'FRIP).
    label: str
            Sample label. Default: None.
    path_to_regions: str
            Path to regions file to use for FRIP.
    valid_bc: list, optional
            A list containing selected barcodes. This parameter is ignored if n_frag or n_bc are specified. Default: None.
    n_frag: int, optional
            Minimal number of fragments assigned to a barcode to be kept. Either n_frag or n_bc can be specified. Default: None.
    n_bc: int, optional
            Number of barcodes to select. Either n_frag or n_bc can be specified. Default: None.
    tss_window: int, optional
            Window around the TSS used to count fragments in the TSS when calculating the TSS enrichment per barcode. Default: 50 (+/- 50 bp).
    tss_flank_window: int, optional
            Flanking window around the TSS. Default: 1000 (+/- 1000 bp).
    tss_minimum_signal_window: int, optional
            Tail window use to normalize the TSS enrichment. Default: 100 (average signal in the 100bp in the extremes of the TSS window).
    tss_rolling_window: int, optional
            Rolling window used to smooth signal. Default: 10.
    min_norm: int, optional
            Minimum normalization score. If the average minimum signal value is below this value, this number is used to normalize the TSS signal. This approach penalizes cells with fewer reads.
    check_for_duplicates: bool, optional
            If no duplicate counts are provided per row in the fragments file, whether to collapse duplicates. Default: True.
    remove_duplicates: bool, optional
            Whether to remove duplicates. Default: True.

    Return
    ---
    pd.DataFrame or list and list
            A list with the barcode statistics for all samples (or a combined data frame with a column 'Sample' indicating the sample of origin) and a list of dictionaries with the sample-level profiles for each sample.
    """

    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')
    # Compute stats
    metrics = {}
    metadata_bc_dict = {}
    profile_data_dict = {}
    # Prepare fragments
    if isinstance(fragments, str):
        log.info('Reading ' + label)
        fragments_df = pr.read_bed(fragments).df
    else:
        fragments_df = fragments
    # Convert to category for memory efficiency
    fragments_df["Name"] = fragments_df["Name"].astype("category")
    # Check for duplicates
    if 'Score' not in fragments_df or all(fragments_df['Score'] == '.'):
        fragments_df = fragments_df[['Chromosome', 'Start', 'End', 'Name']]
        if check_for_duplicates:
            log.info("Collapsing duplicates")
            fragments_df = pd.concat([collapse_duplicates(fragments_df[fragments_df.Chromosome == x])
             for x in fragments_df.Chromosome.cat.categories.values])
        else:
            fragments_df['Score'] = 1
    else:
        fragments_df = fragments_df[[
            'Chromosome', 'Start', 'End', 'Name', 'Score']]
    fragments_df["Score"] = fragments_df["Score"].astype("int32")
    # Prepare valid barcodes
    if valid_bc is not None:
        if n_bc is not None or n_frag is not None:
            valid_bc = None
    # Rank plot
    if 'barcode_rank_plot' in stats:
        # Rank plot
        log.info('Computing barcode rank plot for ' + label)
        metrics['barcode_rank_plot'] = barcode_rank_plot(
            fragments=fragments_df,
            valid_bc=valid_bc,
            n_frag=n_frag,
            n_bc=n_bc,
            remove_duplicates=remove_duplicates,
            plot=False,
            return_bc=True,
            return_plot_data=True)
        if valid_bc is None:
            fragments_df = fragments_df[fragments_df.Name.isin(set(metrics['barcode_rank_plot']['valid_bc']))]

    gc.collect()
    # Duplicate rate
    if 'duplicate_rate' in stats:
        # Duplicate rate
        log.info('Computing duplicate rate plot for ' + label)
        metrics['duplicate_rate'] = duplicate_rate(fragments=fragments_df,
                                                   valid_bc=valid_bc,
                                                   plot=False,
                                                   return_plot_data=True)
    
    gc.collect()
    # Fragment size
    if 'insert_size_distribution' in stats:
        # Fragment size
        log.info('Computing insert size distribution for ' + label)
        metrics['insert_size_distribution'] = insert_size_distribution(
            fragments=fragments_df,
            valid_bc=valid_bc,
            remove_duplicates=remove_duplicates,
            plot=False,
            return_plot_data=True)
    fragments_df = pr.PyRanges(fragments_df)
    gc.collect()
    # TSS
    if 'profile_tss' in stats:
        # TSS
        log.info('Computing TSS profile for ' + label)
        profile_tss_metrics = profile_tss(
            fragments=fragments_df,
            annotation=tss_annotation,
            valid_bc=valid_bc,
            plot=False,
            n_cpu=1,
            partition = partition,
            flank_window=tss_flank_window,
            tss_window=tss_window,
            minimum_signal_window=tss_minimum_signal_window,
            rolling_window=tss_rolling_window,
            min_norm=min_norm,
            return_TSS_enrichment_per_barcode=True,
            return_TSS_coverage_matrix_per_barcode=True,
            return_plot_data=True)
        if profile_tss_metrics is not None:
            metrics['profile_tss'] = profile_tss_metrics
    gc.collect()
    # FRIP
    if 'frip' in stats:
        # FRIP
        log.info('Computing FRIP profile for ' + label)
        metrics['frip'] = frip(fragments=fragments_df,
                               path_to_regions=path_to_regions,
                               valid_bc=valid_bc,
                               remove_duplicates=remove_duplicates,
                               n_cpu=1,
                               plot=False,
                               return_plot_data=True)
    del fragments_df
    gc.collect()
    metadata_bc, profile_data = metrics2data(metrics)
    metadata_bc = metadata_bc.fillna(0)
    metadata_bc_dict = {label: metadata_bc}
    profile_data_dict = {label: profile_data}
    log.info('Sample ' + label + ' done!')

    return metadata_bc_dict, profile_data_dict


def plot_sample_metrics(profile_data_dict: Dict[str,
                                                pd.DataFrame],
                        profile_list: Optional[Union['barcode_rank_plot',
                                                     'duplicate_rate',
                                                     'insert_size_distribution',
                                                     'profile_tss',
                                                     'frip']] = ['barcode_rank_plot',
                                                                 'duplicate_rate',
                                                                 'insert_size_distribution',
                                                                 'profile_tss',
                                                                 'frip'],
                        remove_duplicates: Optional[bool] = True,
                        color: Optional[List[List[Union[str]]]] = None,
                        cmap: Optional[str] = None,
                        ncol: Optional[int] = 1,
                        figsize: Optional[Tuple[int,
                                                int]] = None,
                        insert_size_distriubtion_xlim: Optional[List[int]] = None,
                        legend_outside: Optional[bool] = False,
                        duplicate_rate_as_hexbin: Optional[bool] = False,
                        plot: Optional[bool] = True,
                        save: Optional[str] = None):
    """
    Plot sample-level profiles given a list of sample-level profiles dictionaries (one per sample).

    Parameters
    ---------
    profile_data_dict: list of dict
            Dictionary of dictionaries with keys 'barcode_rank_plot', 'insert_size_distribution', 'profile_tss' and/or 'FRIP', containing the sample-level profiles for each metric. This dictionary is an output of `metrics2data`.
    profile_list: list, optional
            List of the sample-level profiles to plot. Default: All.
    remove_duplicates: optional, bool
            Whether duplicated should not be considered for the barcode rank plot. Default: True
    color: list, optional
            List containing the colors to each for sample. When using barcode_rank_plot, at least two colors must be provided per sample. Default: None.
    cmap: list, optional
            Color map to color the plot by density for the duplicate rate plot. Default: None
    ncol: int, optional
            Number of columns for grid plot. If 1 each plot will be drawn independently, while the number of rows is automatically adjusted. Default: 1.
    figsize: tuple, optional
            Figure size. If drawing each plot independently it corresponds to the size of each plot, if using grid plotting it will correspond to the total size of the figure.
    insert_size_distriubtion_xlim: list, optional
            A list with two numbers that indicate the x axis limits. Default: None
    plot: bool, optional
            Whether the plots should be returned to the console. Default: True
    save: str, optional
            Output file to save plot. Default: None.
    """

    plot_sample_metrics_generator_obj = plot_sample_metrics_generator(
        profile_data_dict=profile_data_dict,
        profile_list=profile_list,
        remove_duplicates=remove_duplicates,
        color=color,
        cmap=cmap,
        insert_size_distriubtion_xlim=insert_size_distriubtion_xlim,
        legend_outside=legend_outside,
        duplicate_rate_as_hexbin=duplicate_rate_as_hexbin)
    label_list = list(profile_data_dict.keys())
    if ('duplicate_rate' in profile_list) & (not isinstance(
            profile_data_dict[label_list[0]], pd.DataFrame)):
        n_plots = len(profile_list) - 1 + len(profile_data_dict)
    else:
        n_plots = len(profile_list)

    multiplot_from_generator(g=plot_sample_metrics_generator_obj,
                             num_columns=ncol,
                             n_plots=n_plots,
                             figsize=figsize,
                             plot=plot,
                             save=save)


def plot_sample_metrics_generator(profile_data_dict: Dict[str,
                                                          pd.DataFrame],
                                  profile_list: Optional[Union['barcode_rank_plot',
                                                               'duplicate_rate',
                                                               'insert_size_distribution',
                                                               'profile_tss',
                                                               'frip']] = ['barcode_rank_plot',
                                                                           'duplicate_rate',
                                                                           'insert_size_distribution',
                                                                           'profile_tss',
                                                                           'frip'],
                                  remove_duplicates: Optional[bool] = True,
                                  color: Optional[List[List[Union[str]]]] = None,
                                  cmap: Optional[str] = None,
                                  insert_size_distriubtion_xlim: Optional[List[int]] = None,
                                  legend_outside: Optional[bool] = False,
                                  duplicate_rate_as_hexbin: Optional[bool] = False):
    """
    Plot sample-level profiles given a list of sample-level profiles dictionaries (one per sample). This function creates the generator to pass to the multiplotting function.

    Parameters
    ---------
    profile_data_dict: list of dict
        Dictionary of dictionaries with keys 'barcode_rank_plot', 'insert_size_distribution', 'profile_tss' and/or 'FRIP', containing the sample-level profiles for each metric. This dictionary is an output of `metrics2data`.
    profile_list: list, optional
        List of the sample-level profiles to plot. Default: All.
    remove_duplicates: optional, bool
        Whether duplicated should not be considered for the barcode rank plot. Default: True
    color: list, optional
        List containing the colors to each for sample. When using barcode_rank_plot, at least two colors must be provided per sample. Default: None.
    cmap: list, optional
        Color map to color the plot by density for the duplicate rate plot.
    insert_size_distriubtion_xlim: list, optional
        A list with two numbers that indicate the x axis limits. Default: None
    duplicate_rate_as_hexbin: bool, optional
        A boolean indicating if the duplicate rate should be plotted as an hexagonal binning plot. The quality of the plot will be reduced, but is a faster alternative
        when dealing with a large number of points. Default: False.
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    # Prepare labels
    label_list = list(profile_data_dict.keys())

    # Check if it is only one sample, without sample name as entry in dict
    if isinstance(profile_data_dict[label_list[0]], pd.DataFrame):
        profile_data_dict = {'Sample': profile_data_dict}
        label_list = list(profile_data_dict.keys())

    # Rank plot
    if 'barcode_rank_plot' in profile_list:
        yield

        legend_labels = []
        for i in range(len(profile_data_dict)):
            if 'barcode_rank_plot' not in profile_data_dict[label_list[i]]:
                log.error(
                    'barcode_rank_plot is not included in the profiles dictionary')
            plot_data = profile_data_dict[label_list[i]]['barcode_rank_plot']

            sel = np.ma.masked_where(
                plot_data['Selected'],
                plot_data['Barcode_rank'])
            nosel = np.ma.masked_where(
                plot_data['Selected'] == False,
                plot_data['Barcode_rank'])

            if (len(label_list) > 1) | (label_list[0] != ''):
                legend_labels.append(f'{label_list[i]} ({sum(sel.mask)})')
            else:
                legend_labels.append(f'Selected BC: {sum(sel.mask)}')
                legend_labels.append(
                    f'Non-selected BC: {plot_data.shape[0]-sum(sel.mask)}')

            if not remove_duplicates:
                NF = plot_data['Total_nr_frag']
            else:
                NF = plot_data['Unique_nr_frag']

            if color is None:
                plt.plot(nosel, NF)
                if (len(label_list) > 1) | (label_list[0] != ''):
                    plt.plot(sel, NF, 'grey', label='_nolegend_')
                else:
                    plt.plot(sel, NF, 'grey')
            else:
                if len(color[i]) < 2:
                    plt.plot(nosel, NF, color[i])
                    if (len(label_list) > 1) | (label_list[0] != ''):
                        plt.plot(sel, NF, 'grey', label='_nolegend_')
                    else:
                        plt.plot(sel, NF, 'grey')
                else:
                    plt.plot(nosel, NF, color[i][0])
                    if (len(label_list) > 1) | (label_list[0] != ''):
                        plt.plot(sel, NF, color[i][1], label='_nolegend_')
                    else:
                        plt.plot(sel, NF, color[i][1])

            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Barcode Rank", fontsize=10)
            plt.ylabel("Number of fragments in regions", fontsize=10)

            if legend_outside:
                plt.legend(
                    legend_labels, bbox_to_anchor=(
                        1.04, 1), loc="upper left")
            else:
                plt.legend(legend_labels)

    # Fragment size
    if 'insert_size_distribution' in profile_list:
        yield

        for i in range(len(profile_data_dict)):
            if 'insert_size_distribution' not in profile_data_dict[label_list[i]]:
                log.error(
                    'insert_size_distribution is not included in the profiles dictionary')
            plot_data = profile_data_dict[label_list[i]
                                          ]['insert_size_distribution']
            if color is not None:
                if len(color[i]) > 2:
                    selected_color = color[i][0]
                else:
                    selected_color = None
            else:
                selected_color = None

            plt.plot(
                plot_data['Width'],
                plot_data['Ratio_frag'],
                color=selected_color)
            plt.xlabel("Fragment size", fontsize=10)
            plt.ylabel("Fragments ratio", fontsize=10)
            if insert_size_distriubtion_xlim is not None:
                plt.xlim(
                    insert_size_distriubtion_xlim[0],
                    insert_size_distriubtion_xlim[1])
        if (len(label_list) > 1) | (label_list[0] != ''):
            if legend_outside:
                plt.legend(
                    label_list, bbox_to_anchor=(
                        1.04, 1), loc="upper left")
            else:
                plt.legend(label_list)

    # TSS
    if 'profile_tss' in profile_list:
        yield

        for i in range(len(profile_data_dict)):
            if 'profile_tss' not in profile_data_dict[label_list[i]]:
                log.error(
                    'profile_tss is not included in the profiles dictionary')
            plot_data = profile_data_dict[label_list[i]]['profile_tss']
            if color is not None:
                if len(color[i]) > 2:
                    selected_color = color[i][0]
                else:
                    selected_color = None
            else:
                selected_color = None

            plt.plot(
                plot_data['Position'],
                plot_data['TSS_enrichment'],
                color=selected_color)

            plt.xlim(min(plot_data['Position']), max(plot_data['Position']))
            plt.xlabel("Position from TSS", fontsize=10)
            plt.ylabel("Normalized enrichment", fontsize=10)
        if (len(label_list) > 1) | (label_list[0] != ''):
            if legend_outside:
                plt.legend(
                    label_list, bbox_to_anchor=(
                        1.04, 1), loc="upper left")
            else:
                plt.legend(label_list)

    # FRIP
    if 'frip' in profile_list:
        yield

        for i in range(len(profile_data_dict)):
            if 'frip' not in profile_data_dict[label_list[i]]:
                log.error('frip is not included in the profiles dictionary')
            plot_data = profile_data_dict[label_list[i]]['frip']
            if color is not None:
                if len(color[i]) > 2:
                    selected_color = color[i][0]
                else:
                    selected_color = None
            else:
                selected_color = None

            sns.distplot(
                plot_data['FRIP'],
                hist=False,
                kde=True,
                color=selected_color,
                label=label_list[i])

        if (len(label_list) > 1) | (label_list[0] != ''):
            if legend_outside:
                plt.legend(
                    label_list, bbox_to_anchor=(
                        1.04, 1), loc="upper left")
            else:
                plt.legend(label_list)

    # Duplicate rate
    if 'duplicate_rate' in profile_list:
        for i in range(len(profile_data_dict)):
            yield
            if 'duplicate_rate' not in profile_data_dict[label_list[i]]:
                log.error(
                    'duplicate_rate is not included in the profiles dictionary')
            plot_data = profile_data_dict[label_list[i]]['duplicate_rate']
            x = plot_data['Unique_nr_frag']
            y = plot_data['Dupl_rate']

            if not duplicate_rate_as_hexbin:
                try:
                    xy = np.vstack([np.log(x), y])
                    z = gaussian_kde(xy)(xy)
                    idx = z.argsort()
                    x, y, z = x[idx], y[idx], z[idx]
                    plt.scatter(x, y, c=z, s=10, edgecolor=None, cmap=cmap)
                except BaseException:
                    log.info('All fragments are unique')
                    plt.scatter(x, y, s=10, edgecolor=None, cmap=cmap)
            else:
                plt.hexbin(
                    x,
                    y,
                    edgecolor=None,
                    cmap=cmap,
                    gridsize=100,
                    xscale='log',
                    bins='log',
                    mincnt=1)
            plt.title(label_list[i])
            plt.ylim(0, 1)
            plt.xscale("log")
            plt.xlabel("Number of (unique) fragments", fontsize=10)
            plt.ylabel("Duplication rate", fontsize=10)
            plt.colorbar().set_label('Density')
            plt.xlim(min(x), max(x))


def plot_barcode_profile_tss(tss_profile_per_barcode: pd.DataFrame,
                             barcode: Union[List[str], str],
                             rolling_window: Optional[int] = 10,
                             plot: Optional[bool] = True,
                             color: Optional[List[List[Union[str]]]] = None,
                             save: Optional[str] = None):
    """"
    Plot TSS profile per barcode.

    Parameters
    ---
    tss_profile_per_barcode: pd.DataFrame
            A :class:`matrix` with the normalized enrichment scores in each position for each barcode, obtained when setting `return_TSS_enrichment_per_barcode=True` in the function `profile_tss`.
    barcode: str or list, optional
            Barcode or list of barcodes to plot.
    rolling_window: int, optional
            Rolling window used to smooth signal. Default: 10.
    plot: bool, optional
            Whether to return the plot to the console. Default: True.
    color: str, optional
            Line color for the plot. Default: None.
    save: str, optional
            Output file to save plot. Default: None.
    """

    fig = plt.figure()

    if save is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)

    if isinstance(barcode, str):
        plot_data_df = pd.DataFrame(tss_profile_per_barcode.loc[barcode, :]).T
        i_iter = [0]
    else:
        plot_data_df = tss_profile_per_barcode.loc[barcode, :]
        i_iter = range(plot_data_df.shape[0])

    plot_data_df = plot_data_df.rolling(
        window=100, min_periods=0, axis=1).mean()

    for i in i_iter:
        plot_data = plot_data_df.iloc[i, :]
        if color is not None:
            if len(color[i]) > 2:
                selected_color = color[i][0]
            else:
                selected_color = None
        else:
            selected_color = None
        plt.plot(range(-(int(plot_data_df.shape[1] / 2) + 1), int(
            plot_data_df.shape[1] / 2)), plot_data, color=selected_color)
        plt.xlim(-(int(plot_data_df.shape[1] / 2) + 1),
                 int(plot_data_df.shape[1] / 2))
        plt.xlabel("Position from TSS", fontsize=10)
        plt.ylabel("Normalized enrichment", fontsize=10)
    plt.legend(barcode)

    if save is not None:
        pdf.savefig(fig, bbox_inches='tight')
        pdf.close()

    if plot is not False:
        plt.show()
    else:
        plt.close(fig)


def plot_barcode_metrics_per_group(input_metrics: Dict,
                                   var_x: str,
                                   var_y: Optional[str] = None,
                                   min_x: Optional[int] = None,
                                   max_x: Optional[int] = None,
                                   min_y: Optional[int] = None,
                                   max_y: Optional[int] = None,
                                   color: Optional[str] = '#440154FF',
                                   cmap: Optional[str] = 'viridis',
                                   as_density: Optional[bool] = False,
                                   add_hist: Optional[bool] = True,
                                   n_bins: Optional[int] = 100,
                                   plot_as_hexbin: Optional[bool] = False,
                                   plot: Optional[bool] = True,
                                   save: Optional[str] = None):
    """"
    Plot barcode metrics and filter based on used-provided thresholds.

    Parameters
    ---
    input_metrics: dictionary
            A dictionary with group labels as keys and barcode metrics per sample as values.
    var_x: str
            Metric to plot.
    var_group: str, optional
            Variable to divide the plot by groups. Default: None.
    var_y: str, optional
            A second metric to plot in combination with `var_x`. When provided, the function returns a 2D plot with `var_x` and `var_y` as axes, if not provided the function returns and histogram or density plot for `var_x`. Default: None.
    min_x: float, optional
            Minimum value on `var_x` to keep the barcode/cell. Default: None.
    max_x: float, optional
            Maximum value on `var_x` to keep the barcode/cell. Default: None.
    min_y: float, optional
            Minimum value on `var_y` to keep the barcode/cell. Default: None.
    max_y: float, optional
            Maximum value on `var_y` to keep the barcode/cell. Default: None.
    color: str, optional
            Color to use on histograms and/or density plots. Default: None.
    cmap: str, optional
            Color map to color 2D dot plots by density. Default: None.
    as_density: bool, optional
            Whether to plot variables as density plots rather than histograms. Default: True.
    add_hist: bool, optional
            Whether to show the histogram together with the density plots when `as_density=True`. Default: True.
    n_bins: int, optional
            Number of bins to use when plotting the variable histogram. Default: 100.
    plot_as_hexbin: bool, optional
            A boolean indicating if the data should be plotted as an hexagonal binning plot. The quality of the plot will be reduced, but is a faster alternative
            when dealing with a large number of points. Default: False.
    plot: bool, optional
            Whether the plots should be returned to the console. Default: True.
    save: bool, optional
            Path to save plots as a file. Default: None.

    Return
    ---
    dict or list
            If var_group is provided or the input is a dictionary, the function returns a dictionary with the selected cells per group based on user provided thresholds; otherwise a list with the selected cells.
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    selected_cells = {}
    fig_dict = {}
    if save is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)
        if not os.path.exists(os.path.dirname(save)):
            if os.path.dirname(save) != '':
                os.makedirs(os.path.dirname(save))
    for key in input_metrics.keys():
        x = input_metrics[key][var_x]
        if var_y in (set(input_metrics[key].columns)):
            # Take cell data
            y = input_metrics[key][var_y]
            # Plot xy
            plt.close()
            fig = plt.figure(figsize=(5, 5))
            fig.add_axes([0, 0, 0.8, 0.8])
            if not plot_as_hexbin:
                # Color by density
                try:
                    xy = np.vstack([x, y])
                    z = gaussian_kde(xy)(xy)
                    idx = z.argsort()
                    x, y, z = x[idx], y[idx], z[idx]
                    plt.scatter(x, y, c=z, s=10, edgecolor=None, cmap=cmap)
                except BaseException:
                    plt.scatter(x, y, s=10, edgecolor=None, cmap=cmap)
            else:
                plt.hexbin(
                    x,
                    y,
                    edgecolor=None,
                    cmap=cmap,
                    gridsize=100,
                    mincnt=0.1)
            plt.xlabel(var_x, fontsize=10)
            plt.ylabel(var_y, fontsize=10)
            plt.xlim(min(x), max(x))
            plt.ylim(min(y), max(y))

            if len(input_metrics) > 1:
                plt.title(key, y=-0.2, fontsize=10, fontweight="bold")
            # Add limits
            if min_x is not None:
                plt.axvline(x=min_x, color='skyblue', linestyle='--')
                input_metrics[key] = input_metrics[key].loc[input_metrics[key]
                                                            [var_x] > min_x, :]
            if max_x is not None:
                plt.axvline(x=max_x, color='tomato', linestyle='--')
                input_metrics[key] = input_metrics[key].loc[input_metrics[key]
                                                            [var_x] < max_x, :]
            if min_y is not None:
                plt.axhline(y=min_y, color='skyblue', linestyle='--')
                input_metrics[key] = input_metrics[key].loc[input_metrics[key]
                                                            [var_y] > min_y, :]
            if max_y is not None:
                plt.axhline(y=max_y, color='tomato', linestyle='--')
                input_metrics[key] = input_metrics[key].loc[input_metrics[key]
                                                            [var_y] < max_y, :]

            # first barplot on axis
            fig.add_axes([0, 0.8, 0.8, 0.2])
            if as_density:
                sns.distplot(
                    x,
                    hist=add_hist,
                    kde=True,
                    color=color,
                    kde_kws={
                        'shade': True},
                    bins=n_bins)
            else:
                plt.hist(x, bins=n_bins, color=color)
            plt.xlim(min(x), max(x))
            plt.axis('off')
            # second barplot on axis
            fig.add_axes([0.8, 0, 0.2, 0.8])
            if as_density:
                sns.distplot(
                    y,
                    hist=add_hist,
                    kde=True,
                    color=color,
                    kde_kws={
                        'shade': True},
                    vertical=True,
                    bins=n_bins)
            else:
                plt.hist(y, bins=n_bins, orientation='horizontal', color=color)
            plt.ylim(min(y), max(y))
            plt.axis('off')

            if save is not None:
                pdf.savefig(fig, bbox_inches='tight')

            if plot is not False:
                plt.show()
            else:
                plt.close(fig)

        else:
            plt.close()
            fig = plt.figure()
            if isinstance(var_y, str):
                log.info(
                    'The given var_y is not a column in cistopic_obj.cell_data')

            # first barplot on axis
            if as_density:
                sns.distplot(
                    x,
                    hist=add_hist,
                    kde=True,
                    color=color,
                    kde_kws={
                        'shade': True},
                    bins=n_bins)
            else:
                plt.hist(x, bins=n_bins, color=color)
                plt.xlim(min(x), max(x))

            if len(input_metrics) > 1:
                plt.legend([key])

            # Add limits
            if min_x is not None:
                plt.axvline(x=min_x, color='skyblue', linestyle='--')
                input_metrics[key] = input_metrics[key].loc[input_metrics[key]
                                                            [var_x] > min_x, :]
                if max_x is not None:
                    plt.axvline(x=max_x, color='tomato', linestyle='--')
                    input_metrics[key] = input_metrics[key].loc[input_metrics[key]
                                                                [var_x] < max_x, :]

            if save is not None:
                pdf.savefig(fig, bbox_inches='tight')

            if plot is not False:
                plt.show()
            else:
                plt.close(fig)

        selected_cells[key] = input_metrics[key].index.to_list()
        fig_dict[key] = fig

    if save is not None:
        pdf.close()

    return fig_dict, selected_cells


def plot_barcode_metrics(input_metrics: Union[Dict,
                                              pd.DataFrame,
                                              'CistopicObject'],
                         var_x: str,
                         var_group: Optional[str] = None,
                         var_y: Optional[str] = None,
                         min_x: Optional[float] = None,
                         max_x: Optional[float] = None,
                         min_y: Optional[float] = None,
                         max_y: Optional[float] = None,
                         color: Optional[str] = '#440154FF',
                         cmap: Optional[str] = 'viridis',
                         as_density: Optional[bool] = True,
                         add_hist: Optional[bool] = True,
                         n_bins: Optional[int] = 100,
                         plot_as_hexbin: Optional[bool] = False,
                         combine_samples_ridgeline: Optional[bool] = True,
                         overlap_ridgeline: Optional[float] = .85,
                         plot: Optional[bool] = True,
                         save: Optional[str] = None,
                         return_cells: Optional[bool] = True,
                         return_fig: Optional[bool] = False):
    """"
    Plot barcode metrics and filter based on used-provided thresholds.

    Parameters
    ---
    input_metrics: dictionary, pd.DataFrame or CistopicObject
    A dictionary with group labels as keys and barcode metrics per sample as values, a dataframe with barcode metrics (for one or more samples) or a cisTopicObjbect with metrics in `class::CistopicObject.cell_data`.
    var_x: str
            Metric to plot.
    var_group: str, optional
            Variable to divide the plot by groups. Default: None.
    var_y: str, optional
            A second metric to plot in combination with `var_x`. When provided, the function returns a 2D plot with `var_x` and `var_y` as axes, if not provided the function returns and histogram or density plot for `var_x`. Default: None.
    min_x: float, optional
            Minimum value on `var_x` to keep the barcode/cell. Default: None.
    max_x: float, optional
            Maximum value on `var_x` to keep the barcode/cell. Default: None.
    min_y: float, optional
            Minimum value on `var_y` to keep the barcode/cell. Default: None.
    max_y: float, optional
            Maximum value on `var_y` to keep the barcode/cell. Default: None.
    color: str, optional
            Color to use on histograms and/or density plots. Default: None.
    cmap: str, optional
            Color map to color 2D dot plots by density. Default: None.
    as_density: bool, optional
            Whether to plot variables as density plots rather than histograms. Default: True.
    add_hist: bool, optional
            Whether to show the histogram together with the density plots when `as_density=True`. Default: True.
    n_bins: int, optional
            Number of bins to use when plotting the variable histogram. Default: 100.
    plot_as_hexbin: bool, optional
            A boolean indicating if the data should be plotted as an hexagonal binning plot. The quality of the plot will be reduced, but is a faster alternative
            when dealing with a large number of points. Default: False.
    combine_samples_ridgeline: bool, optional
            When a group variable is provided and only one metric is given, the distribution of the metric can be plotted in all groups using a ridgeline plot. If False, an histogram per sample will be returned. Default: True.
    overlap_ridgeline: float, optional
            Overlap between the ridgeline plot tracks. Default=.85
    plot: bool, optional
            Whether the plots should be returned to the console. Default: True.
    save: bool, optional
            Path to save plots as a file. Default: None.
    return_cells: bool, optional
            Whether to return selected cells based on user-given thresholds. Default: True.
    return_fig: bool, optional
            Whether to return the plot figure; if several samples it will return a dictionary with the figures per sample. Default: False.

    Return
    ---
    dict or list
            If var_group is provided or the input is a dictionary, the function returns a dictionary with the selected cells per group based on user provided thresholds; otherwise a list with the selected cells.
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    # Take cell data
    if isinstance(input_metrics, CistopicObject):
        input_metrics = input_metrics.cell_data
    if isinstance(input_metrics, dict):
        input_metrics = merge_metadata(input_metrics)
        var_group = 'Sample'

    # If there is only one sample
    if var_group is None:
        input_metrics = {'Sample': input_metrics}
        fig_dict, selected_cells = plot_barcode_metrics_per_group(
            input_metrics, var_x, var_y, min_x, max_x, min_y, max_y, color, cmap, as_density, add_hist, n_bins, plot_as_hexbin, plot, save)
    else:
        input_metrics_dict = {x: input_metrics[input_metrics.loc[:, var_group] == x] for x in list(
            set(input_metrics.loc[:, var_group]))}
        if combine_samples_ridgeline is True and var_y is None:
            selected_cells = plot_barcode_metrics_per_group(
                input_metrics_dict,
                var_x,
                var_y,
                min_x,
                max_x,
                min_y,
                max_y,
                color,
                cmap,
                as_density,
                add_hist,
                n_bins,
                plot_as_hexbin,
                plot=False,
                save=None)
            if save is not None:
                if not os.path.exists(os.path.dirname(save)):
                    os.makedirs(os.path.dirname(save))
                pdf = matplotlib.backends.backend_pdf.PdfPages(save)
            plt.close()
            fig = plt.figure()
            grouped = [(v, d.loc[:, var_x].dropna().values)
                       for v, d in input_metrics.groupby(var_group)]
            sample, data = zip(*grouped)
            if color is None:
                color = 'skyblue'
            ridgeline(
                data,
                labels=sample,
                overlap=overlap_ridgeline,
                fill=color)
            plt.xlabel(var_x)
            plt.grid(zorder=0)
            # Add limits
            if min_x is not None:
                plt.axvline(x=min_x, color='skyblue', linestyle='--')
            if max_x is not None:
                plt.axvline(x=max_x, color='tomato', linestyle='--')
            if min_y is not None:
                plt.axhline(y=min_y, color='skyblue', linestyle='--')
            if max_y is not None:
                plt.axhline(y=max_y, color='tomato', linestyle='--')

            if save is not None:
                pdf.savefig(fig, bbox_inches='tight')
                pdf.close()

            if plot is not False:
                plt.show()
            else:
                plt.close(fig)
            fig_dict = fig

        else:
            fig_dict, selected_cells = plot_barcode_metrics_per_group(
                input_metrics_dict, var_x, var_y, min_x, max_x, min_y, max_y, color, cmap, as_density, add_hist, n_bins, plot_as_hexbin, plot, save)

    if return_cells:
        if len(selected_cells) == 1:
            selected_cells = selected_cells[list(selected_cells.keys())[0]]
        if return_fig:
            if len(fig_dict) == 1:
                fig_dict = fig_dict[list(fig_dict.keys())[0]]
            return fig_dict, selected_cells
        else:
            return selected_cells
    else:
        if return_fig:
            if len(fig_dict) == 1:
                fig_dict = fig_dict[list(fig_dict.keys())[0]]
            return fig_dict


def merge_metadata(metadata_bc_dict: Dict):
    """
    Merge barcode-level statistics from different samples.

    Parameters
    ---
    metadata_bc_dict: dict
            Dictionary containing `class::pd.DataFrame` with the barcode-level statistics for each sample

    Return
    ---
    pd.DataFrame
            A data frame containing the combined barcode statistics with an additional column called sample.
    """

    for key in metadata_bc_dict.keys():
        metadata_bc_dict[key]['Sample'] = [
            key] * metadata_bc_dict[key].shape[0]

    metadata_bc_list = [metadata_bc_dict[key]
                        for key in metadata_bc_dict.keys()]
    metadata_bc_combined = pd.concat(metadata_bc_list, axis=0, sort=False)
    return metadata_bc_combined


def ridgeline(data: List,
              overlap: Optional[float] = 0,
              fill: Optional[str] = None,
              labels: Optional[List] = None,
              n_points: Optional[int] = 150):
    """
    Create a standard ridgeline plot.

    Parameters
    ---
    data: list of lists.
            Data per group to plot
    overlap: float, optional
            Overlap between distributions. Default: 0.
    fill: str, optional
            Color to fill the distributions. Default: None.
    labels: List
            Values to place on the y axis to describe the distributions.
    n_points: int, optional
            Number of points to evaluate each distribution function. Default: 150/
    """
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    xx = np.linspace(np.min(np.concatenate(data)),
                     np.max(np.concatenate(data)), n_points)
    curves = []
    ys = []
    for i, d in enumerate(data):
        pdf = gaussian_kde(d)
        y = i * (1.0 - overlap)
        ys.append(y)
        curve = pdf(xx)
        if fill:
            plt.fill_between(xx, np.ones(n_points) * y,
                             curve + y, zorder=len(data) - i + 1, color=fill)
            plt.plot(xx, curve + y, c='k', zorder=len(data) - i + 1)
    if labels:
        plt.yticks(ys, labels)
