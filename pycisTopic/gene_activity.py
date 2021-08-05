import logging
import numpy as np
import pandas as pd
import pyranges as pr
import scipy.sparse as sparse
import sys
from typing import List, Optional, Union

from .diff_features import *
from .utils import *

pd.options.mode.chained_assignment = None


def get_gene_activity(imputed_acc_object: 'CistopicImputedFeatures',
                      pr_annot: pr.PyRanges,
                      chromsizes: pr.PyRanges,
                      use_gene_boundaries: Optional[bool] = True,
                      upstream: Optional[List[int]] = [1000, 100000],
                      downstream: Optional[List[int]] = [1000, 100000],
                      distance_weight: Optional[bool] = True,
                      decay_rate: Optional[float] = 1,
                      extend_gene_body_upstream: Optional[int] = 5000,
                      extend_gene_body_downstream: Optional[int] = 0,
                      gene_size_weight: Optional[bool] = False,
                      gene_size_scale_factor: Optional[Union[str, int]] = 'median',
                      remove_promoters: Optional[bool] = False,
                      scale_factor: Optional[float] = 1,
                      average_scores: Optional[bool] = True,
                      extend_tss: Optional[List[int]] = [10, 10],
                      return_weights: Optional[bool] = True,
                      gini_weight: Optional[bool] = True,
                      project: Optional[str] = 'Gene_activity'):
    """
    Infer gene activity.

    Parameters
    ---------
    imputed_features_obj: :class:`CistopicImputedFeatures`
        A cisTopic imputation data object.
    pr_annot: pr.PyRanges
        A :class:`pr.PyRanges` containing gene annotation, including Chromosome, Start, End, Strand (as '+' and '-'), Gene name
        and Transcription Start Site.
    chromsizes: pr.PyRanges
        A :class:`pr.PyRanges` containing size of each chromosome, containing 'Chromosome', 'Start' and 'End' columns.
    use_gene_boundaries: bool, optional
        Whether to use the whole search space or stop when encountering another gene. Default: True
    upstream: List, optional
        Search space upstream. The minimum (first position) means that even if there is a gene right next to it these
        bp will be taken. The second position indicates the maximum distance. Default: [1000,100000]
    downstream: List, optional
        Search space downstream. The minimum (first position) means that even if there is a gene right next to it these
        bp will be taken. The second position indicates the maximum distance. Default: [1000,100000]
    distance_weight: bool, optional
        Whether to add a distance weight (an exponential function, the weight will decrease with distance). Default: True
    decay_rate: float, optional
        Exponent for the distance exponential funciton (the higher the faster will be the decrease). Default: 1
    extend_gene_body_upstream: int, optional
        Number of bp upstream immune to the distance weight (their value will be maximum for this weight). Default: 5000
    extend_gene_body_downstream: int, optional
        Number of bp downstream immune to the distance weight (their value will be maximum for this weight). Default: 0
    gene_size_weight: bool, optional
        Whether to add a weights based on th length of the gene. Default: False
    gene_size_scale_factor: str or int, optional
        Dividend to calculate the gene size weigth. Default is the median value of all genes in the genome.
    remove_promoters: bool, optional
        Whether to ignore promoters when computing gene activity. Default: False
    average_scores: bool, optional
        Whether to divide by the total number of region assigned to a gene when calculating the gene activity
        score. Default: True
    scale_factor: int, optional
        Value to multiply for the final gene activity matrix. Default: 1
    extend_tss: list, optional
        Space around the TSS consider as promoter. Default: [10,10]
    return_weights: bool, optional
        Whether to return the final weight values. Default: True
    gini_weight: bool, optional
        Whether to add a gini index weigth. The more unique the region is, the higher this weight will be.
        Default: True
    project: str, optional;
        Project name for the :class:`CistopicImputedFeatures` with the gene activity

    Return
    ------
    CistopicImputedFeatures
    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger('cisTopic')
    # Calculate region weights
    region_weights_df = region_weights(imputed_acc_object,
                                       pr_annot,
                                       chromsizes,
                                       use_gene_boundaries,
                                       upstream,
                                       downstream,
                                       distance_weight,
                                       decay_rate,
                                       extend_gene_body_upstream,
                                       extend_gene_body_downstream,
                                       gene_size_weight,
                                       gene_size_scale_factor,
                                       remove_promoters,
                                       extend_tss,
                                       gini_weight)
    # Weigthed aggregation
    region_weights_df.loc[:, 'Index'] = get_position_index(
        region_weights_df.Name, imputed_acc_object.feature_names)
    region_weights_df.loc[:, 'Weight'] = region_weights_df.Gene_size_weight * \
                                         region_weights_df.Distance_weight * region_weights_df.Gini_weight
    region_weights_df = region_weights_df.loc[region_weights_df.Weight > 0]
    genes = list(set(region_weights_df.Gene))
    log.info('Getting gene activity scores')
    gene_act = np.array([weighted_aggregation(imputed_acc_object.mtx,
                                              region_weights_df[region_weights_df.Gene == gene],
                                              average_scores) for gene in genes])
    log.info('Creating imputed features object')
    if scale_factor != 1:
        log.info('Scaling matrix')
        gene_act = gene_act * scale_factor
        gene_act = gene_act.round()
        gene_act = sparse.csr_matrix(gene_act)
        keep_features_index = non_zero_rows(gene_act)
        gene_act = gene_act[keep_features_index, ]
        genes = subset_list(genes, keep_features_index)
    gene_act = CistopicImputedFeatures(
        gene_act, genes, imputed_acc_object.cell_names, project)
    if return_weights:
        return gene_act, region_weights_df
    else:
        return gene_act


def weighted_aggregation(imputed_acc_obj_mtx: sparse.csr_matrix,
                         region_weights_df_per_gene: pd.DataFrame,
                         average_scores: bool):
    """
    Weighted aggregation of region probabilities into gene activity

    Parameters
    ---------
    imputed_acc_obj_mtx: sparse.csr_matrix
        A sparse matrix with regions as rows and cells as columns.
    region_weights_df_per_gene: pd.DataFrame
        A data frame with region index (from the sparse matrix) for the gene
    average_score: bool
        Whether final values should be divided by the total number of regions aggregated
    """
    if average_scores:
        gene_act = imputed_acc_obj_mtx[region_weights_df_per_gene.Index, :].T.dot(
            (region_weights_df_per_gene.Weight.values)) / region_weights_df_per_gene.shape[0]
    else:
        gene_act = imputed_acc_obj_mtx[region_weights_df_per_gene.Index, :].T.dot(
            (region_weights_df_per_gene.Weight.values))
    return gene_act


def region_weights(imputed_acc_object,
                   pr_annot,
                   chromsizes,
                   use_gene_boundaries=True,
                   upstream=[1000, 100000],
                   downstream=[1000, 100000],
                   distance_weight=True,
                   decay_rate=1,
                   extend_gene_body_upstream=5000,
                   extend_gene_body_downstream=0,
                   gene_size_weight=True,
                   gene_size_scale_factor='median',
                   remove_promoters=True,
                   extend_tss=[10, 10],
                   gini_weight=True):
    """
    Calculate region weights.

    Parameters
    ---------
    imputed_features_obj: :class:`CistopicImputedFeatures`
        A cisTopic imputation data object.
    pr_annot: pr.PyRanges
        A :class:`pr.PyRanges` containing gene annotation, including Chromosome, Start, End, Strand (as '+' and '-'), Gene name
        and Transcription Start Site.
    chromsizes: pr.PyRanges
        A :class:`pr.PyRanges` containing size of each chromosome, containing 'Chromosome', 'Start' and 'End' columns.
    use_gene_boundaries: bool, optional
        Whether to use the whole search space or stop when encountering another gene. Default: True
    upstream: List, optional
        Search space upstream. The minimum (first position) means that even if there is a gene right next to it these
        bp will be taken. The second position indicates the maximum distance. Default: [1000,100000]
    downstream: List, optional
        Search space downstream. The minimum (first position) means that even if there is a gene right next to it these
        bp will be taken. The second position indicates the maximum distance. Default: [1000,100000]
    distance_weight: bool, optional
        Whether to add a distance weight (an exponential function, the weight will decrease with distance). Default: True
    decay_rate: float, optional
        Exponent for the distance exponential funciton (the higher the faster will be the decrease). Default: 1
    extend_gene_body_upstream: int, optional
        Number of bp upstream immune to the distance weight (their value will be maximum for this weight). Default: 5000
    extend_gene_body_downstream: int, optional
        Number of bp downstream immune to the distance weight (their value will be maximum for this weight). Default: 0
    gene_size_weight: bool, optional
        Whether to add a weights based on th length of the gene. Default: False
    gene_size_scale_factor: str or int, optional
        Dividend to calculate the gene size weigth. Default is the median value of all genes in the genome.
    remove_promoters: bool, optional
        Whether to ignore promoters when computing gene activity. Default: False
    extend_tss: list, optional
        Space around the TSS consider as promoter. Default: [10,10]
    gini_weight: bool, optional
        Whether to add a gini index weigth. The more unique the region is, the higher this weight will be.
        Default: True


    Return
    ------
    pd.DataFrame
        A data frame for with weights for each region and the gene they are linked to.
    """
    # Create cisTopic logger
    level = logging.INFO
    log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    log = logging.getLogger('cisTopic')
    # Check up on imputed_acc_object
    features_index = non_zero_rows(imputed_acc_object.mtx)
    imputed_acc_object.mtx = imputed_acc_object.mtx[features_index, :]
    imputed_acc_object.feature_names = subset_list(
        imputed_acc_object.feature_names, features_index)
    # Load regions
    pr_regions = region_names_to_coordinates(imputed_acc_object.feature_names)
    pr_regions.loc[:, 'Name'] = imputed_acc_object.feature_names
    pr_regions = pr.PyRanges(pr_regions)
    # Add gene width
    if pr_annot.df['Gene'].isnull().values.any():
        pr_annot = pr.PyRanges(pr_annot.df.fillna(value={'Gene': 'na'}))
    pr_annot.Gene_width = abs(pr_annot.End - pr_annot.Start).astype(np.int32)
    if gene_size_weight:
        log.info('Calculating gene size weights')
        if isinstance(gene_size_scale_factor, str):
            gene_size_scale_factor = np.median(pr_annot.Gene_width)
        pr_annot.Gene_size_weight = gene_size_scale_factor / pr_annot.Gene_width
        log.info('Gene size weights done')
    else:
        pr_annot.Gene_size_weight = 1

    # Prepare promoters annotation
    pd_promoters = pr_annot.df.loc[:, ['Chromosome', 'Transcription_Start_Site', 'Strand', 'Gene']]
    pd_promoters['Transcription_Start_Site'] = (
        pd_promoters.loc[:, 'Transcription_Start_Site']
    ).astype(np.int32)
    pd_promoters['End'] = (pd_promoters.loc[:, 'Transcription_Start_Site']).astype(np.int32)
    pd_promoters.columns = ['Chromosome', 'Start', 'Strand', 'Gene', 'End']
    pd_promoters = pd_promoters.loc[:, ['Chromosome', 'Start', 'End', 'Strand', 'Gene']]
    pr_promoters = pr.PyRanges(pd_promoters)
    pr_promoters = extend_pyranges(pr_promoters, extend_tss[0], extend_tss[1])

    if use_gene_boundaries:
        log.info('Calculating gene boundaries')
        # Add chromosome limits
        chromsizes_begin_pos = chromsizes.df.copy()
        chromsizes_begin_pos['End'] = 1
        chromsizes_begin_pos['Strand'] = '+'
        chromsizes_begin_pos['Gene'] = 'Chrom_Begin'
        chromsizes_begin_neg = chromsizes_begin_pos.copy()
        chromsizes_begin_neg['Strand'] = '-'
        chromsizes_end_pos = chromsizes.df.copy()
        chromsizes_end_pos['Start'] = chromsizes_end_pos['End'] - 1
        chromsizes_end_pos['Strand'] = '+'
        chromsizes_end_pos['Gene'] = 'Chrom_End'
        chromsizes_end_neg = chromsizes_end_pos.copy()
        chromsizes_end_neg['Strand'] = '-'
        pr_gene_bound = pr.PyRanges(
            pd.concat(
                [
                    pr_promoters.df,
                    chromsizes_begin_pos,
                    chromsizes_begin_neg,
                    chromsizes_end_pos,
                    chromsizes_end_neg
                ]
            )
        )
        # Get distance to nearest promoter (of a differrent gene)
        pr_annot_nodup = pr_annot[['Chromosome',
                                   'Start',
                                   'End',
                                   'Strand',
                                   'Gene',
                                   'Gene_width',
                                   'Gene_size_weight']].drop_duplicate_positions().copy()
        pr_annot_nodup = pr.PyRanges(
            pr_annot_nodup.df.drop_duplicates(
                subset="Gene", keep="first"))
        closest_promoter_upstream = pr_annot_nodup.nearest(
            pr_gene_bound, overlap=False, how='upstream')
        closest_promoter_upstream = closest_promoter_upstream[[
            'Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Distance']]
        closest_promoter_downstream = pr_annot_nodup.nearest(
            pr_gene_bound, overlap=False, how='downstream')
        closest_promoter_downstream = closest_promoter_downstream[[
            'Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Distance']]
        # Add distance information and limit if above/below thresholds
        pr_annot_df = pr_annot_nodup.df
        pr_annot_df = pr_annot_df.set_index('Gene')
        closest_promoter_upstream_df = closest_promoter_upstream.df.set_index(
            'Gene').Distance
        closest_promoter_upstream_df.name = 'Distance_upstream'
        pr_annot_df = pd.concat(
            [pr_annot_df, closest_promoter_upstream_df], axis=1, sort=False)
        closest_promoter_downstream_df = closest_promoter_downstream.df.set_index(
            'Gene').Distance
        closest_promoter_downstream_df.name = 'Distance_downstream'
        pr_annot_df = pd.concat(
            [pr_annot_df, closest_promoter_downstream_df], axis=1, sort=False).reset_index()
        pr_annot_df.loc[pr_annot_df.Distance_upstream <
                        upstream[0], 'Distance_upstream'] = upstream[0]
        pr_annot_df.loc[pr_annot_df.Distance_upstream >
                        upstream[1], 'Distance_upstream'] = upstream[1]
        pr_annot_df.loc[pr_annot_df.Distance_downstream <
                        downstream[0], 'Distance_downstream'] = downstream[0]
        pr_annot_df.loc[pr_annot_df.Distance_downstream >
                        downstream[1], 'Distance_downstream'] = downstream[1]
        pr_annot_nodup = pr.PyRanges(pr_annot_df.dropna(axis=0))
        # Extend to search space
        extended_annot = extend_pyranges_with_limits(pr_annot_nodup)
        extended_annot = extended_annot[['Chromosome',
                                         'Start',
                                         'End',
                                         'Strand',
                                         'Gene',
                                         'Gene_width',
                                         'Gene_size_weight',
                                         'Distance_upstream',
                                         'Distance_downstream']]
    else:
        extended_annot = extend_pyranges(pr_annot, upstream[1], downstream[1])
        extended_annot = extended_annot[[
            'Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Gene_width', 'Gene_size_weight']]
    # Format search space
    extended_annot = extended_annot.drop_duplicate_positions()
    # Intersect regions
    regions_per_gene = pr_regions.join(extended_annot)
    regions_per_gene.Width = abs(
        regions_per_gene.End -
        regions_per_gene.Start).astype(
        np.int32)
    regions_per_gene.Start = round(
        regions_per_gene.Start +
        regions_per_gene.Width /
        2).astype(
        np.int32)
    regions_per_gene.End = (regions_per_gene.Start + 1).astype(np.int32)
    # Calculate distance
    log.info('Calculating distances')
    if use_gene_boundaries:
        regions_per_gene = reduce_pyranges_with_limits_b(regions_per_gene)
        regions_per_gene = calculate_distance_with_limits_join(
            regions_per_gene)
    else:
        regions_per_gene = reduce_pyranges_b(
            regions_per_gene, upstream[1], downstream[1])
        regions_per_gene = calculate_distance_join(regions_per_gene)
        regions_per_gene.Distance_weight = 1
    if distance_weight:
        log.info('Calculating distance weigths')
        # Distance weight
        regions_gene_list = []
        regions_gene_body = regions_per_gene[(regions_per_gene.Distance <= extend_gene_body_upstream) & (
                regions_per_gene.Distance >= extend_gene_body_downstream)]
        if len(regions_gene_body) > 0:
            regions_gene_body.Distance_weight = 1 + np.exp(-1)
            regions_gene_list.append(regions_gene_body.df)
        regions_gene_upstream = regions_per_gene[regions_per_gene.Distance >
                                                 extend_gene_body_upstream]
        if len(regions_gene_upstream) > 0:
            regions_gene_upstream.Distance_weight = np.exp(
                (-decay_rate * abs(regions_gene_upstream.Distance) / (5000)).astype(float)) + np.exp(-1)
            regions_gene_list.append(regions_gene_upstream.df)
        regions_gene_downstream = regions_per_gene[regions_per_gene.Distance <
                                                   extend_gene_body_downstream]
        if len(regions_gene_downstream) > 0:
            regions_gene_downstream.Distance_weight = np.exp(
                (-decay_rate * abs(regions_gene_downstream.Distance) / (5000)).astype(float)) + np.exp(-1)
            regions_gene_list.append(regions_gene_downstream.df)
        if len(regions_gene_list) > 0:
            regions_per_gene = pr.PyRanges(
                pd.concat(regions_gene_list, axis=0, sort=False))
            log.info('Distance weights done')
    else:
        regions_per_gene.Distance_weight = 1
    # Remove promoters
    if remove_promoters:
        log.info('Removing distal regions overlapping promoters')
        regions_per_gene_promoters = regions_per_gene[regions_per_gene.Distance == 0]
        regions_per_gene_distal = regions_per_gene[regions_per_gene.Distance != 0]
        regions_per_gene_distal_wo_promoters = regions_per_gene_distal.overlap(
            pr_promoters, invert=True)
        regions_per_gene = pr.PyRanges(pd.concat(
            [regions_per_gene_promoters.df, regions_per_gene_distal_wo_promoters.df]))
    # Calculate variability weight
    if gini_weight:
        log.info('Calculating gini weights')
        subset_imputed_acc_object = imputed_acc_object.subset(
            cells=None, features=list(set(regions_per_gene.Name)), copy=True)
        x = subset_imputed_acc_object.mtx
        if sparse.issparse(x):
            gini_weight = [gini(x[i, :].toarray()) for i in range(x.shape[0])]
        else:
            gini_weight = [gini(x[i, :]) for i in range(x.shape[0])]
        gini_weight = pd.DataFrame(
            gini_weight,
            columns=['Gini'],
            index=subset_imputed_acc_object.feature_names)
        gini_weight['Gini_weight'] = np.exp(
            (1 - gini_weight['Gini'])) + np.exp(-1)
        gini_weight = gini_weight.loc[regions_per_gene.Name,]
        regions_per_gene.Gini_weight = gini_weight.loc[:, 'Gini_weight']
    else:
        regions_per_gene.Gini_weight = 1
    # Return weights
    if use_gene_boundaries:
        weights_df = regions_per_gene.df.loc[:,
                     ['Name',
                      'Gene',
                      'Distance',
                      'Distance_upstream',
                      'Distance_downstream',
                      'Gene_size_weight',
                      'Distance_weight',
                      'Gini_weight']]
    else:
        weights_df = regions_per_gene.df.loc[
                     :, ['Name', 'Gene', 'Distance', 'Gene_size_weight', 'Distance_weight', 'Gini_weight']
                     ]
    return weights_df


def extend_pyranges_with_limits(pr_obj: pr.PyRanges):
    """
    A helper function to extend coordinates downstream/upstream in a pyRanges with Distance_upstream and
    Distance_downstream columns.
    """
    # Split per strand
    positive_pr = pr_obj[pr_obj.Strand == '+']
    negative_pr = pr_obj[pr_obj.Strand == '-']
    # Extend space
    if len(positive_pr) > 0:
        positive_pr.Start = (positive_pr.Start - positive_pr.Distance_upstream).astype(np.int32)
        positive_pr.End = (positive_pr.End + positive_pr.Distance_downstream).astype(np.int32)
    if len(negative_pr) > 0:
        negative_pr.Start = (negative_pr.Start - negative_pr.Distance_downstream).astype(np.int32)
        negative_pr.End = (negative_pr.End + negative_pr.Distance_upstream).astype(np.int32)
    extended_pr = pr.PyRanges(pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
    return extended_pr


def reduce_pyranges_with_limits_b(pr_obj: pr.PyRanges):
    """
    A helper function to reduce coordinates downstream/upstream in a pyRanges with Distance_upstream and
    Distance_downstream columns.
    """
    # Split per strand
    positive_pr = pr_obj[pr_obj.Strand == '+']
    negative_pr = pr_obj[pr_obj.Strand == '-']
    # Extend space
    if len(positive_pr) > 0:
        positive_pr.Start_b = (positive_pr.Start_b + positive_pr.Distance_upstream).astype(np.int32)
        positive_pr.End_b = (positive_pr.End_b - positive_pr.Distance_downstream).astype(np.int32)
    if len(negative_pr) > 0:
        negative_pr.Start_b = (negative_pr.Start_b + negative_pr.Distance_downstream).astype(np.int32)
        negative_pr.End_b = (negative_pr.End_b - negative_pr.Distance_upstream).astype(np.int32)
    extended_pr = pr.PyRanges(pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
    return extended_pr


def extend_pyranges(pr_obj: pr.PyRanges,
                    upstream: int,
                    downstream: int):
    """
    A helper function to extend coordinates downstream/upstream in a pyRanges given upstream and downstream
    distances.
    """
    # Split per strand
    positive_pr = pr_obj[pr_obj.Strand == '+']
    negative_pr = pr_obj[pr_obj.Strand == '-']
    # Extend space
    if len(positive_pr) > 0:
        positive_pr.Start = (positive_pr.Start - upstream).astype(np.int32)
        positive_pr.End = (positive_pr.End + downstream).astype(np.int32)
    if len(negative_pr) > 0:
        negative_pr.Start = (negative_pr.Start - downstream).astype(np.int32)
        negative_pr.End = (negative_pr.End + upstream).astype(np.int32)
    extended_pr = pr.PyRanges(
        pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
    return extended_pr


def reduce_pyranges_b(pr_obj: pr.PyRanges,
                      upstream: int,
                      downstream: int):
    """
    A helper function to reduce coordinates downstream/upstream in a pyRanges given upstream and downstream
    distances.
    """
    # Split per strand
    positive_pr = pr_obj[pr_obj.Strand == '+']
    negative_pr = pr_obj[pr_obj.Strand == '-']
    # Extend space
    if len(positive_pr) > 0:
        positive_pr.Start_b = (positive_pr.Start_b + upstream).astype(np.int32)
        positive_pr.End_b = (positive_pr.End_b - downstream).astype(np.int32)
    if len(negative_pr) > 0:
        negative_pr.Start_b = (negative_pr.Start_b + downstream).astype(np.int32)
        negative_pr.End_b = (negative_pr.End_b - upstream).astype(np.int32)
    extended_pr = pr.PyRanges(pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
    return extended_pr


def calculate_distance_join(pr_obj: pr.PyRanges):
    """
    A helper function to calculate distances between regions and genes.
    """
    # Split per strand
    pr_obj_df = pr_obj.df
    distance_df = pd.DataFrame(
        [
            pr_obj_df.Start_b - pr_obj_df.Start,
            pr_obj_df.End_b - pr_obj_df.Start,
            pr_obj_df.Strand
        ],
        index=['start_dist', 'end_dist', 'strand']
    )
    distance_df = distance_df.transpose()
    distance_df.loc[:, 'min_distance'] = abs(distance_df.loc[:, ['start_dist', 'end_dist']].transpose()).min()
    distance_df.strand[distance_df.strand == '+'] = 1
    distance_df.strand[distance_df.strand == '-'] = -1
    distance_df.loc[:, 'location'] = 0
    distance_df.loc[(distance_df.start_dist > 0) & (distance_df.end_dist > 0), 'location'] = 1
    distance_df.loc[(distance_df.start_dist < 0) & (distance_df.end_dist < 0), 'location'] = -1
    distance_df.loc[:, 'location'] = distance_df.loc[:, 'location'] * distance_df.loc[:, 'strand']
    pr_obj.Distance = distance_df.loc[:, 'location'] * distance_df.loc[:, 'min_distance'].astype(np.int32)
    pr_obj = pr_obj[['Chromosome',
                     'Start',
                     'End',
                     'Strand',
                     'Name',
                     'Gene',
                     'Gene_width',
                     'Gene_size_weight',
                     'Distance']]
    return pr_obj


def calculate_distance_with_limits_join(pr_obj: pr.PyRanges):
    """
    A helper function to calculate distances between regions and genes, returning information on what is the relative
    distance to the TSS and end of the gene.
    """
    # Split per strand
    pr_obj_df = pr_obj.df
    distance_df = pd.DataFrame(
        [
            pr_obj_df.Start_b -
            pr_obj_df.Start,
            pr_obj_df.End_b -
            pr_obj_df.Start,
            pr_obj_df.Strand
        ],
        index=['start_dist', 'end_dist', 'strand'])
    distance_df = distance_df.transpose()
    distance_df.loc[:, 'min_distance'] = abs(distance_df.loc[:, ['start_dist', 'end_dist']].transpose()).min()
    distance_df.strand[distance_df.strand == '+'] = 1
    distance_df.strand[distance_df.strand == '-'] = -1
    distance_df.loc[:, 'location'] = 0
    distance_df.loc[(distance_df.start_dist > 0) & (distance_df.end_dist > 0), 'location'] = 1
    distance_df.loc[(distance_df.start_dist < 0) & (distance_df.end_dist < 0), 'location'] = -1
    distance_df.loc[:, 'location'] = distance_df.loc[:, 'location'] * distance_df.loc[:, 'strand']
    pr_obj.Distance = distance_df.loc[:, 'location'] * distance_df.loc[:, 'min_distance'].astype(np.int32)
    pr_obj = pr_obj[['Chromosome',
                     'Start',
                     'End',
                     'Strand',
                     'Name',
                     'Gene',
                     'Gene_width',
                     'Gene_size_weight',
                     'Distance',
                     'Distance_upstream',
                     'Distance_downstream']]
    return pr_obj
