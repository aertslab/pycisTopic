import pandas as pd
import numpy as np
import pyranges as pr
import scipy.sparse as sparse
import logging
import sys

from .utils import *
from .diff_features import *

pd.options.mode.chained_assignment = None

def get_gene_activity(imputed_acc_object,
				pr_annot,
				chromsizes,
				use_gene_boundaries=True,
				upstream=[1000,100000],
				downstream=[1000,100000],
				distance_weight=True,
				decay_rate=2.5,
				extend_gene_body_upstream=1000,
				extend_gene_body_downstream=0,
				gene_size_weight=True,
				gene_size_scale_factor='median',
				remove_promoters=True,
				scale_factor=1,
				average_scores=False,
				extend_tss=[10,10],
				return_weights=True,
				gini_weight = True,
				project='Gene_activity'):
	# Create cisTopic logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
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
	region_weights_df.loc[:,'Index'] = get_position_index(region_weights_df.Name, imputed_acc_object.feature_names)
	region_weights_df.loc[:,'Weight'] = region_weights_df.Gene_size_weight*region_weights_df.Distance_weight*region_weights_df.Gini_weight
	region_weights_df = region_weights_df.loc[region_weights_df.Weight > 0,:]
	genes = list(set(region_weights_df.Gene))
	log.info('Getting gene activity scores')
	gene_act = np.array([weighted_aggregation(imputed_acc_object.mtx, region_weights_df[region_weights_df.Gene == gene], average_scores) for gene in genes])
	log.info('Creating imputed features object')
	if scale_factor != 1:
		log.info('Scaling matrix')
		gene_act = gene_act*scale_factor
		gene_act = gene_act.round()
		gene_act=sparse.csr_matrix(gene_act)
		keep_features_index = non_zero_rows(gene_act)
		gene_act=gene_act[keep_features_index,]
		genes=subset_list(genes, keep_features_index)
	gene_act=CistopicImputedFeatures(gene_act, genes, imputed_acc_object.cell_names, project)
	if return_weights == True:
		return gene_act, region_weights_df
	else:
		return gene_act

def weighted_aggregation(imputed_acc_obj_mtx, region_weights_df_per_gene, average_scores):
	if average_scores == True:
		gene_act = imputed_acc_obj_mtx[region_weights_df_per_gene.Index,:].T.dot((region_weights_df_per_gene.Weight.values))/region_weights_df_per_gene.shape[0]
	else:
		gene_act = imputed_acc_obj_mtx[region_weights_df_per_gene.Index,:].T.dot((region_weights_df_per_gene.Weight.values))
	return gene_act

def region_weights(imputed_acc_object,
				  pr_annot,
				  chromsizes,
				  use_gene_boundaries=True,
				  upstream=[1000,100000],
				  downstream=[1000, 100000],
				  distance_weight=True,
				  decay_rate=1,
				  extend_gene_body_upstream=5000,
				  extend_gene_body_downstream=0,
				  gene_size_weight=True,
				  gene_size_scale_factor='median',
				  remove_promoters=True,
				  extend_tss=[10,10],
				  gini_weight = True):
	# Create cisTopic logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
	log = logging.getLogger('cisTopic')
	# Load regions
	pr_regions = region_names_to_coordinates(imputed_acc_object.feature_names)
	pr_regions.loc[:,'Name'] = imputed_acc_object.feature_names
	pr_regions = pr.PyRanges(pr_regions)
	# Add gene width
	if pr_annot.df['Gene'].isnull().values.any():
		pr_annot=pr.PyRanges(pr_annot.df.fillna(value={'Gene':'na'}))
	pr_annot.Gene_width = abs(pr_annot.End-pr_annot.Start).astype(np.int32)
	if gene_size_weight == True:
		log.info('Calculating gene size weigths')
		if isinstance(gene_size_scale_factor, str):
			gene_size_scale_factor=np.median(pr_annot.Gene_width)
		pr_annot.Gene_size_weight = gene_size_scale_factor/pr_annot.Gene_width
		log.info('Gene size weights done')
	else:
		pr_annot.Gene_size_weight = 1
	
	# Prepare promoters annotation
	pd_promoters = pr_annot.df.loc[:, ['Chromosome', 'Transcription_Start_Site', 'Strand', 'Gene']]
	pd_promoters['Transcription_Start_Site'] = (pd_promoters.loc[:,'Transcription_Start_Site']).astype(np.int32)
	pd_promoters['End'] = (pd_promoters.loc[:,'Transcription_Start_Site']).astype(np.int32)
	pd_promoters.columns = ['Chromosome', 'Start', 'Strand', 'Gene', 'End']
	pd_promoters = pd_promoters.loc[:, ['Chromosome', 'Start', 'End', 'Strand', 'Gene']]
	pr_promoters = pr.PyRanges(pd_promoters)
	pr_promoters = extend_pyranges(pr_promoters, extend_tss[0], extend_tss[1])

	
	if use_gene_boundaries == True:
		log.info('Calculating gene boundaries')
		# Add chromosome limits
		chromsizes_begin_pos=  chromsizes.df
		chromsizes_begin_pos['End'] = [1]*len(chromsizes_begin_pos)
		chromsizes_begin_pos['Strand'] = ['+']*len(chromsizes_begin_pos)
		chromsizes_begin_pos['Gene'] = ['Chrom_Begin']*len(chromsizes_begin_pos)
		chromsizes_begin_neg = chromsizes_begin_pos
		chromsizes_begin_neg.loc[:,'Strand'] = ['-']*len(chromsizes_begin_pos)
		chromsizes_end_pos =  chromsizes.df
		chromsizes_end_pos['Start'] = chromsizes_end_pos['End']-1
		chromsizes_end_pos['Strand'] = ['+']*len(chromsizes_end_pos)
		chromsizes_end_pos['Gene'] = ['Chrom_End']*len(chromsizes_end_pos)
		chromsizes_end_neg = chromsizes_end_pos
		chromsizes_end_neg.loc[:,'Strand'] = ['-']*len(chromsizes_end_pos)
		pr_gene_bound = pr.PyRanges(pd.concat([pr_promoters.df, chromsizes_begin_pos, chromsizes_begin_neg, chromsizes_end_pos, chromsizes_end_neg]))
		# Get distance to nearest promoter (of a differrent gene)
		pr_annot_nodup=pr_annot[['Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Gene_width', 'Gene_size_weight']].drop_duplicate_positions().copy()
		pr_annot_nodup=pr.PyRanges(pr_annot_nodup.df.drop_duplicates(subset ="Gene", keep = "first"))
		closest_promoter_upstream=pr_annot_nodup.nearest(pr_gene_bound, overlap=False, how='upstream')
		closest_promoter_upstream=closest_promoter_upstream[['Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Distance']]
		closest_promoter_downstream=pr_annot_nodup.nearest(pr_gene_bound, overlap=False, how='downstream')
		closest_promoter_downstream=closest_promoter_downstream[['Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Distance']]
		# Add distance information and limit if above/below thresholds
		pr_annot_df = pr_annot_nodup.df
		pr_annot_df = pr_annot_df.set_index('Gene')
		closest_promoter_upstream_df = closest_promoter_upstream.df.set_index('Gene').Distance
		closest_promoter_upstream_df.name = 'Distance_upstream'
		pr_annot_df = pd.concat([pr_annot_df, closest_promoter_upstream_df], axis=1, sort=False)
		closest_promoter_downstream_df = closest_promoter_downstream.df.set_index('Gene').Distance
		closest_promoter_downstream_df.name = 'Distance_downstream'
		pr_annot_df = pd.concat([pr_annot_df, closest_promoter_downstream_df], axis=1, sort=False).reset_index()
		pr_annot_df.loc[pr_annot_df.Distance_upstream < upstream[0],'Distance_upstream'] = upstream[0]
		pr_annot_df.loc[pr_annot_df.Distance_upstream > upstream[1],'Distance_upstream'] = upstream[1]
		pr_annot_df.loc[pr_annot_df.Distance_downstream < downstream[0],'Distance_downstream'] = downstream[0]
		pr_annot_df.loc[pr_annot_df.Distance_downstream > downstream[1],'Distance_downstream'] = downstream[1]
		pr_annot_nodup=pr.PyRanges(pr_annot_df.dropna(axis = 0))
		# Extend to search space
		extended_annot = extend_pyranges_with_limits(pr_annot_nodup)
		extended_annot = extended_annot[['Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Gene_width', 'Gene_size_weight', 'Distance_upstream', 'Distance_downstream']]
	else:
		extended_annot = extend_pyranges(pr_annot, upstream[1], downstream[1])
		extended_annot = extended_annot[['Chromosome', 'Start', 'End', 'Strand', 'Gene', 'Gene_width', 'Gene_size_weight']]
	# Format search space
	extended_annot = extended_annot.drop_duplicate_positions()
	# Intersect regions
	regions_per_gene = pr_regions.join(extended_annot)
	regions_per_gene.Width = abs(regions_per_gene.End-regions_per_gene.Start).astype(np.int32)
	regions_per_gene.Start = round(regions_per_gene.Start+regions_per_gene.Width/2).astype(np.int32)
	regions_per_gene.End = (regions_per_gene.Start + 1).astype(np.int32)
	# Calculate distance
	log.info('Calculating distances')
	if use_gene_boundaries == True:
		regions_per_gene = reduce_pyranges_with_limits_b(regions_per_gene)
		regions_per_gene = calculate_distance_with_limits_join(regions_per_gene)
	else:
		regions_per_gene = reduce_pyranges_b(regions_per_gene, upstream[1], downstream[1])
		regions_per_gene = calculate_distance_join(regions_per_gene)
		regions_per_gene.Distance_weight = 1
	if distance_weight == True:
		log.info('Calculating distance weigths')
		# Distance weight
		regions_gene_list=[]
		regions_gene_body = regions_per_gene[(regions_per_gene.Distance <= extend_gene_body_upstream) & (regions_per_gene.Distance >= extend_gene_body_downstream)]
		if len(regions_gene_body) > 0:
			regions_gene_body.Distance_weight =1+np.exp(-1)
			regions_gene_list.append(regions_gene_body.df)
		regions_gene_upstream = regions_per_gene[regions_per_gene.Distance > extend_gene_body_upstream]
		if len(regions_gene_upstream) > 0:
			regions_gene_upstream.Distance_weight = np.exp((-decay_rate*abs(regions_gene_upstream.Distance)/(5000)).astype(float))+np.exp(-1)
			regions_gene_list.append(regions_gene_upstream.df)
		regions_gene_downstream = regions_per_gene[regions_per_gene.Distance < extend_gene_body_downstream]
		if len(regions_gene_downstream) > 0:
			regions_gene_downstream.Distance_weight = np.exp((-decay_rate*abs(regions_gene_downstream.Distance)/(5000)).astype(float))+np.exp(-1)
			regions_gene_list.append(regions_gene_downstream.df)
		if len(regions_gene_list) > 0:
			regions_per_gene = pr.PyRanges(pd.concat(regions_gene_list, axis=0, sort=False))
			log.info('Distance weights done')
	else:
		regions_per_gene.Distance_weight = 1
	# Remove promoters
	if remove_promoters == True:
		log.info('Removing distal regions overlapping promoters')
		regions_per_gene_promoters = regions_per_gene[regions_per_gene.Distance == 0]
		regions_per_gene_distal = regions_per_gene[regions_per_gene.Distance != 0]
		regions_per_gene_distal_wo_promoters = regions_per_gene_distal.overlap(pr_promoters, invert=True)
		regions_per_gene = pr.PyRanges(pd.concat([regions_per_gene_promoters.df, regions_per_gene_distal_wo_promoters.df]))
	# Calculate variability weight
	if gini_weight == True:
		log.info('Calculating gini weights')
		subset_imputed_acc_object = imputed_acc_object.subset(cells=None, features=list(set(regions_per_gene.Name)), copy=True)
		x = subset_imputed_acc_object.mtx
		if sparse.issparse(x):
			gini_weight = [gini(x[i,:].toarray()) for i in range(x.shape[0])]
		else:
			gini_weight = [gini(x[i,:]) for i in range(x.shape[0])]
		gini_weight = pd.DataFrame(gini_weight, columns=['Gini'], index=subset_imputed_acc_object.feature_names)
		gini_weight['Gini_weight'] = np.exp((1-gini_weight['Gini'])) + np.exp(-1)
		gini_weight = gini_weight.loc[regions_per_gene.Name,]
		regions_per_gene.Gini_weight = gini_weight.loc[:,'Gini_weight']
	else:
		regions_per_gene.Gini_weight = 1
	# Return weights
	if use_gene_boundaries == True:
		weights_df = regions_per_gene.df.loc[:, ['Name', 'Gene', 'Distance', 'Distance_upstream', 'Distance_downstream', 'Gene_size_weight', 'Distance_weight', 'Gini_weight']]
	else:
		weights_df = regions_per_gene.df.loc[:, ['Name', 'Gene', 'Distance', 'Gene_size_weight', 'Distance_weight', 'Gini_weight']]
	return weights_df

def extend_pyranges_with_limits(pr_obj):
	# Split per strand
	positive_pr = pr_obj[pr_obj.Strand == '+']
	negative_pr = pr_obj[pr_obj.Strand == '-']
	# Extend space
	if len(positive_pr) > 0:
		positive_pr.Start = (positive_pr.Start-positive_pr.Distance_upstream).astype(np.int32)
		positive_pr.End = (positive_pr.End+positive_pr.Distance_downstream).astype(np.int32)
	if len(negative_pr) > 0:
		negative_pr.Start = (negative_pr.Start-negative_pr.Distance_downstream).astype(np.int32)
		negative_pr.End = (negative_pr.End+negative_pr.Distance_upstream).astype(np.int32)
	extended_pr = pr.PyRanges(pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
	return extended_pr

def reduce_pyranges_with_limits_b(pr_obj):
	# Split per strand
	positive_pr = pr_obj[pr_obj.Strand == '+']
	negative_pr = pr_obj[pr_obj.Strand == '-']
	# Extend space
	if len(positive_pr) > 0:
		positive_pr.Start_b = (positive_pr.Start_b+positive_pr.Distance_upstream).astype(np.int32)
		positive_pr.End_b = (positive_pr.End_b-positive_pr.Distance_downstream).astype(np.int32)
	if len(negative_pr) > 0:
		negative_pr.Start_b = (negative_pr.Start_b+negative_pr.Distance_downstream).astype(np.int32)
		negative_pr.End_b = (negative_pr.End_b-negative_pr.Distance_upstream).astype(np.int32)
	extended_pr = pr.PyRanges(pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
	return extended_pr

def extend_pyranges(pr_obj, upstream, downstream):
	# Split per strand
	positive_pr = pr_obj[pr_obj.Strand == '+']
	negative_pr = pr_obj[pr_obj.Strand == '-']
	# Extend space
	if len(positive_pr) > 0:
		positive_pr.Start = (positive_pr.Start-upstream).astype(np.int32)
		positive_pr.End = (positive_pr.End+downstream).astype(np.int32)
	if len(negative_pr) > 0:
		negative_pr.Start = (negative_pr.Start-downstream).astype(np.int32)
		negative_pr.End = (negative_pr.End+upstream).astype(np.int32)
	extended_pr = pr.PyRanges(pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
	return extended_pr

def reduce_pyranges_b(pr_obj, upstream, downstream):
	# Split per strand
	positive_pr = pr_obj[pr_obj.Strand == '+']
	negative_pr = pr_obj[pr_obj.Strand == '-']
	# Extend space
	if len(positive_pr) > 0:
		positive_pr.Start_b = (positive_pr.Start_b+upstream).astype(np.int32)
		positive_pr.End_b = (positive_pr.End_b-downstream).astype(np.int32)
	if len(negative_pr) > 0:
		negative_pr.Start_b = (negative_pr.Start_b+downstream).astype(np.int32)
		negative_pr.End_b = (negative_pr.End_b-upstream).astype(np.int32)
	extended_pr = pr.PyRanges(pd.concat([positive_pr.df, negative_pr.df], axis=0, sort=False))
	return extended_pr

def calculate_distance_join(pr_obj):
	# Split per strand
	pr_obj_df = pr_obj.df
	distance_df = pd.DataFrame([pr_obj_df.Start_b-pr_obj_df.Start, pr_obj_df.End_b-pr_obj_df.Start, pr_obj_df.Strand], index=['start_dist', 'end_dist', 'strand'])
	distance_df = distance_df.transpose()
	distance_df.loc[:,'min_distance'] = abs(distance_df.loc[:,['start_dist', 'end_dist']].transpose()).min()
	distance_df.strand[distance_df.strand == '+'] = 1
	distance_df.strand[distance_df.strand == '-'] = -1
	distance_df.loc[:,'location'] = 0
	distance_df.loc[(distance_df.start_dist > 0) & (distance_df.end_dist > 0), 'location'] = 1
	distance_df.loc[(distance_df.start_dist < 0) & (distance_df.end_dist < 0), 'location'] = -1
	distance_df.loc[:,'location'] = distance_df.loc[:,'location']*distance_df.loc[:,'strand']
	pr_obj.Distance = distance_df.loc[:,'location']*distance_df.loc[:,'min_distance'].astype(np.int32)
	pr_obj =  pr_obj[['Chromosome', 'Start', 'End', 'Strand', 'Name', 'Gene', 'Gene_width', 'Gene_size_weight', 'Distance']]
	return pr_obj

def calculate_distance_with_limits_join(pr_obj):
	# Split per strand
	pr_obj_df = pr_obj.df
	distance_df = pd.DataFrame([pr_obj_df.Start_b-pr_obj_df.Start, pr_obj_df.End_b-pr_obj_df.Start, pr_obj_df.Strand], index=['start_dist', 'end_dist', 'strand'])
	distance_df = distance_df.transpose()
	distance_df.loc[:,'min_distance'] = abs(distance_df.loc[:,['start_dist', 'end_dist']].transpose()).min()
	distance_df.strand[distance_df.strand == '+'] = 1
	distance_df.strand[distance_df.strand == '-'] = -1
	distance_df.loc[:,'location'] = 0
	distance_df.loc[(distance_df.start_dist > 0) & (distance_df.end_dist > 0), 'location'] = 1
	distance_df.loc[(distance_df.start_dist < 0) & (distance_df.end_dist < 0), 'location'] = -1
	distance_df.loc[:,'location'] = distance_df.loc[:,'location']*distance_df.loc[:,'strand']
	pr_obj.Distance = distance_df.loc[:,'location']*distance_df.loc[:,'min_distance'].astype(np.int32)
	pr_obj =  pr_obj[['Chromosome', 'Start', 'End', 'Strand', 'Name', 'Gene', 'Gene_width', 'Gene_size_weight', 'Distance', 'Distance_upstream', 'Distance_downstream']]
	return pr_obj


