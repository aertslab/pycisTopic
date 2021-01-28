import logging
import pandas as pd
import pyranges as pr
import sys

from typing import Optional, Union
from typing import List, Dict

def get_consensus_peaks(narrow_peaks_dict: Dict[str, pr.PyRanges],
				   peak_half_width: int,
				   chromsizes: Optional[Union[pr.PyRanges, pd.DataFrame]] = None,
				   path_to_blacklist: Optional[str] = None):
	"""
	Returns consensus peaks from a set of MACS narrow peak results. First, each summit is extended a `peak_half_width` in each direction 
	and then we iteratively filter out less significant peaks that overlap with a more significant one. During this procedure peaks will
	be merged and depending on the number of peaks included into them, different processes will happen:
	* **1 peak**:  The original peak region will be kept
	* **2 peaks**:  The original peak region with the highest score will be kept
	* **3 or more peaks**:  The orignal peak region with the most significant score will be taken, and all the original peak regions in 
	this merged peak region that overlap with the significant peak region will be removed. The process is repeated with the next most 
	significant peak (if it was not removed already) until all peaks are processed.

	This proccess will happen twice, first in each pseudobulk peaks; and after peak score normalization, to process all peaks together.
	
	This approach is described in Corces et al. 2018.
	
	Parameters
	---------
	narrow_peaks_dict: dict
		A dictionary containing group labels as keys and pr.PyRanges with the narrowPeak results from MACS2 as values (as returned by .pseudobulkPeakCalling.peakCalling()).
	peak_half_width: int
		Number of base pairs that each summit will be extended in each direction.
	chromsizes: pd.PyRanges or pd.DataFrame
		A data frame or :class:`pr.PyRanges` containing size of each column, containing 'Chromosome', 'Start' and 'End' columns.
	path_to_blacklist: str, optional
		Path to bed file containing blacklist regions (Amemiya et al., 2019). Default: None
		
	Return
	------
	pr.PyRanges
		A pyRanges containing chromosome, start and end coordinates and the peaks ids of the original peaks that are contained in the 
		consensus region.
	
	References
	------
	Corces, M. R., Granja, J. M., Shams, S., Louie, B. H., Seoane, J. A., Zhou, W., ... & Chang, H. Y. (2018). The chromatin accessibility
	landscape of primary human cancers. Science, 362(6413).
	Amemiya, H. M., Kundaje, A., & Boyle, A. P. (2019). The ENCODE blacklist: identification of problematic regions of the genome. 
	Scientific reports, 9(1), 1-5.
	"""
	# Create logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
	log = logging.getLogger('cisTopic')
	
	if isinstance(chromsizes, pd.DataFrame):
		chromsizes = chromsizes.loc[:,['Chromosome', 'Start', 'End']]
		chromsizes = pr.PyRanges(chromsizes)
	
	log.info('Extending and merging peaks per class')
	center_extended_peaks=[iterative_peak_filtering(calculate_peaks_and_extend(narrow_peaks_dict[x], peak_half_width, chromsizes, path_to_blacklist)).df for x in list(narrow_peaks_dict.keys())]
	log.info('Normalizing peak scores')
	center_extended_peaks_norm=[cpm(x, 'Score') for x in center_extended_peaks]  
	center_extended_peaks_norm=pr.PyRanges(pd.concat(center_extended_peaks_norm, axis=0, sort=False))
	log.info('Merging peaks')
	consensus_peaks = iterative_peak_filtering(center_extended_peaks_norm)
	consensus_peaks.Start = (consensus_peaks.Start + 1).astype(int)
	log.info('Done!')
	return consensus_peaks
	
def cpm(x: pr.PyRanges,
		column: str):
	"""
	cpm normalization
	
	Parameters
	---------
	x: pr.PyRanges
		 A pyRanges object
	column: str
		Name of the column that has to be normalized
		
	Return
	------
	pr.PyRanges
		A pyRanges with the normalized column.
	"""
	x.loc[:,column] = x.loc[:,column]/sum(x.loc[:,column])*1000000
	return x

def calculate_peaks_and_extend(narrow_peaks: pr.PyRanges,
							   peak_half_width: int,
							   chromsizes: Optional[Union[pr.PyRanges, pd.DataFrame]] = None,
				   			   path_to_blacklist: Optional[str] = None): 
	"""
	Extend peaks a number of base pairs in eca direction from the summit
	
	Parameters
	---------
	narrow_peaks: pr.PyRanges
		A pr.PyRanges with the narrowPeak results from MACS2.
	peak_half_width: int
		Number of base pairs that each summit will be extended in each direction.
	chromsizes: pd.PyRanges or pd.DataFrame
		A data frame or :class:`pr.PyRanges` containing size of each column, containing 'Chromosome', 'Start' and 'End' columns.
	path_to_blacklist: str, optional
		Path to bed file containing blacklist regions (Amemiya et al., 2019). Default: None
		
	Return
	------
	pr.PyRanges
		A pyRanges containing chromosome, start and end coordinates of the extended peaks.
	"""
	center_extended_peaks = pr.PyRanges(chromosomes=narrow_peaks.Chromosome, starts=narrow_peaks.Start+narrow_peaks.Summit-peak_half_width, ends=narrow_peaks.Start+narrow_peaks.Summit+peak_half_width+1)
	center_extended_peaks.Name = narrow_peaks.Name
	center_extended_peaks.Score= narrow_peaks.Score
	if isinstance(chromsizes, pr.PyRanges):
		center_extended_peaks = center_extended_peaks.intersect(chromsizes, how='containment')
	if isinstance(path_to_blacklist, str):
		blacklist = pr.read_bed(path_to_blacklist)
		center_extended_peaks = center_extended_peaks.overlap(blacklist, invert=True)
	return center_extended_peaks

def iterative_peak_filtering(center_extended_peaks: pr.PyRanges):
	"""
	Returns consensus peaks from a set of MACS narrow peak results. First, each summit is extended a `peak_half_width` in each direction 
	and then we iteratively filter out less significant peaks that overlap with a more significant one. During this procedure, described 
	in this functions, peaks will be merged and depending on the number of peaks included into them, different processes will happen:
	* **1 peak**:  The original peak region will be kept
	* **2 peaks**:  The original peak region with the highest score will be kept
	* **3 or more peaks**:  The orignal peak region with the most significant score will be taken, and all the original peak regions in 
	this merged peak region that overlap with the significant peak region will be removed. The process is repeated with the next most 
	significant peak (if it was not removed already) until all peaks are processed.

	This proccess will happen twice, first in each pseudobulk peaks; and after peak score normalization, to process all peaks together.
	
	This approach is described in Corces et al. 2018.
	
	Parameters
	---------
	center_extended_peaks: pr.PyRanges
		A pr.PyRanges with all the peaks to be combined (and their MACS score), after centering and extending the peaks.
		
	Return
	------
	pr.PyRanges
		A pyRanges containing chromosome, start and end coordinates and the peaks ids of the original peaks that are contained in the 
		consensus region.
	
	References
	------
	Corces, M. R., Granja, J. M., Shams, S., Louie, B. H., Seoane, J. A., Zhou, W., ... & Chang, H. Y. (2018). The chromatin accessibility
	landscape of primary human cancers. Science, 362(6413).
	Amemiya, H. M., Kundaje, A., & Boyle, A. P. (2019). The ENCODE blacklist: identification of problematic regions of the genome. 
	Scientific reports, 9(1), 1-5.
	"""
	center_extended_peaks_merged = center_extended_peaks.merge(count=True)
	# Take original print region if the number of merged regions is equal to 1
	center_extended_peaks_selected = [center_extended_peaks_merged[center_extended_peaks_merged.Count == 1][['Chromosome', 'Start', 'End']].df]
	center_extended_peaks_merged = center_extended_peaks_merged[center_extended_peaks_merged.Count != 1]
	# Take peak with the highest score if the number of merged regions is equal to 2
	center_extended_peaks_with_2_counts = center_extended_peaks_merged[center_extended_peaks_merged.Count == 2]
	if len(center_extended_peaks_with_2_counts) > 0:
		center_extended_peaks_with_2_counts.Name = center_extended_peaks_with_2_counts.Chromosome.astype(str) + ':' + \
		center_extended_peaks_with_2_counts.Start.astype(str) + '-' + center_extended_peaks_with_2_counts.End.astype(str)
		original_and_merged_coordinates = center_extended_peaks.join(center_extended_peaks_with_2_counts)
		selected_regions = pr.PyRanges(original_and_merged_coordinates.df.iloc[original_and_merged_coordinates.df.groupby(['Name_b'], as_index=False, sort=False)['Score'].idxmax()['Score'].tolist()])
		selected_coordinates = pr.PyRanges(chromosomes=selected_regions.Chromosome, starts=selected_regions.Start, ends = selected_regions.End)
		center_extended_peaks_selected.append(selected_coordinates.df)
	# For peaks with more than 3 counts, take region with highest score, remove those overlapping it and repeat
	center_extended_peaks_with_more_than_2_counts = center_extended_peaks_merged[center_extended_peaks_merged.Count > 2]
	if len(center_extended_peaks_with_more_than_2_counts):
		center_extended_peaks_with_more_than_2_counts.Name = center_extended_peaks_with_more_than_2_counts.Chromosome.astype(str) + ':' + \
		center_extended_peaks_with_more_than_2_counts.Start.astype(str) + '-' + center_extended_peaks_with_more_than_2_counts.End.astype(str)
		original_and_merged_coordinates = center_extended_peaks.join(center_extended_peaks_with_more_than_2_counts)
		selected_regions = pr.PyRanges(original_and_merged_coordinates.df.iloc[original_and_merged_coordinates.df.groupby(['Name_b'], as_index=False, sort=False)['Score'].idxmax()['Score'].tolist()])
		selected_coordinates = pr.PyRanges(chromosomes=selected_regions.Chromosome, starts=selected_regions.Start, ends = selected_regions.End)
		center_extended_peaks_selected.append(selected_coordinates.df)
		remaining_coordinates = original_and_merged_coordinates.overlap(selected_coordinates, invert=True)
	
		while len(remaining_coordinates) > 0:
			selected_regions = pr.PyRanges(remaining_coordinates.df.iloc[remaining_coordinates.df.groupby(['Name_b'], as_index=False, sort=False)['Score'].idxmax()['Score'].tolist()])
			selected_coordinates = pr.PyRanges(chromosomes=selected_regions.Chromosome, starts=selected_regions.Start, ends = selected_regions.End)
			center_extended_peaks_selected.append(selected_coordinates.df)
			remaining_coordinates = remaining_coordinates.overlap(selected_coordinates, invert=True)
		
	selected_coordinates = pr.PyRanges(pd.concat(center_extended_peaks_selected, axis=0, sort=False))
	selected_coordinates.Name = selected_coordinates.Chromosome.astype(str) + ':' + \
	selected_coordinates.Start.astype(str) + '-' + selected_coordinates.End.astype(str)
	add_peak_id_and_score = selected_coordinates.join(center_extended_peaks)
	selected_coordinates.Name = add_peak_id_and_score.df.groupby(['Name'], sort=False)['Name_b'].apply(','.join).reset_index().Name_b
	selected_coordinates.Score = add_peak_id_and_score.df.groupby(['Name'],sort=False)['Score'].max()
	
	return selected_coordinates 
