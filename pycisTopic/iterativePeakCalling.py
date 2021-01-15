import pandas as pd
import pyranges as pr
import logging
import sys

def consensusPeaks(narrow_peaks_dict, peak_half_width, chromsizes=None, path_to_blacklist=None):
    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    log.info('Extending and merging peaks per class')
    center_extended_peaks=[iterative_peak_filtering(calculate_peaks_and_extend(narrow_peaks_dict[x], peak_half_width, chromsizes, path_to_blacklist)).df for x in list(narrow_peaks_dict.keys())]
    log.info('Normalizing peak scores')
    center_extended_peaks_norm=[CPM(x, 'Score') for x in center_extended_peaks]  
    center_extended_peaks_norm=pr.PyRanges(pd.concat(center_extended_peaks_norm, axis=0, sort=False))
    log.info('Merging peaks')
    consensus_peaks = iterative_peak_filtering(center_extended_peaks_norm)
    log.info('Done!')
    return consensus_peaks
    
def CPM(x, column):
    x.loc[:,column] = x.loc[:,column]/sum(x.loc[:,column])*1000000
    return x

def calculate_peaks_and_extend(narrow_peaks, peak_half_width, chromsizes=None, path_to_blacklist=None): 
    center_extended_peaks = pr.PyRanges(chromosomes=narrow_peaks.Chromosome, starts=narrow_peaks.Start+narrow_peaks.Summit-peak_half_width, ends=narrow_peaks.Start+narrow_peaks.Summit+peak_half_width+1)
    center_extended_peaks.Name = narrow_peaks.Name
    center_extended_peaks.Score= narrow_peaks.Score
    if isinstance(chromsizes, pr.PyRanges):
        center_extended_peaks = center_extended_peaks.intersect(chromsizes, how='containment')
    if isinstance(path_to_blacklist, str):
        blacklist = pr.read_bed(path_to_blacklist)
        center_extended_peaks = center_extended_peaks.overlap(blacklist, invert=True)
    return center_extended_peaks

def iterative_peak_filtering(center_extended_peaks):
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
        selected_regions = pr.PyRanges(original_and_merged_coordinates.df.iloc[original_and_merged_coordinates.df.groupby(['Name_b'], as_index=False, sort=False)['Score'].idxmax().tolist()])
        selected_coordinates = pr.PyRanges(chromosomes=selected_regions.Chromosome, starts=selected_regions.Start, ends = selected_regions.End)
        center_extended_peaks_selected.append(selected_coordinates.df)
    # For peaks with more than 3 counts, take region with highest score, remove those overlapping it and repeat
    center_extended_peaks_with_more_than_2_counts = center_extended_peaks_merged[center_extended_peaks_merged.Count > 2]
    if len(center_extended_peaks_with_more_than_2_counts):
        center_extended_peaks_with_more_than_2_counts.Name = center_extended_peaks_with_more_than_2_counts.Chromosome.astype(str) + ':' + \
        center_extended_peaks_with_more_than_2_counts.Start.astype(str) + '-' + center_extended_peaks_with_more_than_2_counts.End.astype(str)
        original_and_merged_coordinates = center_extended_peaks.join(center_extended_peaks_with_more_than_2_counts)
        selected_regions = pr.PyRanges(original_and_merged_coordinates.df.iloc[original_and_merged_coordinates.df.groupby(['Name_b'], as_index=False, sort=False)['Score'].idxmax().tolist()])
        selected_coordinates = pr.PyRanges(chromosomes=selected_regions.Chromosome, starts=selected_regions.Start, ends = selected_regions.End)
        center_extended_peaks_selected.append(selected_coordinates.df)
        remaining_coordinates = original_and_merged_coordinates.overlap(selected_coordinates, invert=True)
    
        while len(remaining_coordinates) > 0:
            selected_regions = pr.PyRanges(remaining_coordinates.df.iloc[remaining_coordinates.df.groupby(['Name_b'], as_index=False, sort=False)['Score'].idxmax().tolist()])
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
