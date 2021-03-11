import gc
import logging
import os
import pandas as pd
import pyBigWig
import pyranges as pr
import ray
import re
import subprocess
import sys

from typing import Optional, Union
from typing import List, Dict

from .cistopic_class import *
from .utils import *


def export_pseudobulk(input_data: Union['CistopicObject', pd.DataFrame, Dict[str, pd.DataFrame]],
					 variable: str,
					 chromsizes: Union[pd.DataFrame, pr.PyRanges],
					 bed_path: str,
					 bigwig_path: str,
					 path_to_fragments: Optional[Dict[str, str]] = None,
					 sample_id_col: Optional[str] = 'sample_id',
					 n_cpu: Optional[int] = 1,
					 normalize_bigwig: Optional[bool] = True,
					 remove_duplicates: Optional[bool] = True,
					 **kwargs):
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
		A data frame or :class:`pr.PyRanges` containing size of each column, containing 'Chromosome', 'Start' and 'End' columns.
	bed_path: str
		Path to folder where the fragments bed files per group will be saved.
	bigwig_path: str
		Path to folder where the bigwig files per group will be saved.
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
	remove_duplicates: bool, optional
		Whether duplicates should be removed before converting the data to bigwig.
		
	Return
	------
	dict
		A dictionary containing the paths to the newly created bed fragments files per group a dictionary containing the paths to the
		newly created bigwig files per group.
	"""
	# Create logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
	log = logging.getLogger('cisTopic')
	
	# Get fragments file
	if isinstance(input_data, CistopicObject):
		path_to_fragments = cistopic_obj.path_to_fragments
		if path_to_fragments == None:
			log.error('No path_to_fragments in this cisTopic object.')
		cell_data = cistopic_obj.cell_data
	elif isinstance(input_data, pd.DataFrame):
		if path_to_fragments == None:
			log.error('Please, provide path_to_fragments.')
		cell_data = input_data
	# Check for sample_id column
	try:
		sample_ids = list(set(cell_data[sample_id_col]))
	except ValueError:
		print('Please, include a sample identification column (e.g. "sample_id") in your cell metadata!')
		
	# Get fragments
	fragments_df_dict={}
	for sample_id in path_to_fragments.keys():
		if isinstance(input_data, pd.DataFrame):
			if sample_id not in sample_ids:
				log.info('The following path_to_fragments entry is not found in the cell metadata sample_id_col: ', sample_id, '. It will be ignored.')
			else:	
				log.info('Reading fragments from ' + path_to_fragments[sample_id])
				fragments_df=pr.read_bed(path_to_fragments[sample_id], as_df=True)
				if 'barcode' in cell_data:
					fragments_df = fragments_df.loc[fragments_df['Name'].isin(cell_data['barcode'].tolist())]	
				else:
					fragments_df = fragments_df.loc[fragments_df['Name'].isin(prepare_tag_cells(cell_data.index.tolist()))]	
				fragments_df_dict[sample_id] = fragments_df

	# Set groups
	if 'barcode' in cell_data:
		cell_data = cell_data.loc[:,[variable, sample_id_col, 'barcode']]
	else:
		cell_data = cell_data.loc[:,[variable, sample_id_col]]
	cell_data[variable] = cell_data[variable].replace(' ', '', regex=True)
	cell_data[variable] = cell_data[variable].replace('[^A-Za-z0-9]+', '_', regex=True)
	groups = sorted(list(set(cell_data[variable])))
	# Check chromosome sizes
	if isinstance(chromsizes, pd.DataFrame):
		chromsizes = chromsizes.loc[:,['Chromosome', 'Start', 'End']]
		chromsizes = pr.PyRanges(chromsizes)
	# Check that output paths exist
	if not os.path.exists(bed_path):
		os.makedirs(bed_path)
	if not os.path.exists(bigwig_path):
		os.makedirs(bigwig_path)
	# Create pseudobulks
	ray.init(num_cpus = n_cpu, **kwargs)
	ray_handle = ray.wait([export_pseudobulk_ray.remote(cell_data,
								group,
								fragments_df_dict, 
								chromsizes,
								bigwig_path,
								bed_path,
								sample_id_col,
								normalize_bigwig,
								remove_duplicates) for group in groups], num_returns=len(groups))
	ray.shutdown()
	bw_paths = {group: os.path.join(bigwig_path, str(group) + '.bw') for group in groups}
	bed_paths = {group: os.path.join(bed_path, str(group) + '.bed.gz') for group in groups}
	return bw_paths, bed_paths

@ray.remote
def export_pseudobulk_ray(cell_data: pd.DataFrame,
						 group: str,
						 fragments_df_dict: Dict[str, pd.DataFrame],
						 chromsizes: pr.PyRanges,
						 bigwig_path: str,
						 bed_path: str,
						 sample_id_col: Optional[str] = 'sample_id',
						 normalize_bigwig: Optional[bool] = True,
					 	 remove_duplicates: Optional[bool] = True):
	"""
	Create pseudobulk as bed and bigwig from single cell fragments file given a barcode annotation and a group. 

	Parameters
	---------
	cell_data: pd.DataFrame
		A cell metadata :class:`pd.Dataframe` containing barcodes, their annotation and their sample of origin.
	group: str
		A character string indicating the group for which pseudobulks will be created.
	fragments_df_dict: dict
		A dictionary containing data frames as values with 'Chromosome', 'Start', 'End', 'Name', and 'Score' as columns; and sample label
		as keys. 'Score' indicates the number of times that a fragments is found assigned to that barcode. 
	chromsizes: pr.PyRanges
		A :class:`pr.PyRanges` containing size of each column, containing 'Chromosome', 'Start' and 'End' columns.
	bed_path: str
		Path to folder where the fragments bed file will be saved.
	bigwig_path: str
		Path to folder where the bigwig file will be saved.
	sample_id_col: str, optional
		Name of the column containing the sample name per barcode in the input :class:`CistopicObject.cell_data` or class:`pd.DataFrame`. Default: 'sample_id'.
	normalize_bigwig: bool, optional
		Whether bigwig files should be CPM normalized. Default: True.
	remove_duplicates: bool, optional
		Whether duplicates should be removed before converting the data to bigwig.
	"""
	# Create logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
	log = logging.getLogger('cisTopic')
	
	log.info('Creating pseudobulk for '+ str(group))
	group_fragments_dict={}
	for sample_id in fragments_df_dict:
		sample_data = cell_data[cell_data.loc[:,sample_id_col].isin([sample_id])]
		if 'barcode' in sample_data:
			sample_data.index = sample_data['barcode'].tolist()
		else:
			sample_data.index = prepare_tag_cells(sample_data.index.tolist())
		group_var = sample_data.iloc[:,0]
		barcodes=group_var[group_var.isin([group])].index.tolist()
		fragments_df = fragments_df_dict[sample_id]
		group_fragments=fragments_df.loc[fragments_df['Name'].isin(barcodes)]
		if len(fragments_df_dict) > 1:
			group_fragments_dict[sample_id] = group_fragments
			
	if len(fragments_df_dict) > 1:
		group_fragments_list = [group_fragments_dict[list(group_fragments_dict.keys())[x]] for x in range(len(fragments_df_dict))]
		group_fragments = group_fragments_list[0].append(group_fragments_list[1:])
		
	group_pr=pr.PyRanges(group_fragments)
	bigwig_path_group = os.path.join(bigwig_path, str(group) + '.bw')
	bed_path_group = os.path.join(bed_path, str(group) + '.bed.gz')
	if isinstance(bigwig_path, str):
		if remove_duplicates == True:
			group_pr.to_bigwig(path=bigwig_path_group, chromosome_sizes=chromsizes, rpm=normalize_bigwig)
		else:
			group_pr.to_bigwig(path=bigwig_path_group, chromsizes=chromsizes, rpm=normalize_bigwig, value_col='Score')
	if isinstance(bed_path, str):
		group_pr.to_bed(path=bed_path_group, keep=True, compression='infer', chain=False)
	gc.collect()
	log.info(str(group)+' done!')


def peak_calling(macs_path: str,
				bed_paths: Dict,
			 	outdir: str,
			 	genome_size: str,
			 	n_cpu: Optional[int] = 1,
			 	input_format: Optional[str] = 'BEDPE',
			 	shift: Optional[int] = 73,
			 	ext_size: Optional[int] = 146,
			 	keep_dup: Optional[str] = 'all',
			 	q_value: Optional[float] = 0.05,
			 	**kwargs):
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
	
	Return
	------
	dict
		A dictionary containing each group label as names and :class:`pr.PyRanges` with MACS2 narrow peaks as values.
	"""
	if not os.path.exists(outdir):
		os.makedirs(outdir)
		
	ray.init(num_cpus=n_cpu, **kwargs)
	narrow_peaks = ray.get([macs_call_peak_ray.remote(macs_path,
								bed_paths[name],
								name,
								outdir, 
								genome_size,
								input_format,
								shift,
								ext_size,
								keep_dup,
								q_value) for name in list(bed_paths.keys())])
	ray.shutdown()
	narrow_peaks_dict={list(bed_paths.keys())[i]: narrow_peaks[i].narrow_peak for i in range(len(narrow_peaks))} 
	return narrow_peaks_dict


@ray.remote
def macs_call_peak_ray(macs_path: str,
					  bed_path: str,
					  name: str,
					  outdir: str,
					  genome_size: str,
					  input_format: Optional[str] = 'BEDPE',
				 	  shift: Optional[int] = 73,
				 	  ext_size: Optional[int] = 146, 
				  	  keep_dup: Optional[str] = 'all',
				 	  q_value: Optional[int] = 0.05):
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
	
	Return
	------
	dict
		A :class:`pr.PyRanges` with MACS2 narrow peaks as values.
	"""
	# Create logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
	log = logging.getLogger('cisTopic')
	
	MACS_peak_calling = MACSCallPeak(macs_path, bed_path, name, outdir, genome_size, input_format=input_format, shift=shift, ext_size=ext_size, keep_dup = keep_dup, q_value = q_value)
	log.info(name + ' done!')
	return MACS_peak_calling
	

class MACSCallPeak():
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
	"""
	def __init__(self,
				 macs_path: str,
				 bed_path: str,
				 name: str,
				 outdir: str,
				 genome_size: str,
				 input_format: Optional[str] = 'BEDPE',
				 shift: Optional[int] = 73,
				 ext_size: Optional[int] = 146, 
				 keep_dup: Optional[str] = 'all',
				 q_value: Optional[int] = 0.05):
		self.macs_path = macs_path
		self.treatment = bed_path
		self.name = name
		self.outdir = outdir
		self.format = input_format
		self.gsize = genome_size
		self.shift = shift
		self.ext_size = ext_size
		self.keep_dup = keep_dup
		self.qvalue = q_value
		self.call_peak()

	def call_peak(self):
		"""
		Run MACS2 peak calling.
		"""
		# Create logger
		level	= logging.INFO
		format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
		handlers = [logging.StreamHandler(stream=sys.stdout)]
		logging.basicConfig(level = level, format = format, handlers = handlers)
		log = logging.getLogger('cisTopic')
		
		cmd = self.macs_path + ' callpeak --treatment %s --name %s  --outdir %s --format %s --gsize %s '\
			'--qvalue %s --nomodel --shift %s --extsize %s --keep-dup %s --call-summits --nolambda'

		cmd = cmd % (
			self.treatment, self.name, self.outdir, self.format, self.gsize,
			self.qvalue, self.shift, self.ext_size, self.keep_dup
		)
		log.info("Calling peaks for " + self.name + " with %s", cmd)
		try:
			subprocess.check_output(args=cmd, shell=True, stderr=subprocess.STDOUT)
		except subprocess.CalledProcessError as e:
			raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
		self.narrow_peak = self.load_narrow_peak()
		
	def load_narrow_peak(self):
		"""
		Load MACS2 narrow peak files as :class:`pr.PyRanges`.
		"""
		narrow_peak = pd.read_csv(os.path.join(self.outdir, self.name + '_peaks.narrowPeak'), sep='\t', header = None)
		narrow_peak.columns = ['Chromosome', 'Start', 'End', 'Name', 'Score', 'Strand', 'FC_summit', '-log10_pval', '-log10_qval', 'Summit']
		narrow_peak_pr = pr.PyRanges(narrow_peak)
		return narrow_peak_pr
