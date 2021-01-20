import collections as cl
import logging
import numpy as np
import pandas as pd
import pyranges as pr
import ray
from scipy import sparse
import sklearn.preprocessing as sp
import sys
from typing import Optional, Union
from typing import List, Dict
from .LDAModels import *
from .utils import *

dtype = pd.SparseDtype(int, fill_value=0)
pd.options.mode.chained_assignment = None

class cisTopicObject:
	"""
	cisTopic data class.
	
	:class:`cisTopicObject` contains the cell by fragment matrices (stored as counts :attr:`fragment_matrix` and as binary accessibility :attr:`binary_matrix`),
	cell metadata :attr:`cell_data`, region metadata :attr:`region_data` and path/s to the fragments file/s :attr:`path_to_fragments`.
	
	LDA models from :class:`cisTopicCGSModel` can be stored :attr:`selected_model` as well as cell/region projections :attr:`projections` as a dictionary.
	
	Attributes
	---------
	fragment_matrix: sparse.csr_matrix
		A matrix containing cell names as column names, regions as row names and fragment counts as values.
	binary_matrix: sparse.csr_matrix
		A matrix containing cell names as column names, regions as row names and whether regions as accessible (0: Not accessible; 1: Accessible) as values.
	cell_names: list
		A list containing cell names.
	region_names: list
		A list containing region names.
	cell_data: pd.DataFrame
		A data frame containing cell information, with cells as indexes and attributes as columns.
	region_data: pd.DataFrame
		A data frame containing region information, with region as indexes and attributes as columns.
	path_to_fragments: str or dict
		A list containing the paths to the fragments files used to generate the :class:`cisTopicObject`.
	project: str
		Name of the cisTopic project.
	"""
	def __init__(self,
			   fragment_matrix: sparse.csr_matrix,
			   binary_matrix: sparse.csr_matrix,
			   cell_names: List[str],
			   region_names: List[str],
			   cell_data: pd.DataFrame,
			   region_data: pd.DataFrame,
			   path_to_fragments: Union[str, Dict[str, str]],
			   project: Optional[str] = 'cisTopic'):
		self.fragment_matrix = fragment_matrix
		self.binary_matrix = binary_matrix
		self.cell_names = cell_names
		self.region_names = region_names
		self.cell_data = cell_data
		self.region_data = region_data
		self.project = project
		if isinstance(path_to_fragments, str):
			path_to_fragments = {project: path_to_fragments}
		self.path_to_fragments = path_to_fragments
		self.selected_model = []
		self.projections= {'cell': {}, 'region': {}}
	
	
	def __str__(self):
		descr = f"cisTopicObject from project {self.project} with nCells × nRegions = {len(self.cell_names)} × {len(self.region_names)}"
		return(descr)

	def addCellData(self,
					cell_data: pd.DataFrame):
		"""
		Add cell metadata to :class:`cisTopicObject`. If the column already exist on the cell metadata, it will be overwritten.
		
		Parameters
		---------
		cell_data: pd.DataFrame
			A data frame containing metadata information, with cell names as indexes. If cells are missing from the metadata, values will be filled with Nan.
			
		Return
		------
		cisTopicObject
			The input :class:`cisTopicObject` with :attr:`cell_data` updated.
		"""
		
		flag=False
		if len(set(self.cell_names) & set(cell_data.index)) < len(self.cell_names):
			check_cell_names = prepare_tag_cells(self.cell_names)
			if len(set(check_cell_names) & set(cell_data.index)) < len(set(self.cell_names) & set(cell_data.index)):
				print("Warning: Some cells in this cisTopicObject are not present in this cell_data. Values will be filled with Nan \n")
			else:
				flag=True
		if len(set(self.cell_data.columns) & set(cell_data.columns)) > 0:
			print(f"Columns {list(set(self.cell_data.columns.values) & set(cell_data.columns.values))} will be overwritten")
			self.cell_data = self.cell_data.loc[:,list(set(self.cell_data.columns).difference(set(cell_data.columns)))]
		if flag==False:
			cell_data = cell_data.loc[list(set(self.cell_names) & set(cell_data.index)),]
			new_cell_data = pd.concat([self.cell_data, cell_data], axis=1, sort=False)
		elif flag==True:
			self.cell_data.index = prepare_tag_cells(self.cell_names)
			cell_data = cell_data.loc[list(set(self.cell_data.index.tolist()) & set(cell_data.index)),]
			new_cell_data = pd.concat([self.cell_data, cell_data], axis=1, sort=False)
			new_cell_data = new_cell_data.loc[prepare_tag_cells(self.cell_names),:]
			new_cell_data.index = self.cell_names
		
		self.cell_data = new_cell_data.loc[self.cell_names,:]

	def addRegionData(self,
					  region_data: pd.DataFrame):
		"""
		Add region metadata to :class:`cisTopicObject`. If the column already exist on the region metadata, it will be overwritten.
		
		Parameters
		---------
		region_data: pd.DataFrame
			A data frame containing metadata information, with region names as indexes. If regions are missing from the metadata, values will be filled with Nan.
		
		Return
		------
		cisTopicObject
			The input :class:`cisTopicObject` with :attr:`region_data` updated.
		"""
		if len(set(self.region_names) & set(region_data.index)) < len(self.region_names):
			print("Warning: Some regions in this cisTopicObject are not present in this region_data. Values will be filled with Nan \n")
		if len(set(self.region_data.columns.values) & set(region_data.columns.values)) > 0:
			print(f"Columns {list(set(self.region_data.columns.values) & set(region_data.columns.values))} will be overwritten")
			self.region_data = self.region_data.loc[:,list(set(self.region_data.columns.values).difference(set(self.columns.values)))]
		region_data = region_data.loc[list(set(self.region_names) & set(region_data.index)),]
		new_region_data = pd.concat([self.region_data, region_data], axis=1, sort=False)
		self.region_data = new_region_data.loc[self.region_names,:]

	def subset(self,
			cells: Optional[List[str]] = None,
			regions: Optional[List[str]] = None,
			copy: Optional[bool] = False):
		"""
		Subset cells and/or regions from :class:`cisTopicObject`. Existent :class:`cisTopicCGSModel` and projections will be deleted. This is to ensure that models contained in a :class:`cisTopicObject` are derived from the cells it contains.
		
		Parameters
		---------
		cells: list, optional
			A list containing the names of the cells to keep.
		regions: list, optional
			A list containing the names of the regions to keep.
		copy: bool, optional
			Whether changes should be done on the input :class:`cisTopicObject` or a new object should be returned
		
		Return
		------
		cisTopicObject
			A :class:`cisTopicObject` containing the selected cells and/or regions.
		"""
		# Create logger
		level	= logging.INFO
		format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
		handlers = [logging.StreamHandler(stream=sys.stdout)]
		logging.basicConfig(level = level, format = format, handlers = handlers)
		log = logging.getLogger('cisTopic')
		
		# Select cells
		if cells is not None:
			try:
				keep_cells_index = getPositionIndex(cells, self.cell_names)
			except:
				try:
					keep_cells_index = getPositionIndex(cells, prepare_tag_cells(self.cell_names))
				except:
					log.error('None of the given cells is contained in this cisTopic object!')	
		else:
			keep_cells_index = list(range(len(self.cell_names)))
		# Select regions
		if regions is not None:
			keep_regions_index = getPositionIndex(regions, self.region_names)
		else:
			keep_regions_index = list(range(len(self.region_names)))
		# Subset
		fragment_matrix = self.fragment_matrix[:, keep_cells_index]
		fragment_matrix = fragment_matrix[keep_regions_index,:]
		binary_matrix = self.binary_matrix[:, keep_cells_index]
		binary_matrix = binary_matrix[keep_regions_index, :]
		region_names = subsetList(self.region_names, keep_regions_index) # Subset selected regions
		keep_regions_index = nonZeroRows(binary_matrix)
		fragment_matrix = fragment_matrix[keep_regions_index,]
		binary_matrix = binary_matrix[keep_regions_index,]
		# Update
		cell_names = subsetList(self.cell_names, keep_cells_index)
		region_names = subsetList(region_names, keep_regions_index) # Subset regions with all zeros
		cell_data = self.cell_data.iloc[keep_cells_index,]
		region_data = self.region_data.iloc[keep_regions_index,]
		path_to_fragments = self.path_to_fragments
		project = self.project
		# Create new object
		if copy == True:
			subset_cisTopic_obj = cisTopicObject(fragment_matrix, binary_matrix, cell_names, region_names, cell_data, region_data, path_to_fragments, project)
			return subset_cisTopic_obj
		else:
			self.fragment_matrix = fragment_matrix
			self.binary_matrix = binary_matrix
			self.cell_names = cell_names
			self.region_names = region_names
			self.cell_data = cell_data
			self.region_data = region_data
			self.selected_model = []
			self.projections= {}

	def merge(self,
			  cisTopic_obj_list: List['cisTopicObject'],
			  is_acc: Optional[int] = 1,
			  project: Optional[str] = 'cisTopic_merge',
			  copy: Optional[bool] = False):
		
		"""
		Merge a list of :class:`cisTopicObject` to the input :class:`cisTopicObject`. Reference coordinates must be the same between the objects. Existent :class:`cisTopicCGSModel` and projections will be deleted. This is to ensure that models contained in a :class:`cisTopicObject` are derived from the cells it contains.
		
		Parameters
		---------
		cisTopic_obj_list: list
			A list containing one or more :class:`cisTopicObject` to merge.
		is_acc: int, optional
			Minimal number of fragments for a region to be considered accessible. Default: 1.
		project: str, optional
			Name of the cisTopic project.
		copy: bool, optional
			Whether changes should be done on the input :class:`cisTopicObject` or a new object should be returned
		Return
		------
		cisTopicObject
			A combined :class:`cisTopicObject`. Two new columns in :attr:`cell_data` indicate the :class:`cisTopicObject` of origin (`cisTopic_id`) and the fragment file from which the cell comes from (`path_to_fragments`).
		"""
		# Create logger
		level	= logging.INFO
		format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
		handlers = [logging.StreamHandler(stream=sys.stdout)]
		logging.basicConfig(level = level, format = format, handlers = handlers)
		log = logging.getLogger('cisTopic')
		
		cisTopic_obj_list.insert(0, self)
		fragment_matrix_list = [x.fragment_matrix for x in cisTopic_obj_list]
		region_names_list = [x.region_names for x in cisTopic_obj_list]
		cell_names_list = [x.cell_names for x in cisTopic_obj_list]
		cell_data_list = [x.cell_data.copy() for x in cisTopic_obj_list]
		project_list = [x.project for x in cisTopic_obj_list]
		path_to_fragments_list = [x.path_to_fragments for x in cisTopic_obj_list]
		path_to_fragments_dict = {k: v for ptf in path_to_fragments_list for k,v in ptf.items()}
        
		if len(project_list) > len(set(project_list)):
			ori_project_list = project_list
			log.info('You cannot merge objects with the same project id. Project id will be updated.')
			project_list=list(map(lambda x: x[1] + '_' + str(project_list[:x[0]].count(x[1]) + 1) if project_list.count(x[1]) > 1 else x[1], enumerate(project_list)))
			for i in range(len(project_list)):
				print(i)
				if len(list(set(cell_data_list[i]['sample_id']))) <= 1:
					if (cell_data_list[i]['sample_id'][0] == ori_project_list[i]) & (cell_data_list[i]['sample_id'][0] != project_list[i]):
						log.info('Conflicting sample_id on project ' + ori_project_list[i] + ' will be updated to match with the new project name.')
						cell_data_list[i]['sample_id'] = [project_list[i]]*len(cell_data_list[i]['sample_id'])
				if list(path_to_fragments_list[i].keys()) == 1:
					if list(path_to_fragments_list[i].keys()) == ori_project_list[i]:
						log.info('Conflicting path_to_fragments key on project ' + project_list[i] + ' will be updated to match with the new project name.')
						path_to_fragments_list[project_list[i]] = path_to_fragments_list.pop(ori_project_list[i])
                                
		cell_names_list = [prepare_tag_cells(cell_names_list[x]) for x in range(len(cell_names_list))]
		fragment_matrix = fragment_matrix_list[0]
		region_names= region_names_list[0]
		cell_names = [n+'-'+s for n,s in zip(cell_names_list[0],cell_data_list[0]['sample_id'].tolist())]
		object_id = [project_list[0]]*len(cell_names)

		cell_data_list[0].index = cell_names
		
		for i in range(1,len(region_names_list)):
			region_names_to_add=region_names_list[i]
			fragment_matrix_to_add=fragment_matrix_list[i]
			cell_names_to_add = cell_names_list[i]
			object_id_to_add = [project_list[i]]*len(cell_names_to_add)
			cell_names_to_add = [n+'-'+s for n,s in zip(cell_names_to_add,cell_data_list[i]['sample_id'].tolist())]
			cell_data_list[i].index = cell_names_to_add
			cell_names=cell_names+cell_names_to_add
			
			object_id=object_id+object_id_to_add
			common_regions=list(set(region_names) & set(region_names_to_add))
			diff_regions=list(set(region_names) ^ set(region_names_to_add))
			
			common_index_fm = getPositionIndex(common_regions, region_names)
			common_index_fm_to_add = getPositionIndex(common_regions, region_names_to_add)
			fragment_matrix=sparse.hstack([fragment_matrix[common_index_fm,], fragment_matrix_to_add[common_index_fm_to_add,]])
			region_names=common_regions
			
			if len(diff_regions) > 0:
				diff_fragment_matrix=np.zeros((len(diff_regions), fragment_matrix.shape[1]))
				fragment_matrix=sparse.vstack([fragment_matrix, diff_fragment_matrix])
				region_names=common_regions+diff_regions
			
			fragment_matrix = sparse.csr_matrix(fragment_matrix).astype(int)
			log.info(f"cisTopic object {i} merged")
		
		
		binary_matrix = sp.binarize(fragment_matrix, threshold=is_acc-1)
		cell_data = pd.concat(cell_data_list, axis=0, sort=False)
		cell_data.index = cell_names
		region_data = [x.region_data for x in cisTopic_obj_list]
		region_data = pd.concat(region_data, axis=0, sort=False)
		if copy is True:
			cisTopic_obj=cisTopicObject(fragment_matrix, binary_matrix, cell_names, region_names, cell_data, region_data, path_to_fragments_dict, project)
			return cisTopic_obj
		else:
			self.fragment_matrix = fragment_matrix
			self.binary_matrix = binary_matrix
			self.cell_names = cell_names
			self.region_names = region_names
			self.cell_data = cell_data
			self.region_data = region_data
			self.path_to_fragments = path_to_fragments_dict
			self.project = project
			self.selected_model = []
			self.projections= {}

	def addLDAModel(self,
					model: 'cisTopicLDAModel'):
		"""
		Add LDA model to a cisTopic object.
		
		Parameters
		---
		model: cisTopicLDAModel
			Selected cisTopic LDA model results (see `LDAModels.evaluateModels`)
		"""
		# Check that region and cell names are in the same order
		model.region_topic = model.topic_region.loc[self.region_names,:]
		model.cell_topic = model.cell_topic.loc[:,self.cell_names]
		self.selected_model = model


def createcisTopicObject(fragment_matrix: Union[pd.DataFrame, sparse.csr_matrix],
						 cell_names: Optional[List[str]] = None,
						 region_names: Optional[List[str]] = None,
						 path_to_blacklist: Optional[str] = None,
						 min_frag: Optional[int] = 1,
						 min_cell: Optional[int] = 1,
						 is_acc: Optional[int] = 1,
						 path_to_fragments: Optional[Union[str, Dict[str, str]]] = {},
						 project: Optional[str] = 'cisTopic',
						 tag_cells: Optional[bool] = True):
	"""
	Creates a cisTopicObject from a count matrix.
		
	Parameters
	---------
	fragment_matrix: pd.DataFrame or sparse.csr_matrix
		A data frame containing cell names as column names, regions as row names and fragment counts as values or :class:`sparse.csr_matrix` containing cells as columns and regions as rows.
	cell_names: list, optional
		A list containing cell names. Only used if the fragment matrix is :class:`sparse.csr_matrix`.
	region_names: list, optional
		A list containing region names. Only used if the fragment matrix is :class:`sparse.csr_matrix`.
	path_to_blacklist: str, optional
		Path to bed file containing blacklist regions (Amemiya et al., 2019).
	min_frag: int, optional
		Minimal number of fragments in a cell for the cell to be kept. Default: 1
	min_cell: int, optional
		Minimal number of cell in which a region is detected to be kept. Default: 1
	is_acc: int, optional
		Minimal number of fragments for a region to be considered accessible. Default: 1
	path_to_fragments: str, dict
		A dict or str containing the paths to the fragments files used to generate the :class:`cisTopicObject`. Default: {}.
	project: str, optional
		Name of the cisTopic project. Default: 'cisTopic'
	tag_cells: bool, optional
		Whether to add the project name as suffix to the cell names. Default: True
		
	Return
	------
	cisTopicObject
	
	References
	------
	Amemiya, H. M., Kundaje, A., & Boyle, A. P. (2019). The ENCODE blacklist: identification of problematic regions of the genome. Scientific reports, 9(1), 1-5.
	"""
	# Create logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
	log = logging.getLogger('cisTopic')
	
	if isinstance(fragment_matrix, pd.DataFrame):
		log.info('Converting fragment matrix to sparse matrix')
		region_names = list(fragment_matrix.index)
		cell_names = list(fragment_matrix.columns.values)
		fragment_matrix = sparse.csr_matrix(fragment_matrix.to_numpy())
	
	if tag_cells == True: 
		cell_names=prepare_tag_cells(cell_names)
		cell_names=[cell_names[x] + '-' + project for x in range(len(cell_names))]

	if isinstance(path_to_blacklist, str):
		log.info('Removing blacklisted regions')
		regions = pr.PyRanges(regionNamesToCoordinates(region_names))
		blacklist = pr.read_bed(path_to_blacklist)
		regions = regions.overlap(blacklist, invert=True)
		selected_regions = [str(chrom) + ":" + str(start) + '-' + str(end) for chrom, start, end in zip(list(regions.Chromosome), list(regions.Start), list(regions.End))]
		index = getPositionIndex(selected_regions, region_names)
		fragment_matrix = fragment_matrix[index,]
		region_names = selected_regions

	log.info('Creating cisTopicObject')
	binary_matrix = sp.binarize(fragment_matrix, threshold=is_acc-1)
	selected_regions = nonZeroRows(binary_matrix)
	fragment_matrix = fragment_matrix[selected_regions,]
	binary_matrix = binary_matrix[selected_regions,]
	region_names = subsetList(region_names, selected_regions)
	
	cisTopic_nr_frag=np.array(fragment_matrix.sum(axis=0)).flatten()
	cisTopic_nr_acc=np.array(binary_matrix.sum(axis=0)).flatten()

	cell_data = pd.DataFrame([cisTopic_nr_frag, np.log10(cisTopic_nr_frag), cisTopic_nr_acc, np.log10(cisTopic_nr_acc), [project]*len(cell_names)], columns=cell_names, index=['cisTopic_nr_frag', 'cisTopic_log_nr_frag', 'cisTopic_nr_acc', 'cisTopic_log_nr_acc', 'sample_id']).transpose()

	if min_frag != 1:
		selected_cells = cell_data.cisTopic_nr_frag >= min_frag
		fragment_matrix = fragment_matrix[:,selected_cells]
		binary_matrix = binary_matrix[:,selected_cells]
		cell_data = cell_data.loc[selected_cells,]
		cell_names = cell_data.index.to_list()

	region_data=regionNamesToCoordinates(region_names)
	region_data['Width'] = abs(region_data.End-region_data.Start).astype(dtype)
	region_data['cisTopic_nr_frag'] = np.array(fragment_matrix.sum(axis=1)).flatten()
	region_data['cisTopic_log_nr_frag'] = np.log10(region_data['cisTopic_nr_frag'])
	region_data['cisTopic_nr_acc'] = np.array(binary_matrix.sum(axis=1)).flatten()
	region_data['cisTopic_log_nr_acc'] = np.log10(region_data['cisTopic_nr_acc'])

	if min_cell != 1:
		selected_regions = region_data.cisTopic_nr_acc >= min_cell
		fragment_matrix = fragment_matrix[selected_regions,:]
		binary_matrix = binary_matrix[selected_regions,:]
		region_data = region_data[selected_regions,:]
		region_names = region_data.index.to_list()
	
	cisTopic_obj = cisTopicObject(fragment_matrix, binary_matrix, cell_names, region_names, cell_data, region_data, path_to_fragments, project)
	log.info('Done!')
	return(cisTopic_obj)

def createcisTopicObjectFromMatrixFile(fragment_matrix_file: str,
									   path_to_blacklist: Optional[str] = None,
									   compression: Optional[str] = None,
									   min_frag: Optional[int] = 1,
									   min_cell: Optional[int] = 1,
									   is_acc: Optional[int] = 1,
									   path_to_fragments: Optional[Dict[str, str]] = {},
									   sample_id: Optional[pd.DataFrame] = None,
									   project: Optional[str] = 'cisTopic'):
	"""
	Creates a cisTopicObject from a count matrix file (tsv).
	
	Parameters
	---------
	fragment_matrix: str
		Path to a tsv file containing cell names as column names, regions as row names and fragment counts as values.
	path_to_blacklist: str, optional
		Path to bed file containing blacklist regions (Amemiya et al., 2019). Default: None
	compression: str, None
		Whether the file is compressed (e.g. bzip). Default: None
	min_frag: int, optional
		Minimal number of fragments in a cell for the cell to be kept. Default: 1
	min_cell: int, optional
		Minimal number of cell in which a region is detected to be kept. Default: 1
	is_acc: int, optional
		Minimal number of fragments for a region to be considered accessible. Default: 1
	path_to_fragments: dict, optional
		A list containing the paths to the fragments files used to generate the :class:`cisTopicObject`. Default: None.
	sample_id: pd.DataFrame, optional
		A data frame indicating from which sample each barcode is derived. Required if path_to_fragments is provided. Default: None.
	project: str, optional
		Name of the cisTopic project. Default: 'cisTopic'
		
	Return
	------
	cisTopicObject
	
	References
	------
	Amemiya, H. M., Kundaje, A., & Boyle, A. P. (2019). The ENCODE blacklist: identification of problematic regions of the genome. Scientific reports, 9(1), 1-5.
	"""
	# Create logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
	log = logging.getLogger('cisTopic')
	
	log.info('Reading data')
	if compression is not None:
		fragment_matrix = pd.read_csv(fragment_matrix_file,
			sep='\t',
			header=0,
			compression=compression)
	else:
		fragment_matrix = pd.read_csv(fragment_matrix_file,
			sep='\t',
			header=0)

	cisTopic_obj = createcisTopicObject(fragment_matrix=fragment_matrix,
										path_to_blacklist=path_to_blacklist,
										min_frag=min_frag,
										min_cell=min_cell,
										is_acc=is_acc,
										path_to_fragments=path_to_fragments,
										project=project)
	if sample_id is not None:
		cisTopic_obj.addCellData(sample_id)
	return(cisTopic_obj)

def createcisTopicObjectFromFragments(path_to_fragments: str,
									  path_to_regions: str,
									  path_to_blacklist: Optional[str] = None,
									  metrics: Optional[Union[str, pd.DataFrame]] = None,
									  valid_bc: Optional[List[str]] = None,
									  n_cpu: Optional[int] = 1,
									  min_frag: Optional[int] = 1,
									  min_cell: Optional[int] = 1,
									  is_acc: Optional[int] = 1,
									  remove_duplicates: Optional[bool] = True,
									  project: Optional[str] = 'cisTopic',
									  partition: Optional[int] = 5,
									  fragments_df: Optional[Union[pd.DataFrame, pr.PyRanges]] = None):
	"""
	Creates a cisTopicObject from a fragments file and defined genomic intervals (compatible with CellRangerATAC output)
	
	Parameters
	---------
	path_to_fragments: str
		The path to the fragments file containing chromosome, start, end and assigned barcode for each read (e.g. from CellRanger ATAC (/outs/fragments.tsv.gz)).
	path_to_regions: str
		Path to the bed file with the defined regions.
	path_to_blacklist: str, optional
		Path to bed file containing blacklist regions (Amemiya et al., 2019). Default: None
	metrics: str, optional
		Data frame of CellRanger ot similar, with barcodes and metrics (e.g. from CellRanger ATAC /outs/singlecell.csv). If it is an output from CellRanger, only cells for which is__cell_barcode is 1 will be considered, otherwise only barcodes included in the metrics will be taken. Default: None
	valid_bc: list, optional
		A list with valid cell barcodes can be provided, only used if path_to_metrics is not provided. Default: None
	n_cpu: int, optional
		Number of cores to use. Default: 1.
	min_frag: int, optional
		Minimal number of fragments in a cell for the cell to be kept. Default: 1
	min_cell: int, optional
		Minimal number of cell in which a region is detected to be kept. Default: 1
	is_acc: int, optional
		Minimal number of fragments for a region to be considered accessible. Default: 1
	remove_duplicates: bool, optional
		Whether to consider duplicates when counting fragments. Default: True
	project: str, optional
		Name of the cisTopic project. It will also be used as name for sample_id in the cell_data :class:`cisTopicObject.cell_data`. Default: 'cisTopic'
	partition: int, optional
		When using Pandas > 0.21, counting may fail (https://github.com/pandas-dev/pandas/issues/26314). In that case, the fragments data frame is divided in this number of partitions, and after counting data is merged.
		
	Return
	------
	cisTopicObject
	
	References
	------
	Amemiya, H. M., Kundaje, A., & Boyle, A. P. (2019). The ENCODE blacklist: identification of problematic regions of the genome. Scientific reports, 9(1), 1-5.
	"""
	# Create logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
	log = logging.getLogger('cisTopic')
	
	# Read data
	log.info('Reading data for ' + project)
	if isinstance(fragments_df, pd.DataFrame):
		fragments = pr.PyRanges(fragments_df)
		if path_to_fragments != None:
			log.info('Using fragments of provided pandas data frame')
	else:
		fragments = pr.read_bed(path_to_fragments)
	regions = pr.read_bed(path_to_regions)
	regions = regions[['Chromosome', 'Start', 'End']]
	regions.regionID= [str(chrom) + ":" + str(start) + '-' + str(end) for chrom, start, end in zip(list(regions.Chromosome), list(regions.Start), list(regions.End))]

	# If CellRanger metrics, select valid barcodes
	if metrics is not None:
		log.info('metrics provided!')
		if isinstance(metrics, str):
			metrics = pd.read_csv(metrics)
		if 'is__cell_barcode' in metrics.columns:
			metrics = metrics[metrics.is__cell_barcode == 1]
			metrics.index = metrics.barcode
			metrics = metrics.iloc[:,2:]
		fragments = fragments[fragments.Name.isin(set(metrics.index))]
	if isinstance(valid_bc, list):
		log.info('valid_bc provided, selecting barcodes!')
		fragments = fragments[fragments.Name.isin(set(valid_bc))]
	if metrics is None:
		log.info('Counting total number of fragments (Total_nr_frag)')
		fragments_per_barcode=cl.Counter(fragments.Name.to_list())
		fragments_per_barcode=[fragments_per_barcode[x] for x in set(fragments.Name.to_list())]
		FPB_DF=pd.DataFrame(fragments_per_barcode)
		FPB_DF.index=set(fragments.Name.to_list())
		FPB_DF.columns=['Total_nr_frag']
	# Count fragments in regions
	log.info('Counting fragments in regions')
	fragments_in_regions=regions.join(fragments, nb_cpu=n_cpu)
	# Convert to pandas
	counts_df = pd.concat([fragments_in_regions.regionID, fragments_in_regions.Name, fragments_in_regions.Score.astype(dtype)], axis=1, sort=False)
	if remove_duplicates == True:
		log.info('Duplicate removal')
		counts_df.Score = 1
	log.info('Creating fragment matrix')
	try:
		fragment_matrix = counts_df.groupby(["Name", "regionID"]).agg({"Score": np.sum}).unstack(level="Name").fillna(0).astype(dtype)
		fragment_matrix.columns.names = [None, None]
		fragment_matrix.columns=[x[1] for x in fragment_matrix.columns.values]
	except ValueError:
		log.info('Data is too big, making partitions. This is a reported error in Pandas versions > 0.21 (https://github.com/pandas-dev/pandas/issues/26314)')
		barcode_list = np.array_split(list(set(counts_df.Name.to_list())), partition)
		dfList = [counts_df[counts_df.Name.isin(set(barcode_list[x]))] for x in range(0,partition)]
		dfList = [x.groupby(["Name", "regionID"]).agg({"Score": np.sum}).unstack(level="Name").fillna(0).astype(dtype) for x in dfList]
		fragment_matrix  = pd.concat(dfList, axis=1, sort=False).fillna(0).astype(dtype)
		fragment_matrix.columns.names = [None, None]
		fragment_matrix.columns=[x[1] for x in fragment_matrix.columns.values]

	# Create cisTopicObject
	cisTopic_obj = createcisTopicObject(fragment_matrix=fragment_matrix,
										path_to_blacklist=path_to_blacklist,
										min_frag=min_frag,
										min_cell=min_cell,
										is_acc=is_acc,
										path_to_fragments={project: path_to_fragments},
										project=project)
	if metrics is not None:
		metrics['barcode']=metrics.index.tolist()
		cisTopic_obj.addCellData(metrics)
	else:
		FPB_DF['barcode']=FPB_DF.index.tolist()
		cisTopic_obj.addCellData(FPB_DF)
	return(cisTopic_obj)

def merge(cisTopic_obj_list: List['cisTopicObject'],
		  is_acc: Optional[int] = 1,
		  project: Optional[str] = 'cisTopic_merge'):
	
	"""
	Merge a list of :class:`cisTopicObject` to the input :class:`cisTopicObject`. Reference coordinates must be the same between the objects. Existent :class:`cisTopicCGSModel` and projections will be deleted. This is to ensure that models contained in a :class:`cisTopicObject` are derived from the cells it contains.
		
	Parameters
	---------
	cisTopic_obj_list: list
		A list containing one or more :class:`cisTopicObject` to merge.
	is_acc: int, optional
		Minimal number of fragments for a region to be considered accessible. Default: 1.
	project: str, optional
		Name of the cisTopic project.

	Return
	------
	cisTopicObject
		A combined :class:`cisTopicObject`. Two new columns in :attr:`cell_data` indicate the :class:`cisTopicObject` of origin (`cisTopic_id`) and the fragment file from which the cell comes from (`path_to_fragments`).
	"""

	merged_cisTopic_obj = cisTopic_obj_list[0].merge(cisTopic_obj_list[1:], is_acc=is_acc, project=project, copy=True)
	return merged_cisTopic_obj

