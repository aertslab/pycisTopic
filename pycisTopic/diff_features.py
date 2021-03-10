import sklearn
import scipy
from scipy.stats import ranksums
import numpy as np
import ray
import logging
import sys
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import matplotlib

from .utils import *

class CistopicImputedFeatures:
	def __init__(self, imputed_acc, feature_names, cell_names, project):
		self.mtx=imputed_acc
		self.feature_names=feature_names
		self.cell_names=cell_names
		self.project=project
	
	def __str__(self):
		descr = f"CistopicImputedFeatures from project {self.project} with nCells × nFeatures = {len(self.cell_names)} × {len(self.feature_names)}"
		return(descr)
		
	def subset(self, cells=None, features=None, copy=False):
		mtx = self.mtx
		cell_names = self.cell_names
		feature_names = self.feature_names
		
		if cells != None:
			try:
				cells_index = get_position_index(cells, self.cell_names)
			except:
				try:
					cells_index = get_position_index(cells, prepare_tag_cells(self.cell_names))
				except:
					log.error('None of the given cells is contained in this cisTopic object!')	
			mtx = mtx[:,cells_index]
			cell_names = subset_list(cell_names, cells_index)
		
		if features != None:
			features_index = get_position_index(features, feature_names)
			mtx = mtx[features_index,:]
			feature_names = subset_list(feature_names, features_index)
			
		features_index = non_zero_rows(mtx)
		mtx = mtx[features_index,:]
		feature_names = subset_list(feature_names, features_index)
			
		if copy == True:
			return CistopicImputedFeatures(mtx, feature_names, cell_names, self.project)
		else:
			self.mtx = mtx
			self.cell_names = cell_names
			self.feature_names = feature_names
	
	def merge(self, cistopic_imputed_features_list, project='cisTopic_impute_merge', copy=False):
		# Create cisTopic logger
		level	= logging.INFO
		format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
		handlers = [logging.StreamHandler(stream=sys.stdout)]
		logging.basicConfig(level = level, format = format, handlers = handlers)
		log = logging.getLogger('cisTopic')
		
		cistopic_imputed_features_list.insert(0, self)
		mtx_list = [x.mtx for x in cistopic_imputed_features_list]
		feature_names_list = [x.feature_names for x in cistopic_imputed_features_list]
		cell_names_list = [x.cell_names for x in cistopic_imputed_features_list]
		
		mtx = mtx_list[0]
		feature_names= feature_names_list[0]
		cell_names = cell_names_list[0]

		for i in range(1,len(feature_names_list)):
			feature_names_to_add=feature_names_list[i]
			mtx_to_add=mtx_list[i]
			cell_names_to_add = cell_names_list[i]
			cell_names=cell_names+cell_names_to_add
			
			common_features=list(set(feature_names) & set(feature_names_to_add))
			diff_features=list(set(feature_names) ^ set(feature_names_to_add))
			
			common_index_fm = get_position_index(common_features, feature_names)
			common_index_fm_to_add = get_position_index(common_features, feature_names_to_add)
			mtx=sparse.hstack([mtx[common_index_fm,], mtx_to_add[common_index_fm_to_add,]])
			feature_names=common_features
			
			if len(diff_features) > 0:
				diff_mtx=np.zeros((len(diff_features), mtx.shape[1]))
				mtx=sparse.vstack([mtx, diff_mtx])
				feature_names=common_features+diff_features
			
			mtx = sparse.csr_matrix(mtx).astype(int)
			log.info(f"cisTopic imputed features object {i} merged")
			
		if copy == True:
			return CistopicImputedFeatures(mtx, feature_names, cell_names, project)
		else:
			self.mtx = mtx
			self.cell_names = cell_names
			self.feature_names = feature_names
			self.project = project
			

def impute_accessibility(cistopic_obj, selected_cells=None, selected_regions=None, scale_factor=10**6, project='cisTopic_Impute'):
    # Create cisTopic logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    model=cistopic_obj.selected_model
    cell_names=cistopic_obj.cell_names
    cell_topic=model.cell_topic.loc[:,cell_names]
    region_names=cistopic_obj.region_names
    topic_region=model.topic_region.loc[region_names,:]
    
    if selected_cells != None:
        cell_topic=cell_topic.loc[:,selected_cells]
        cell_names=selected_cells
    if selected_regions != None:
        topic_region=topic_region.loc[selected_regions,:]
        region_names=selected_regions
        
    cell_topic = cell_topic.to_numpy().astype('float16')
    topic_region = topic_region.to_numpy().astype('float16')
    
    log.info('Imputing drop-outs')
    imputed_acc = topic_region @ cell_topic

    if isinstance(scale_factor, int):
        log.info('Scaling')
        # Set all values smaller or equal than (1 / scale_factor) to zero.
        np.place(imputed_acc, imputed_acc < (1 / scale_factor), [0])
        imputed_acc = imputed_acc * scale_factor
        imputed_acc = sparse.csr_matrix(imputed_acc, dtype=np.float64)
        if scale_factor != 1:
            log.info('Converting to sparse matrix')
            keep_regions_index = non_zero_rows(imputed_acc)
            imputed_acc=imputed_acc[keep_regions_index,]
            region_names=subset_list(region_names, keep_regions_index)
    imputed_acc_obj=CistopicImputedFeatures(imputed_acc, region_names, cell_names, project)
    log.info('Done!')  
    return(imputed_acc_obj)

def normalize_scores(input_mat, scale_factor=10**4):
	# Create cisTopic logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
	log = logging.getLogger('cisTopic')
	
	log.info('Normalizing imputed data')
	if isinstance(input_mat, CistopicImputedFeatures):
		mtx = np.log1p(input_mat.mtx/input_mat.mtx.sum(0)*scale_factor)
		output=CistopicImputedFeatures(mtx, input_mat.feature_names, input_mat.cell_names, input_mat.project)
	elif isinstance(input_mat, pd.DataFrame):
		output = np.log1p(input_mat.values/input_mat.values.sum(0)*scale_factor)
		output = pd.DataFrame(output, index=input_mat.index.tolist(), columns=input_mat.columns)
	log.info('Done!')
	return(output)

def find_high_variable_features(input_mat, min_disp = 0.05, min_mean = 0.0125, max_mean = 3, max_disp = np.inf, n_bins=20, n_top_features=None, plot=True, save=None):
	# Create cisTopic logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
	log = logging.getLogger('cisTopic')
	
	log.info('Calculating mean and variance')
	if isinstance(input_mat, pd.DataFrame):
		mat = input_mat.values
		features = input_mat.index.tolist()
	else:
		mat = input_mat.mtx
		features = input_mat.feature_names


	if not sparse.issparse(mat):
		mat=sparse.csr_matrix(mat)

	mean, var = sklearn.utils.sparsefuncs.mean_variance_axis(mat, axis=1)
	mean[mean == 0] = 1e-12
	dispersion = var / mean
	# Logarithmic dispersion as in Seurat
	dispersion[dispersion == 0] = np.nan
	dispersion = np.log(dispersion)
	df = pd.DataFrame()
	df['means'] = mean
	df['dispersions'] = dispersion
	
	df['mean_bin'] = pd.cut(df['means'], bins=n_bins)
	disp_grouped = df.groupby('mean_bin')['dispersions']
	disp_mean_bin = disp_grouped.mean()
	disp_std_bin = disp_grouped.std(ddof=1)
	# retrieve those regions that have nan std, these are the ones where
	# only a single gene fell in the bin and implicitly set them to have
	# a normalized disperion of 1
	one_feature_per_bin = disp_std_bin.isnull()
	feature_indices = np.where(one_feature_per_bin[df['mean_bin'].values])[0].tolist()
	
	if len(feature_indices) > 0:
		log.debug(
			f'Feature indices {feature_indices} fell into a single bin: their '
			'normalized dispersion was set to 1.\n	'
			'Decreasing `n_bins` will likely avoid this effect.'
		)
	
	disp_std_bin[one_feature_per_bin.values] = disp_mean_bin[one_feature_per_bin.values].values
	disp_mean_bin[one_feature_per_bin.values] = 0
	# Normalize
	df['dispersions_norm'] = (
		(
			df['dispersions'].values  # use values here as index differs
			- disp_mean_bin[df['mean_bin'].values].values
		) / disp_std_bin[df['mean_bin'].values].values
	)
	
	dispersion_norm = df['dispersions_norm'].values.astype('float32')
	
	if n_top_features is not None:
		dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
		dispersion_norm[::-1].sort() 
		disp_cut_off = dispersion_norm[n_top_features-1]
		feature_subset = np.nan_to_num(df['dispersions_norm'].values) >= disp_cut_off
		log.debug(
			f'the {n_top_features} top features correspond to a '
			f'normalized dispersion cutoff of {disp_cut_off}'
		)
	else:
		dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
		feature_subset = np.logical_and.reduce((
			mean > min_mean, mean < max_mean,
			dispersion_norm > min_disp,
			dispersion_norm < max_disp,
		))

	df['highly_variable'] = feature_subset
	var_features = [features[i] for i in df[df.highly_variable == True].index.to_list()]
	

	if plot == True:
		matplotlib.rcParams['agg.path.chunksize'] = 10000
		plt.scatter(df['means'], df['dispersions_norm'], c=feature_subset, s=10, alpha=0.1)
		plt.xlabel('Mean measurement of features')
		plt.ylabel('Normalized dispersion of the features')
		if save != None:
			fig.savefig(save)
		plt.show()
		
	log.info('Done!')
	return var_features
	
def find_diff_features(cistopic_obj, imputed_features_obj, variable, var_features=None, contrasts=None, contrast_name='contrast', adjpval_thr=0.05, log2fc_thr=1, n_cpu=1):
	# Create cisTopic logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
	log = logging.getLogger('cisTopic')
	
	group_var=cistopic_obj.cell_data.loc[:,variable]
	if contrasts == None:
		levels=sorted(list(set(group_var.tolist())))
		contrasts=[[[x], levels[:levels.index(x)] + levels[levels.index(x)+1:]] for x in levels]
		contrasts_names=levels
	else:
		contrasts_names=['_'.join(contrasts[i][0]) + '_VS_' +'_'.join(contrasts[i][1]) for i in range(len(contrasts))]
	# Get barcodes in each class per contrats
	barcode_groups = [[group_var[group_var.isin(contrasts[x][0])].index.tolist(), group_var[group_var.isin(contrasts[x][1])].index.tolist()] for x in range(len(contrasts))]
	# Subset imputed accessibility matrix
	subset_imputed_features_obj = imputed_features_obj.subset(cells=None, features=var_features, copy=True)
	# Compute p-val and log2FC
	ray.init(num_cpus=n_cpu)
	markers_list=ray.get([markers_ray.remote(subset_imputed_features_obj, barcode_groups[i], contrasts_names[i], adjpval_thr=adjpval_thr, log2fc_thr=log2fc_thr) for i in range(len(contrasts))])
	ray.shutdown()
	markers_dict={contrasts_names[i]: markers_list[i] for i in range(len(markers_list))} 
	return markers_dict

def p_adjust_bh(p):
	"""Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
	p = np.asfarray(p)
	by_descend = p.argsort()[::-1]
	by_orig = by_descend.argsort()
	steps = float(len(p)) / np.arange(len(p), 0, -1)
	q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
	return q[by_orig]

@ray.remote
def markers_ray(input_mat, barcode_group, contrast_name, adjpval_thr=0.05, log2fc_thr=1):
	# Create cisTopic logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
	log = logging.getLogger('cisTopic')
	
	if isinstance(input_mat, pd.DataFrame):
		mat = input_mat.values
		features = input_mat.index.tolist()
		samples = input_mat.columns
	else:
		mat = input_mat.mtx
		features = input_mat.feature_names
		samples = input_mat.cell_names
	
	fg_cells_index = get_position_index(barcode_group[0], samples)
	bg_cells_index = get_position_index(barcode_group[1], samples)
	log.info('Computing p-value for ' + contrast_name)
	if sparse.issparse(mat):
		wilcox_test = [ranksums(mat[x, fg_cells_index].toarray()[0], y=mat[x, bg_cells_index].toarray()[0]) for x in range(mat.shape[0])]
	else:
		wilcox_test = [ranksums(mat[x, fg_cells_index], y=mat[x, bg_cells_index]) for x in range(mat.shape[0])]
	
	log.info('Computing log2FC for ' + contrast_name)
	if sparse.issparse(mat):
		logFC = [np.log2((np.mean(mat[x, fg_cells_index].toarray()[0])+10**-12)/((np.mean(mat[x, bg_cells_index].toarray()[0])+10**-12))) for x in range(mat.shape[0])]
	else:
		logFC = [np.log2((np.mean(mat[x, fg_cells_index])+10**-12)/((np.mean(mat[x, bg_cells_index])+10**-12))) for x in range(mat.shape[0])]

	
	pvalue = [wilcox_test[x].pvalue for x in range(len(wilcox_test))] 
	adj_pvalue = p_adjust_bh(pvalue)
	name = [contrast_name]*len(adj_pvalue)
	markers_dataframe = pd.DataFrame([logFC, adj_pvalue, name], index=['Log2FC', 'Adjusted_pval', 'Contrast'], columns=features).transpose()
	markers_dataframe = markers_dataframe.loc[markers_dataframe['Adjusted_pval'] <= adjpval_thr,:]
	markers_dataframe = markers_dataframe.loc[markers_dataframe['Log2FC'] >= log2fc_thr,:]
	markers_dataframe = markers_dataframe.sort_values(['Log2FC', 'Adjusted_pval'], ascending=[False, True])
	log.info(contrast_name + ' done!')
	return markers_dataframe
