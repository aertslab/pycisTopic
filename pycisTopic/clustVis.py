import fitsne
import harmonypy as hm
import igraph as ig
import leidenalg as la
import logging
import matplotlib.backends.backend_pdf
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
import sklearn
from sklearn.neighbors import kneighbors_graph
import sys
import umap

from typing import Optional, Union
from typing import List, Iterable

from pycisTopic.cisTopicClass import *

def findClusters(cisTopic_obj: 'cisTopicObject',
				 target: Optional[str] = 'cell',
				 k: Optional[int] = 10,
				 res: Optional[float] = 0.6,
				 seed: Optional[int] = 555,
				 scale: Optional[bool] = False,
				 prefix: Optional[str] = '',
				 selected_topics: Optional[List[int]] = None,
				 selected_features: Optional[List[str]] = None,
				 harmony : Optional[bool] =False):
				 
	"""
	Performing leiden cell or region clustering and add results to cisTopic object's metadata. 
	
	Parameters
	---------
	cisTopic_obj: `class::cisTopicObject`
		A cisTopic object with a model in `class::cisTopicObject.selected_model`.
	target: str, optional
		Whether cells ('cell') or regions ('region') should be clustered. Default: 'cell'
	k: int, optional
		Number of neighbours in the k-neighbours graph. Default: 10
	res: float, optional
		Resolution parameter for the leiden algorithm step. Default: 0.6
	seed: int, optional
		Seed parameter for the leiden algorithm step. Default: 555
	scale: bool, optional
		Whether to scale the cell-topic or topic-regions contributions prior to the clustering. Default: False
	prefix: str, optional
		Prefix to add to the clustering name when adding it to the correspondent metadata attribute. Default: ''
	selected_topics: list, optional
		A list with selected topics to be used for clustering. Default: None (use all topics)
	selected_features: list, optional
		A list with selected features (cells or regions) to cluster. This is recommended when working with regions (e.g. selecting 
		regions in binarized topics), as working with all regions can be time consuming. Default: None (use all features)
	harmony: bool, optional
		If target is 'cell', whether to use harmony processed topic contributions. Default: False.
	"""
	
    # Create cisTopic logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    log.info(f"Finding neighbours")
    model=cisTopic_obj.selected_model
    
    if target == 'cell':
    	if (harmony == True):
        	data_mat=model.cell_topic_harmony
        	prefix='harmony_'+prefix
    	else:
        	data_mat=model.cell_topic

    	data_names=cisTopic_obj.cell_names
    	
    if target == 'region':
    	data_mat=model.topic_region.T
    	data_names=cisTopic_obj.region_names
     
    if selected_topics != None:
        data_mat=data_mat.loc[['Topic' + str(x) for x in selected_topics],:]
    if selected_features != None:
        data_mat=data_mat[selected_features]
        data_names=selected_cells
    
    if scale == True:
        data_mat = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(data_mat), index=data_mat.index.to_list(), columns=data_mat.columns)
    data_mat = data_mat.T
    A = kneighbors_graph(data_mat, k)
    sources, targets = A.nonzero()
    G = ig.Graph(directed=True)
    G.add_vertices(A.shape[0])
    edges = list(zip(sources, targets))
    G.add_edges(edges)
    log.info(f"Finding clusters")
    partition = la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter = res, seed = seed)
    cluster = pd.DataFrame(partition.membership, index=data_names, columns=[prefix + 'leiden_' + str(k) + '_' + str(res)]).astype(str)
    if target == 'cell':
    	cisTopic_obj.addCellData(cluster)
    if target == 'region':
    	cisTopic_obj.addRegionData(cluster)

def runUMAP(cisTopic_obj: 'cisTopicObject',
			target: Optional[str] = 'cell',
			scale: Optional[bool] = False,
			reduction_name: Optional[str] = 'UMAP',
		    random_state: Optional[int] = 555,
		    selected_topics: Optional[List[int]] = None,
		    selected_cells: Optional[List[str]] = None,
		    harmony: Optional[bool] = False):
	
	"""
	Run UMAP and add it to the dimensionality reduction dictionary. 
	
	Parameters
	---------
	cisTopic_obj: `class::cisTopicObject`
		A cisTopic object with a model in `class::cisTopicObject.selected_model`.
	target: str, optional
		Whether cells ('cell') or regions ('region') should be used. Default: 'cell'
	scale: bool, optional
		Whether to scale the cell-topic or topic-regions contributions prior to the dimensionality reduction. Default: False
	reduction_name: str, optional
		Reduction name to use as key in the dimensionality reduction dictionary. Default: 'UMAP'
	random_state: int, optional
		Seed parameter for running UMAP. Default: 555
	selected_topics: list, optional
		A list with selected topics to be used for clustering. Default: None (use all topics)
	selected_features: list, optional
		A list with selected features (cells or regions) to cluster. This is recommended when working with regions (e.g. selecting 
		regions in binarized topics), as working with all regions can be time consuming. Default: None (use all features)
	harmony: bool, optional
		If target is 'cell', whether to use harmony processed topic contributions. Default: False.
	"""
		    
    # Create cisTopic logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    model=cisTopic_obj.selected_model
    
    if target == 'cell':
    	if (harmony == True):
        	data_mat=model.cell_topic_harmony
        	prefix='harmony_'+prefix
    	else:
        	data_mat=model.cell_topic

    	data_names=cisTopic_obj.cell_names
    	
    if target == 'region':
    	data_mat=model.topic_region.T
    	data_names=cisTopic_obj.region_names
     
    if selected_topics != None:
        data_mat=data_mat.loc[['Topic' + str(x) for x in selected_topics],:]
    if selected_features != None:
        data_mat=data_mat[selected_features]
        data_names=selected_cells
    
    if scale == True:
        data_mat = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(data_mat), index=data_mat.index.to_list(), columns=data_mat.columns)
    
    data_mat = data_mat.T

    log.info(f"Running UMAP")
    reducer=umap.UMAP(random_state=random_state)
    embedding = reducer.fit_transform(data_mat)
    dr = pd.DataFrame(embedding, index=data_names, columns=['UMAP_1', 'UMAP_2'])
    if target == 'cell':
    	cisTopic_obj.projections['cell'][reduction_name] = dr
    if target == 'region':
    	cisTopic_obj.projections['region'][reduction_name] = dr

def runTSNE(cisTopic_obj: 'cisTopicObject',
			target: Optional[str] = 'cell',
			scale: Optional[bool] = False,
			reduction_name: Optional[str] = 'tSNE',
			seed: Optional[int] = 555,
			perplexity: Optional[int] = 30,
			selected_topics: Optional[List[int]] = None,
		    selected_cells: Optional[List[str]] = None,
		    harmony: Optional[bool] = False):
    
    
    """
	Run tSNE and add it to the dimensionality reduction dictionary. 
	
	Parameters
	---------
	cisTopic_obj: `class::cisTopicObject`
		A cisTopic object with a model in `class::cisTopicObject.selected_model`.
	target: str, optional
		Whether cells ('cell') or regions ('region') should be used. Default: 'cell'
	scale: bool, optional
		Whether to scale the cell-topic or topic-regions contributions prior to the dimensionality reduction. Default: False
	reduction_name: str, optional
		Reduction name to use as key in the dimensionality reduction dictionary. Default: 'tSNE'
	random_state: int, optional
		Seed parameter for running UMAP. Default: 555
	perplexity: int, optional
		Perplexity parameter for FitSNE. Default: 30
	selected_topics: list, optional
		A list with selected topics to be used for clustering. Default: None (use all topics)
	selected_features: list, optional
		A list with selected features (cells or regions) to cluster. This is recommended when working with regions (e.g. selecting 
		regions in binarized topics), as working with all regions can be time consuming. Default: None (use all features)
	harmony: bool, optional
		If target is 'cell', whether to use harmony processed topic contributions. Default: False.
	"""
    # Create cisTopic logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    
    model=cisTopic_obj.selected_model
    
    if target == 'cell':
    	if (harmony == True):
        	data_mat=model.cell_topic_harmony
        	prefix='harmony_'+prefix
    	else:
        	data_mat=model.cell_topic

    	data_names=cisTopic_obj.cell_names
    	
    if target == 'region':
    	data_mat=model.topic_region.T
    	data_names=cisTopic_obj.region_names
     
    if selected_topics != None:
        data_mat=data_mat.loc[['Topic' + str(x) for x in selected_topics],:]
    if selected_features != None:
        data_mat=data_mat[selected_features]
        data_names=selected_cells
    
    if scale == True:
        data_mat = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(data_mat), index=data_mat.index.to_list(), columns=data_mat.columns)
    
    data_mat = data_mat.T

    log.info(f"Running tSNE")
    embedding = fitsne.FItSNE(np.ascontiguousarray(data_mat.to_numpy()), rand_seed=seed, perplexity=perplexity)
    dr = pd.DataFrame(embedding, index=data_names, columns=['tSNE_1', 'tSNE_2'])

	if target == 'cell':
    	cisTopic_obj.projections['cell'][reduction_name] = dr
    if target == 'region':
    	cisTopic_obj.projections['region'][reduction_name] = dr

def plotMetaData(cisTopic_obj: 'cisTopicObject',
 				 reduction_name: str,
  				 variable: str,
  				 target: Optional[str] = 'cell',
  				 cmap: Optional[Union[str, 'matplotlib.cm']] = cm.viridis, 
  				 s: Optional[int] = 10, 
  				 alpha: Optional[Union[float, int]] = 1, 
  				 seed=123, 
  				 color_dictionary={}, 
  				 selected_cells=None,
  				 save=None):
    cell_data=cisTopic_obj.cell_data
    embedding=cisTopic_obj.projections[reduction_name]
    if selected_cells != None:
        cell_data=cell_data.loc[selected_cells]
        embedding=embedding.loc[selected_cells]
    cell_data=cell_data.loc[embedding.index.to_list()]
    if save != None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)
    for var in variable:
        fig=plt.figure()
        var_data=cell_data.loc[:,var].to_list()
        if isinstance(var_data[0], str):
            categories = set(var_data)
            try:
                color_dict = color_dictionary[var]
            except:
                random.seed(seed)
                color = list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(len(categories))))
                color_dict = dict(zip(categories, color))
            plt.scatter(embedding.iloc[:, 0], embedding.iloc[:, 1], c=cell_data.loc[:,var].apply(lambda x: color_dict[x]), s=s, alpha=alpha)
            plt.xlabel(embedding.columns[0])
            plt.ylabel(embedding.columns[1])
            plt.title(var)
            patchList = []
            for key in color_dict:
                data_key = mpatches.Patch(color=color_dict[key], label=key)
                patchList.append(data_key)
            plt.legend(handles=patchList, bbox_to_anchor=(1.04,1), loc="upper left")
            if save != None:
                pdf.savefig(fig, bbox_inches='tight')
            plt.show()
        else:
            o = np.argsort(var_data)
            plt.scatter(embedding.iloc[o, 0], embedding.iloc[o, 1], c=subsetList(var_data,o), cmap=cmap, s=s,  alpha=alpha)
            plt.xlabel(embedding.columns[0])
            plt.ylabel(embedding.columns[1])
            plt.title(var)
            # setup the colorbar
            normalize = mcolors.Normalize(vmin=np.array(var_data).min(), vmax=np.array(var_data).max())
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
            scalarmappaple.set_array(var_data)
            plt.colorbar(scalarmappaple)
            if save != None:
                pdf.savefig(fig, bbox_inches='tight')
            plt.show()

    if save != None:
        pdf = pdf.close()


def plotTopic(cisTopic_obj, reduction_name, topic=None, cmap=cm.viridis, s=10, alpha=1, scale=False, selected_topics=None,  selected_cells=None, harmony=False, save=None):
    embedding=cisTopic_obj.projections[reduction_name]
    model=cisTopic_obj.selected_model
    if harmony == True:
        cell_topic=model.cell_topic_harmony
        prefix='harmony_'
    else:
        cell_topic=model.cell_topic

    if selected_cells != None:
        cell_data=cell_data.loc[selected_cells]
        embedding=embedding.loc[selected_cells]
    cell_topic=cell_topic.loc[:,embedding.index.to_list()]
    
    if selected_topics != None:
        cell_topic=cell_topic.loc[['Topic' + str(x) for x in selected_topics],]
    
    if scale == True:
        cell_topic = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(cell_topic), index=cell_topic.index.to_list(), columns=cell_topic.columns)
    cell_topic = cell_topic.transpose()
    
    if topic == None:
        topic = cell_topic.columns.to_list()
    else:
        topic = ['Topic'+str(t) for t in topic]

    if save != None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)
    for var in topic:
        var_data = cell_topic.loc[:,var]
        var_data = var_data.sort_values()
        embedding_plot = embedding.loc[var_data.index.tolist(),:]
        fig=plt.figure()
        o = np.argsort(var_data)
        if scale == False:
            plt.scatter(embedding_plot.iloc[o, 0], embedding_plot.iloc[o, 1], c=subsetList(var_data,o), cmap=cmap, s=s, alpha=alpha, vmin=0, vmax=max(var_data))
            normalize = mcolors.Normalize(vmin=0, vmax=np.array(var_data).max())
        else:
            plt.scatter(embedding_plot.iloc[o, 0], embedding_plot.iloc[o, 1], c=subsetList(var_data,o), cmap=cmap, s=s, alpha=alpha)
            normalize = mcolors.Normalize(vmin=np.array(var_data).min(), vmax=np.array(var_data).max())
        plt.xlabel(embedding_plot.columns[0])
        plt.ylabel(embedding_plot.columns[1])
        plt.title(var)
        # setup the colorbar
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        scalarmappaple.set_array(var_data)
        plt.colorbar(scalarmappaple)
        if save != None:
            pdf.savefig(fig, bbox_inches='tight')
        plt.show()

    if save != None:
        pdf.close()

def plotImputedFeatures(cisTopic_obj, reduction_name, imputed_data, features, cmap=cm.viridis, s=10, alpha=1, selected_cells=None, save=None):
    if save != None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)
    for feature in features:
        embedding=cisTopic_obj.projections[reduction_name]
        if selected_cells != None:
            embedding=embedding.loc[selected_cells]
        feature_index = getPositionIndex([feature], imputed_data.feature_names)
        feature_data = imputed_data.mtx[feature_index,:]
        if isinstance(feature_data, sparse.csr_matrix):
            color_data = pd.DataFrame(feature_data.transpose().todense(), index=embedding.index.tolist())
        else:
            color_data = pd.DataFrame(feature_data.transpose(), index=embedding.index.tolist())
        color_data = color_data.sort_values(by=0)
        embedding = embedding.loc[color_data.index.tolist(),:]
        var_data=color_data.iloc[:,0].to_list()
        o = np.argsort(var_data)
        plt.scatter(embedding.iloc[:, 0], embedding.iloc[:, 1], c=subsetList(var_data,o), s=s, alpha=alpha)
        plt.xlabel(embedding.columns[0])
        plt.ylabel(embedding.columns[1])
        plt.title(feature)
        # setup the colorbar
        normalize = mcolors.Normalize(vmin=np.array(color_data).min(), vmax=np.array(color_data).max())
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        scalarmappaple.set_array(color_data)
        plt.colorbar(scalarmappaple)
        if save != None:
            pdf.savefig(fig, bbox_inches='tight')
        plt.show()
    if save != None:
        pdf = pdf.close()

def cellTopicHeatmap(cisTopic_obj, variables, scale=False, cluster_topics=False, color_dict={}, seed=123, legend_loc_x=1.2, legend_loc_y=-0.5, legend_dist_y=-1, selected_topics=None, selected_cells=None, harmony=False, save=None):
    model=cisTopic_obj.selected_model
    if harmony == True:
        cell_topic=model.cell_topic_harmony
    else:
        cell_topic=model.cell_topic
    cell_data=cisTopic_obj.cell_data
    
    if selected_topics != None:
        cell_topic=cell_topic.loc[['Topic' + str(x) for x in selected_topics],]
    if selected_cells != None:
        cell_topic=cell_topic.loc[:,selected_cells]
        cell_data=cell_data.loc[selected_cells]
    
    if scale == True:
        cell_topic = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(cell_topic), index=cell_topic.index.to_list(), columns=cell_topic.columns)
    cell_topic = cell_topic.transpose()

    var = variables[0]
    var_data = cell_data.loc[:,var].sort_values()
    cell_topic = cell_topic.loc[var_data.index.to_list(),:]
    df = pd.concat([cell_topic, var_data], axis=1, sort=False)
    topic_order = df.groupby(var).mean().idxmax().sort_values().index.to_list()
    cell_topic = cell_topic.loc[:,topic_order]
    # Color dict
    col_colors={}
    for var in variables:
        var_data = cell_data.loc[:,var].sort_values()
        categories = set(var_data)
        try:
            color_dict = color_dictionary[var]
        except:
            random.seed(seed)
            color = list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(len(categories))))
            color = [mcolors.to_rgb(x) for x in color]
            color_dict[var] = dict(zip(categories, color))
        col_colors[var] = var_data.map(color_dict[var])
        seed=seed+1

    cell_topic = cell_topic.transpose()
    col_colors = pd.concat([col_colors[var] for var in variables], axis=1, sort=False)

    fig=plt.figure()
    g=sns.clustermap(cell_topic,
                     row_cluster=cluster_topics,
                     col_cluster=False,
                     col_colors=col_colors,
                     cmap=cm.viridis,
                     xticklabels=False,
                     figsize=(8,8))
        
    cbar = g.cax
    cbar.set_position([legend_loc_x, 0.55, 0.05, 0.2])
    g.ax_col_dendrogram.set_visible(False)
    g.ax_row_dendrogram.set_visible(False)
                     
    pos= legend_loc_y
    for key in color_dict:
        patchList = []
        for subkey in color_dict[key]:
                data_key = mpatches.Patch(color=color_dict[key][subkey], label=subkey)
                patchList.append(data_key)
        legend = plt.legend(handles=patchList, bbox_to_anchor=(legend_loc_x, pos), loc="center", title=key)
        ax = plt.gca().add_artist(legend)
        pos += legend_dist_y

    if save != None:
        g.savefig(save, bbox_inches='tight')
    plt.show()

def harmony(cisTopic_obj, vars_use, scale=True, random_state = 0):
    cell_data=cisTopic_obj.cell_data
    model= cisTopic_obj.selected_model
    cell_topic=model.cell_topic
    if scale == True:
        cell_topic = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(cell_topic), index=cell_topic.index.to_list(), columns=cell_topic.columns)
    cell_topic=cell_topic.transpose().to_numpy()
    ho = hm.run_harmony(cell_topic, cell_data, vars_use, random_state=random_state)
    cell_topic_harmony = pd.DataFrame(ho.Z_corr, index=model.cell_topic.index.to_list(), columns=model.cell_topic.columns)
    cisTopic_obj.selected_model.cell_topic_harmony = cell_topic_harmony
    return cisTopic_obj



