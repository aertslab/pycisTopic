from sklearn.neighbors import kneighbors_graph
import sys
import leidenalg as la
import igraph as ig
import pandas as pd
import umap
import sklearn
import random
import matplotlib.patches as mpatches
import fitsne
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import harmonypy as hm
import matplotlib.backends.backend_pdf
from pycisTopic.cisTopicClass import *


def findClusters(cisTopic_obj, k=10, res=0.6, seed=555, scale=False, prefix='', selected_topics=None, selected_cells=None, harmony=False):
    # Create cisTopic logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    log.info(f"Finding neighbours")
    model=cisTopic_obj.selected_model
    if harmony == True:
        cell_topic=model.cell_topic_harmony
        prefix='harmony_'+prefix
    else:
        cell_topic=model.cell_topic

    cell_names=cisTopic_obj.cell_names
    
    if selected_topics != None:
        cell_topic=cell_topic.loc[['Topic' + str(x) for x in selected_topics],]
    if selected_cells != None:
        cell_topic=cell_topic.loc[:,selected_cells]
        cell_names=selected_cells
    
    if scale == True:
        cell_topic = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(cell_topic), index=cell_topic.index.to_list(), columns=cell_topic.columns)
    cell_topic = cell_topic.transpose()
    A = kneighbors_graph(cell_topic, k)
    sources, targets = A.nonzero()
    G = ig.Graph(directed=True)
    G.add_vertices(A.shape[0])
    edges = list(zip(sources, targets))
    G.add_edges(edges)
    log.info(f"Finding clusters")
    partition = la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter = res, seed = seed)
    cluster = pd.DataFrame(partition.membership, index=cell_names, columns=[prefix + 'Leiden_' + str(k) + '_' + str(res)]).astype(str)
    cisTopic_obj.addCellData(cluster)
    return cisTopic_obj

def runUMAP(cisTopic_obj, scale=False, reduction_name='UMAP', random_state=123, selected_topics=None, selected_cells=None, harmony=False):
    # Create cisTopic logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    model=cisTopic_obj.selected_model
    if harmony == True:
        cell_topic=model.cell_topic_harmony
    else:
        cell_topic=model.cell_topic
    cell_names=cisTopic_obj.cell_names
    
    if selected_topics != None:
        cell_topic=cell_topic.loc[['Topic' + str(x) for x in selected_topics],]
    if selected_cells != None:
        cell_topic=cell_topic.loc[:,selected_cells]
        cell_names=selected_cells
    
    if scale == True:
        cell_topic = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(cell_topic), index=cell_topic.index.to_list(), columns=cell_topic.columns)
    cell_topic = cell_topic.transpose()

    log.info(f"Running UMAP")
    reducer=umap.UMAP(random_state=random_state)
    embedding = reducer.fit_transform(cell_topic)
    dr = pd.DataFrame(embedding, index=cell_names, columns=['UMAP_1', 'UMAP_2'])
    dr = dr.loc[cell_names]
    cisTopic_obj.projections[reduction_name] = dr
    return cisTopic_obj

def runTSNE(cisTopic_obj, scale=False, reduction_name='tSNE', seed=123, perplexity=30, selected_topics=None, selected_cells=None, harmony=False):
    # Create cisTopic logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    
    model=cisTopic_obj.selected_model
    if harmony == True:
        cell_topic=model.cell_topic_harmony
        prefix='harmony_'
    else:
        cell_topic=model.cell_topic
    cell_names=cisTopic_obj.cell_names
    
    if selected_topics != None:
        cell_topic=cell_topic.loc[['Topic' + str(x) for x in selected_topics],]
    if selected_cells != None:
        cell_topic=cell_topic.loc[:,selected_cells]
        cell_names=selected_cells
    
    if scale == True:
        cell_topic = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(cell_topic), index=cell_topic.index.to_list(), columns=cell_topic.columns)
    cell_topic = cell_topic.transpose()

    log.info(f"Running tSNE")
    embedding = fitsne.FItSNE(np.ascontiguousarray(cell_topic.to_numpy()), rand_seed=seed, perplexity=perplexity)
    dr = pd.DataFrame(embedding, index=cell_names, columns=['tSNE_1', 'tSNE_2'])
    dr = dr.loc[cell_names]
    cisTopic_obj.projections[reduction_name] = dr
    return cisTopic_obj

def plotMetaData(cisTopic_obj, reduction_name, variable, cmap=cm.viridis, s=10, alpha=1, seed=123, color_dictionary={}, selected_cells=None, save=None):
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



