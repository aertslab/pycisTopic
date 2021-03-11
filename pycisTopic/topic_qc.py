from adjustText import adjust_text
from itertools import compress
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats import proportion

from typing import Optional, Union
from typing import Dict, Tuple

from .cistopic_class import *
	
def compute_topic_metrics(cistopic_obj: 'CistopicObject',
					 return_metrics: Optional[bool] = True):
	"""
	Compute topic quality control metrics. 
	
	Parameters
	---------
	cistopic_obj: `class::CistopicObject`
		A cisTopic object with a model in `class::CistopicObject.selected_model`.
	return_metrics: bool, optional
		Whether to return metrics as `class::pd.DataFrame`. The metrics will be also appended to
		`class::CistopicObject.selected_model.topic_qc_metrics` despite the value of this parameter. Default: True.
 
	Return
	---------
	pd.DataFrame
		Data frame containing a column with topic metrics: the number of assignments, the topic coherence (Mimno et al., 2011), the 
		marginal topic distribution (which indicates how much each topic contributes to the model), and the gini index (which indicates
		the specificity of topics. If topics have been binarized, the number of regions/cells per topic will be added.

	References
	----------
	Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A. (2011). Optimizing semantic coherence in topic models. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing (pp. 262-272).
	
	"""
	model = cistopic_obj.selected_model
	topic_coh = model.coherence['Mimno_2011']
	topic_ass = model.topic_ass.drop('Topic', axis=1)
	marginal_dist = model.marg_topic['Marg_Topic']
	gini_values = pd.DataFrame([gini_coefficient(model.cell_topic.iloc[i,:].to_numpy()) for i in range(model.cell_topic.shape[0])])
	topic_qc_metrics = pd.concat([np.log10(topic_ass['Assignments']), topic_ass, topic_coh, marginal_dist, gini_values], axis=1)
	topic_qc_metrics.columns = ['Log10_Assignments'] + topic_ass.columns.tolist() + ['Coherence', 'Marginal_topic_dist', 'Gini_index']
	topic_qc_metrics.index = ['Topic'+str(i) for i in range(1, model.cell_topic.shape[0]+1)]
	cistopic_obj.selected_model.topic_qc_metrics = topic_qc_metrics
	if return_metrics is True:
		return topic_qc_metrics

def plot_topic_qc(topic_qc_metrics: Union[pd.DataFrame, 'CistopicObject'], 
								var_x: str,
								var_y: str,
								min_x: Optional[int] = None,
								max_x: Optional[int] = None,
								min_y: Optional[int] = None,
								max_y: Optional[int] = None,
								var_color: Optional[str] = None,
								cmap: Optional[str] = 'viridis',
								dot_size: Optional[int] = 10,
								text_size: Optional[int] = 10,
								plot: Optional[bool] = False,
								save: Optional[str] = None,
								return_topics: Optional[bool] = False,
								return_fig: Optional[bool] = False):
	"""
	Plotting topic qc metrics and filtering. 
	
	Parameters
	---------
	topic_qc_metrics: `class::pd.DataFrame` or `class::CistopicObject`
		A topic metrics dataframe or a cisTopic object with `class::CistopicObject.selected_model.topic_qc_metrics` filled.
	var_x: str
		Metric to plot.
	var_y: str, optional
		A second metric to plot in combination with `var_x`. 
	min_x: float, optional
		Minimum value on `var_x` to keep the barcode/cell. Default: None.
	max_x: float, optional
		Maximum value on `var_x` to keep the barcode/cell. Default: None.
	min_y: float, optional
		Minimum value on `var_y` to keep the barcode/cell. Default: None.
	max_y: float, optional
		Maximum value on `var_y` to keep the barcode/cell. Default: None.
	var_color: str, optional
		Metric to color plot by. Default: None
	cmap: str, optional
		Color map to color 2D dot plots by density. Default: None.
	dot_size: int, optional
		Dot size in the plot. Default: 10
	text_size: int, optional
		Size of the labels in the plot. Default: 10
	plot: bool, optional
		Whether the plots should be returned to the console. Default: True.
	save: bool, optional
		Path to save plots as a file. Default: None.
	return_topics: bool, optional
		Whether to return selected topics based on user-given thresholds. Default: True.
	return_fig: bool, optional
		Whether to return the plot figure; if several samples it will return a dictionary with the figures per sample. Default: False.
 
	Return
	---
	list
		A list with the selected topics.
	"""
	
	if not isinstance(topic_qc_metrics, pd.DataFrame):
		try:
			topic_qc_metrics=cistopic_obj.selected_model.topic_qc_metrics
		except:
			log.error('This cisTopic object does not include topic qc metrics. Please run compute_topic_metrics() first.')
			
	# Plot xy
	fig=plt.figure()
	if var_color is not None:
		plt.scatter(topic_qc_metrics[var_x], topic_qc_metrics[var_y], c=topic_qc_metrics[var_color], cmap=cmap, s=dot_size)
	else: 
		plt.scatter(topic_qc_metrics[var_x], topic_qc_metrics[var_y], s=dot_size)
		
	# Topics
	n = topic_qc_metrics.index.tolist()
		
	plt.xlabel(var_x, fontsize=10)
	plt.ylabel(var_y, fontsize=10)
	# Add topic number
	texts=[]
	for i, txt in enumerate(n):
		texts.append(plt.text(topic_qc_metrics[var_x][i], topic_qc_metrics[var_y][i], i+1,
					horizontalalignment='center',
					verticalalignment='center',
					size=text_size, weight='bold',
					path_effects=[PathEffects.withStroke(linewidth=3, foreground='w')]))
	adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=.5))
	
	# Add limits
	x=topic_qc_metrics[var_x]
	y=topic_qc_metrics[var_y]
	if min_x != None:
		plt.axvline(x=min_x, color='skyblue', linestyle='--')
		n=list(compress(n, x > min_x))
		y=y[list(x > min_x)]
		x=x[list(x > min_x)]
	if max_x != None:
		plt.axvline(x=max_x, color='tomato', linestyle='--')
		n=list(compress(n, x < max_x))
		y=y[list(x < max_x)]
		x=x[list(x < max_x)]
	if min_y != None:
		plt.axhline(y=min_y, color='skyblue', linestyle='--')
		n=list(compress(n, y > min_y))
		x=x[list(y > min_y)]
		y=y[list(y > min_y)]
	if max_y != None:
		plt.axhline(y=max_y, color='tomato', linestyle='--')
		n=list(compress(n, y < max_y))
		x=x[list(y < max_y)]
		y=y[list(y < max_y)]
		
	# setup the colorbar
	if var_color is not None:
		scalarmappaple = cm.ScalarMappable(cmap=cmap)
		scalarmappaple.set_array(topic_qc_metrics[var_color])
		cbar=plt.colorbar(scalarmappaple)  
		cbar.set_label(var_color, rotation=270, labelpad=15)
	
	plt.tight_layout()
	
	if save is not None:
		fig.savefig(save, bbox_inches='tight')
		
	if plot is not False:
		plt.show()
	else:
		plt.close(fig)
		
	if return_topics is True:
		if return_fig is True:
			return fig, n
		else:
			return n
	else:
		if return_fig is True:
			return fig
		
def topic_annotation(cistopic_obj: 'CistopicObject',
					annot_var: str,
					binarized_cell_topic: Optional[Dict[str, pd.DataFrame]] = None,
					general_topic_thr: Optional[float] = 0.2,
					**kwargs):
	"""
	Automatic annotation of topics.
	
	Parameters
	---------
	cistopic_obj: `class::CistopicObject`
		A cisTopic object with a model in `class::CistopicObject.selected_model`.
	annot_var: str
		Name of the variable (contained in 'class::CistopicObject.cell_data') to use for annotation
	binarized_cell_topic: Dict, optional
		A dictionary containing binarized cell topic distributions (from `binarize_topics()`). If not provided, `binarized_topics()`
		will be run. Default: None.
	general_topic_thr: float, optional
		Threshold for considering a topic as general. After assigning topics to annotations, the ratio of cells in the binarized topic
		in the whole population is compared with the ratio of the total number of cells in the assigned groups versus the whole population.
		If the difference is above this threshold, the topic is considered general. Default: 0.2.
	**kwargs
		Arguments to pass to `binarize_topics()`
	   
	Return
	---------
	pd.DataFrame
		Data frame containing a column with the annotations (separated by ,), the ratio of cells in the binarized topic and the ratio of
		cells assigned to a topic based on the annotated groups.
	"""
	model = cistopic_obj.selected_model
	cell_topic = model.cell_topic
	annot = cistopic_obj.cell_data[annot_var]
	if binarized_cell_topic is None:
		binarized_cell_topic = binarize_topics(cistopic_obj, target='cell', **kwargs)
		
	topic_annot_dict = {topic:[] for topic in cell_topic.index.tolist()}
	group_size_dict = {topic:[] for topic in cell_topic.index.tolist()}
	for group in set(filter(lambda x: x == x , set(annot))):
		cells_in_group = annot[annot == group].index.tolist()
		nobs = len(cells_in_group)
		for topic in cell_topic.index.tolist():
			count =  len(list(set(cells_in_group) & set(binarized_cell_topic[topic].index.tolist())))
			value = binarized_cell_topic[topic].shape[0]/cell_topic.shape[1]
			stat, pval = proportion.proportions_ztest(count, nobs, value=value, alternative='larger')
			if pval < 0.05:
				topic_annot_dict[topic].append(group)
				group_size_dict[topic].append(nobs)
				   
	topic_annot_dict = {x:', '.join(topic_annot_dict[x]) for x in topic_annot_dict.keys()}
	topic_annot = pd.DataFrame([list(topic_annot_dict.values()),
							   [binarized_cell_topic[topic].shape[0]/cell_topic.shape[1] for topic in cell_topic.index.tolist()],
							   [sum(group_size_dict[topic])/cell_topic.shape[1] for topic in cell_topic.index.tolist()]],
							   index=[annot_var, 'Ratio_cells_in_topic', 'Ratio_group_in_population']).T
	topic_annot.index = list(topic_annot_dict.keys())
	topic_annot['is_general'] = (topic_annot['Ratio_cells_in_topic'] - topic_annot['Ratio_group_in_population']) > general_topic_thr		
	return topic_annot 
	
def gini_coefficient(x):
	"""
	Compute Gini coefficient of array of values
	"""
	diffsum = 0
	for i, xi in enumerate(x[:-1], 1):
		diffsum += np.sum(np.abs(xi - x[i:]))
	return diffsum / (len(x)**2 * np.mean(x))	
