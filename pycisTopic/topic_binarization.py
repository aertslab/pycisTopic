import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from .cistopic_class import *

def binarize_topics(cistopic_obj: 'CistopicObject',
					method: Optional[str] = 'otsu',
					smooth_topics: Optional[bool] = False,
					ntop: Optional[int] = 2000,
					nbins: Optional[int] = 100,
					plot: Optional[bool] = False,
					figsize: Optional[Tuple[float, float]] = (6.4,4.8),
					num_columns: Optional[int] = 1,
					save: Optional[str]=None):
	"""
	Binarize topic-region distributions.
	
	Parameters
	---------
	cistopic_obj: `class::CistopicObject`
		A cisTopic object with a model in `class::CistopicObject.selected_model`.
	method: str, optional
		Method to use for topic binarization. Possible options are: 'otsu' [Otsu, 1979], 'yen' [Yen et al., 1995], 'li'
		[Li & Lee, 1993] or 'ntop' [Taking the top n regions per topic]. Default: 'otsu'
	smooth_topics: bool, optional
		Whether to smooth topics distributions to penalize regions enriched across many topics. The formula applied is
		\eqn{\beta_{w, k} (\log\beta_{w,k} - 1 / K \sum_{k'} \log \beta_{w,k'})}
	ntop: int, optional
		Number of top regions to select when using method='ntop'. Default: 2000
	nbins: int, optional
		Number of bins to use in the histogram used for otsu, yen and li thresholding. Default: 100
	plot: bool, optional
		Whether to plot region-topic distributions and their threshold. Default: False
	figsize: tuple, optional
		Size of the figure. If num_columns is 1, this is the size for each figure; if num_columns is above 1, this is the overall size of the figure (if keeping
		default, it will be the size of each subplot in the figure). Default: (6.4, 4.8)
	num_columns: int, optional
		For multiplot figures, indicates the number of columns (the number of rows will be automatically determined based on the number of plots). Default: 1
	save: str, optional
		Path to save plot. Default: None.
		
	Return
	---------
	dict
		A dictionary containing a pd.DataFrame with the selected regions with region names as indexes and a topic score 
		column.
		
	References
	---------	
	Otsu, N., 1979. A threshold selection method from gray-level histograms. IEEE transactions on systems, man, and 
	cybernetics, 9(1), pp.62-66.
	Yen, J.C., Chang, F.J. and Chang, S., 1995. A new criterion for automatic multilevel thresholding. IEEE Transactions on 
	Image Processing, 4(3), pp.370-378.
	Li, C.H. and Lee, C.K., 1993. Minimum cross entropy thresholding. Pattern recognition, 26(4), pp.617-625.
	"""
	# Create cisTopic logger
	level	= logging.INFO
	format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
	handlers = [logging.StreamHandler(stream=sys.stdout)]
	logging.basicConfig(level = level, format = format, handlers = handlers)
	log = logging.getLogger('cisTopic')
	
	topic_region = cistopic_obj.selected_model.topic_region
	
	if smooth_topics == True:
		topic_region = smooth_topics(topic_region)
	
	binarized_topics = {}
	if save != None:
		pdf = matplotlib.backends.backend_pdf.PdfPages(save)
		
	fig = plt.figure(figsize=figsize)
	i = 1
	for i in range(topic_region.shape[1]):
		l = np.asarray(topic_region.iloc[:,i])
		l_norm = (l - np.min(l))/np.ptp(l)
		if method == 'otsu':
			thr = threshold_otsu(l_norm, nbins=nbins)
		elif method == 'yen':
			thr = threshold_yen(l_norm, nbins=nbins)
		elif method == 'li':
			thresholds = np.arange(np.min(l_norm) + 0.01, np.max(l_norm) - 0.01, 0.01)
			entropies = [_cross_entropy(l_norm, t, nbins=nbins) for t in thresholds]
			thr = thresholds[np.argmin(entropies)]
		elif method == 'ntop':
			data =  pd.DataFrame(l_norm).sort_values(0, ascending=False)
			thr = float(data.iloc[ntop,])
		else:
			log.info('Binarization method not found. Please choose: "otsu", "yen", "li" or "ntop".')
		
		if plot == True:
			if num_columns > 1:
				plt.subplot(num_rows, num_columns, i)
				i = i + 1
			plt.hist(l_norm, bins=100)
			plt.axvline(thr, color='tomato', linestyle='--')
			plt.xlabel('Probability Topic ' + str(i+1) + '\n' + 'Selected regions:' + str(sum(l_norm>thr)), fontsize=10)
			if num_columns == 1:
				if save != None:
					pdf.savefig(fig, bbox_inches='tight')
				if plot == True:
					plt.show()
		binarized_topics['Topic' + str(i+1)] =  pd.DataFrame(topic_region.iloc[l_norm>thr,i]).sort_values('Topic'+str(i+1), ascending=False)
   
	if num_columns > 1:
		plt.tight_layout()
		if save != None:
				pdf.savefig(fig, bbox_inches='tight')
		if plot == True:
			plt.show()
		else:
			plt.close()

	if save != None:
		pdf.close() 
	
	return binarized_topics

def smooth_topics(topic_region):
	"""
	Smooth topic-region distributions.
	
	Parameters
	---------
	topic_region: `class::pd.DataFrame`
		A pandas dataframe with topic-region distributions (with topics as columns and regions as rows)

	Return
	---------
	pd.DataFrame
	"""
	topic_region_np = np.apply_along_axis(norm, 1, topic_region.values)
	topic_region = pd.DataFrame(topic_region_np, index=topic_region.index.tolist(), columns=topic_region.columns)
	return topic_region

def norm(x):
	return x*(np.log(x+1e-05) - np.sum(np.log(x+1e-05))/len(x))

def threshold_yen(array, nbins=100):
	hist, bin_centers = histogram(array, nbins)
	# Calculate probability mass function
	pmf = hist.astype(np.float32) / hist.sum()
	P1 = np.cumsum(pmf)  # Cumulative normalized histogram
	P1_sq = np.cumsum(pmf ** 2)
	# Get cumsum calculated from end of squared array
	P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
	# P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid
	# '-inf' in crit. ImageJ Yen implementation replaces those values by zero.
	crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) *
				  (P1[:-1] * (1.0 - P1[:-1])) ** 2)
	return bin_centers[crit.argmax()]

def threshold_otsu(array, nbins=100):
	hist, bin_centers = histogram(array, nbins)
	hist = hist.astype(float)
	# class probabilities for all possible thresholds
	weight1 = np.cumsum(hist)
	weight2 = np.cumsum(hist[::-1])[::-1]
	# class means for all possible thresholds
	mean1 = np.cumsum(hist * bin_centers) / weight1
	mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
	# Clip ends to align class 1 and class 2 variables:
	# The last value of ``weight1``/``mean1`` should pair with zero values in
	# ``weight2``/``mean2``, which do not exist.
	variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
	idx = np.argmax(variance12)
	threshold = bin_centers[:-1][idx]
	return threshold
	
# Computing a histogram using np.histogram on a uint8 image with bins=256
# doesn't work and results in aliasing problems. We use a fully specified set
# of bins to ensure that each uint8 value false into its own bin.
def _cross_entropy(array, threshold, nbins=100):
	hist, bin_centers = histogram(array, nbins=nbins)
	t = np.flatnonzero(bin_centers > threshold)[0]
	m0a = np.sum(hist[:t])  # 0th moment, background
	m0b = np.sum(hist[t:])
	m1a = np.sum(hist[:t] * bin_centers[:t])  # 1st moment, background
	m1b = np.sum(hist[t:] * bin_centers[t:])
	mua = m1a / m0a # mean value, background
	mub = m1b / m0b 
	nu = -m1a * np.log(mua) - m1b * np.log(mub)
	return nu
	
def threshold_li(array, tolerance=None, initial_guess=None,
				 iter_callback=None):

	# Get tolerance
	tolerance = tolerance or np.min(np.diff(np.unique(image))) / 2
	# Initial estimate for iteration. See "initial_guess" in the parameter list
	if initial_guess is None:
		t_next = np.mean(array)
	elif callable(initial_guess):
		t_next = initial_guess(array)
	else:
		raise TypeError('Incorrect type for `initial_guess`; should be '
						'a floating point value, or a function mapping an '
						'array to a floating point value.')
	# initial value for t_curr must be different from t_next by at
	# least the tolerance. Since the image is positive, we ensure this
	# by setting to a large-enough negative number
	t_curr = -0.01*tolerance
	# Callback on initial iterations
	if iter_callback is not None:
		iter_callback(t_next)
	# Stop the iterations when the difference between the
	# new and old threshold values is less than the tolerance
	while abs(t_next - t_curr) > tolerance:
		t_curr = t_next
		foreground = (array > t_curr)
		mean_fore = np.mean(array[foreground])
		mean_back = np.mean(array[~foreground])
		t_next = ((mean_back - mean_fore) /
				  (np.log(mean_back) - np.log(mean_fore)))
		if iter_callback is not None:
			iter_callback(t_next)
	threshold = t_next
	return threshold


def histogram(array, nbins=100):
	array = array.ravel().flatten()
	hist, bin_edges = np.histogram(array, bins=nbins, range=None)
	bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
	return hist, bin_centers
