import pandas as pd
import numpy as np
import math
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
from PIL import Image
import pyranges as pr
import re
from scipy import sparse


def region_names_to_coordinates(region_names):
    chrom=pd.DataFrame([i.split(':', 1)[0] for i in region_names])
    coor=[i.split(':', 1)[1] for i in region_names]
    start=pd.DataFrame([int(i.split('-', 1)[0]) for i in coor])
    end=pd.DataFrame([int(i.split('-', 1)[1]) for i in coor])
    regiondf=pd.concat([chrom, start, end], axis=1, sort=False)
    regiondf.index=region_names
    regiondf.columns=['Chromosome', 'Start', 'End']
    return(regiondf)

def get_position_index(query_list, target_list):
    d = {k:v for v,k in enumerate(target_list)}
    index=(d[k] for k in query_list)
    return list(index)
    
def non_zero_rows(X):
    if isinstance(X, sparse.csr_matrix):       
         # Remove all explicit zeros in sparse matrix.                                                                                                                                                              
         X.eliminate_zeros()
         # Get number of non zeros per row and get indices for each row which is not completely zero.                                                                                                                                                                                           
         return np.nonzero(X.getnnz(axis=1))[0]
    else:
         # For non sparse matrices.
         return np.nonzero(np.count_nonzero(x, axis=1))[0]

def subset_list(target_list, index_list):
    X = list(map(target_list.__getitem__, index_list))
    return X

def loglikelihood(nzw, ndz, alpha, eta):
    D = ndz.shape[0]
    n_topics = ndz.shape[1]
    vocab_size = nzw.shape[1]
    
    const_prior = (n_topics * math.lgamma(alpha) - math.lgamma(alpha*n_topics)) * D
    const_ll = (vocab_size* math.lgamma(eta) - math.lgamma(eta*vocab_size)) * n_topics
    
    # calculate log p(w|z)
    topic_ll = 0
    for k in range(n_topics):
        sum = eta*vocab_size
        for w in range(vocab_size):
            if nzw[k, w] > 0:
                topic_ll=math.lgamma(nzw[k, w]+eta)
                sum += nzw[k, w]
        topic_ll -= math.lgamma(sum)
    
    # calculate log p(z)
    doc_ll = 0
    for d in range(D):
        sum = alpha*n_topics
        for k in range(n_topics):
            if ndz[d, k] > 0:
                doc_ll=math.lgamma(ndz[d, k] + alpha)
                sum += ndz[d, k]
        doc_ll -= math.lgamma(sum)

    ll=doc_ll-const_prior+topic_ll-const_ll
    return ll

def sparse2bow(X):
    for indprev, indnow in zip(X.indptr, X.indptr[1:]):
        yield np.array(X.indices[indprev:indnow])

def chunks(l, n):
    return [l[x: x+n] for x in xrange(0, len(l), n)]

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.000000000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

def regions_overlap(target, query):
    # Read input
    if isinstance(target, str):
        target_pr=pr.read_bed(target)
    if isinstance(target, list):
        target_pr=pr.PyRanges(regionNamesToCoordinates(target))
    if isinstance(target, pr.PyRanges):
        target_pr=target
    # Read input
    if isinstance(query, str):
        query_pr=pr.read_bed(query)
    if isinstance(query, list):
        query_pr=pr.PyRanges(regionNamesToCoordinates(query))
    if isinstance(query, pr.PyRanges):
        query_pr=query
    
    target_pr = target_pr.overlap(query_pr)
    selected_regions = [str(chrom) + ":" + str(start) + '-' + str(end) for chrom, start, end in zip(list(target_pr.Chromosome), list(target_pr.Start), list(target_pr.End))]
    return selected_regions

def format_input_regions(input_data):
    new_data = {}
    for key in input_data.keys():
        data = input_data[key]
        if isinstance(data, pd.DataFrame):
            regions = data.index.tolist()
            if len(regions) > 0:
                new_data[key] = pr.PyRanges(regionNamesToCoordinates(regions))
        else:
            new_data[key] = data
    return new_data
    
def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            return
    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        s = s.replace(old_string, new_string)
        f.write(s)     

def load_cisTopic_model(path_to_cisTopic_model_matrices):
    metrics = None
    coherence = None
    marg_topic = None
    topic_ass = None
    cell_topic = pd.read_feather(path_to_cisTopic_model_matrices + 'cell_topic.feather')
    cell_topic.index = ['Topic'+str(x) for x in range(1,cell_topic.shape[0]+1)]
    topic_region = pd.read_feather(path_to_cisTopic_model_matrices + 'topic_region.feather')
    topic_region.index = ['Topic'+str(x) for x in range(1,topic_region.shape[0]+1)]
    topic_region = topic_region.T
    parameters = None
    model = cisTopicLDAModel(metrics, coherence, marg_topic, topic_ass, cell_topic, topic_region, parameters)
    return model

def prepare_tag_cells(cell_names):
	new_cell_names = [re.findall(r"^[ACGT]*-[0-9]+-", x)[0].rstrip('-') if len(re.findall(r"^[ACGT]*-[0-9]+-", x)) != 0 else x for x in cell_names]
	new_cell_names = [re.findall(r"^\w*-[0-9]*", new_cell_names[i])[0].rstrip('-') if (len(re.findall(r"^\w*-[0-9]*", new_cell_names[i])) != 0) & (new_cell_names[i] == cell_names[i]) else new_cell_names[i] for i in range(len(new_cell_names))]
	return new_cell_names
	
def multiplot_from_generator(g, num_columns, n_plots, figsize=None, plot=True, save=None):
    if save is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)
    # call 'next(g)' to get past the first 'yield'
    next(g)
    # default to 15-inch rows, with square subplots
    if figsize is None:
        if num_columns == 1:
            figsize = (5, 5)
        else:
            num_rows = int(np.ceil(n_plots/num_columns))
            figsize = (6.4*num_columns, 4.8*num_rows)
              
    if num_columns > 1:
        fig=plt.figure(figsize=figsize) 
        num_rows = int(np.ceil(n_plots/num_columns))
    plot = 0    
    try:
        while True:
            # call plt.figure once per row
            if num_columns == 1:
                fig=plt.figure(figsize=figsize)
                ax = plt.subplot(1, 1, 1)
                if save is not None:
                        pdf.savefig(fig, bbox_inches='tight')
            if num_columns > 1:
                ax = plt.subplot(num_rows, num_columns, plot+1)
                ax.autoscale(enable=True)
                plot = plot + 1
            next(g)
    except StopIteration:
        if num_columns == 1:
            if save is not None:
                pdf.savefig(fig, bbox_inches='tight')
        pass
    if num_columns > 1:
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if save is not None:
            pdf.savefig(fig, bbox_inches='tight')
    if save != None:
        pdf.close()
    if plot == False:
    	plt.close()

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', format='png', dpi=500, transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    return img
