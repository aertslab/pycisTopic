# Take cell data
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import compress
import numpy as np

def topic_selection(cistopic_obj, min_coherence=None, max_coherence=None, min_assignments=None, max_assignments=None, cmap=cm.viridis, save=None):
    model=cistopic_obj.selected_model
    topic_coh=model.coherence
    topic_ass=model.topic_ass
    marginal_dist=model.marg_topic
    var_x='Coherence'
    var_y='Log assignments'
    x=topic_coh.Mimno_2011
    y=np.log10(topic_ass.Assignments)
    n=topic_coh.Topic.to_list()
    n=list(map(int, n))
    gini_values=[gini_coefficient(model.cell_topic.iloc[i,:].to_numpy()) for i in range(model.cell_topic.shape[0])]
    # Plot xy
    fig=plt.figure()
    plt.scatter(x, y, c=gini_values, cmap=cmap, vmin=0, vmax=1)
    plt.xlabel(var_x, fontsize=10)
    plt.ylabel(var_y, fontsize=10)
    # Add topic number
    for i, txt in enumerate(n):
        plt.annotate(txt, (x[i]+0.01, y[i]+0.02))
    
    # Add limits
    min_x=min_coherence
    min_y=min_assignments
    max_x=max_coherence
    max_y=max_assignments

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
    normalize = mcolors.Normalize(vmin=0, vmax=1)
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
    scalarmappaple.set_array(gini_values)
    cbar=plt.colorbar(scalarmappaple)  
    cbar.set_label('Gini index', rotation=270, labelpad=15)
    if save != None:
        fig.savefig(save, bbox_inches='tight')
    plt.show()
    return(n)
    
def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))
