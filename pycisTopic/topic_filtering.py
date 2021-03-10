# Take cell data
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import compress
import numpy as np

def compute_topic_metrics(cistopic_obj: 'CistopicObject',
                     return_metrics: Optional[bool] = True):
    model = cistopic_obj.selected_model
    topic_coh = model.coherence['Mimno_2011']
    try:
        topic_ass = model.topic_ass[['Assignments', 'Features_in_binarized_topic']]
    except:
        topic_ass = model.topic_ass['Assignments']
    marginal_dist = model.marg_topic['Marg_Topic']
    gini_values = pd.DataFrame([gini_coefficient(model.cell_topic.iloc[i,:].to_numpy()) for i in range(model.cell_topic.shape[0])])
    topic_qc_metrics = pd.concat([np.log10(topic_ass['Assignments']), topic_ass, topic_coh, marginal_dist, gini_values], axis=1)
    topic_qc_metrics.columns = ['Log10_Assignments', 'Assignments', 'Features_in_binarized_topic', 'Coherence', 'Marginal_topic_dist', 'Gini_index']
    topic_qc_metrics.index = ['Topic'+str(i) for i in range(1, model.cell_topic.shape[0]+1)]
    cistopic_obj.selected_model.topic_qc_metrics = topic_qc_metrics
    if return_metrics == True:
        return topic_qc_metrics

def topic_qc_plot(topic_qc_metrics: Union[pd.DataFrame, 'CistopicObject'], 
                                var_x: str,
                                var_y: Optional[str] = None,
                                min_x: Optional[int] = None,
                                max_x: Optional[int] = None,
                                min_y: Optional[int] = None,
                                max_y: Optional[int] = None,
                                var_color: Optional[str] = None,
                                cmap: Optional[str] = 'viridis',
                                dot_size: Optional[int] = 10,
                                text_size: Optional[int] = 10,
                                plot: Optional[bool] = True,
                                save: Optional[str] = None,
                                return_topics: Optional[bool] = False,
                                return_fig: Optional[bool] = False):
    
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
    
    if save != None:
        fig.savefig(save, bbox_inches='tight')
        
    if plot == True:
        plt.show()
    else:
        plt.close(fig)
        
    if return_topics == True:
        if return_fig == True:
            return fig, n
        else:
            return n
    else:
        if return_fig == True:
            return fig
    
def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))
