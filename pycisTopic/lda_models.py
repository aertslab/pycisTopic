from gensim import utils, matutils, corpora
from gensim.models import basemodel
from gensim.models.ldamodel import LdaModel
from gensim.utils import check_output, revdict
from itertools import chain
import lda
import logging
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import random
import ray 
from scipy import sparse
import subprocess
import sys
import tempfile
import tmtoolkit
from typing import Optional, Union
from typing import List, Iterable, Tuple
import warnings
import xml.etree.ElementTree as et
import zipfile

from .cistopic_class import *
from .utils import *


class CistopicLDAModel:
    """
    cisTopic LDA model class
    
    :class:`cistopicLdaModel` contains model quality metrics (model coherence (adaptation from Mimno et al., 2011), log-likelihood (Griffiths and Steyvers, 2004), density-based (Cao Juan et al., 2009) and divergence-based (Arun et al., 2010)), topic quality metrics (coherence, marginal distribution and total number of assignments), cell-topic and topic-region distribution, model parameters and model dimensions.
    
    Parameters
    ----------
    metrics: pd.DataFrame
        :class:`pd.DataFrame` containing model quality metrics, including model coherence (adaptation from Mimno et al., 2011), log-likelihood and density and divergence-based methods (Cao Juan et al., 2009; Arun et al., 2010).
    coherence: pd.DataFrame
        :class:`pd.DataFrame` containing the coherence of each topic (Mimno et al., 2011).
    marginal_distribution: pd.DataFrame
        :class:`pd.DataFrame` containing the marginal distribution for each topic. It can be interpreted as the importance of each topic for the whole corpus.
    topic_ass: pd.DataFrame
        :class:`pd.DataFrame` containing the total number of assignments per topic.
    cell_topic: pd.DataFrame
        :class:`pd.DataFrame` containing the topic cell distributions, with cells as columns, topics as rows and the probability of each topic in each cell as values.
    topic_region: pd.DataFrame
        :class:`pd.DataFrame` containing the topic cell distributions, with topics as columns, regions as rows and the probability of each region in each topic as values.
    parameters: pd.DataFrame
        :class:`pd.DataFrame` containing parameters used for the model.
    n_cells: int
        Number of cells in the model.
    n_regions: int
        Number of regions in the model.
    n_topic: int
        Number of topics in the model.
        
    References
    ----------
    Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A. (2011). Optimizing semantic coherence in topic models. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing (pp. 262-272).
    
    Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. Proceedings of the National academy of Sciences, 101(suppl 1), 5228-5235.
    
    Cao, J., Xia, T., Li, J., Zhang, Y., & Tang, S. (2009). A density-based method for adaptive LDA model selection. Neurocomputing, 72(7-9), 1775-1781.
    
    Arun, R., Suresh, V., Madhavan, C. V., & Murthy, M. N. (2010). On finding the natural number of topics with latent dirichlet allocation: Some observations. In Pacific-Asia conference on knowledge discovery and data mining (pp. 391-402). Springer, Berlin, Heidelberg.
    """
    def __init__(self,
                 metrics: pd.DataFrame,
                 coherence: pd.DataFrame,
                 marg_topic: pd.DataFrame,
                 topic_ass: pd.DataFrame,
                 cell_topic: pd.DataFrame,
                 topic_region: pd.DataFrame,
                 parameters: pd.DataFrame):
        self.metrics = metrics
        self.coherence = coherence
        self.marg_topic = marg_topic
        self.topic_ass = topic_ass
        self.cell_topic = cell_topic
        self.cell_topic_harmony = []
        self.topic_region = topic_region
        self.parameters = parameters
        self.n_cells = cell_topic.shape[1]
        self.n_regions = topic_region.shape[0]
        self.n_topic = cell_topic.shape[0]
        
        
    def __str__(self):
        descr = f"CistopicLDAModel with {self.n_topic} topics and n_cells × n_regions = {self.n_cells} × {self.n_regions}"
        return(descr)

def run_cgs_models(cistopic_obj:'cisTopicObject',
                 n_topics:List[int],
                 n_cpu: Optional[int] = 1,
                 n_iter: Optional[int] = 150,
                 random_state: Optional[int] = 555,
                 alpha: Optional[float] = 50,
                 alpha_by_topic: Optional[bool] = True,
                 eta: Optional[float] = 0.1,
                 eta_by_topic: Optional[bool] = False,
                 top_topics_coh: Optional[int] = 5,
                 save_path: Optional[str] = None,
                 **kwargs):
    """
    Run Latent Dirichlet Allocation using Gibbs Sampling as described in Griffiths and Steyvers, 2004.
        
    Parameters
    ----------
    cistopic_obj: cisTopicObject
        A :class:`cisTopicObject`. Note that cells/regions have to be filtered before running any LDA model.
    n_topics: list of int
        A list containing the number of topics to use in each model.
    n_cpu: int, optional
        Number of cpus to use for modelling. In this function parallelization is done per model, that is, one model will run entirely in a unique cpu. We recommend to set the number of cpus as the number of models that will be inferred, so all models start at the same time.
    n_iter: int, optional
        Number of iterations for which the Gibbs sampler will be run. Default: 150.
    random_state: int, optional
        Random seed to initialize the models. Default: 555.
    alpha: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic proportions. Default: 50.
    alpha_by_topic: bool, optional
        Boolean indicating whether the scalar given in alpha has to be divided by the number of topics. Default: True
    eta: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic multinomials. Default: 0.1.
    eta_by_topic: bool, optional
        Boolean indicating whether the scalar given in beta has to be divided by the number of topics. Default: False
    top_topics_coh: int, optional
        Number of topics to use to calculate the model coherence. For each model, the coherence will be calculated as the average of the top coherence values. Default: 5.
    save_path: str, optional
        Path to save models as independent files as they are completed. This is recommended for large data sets. Default: None.
        
    Return
    ------
    list of :class:`CistopicLDAModel`
        A list with cisTopic LDA models.
        
    References
    ----------
    Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. Proceedings of the National academy of Sciences, 101(suppl 1), 5228-5235.
    """
    
    binary_matrix = cistopic_obj.binary_matrix.transpose()
    region_names = cistopic_obj.region_names
    cell_names = cistopic_obj.cell_names
    ray.init(num_cpus=n_cpu, **kwargs)
    model_list=ray.get([run_cgs_model.remote(binary_matrix,
                              n_topics=n_topic,
                              cell_names=cell_names,
                              region_names=region_names,
                              n_iter=n_iter,
                              random_state=random_state,
                              alpha=alpha,
                              alpha_by_topic=alpha_by_topic,
                              eta=eta,
                              eta_by_topic=eta_by_topic,
                              top_topics_coh = top_topics_coh,
                              save_path=save_path) for n_topic in n_topics])
    ray.shutdown()
    return model_list

@ray.remote
def run_cgs_model(binary_matrix: sparse.csr_matrix,
                n_topics: int,
                cell_names: List[str],
                region_names: List[str],
                n_iter: Optional[int]=150,
                random_state: Optional[int]=555,
                alpha: Optional[float]=50,
                alpha_by_topic: Optional[bool]=True,
                eta: Optional[float]=0.1,
                eta_by_topic: Optional[bool]=False,
                top_topics_coh: Optional[int] = 5,
                save_path: Optional[str] = None):
    """
    Run Latent Dirichlet Allocation per model using Gibbs Sampling as described in Griffiths and Steyvers, 2004.
        
    Parameters
    ----------
    binary_matrix: sparse.csr_matrix
        Binary sparse matrix containing cells as columns, regions as rows, and 1 if a regions is considered accesible on a cell (otherwise, 0).
    n_topics: int
        Number of topics to use in the model.
    cell_names: list of str
        List containing cell names as ordered in the binary matrix columns.
    region_names: list of str
        List containing region names as ordered in the binary matrix rows.
    n_iter: int, optional
        Number of iterations for which the Gibbs sampler will be run. Default: 150.
    random_state: int, optional
        Random seed to initialize the models. Default: 555.
    alpha: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic proportions. Default: 50.
    alpha_by_topic: bool, optional
        Boolean indicating whether the scalar given in alpha has to be divided by the number of topics. Default: True
    eta: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic multinomials. Default: 0.1.
    eta_by_topic: bool, optional
        Boolean indicating whether the scalar given in beta has to be divided by the number of topics. Default: False
    top_topics_coh: int, optional
        Number of topics to use to calculate the model coherence. For each model, the coherence will be calculated as the average of the top coherence values. Default: 5.
    save_path: str, optional
        Path to save models as independent files as they are completed. This is recommended for large data sets. Default: None.
    
    Return
    ------
    CistopicLDAModel
        A cisTopic LDA model.
    
    References
    ----------
    Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. Proceedings of the National academy of Sciences, 101(suppl 1), 5228-5235.
    """
    # Create cisTopic logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    # Suppress lda logger
    lda_log = logging.getLogger('lda')
    lda_log.addHandler(logging.NullHandler())
    lda_log.propagate = False
    warnings.filterwarnings('ignore')
    
    # Set models
    if alpha_by_topic == True and eta_by_topic == True:
        model=lda.LDA(n_topics=n_topics,
                      n_iter=n_iter,
                      random_state=random_state,
                      alpha=alpha/n_topics,
                      eta=eta/n_topics, 
                      refresh=n_iter)
    elif alpha_by_topic == True and eta_by_topic == False:
        model=lda.LDA(n_topics=n_topics,
                      n_iter=n_iter,
                      random_state=random_state,
                      alpha=alpha/n_topics,
                      eta=eta, 
                      refresh=n_iter)
    elif alpha_by_topic == False and eta_by_topic == True:
        model=lda.LDA(n_topics=n_topics,
                      n_iter=n_iter,
                      random_state=random_state,
                      alpha=alpha,
                      eta=eta/n_topics, 
                      refresh=n_iter)
    else:
        model=lda.LDA(n_topics=n_topics,
                      n_iter=n_iter,
                      random_state=random_state,
                      alpha=alpha,
                      eta=eta, 
                      refresh=n_iter)
        
    # Running model
    log.info(f"Running model with {n_topics} topics")
    model.fit(binary_matrix)
    
    # Model evaluation
    arun_2010=tmtoolkit.topicmod.evaluate.metric_arun_2010(model.topic_word_, model.doc_topic_, np.asarray(binary_matrix.sum(axis=1)).astype(float))
    cao_juan_2009=tmtoolkit.topicmod.evaluate.metric_cao_juan_2009(model.topic_word_)
    mimno_2011=tmtoolkit.topicmod.evaluate.metric_coherence_mimno_2011(model.topic_word_, dtm=binary_matrix, top_n=20, eps=1e-12, normalize=True, return_mean=False)
    ll=loglikelihood(model.nzw_, model.ndz_, alpha, eta)
    
    # Organinze data
    if len(mimno_2011) <= top_topics_coh:
        metrics = pd.DataFrame([arun_2010, cao_juan_2009, np.mean(mimno_2011), ll], index=['Arun_2010', 'Cao_Juan_2009', 'Mimno_2011', 'loglikelihood'], columns=['Metric']).transpose()
    else:
        metrics = pd.DataFrame([arun_2010, cao_juan_2009, np.mean(mimno_2011[np.argpartition(mimno_2011, -top_topics_coh)[-top_topics_coh:]]), ll], index=['Arun_2010', 'Cao_Juan_2009', 'Mimno_2011', 'loglikelihood'], columns=['Metric']).transpose()
    coherence = pd.DataFrame([range(1, n_topics+1), mimno_2011], index=['Topic', 'Mimno_2011']).transpose()
    marg_topic = pd.DataFrame([range(1, n_topics+1), list(chain.from_iterable(tmtoolkit.topicmod.model_stats.marginal_topic_distrib(model.doc_topic_, binary_matrix.sum(axis=1)).tolist()))], index=['Topic', 'Marg_Topic']).transpose()
    topic_ass = pd.DataFrame.from_records([range(1, n_topics+1), model.nz_], index=['Topic', 'Assignments']).transpose()
    cell_topic = pd.DataFrame.from_records(model.doc_topic_, index = cell_names, columns=['Topic'+ str(i) for i in range(1, n_topics+1)]).transpose()
    topic_region = pd.DataFrame.from_records(model.topic_word_, columns = region_names, index=['Topic'+ str(i) for i in range(1, n_topics+1)]).transpose()
    parameters = pd.DataFrame(['lda', n_topics, n_iter, random_state, alpha, alpha_by_topic, eta, eta_by_topic, top_topics_coh], index=['package', 'n_topics', 'n_iter', 'random_state', 'alpha', 'alpha_by_topic', 'eta', 'eta_by_topic', 'top_topics_coh'], columns=['Parameter'])
    # Create object 
    model = CistopicLDAModel(metrics, coherence, marg_topic, topic_ass, cell_topic, topic_region, parameters)
    log.info(f"Model with {n_topics} topics done!")
    if isinstance(save_path, str):
        log.info(f"Saving model with {n_topics} topics at {save_path}")
        with open(os.path.join(save_path, 'Topic' + str(n_topics) + '.pkl'), 'wb') as f:
            pickle.dump(model, f)
    return model


class LDAMallet(utils.SaveLoad, basemodel.BaseTopicModel):
    """
    Wrapper class to run LDA models with Mallet. This class has been adapted from gensim (https://github.com/RaRe-Technologies/gensim/blob/27bbb7015dc6bbe02e00bb1853e7952ac13e7fe0/gensim/models/wrappers/ldamallet.py).
    
    Parameters
    ----------
    mallet_path: str
        Path to the mallet binary (e.g. /xxx/Mallet/bin/mallet).
    num_topics: int
        The number of topics to use in the model.
    corpus: iterable of iterable of (int, int), optional
        Collection of texts in BoW format. Default: None.
    alpha: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic proportions. Default: 50.
    id2word : :class:`gensim.corpora.dictionary.Dictionary`, optional
        Mapping between tokens ids and words from corpus, if not specified - will be inferred from `corpus`. Default: None.
    n_cpu : int, optional
        Number of threads that will be used for training. Default: 1.
    tmp_dir : str, optional
        tmp_dir for produced temporary files. Default: None.
    optimize_interval : int, optional
        Optimize hyperparameters every `optimize_interval` iterations (sometimes leads to Java exception 0 to switch off hyperparameter optimization). Default: 0.
    iterations : int, optional
        Number of training iterations. Default: 150.
    topic_threshold : float, optional
        Threshold of the probability above which we consider a topic. Default: 0.0.
    random_seed: int, optional
        Random seed to ensure consistent results, if 0 - use system clock. Default: 555.
    """
    
    def __init__(self,
                 mallet_path: str,
                 num_topics: int,
                 corpus: Optional[Iterable] = None,
                 alpha: Optional[float]=50,
                 eta: Optional[float]=0.1,
                 id2word: Optional['gensim.corpora.dictionary.Dictionary'] = None,
                 n_cpu: Optional[int]=1,
                 tmp_dir: Optional[str]=None,
                 optimize_interval: Optional[int]=0,
                 iterations: Optional[int]=150,
                 topic_threshold: Optional[float]=0.0,
                 random_seed: Optional[int]=555):

        logger = logging.getLogger('LDAMalletWrapper')
        self.mallet_path = mallet_path
        self.id2word = id2word
        if self.id2word is None:
            logger.warning("No word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        else:
            self.num_terms = 0 if not self.id2word else 1 + max(self.id2word.keys())
        if self.num_terms == 0:
            raise ValueError("Cannot compute LDA over an empty collection (no terms)")
        self.num_topics = num_topics
        self.topic_threshold = topic_threshold
        self.alpha = alpha
        self.eta = eta
        if tmp_dir is None:
            tmp_dir = os.path.join(tempfile.gettempdir()) + '/'
        self.tmp_dir = tmp_dir
        self.random_label = hex(random.randint(0, 0xffffff))[2:] + '_'
        self.n_cpu = n_cpu
        self.optimize_interval = optimize_interval
        self.iterations = iterations
        self.random_seed = random_seed
        if corpus is not None:
            self.train(corpus)

    def corpus2mallet(self, corpus, file_like):
        """
        Convert `corpus` to Mallet format and write it to `file_like` descriptor.
        
            
        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
        Collection of texts in BoW format.
        file_like : file-like object.
        """
        for docno, doc in enumerate(corpus):
            if self.id2word:
                tokens = chain.from_iterable([self.id2word[tokenid]] * int(cnt) for tokenid, cnt in doc)
            else:
                tokens = chain.from_iterable([str(tokenid)] * int(cnt) for tokenid, cnt in doc)
            file_like.write(utils.to_utf8("%s 0 %s\n" % (docno, ' '.join(tokens))))

    def convert_input(self, corpus, infer=False, serialize_corpus=True):
        """
        Convert corpus to Mallet format and save it to a temporary text file.
            
        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
        Collection of texts in BoW format.
        infer : bool, optional
        serialize_corpus : bool, optional
        """
        logger = logging.getLogger('LDAMalletWrapper')
        if serialize_corpus:
            logger.info("Serializing temporary corpus to %s", self.fcorpustxt())
            with utils.open(self.fcorpustxt(), 'wb') as fout:
                self.corpus2mallet(corpus, fout)
                    
        cmd = \
            self.mallet_path + \
            " import-file --preserve-case --keep-sequence " \
            "--remove-stopwords --token-regex \"\\S+\" --input %s --output %s"
        if infer:
            cmd += ' --use-pipe-from ' + self.fcorpusmallet()
            cmd = cmd % (self.fcorpustxt(), self.fcorpusmallet() + '.infer')
        else:
            cmd = cmd % (self.fcorpustxt(), self.fcorpusmallet())
        logger.info("Converting temporary corpus to MALLET format with %s", cmd)
        check_output(args=cmd, shell=True)

    def train(self, corpus):
        """
        Train Mallet LDA.
            
        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Corpus in BoW format
        """
        logger = logging.getLogger('LDAMalletWrapper')
        if os.path.isfile(self.fcorpusmallet()) is False:
            self.convert_input(corpus, infer=False)
        else:
            logger.info("MALLET corpus already exists, training model")
        cmd = self.mallet_path + ' train-topics --input %s --num-topics %s  --alpha %s --beta %s --optimize-interval %s '\
            '--num-threads %s --output-state %s --output-doc-topics %s --output-topic-keys %s '\
            '--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s  --random-seed %s'
                    
        cmd = cmd % (
            self.fcorpusmallet(), self.num_topics, self.alpha, self.eta, self.optimize_interval,
            self.n_cpu, self.fstate(), self.fdoctopics(), self.ftopickeys(), self.iterations,
            self.finferencer(), self.topic_threshold, str(self.random_seed)
        )
        logger.info("Training MALLET LDA with %s", cmd)
        cmd = cmd.split()
        try:
            subprocess.check_output(args=cmd, shell=False, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        self.word_topics = self.load_word_topics()
        self.wordtopics = self.word_topics

    def load_word_topics(self):
        """
        Load words X topics matrix from :meth:`gensim.models.wrappers.LDAMallet.LDAMallet.fstate` file.
            
        Returns
        -------
        np.ndarray
            Matrix words X topics.
        """
        logger = logging.getLogger('LDAMalletWrapper')
        logger.info("loading assigned topics from %s", self.fstate())
        word_topics = np.zeros((self.num_topics, self.num_terms), dtype=np.float64)
        if hasattr(self.id2word, 'token2id'):
            word2id = self.id2word.token2id
        else:
            word2id = revdict(self.id2word)
                                
        with utils.open(self.fstate(), 'rb') as fin:
            _ = next(fin)  # header
            self.alpha = np.fromiter(next(fin).split()[2:], dtype=float)
            assert len(self.alpha) == self.num_topics, "Mismatch between MALLET vs. requested topics"
            _ = next(fin)  # noqa:F841 beta
            for lineno, line in enumerate(fin):
                line = utils.to_unicode(line)
                doc, source, pos, typeindex, token, topic = line.split(" ")
                if token not in word2id:
                    continue
                tokenid = word2id[token]
                word_topics[int(topic), tokenid] += 1.0
        return word_topics


    def get_topics(self):
        """
        Get topics X words matrix.
            
        Returns
        -------
        np.ndarray
            Topics X words matrix, shape `num_topics` x `vocabulary_size`.
        """
        topics = self.word_topics
        return topics / topics.sum(axis=1)[:, None]

    def fcorpustxt(self):
        """
        Get path to corpus text file.
            
        Returns
        -------
        str
            Path to corpus text file.
        """
        return self.tmp_dir + 'corpus.txt'

    def fcorpusmallet(self):
        """
        Get path to corpus.mallet file.
                
        Returns
        -------
        str
            Path to corpus.mallet file.
        """
        return self.tmp_dir + 'corpus.mallet'

    def fstate(self):
        """
        Get path to temporary file.
            
        Returns
        -------
        str
            Path to file.
            
        """
        return self.tmp_dir + self.random_label + 'state.mallet.gz'

    def fdoctopics(self):
        """
        Get path to document topic text file.
                
        Returns
        -------
        str
            Path to document topic text file.
        """
        return self.tmp_dir + self.random_label + 'doctopics.txt'

    def finferencer(self):
        """
        Get path to inferencer.mallet file.
            
        Returns
        -------
        str
            Path to inferencer.mallet file.
        """
        return self.tmp_dir + self.random_label + 'inferencer.mallet'
            
    def ftopickeys(self):
        """
        Get path to topic keys text file.
                
        Returns
        -------
        str
            Path to topic keys text file.
                
        """
        return self.tmp_dir + self.random_label + 'topickeys.txt'


def run_cgs_models_mallet(path_to_mallet_binary: str,
                       cistopic_obj: 'cisTopicObject',
                       n_topics: List[int],
                       n_cpu: Optional[int]=1,
                       n_iter: Optional[int]=150,
                       random_state: Optional[int]=555,
                       alpha: Optional[float]=50,
                       alpha_by_topic: Optional[bool]=True,
                       eta: Optional[float]=0.1,
                       eta_by_topic: Optional[bool]=False,
                       top_topics_coh: Optional[int]=5,
                       tmp_path: Optional[str]=None,
                       save_path: Optional[str]=None):
    """
    Run Latent Dirichlet Allocation per model as implemented in Mallet (McCallum, 2002).
        
    Parameters
    ----------
    path_to_mallet_binary: str
        Path to the mallet binary (e.g. /xxx/Mallet/bin/mallet).
    cistopic_obj: cisTopicObject
        A :class:`cisTopicObject`. Note that cells/regions have to be filtered before running any LDA model.
    n_topics: list of int
        A list containing the number of topics to use in each model.
    n_cpu: int, optional
        Number of cpus to use for modelling. In this function parallelization is done per model, that is, one model will run entirely in a unique cpu. We recommend to set the number of cpus as the number of models that will be inferred, so all models start at the same time.
    n_iter: int, optional
        Number of iterations for which the Gibbs sampler will be run. Default: 150.
    random_state: int, optional
        Random seed to initialize the models. Default: 555.
    alpha: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic proportions. Default: 50.
    alpha_by_topic: bool, optional
        Boolean indicating whether the scalar given in alpha has to be divided by the number of topics. Default: True
    eta: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic multinomials. Default: 0.1.
    eta_by_topic: bool, optional
        Boolean indicating whether the scalar given in beta has to be divided by the number of topics. Default: False
    top_topics_coh: int, optional
        Number of topics to use to calculate the model coherence. For each model, the coherence will be calculated as the average of the top coherence values. Default: 5.
    tmp_path: str, optional
        Path to a temporary folder for Mallet. Default: None.
    save_path: str, optional
        Path to save models as independent files as they are completed. This is recommended for large data sets. Default: None.
    
    Return
    ------
    list of :class:`CistopicLDAModel`
        A list with cisTopic LDA models.
    
    References
    ----------
    McCallum, A. K. (2002). Mallet: A machine learning for language toolkit. http://mallet.cs.umass.edu.
    """
    # Create cisTopic logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    binary_matrix = cistopic_obj.binary_matrix
    region_names = cistopic_obj.region_names
    cell_names = cistopic_obj.cell_names
    
    log.info(f"Formatting input to corpus")
    corpus = matutils.Sparse2Corpus(binary_matrix)
    names_dict = {x:str(x) for x in range(len(region_names))}
    id2word = corpora.Dictionary.from_corpus(corpus, names_dict)
    
    model_list=[run_cgs_model_mallet(path_to_mallet_binary, binary_matrix,
    							  corpus,
    							  id2word,
                                  n_topics=n_topic,
                                  cell_names=cell_names,
                                  region_names=region_names,
                                  n_cpu=n_cpu,
                                  n_iter=n_iter,
                                  random_state=random_state,
                                  alpha=alpha,
                                  alpha_by_topic=alpha_by_topic,
                                  eta=eta,
                                  eta_by_topic=eta_by_topic,
                                  top_topics_coh=top_topics_coh,
                                  tmp_path=tmp_path,
                                  save_path=save_path) for n_topic in n_topics]
    return model_list

def run_cgs_model_mallet(path_to_mallet_binary: str,
                      binary_matrix: sparse.csr_matrix,
                      corpus: Iterable,
                      id2word: 'gensim.corpora.dictionary.Dictionary',
                      n_topics: List[int],
                      cell_names: List[str],
                      region_names: List[str],
                      n_cpu: Optional[int]=1,
                      n_iter: Optional[int]=500,
                      random_state: Optional[int]=555,
                      alpha: Optional[float]=50,
                      alpha_by_topic: Optional[bool]=True,
                      eta: Optional[float]=0.1,
                      eta_by_topic: Optional[bool]=False,
                      top_topics_coh: Optional[int]=5,
                      tmp_path: Optional[str]=None,
                      save_path: Optional[str]=None):
    
    """
    Run Latent Dirichlet Allocation in a model as implemented in Mallet (McCallum, 2002).
        
    Parameters
    ----------
    path_to_mallet_binary: str
        Path to the mallet binary (e.g. /xxx/Mallet/bin/mallet).
    binary_matrix: sparse.csr_matrix
        Binary sparse matrix containing cells as columns, regions as rows, and 1 if a regions is considered accesible on a cell (otherwise, 0).
    n_topics: list of int
        A list containing the number of topics to use in each model.
    cell_names: list of str
        List containing cell names as ordered in the binary matrix columns.
    region_names: list of str
        List containing region names as ordered in the binary matrix rows.
    n_cpu: int, optional
        Number of cpus to use for modelling. In this function parallelization is done per model, that is, one model will run entirely in a unique cpu. We recommend to set the number of cpus as the number of models that will be inferred, so all models start at the same time.
    n_iter: int, optional
        Number of iterations for which the Gibbs sampler will be run. Default: 150.
    random_state: int, optional
        Random seed to initialize the models. Default: 555.
    alpha: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic proportions. Default: 50.
    alpha_by_topic: bool, optional
        Boolean indicating whether the scalar given in alpha has to be divided by the number of topics. Default: True
    eta: float, optional
        Scalar value indicating the symmetric Dirichlet hyperparameter for topic multinomials. Default: 0.1.
    eta_by_topic: bool, optional
        Boolean indicating whether the scalar given in beta has to be divided by the number of topics. Default: False
    top_topics_coh: int, optional
        Number of topics to use to calculate the model coherence. For each model, the coherence will be calculated as the average of the top coherence values. Default: 5.
    tmp_path: str, optional
        Path to a temporary folder for Mallet. Default: None.
    save_path: str, optional
        Path to save models as independent files as they are completed. This is recommended for large data sets. Default: None.
        
    Return
    ------
    CistopicLDAModel
        A cisTopic LDA model.
        
    References
    ----------
    McCallum, A. K. (2002). Mallet: A machine learning for language toolkit. http://mallet.cs.umass.edu.
    """
    # Create cisTopic logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    # Set models
    if alpha_by_topic == False:
        alpha=alpha*n_topics
    if eta_by_topic == True:
        eta=eta/n_topics
    
    # Running model
    log.info(f"Running model with {n_topics} topics")
    model = LDAMallet(path_to_mallet_binary, corpus=corpus, id2word=id2word, num_topics=n_topics, iterations=n_iter, alpha=alpha, eta=eta, n_cpu=n_cpu, tmp_dir=tmp_path, random_seed=random_state)

    # Get distributions
    topic_word = model.get_topics()
    doc_topic = pd.read_csv(model.fdoctopics(), header=None, sep='\t').iloc[:,2:].to_numpy()

    # Model evaluation
    cell_cov=np.asarray(binary_matrix.sum(axis=0)).astype(float)
    arun_2010=tmtoolkit.topicmod.evaluate.metric_arun_2010(topic_word, doc_topic, cell_cov)
    cao_juan_2009=tmtoolkit.topicmod.evaluate.metric_cao_juan_2009(topic_word)
    mimno_2011=tmtoolkit.topicmod.evaluate.metric_coherence_mimno_2011(topic_word, dtm=binary_matrix.transpose(), top_n=20, eps=1e-12, normalize=True, return_mean=False)
    topic_word_assig=model.word_topics
    doc_topic_assig=(doc_topic.T * (cell_cov)).T
    ll=loglikelihood(topic_word_assig, doc_topic_assig, alpha, eta)

    # Organinze data
    if len(mimno_2011) <= top_topics_coh:
        metrics = pd.DataFrame([arun_2010, cao_juan_2009, np.mean(mimno_2011), ll], index=['Arun_2010', 'Cao_Juan_2009', 'Mimno_2011', 'loglikelihood'], columns=['Metric']).transpose()
    else:
        metrics = pd.DataFrame([arun_2010, cao_juan_2009, np.mean(mimno_2011[np.argpartition(mimno_2011, -top_topics_coh)[-top_topics_coh:]]), ll], index=['Arun_2010', 'Cao_Juan_2009', 'Mimno_2011', 'loglikelihood'], columns=['Metric']).transpose()
    coherence = pd.DataFrame([range(1, n_topics+1), mimno_2011], index=['Topic', 'Mimno_2011']).transpose()
    marg_topic = pd.DataFrame([range(1, n_topics+1), tmtoolkit.topicmod.model_stats.marginal_topic_distrib(doc_topic, cell_cov)], index=['Topic', 'Marg_Topic']).transpose()
    topic_ass = pd.DataFrame.from_records([range(1, n_topics+1),  list(chain.from_iterable(model.word_topics.sum(axis=1)[:,None]))], index=['Topic', 'Assignments']).transpose()
    cell_topic = pd.DataFrame.from_records(doc_topic, index = cell_names, columns=['Topic'+ str(i) for i in range(1, n_topics+1)]).transpose()
    topic_region = pd.DataFrame.from_records(topic_word, columns = region_names, index=['Topic'+ str(i) for i in range(1, n_topics+1)]).transpose()
    parameters = pd.DataFrame(['Mallet', n_topics, n_iter, random_state, alpha, alpha_by_topic, eta, top_topics_coh], index=['package', 'n_topics', 'n_iter', 'random_state', 'alpha', 'alpha_by_topic', 'eta', 'top_topics_coh'], columns=['Parameter'])
    # Create object
    model = CistopicLDAModel(metrics, coherence, marg_topic, topic_ass, cell_topic, topic_region, parameters)
    log.info(f"Model with {n_topics} topics done!")
    if isinstance(save_path, str):
        log.info(f"Saving model with {n_topics} topics at {save_path}")
        with open(os.path.join(save_path, 'Topic' + str(n_topics) + '.pkl'), 'wb') as f:
            pickle.dump(model, f)
    return model

def evaluate_models(models: List['CistopicLDAModel'],
                   select_model: Optional[int]=None,
                   return_model: Optional[bool]=True,
                   metrics: Optional[str]=['Minmo_2011', 'loglikelihood', 'Cao_Juan_2009', 'Arun_2010'],
                   min_topics_coh: Optional[int]=5,
                   plot: Optional[bool]=True,
                   figsize: Optional[Tuple[float, float]] = (6.4,4.8),
                   plot_metrics: Optional[bool]=False,
                   save: Optional[str]=None):
    """
    Model selection based on model quality metrics (model coherence (adaptation from Mimno et al., 2011), log-likelihood (Griffiths and Steyvers, 2004), density-based (Cao Juan et al., 2009) and divergence-based (Arun et al., 2010)).
        
    Parameters
    ----------
    models: list of :class:`CistopicLDAModel`
        A list containing cisTopic LDA models, as returned from run_cgs_models or run_cgs_modelsMallet.
    selected_model: int, optional
        Integer indicating the number of topics of the selected model. If not provided, the best model will be selected automatically based on the model quality metrics. Default: None.
    return_model: bool, optional
        Whether to return the selected model as :class:`CistopicLDAModel`
    metrics: list of str
        Metrics to use for plotting and model selection:
            Minmo_2011: Uses the average model coherence as calculated by Mimno et al (2011). In order to reduce the impact of the number of topics, we calculate the average coherence based on the top selected average values. The better the model, the higher coherence.
            log-likelihood: Uses the log-likelihood in the last iteration as calculated by Griffiths and Steyvers (2004). The better the model, the higher the log-likelihood.
            Arun_2010: Uses a density-based metric as in Arun et al (2010) using the topic-region distribution, the cell-topic distribution and the cell coverage. The better the model, the lower the metric.
            Cao_Juan_2009: Uses a divergence-based metric as in Cao Juan et al (2009) using the topic-region distribution. The better the model, the lower the metric.
        Default: all metrics.
    min_topics_coh: int, optional
        Minimum number of topics on a topic to use its coherence for model selection. Default: 5.
    plot: bool, optional
        Whether to return plot to the console. Default: True.
    figsize: tuple, optional
		Size of the figure. Default: (6.4, 4.8)
    plot_metrics: bool, optional
        Whether to plot metrics independently. Default: False.
    save: str, optional
        Output file to save plot. Default: None.
        
    Return
    ------
    plot
        Plot with the combined metrics in which the best model should have high values for all metrics (Arun_2010 and Cao_Juan_2011 are inversed).
        
    References
    ----------
    Mimno, D., Wallach, H., Talley, E., Leenders, M., & McCallum, A. (2011). Optimizing semantic coherence in topic models. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing (pp. 262-272).
        
    Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. Proceedings of the National academy of Sciences, 101(suppl 1), 5228-5235
        
    Cao, J., Xia, T., Li, J., Zhang, Y., & Tang, S. (2009). A density-based method for adaptive LDA model selection. Neurocomputing, 72(7-9), 1775-1781.
        
    Arun, R., Suresh, V., Madhavan, C. V., & Murthy, M. N. (2010). On finding the natural number of topics with latent dirichlet allocation: Some observations. In Pacific-Asia conference on knowledge discovery and data mining (pp. 391-402). Springer, Berlin, Heidelberg.
    """
    models=[models[i] for i in np.argsort([m.n_topic for m in models])]
    all_topics=sorted([models[x].n_topic for x in range(0, len(models))])
    metrics_dict = {}
    fig = plt.figure(figsize=figsize)
    if 'Minmo_2011' in metrics:
        in_index = [i for i in range(len(all_topics)) if all_topics[i] >= min_topics_coh]
    if 'Arun_2010' in metrics:
        arun_2010 = [models[index].metrics.loc['Metric', 'Arun_2010'] for index in range(0, len(all_topics))]
        arun_2010_negative = [ -x for x in arun_2010]
        arun_2010_rescale = (arun_2010_negative-min(arun_2010_negative))/(max(arun_2010_negative)-min(arun_2010_negative))
        if 'Minmo_2011' in metrics:
            metrics_dict['Arun_2010'] = np.array(subset_list(arun_2010_rescale, in_index))
        else:
            metrics_dict['Arun_2010'] = arun_2010_rescale
        plt.plot(all_topics, arun_2010_rescale, linestyle='--', marker='o', label='Inv_Arun_2010')
    
    if 'Cao_Juan_2009' in metrics:
        Cao_Juan_2009 = [models[index].metrics.loc['Metric', 'Cao_Juan_2009'] for index in range(0, len(all_topics))]
        Cao_Juan_2009_negative = [ -x for x in Cao_Juan_2009 ]
        Cao_Juan_2009_rescale = (Cao_Juan_2009_negative-min(Cao_Juan_2009_negative))/(max(Cao_Juan_2009_negative)-min(Cao_Juan_2009_negative))
        if 'Minmo_2011' in metrics:
            metrics_dict['Cao_Juan_2009'] = np.array(subset_list(Cao_Juan_2009_rescale, in_index))
        else:
            metrics_dict['Cao_Juan_2009'] = Cao_Juan_2009_rescale
        plt.plot(all_topics, Cao_Juan_2009_rescale, linestyle='--', marker='o', label='Inv_Cao_Juan_2009')
    
    if 'Minmo_2011' in metrics:
        Mimno_2011 = [models[index].metrics.loc['Metric', 'Mimno_2011'] for index in range(0, len(all_topics))]
        Mimno_2011 = subset_list(Mimno_2011, in_index)
        Mimno_2011_all_topics = subset_list(all_topics, in_index)
        Mimno_2011_rescale = (Mimno_2011-min(Mimno_2011))/(max(Mimno_2011)-min(Mimno_2011))
        metrics_dict['Minmo_2011'] = np.array(Mimno_2011_rescale)
        plt.plot(Mimno_2011_all_topics, Mimno_2011_rescale, linestyle='--', marker='o', label='Mimno_2011')
    
    if 'loglikelihood' in metrics:
        loglikelihood = [models[index].metrics.loc['Metric', 'loglikelihood'] for index in range(0, len(all_topics))]
        loglikelihood_rescale = (loglikelihood-min(loglikelihood))/(max(loglikelihood)-min(loglikelihood))
        if 'Minmo_2011' in metrics:
            metrics_dict['loglikelihood'] = np.array(subset_list(loglikelihood_rescale, in_index))
        else:
            metrics_dict['loglikelihood'] = loglikelihood_rescale
        plt.plot(all_topics, loglikelihood_rescale, linestyle='--', marker='o', label='Loglikelihood')
    
    if select_model == None:
        combined_metric = sum(metrics_dict.values())
        if 'Minmo_2011' in metrics:
            best_model = Mimno_2011_all_topics[combined_metric.tolist().index(max(combined_metric))]
        else:
            best_model = all_topics[combined_metric.tolist().index(max(combined_metric))]
    else:
        combined_metric = None
        best_model = select_model
    
    plt.axvline(best_model, linestyle='--', color='grey')
    plt.xlabel('Number of topics\nOptimal number of topics: '+ str(best_model))
    plt.ylabel('Rescaled metric')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    if save != None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)
        pdf.savefig(fig, bbox_inches='tight')
    if plot is True:
        plt.show()
    else:
        plt.close(fig)
    
    if plot_metrics == True:
        if 'Arun_2010' in metrics:
            fig=plt.figure()
            plt.plot(all_topics, arun_2010, linestyle='--', marker='o')
            plt.axvline(best_model, linestyle='--', color='grey')
            plt.title('Arun_2010 - Minimize')
            if save != None:
                pdf.savefig(fig)
            plt.show()
        
        if 'Cao_Juan_2009' in metrics:
            fig=plt.figure()
            plt.plot(all_topics, Cao_Juan_2009, linestyle='--', marker='o')
            plt.axvline(best_model, linestyle='--', color='grey')
            plt.title('Cao_Juan_2009 - Minimize')
            if save != None:
                pdf.savefig(fig)
            plt.show()
        if 'Minmo_2011' in metrics:
            fig=plt.figure()
            plt.plot(Mimno_2011_all_topics,  Mimno_2011, linestyle='--', marker='o')
            plt.axvline(best_model, linestyle='--', color='grey')
            plt.title('Mimno_2011 - Maximize')
            if save != None:
                pdf.savefig(fig)
            plt.show()
        
        if 'loglikelihood' in metrics:
            fig=plt.figure()
            plt.plot(all_topics,  loglikelihood, linestyle='--', marker='o')
            plt.axvline(best_model, linestyle='--', color='grey')
            plt.title('Loglikelihood - Maximize')
            if save != None:
                pdf.savefig(fig)
            plt.show()

    if save != None:
        pdf.close()
    
    if return_model == True:
        return models[all_topics.index(best_model)]




