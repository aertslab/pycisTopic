ssh ...
my_conda_initialize

# Create conda environment 
conda create -n pycisTopic_env python=3.7.4

# Install dependencies (& any extra packages)
conda activate pycisTopic_env

conda install numpy pandas matplotlib seaborn scipy 
conda install -c bioconda pyBigWig pyranges pybedtools pyfasta umap harmonypy scanorama pybiomart 
conda install -c conda-forge loompy igraph python-igraph leidenalg lda IPython gensim networkx typing adjusttext gcc openssl 
conda install ipykernel  # for Jupyter
pip install ray # https://docs.ray.io/en/master/installation.html
pip install -U "tmtoolkit[recommended]"
conda install -c conda-forge cython fftw # for fitsne
pip install fitsne

## pyscenic (required for pycisTopic)
conda install -c anaconda xlrd cytoolz
pip install pyscenic # incompatible version with the one required by tmtoolkit

# conda list
# These "imports" are not listed in conda but are available (i.e. default from python? or installed through another dependency?):
#      PIL sklearn pickle zipfile ssl xml random math operator base64 collections itertools json multiprocessing subprocess logging io os sys tempfile warnings

## install pycisTopic
cd $SAIBAR/software/ # (temporary)
git clone https://github.com/aertslab/pycisTopic
cd pycisTopic
python setup.py install

# To confirm whether it has been installed properly, try:
python
import pycisTopic

########
# Other packages installed:
## scanpy (https://scanpy.readthedocs.io/en/latest/installation.html) - not required?
conda install scikit-learn statsmodels numba pytables
conda install -c conda-forge python-igraph louvain multicore-tsne
pip install scanpy   #  requires pandas!=1.1,>=1.0

module load GCC
pip install scrublet

#########
# trying to run tutorial... started complainin about missing...
pip install ray # https://docs.ray.io/en/master/installation.html
# needed to re-install (why??):
conda install -c conda-forge gensim 
conda install smart_open==2.0.0
pip install lda  # no longer works with conda-forge...
conda install cython
conda install -c bioconda pyBigWig 
pip install pyscenic # tmtoolkit 0.10.0 requires pandas<1.2,>=1.1.0, but you have pandas 0.25.3 which is incompatible.
pip install pybiomart
conda install -c bioconda harmonypy
conda install -c conda-forge python-igraph 
conda install -c conda-forge leidenalg
conda install pandas==1.2.0  # important

#########
# After installing any new package, check pandas version. It keeps getting downgraded:
conda list | grep pandas


