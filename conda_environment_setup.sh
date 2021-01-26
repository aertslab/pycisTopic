ssh ...
my_conda_initialize

# Create conda environment 
conda create -n pyCisTopic_env python=3.7.4

# Install dependencies (& any extra packages)
conda activate pyCisTopic_env

conda install numpy pandas matplotlib seaborn scipy 
conda install -c bioconda pyBigWig pyranges pybedtools pyfasta umap harmonypy scanorama ray
conda install -c conda-forge loompy igraph leidenalg lda IPython gensim networkx typing fit-sne
conda install ipykernel  # for Jupyter
pip install -U "tmtoolkit[recommended]"

## pyScenic (required for pyCisTopic)
conda install -c anaconda xlrd cytoolz
pip install pyscenic

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
# pip install scanpy   #  ERROR: anndata 0.7.5 has requirement pandas!=1.1,>=1.0, but you'll have pandas 0.25.3 which is incompatible.


      # Install environment as kernel for Jupyter???
      # pip install --user ipykernel
      # python -m ipykernel install --user --name=pyCisTopic

      #DL:     pydot shap tensorflow
