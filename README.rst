.. image:: https://zenodo.org/badge/329905726.svg
   :target: https://zenodo.org/badge/latestdoi/329905726

pycisTopic
==========

pycisTopic is a Python module to simultaneously identify cell states and cis-regulatory topics from single cell epigenomics data.

Installation
**********************

To install pycisTopic::


	conda create --name scenicplus python=3.11 -y
	conda activate scenicplus
	git clone https://github.com/aertslab/scenicplus.git
	cd scenicplus
	git checkout development
	pip install .

Check version
**********************

To check your pycisTopic version::

	import pycisTopic
	pycisTopic.__version__

Tutorials & documentation
**********************

Tutorial and documentation are available at https://pycistopic.readthedocs.io/.

Questions?
**********************

* If you have **technical questions or problems**, such as bug reports or ideas for new features, please open an issue under the issues tab.
* If you have **questions about the interpretation of results or your analysis**, please start a Discussion under the Discussions tab.


Reference
**********************

`Bravo Gonzalez-Blas, C. & De Winter, S. et al. (2022). SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks <https://www.biorxiv.org/content/10.1101/2022.08.19.504505v1>`_
