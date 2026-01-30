.. image:: https://zenodo.org/badge/329905726.svg
   :target: https://zenodo.org/badge/latestdoi/329905726

pycisTopic
==========

pycisTopic is a Python module to simultaneously identify cell states and cis-regulatory topics from single cell epigenomics data.

Installation
************

pycisTopic can be installed with your environment manager of choice. One option is ``conda``::

	conda create --name pycistopic python=3.12 -y
	conda activate pycistopic
	git clone https://github.com/aertslab/pycisTopic.git
	cd pycisTopic
	pip install .

You can also install `SCENIC+`_, which includes pycisTopic::

	conda create --name scenicplus python=3.11 -y
	conda activate scenicplus
	git clone https://github.com/aertslab/scenicplus.git
	cd scenicplus
	pip install .

.. _SCENIC+: https://github.com/aertslab/scenicplus

Check version
*************

To check your pycisTopic version::

	import pycisTopic
	pycisTopic.__version__

Tutorials & documentation
*************************

Tutorial and documentation are available at https://pycistopic.readthedocs.io/.

Questions?
**********

* If you have **technical questions or problems**, such as bug reports or ideas for new features, please open an issue under the issues tab.
* If you have **questions about the interpretation of results or your analysis**, please start a Discussion under the Discussions tab.


Reference
*********

`Bravo González-Blas, C., De Winter, S., et al. SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks. Nat Methods 20, 1355–1367 (2023). <https://doi.org/10.1038/s41592-023-01938-4>`_
