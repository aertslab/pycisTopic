pycisTopic
==========

pycisTopic is a Python module to simultaneously identify cell states and cis-regulatory topics from single cell epigenomics data.

Under development.

Installation
**********************

To install pycisTopic::

	git clone https://github.com/aertslab/pycisTopic.git
	cd pycisTopic
	pip install . 
	
Depending on your pip version, you may need to run this pip command instead::

	pip install . --use-feature=in-tree-build


Creating a Docker/Singularity Image
-----------------------------------

To build a Docker image, then create a Singularity image from this::

    git clone https://github.com/aertslab/pycisTopic.git

    docker build -t aertslab/pycistopic:latest . -f pycisTopic/Dockerfile

    singularity build aertslab-pycistopic-latest.sif docker-daemon://aertslab/pycistopic:latest


Check version
**********************

To check your pycisTopic version::

	import pycisTopic
	pycisTopic.__version__

Tutorials
**********************

-  `Single sample - Complete workflow <https://htmlpreview.github.io/?https://github.com/aertslab/pycisTopic/blob/master/notebooks/Single_sample_workflow.html>`__
