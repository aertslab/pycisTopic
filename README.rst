pycisTopic
==========

pycisTopic is a Python module to simultaneously identify cell states and cis-regulatory topics from single cell epigenomics data.

Installation
**********************

To install pycisTopic::

	git clone https://github.com/aertslab/pycisTopic.git
	cd pycisTopic
	pip install . 
	
Depending on your pip version, you may need to run this pip command instead::

	pip install -e .


Creating a Docker/Singularity Image
-----------------------------------

To build a Docker image, then create a Singularity image from this::

	# Clone repositories (pycisTopic and pycistarget)
	git clone https://github.com/aertslab/pycisTopic.git
	git clone https://github.com/aertslab/pycistarget.git

	# Build image
	podman build -t aertslab/pycistopic:latest . -f pycisTopic/Dockerfile

	# Export to oci 
	podman save --format oci-archive --output pycistopic_img.tar localhost/aertslab/pycistopic

	# Build to singularity
	singularity build pycistopic.sif oci-archive://pycistopic_img.tar

	# Add all binding paths where you would need to access
	singularity exec -B /lustre1,/staging,/data,/vsc-hard-mounts,/scratch pycistopic.sif ipython3


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

:ref:`Bravo Gonzalez-Blas, C. & De Winter, S. *et al.* (2022). SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks<https://www.biorxiv.org/content/10.1101/2022.08.19.504505v1>``
