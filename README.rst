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

	pip install -e .


Creating a Docker/Singularity Image
-----------------------------------

To build a Docker image, then create a Singularity image from this::

	# Clone repositories (pycisTopic and pycistarget)
	git clone https://github.com/aertslab/pycisTopic.git
	git clone https://github.com/aertslab/pycistarget.git

	# Copy your target Dockerfile to your workdir and build image
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

Tutorials
**********************

-  `Single sample - Complete workflow <https://htmlpreview.github.io/?https://github.com/aertslab/pycisTopic/blob/master/notebooks/Single_sample_workflow.html>`__
