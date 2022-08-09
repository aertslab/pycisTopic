**************
Installation
**************

.. _installation:


To install pycisTopic::

	git clone https://github.com/aertslab/pycisTopic.git
	cd pycisTopic
	pip install -e . 
	

To check your pycisTopic version::

	import pycisTopic
	pycisTopic.__version__
	

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
	
.. warning::

	If you are planning to run the whole SCENIC+ workflow we recommend to install the SCENIC+ image. You can find more information at https://scenicplus.readthedocs.io/en/latest/install.html.

