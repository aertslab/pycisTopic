# Installation

You need to have Python 3.XX or newer installed on your system. 

We recommend using [uv](https://docs.astral.sh/uv/) for package installation, which is significantly faster than pip and has better dependency resolution:

```bash
pip install uv
```

## Basic Installation 

To install pycisTopic run the following commands

1. Pull the github repository

```bash
# Clone pycisTopic git.
git clone https://github.com/aertslab/pycisTopic/
cd pycisTopic
```

2. Create a virtual environment within the pycisTopic directory

```bash
uv venv 
uv sync --extra development
```

3. Activate the virtual environment

```bash
source .venv/bin/activate
```

