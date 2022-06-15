from setuptools import find_packages, setup


def read_requirements(fname):
    with open(fname, 'r', encoding='utf-8') as file:
        return [line.rstrip() for line in file]


setup(
    name='pycisTopic',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    include_dirs=["."],
    install_requires=read_requirements('requirements.txt'),
    author="Carmen Bravo",
    author_email="carmen.bravogonzalezblas@kuleuven.be",
    description="pycisTopic is a Python module to simultaneously identify cell states and cis-regulatory topics from single cell epigenomics data.",
    long_description=open('README.rst').read(),
    url="https://github.com/aertslab/pycisTopic",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
