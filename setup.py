from setuptools import find_packages, setup

fname = 'requirements.txt'
with open(fname, 'r', encoding='utf-8') as f:
        requirements =  f.read().splitlines()

required = []
dependency_links = []

# Do not add to required lines pointing to Git repositories
EGG_MARK = '#egg='
for line in requirements:
        if line.startswith('-e git:') or line.startswith('-e git+') or \
                line.startswith('git:') or line.startswith('git+'):
                line = line.lstrip('-e ')  # in case that is using "-e"
                if EGG_MARK in line:
                        package_name = line[line.find(EGG_MARK) + len(EGG_MARK):]
                        repository = line[:line.find(EGG_MARK)]
                        required.append('%s @ %s' % (package_name, repository))
                        dependency_links.append(line)
                else:
                        print('Dependency to a git repository should have the format:')
                        print('git+ssh://git@github.com/xxxxx/xxxxxx#egg=package_name')
        else:
                required.append(line)

setup(
    name='pycisTopic',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    include_dirs=["."],
    install_requires=required,
    dependency_links=dependency_links,
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
