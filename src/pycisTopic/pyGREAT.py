from __future__ import annotations

import os
import random
import sys
import tempfile

import pandas as pd
import pyranges as pr
import ray
import requests
from bs4 import BeautifulSoup
from pycisTopic.utils import coord_to_region_names, region_names_to_coordinates

# Set stderr to null when using ray.init to avoid ray printing Broken pipe million times
_stderr = sys.stderr
null = open(os.devnull, "wb")


def pyGREAT(
    region_sets: dict[str, pr.PyRanges],
    species: str,
    rule: str = "basalPlusExt",
    span: float = 1000.0,
    upstream: float = 5.0,
    downstream: float = 1.0,
    two_distance: float = 1000.0,
    one_distance: float = 1000.0,
    include_curated_reg_doms: int = 1,
    bg_choice: str = "wholeGenome",
    tmp_dir: str | None = None,
    n_cpu: int = 1,
    **kwargs
):
    """
    Running GREAT (McLean et al., 2010) on a dictionary of pyranges. For more details in GREAT parameters, please visit http://great.stanford.edu/public/html/.

    Parameters
    ----------
    region_sets: Dict
        A dictionary containing region sets to query as pyRanges objects.
    species: str
        Genome assembly from where the coordinates come from. Possible values are: 'mm9', 'mm10', 'hg19', 'hg38'
    rule: str
        How to associate genomic regions to genes. Possible options are 'basalPlusExt', 'twoClosest', 'oneClosest'. Default: 'basalPlusExt'
    span: float
        Unit: kb, only used when rule is 'basalPlusExt'. Default: 1000.0
    upstream: float
        Unit: kb, only used when rule is 'basalPlusExt'. Default: 5.0
    downstream: float
        Unit: kb, only used when rule is 'basalPlusExt'. Default: 1.0
    two_distance: float
        Unit: kb, only used when rule is 'twoClosest'. Default: 1000.0
    one_distance: float
        Unit: kb, only used when rule is 'oneClosest'. Default: 1000.0
    include_curated_reg_doms: int
        Whether to include curated regulatory domains. Default: 1
    bg_choice: str
        A path to the background file or a string. Default: 'wholeGenome'
    tmp_dir: str
        Temporary directory to save region sets as bed files for GREAT. Default: None
    n_cpu: int
        Number of cores to use. Default: 1
    ***kwargs
        Other parameters to pass to ray.init

    Return
    ------
    Dictionary with pyGREAT results

    References
    ----------
    McLean, C. Y., Bristor, D., Hiller, M., Clarke, S. L., Schaar, B. T., Lowe, C. B., ... & Bejerano, G. (2010). GREAT improves functional interpretation of
    cis-regulatory regions. Nature biotechnology, 28(5), 495-501.

    """
    if n_cpu > 1:
        ray.init(num_cpus=n_cpu, **kwargs)
        sys.stderr = null
        pyGREAT_dict = ray.get(
            [
                pyGREAT_oneset_ray.remote(
                    region_sets[key],
                    species,
                    rule,
                    span,
                    upstream,
                    downstream,
                    two_distance,
                    one_distance,
                    include_curated_reg_doms,
                    bg_choice,
                    tmp_dir,
                )
                for key in region_sets.keys()
            ]
        )
        ray.shutdown()
        sys.stderr = sys.__stderr__
        pyGREAT_dict = {
            key: pyGREAT_result
            for key, pyGREAT_result in zip(list(region_sets.keys()), pyGREAT_dict)
        }
    else:
        pyGREAT_dict = {
            key: pyGREAT_oneset(
                region_sets[key],
                species,
                rule,
                span,
                upstream,
                downstream,
                two_distance,
                one_distance,
                include_curated_reg_doms,
                bg_choice,
                tmp_dir,
            )
            for key in region_sets.keys()
        }
    return pyGREAT_dict


@ray.remote
def pyGREAT_oneset_ray(
    region_set: pr.PyRanges,
    species: str,
    rule: str = "basalPlusExt",
    span: float = 1000.0,
    upstream: float = 5.0,
    downstream: float = 1.0,
    two_distance: float = 1000.0,
    one_distance: float = 1000.0,
    include_curated_reg_doms: int = 1,
    bg_choice: str = "wholeGenome",
    tmp_dir: str | None = None,
):
    """
    Running GREAT (McLean et al., 2010) on a set of pyranges. For more details in GREAT parameters, please visit http://great.stanford.edu/public/html/.

    Parameters
    ----------
    region_sets: Dict
        A dictionary containing region sets to query as pyRanges objects.
    species: str
        Genome assembly from where the coordinates come from. Possible values are: 'mm9', 'mm10', 'hg19', 'hg38'
    rule: str
        How to associate genomic regions to genes. Possible options are 'basalPlusExt', 'twoClosest', 'oneClosest'. Default: 'basalPlusExt'
    span: float
        Unit: kb, only used when rule is 'basalPlusExt'. Default: 1000.0
    upstream: float
        Unit: kb, only used when rule is 'basalPlusExt'. Default: 5.0
    downstream: float
        Unit: kb, only used when rule is 'basalPlusExt'. Default: 1.0
    two_distance: float
        Unit: kb, only used when rule is 'twoClosest'. Default: 1000.0
    one_distance: float
        Unit: kb, only used when rule is 'oneClosest'. Default: 1000.0
    include_curated_reg_doms: int
        Whether to include curated regulatory domains. Default: 1
    bg_choice: str
        A path to the background file or a string. Default: 'wholeGenome'
    tmp_dir: str
        Temporary directory to save region sets as bed files for GREAT. Default: None

    Return
    ------
    Dictionary with pyGREAT results

    References
    ----------
    McLean, C. Y., Bristor, D., Hiller, M., Clarke, S. L., Schaar, B. T., Lowe, C. B., ... & Bejerano, G. (2010). GREAT improves functional interpretation of
    cis-regulatory regions. Nature biotechnology, 28(5), 495-501.

    """
    return pyGREAT_oneset(
        region_set,
        species,
        rule,
        span,
        upstream,
        downstream,
        two_distance,
        one_distance,
        include_curated_reg_doms,
        bg_choice,
        tmp_dir,
    )


def pyGREAT_oneset(
    region_set: pr.PyRanges,
    species: str,
    rule: str = "basalPlusExt",
    span: float = 1000.0,
    upstream: float = 5.0,
    downstream: float = 1.0,
    two_distance: float = 1000.0,
    one_distance: float = 1000.0,
    include_curated_reg_doms: int = 1,
    bg_choice: str = "wholeGenome",
    tmp_dir: str | None = None,
):
    """
    Running GREAT (McLean et al., 2010) on a pyranges object. For more details in GREAT parameters, please visit http://great.stanford.edu/public/html/.

    Parameters
    ----------
    region_sets: Dict
        A dictionary containing region sets to query as pyRanges objects.
    species: str
        Genome assembly from where the coordinates come from. Possible values are: 'mm9', 'mm10', 'hg19', 'hg38'
    rule: str
        How to associate genomic regions to genes. Possible options are 'basalPlusExt', 'twoClosest', 'oneClosest'. Default: 'basalPlusExt'
    span: float
        Unit: kb, only used when rule is 'basalPlusExt'. Default: 1000.0
    upstream: float
        Unit: kb, only used when rule is 'basalPlusExt'. Default: 5.0
    downstream: float
        Unit: kb, only used when rule is 'basalPlusExt'. Default: 1.0
    two_distance: float
        Unit: kb, only used when rule is 'twoClosest'. Default: 1000.0
    one_distance: float
        Unit: kb, only used when rule is 'oneClosest'. Default: 1000.0
    include_curated_reg_doms: int
        Whether to include curated regulatory domains. Default: 1
    bg_choice: str
        A path to the background file or a string. Default: 'wholeGenome'
    tmp_dir: str
        Temporary directory to save region sets as bed files for GREAT. Default: None
    n_cpu: int
        Number of cores to use. Default: 1
    ***kwargs
        Other parameters to pass to ray.init

    Return
    ------
    Dictionary with pyGREAT results

    References
    ----------
    McLean, C. Y., Bristor, D., Hiller, M., Clarke, S. L., Schaar, B. T., Lowe, C. B., ... & Bejerano, G. (2010). GREAT improves functional interpretation of
    cis-regulatory regions. Nature biotechnology, 28(5), 495-501.

    """
    # Params
    region_set.Name = coord_to_region_names(region_set)
    random_label = hex(random.randint(0, 0xFFFFFF))[2:]
    bed_file = os.path.join(
        tmp_dir if tmp_dir else tempfile.gettempdir(),
        f"{random_label}_great.bed"
    )
    region_set.df.to_csv(bed_file, sep="\t", index=False, header=False)

    # GREAT job
    url = "http://great.stanford.edu/public/cgi-bin/greatWeb.php"
    params = {
        "species": species,
        "rule": rule,
        "span": span,
        "upstream": upstream,
        "downstream": downstream,
        "twoDistance": two_distance,
        "oneDistance": one_distance,
        "includeCuratedRegDoms": include_curated_reg_doms,
        "bgChoice": "file" if (bg_choice != 'wholeGenome') else 'wholeGenome',
        "fgChoice": "file",
    }

    if bg_choice == 'wholeGenome':
        files = {"fgFile": open(bed_file, "r")}
    else:
        files = {"fgFile": open(bed_file, "r"), "bgFile": open(bg_choice, "r")}

    # Launch job
    r = requests.post(url, data=params, files=files)
    # Get results
    if r.status_code:
        soup = BeautifulSoup(r.text, "lxml")
        jobId = soup.find("div", attrs={"class": "job_desc_info"}).text

        downloadUrl = "http://great.stanford.edu/public/cgi-bin/downloadAllTSV.php"

        params = {
            "outputDir": "/scratch/great/tmp/results/" + jobId + ".d/",
            "outputPath": "/tmp/results/" + jobId + ".d/",
            "species": species,
            "ontoName": "",
            "ontoList": "GOMolecularFunction@1-Inf,GOBiologicalProcess@1-Inf,GOCellularComponent@1-Inf,MGIPhenotype@1-Inf,HumanPhenotypeOntology@1-Inf",
        }

        r2 = requests.post(downloadUrl, data=params)
        results = r2.text.split("\n")
        results = [x.split("\t") for x in results[3:]]
        pd_results = pd.DataFrame(results)
        pd_results = pd_results.rename(columns=pd_results.iloc[0]).drop(
            pd_results.index[0]
        )
        pd_results.rename(columns={"# Ontology": "Ontology"}, inplace=True)
        pd_results = pd_results[~pd_results.iloc[:, 0].str.contains("#")]
        pd_results = pd_results[~pd_results.iloc[:, 0].str.contains("Ensembl Genes")]
        pd_results = pd_results[~pd_results.iloc[:, 0].isin([""])]
        ontologies = set(pd_results.iloc[:, 0])
        pygreat_dict = {
            key: pd_results[pd_results.iloc[:, 0].isin([key])]
            .reset_index()
            .drop("index", axis=1)
            for key in ontologies
        }
        os.remove(bed_file)
        return pygreat_dict


def get_region_signature(
    pyGREAT_results: dict[str, pd.DataFrame],
    region_set_key: str,
    ontology: str,
    term: str,
):
    """
    Retriving GO region signature from GREAT results.

    Parameters
    ----------
    pyGREAT_results: Dict
        A dictionary with pyGREAT results.
    region_set_key: str
        Key of the region set to query
    ontology: str
        Ontology to query
    term: str
        Term to retrive regions from

    Return
    ------
    Signature as pyranges

    """
    data = pyGREAT_results[region_set_key][ontology]
    regions = list(
        set(data.loc[data.iloc[:, 2].isin([term]), "Regions"].tolist()[0].split(","))
    )
    return pr.PyRanges(region_names_to_coordinates(regions))
