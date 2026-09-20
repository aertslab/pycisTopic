import pytest
import urllib.request
import urllib
import gzip
import os
import pathlib
import shutil

from pycistopic_lib import split_fragment_files_by_cell_type

TEST_DIRECTORY = pathlib.Path(__file__).parent.absolute()

FRAGMENT_FILEs = [
    TEST_DIRECTORY / pathlib.Path("atac_pbmc_500_nextgem_fragments.tsv.gz"),
    TEST_DIRECTORY / pathlib.Path("atac_pbmc_500_v1_fragments.tsv.gz")
]

INDEX_FILEs= [
    TEST_DIRECTORY / pathlib.Path("atac_pbmc_500_nextgem_fragments.tsv.gz.tbi"),
    TEST_DIRECTORY / pathlib.Path("atac_pbmc_500_v1_fragments.tsv.gz.tbi")
]

@pytest.fixture
def fragment_file(tmp_path):
    for f in FRAGMENT_FILEs:
        shutil.copy(f, tmp_path)
    return [tmp_path / f.name for f in FRAGMENT_FILEs]

@pytest.fixture
def fragment_file_and_index(tmp_path):
    for f in FRAGMENT_FILEs:
        shutil.copy(f, tmp_path)
    for f in INDEX_FILEs:
        shutil.copy(f, tmp_path)
    return [tmp_path / f.name for f in FRAGMENT_FILEs]

def count_number_of_fragments_per_chrom_for_cbs(file_name: str, cbs: list[str]):
    n_fragment_per_cb_per_chrom = {
        cb: {} for cb in cbs
    }
    with gzip.open(file_name) as f:
        for fragment in f:
            chrom, _, _, cb = fragment.decode().split()[0:4]
            if cb in cbs:
                if chrom not in n_fragment_per_cb_per_chrom[cb]:
                    n_fragment_per_cb_per_chrom[cb][chrom] = 0
                n_fragment_per_cb_per_chrom[cb][chrom] += 1
    return n_fragment_per_cb_per_chrom

def fragment_file_correct_order(file_name: str):
    chrom_to_start = {}
    with gzip.open(file_name) as f:
        for l in f:
            l = l.decode()
            if l.startswith("#"):
                continue
            chrom, start = l.split()[0:2]
            if chrom not in chrom_to_start:
                chrom_to_start[chrom] = []
            chrom_to_start[chrom].append(int(start))
    def always_inc(l):
        x = l[0]
        for y in l[1:]:
            if y < x:
                return False
        return True
    return all([always_inc(chrom_to_start[chrom]) for chrom in chrom_to_start])

def test_pseudobulk_indexed(fragment_file_and_index, tmp_path):
    fragment_files = fragment_file_and_index
    bcs_1 = []
    bcs_2 = []
    with open(TEST_DIRECTORY / pathlib.Path("bcs_nextgem.txt")) as f:
        for l in f:
            bcs_1.append(l.strip())

    with open(TEST_DIRECTORY / pathlib.Path("bcs_v1.txt")) as f:
        for l in f:
            bcs_2.append(l.strip())
    sample_to_cb = {
        str(fragment_files[0]): bcs_1,
        str(fragment_files[1]): bcs_2,
    }
    split_fragment_files_by_cell_type(
        fragment_file_paths=[str(f) for f in fragment_files],
        output_file_prefix=str(tmp_path) +  "/",
        cell_type_to_fragment_file_to_cell_barcode={
            "cell_type": sample_to_cb
        },
        chromosomes=[*[f"chr{x + 1}" for x in range(22)], "chrY", "chrX"]
    )
    assert(fragment_file_correct_order(tmp_path / "_cell_type.tsv.gz"))
    n_fragment_per_cb_per_chrom_outf = count_number_of_fragments_per_chrom_for_cbs(
        tmp_path / "_cell_type.tsv.gz",
        [*list(sample_to_cb.values())[0], *list(sample_to_cb.values())[1]]
    )
    n_fragment_per_cb_per_chrom_inf = {}
    for sample in fragment_files:
        n_fragment_per_cb_per_chrom_sample = count_number_of_fragments_per_chrom_for_cbs(
            sample, sample_to_cb[str(sample)])
        for cb in n_fragment_per_cb_per_chrom_sample:
            if cb not in n_fragment_per_cb_per_chrom_inf:
                n_fragment_per_cb_per_chrom_inf[cb] = {}
            for chrom in n_fragment_per_cb_per_chrom_sample[cb]:
                if chrom not in n_fragment_per_cb_per_chrom_inf[cb]:
                    n_fragment_per_cb_per_chrom_inf[cb][chrom] = 0
                n_fragment_per_cb_per_chrom_inf[cb][chrom] += n_fragment_per_cb_per_chrom_sample[cb][chrom]
    for cb in n_fragment_per_cb_per_chrom_inf:
        for chrom in n_fragment_per_cb_per_chrom_inf[cb]:
            assert(n_fragment_per_cb_per_chrom_inf[cb][chrom] == n_fragment_per_cb_per_chrom_outf[cb][chrom])


def test_pseudobulk_not_indexed(fragment_file, tmp_path):
    fragment_files = fragment_file
    bcs_1 = []
    bcs_2 = []
    with open(TEST_DIRECTORY / pathlib.Path("bcs_nextgem.txt")) as f:
        for l in f:
            bcs_1.append(l.strip())

    with open(TEST_DIRECTORY / pathlib.Path("bcs_v1.txt")) as f:
        for l in f:
            bcs_2.append(l.strip())
    sample_to_cb = {
        str(fragment_files[0]): bcs_1,
        str(fragment_files[1]): bcs_2,
    }
    split_fragment_files_by_cell_type(
        fragment_file_paths=[str(f) for f in fragment_files],
        output_file_prefix=str(tmp_path) +  "/",
        cell_type_to_fragment_file_to_cell_barcode={
            "cell_type": sample_to_cb
        },
        chromosomes=[*[f"chr{x + 1}" for x in range(22)], "chrY", "chrX"]
    )
    assert(fragment_file_correct_order(tmp_path / "_cell_type.tsv.gz"))
    n_fragment_per_cb_per_chrom_outf = count_number_of_fragments_per_chrom_for_cbs(
        tmp_path / "_cell_type.tsv.gz",
        [*list(sample_to_cb.values())[0], *list(sample_to_cb.values())[1]]
    )
    n_fragment_per_cb_per_chrom_inf = {}
    for sample in fragment_files:
        n_fragment_per_cb_per_chrom_sample = count_number_of_fragments_per_chrom_for_cbs(
            sample, sample_to_cb[str(sample)])
        for cb in n_fragment_per_cb_per_chrom_sample:
            if cb not in n_fragment_per_cb_per_chrom_inf:
                n_fragment_per_cb_per_chrom_inf[cb] = {}
            for chrom in n_fragment_per_cb_per_chrom_sample[cb]:
                if chrom not in n_fragment_per_cb_per_chrom_inf[cb]:
                    n_fragment_per_cb_per_chrom_inf[cb][chrom] = 0
                n_fragment_per_cb_per_chrom_inf[cb][chrom] += n_fragment_per_cb_per_chrom_sample[cb][chrom]
    for cb in n_fragment_per_cb_per_chrom_inf:
        for chrom in n_fragment_per_cb_per_chrom_inf[cb]:
            assert(n_fragment_per_cb_per_chrom_inf[cb][chrom] == n_fragment_per_cb_per_chrom_outf[cb][chrom])