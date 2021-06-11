import collections as cl
import logging
import numpy as np
import pandas as pd
import pyranges as pr
import ray
import sklearn.preprocessing as sp
import sys
from scipy import sparse
from typing import List, Dict
from typing import Optional, Union

from .lda_models import *
from .utils import *

dtype = pd.SparseDtype(int, fill_value=0)
pd.options.mode.chained_assignment = None


class CistopicObject:
    """
    cisTopic data class.

    :class:`CistopicObject` contains the cell by fragment matrices (stored as counts :attr:`fragment_matrix` and as binary accessibility :attr:`binary_matrix`),
    cell metadata :attr:`cell_data`, region metadata :attr:`region_data` and path/s to the fragments file/s :attr:`path_to_fragments`.

    LDA models from :class:`CisTopicLDAModel` can be stored :attr:`selected_model` as well as cell/region projections :attr:`projections` as a dictionary.

    Attributes
    ---------
    fragment_matrix: sparse.csr_matrix
        A matrix containing cell names as column names, regions as row names and fragment counts as values.
    binary_matrix: sparse.csr_matrix
        A matrix containing cell names as column names, regions as row names and whether regions as accessible (0: Not accessible; 1: Accessible) as values.
    cell_names: list
        A list containing cell names.
    region_names: list
        A list containing region names.
    cell_data: pd.DataFrame
        A data frame containing cell information, with cells as indexes and attributes as columns.
    region_data: pd.DataFrame
        A data frame containing region information, with region as indexes and attributes as columns.
    path_to_fragments: str or dict
        A list containing the paths to the fragments files used to generate the :class:`CistopicObject`.
    project: str
        Name of the cisTopic project.
    """

    def __init__(self,
                 fragment_matrix: sparse.csr_matrix,
                 binary_matrix: sparse.csr_matrix,
                 cell_names: List[str],
                 region_names: List[str],
                 cell_data: pd.DataFrame,
                 region_data: pd.DataFrame,
                 path_to_fragments: Union[str, Dict[str, str]],
                 project: Optional[str] = 'cisTopic'):
        self.fragment_matrix = fragment_matrix
        self.binary_matrix = binary_matrix
        self.cell_names = cell_names
        self.region_names = region_names
        self.cell_data = cell_data
        self.region_data = region_data
        self.project = project
        if isinstance(path_to_fragments, str):
            path_to_fragments = {project: path_to_fragments}
        self.path_to_fragments = path_to_fragments
        self.selected_model = []
        self.projections = {'cell': {}, 'region': {}}

    def __str__(self):
        descr = f"CistopicObject from project {self.project} with n_cells × n_regions = {len(self.cell_names)} × {len(self.region_names)}"
        return (descr)

    def add_cell_data(self,
                      cell_data: pd.DataFrame):
        """
        Add cell metadata to :class:`CistopicObject`. If the column already exist on the cell metadata, it will be overwritten.

        Parameters
        ---------
        cell_data: pd.DataFrame
            A data frame containing metadata information, with cell names as indexes. If cells are missing from the metadata, values will be filled with Nan.

        Return
        ------
        CistopicObject
            The input :class:`CistopicObject` with :attr:`cell_data` updated.
        """

        flag = False
        if len(set(self.cell_names) & set(
                cell_data.index)) < len(self.cell_names):
            check_cell_names = prepare_tag_cells(self.cell_names)
            if len(set(check_cell_names) & set(cell_data.index)) < len(
                    set(self.cell_names) & set(cell_data.index)):
                print(
                    "Warning: Some cells in this CistopicObject are not present in this cell_data. Values will be "
                    "filled with Nan\n"
                )
            else:
                flag = True
        if len(set(self.cell_data.columns) & set(cell_data.columns)) > 0:
            print(
                f"Columns {list(set(self.cell_data.columns.values) & set(cell_data.columns.values))} will be overwritten")
            self.cell_data = self.cell_data.loc[:, list(
                set(self.cell_data.columns).difference(set(cell_data.columns)))]
        if not flag:
            cell_data = cell_data.loc[list(
                set(self.cell_names) & set(cell_data.index)), ]
            new_cell_data = pd.concat(
                [self.cell_data, cell_data], axis=1, sort=False)
        elif flag:
            self.cell_data.index = prepare_tag_cells(self.cell_names)
            cell_data = cell_data.loc[list(
                set(self.cell_data.index.tolist()) & set(cell_data.index)), ]
            new_cell_data = pd.concat(
                [self.cell_data, cell_data], axis=1, sort=False)
            new_cell_data = new_cell_data.loc[prepare_tag_cells(
                self.cell_names), :]
            new_cell_data.index = self.cell_names

        self.cell_data = new_cell_data.loc[self.cell_names, :]

    def add_region_data(self,
                        region_data: pd.DataFrame):
        """
        Add region metadata to :class:`CistopicObject`. If the column already exist on the region metadata, it will be overwritten.

        Parameters
        ---------
        region_data: pd.DataFrame
            A data frame containing metadata information, with region names as indexes. If regions are missing from the metadata, values will be filled with Nan.

        Return
        ------
        CistopicObject
            The input :class:`CistopicObject` with :attr:`region_data` updated.
        """
        if len(set(self.region_names) & set(
                region_data.index)) < len(self.region_names):
            print(
                "Warning: Some regions in this CistopicObject are not present in this region_data. Values will be "
                "filled with Nan\n"
            )
        if len(set(self.region_data.columns.values) &
               set(region_data.columns.values)) > 0:
            print(
                f"Columns {list(set(self.region_data.columns.values) & set(region_data.columns.values))} will be overwritten")
            self.region_data = self.region_data.loc[:, list(set(
                self.region_data.columns.values).difference(set(region_data.columns.values)))]
        region_data = region_data.loc[list(
            set(self.region_names) & set(region_data.index)), ]
        new_region_data = pd.concat(
            [self.region_data, region_data], axis=1, sort=False)
        self.region_data = new_region_data.loc[self.region_names, :]

    def subset(self,
               cells: Optional[List[str]] = None,
               regions: Optional[List[str]] = None,
               copy: Optional[bool] = False):
        """
        Subset cells and/or regions from :class:`CistopicObject`. Existent :class:`CisTopicLDAModel` and projections will be deleted. This is to ensure that
        models contained in a :class:`CistopicObject` are derived from the cells it contains.

        Parameters
        ---------
        cells: list, optional
            A list containing the names of the cells to keep.
        regions: list, optional
            A list containing the names of the regions to keep.
        copy: bool, optional
            Whether changes should be done on the input :class:`CistopicObject` or a new object should be returned

        Return
        ------
        CistopicObject
            A :class:`CistopicObject` containing the selected cells and/or regions.
        """
        # Create logger
        level = logging.INFO
        format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level=level, format=format, handlers=handlers)
        log = logging.getLogger('cisTopic')

        # Select cells
        if cells is not None:
            try:
                keep_cells_index = get_position_index(cells, self.cell_names)
            except BaseException:
                try:
                    keep_cells_index = get_position_index(
                        cells, prepare_tag_cells(self.cell_names))
                except BaseException:
                    log.error(
                        'None of the given cells is contained in this cisTopic object!')
        else:
            keep_cells_index = list(range(len(self.cell_names)))
        # Select regions
        if regions is not None:
            keep_regions_index = get_position_index(regions, self.region_names)
        else:
            keep_regions_index = list(range(len(self.region_names)))
        # Subset
        fragment_matrix = self.fragment_matrix[:, keep_cells_index]
        fragment_matrix = fragment_matrix[keep_regions_index, :]
        binary_matrix = self.binary_matrix[:, keep_cells_index]
        binary_matrix = binary_matrix[keep_regions_index, :]
        region_names = subset_list(
            self.region_names,
            keep_regions_index)  # Subset selected regions
        keep_regions_index = non_zero_rows(binary_matrix)
        fragment_matrix = fragment_matrix[keep_regions_index, ]
        binary_matrix = binary_matrix[keep_regions_index, ]
        # Update
        cell_names = subset_list(self.cell_names, keep_cells_index)
        # Subset regions with all zeros
        region_names = subset_list(region_names, keep_regions_index)
        cell_data = self.cell_data.iloc[keep_cells_index, ]
        region_data = self.region_data.iloc[keep_regions_index, ]
        path_to_fragments = self.path_to_fragments
        project = self.project
        # Create new object
        if copy:
            subset_cistopic_obj = CistopicObject(
                fragment_matrix,
                binary_matrix,
                cell_names,
                region_names,
                cell_data,
                region_data,
                path_to_fragments,
                project)
            return subset_cistopic_obj
        else:
            self.fragment_matrix = fragment_matrix
            self.binary_matrix = binary_matrix
            self.cell_names = cell_names
            self.region_names = region_names
            self.cell_data = cell_data
            self.region_data = region_data
            self.selected_model = []
            self.projections = {}

    def merge(self,
              cistopic_obj_list: List['CistopicObject'],
              is_acc: Optional[int] = 1,
              project: Optional[str] = 'cisTopic_merge',
              copy: Optional[bool] = False):
        """
        Merge a list of :class:`CistopicObject` to the input :class:`CistopicObject`. Reference coordinates must be the same between the objects. Existent :class:`cisTopicCGSModel` and projections will be deleted. This is to ensure that models contained in a :class:`CistopicObject` are derived from the cells it contains.

        Parameters
        ---------
        cistopic_obj_list: list
            A list containing one or more :class:`CistopicObject` to merge.
        is_acc: int, optional
            Minimal number of fragments for a region to be considered accessible. Default: 1.
        project: str, optional
            Name of the cisTopic project.
        copy: bool, optional
            Whether changes should be done on the input :class:`CistopicObject` or a new object should be returned
        Return
        ------
        CistopicObject
            A combined :class:`CistopicObject`. Two new columns in :attr:`cell_data` indicate the :class:`CistopicObject` of origin (`cisTopic_id`) and the fragment file from which the cell comes from (`path_to_fragments`).
        """
        # Create logger
        level = logging.INFO
        format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level=level, format=format, handlers=handlers)
        log = logging.getLogger('cisTopic')

        cistopic_obj_list.insert(0, self)
        fragment_matrix_list = [x.fragment_matrix for x in cistopic_obj_list]
        region_names_list = [x.region_names for x in cistopic_obj_list]
        cell_names_list = [x.cell_names for x in cistopic_obj_list]
        cell_data_list = [x.cell_data.copy() for x in cistopic_obj_list]
        project_list = [x.project for x in cistopic_obj_list]
        path_to_fragments_list = [
            x.path_to_fragments for x in cistopic_obj_list]
        path_to_fragments_dict = {
            k: v
            for ptf in path_to_fragments_list
            for k, v in ptf.items()
        }

        if len(project_list) > len(set(project_list)):
            ori_project_list = project_list
            log.info(
                'You cannot merge objects with the same project id. Project id will be updated.')
            project_list = list(map(lambda x: x[1] + '_' + str(project_list[:x[0]].count(
                x[1]) + 1) if project_list.count(x[1]) > 1 else x[1], enumerate(project_list)))
            for i in range(len(project_list)):
                print(i)
                if len(list(set(cell_data_list[i]['sample_id']))) <= 1:
                    if (cell_data_list[i]['sample_id'][0] == ori_project_list[i]) & (
                            cell_data_list[i]['sample_id'][0] != project_list[i]):
                        log.info(
                            'Conflicting sample_id on project ' +
                            ori_project_list[i] +
                            ' will be updated to match with the new project name.')
                        cell_data_list[i]['sample_id'] = [project_list[i]] * len(cell_data_list[i]['sample_id'])
                if list(path_to_fragments_list[i].keys()) == 1:
                    if list(
                            path_to_fragments_list[i].keys()) == ori_project_list[i]:
                        log.info(
                            'Conflicting path_to_fragments key on project ' +
                            project_list[i] +
                            ' will be updated to match with the new project name.')
                        path_to_fragments_list[project_list[i]] = path_to_fragments_list.pop(
                            ori_project_list[i])

        cell_names_list = [
            prepare_tag_cells(
                cell_names_list[x]) for x in range(
                len(cell_names_list))]
        fragment_matrix = fragment_matrix_list[0]
        region_names = region_names_list[0]
        cell_names = [
            n + '-' + s
            for n, s in zip(
                cell_names_list[0],
                cell_data_list[0]['sample_id'].tolist())
        ]
        object_id = [project_list[0]] * len(cell_names)

        cell_data_list[0].index = cell_names

        for i in range(1, len(region_names_list)):
            region_names_to_add = region_names_list[i]
            fragment_matrix_to_add = fragment_matrix_list[i]
            cell_names_to_add = cell_names_list[i]
            object_id_to_add = [project_list[i]] * len(cell_names_to_add)
            cell_names_to_add = [
                n + '-' + s
                for n, s in zip(
                    cell_names_to_add,
                    cell_data_list[i]['sample_id'].tolist())
            ]
            cell_data_list[i].index = cell_names_to_add
            cell_names = cell_names + cell_names_to_add

            object_id = object_id + object_id_to_add
            common_regions = list(set(region_names) & set(region_names_to_add))
            diff_regions = list(set(region_names) ^ set(region_names_to_add))

            common_index_fm = get_position_index(common_regions, region_names)
            common_index_fm_to_add = get_position_index(
                common_regions, region_names_to_add)
            fragment_matrix_common = sparse.hstack(
                [fragment_matrix[common_index_fm, ], fragment_matrix_to_add[common_index_fm_to_add, ]])

            if len(diff_regions) > 0:
                diff_regions_1 = list(
                    np.setdiff1d(
                        region_names,
                        region_names_to_add))
                diff_index_fm_1 = get_position_index(
                    diff_regions_1, region_names)
                fragment_matrix_diff_1 = sparse.hstack([fragment_matrix[diff_index_fm_1, ], np.zeros(
                    (len(diff_regions_1), fragment_matrix_to_add.shape[1]))])

                diff_regions_2 = list(
                    np.setdiff1d(
                        region_names_to_add,
                        region_names))
                diff_index_fm_2 = get_position_index(
                    diff_regions_2, region_names_to_add)
                fragment_matrix_diff_2 = sparse.hstack([np.zeros(
                    (len(diff_regions_2), fragment_matrix.shape[1])), fragment_matrix_to_add[diff_index_fm_2, ]])

                fragment_matrix = sparse.vstack(
                    [fragment_matrix_common, fragment_matrix_diff_1, fragment_matrix_diff_2])
                region_names = common_regions + diff_regions_1 + diff_regions_2
            else:
                fragment_matrix = fragment_matrix_common
                region_names = common_regions

            fragment_matrix = sparse.csr_matrix(
                fragment_matrix, dtype=np.int32)
            log.info(f"cisTopic object {i} merged")

        binary_matrix = sp.binarize(fragment_matrix, threshold=is_acc - 1)
        cell_data = pd.concat(cell_data_list, axis=0, sort=False)
        cell_data.index = cell_names
        region_data = [x.region_data for x in cistopic_obj_list]
        region_data = pd.concat(region_data, axis=0, sort=False)
        if copy is True:
            cistopic_obj = CistopicObject(
                fragment_matrix,
                binary_matrix,
                cell_names,
                region_names,
                cell_data,
                region_data,
                path_to_fragments_dict,
                project)
            return cistopic_obj
        else:
            self.fragment_matrix = fragment_matrix
            self.binary_matrix = binary_matrix
            self.cell_names = cell_names
            self.region_names = region_names
            self.cell_data = cell_data
            self.region_data = region_data
            self.path_to_fragments = path_to_fragments_dict
            self.project = project
            self.selected_model = []
            self.projections = {}

    def add_LDA_model(self,
                      model: 'CistopicLDAModel'):
        """
        Add LDA model to a cisTopic object.

        Parameters
        ---
        model: CistopicLDAModel
            Selected cisTopic LDA model results (see `LDAModels.evaluate_models`)
        """
        # Check that region and cell names are in the same order
        model.region_topic = model.topic_region.loc[self.region_names, :]
        model.cell_topic = model.cell_topic.loc[:, self.cell_names]
        self.selected_model = model


def create_cistopic_object(fragment_matrix: Union[pd.DataFrame, sparse.csr_matrix],
                           cell_names: Optional[List[str]] = None,
                           region_names: Optional[List[str]] = None,
                           path_to_blacklist: Optional[str] = None,
                           min_frag: Optional[int] = 1,
                           min_cell: Optional[int] = 1,
                           is_acc: Optional[int] = 1,
                           path_to_fragments: Optional[Union[str, Dict[str, str]]] = {},
                           project: Optional[str] = 'cisTopic',
                           tag_cells: Optional[bool] = True):
    """
    Creates a CistopicObject from a count matrix.

    Parameters
    ---------
    fragment_matrix: pd.DataFrame or sparse.csr_matrix
        A data frame containing cell names as column names, regions as row names and fragment counts as values or :class:`sparse.csr_matrix` containing cells as columns and regions as rows.
    cell_names: list, optional
        A list containing cell names. Only used if the fragment matrix is :class:`sparse.csr_matrix`.
    region_names: list, optional
        A list containing region names. Only used if the fragment matrix is :class:`sparse.csr_matrix`.
    path_to_blacklist: str, optional
        Path to bed file containing blacklist regions (Amemiya et al., 2019).
    min_frag: int, optional
        Minimal number of fragments in a cell for the cell to be kept. Default: 1
    min_cell: int, optional
        Minimal number of cell in which a region is detected to be kept. Default: 1
    is_acc: int, optional
        Minimal number of fragments for a region to be considered accessible. Default: 1
    path_to_fragments: str, dict
        A dict or str containing the paths to the fragments files used to generate the :class:`CistopicObject`. Default: {}.
    project: str, optional
        Name of the cisTopic project. Default: 'cisTopic'
    tag_cells: bool, optional
        Whether to add the project name as suffix to the cell names. Default: True

    Return
    ------
    CistopicObject

    References
    ------
    Amemiya, H. M., Kundaje, A., & Boyle, A. P. (2019). The ENCODE blacklist: identification of problematic regions of the genome. Scientific reports, 9(1), 1-5.
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    if isinstance(fragment_matrix, pd.DataFrame):
        log.info('Converting fragment matrix to sparse matrix')
        region_names = list(fragment_matrix.index)
        cell_names = list(fragment_matrix.columns.values)
        fragment_matrix = sparse.csr_matrix(
            fragment_matrix.to_numpy(), dtype=np.int32)

    if tag_cells:
        cell_names = [
            cell_names[x] +
            '-' +
            project for x in range(
                len(cell_names))]

    if isinstance(path_to_blacklist, str):
        log.info('Removing blacklisted regions')
        regions = pr.PyRanges(region_names_to_coordinates(region_names))
        blacklist = read_fragments_from_file(path_to_blacklist)
        regions = regions.overlap(blacklist, invert=True)
        selected_regions = [
            str(chrom) +
            ":" +
            str(start) +
            '-' +
            str(end)
            for chrom, start, end in zip(
                list(regions.Chromosome),
                list(regions.Start),
                list(regions.End)
            )
        ]
        index = get_position_index(selected_regions, region_names)
        fragment_matrix = fragment_matrix[index, ]
        region_names = selected_regions

    log.info('Creating CistopicObject')
    binary_matrix = sp.binarize(fragment_matrix, threshold=is_acc - 1)
    selected_regions = non_zero_rows(binary_matrix)
    fragment_matrix = fragment_matrix[selected_regions, ]
    binary_matrix = binary_matrix[selected_regions, ]
    region_names = subset_list(region_names, selected_regions)

    cisTopic_nr_frag = np.array(fragment_matrix.sum(axis=0)).flatten()
    cisTopic_nr_acc = np.array(binary_matrix.sum(axis=0)).flatten()

    cell_data = pd.DataFrame([cisTopic_nr_frag,
                              np.log10(cisTopic_nr_frag),
                              cisTopic_nr_acc,
                              np.log10(cisTopic_nr_acc),
                              [project] * len(cell_names)],
                             columns=cell_names,
                             index=['cisTopic_nr_frag',
                                    'cisTopic_log_nr_frag',
                                    'cisTopic_nr_acc',
                                    'cisTopic_log_nr_acc',
                                    'sample_id']).transpose()

    if min_frag != 1:
        selected_cells = cell_data.cisTopic_nr_frag >= min_frag
        fragment_matrix = fragment_matrix[:, selected_cells]
        binary_matrix = binary_matrix[:, selected_cells]
        cell_data = cell_data.loc[selected_cells, ]
        cell_names = cell_data.index.to_list()

    region_data = region_names_to_coordinates(region_names)
    region_data['Width'] = abs(
        region_data.End -
        region_data.Start).astype(
        np.int32)
    region_data['cisTopic_nr_frag'] = np.array(
        fragment_matrix.sum(axis=1)).flatten()
    region_data['cisTopic_log_nr_frag'] = np.log10(
        region_data['cisTopic_nr_frag'])
    region_data['cisTopic_nr_acc'] = np.array(
        binary_matrix.sum(axis=1)).flatten()
    region_data['cisTopic_log_nr_acc'] = np.log10(
        region_data['cisTopic_nr_acc'])

    if min_cell != 1:
        selected_regions = region_data.cisTopic_nr_acc >= min_cell
        fragment_matrix = fragment_matrix[selected_regions, :]
        binary_matrix = binary_matrix[selected_regions, :]
        region_data = region_data[selected_regions, :]
        region_names = region_data.index.to_list()

    cistopic_obj = CistopicObject(
        fragment_matrix,
        binary_matrix,
        cell_names,
        region_names,
        cell_data,
        region_data,
        path_to_fragments,
        project)
    log.info('Done!')
    return cistopic_obj


def create_cistopic_object_from_matrix_file(fragment_matrix_file: str,
                                            path_to_blacklist: Optional[str] = None,
                                            compression: Optional[str] = None,
                                            min_frag: Optional[int] = 1,
                                            min_cell: Optional[int] = 1,
                                            is_acc: Optional[int] = 1,
                                            path_to_fragments: Optional[Dict[str, str]] = {},
                                            sample_id: Optional[pd.DataFrame] = None,
                                            project: Optional[str] = 'cisTopic'):
    """
    Creates a CistopicObject from a count matrix file (tsv).

    Parameters
    ---------
    fragment_matrix: str
        Path to a tsv file containing cell names as column names, regions as row names and fragment counts as values.
    path_to_blacklist: str, optional
        Path to bed file containing blacklist regions (Amemiya et al., 2019). Default: None
    compression: str, None
        Whether the file is compressed (e.g. bzip). Default: None
    min_frag: int, optional
        Minimal number of fragments in a cell for the cell to be kept. Default: 1
    min_cell: int, optional
        Minimal number of cell in which a region is detected to be kept. Default: 1
    is_acc: int, optional
        Minimal number of fragments for a region to be considered accessible. Default: 1
    path_to_fragments: dict, optional
        A list containing the paths to the fragments files used to generate the :class:`CistopicObject`. Default: None.
    sample_id: pd.DataFrame, optional
        A data frame indicating from which sample each barcode is derived. Required if path_to_fragments is provided. Levels must agree with keys in path_to_fragments. Default: None.
    project: str, optional
        Name of the cisTopic project. Default: 'cisTopic'

    Return
    ------
    CistopicObject

    References
    ------
    Amemiya, H. M., Kundaje, A., & Boyle, A. P. (2019). The ENCODE blacklist: identification of problematic regions of the genome. Scientific reports, 9(1), 1-5.
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    log.info('Reading data')
    if compression is not None:
        fragment_matrix = pd.read_csv(fragment_matrix_file,
                                      sep='\t',
                                      header=0,
                                      compression=compression)
    else:
        fragment_matrix = pd.read_csv(fragment_matrix_file,
                                      sep='\t',
                                      header=0)

    cistopic_obj = create_cistopic_object(fragment_matrix=fragment_matrix,
                                          path_to_blacklist=path_to_blacklist,
                                          min_frag=min_frag,
                                          min_cell=min_cell,
                                          is_acc=is_acc,
                                          path_to_fragments=path_to_fragments,
                                          project=project)

    if sample_id is not None:
        if (isinstance(path_to_fragments, dict)):
            cistopic_obj.add_cell_data(sample_id)
        else:
            log.error(
                'Provide path_to_fragments with keys matching levels in sample_id!')
    return cistopic_obj


def create_cistopic_object_from_fragments(path_to_fragments: str,
                                          path_to_regions: str,
                                          path_to_blacklist: Optional[str] = None,
                                          metrics: Optional[Union[str, pd.DataFrame]] = None,
                                          valid_bc: Optional[List[str]] = None,
                                          n_cpu: Optional[int] = 1,
                                          min_frag: Optional[int] = 1,
                                          min_cell: Optional[int] = 1,
                                          is_acc: Optional[int] = 1,
                                          check_for_duplicates: Optional[bool] = True,
                                          project: Optional[str] = 'cisTopic',
                                          partition: Optional[int] = 5,
                                          fragments_df: Optional[Union[pd.DataFrame, pr.PyRanges]] = None):
    """
    Creates a CistopicObject from a fragments file and defined genomic intervals (compatible with CellRangerATAC output)

    Parameters
    ---------
    path_to_fragments: str
        The path to the fragments file containing chromosome, start, end and assigned barcode for each read (e.g. from CellRanger ATAC (/outs/fragments.tsv.gz)).
    path_to_regions: str
        Path to the bed file with the defined regions.
    path_to_blacklist: str, optional
        Path to bed file containing blacklist regions (Amemiya et al., 2019). Default: None
    metrics: str, optional
        Data frame of CellRanger ot similar, with barcodes and metrics (e.g. from CellRanger ATAC /outs/singlecell.csv). If it is an output from CellRanger, only cells for which is__cell_barcode is 1 will be considered, otherwise only barcodes included in the metrics will be taken. Default: None
    valid_bc: list, optional
        A list with valid cell barcodes can be provided, only used if path_to_metrics is not provided. Default: None
    n_cpu: int, optional
        Number of cores to use. Default: 1.
    min_frag: int, optional
        Minimal number of fragments in a cell for the cell to be kept. Default: 1
    min_cell: int, optional
        Minimal number of cell in which a region is detected to be kept. Default: 1
    is_acc: int, optional
        Minimal number of fragments for a region to be considered accessible. Default: 1
    check_for_duplicates: bool, optional
        If no duplicate counts are provided per row in the fragments file, whether to collapse duplicates. Default: True.
    project: str, optional
        Name of the cisTopic project. It will also be used as name for sample_id in the cell_data :class:`CistopicObject.cell_data`. Default: 'cisTopic'
    partition: int, optional
        When using Pandas > 0.21, counting may fail (https://github.com/pandas-dev/pandas/issues/26314). In that case, the fragments data frame is divided in this number of partitions, and after counting data is merged.
    fragments_df: pd.DataFrame or pr.PyRanges, optional
        A PyRanges or DataFrame containing chromosome, start, end and assigned barcode for each read, corresponding to the data in path_to_fragments.
    Return
    ------
    CistopicObject

    References
    ------
    Amemiya, H. M., Kundaje, A., & Boyle, A. P. (2019). The ENCODE blacklist: identification of problematic regions of the genome. Scientific reports, 9(1), 1-5.
    """
    # Create logger
    level = logging.INFO
    format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level=level, format=format, handlers=handlers)
    log = logging.getLogger('cisTopic')

    # Read data
    log.info('Reading data for ' + project)
    if isinstance(fragments_df, pd.DataFrame):
        fragments = pr.PyRanges(fragments_df)
        if path_to_fragments is not None:
            log.info('Using fragments of provided pandas data frame')
    else:
        fragments = read_fragments_from_file(path_to_fragments)

    if 'Score' not in fragments.df:
        fragments_df = fragments.df
        if check_for_duplicates:
            log.info("Collapsing duplicates")
            fragments_df = pd.concat([
                collapse_duplicates(fragments_df[fragments_df.Chromosome == x])
                for x in fragments_df.Chromosome.cat.categories.values]
            )
        else:
            fragments_df['Score'] = 1
        fragments = pr.PyRanges(fragments_df)

    regions = read_fragments_from_file(path_to_regions)
    regions = regions[['Chromosome', 'Start', 'End']]
    regions.regionID = [
        str(chrom) +
        ":" +
        str(start) +
        '-' +
        str(end)
        for chrom, start, end in zip(
            list(regions.Chromosome),
            list(regions.Start),
            list(regions.End))
    ]

    # If CellRanger metrics, select valid barcodes
    if metrics is not None:
        log.info('metrics provided!')
        if isinstance(metrics, str):
            metrics = pd.read_csv(metrics)
        if 'is__cell_barcode' in metrics.columns:
            metrics = metrics[metrics.is__cell_barcode == 1]
            metrics.index = metrics.barcode
            metrics = metrics.iloc[:, 2:]
        fragments = fragments[fragments.Name.isin(set(metrics.index))]
    if isinstance(valid_bc, list):
        log.info('valid_bc provided, selecting barcodes!')
        fragments = fragments[fragments.Name.isin(set(valid_bc))]
    if metrics is None:
        log.info('Counting total number of fragments (Total_nr_frag)')
        fragments_per_barcode = cl.Counter(fragments.Name.to_list())
        fragments_per_barcode = [fragments_per_barcode[x]
                                 for x in set(fragments.Name.to_list())]
        FPB_DF = pd.DataFrame(fragments_per_barcode)
        FPB_DF.index = set(fragments.Name.to_list())
        FPB_DF.columns = ['Total_nr_frag']
    # Count fragments in regions
    log.info('Counting fragments in regions')
    fragments_in_regions = regions.join(fragments, nb_cpu=n_cpu)
    # Convert to pandas
    counts_df = pd.concat([fragments_in_regions.regionID.astype("category"),
                           fragments_in_regions.Name.astype("category"),
                           fragments_in_regions.Score.astype(np.int32)],
                          axis=1,
                          sort=False)

    log.info('Creating fragment matrix')
    try:
        fragment_matrix = counts_df.groupby(["Name", "regionID"], sort=False, observed=True).size().unstack(
            level="Name", fill_value=0).astype(np.int32)
        fragment_matrix.columns.names = [None]
        # Create CistopicObject
        cistopic_obj = create_cistopic_object(
            fragment_matrix=fragment_matrix,
            path_to_blacklist=path_to_blacklist,
            min_frag=min_frag,
            min_cell=min_cell,
            is_acc=is_acc,
            path_to_fragments={
            project: path_to_fragments},
            project=project)
    except (ValueError, MemoryError):
        log.info(
            'Data is too big, making partitions. This is a reported error in Pandas versions > 0.21 (https://github.com/pandas-dev/pandas/issues/26314)')
        barcode_list = np.array_split(
            list(set(counts_df.Name.to_list())), partition)
        cistopic_obj_list = [counts_df[counts_df.Name.isin(
            set(barcode_list[x]))] for x in range(0, partition)]
        del counts_df
        cistopic_obj_list = [create_cistopic_object_chunk(cistopic_obj_list[i],
                                path_to_blacklist,
                                min_frag,
                                min_cell,
                                is_acc,
                                path_to_fragments={
                                project: path_to_fragments},
                                project=str(i),
                                project_all=project) for i in range(partition)]
        cistopic_obj = merge(cistopic_obj_list, project=project)
        cistopic_obj.project = project
        cistopic_obj.path_to_fragments = {
            project: path_to_fragments}
        
    if metrics is not None:
        metrics['barcode'] = metrics.index.tolist()
        cistopic_obj.add_cell_data(metrics)
    else:
        FPB_DF['barcode'] = FPB_DF.index.tolist()
        cistopic_obj.add_cell_data(FPB_DF)
    return cistopic_obj

def create_cistopic_object_chunk(df,
                                path_to_blacklist,
                                min_frag,
                                min_cell,
                                is_acc,
                                path_to_fragments,
                                project,
                                project_all):
    df = df.groupby(["Name", "regionID"], sort=False, observed=True).size().unstack(
            level="Name", fill_value=0).astype(np.int32).rename_axis(None)
    cistopic_obj = create_cistopic_object(
            fragment_matrix=df,
            path_to_blacklist=path_to_blacklist,
            min_frag=min_frag,
            min_cell=min_cell,
            is_acc=is_acc,
            path_to_fragments={
            project: path_to_fragments},
            project=project,
            tag_cells = False)
    cistopic_obj.cell_data['sample_id'] = [project_all] * len(cistopic_obj.cell_names)
    return cistopic_obj


def merge(cistopic_obj_list: List['CistopicObject'],
          is_acc: Optional[int] = 1,
          project: Optional[str] = 'cisTopic_merge'):
    """
    Merge a list of :class:`CistopicObject` to the input :class:`CistopicObject`. Reference coordinates must be the same between the objects. Existent :class:`cisTopicCGSModel` and projections will be deleted. This is to ensure that models contained in a :class:`CistopicObject` are derived from the cells it contains.

    Parameters
    ---------
    cistopic_obj_list: list
        A list containing one or more :class:`CistopicObject` to merge.
    is_acc: int, optional
        Minimal number of fragments for a region to be considered accessible. Default: 1.
    project: str, optional
        Name of the cisTopic project.

    Return
    ------
    CistopicObject
        A combined :class:`CistopicObject`. Two new columns in :attr:`cell_data` indicate the :class:`CistopicObject` of origin (`cisTopic_id`) and the fragment file from which the cell comes from (`path_to_fragments`).
    """

    merged_cistopic_obj = cistopic_obj_list[0].merge(
        cistopic_obj_list[1:], is_acc=is_acc, project=project, copy=True)
    return merged_cistopic_obj
