#!/usr/bin/env python3
# VSN loom functions at https://raw.githubusercontent.com/vib-singlecell-nf/vsn-pipelines/master/src/scenic/bin/export_to_loom.py
import os
import numpy as np
import pandas as pd
import loompy as lp
from sklearn.manifold.t_sne import TSNE
from pyscenic.aucell import aucell
from pyscenic.genesig import Regulon, GeneSignature
from typing import List, Mapping, Sequence, Optional, Dict
from operator import attrgetter
from multiprocessing import cpu_count
from pyscenic.binarization import binarize
from itertools import chain, repeat, islice
import networkx as nx
import zlib
import base64
import json
import re
import sys

from collections import OrderedDict


class Embedding:

    def __init__(self, embedding: pd.DataFrame, embedding_name: str, is_default: bool = False):
        self.embedding = embedding
        self.embedding_name = embedding_name
        self._is_default = is_default

    def get_embedding_name(self):
        return self.embedding_name

    def get_embedding(self):
        return self.embedding

    def is_default(self):
        return self._is_default


class SCopeLoom:

    def __init__(
        self,
        filename: str = None,
        out_fname: str = None,
        ex_mtx: pd.DataFrame = None,
        regulons: List[Regulon] = None,
        cell_annotations: Optional[Mapping[str, str]] = None,
        tree_structure: Sequence[str] = None,
        title: Optional[str] = None,
        nomenclature: str = "Unknown",
        num_workers: int = cpu_count(),
        auc_mtx=None,
        auc_regulon_weights_key='gene2weight',
        auc_thresholds=None,
        compress: bool = False,
        save_additional_regulon_meta_data: bool = False,  # Should be set to true only for multi-runs SCENIC
        set_generic_loom: bool = False,
        tag: str = None,  # Used when merging track and motif-based SCENIC run
        col_attrs: Mapping[str, np.ndarray] = None,
        row_attrs: Mapping[str, np.ndarray] = None,
        global_attrs: Dict = None,
        embeddings: Mapping[str, pd.DataFrame] = None
    ):
        self.filename = filename
        self.out_fname = out_fname
        self.ex_mtx = ex_mtx
        self.regulons = regulons
        self.cell_annotations = cell_annotations
        self.tree_structure = tree_structure if tree_structure else ()
        self.title = title
        self.nomenclature = nomenclature
        self.num_workers = num_workers
        self.auc_mtx = auc_mtx
        self.auc_regulon_weights_key = auc_regulon_weights_key
        self.auc_thresholds = auc_thresholds
        self.compress = compress
        self.save_additional_regulon_meta_data = save_additional_regulon_meta_data
        self.tag = tag
        self.regulon_filter = None
        # Loom representation
        self.col_attrs = col_attrs if col_attrs else {}
        self.row_attrs = row_attrs if row_attrs else {}
        self.global_attrs = global_attrs if global_attrs else {}
        # Internal representation
        self.embeddings = embeddings if embeddings else {}
        # Utils
        self.id2name = OrderedDict()
        # Set Base Loom
        if set_generic_loom:
            self.set_generic_loom()

    def set_generic_loom(self):
        if(self.regulons[0].name.find('_') == -1):
            print(
                "Regulon name does not seem to be compatible with SCOPE. It should include a space to allow selection of the TF.",
                "\nPlease run: \n regulons = [r.rename(r.name.replace('(+)','_('+str(len(r))+'g)')) for r in regulons]",
                "\nor:\n regulons = [r.rename(r.name.replace('(','_(')) for r in regulons]"
            )
        self.set_cell_annotations()
        if self.auc_mtx is None:
            self.auc_mtx = self.calculate_regulon_enrichment()
        if self.auc_thresholds is None:
            self.auc_thresholds = self.binarize_regulon_enrichment()
        if len(self.embeddings.keys()) == 0:
            self.create_loom_default_embedding()
        self.regulon_gene_assignment = self.create_loom_regulon_gene_assignment()
        self.ngenes = self.calculate_nb_genes_per_cell()
        self.clusterings = self.create_loom_clusterings()
        self.regulon_thresholds = self.create_loom_md_regulon_thresholds()
        self.set_generic_col_attrs()
        self.set_generic_row_attrs()
        self.set_generic_global_attrs()
        self.set_tree()

    #######
    # I/O #
    #######

    @staticmethod
    def read_loom(filename: str, tag: str = None):
        with lp.connect(filename, mode='r', validate=False) as loom:

            # Load the content into memory
            # Set the main matrix
            ex_mtx = pd.DataFrame(loom[:, :], index=loom.ra.Gene, columns=loom.ca.CellID).T
            # Set the column, row and global attribute using the underlying Dict of the AttributeManager
            col_attrs = {k: v for k, v in loom.ca.items()}
            row_attrs = {k: v for k, v in loom.ra.items()}
            global_attrs = {k: v for k, v in loom.attrs.items()}
            # Decompress and decode the MetaData global attribute
            try:
                global_attrs["MetaData"] = SCopeLoom.decompress_decode(value=global_attrs["MetaData"])
            except Exception:
                # MetaData is uncompressed
                global_attrs["MetaData"] = json.loads(global_attrs["MetaData"])

        scope_loom = SCopeLoom(
            filename=filename,
            ex_mtx=ex_mtx,
            col_attrs=col_attrs,
            row_attrs=row_attrs,
            global_attrs=global_attrs,
            tag=tag
        )
        if 'embeddings' in scope_loom.get_meta_data():
            scope_loom.convert_loom_embeddings_repr_to_internal_repr()

        # If multi-runs mode
        is_multi_runs_mode = scope_loom.has_scenic_multi_runs_data()
        if is_multi_runs_mode:
            scope_loom.set_scenic_min_genes_regulon(min_genes_regulon=global_attrs["MetaData"]["regulonSettings"]["min_genes_regulon"])
            scope_loom.set_scenic_min_regulon_gene_occurrence(min_regulon_gene_occurrence=global_attrs["MetaData"]["regulonSettings"]["min_regulon_gene_occurrence"])

        return scope_loom

    #############
    # Meta Data #
    #############

    def add_meta_data(self, _dict):
        md = self.global_attrs["MetaData"]
        md.update(_dict)
        self.global_attrs["MetaData"] = md

    def get_meta_data(self):
        return self.global_attrs["MetaData"]

    def set_tree(self):
        assert len(self.tree_structure) <= 3, ""
        self.global_attrs.update(("SCopeTreeL{}".format(idx + 1), category) for idx, category in enumerate(list(islice(chain(self.tree_structure, repeat("")), 3))))

    #############
    # Features  #
    #############

    def get_genes(self):
        return self.row_attrs['Gene']

    ################
    # Observations #
    ################

    def get_cell_ids(self):
        return self.col_attrs['CellID']

    ###############
    # Annotations #
    ###############

    def set_cell_annotations(self):
        if self.cell_annotations is None:
            self.cell_annotations = dict(zip(self.ex_mtx.index.astype(str), ['-'] * self.ex_mtx.shape[0]))

    ###########
    # Metrics #
    ###########

    def calculate_nb_genes_per_cell(self):
        # Calculate the number of genes per cell.
        binary_mtx = self.ex_mtx.copy()
        binary_mtx[binary_mtx != 0] = 1.0
        return binary_mtx.sum(axis=1).astype(int)

    def add_metrics(self, metrics: List[str]):
        md_metrics = []
        for metric in metrics:
            md_metrics.append({"name": metric})
        self.global_attrs["MetaData"].update({'metrics': md_metrics})

    ##############
    # Embeddings #
    ##############

    @staticmethod
    def get_embedding_id(embedding: Embedding, _list):
        """ Returns the appropriate index as a string given the _list

        Parameters:
            None

        Returns:
            str: Returns -1 if the given embedding is the default, 0 if the given _list is empty and length of the given _list minus 1

        """
        if embedding.is_default():
            return '-1'
        elif len(_list) == '0':
            return '0'
        else:
            return str(len(_list) - 1)

    def convert_loom_embeddings_repr_to_internal_repr(self):
        for embedding in self.get_meta_data()['embeddings']:
            self.add_embedding(
                embedding=self.get_embedding_by_id(embedding_id=embedding['id']),
                embedding_name=embedding['name'],
                is_default=True if str(embedding['id']) == '-1' else False
            )

    def get_embedding_by_id(self, embedding_id):
        if str(embedding_id) == '-1':
            return self.col_attrs['Embedding']
        x = self.col_attrs['Embeddings_X'][str(embedding_id)]
        y = self.col_attrs['Embeddings_Y'][str(embedding_id)]
        return np.column_stack((x, y))

    def has_embedding(self, embedding_name):
        return embedding_name in self.embeddings.keys()

    def add_embedding(self, embedding: np.ndarray, embedding_name, is_default: bool = False):
        df_embedding = pd.DataFrame(embedding, columns=['_X', '_Y'], index=self.ex_mtx.index)
        _embedding = Embedding(embedding=df_embedding, embedding_name=embedding_name, is_default=is_default)
        if is_default:
            self.default_embedding = _embedding
        self.embeddings[embedding_name] = _embedding

    def create_loom_default_embedding(self):
        # Create an embedding based on tSNE.
        # Name of columns should be "_X" and "_Y".
        self.add_embedding(embedding=TSNE().fit_transform(self.auc_mtx), embedding_name="tSNE (default)", is_default=True)

    def create_loom_md_embeddings_repr(self):
        """ Returns a Dictionary (Dict) for the global meta data embeddings with preformated data to be stored in Loom file format and compatible with SCope.

        Parameters:
            None

        Returns:
            dict: A Dictionary (Dict) for the global meta data embeddings with preformated data to be stored in Loom file format and compatible with SCope.

        """
        md_embeddings = []
        for _, embedding in self.embeddings.items():
            if embedding.is_default():
                md_embeddings = md_embeddings + [{'id': '-1', 'name': embedding.get_embedding_name()}]
            else:
                md_embeddings = md_embeddings + [
                    {
                        'id': SCopeLoom.get_embedding_id(embedding=embedding, _list=md_embeddings),
                        'name': embedding.get_embedding_name()
                    }
                ]
        return {"embeddings": md_embeddings}

    def create_loom_ca_embeddings_repr(self):
        """ Returns a Dictionary (Dict) for the embeddings with preformated data to be stored in Loom file format and compatible with SCope.

        Parameters:
            None

        Returns:
            dict: A Dictionary (Dict) for the embeddings with preformated data to be stored in Loom file format and compatible with SCope.

        """
        default_embedding = None
        embeddings_X = pd.DataFrame(index=self.ex_mtx.index)
        embeddings_Y = pd.DataFrame(index=self.ex_mtx.index)

        for _, embedding in self.embeddings.items():
            if(embedding.get_embedding().shape[1] != 2):
                raise Exception('The embedding should have two columns.')
            # Set the default embedding
            if embedding.is_default():
                default_embedding = embedding.get_embedding()
            # Update the Embeddings_[X|Y]
            embedding_id = str(SCopeLoom.get_embedding_id(embedding, embeddings_X.columns))
            embedding = embedding.get_embedding().copy()
            embedding.columns = ['_X', '_Y']
            embeddings_X = pd.merge(
                embeddings_X,
                embedding['_X'].to_frame().rename(
                    columns={'_X': embedding_id}
                ).astype('float32'), left_index=True, right_index=True)
            embeddings_Y = pd.merge(
                embeddings_Y,
                embedding['_Y'].to_frame().rename(
                    columns={'_Y': embedding_id}
                ).astype('float32'), left_index=True, right_index=True)
        return {
            "Embedding": SCopeLoom.df_to_named_matrix(df=default_embedding),
            "Embeddings_X": SCopeLoom.df_to_named_matrix(df=embeddings_X),
            "Embeddings_Y": SCopeLoom.df_to_named_matrix(df=embeddings_Y)
        }

    ###############
    # Clusterings #
    ###############

    def create_loom_clusterings(self):
        # Encode cell type clusters.
        # The name of the column should match the identifier of the clustering.
        return pd.DataFrame(
            data=self.ex_mtx.index.values,
            index=self.ex_mtx.index,
            columns=['0']
        ).replace(self.cell_annotations).replace(self.name2idx())

    ##########
    # SCENIC #
    ##########

    def set_scenic_min_genes_regulon(self, min_genes_regulon):
        self.scenic_min_genes_regulon = min_genes_regulon

    def set_scenic_min_regulon_gene_occurrence(self, min_regulon_gene_occurrence):
        self.scenic_min_regulon_gene_occurrence = min_regulon_gene_occurrence

    def set_regulon_filter(self, regulons):
        self.regulon_filter = regulons

    def has_scenic_multi_runs_data(self):
        return (
            'RegulonGeneOccurrences' in self.row_attrs.keys() and 'RegulonGeneWeights' in self.row_attrs.keys()
        ) or (
            'MotifRegulonGeneOccurrences' in self.row_attrs.keys() and 'MotifRegulonGeneWeights' in self.row_attrs.keys()
        ) or (
            'TrackRegulonGeneOccurrences' in self.row_attrs.keys() and 'TrackRegulonGeneWeights' in self.row_attrs.keys()
        )

    def scopify_md_regulon_data(self):
        # Fix regulon objects to display properly in SCope:
        # Rename regulons in the thresholds object, motif
        md_regulon_thresholds = self.global_attrs['MetaData']['regulonThresholds']
        for _, x in enumerate(md_regulon_thresholds):
            tmp = re.sub(r"(_?)\(", '_(', x.get('regulon'))  # + '-motif'
            x.update({'regulon': tmp})
        return {
            'regulonThresholds': md_regulon_thresholds
        }

    def scopify_loom_ca_regulon_data(self):
        regulons_auc_col_attrs = list(filter(lambda col_attrs_key: 'RegulonsAUC' in col_attrs_key, self.col_attrs.keys()))

        def fix(col_attrs_key):
            regulons = pd.DataFrame(self.col_attrs[col_attrs_key], index=self.col_attrs['CellID'])
            # Add underscore for SCope compatibility:
            regulons.columns = regulons.columns.str.replace('_?\\(', '_(')
            return {
                col_attrs_key: SCopeLoom.df_to_named_matrix(regulons)
            }
        # Update the keys
        regulons_auc_col_attrs_update = map(fix, regulons_auc_col_attrs)
        # Convert list of dict to dict
        return {next(iter(x)): x.get(next(iter(x))) for x in regulons_auc_col_attrs_update}

    def scopify_loom_ra_regulon_data(self):
        regulons_row_attrs = list(filter(lambda row_attrs_key: 'Regulon' in row_attrs_key, self.row_attrs.keys()))

        def fix(row_attrs_key):
            regulons = pd.DataFrame(self.row_attrs[row_attrs_key], index=self.row_attrs['Gene'])
            # Add underscore for SCope compatibility:
            regulons.columns = regulons.columns.str.replace('_?\\(', '_(')
            return {
                row_attrs_key: SCopeLoom.df_to_named_matrix(regulons)
            }
        # Update the keys
        regulons_row_attrs_update = map(fix, regulons_row_attrs)
        # Convert list of dict to dict
        return {next(iter(x)): x.get(next(iter(x))) for x in regulons_row_attrs_update}

    def calculate_regulon_enrichment(self):
        # Calculate regulon enrichment per cell using AUCell.
        # Create regulons with weight based on given key
        print("Using {} to weight the genes when running AUCell.".format(self.auc_regulon_weights_key))
        regulon_signatures = list(map(lambda x: GeneSignature(name=x.name, gene2weight=self.get_regulon_gene_data(x, self.auc_regulon_weights_key)), self.regulons))
        auc_mtx = aucell(self.ex_mtx, regulon_signatures, num_workers=self.num_workers)  # (n_cells x n_regulons)
        auc_mtx = auc_mtx.loc[self.ex_mtx.index]
        return auc_mtx

    def binarize_regulon_enrichment(self):
        _, auc_thresholds = binarize(self.auc_mtx)
        return auc_thresholds

    @staticmethod
    def format_tag(tag):
        # return f"{tag[0]}{tag[-1]}"
        return tag

    def get_regulon_gene_weights(self):
        if self.regulon_filter is not None:
            return self.row_attrs['RegulonGeneWeights'][self.regulon_filter]
        return self.row_attrs["RegulonGeneWeights"]

    def get_regulon_gene_occurrences(self):
        if self.regulon_filter is not None:
            return self.row_attrs['RegulonGeneOccurrences'][self.regulon_filter]
        return self.row_attrs["RegulonGeneOccurrences"]

    def get_regulons(self):
        if self.regulon_filter is not None:
            return self.row_attrs['Regulons'][self.regulon_filter]
        return self.row_attrs["Regulons"]

    def merge_regulon_data(self, scope_loom):
        # Check if SCENIC has been run in multi-runs mode
        is_multi_runs_mode = self.has_scenic_multi_runs_data() and scope_loom.has_scenic_multi_runs_data()

        # RegulonsAUC
        # Relabel columns with suffix indicating the regulon source
        auc_mtx = pd.DataFrame(data=self.col_attrs['RegulonsAUC'], index=self.col_attrs['CellID'])
        auc_mtx.columns = auc_mtx.columns + '-' + SCopeLoom.format_tag(self.tag)

        scope_loom_auc_mtx = pd.DataFrame(data=scope_loom.col_attrs['RegulonsAUC'], index=scope_loom.col_attrs['CellID'])
        scope_loom_auc_mtx.columns = scope_loom_auc_mtx.columns + '-' + SCopeLoom.format_tag(scope_loom.tag)

        # Regulons (regulon assignment matrices)
        regulons = pd.DataFrame(self.get_regulons(), index=self.row_attrs['Gene'])
        regulons.columns = regulons.columns + '-' + SCopeLoom.format_tag(self.tag)

        scope_loom_regulons = pd.DataFrame(scope_loom.get_regulons(), index=scope_loom.row_attrs['Gene'])
        scope_loom_regulons.columns = scope_loom_regulons.columns + '-' + SCopeLoom.format_tag(scope_loom.tag)

        # If multi-runs SCENIC
        # Rename Regulons Gene Occurrences
        if is_multi_runs_mode:
            regulon_gene_occurrences = pd.DataFrame(self.get_regulon_gene_occurrences(), index=self.row_attrs['Gene'])
            regulon_gene_occurrences.columns = regulon_gene_occurrences.columns + '-' + SCopeLoom.format_tag(self.tag)

            scope_loom_regulon_gene_occurrences = pd.DataFrame(scope_loom.get_regulon_gene_occurrences(), index=scope_loom.row_attrs['Gene'])
            scope_loom_regulon_gene_occurrences.columns = scope_loom_regulon_gene_occurrences.columns + '-' + SCopeLoom.format_tag(scope_loom.tag)

        # If multi-runs SCENIC
        # Rename Regulons Gene Weights
        if is_multi_runs_mode:
            regulon_gene_weights = pd.DataFrame(self.get_regulon_gene_weights(), index=self.row_attrs['Gene'])
            regulon_gene_weights.columns = regulon_gene_weights.columns + '-' + SCopeLoom.format_tag(self.tag)

            scope_loom_regulon_gene_weights = pd.DataFrame(scope_loom.get_regulon_gene_weights(), index=scope_loom.row_attrs['Gene'])
            scope_loom_regulon_gene_weights.columns = scope_loom_regulon_gene_weights.columns + '-' + SCopeLoom.format_tag(scope_loom.tag)

        # Combine meta data regulons
        # Rename regulons in the thresholds object, motif
        rt = self.global_attrs["MetaData"]["regulonThresholds"]
        for _, x in enumerate(rt):
            tmp = x.get('regulon') + '-' + SCopeLoom.format_tag(self.tag)
            x.update({'regulon': tmp})

        # Rename regulons in the thresholds object, track
        scope_loom_rt = scope_loom.global_attrs["MetaData"]["regulonThresholds"]
        for _, x in enumerate(scope_loom_rt):
            tmp = x.get('regulon') + '-' + SCopeLoom.format_tag(scope_loom.tag)
            x.update({'regulon': tmp})
            # blank out the "motifData" field for track-based regulons:
            x.update({'mofitData': 'NA.png'})

        # merge regulon threshold dictionaries:
        rt_merged = rt + scope_loom_rt

        # Remove because we will save them separately
        del self.row_attrs["Regulons"]
        del self.col_attrs["RegulonsAUC"]

        if is_multi_runs_mode:
            del self.row_attrs["RegulonGeneOccurrences"]
            del self.row_attrs["RegulonGeneWeights"]

        # Update the attributes
        self.row_attrs.update({f'{self.tag.capitalize()}Regulons': SCopeLoom.df_to_named_matrix(regulons)})
        self.col_attrs.update({f'{self.tag.capitalize()}RegulonsAUC': SCopeLoom.df_to_named_matrix(auc_mtx)})

        if is_multi_runs_mode:
            self.row_attrs.update({f'{self.tag.capitalize()}RegulonGeneOccurrences': SCopeLoom.df_to_named_matrix(regulon_gene_occurrences)})
            self.row_attrs.update({f'{self.tag.capitalize()}RegulonGeneWeights': SCopeLoom.df_to_named_matrix(regulon_gene_weights)})

        self.row_attrs.update({f'{scope_loom.tag.capitalize()}Regulons': SCopeLoom.df_to_named_matrix(scope_loom_regulons)})
        self.col_attrs.update({f'{scope_loom.tag.capitalize()}RegulonsAUC': SCopeLoom.df_to_named_matrix(scope_loom_auc_mtx)})

        if is_multi_runs_mode:
            self.row_attrs.update({f'{scope_loom.tag.capitalize()}RegulonGeneOccurrences': SCopeLoom.df_to_named_matrix(scope_loom_regulon_gene_occurrences)})
            self.row_attrs.update({f'{scope_loom.tag.capitalize()}RegulonGeneWeights': SCopeLoom.df_to_named_matrix(scope_loom_regulon_gene_weights)})

        self.global_attrs["MetaData"].update({'regulonThresholds': rt_merged})

    def get_regulon_gene_data(self, regulon, key):
        if key == 'gene2occurrence':
            return regulon.gene2occurrence
        if key == 'gene2weight':
            return regulon.gene2weight
        raise Exception('Cannot retrieve {} from given regulon. Not implemented.'.format(key))

    def get_regulon_meta_data(self, name, threshold):

        def fetch_logo(context):
            for elem in context:
                if elem.endswith('.png'):
                    return elem
            return ""

        name2logo = {reg.name: fetch_logo(reg.context) for reg in self.regulons}

        regulon = list(filter(lambda x: x.name == name, self.regulons))[0]
        regulon_meta_data = {
            "regulon": name,
            "defaultThresholdValue": (threshold if isinstance(threshold, float) else threshold[0]),
            "defaultThresholdName": "guassian_mixture_split",
            "allThresholds": {
                "guassian_mixture_split": (threshold if isinstance(threshold, float) else threshold[0])
            },
            "motifData": name2logo.get(name, "")
        }
        if self.save_additional_regulon_meta_data:
            regulon_meta_data.update({
                "tf": regulon.transcription_factor,
                "score": regulon.score if hasattr(regulon, 'score') else 0.0,
                "orthologousIdentity": regulon.orthologous_identity if hasattr(regulon, 'orthologous_identity') else 0.0,
                "annotation": regulon.annotation if hasattr(regulon, 'annotation') else '',
                "similarityQValue": regulon.score if hasattr(regulon, 'similarity_qvalue') else 0.0
            })
        return regulon_meta_data

    def create_loom_md_regulon_thresholds(self):
        return [self.get_regulon_meta_data(name, threshold) for name, threshold in self.auc_thresholds.iteritems()]

    def create_loom_regulon_gene_assignment(self):
        # Encode genes in regulons as "binary" membership matrix.
        genes = np.array(self.ex_mtx.columns)
        n_genes = len(genes)
        n_regulons = len(self.regulons)
        data = np.zeros(shape=(n_genes, n_regulons), dtype=int)
        for idx, regulon in enumerate(self.regulons):
            data[:, idx] = np.isin(genes, regulon.genes).astype(int)
        regulon_gene_assignment = pd.DataFrame(
            data=data,
            index=self.ex_mtx.columns,
            columns=list(map(attrgetter('name'), self.regulons))
        )
        return regulon_gene_assignment

    # Multi-runs SCENIC additional data

    def encode_regulon_gene_data(self, key):
        genes = np.array(self.ex_mtx.columns)
        n_genes = len(genes)
        n_regulons = len(self.regulons)
        data = np.zeros(shape=(n_genes, n_regulons), dtype=float)
        for idx, regulon in enumerate(self.regulons):
            regulon_data = pd.DataFrame.from_dict(self.get_regulon_gene_data(regulon, key), orient='index')
            data[np.isin(genes, regulon_data.index), idx] = regulon_data[0][genes[np.isin(genes, regulon_data.index)]]
        return data

    def create_loom_regulon_gene_data(self, key):
        return pd.DataFrame(
            data=self.encode_regulon_gene_data(key=key),
            index=self.ex_mtx.columns,
            columns=list(map(attrgetter('name'), self.regulons))
        )

    def add_row_attr_regulon_gene_weights(self):
        regulon_gene_weights = self.create_loom_regulon_gene_data(key='gene2weight')
        if 'RegulonGeneWeights' not in self.row_attrs.keys():
            print("Added 'RegulonGeneWeights' to the row attributes.")
            self.row_attrs['RegulonGeneWeights'] = SCopeLoom.df_to_named_matrix(regulon_gene_weights)

    def add_row_attr_regulon_gene_occurrences(self):
        regulon_gene_occurrences = self.create_loom_regulon_gene_data(key='gene2occurrence')
        if 'RegulonGeneOccurrences' not in self.row_attrs.keys():
            print("Added 'RegulonGeneOccurrences' to the row attributes.")
            self.row_attrs['RegulonGeneOccurrences'] = SCopeLoom.df_to_named_matrix(regulon_gene_occurrences)

    ###########
    # Generic #
    ###########

    def name2idx(self):
        return dict(map(reversed, enumerate(sorted(set(self.cell_annotations.values())))))

    def set_generic_col_attrs(self):
        self.col_attrs = {
            "CellID": self.ex_mtx.index.values.astype('str'),
            "nGene": self.ngenes.values,
            "Embedding": SCopeLoom.df_to_named_matrix(self.default_embedding.get_embedding()),
            "RegulonsAUC": SCopeLoom.df_to_named_matrix(self.auc_mtx),
            "Clusterings": SCopeLoom.df_to_named_matrix(self.clusterings),
            "ClusterID": self.clusterings.values
        }

    def set_generic_row_attrs(self):
        self.row_attrs = {
            "Gene": self.ex_mtx.columns.values.astype('str'),
            "Regulons": SCopeLoom.df_to_named_matrix(self.regulon_gene_assignment)
        }

    def set_generic_global_attrs(self):
        self.global_attrs = {
            "title": os.path.splitext(os.path.basename(self.out_fname))[0] if self.title is None else self.title,
            "MetaData": {
                "embeddings": [{'id': identifier, 'name': name} for identifier, name in self.id2name.items()],
                "annotations": [{
                    "name": "",
                    "values": []
                }],
                "clusterings": [{
                    "id": 0,
                    "group": "celltype",
                    "name": "Cell Type",
                    "clusters": [{"id": idx, "description": name} for name, idx in self.name2idx().items()]
                }],
                "regulonThresholds": self.regulon_thresholds
            },
            "Genome": self.nomenclature
        }

    def sort(self, axis, by):
        if(axis == 0):
            if by not in self.row_attrs:
                sys.exit(f"ERROR: Cannot sort on a row attribute that does not exist")
            sorted_idx = np.argsort(self.row_attrs[by])
            for ra_key in self.row_attrs.keys():
                self.row_attrs[ra_key] = self.row_attrs[ra_key][sorted_idx]
            self.ex_mtx = self.ex_mtx[self.row_attrs[by]]
        else:
            sys.exit(f"ERROR: Sorting by column axis is not implemented")

    def merge(self, loom, overwrite_embeddings=False):
        # Check all the cells and genes are in the same order
        if not np.array_equal(a1=self.get_cell_ids(), a2=loom.get_cell_ids()):
            sys.exit(f"ERROR: Column attribute CellIDs does not match between {os.path.basename(self.filename)} and {os.path.basename(loom.filename)}")
        if not np.array_equal(a1=self.get_genes(), a2=loom.get_genes()):
            sys.exit(f"ERROR: Row attribute 'Gene' does not match between {os.path.basename(self.filename)} and {os.path.basename(loom.filename)}")
        if not np.array_equal(a1=self.get_genes(), a2=self.ex_mtx.columns):
            sys.exit(f"ERROR: Row attribute 'Gene' does not match with the features in the matrix of {os.path.basename(self.filename)}.")
        if not np.array_equal(a1=loom.get_genes(), a2=loom.ex_mtx.columns):
            sys.exit(f"ERROR: Row attribute 'Gene' does not match with the features in the matrix of {os.path.basename(loom.filename)}.")

        # Add the embeddings of the given loom (SCopeLoom) to this SCopeLoom
        if 'embeddings' in loom.get_meta_data():

            print(f"Merging embeddings from {os.path.basename(loom.filename)} into {os.path.basename(self.filename)}...")
            for embedding in loom.get_meta_data()['embeddings']:
                # Only add the embedding if not present in this loom and overwrite_embeddings is False
                if not self.has_embedding(embedding_name=embedding['name']) or overwrite_embeddings:
                    print(f"Adding {embedding['name']} embedding...")
                    self.add_embedding(
                        embedding_name=embedding['name'],
                        embedding=loom.get_embedding_by_id(embedding_id=embedding['id'])
                    )
            print("Done.")

        # Add SCENIC results if present in the given loom
        if any('Regulon' in s for s in loom.row_attrs.keys()) and any('Regulon' in s for s in loom.col_attrs.keys()):

            print(f"Merging SCENIC results from {os.path.basename(loom.filename)} into {os.path.basename(self.filename)}...")

            # Regulon information from row attributes
            # All the row attributes containing the substring 'Regulon' within the given SCopeLoom
            # will be merged with this SCopeLoom
            regulons_row_attrs_keys = list(filter(lambda row_attrs_key: 'Regulon' in row_attrs_key, loom.row_attrs.keys()))
            regulons_row_attrs = {
                regulons_row_attr_key: loom.row_attrs[regulons_row_attr_key] for regulons_row_attr_key in regulons_row_attrs_keys
            }
            self.row_attrs.update(regulons_row_attrs)
            # Regulon information from column attributes
            # All the column attributes containing the substring 'Regulon' within the given SCopeLoom
            # will be merged with this SCopeLoom
            regulons_auc_col_attrs_keys = list(filter(lambda col_attrs_key: 'Regulon' in col_attrs_key, loom.col_attrs.keys()))
            regulons_auc_col_attrs = {
                regulons_auc_col_attr_key: loom.col_attrs[regulons_auc_col_attr_key] for regulons_auc_col_attr_key in regulons_auc_col_attrs_keys
            }
            self.col_attrs.update(regulons_auc_col_attrs)
            # regulonThresholds MetaData global attribute
            self.global_attrs["MetaData"].update(
                {
                    'regulonThresholds': loom.get_meta_data()['regulonThresholds']
                }
            )

            print("Done.")

        # If multi-runs mode
        is_multi_runs_mode = self.has_scenic_multi_runs_data()
        if is_multi_runs_mode:
            self.set_scenic_min_genes_regulon(min_genes_regulon=loom.scenic_min_genes_regulon)
            self.set_scenic_min_regulon_gene_occurrence(min_regulon_gene_occurrence=loom.scenic_min_regulon_gene_occurrence)

    def export(self, out_fname: str, save_embeddings: bool = True, compress_meta_data: bool = False):

        if out_fname is None:
            raise ValueError("The given out_fname cannot be None.")

        ##############
        # Embeddings #
        ##############
        if save_embeddings:
            self.col_attrs.update(self.create_loom_ca_embeddings_repr())
            self.global_attrs["MetaData"].update(self.create_loom_md_embeddings_repr())

        ##########
        # SCENIC #
        ##########
        is_multi_runs_mode = self.has_scenic_multi_runs_data()
        if is_multi_runs_mode:
            if "regulonSettings" not in self.global_attrs["MetaData"].keys():
                self.add_meta_data(_dict={
                    "regulonSettings": {
                        "min_genes_regulon": self.scenic_min_genes_regulon,
                        "min_regulon_gene_occurrence": self.scenic_min_regulon_gene_occurrence
                    }
                })

        if any('Regulon' in s for s in self.row_attrs.keys()):
            self.row_attrs.update(self.scopify_loom_ra_regulon_data())
        if any('RegulonsAUC' in s for s in self.col_attrs.keys()):
            self.col_attrs.update(self.scopify_loom_ca_regulon_data())
        if 'regulonThresholds' in self.global_attrs["MetaData"].keys():
            self.global_attrs["MetaData"].update(self.scopify_md_regulon_data())

        # Compress MetaData global attribute
        # (Should be compressed if Loompy version 2)
        self.global_attrs["MetaData"] = json.dumps(self.global_attrs["MetaData"])
        if compress_meta_data:
            self.global_attrs["MetaData"] = SCopeLoom.compress_encode(value=self.global_attrs["MetaData"])

        # Create loom file for use with the SCope tool.
        # The loom file format opted for rows as genes to facilitate growth along the column axis (i.e add more cells)
        # PySCENIC chose a different orientation because of limitation set by the feather format: selectively reading
        # information from disk can only be achieved via column selection. For the ranking databases this is of utmost
        # importance.
        lp.create(
            filename=out_fname,
            layers=self.ex_mtx.T.values,
            row_attrs=self.row_attrs,
            col_attrs=self.col_attrs,
            file_attrs=self.global_attrs
        )

    #########
    # Utils #
    #########

    @staticmethod
    def df_to_named_matrix(df: pd.DataFrame):
        # Create meta-data structure.
        # Create a numpy structured array
        return np.array([tuple(row) for row in df.values],
                        dtype=np.dtype(list(zip(df.columns, df.dtypes))))

    @staticmethod
    def compress_encode(value):
        '''
        Compress using ZLIB algorithm and encode the given value in base64.
        Taken from: https://github.com/aertslab/SCopeLoomPy/blob/5438da52c4bcf48f483a1cf378b1eaa788adefcb/src/scopeloompy/utils/__init__.py#L7
        '''
        return base64.b64encode(zlib.compress(value.encode('ascii'))).decode('ascii')

    @staticmethod
    def decompress_decode(value):
        try:
            value = value.decode('ascii')
            return json.loads(zlib.decompress(base64.b64decode(value)))
        except AttributeError:
            return json.loads(zlib.decompress(base64.b64decode(value.encode('ascii'))).decode('ascii'))
