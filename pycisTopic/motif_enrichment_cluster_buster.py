import io
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import ssl
from IPython.display import HTML
ssl._create_default_https_context = ssl._create_unverified_context
pd.set_option('display.max_colwidth', None)

from .diff_features import *
from .utils import *


@ray.remote
def run_cluster_buster_for_motif(cluster_buster_path, fasta_filename, motif_filename, motif_name, i, nr_motifs, verbose = False):
    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    if verbose == True:
        log.info('Scoring motif '+str(i)+ ' out of '+str(nr_motifs)+' motifs')
    # Score each region in FASTA file with Cluster-Buster
    # for motif and get top CRM score for each region.
    clusterbuster_command = [cluster_buster_path,
                             '-f', '4',
                             '-c', '0.0',
                             '-r', '10000',
                             '-t', '1',
                             '-l', #Mask repeats
                             motif_filename,
                             fasta_filename]

    try:
        pid = subprocess.Popen(args=clusterbuster_command,
                               bufsize=1,
                               executable=None,
                               stdin=None,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               preexec_fn=None,
                               close_fds=False,
                               shell=False,
                               cwd=None,
                               env=None,
                               universal_newlines=False,
                               startupinfo=None,
                               creationflags=0)
        stdout_data, stderr_data = pid.communicate()
    except OSError as msg:
        print("\nExecution error for: '" + ' '.join(clusterbuster_command) + "': " + str(msg),
              file=sys.stderr)
        sys.exit(1)

    if pid.returncode != 0:
        print("\nError: Non-zero exit status for: " + ' '.join(clusterbuster_command) + "'",
              file=sys.stderr)
        sys.exit(1)

    crm_scores_df = pd.read_csv(
        filepath_or_buffer=io.BytesIO(stdout_data),
        sep='\t',
        header=0,
        names=['seq_name', 'crm_score', 'seq_number', 'rank'],
        index_col='seq_name',
        usecols=['seq_name','crm_score'],
        dtype={'crm_score': np.float32},
        engine='c'
    )
    
    crm_scores_df.columns=[motif_name]
    return crm_scores_df

def get_sequence_names_from_fasta(fasta_filename):
    sequence_names_list = list()
    sequence_names_set = set()
    duplicated_sequences = False

    with open(fasta_filename, 'r') as fh:
        for line in fh:
            if line.startswith('>'):
                # Get sequence name by getting everything after '>' up till the first whitespace.
                sequence_name = line[1:].split(maxsplit=1)[0]

                # Check if all sequence names only appear once.
                if sequence_name in sequence_names_set:
                    print(
                        'Error: Sequence name "{0:s}" is not unique in FASTA file "{1:s}".'.format(
                            sequence_name,
                            fasta_filename
                        ),
                        file=sys.stderr
                    )
                    duplicated_sequences = True

                sequence_names_list.append(sequence_name)
                sequence_names_set.add(sequence_name)

    if duplicated_sequences:
        sys.exit(1)

    return sequence_names_list


def pyranges2names(regions):
    return ['>'+str(chrom) + ":" + str(start) + '-' + str(end) for chrom, start, end in zip(list(regions.Chromosome), list(regions.Start), list(regions.End))]

def grep(l, s):
    return [i for i in l if s in i]

def cluster_buster(cbust_path, input_data, outdir, path_to_fasta, path_to_motifs, n_cpu=1, motifs=None, verbose=False):
    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    # Format input data
    pr_regions_dict = format_input_regions(input_data)
    # Generate fasta file
    if not os.path.exists(outdir):
        os.mkdir(outdir)  
    if not os.path.exists(outdir+'regions.fa'):
        log.info('Getting sequences')
        pr_regions_names_dict = {key: pyranges2names(pr_regions_dict[key]) for key in pr_regions_dict.keys()}
        pr_sequence_list = [pd.DataFrame([pr_regions_names_dict[key], pr.get_fasta(pr_regions_dict[key], path_to_fasta).tolist()], index=['Name', 'Sequence'], columns=pr_regions_names_dict[key]) for key in pr_regions_dict.keys()]
        seq_df = pd.concat(pr_sequence_list, axis=1)
        seq_df = seq_df.loc[:,~seq_df.columns.duplicated()]
        seq_df.T.to_csv(outdir+'regions.fa', header=False, index=False, sep='\n')
        sequence_names =  [seq[1:] for seq in seq_df.columns]
    else:
        sequence_names = get_sequence_names_from_fasta(outdir+'regions.fa')
    # Get motifs and sequence name
    if motifs is None:
        motifs = os.listdir(path_to_motifs)
        motifs = grep(motifs, '.cb')
    

    log.info('Scoring sequences')
    ray.init(num_cpus=n_cpu)
    crm_scores = ray.get([run_cluster_buster_for_motif.remote(cbust_path, outdir+'regions.fa', path_to_motifs+motifs[i], motifs[i], i, len(motifs), verbose) for i in range(len(motifs))])
    ray.shutdown()
    crm_df = pd.concat(crm_scores, axis=1, sort=False).fillna(0).T
    # Remove .cb from motifs names
    crm_df.index = [x.replace('.cb','') for x in crm_df.index.tolist()]
    log.info('Done!')
    return crm_df


def find_enriched_motifs(crm_df, group_dict, var_features=None, contrasts=None, contrast_name='contrast', adjpval_thr=0.05, log2fc_thr=1, n_cpu=1, tmp_dir=None, memory=None, object_store_memory=None):
    # Create cisTopic logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    contrast_keys=[x for x in group_dict.keys()]
    if isinstance(group_dict[contrast_keys[0]], pr.PyRanges):
        group_dict = {key: group_dict[key].df for key in group_dict.keys()}
    
    if isinstance(group_dict[contrast_keys[0]], pd.DataFrame):
        if group_dict[contrast_keys[0]].shape[1] >= 3:
            group_dict = {key: [group_dict[key].iloc[i,0] + ':' + str(group_dict[key].iloc[i,1]) + '-' + str(group_dict[key].iloc[i,2]) for i in range(len(group_dict[key]))] for key in group_dict.keys()}
        else:
            group_dict={key: group_dict[key].index.tolist() for key in group_dict.keys()}

    if contrasts == None:
        levels=list(group_dict.keys())
        contrasts=[[[x], levels[:levels.index(x)] + levels[levels.index(x)+1:]] for x in levels]
        contrasts_names=levels
    else:
        contrasts_names=['_'.join(contrasts[i][0]) + '_VS_' +'_'.join(contrasts[i][1]) for i in range(len(contrasts))]
    # Get region groups
    barcode_groups = [[list(set(sum([group_dict[key] for key in contrasts[x][0]],[]))), list(set(sum([group_dict[key] for key in contrasts[x][1]],[])))] for x in range(len(contrasts))]

    # Subset imputed accessibility matrix
    if var_features is not None:
        crm_df = crm_df.loc[var_features,:]
    # Compute p-val and log2FC
    ray.init(num_cpus=n_cpu, temp_dir = tmp_dir, memory=memory, object_store_memory=object_store_memory)
    markers_list=ray.get([markers_ray.remote(crm_df, barcode_groups[i], contrasts_names[i], adjpval_thr=adjpval_thr, log2fc_thr=log2fc_thr) for i in range(len(contrasts))])
    ray.shutdown()
    markers_dict={contrasts_names[i]: markers_list[i] for i in range(len(markers_list))} 
    return markers_dict

def load_motif_annotations(specie: str,
                           version: str = 'v9',
                           fname: str = None,
                           column_names=('#motif_id', 'gene_name',
                                         'motif_similarity_qvalue', 'orthologous_identity', 'description'),
                           motif_similarity_fdr: float = 0.001,
                           orthologous_identity_threshold: float = 0.0) -> pd.DataFrame:
    """
    Load motif annotations from a motif2TF snapshot.
    :param fname: the snapshot taken from motif2TF.
    :param column_names: the names of the columns in the snapshot to load.
    :param motif_similarity_fdr: The maximum False Discovery Rate to find factor annotations for enriched motifs.
    :param orthologuous_identity_threshold: The minimum orthologuous identity to find factor annotations
        for enriched motifs.
    :return: A dataframe.
    """
    # Create a MultiIndex for the index combining unique gene name and motif ID. This should facilitate
    # later merging.
    if fname is None:
        if specie == 'mus_musculus':
            name='mgi'
        elif specie == 'homo_sapiens':
            name='hgnc'
        elif specie == 'drosophila_melanogaster':
            name='flybase'
        fname = 'https://resources.aertslab.org/cistarget/motif2tf/motifs-'+version+'-nr.'+name+'-m0.001-o0.0.tbl'
    df = pd.read_csv(fname, sep='\t', usecols=column_names)
    df.rename(columns={'#motif_id':"MotifID",
                       'gene_name':"TF",
                       'motif_similarity_qvalue': "MotifSimilarityQvalue",
                       'orthologous_identity': "OrthologousIdentity",
                       'description': "Annotation" }, inplace=True)
    df = df[(df["MotifSimilarityQvalue"] <= motif_similarity_fdr) &
            (df["OrthologousIdentity"] >= orthologous_identity_threshold)]
    
    # Direct annotation
    df_direct_annot = df[(df["MotifSimilarityQvalue"]<= 0) & (df["OrthologousIdentity"] >= 1)]
    df_direct_annot = df_direct_annot.groupby(['MotifID'])['TF'].apply(lambda x: ', '.join(x)).reset_index()
    df_direct_annot.index = df_direct_annot['MotifID']
    df_direct_annot = pd.DataFrame(df_direct_annot['TF'])
    df_direct_annot.columns = ['Direct_annot']
    # Indirect annotation - by motif similarity
    motif_similarity_annot = df[(df["MotifSimilarityQvalue"]> 0) & (df["OrthologousIdentity"] >= 1)]
    motif_similarity_annot = motif_similarity_annot.groupby(['MotifID'])['TF'].apply(lambda x: ', '.join(x)).reset_index()
    motif_similarity_annot.index =  motif_similarity_annot['MotifID']
    motif_similarity_annot = pd.DataFrame(motif_similarity_annot['TF'])
    motif_similarity_annot.columns = ['Motif_similarity_annot']
    # Indirect annotation - by orthology
    orthology_annot = df[(df["MotifSimilarityQvalue"]<= 0) & (df["OrthologousIdentity"] < 1)]
    orthology_annot = orthology_annot.groupby(['MotifID'])['TF'].apply(lambda x: ', '.join(x)).reset_index()
    orthology_annot.index = orthology_annot['MotifID']
    orthology_annot = pd.DataFrame(orthology_annot['TF'])
    orthology_annot.columns = ['Orthology_annot']
    # Indirect annotation - by orthology
    motif_similarity_and_orthology_annot = df[(df["MotifSimilarityQvalue"]> 0) & (df["OrthologousIdentity"] < 1)]
    motif_similarity_and_orthology_annot = motif_similarity_and_orthology_annot.groupby(['MotifID'])['TF'].apply(lambda x: ', '.join(x)).reset_index()
    motif_similarity_and_orthology_annot.index = motif_similarity_and_orthology_annot['MotifID']
    motif_similarity_and_orthology_annot = pd.DataFrame(motif_similarity_and_orthology_annot['TF'])
    motif_similarity_and_orthology_annot.columns = ['Motif_similarity_and_Orthology_annot']
    # Combine
    df = pd.concat([df_direct_annot, motif_similarity_annot, orthology_annot, motif_similarity_and_orthology_annot], axis=1, sort=False)
    return df

def add_motif_annotation(motif_enrichment_dict,
                       specie,
                       version,
                       path_to_motif_annotations=None,
                       motif_similarity_fdr= 0.001,
                       orthologous_identity_threshold= 0.0,
                       add_logo=True):
    # Read motif annotation. 
    annot_df = load_motif_annotations(specie,
                                      version,
                                      fname=path_to_motif_annotations,
                                      motif_similarity_fdr = motif_similarity_fdr,
                                      orthologous_identity_threshold = orthologous_identity_threshold)
        
    # Add info to elements in dict
    motif_enrichment_dict_wAnnot = {key: pd.concat([motif_enrichment_dict[key], annot_df], axis=1, sort=False).loc[motif_enrichment_dict[key].index.tolist(),:] for key in motif_enrichment_dict.keys()}
    if add_logo == True:
        for key in motif_enrichment_dict.keys():
            motif_enrichment_dict_wAnnot[key]['Logo']=['<img src="' +'https://motifcollections.aertslab.org/' + version + '/logos/'+ motif_enrichment_dict_wAnnot[key].index.tolist()[i] + '.png' + '" width="200" >' for i in range(motif_enrichment_dict_wAnnot[key].shape[0])]
        motif_enrichment_dict_wAnnot = {key: motif_enrichment_dict_wAnnot[key].loc[:,['Logo', 'Contrast', 'Direct_annot', 'Motif_similarity_annot', 'Orthology_annot', 'Motif_similarity_and_Orthology_annot', 'Log2FC', 'Adjusted_pval']] for key in motif_enrichment_dict_wAnnot.keys()}
    else:
        motif_enrichment_dict_wAnnot = {key: motif_enrichment_dict_wAnnot[key].loc[:,['Contrast', 'Direct_annot', 'Motif_similarity_annot', 'Orthology_annot', 'Motif_similarity_and_Orthology_annot', 'Log2FC', 'Adjusted_pval']] for key in motif_enrichment_dict_wAnnot.keys()}
        
    return motif_enrichment_dict_wAnnot 

def cbust_results(motif_enrichment_dict_wAnnot, name=None):
    if name is None:
        motif_enrichment_table=pd.concat([motif_enrichment_dict[key] for key in motif_enrichment_dict.keys()], axis=0, sort=False)
    else:
        motif_enrichment_table=motif_enrichment_dict_wAnnot[name]
    return HTML(motif_enrichment_table.to_html(escape=False, col_space=80))
