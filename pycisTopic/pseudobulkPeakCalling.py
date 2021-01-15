import pyranges as pr
import pyBigWig
import pandas as pd
import ray
import logging
import sys
import subprocess

def exportPseudoBulk(cisTopic_obj, variable, chromsizes, n_cpu=1, bed_path=None, bigwig_path=None, normalize_bigwig=True, remove_duplicates=True):
    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    # Get fragments file
    path_to_fragments = cisTopic_obj.path_to_fragments
    if path_to_fragments == None:
        log.error('No fragments path in this cisTopic object. A fragments file is needed for forming pseudobulk profiles.')
    cell_data = cisTopic_obj.cell_data
    
    if isinstance(path_to_fragments , list):
        fragments_df_dict={}
        for path in path_to_fragments:
            log.info('Reading fragments from ' + path)
            path2project = cell_data.loc[cell_data['path_to_fragments'] == path, 'cisTopic_id'][1]
            fragments_df=pr.read_bed(path, as_df=True)
            fragments_df.loc[:,'Name'] = fragments_df.loc[:,'Name'] + '_' + path2project
            fragments_df = fragments_df.loc[fragments_df['Name'].isin(cell_data.index.tolist())]
            fragments_df_dict[path]=fragments_df
        fragments_df_list = [fragments_df_dict[list(fragments_df_dict.keys())[x]] for x in range(len(path_to_fragments))]
        log.info('Merging fragments')
        fragments_df = fragments_df_list[0].append(fragments_df_list[1:])
    else:
        log.info('Reading fragments')
        fragments_df=pr.read_bed(path_to_fragments, as_df=True)
        fragments_df = fragments_df.loc[fragments_df['Name'].isin(cell_data.index.tolist())]
        
    group_var=cisTopic_obj.cell_data.loc[:,variable]
    groups= sorted(list(set(group_var)))
    
    ray.init(num_cpus=n_cpu)
    paths = ray.get([exportPseudoBulk_ray.remote(group_var,
                                group,
                                fragments_df, 
                                chromsizes,
                                bigwig_path,
                                bed_path,
                                normalize_bigwig,
                                remove_duplicates) for group in groups])
    ray.shutdown()
    bw_paths =  {list(paths[x].keys())[0]:paths[x][list(paths[x].keys())[0]][0] for x in range(len(paths))}
    bed_paths = {list(paths[x].keys())[0]:paths[x][list(paths[x].keys())[0]][1] for x in range(len(paths))}
    return bw_paths, bed_paths

@ray.remote
def exportPseudoBulk_ray(group_var, group, fragments_df, chromsizes, bigwig_path, bed_path, normalize_bigwig=True, remove_duplicates=True):
    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    log.info('Creating pseudobulk for '+ str(group))
    barcodes=group_var[group_var.isin([group])].index.tolist()
    group_fragments=fragments_df.loc[fragments_df['Name'].isin(barcodes)]
    group_pr=pr.PyRanges(group_fragments)
    bigwig_path_group = bigwig_path + str(group) + '.bw'
    bed_path_group = bed_path + str(group) + '.bed.gz'
    if isinstance(bigwig_path, str):
        if remove_duplicates == True:
            group_pr.to_bigwig(path=bigwig_path_group, chromosome_sizes=chromsizes, rpm=normalize_bigwig)
        else:
            group_pr.to_bigwig(group_pr, path=bigwig_path_group, chromsizes=chromsizes, rpm=normalize_bigwig, value_col='Score')
    if isinstance(bed_path, str):
        group_pr.to_bed(path=bed_path_group, keep=True, compression='infer', chain=False)
    log.info(str(group)+' done!')
    return {group: [bigwig_path_group, bed_path_group]}

def peakCalling(macs_path, bed_paths, outdir, genome_size, n_cpu=1, input_format='BEDPE', shift=73, ext_size=146, keep_dup = 'all', q_value = 0.05):
    ray.init(num_cpus=n_cpu)
    narrow_peaks = ray.get([MACS_callPeak_ray.remote(macs_path,
                                bed_paths[name],
                                name,
                                outdir, 
                                genome_size,
                                input_format,
                                shift,
                                ext_size,
                                keep_dup,
                                q_value) for name in list(bed_paths.keys())])
    ray.shutdown()
    narrow_peaks_dict={list(bed_paths.keys())[i]: narrow_peaks[i].narrow_peak for i in range(len(narrow_peaks))} 
    return narrow_peaks_dict


@ray.remote
def MACS_callPeak_ray(macs_path, bed_path, name, outdir, genome_size, input_format='BEDPE', shift=73, ext_size=146, keep_dup = 'all', q_value = 0.05):
    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    MACS_peak_calling =MACS_callPeak(macs_path, bed_path, name, outdir, genome_size, input_format=input_format, shift=shift, ext_size=ext_size, keep_dup = keep_dup, q_value = q_value)
    log.info(name + ' done!')
    return MACS_peak_calling
    

class MACS_callPeak(): 
    def __init__(self, macs_path, bed_path, name, outdir, genome_size, input_format='BEDPE', shift=73, ext_size=146, keep_dup = 'all', q_value = 0.05):
        self.macs_path = macs_path
        self.treatment = bed_path
        self.name = name
        self.outdir = outdir
        self.format = input_format
        self.gsize = genome_size
        self.shift = shift
        self.ext_size = ext_size
        self.keep_dup = keep_dup
        self.qvalue = q_value
        self.callpeak()

    def callpeak(self):
        # Create logger
        level    = logging.INFO
        format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level = level, format = format, handlers = handlers)
        log = logging.getLogger('cisTopic')
        
        cmd = self.macs_path + ' callpeak --treatment %s --name %s  --outdir %s --format %s --gsize %s '\
            '--qvalue %s --nomodel --shift %s --extsize %s --keep-dup %s --call-summits --nolambda'

        cmd = cmd % (
            self.treatment, self.name, self.outdir, self.format, self.gsize,
            self.qvalue, self.shift, self.ext_size, self.keep_dup
        )
        log.info("Calling peaks for " + self.name + " with %s", cmd)
        try:
            subprocess.check_output(args=cmd, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        self.narrow_peak = self.load_narrow_peak()
        
    def load_narrow_peak(self):
        narrow_peak = pd.read_csv(self.outdir + self.name + '_peaks.narrowPeak', sep='\t', header = None)
        narrow_peak.columns = ['Chromosome', 'Start', 'End', 'Name', 'Score', 'Strand', 'FC_summit', '-log10_pval', '-log10_qval', 'Summit']
        narrow_peak_pr = pr.PyRanges(narrow_peak)
        return narrow_peak_pr
