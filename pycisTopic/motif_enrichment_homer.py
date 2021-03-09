import logging
import pandas as pd
import sys
import pyranges as pr # with pyfaidx
import ray
import os
import subprocess

from .utils import *
from IPython.display import HTML

def homer_results(homer_dict, name, results='known'):
    if results == 'known':
        file = homer_dict[name].outdir + '/knownResults.html'
    if results == 'denovo':
        file = homer_dict[name].outdir + '/homerResults.html'
    inplace_change(file, 'width="505" height="50"', 'width="1010" height="200"')
    return HTML(file)

def homer_find_motifs_genome(homer_path, input_data, outdir, genome, size='given', mask=True, denovo=False, length='8,10,12', n_cpu=5):
    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    # Format input data
    pr_regions_dict = format_input_regions(input_data)
    # Save regions in dict to the output dir
    bed_paths={}
    bed_dir = outdir+'regions_bed/'
    # Create bed directory
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(bed_dir):
        os.mkdir(bed_dir)
    # Create bed files for Homer
    for key in pr_regions_dict.keys():
        bed_path = bed_dir+key+'.bed'
        pr_regions_dict[key].to_bed(path=bed_path, keep=False, compression='infer', chain=False)
        bed_paths[key] = bed_path
    # Run Homer
    ray.init(num_cpus=n_cpu)
    homer_dict = ray.get([homer_ray.remote(homer_path,
                                bed_paths[name],
                                name,
                                outdir + name, 
                                genome,
                                size,
                                mask,
                                denovo,
                                length) for name in list(bed_paths.keys())])
    ray.shutdown()
    homer_dict={list(bed_paths.keys())[i]: homer_dict[i] for i in range(len(homer_dict))}
    return homer_dict

@ray.remote
def homer_ray(homer_path, bed_path, name, outdir, genome, size='given', mask=True, denovo=False, length='8,10,12'):
    # Create logger
    level    = logging.INFO
    format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(level = level, format = format, handlers = handlers)
    log = logging.getLogger('cisTopic')
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    log.info('Running '+ name)
    Homer_res = Homer(homer_path, bed_path, name, outdir, genome, size, mask, denovo, length)
    log.info(name + ' done!')
    return Homer_res

class Homer(): 
    def __init__(self, homer_path, bed_path, name, outdir, genome, size='given', mask=True, denovo=False, length='8,10,12'):
        self.homer_path = homer_path
        self.bed_path = bed_path
        self.genome = genome
        self.outdir = outdir
        self.size = size
        self.len = length
        self.mask = mask
        self.denovo = denovo
        self.name = name
        self.run()

    def run(self):
        # Create logger
        level    = logging.INFO
        format   = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level = level, format = format, handlers = handlers)
        log = logging.getLogger('cisTopic')
        
        if self.mask == True and self.denovo == False:
            cmd = self.homer_path + ' %s %s %s -preparsedDir %s -size %s -len %s -mask -nomotif'
        if self.mask == True and self.denovo == True:
            cmd = self.homer_path + ' %s %s %s -preparsedDir %s -size %s -len %s -mask'
        if self.mask == False and self.denovo == False:
            cmd = self.homer_path + ' %s %s %s -preparsedDir %s -size %s -len %s -nomotif'
        if self.mask == False and self.denovo == True:
            cmd = self.homer_path + ' %s %s %s -preparsedDir %s -size %s -len %s -nomotif'
            
        cmd = cmd % (self.bed_path, self.genome, self.outdir, self.outdir, self.size, self.len)
        log.info("Running Homer for " + self.name + " with %s", cmd)
        try:
            subprocess.check_output(args=cmd, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        
        try:
            self.known = self.load_known()
        except:
            log.info('No known results found')
        if self.denovo == True:
            try:
                self.denovo_path = self.load_denovo()
            except:
                log.info('No de novo results found')
        
    def load_known(self):
        known = pd.read_csv(self.outdir + '/knownResults.txt', sep='\t')
        return known
    
    def load_denovo(self):
        known = pd.read_csv(self.outdir + '/homerResults.txt', sep='\t')
        return known

