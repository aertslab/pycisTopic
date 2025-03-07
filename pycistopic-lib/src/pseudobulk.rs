use super::custom_errors;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::cmp::Reverse;
use std::io::{BufRead, Write};
use flate2::Compression;
use pyo3::prelude::*;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use std::thread;


#[derive(Eq, PartialEq)]
struct GenomicRange {
    chromosome: String,
    start: usize,
    end: usize,
    cell_barcode: String,
    score: Option<usize>,
    file_index: usize,
    file_name: String
}

impl Ord for GenomicRange {
    fn cmp(&self, other: &GenomicRange) -> std::cmp::Ordering {
           self.start.cmp(&other.start)
            .then(self.end.cmp(&other.end))
    }
}

impl PartialOrd for GenomicRange {
    fn partial_cmp(&self, other: &GenomicRange) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl GenomicRange {
    fn new(line: String, file_index: usize, filename: &str) -> Result<GenomicRange, custom_errors::InvalidFragmentFileError> {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 4 {
            return Err(custom_errors::InvalidFragmentFileError::new(filename));
        }
        Ok(GenomicRange{
            chromosome: fields[0].to_string(),
            start: fields[1].parse::<usize>()
                .map_err(|_| custom_errors::InvalidFragmentFileError::new(filename))?,
            end: fields[2].parse::<usize>()
                .map_err(|_| custom_errors::InvalidFragmentFileError::new(filename))?,
            cell_barcode: fields[3].to_string(),
            score: if fields.len() > 4 {
                    Some(fields[4].parse::<usize>()
                        .map_err(|_| custom_errors::InvalidFragmentFileError::new(filename))?
                    )
                } else {None},
            file_index,
            file_name: filename.to_string()
        })
    }
}

impl fmt::Display for GenomicRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.score {
            Some(score) => write!(
                f,
                "{}\t{}\t{}\t{}\t{}",
                self.chromosome, self.start, self.end, self.cell_barcode, score
            ),
            None => write!(
                f,
                "{}\t{}\t{}\t{}",
                self.chromosome, self.start, self.end, self.cell_barcode
            ),
        }
    }
}


fn split_fragments_by_cell_barcodes_for_chromosome(
    fragment_file_paths: &[&str],
    fragment_file_to_cell_barcode: &HashMap<String, Vec<String>>,
    chromosome: &str,
    gz_output_file: &mut GzEncoder<File>
) -> PyResult<()>{

    // Open fragment files which are gzipped, and pos-sorted.
    let mut readers: Vec<_> = fragment_file_paths
        .iter()
        .map(|&path| {
            let file = File::open(Path::new(path))?;
            let d_file = MultiGzDecoder::new(file);
            Ok(std::io::BufReader::new(d_file))
        })
        .collect::<Result<_, std::io::Error>>()?;

    // A binary heap will be used to write fragments in order from different files
    let mut heap = BinaryHeap::new();
    
    let mut line = "#".to_string();
    let mut bytes_read;

    // Push the first fragment from each file to the heap
    for (
            fragment_file,
            (index, reader)
    ) in fragment_file_paths.iter().zip(readers.iter_mut().enumerate()) {
        // skip header lines
        while line.starts_with("#") {
            line.clear();
            reader.read_line(&mut line)?;
            line = line.trim().to_string();
        }
        // Loop until a fragment with cell barcode, on the correct chromosome, in fragment_file_to_cell_barcode is found
        let mut fragment_found = false;
        // is the barcode found for the first (non-header) line?
        let fragment = GenomicRange::new(
            line.to_string(), index, fragment_file
        )?;
        match fragment_file_to_cell_barcode.get(&fragment_file.to_string()) {
            Some(cell_barcodes) => {
                if cell_barcodes.contains(&fragment.cell_barcode) && fragment.chromosome == chromosome {
                    // Reverse so that "smaller" fragments (i.e. lower genomic location
                    // are written first later on (heap will pop large elements first).
                    heap.push(Reverse(fragment));
                    fragment_found = true;
                }
            },
            None => {
                return Err(
                    custom_errors::ValueError::new(
                        format!(
                            "fragment_file_to_cell_barcode does not contain entry for {}",
                            fragment_file)
                    ).into());
            }
        }

        // barcode not in first line.
        // keep reading until a fragment with correct barcode, on correct chromosome, is found.
        while !fragment_found {
            line.clear();
            bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                // end of file
                break;
            }
            line = line.trim().to_string();
            let fragment = GenomicRange::new(
                line.to_string(), index, fragment_file
            )?;
            match fragment_file_to_cell_barcode.get(&fragment_file.to_string()) {
                Some(cell_barcodes) => {
                    if cell_barcodes.contains(&fragment.cell_barcode) && fragment.chromosome == chromosome {
                        // Reverse so that "smaller" fragments (i.e. lower genomic location
                        // are written first later on (heap will pop large elements first).
                        heap.push(Reverse(fragment));
                        fragment_found = true;
                    }
                },
                None => {
                    return Err(
                        custom_errors::ValueError::new(
                            format!(
                                "fragment_file_to_cell_barcode does not contain entry for {}",
                                fragment_file)
                        ).into());
                }
            }
        }
    }

    while let Some(Reverse(fragment)) = heap.pop() {
        gz_output_file.write_all(format!("{}\n", fragment).as_bytes())?;
        // Loop until a fragment with cell barcode in fragment_file_to_cell_barcode is found
        let mut fragment_found  = false;
        while !fragment_found {
            // Read next range from file that had the smallest range and add this to the heap.
            line.clear();
            bytes_read = readers[fragment.file_index].read_line(&mut line)?;
            if bytes_read == 0 {
                // end of file
                break;
            }
            line = line.trim().to_string();
            let next_fragment = GenomicRange::new(
                line.to_string(), fragment.file_index, &fragment.file_name
            )?;
            // Assuming that the fragment files are sorted.
            // Using the previous while loop, for each file, we should be at the correct location of the file
            // (i.e. where the current chromosomes are located).
            // if the next fragment file has a different chromosome, we should be done with this file and we can skip it.
            if next_fragment.chromosome != chromosome {
                break;
            }
            match fragment_file_to_cell_barcode.get(&next_fragment.file_name) {
                Some(cell_barcodes) => {
                    if cell_barcodes.contains(&next_fragment.cell_barcode) {
                        heap.push(Reverse(next_fragment));
                        fragment_found = true;
                    }
                },
                None => {
                    return Err(
                        custom_errors::ValueError::new(
                            format!(
                                "fragment_file_to_cell_barcode does not contain entry for {}",
                                next_fragment.file_name)
                        ).into()
                    );
                }
            }
        }
    }
    Ok(())
}

#[pyfunction]
pub fn split_fragment_files_by_cell_type(
    fragment_file_paths: Vec<String>,
    output_file_prefix: &str,
    cell_type_to_fragment_file_to_cell_barcode: HashMap<String, HashMap<String, Vec<String>>>,
    chromosomes: Vec<String>
) -> PyResult<()> {
    for cell_type in cell_type_to_fragment_file_to_cell_barcode.keys() {
        let mut handles: Vec<thread::JoinHandle<_>> = Vec::new();
        for chromosome in &chromosomes {
            let output_file_name = format!("{}_{}.{}.tsv.gz", output_file_prefix, cell_type, chromosome);
            let fragment_file_paths = fragment_file_paths.clone(); // Need to clone since threads take ownership
            let fragment_file_to_cell_barcode = cell_type_to_fragment_file_to_cell_barcode
                .get(cell_type)
                .unwrap()
                .clone();
            let file = File::create(output_file_name)?;
            let chromosome = chromosome.clone();
            let handle = thread::spawn(move || {
                let mut gz_output_file = GzEncoder::new(file, Compression::default());
                split_fragments_by_cell_barcodes_for_chromosome(
                    &fragment_file_paths.iter().map(|p| p.as_str()).collect::<Vec<_>>(),
                    &fragment_file_to_cell_barcode,
                    &chromosome,
                    &mut gz_output_file
                )
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.join().expect("Thread panicked")?;
        }
        // concat all chromosomes
        let output_file_name = format!("{}_{}.tsv.gz", output_file_prefix, cell_type);
        let output_file = File::create(&output_file_name)?;
        let mut writer = std::io::BufWriter::new(output_file);
        for chromosome in &chromosomes {
            let input_file_name = format!("{}_{}.{}.tsv.gz", output_file_prefix, cell_type, chromosome);
            let mut input_file = File::open(&input_file_name)?;
            std::io::copy(&mut input_file, &mut writer)?;
        }
        writer.flush()?;
    }
    Ok(())
}
