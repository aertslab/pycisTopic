use pyo3::prelude::*;
use rust_htslib::tbx::{self, Read};
use std::collections::{HashMap, HashSet};
use itertools::Itertools;

fn precompute_reverse_map(hashmap: &HashMap<String, HashSet<String>>) -> HashMap<&str, Vec<&String>> {
    let mut reverse_map: HashMap<&str, Vec<&String>> = HashMap::new();

    for (key, set) in hashmap.iter() {
        for value in set.iter() {
            reverse_map
                .entry(value.as_str())
                .or_insert_with(Vec::new)
                .push(key);
        }
    }

    reverse_map
}

fn keys_containing_value<'a>(
    reverse_map: &'a HashMap<&str, Vec<&String>>,
    value: &'a str,
) -> Option<&'a Vec<&'a String>> {
    reverse_map.get(value)
}

#[pyfunction]
fn get_fragments_for_cell_barcodes(
    path_to_fragments: &str,
    cell_type_to_cell_barcodes: HashMap<String, HashSet<String>>,
    chromsizes: HashMap<String, u64>,
    number_of_threads: usize) -> PyResult<HashMap<String, Vec<String>>> {
        let cell_barcodes_to_cell_types = precompute_reverse_map(&cell_type_to_cell_barcodes);
        let mut tbx_reader = tbx::Reader::from_path(path_to_fragments)
            .expect(&format!("Could not open file {}", path_to_fragments));
        tbx_reader.set_threads(number_of_threads)
            .expect(&format!("Could not set number of threads to {}", number_of_threads));
        let contigs = tbx_reader.seqnames();
        // check that all contigs in chromsizes are also in the fragments file
        for contig in chromsizes.keys() {
            if !contigs.contains(&contig) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Contig {} is not in fragments file\nCheck chromsizes", contig)));
            }
        }
        let mut cell_type_to_fragments: HashMap<String, Vec<String>> = HashMap::new();
        // initialize cell_type_to_fragments
        for cell_type in cell_type_to_cell_barcodes.keys() {
            cell_type_to_fragments.insert(cell_type.clone(), Vec::new());
        }
        for contig in chromsizes.keys().sorted() {
            println!("Processing contig {}", contig);
            let contig_id = tbx_reader.tid(contig)
                .expect(&format!("Could not get contig id for contig {}", contig));
            let contig_size = chromsizes.get(contig).unwrap();
            tbx_reader.fetch(contig_id, 0, *contig_size)
                .expect(&format!("Could not fetch contig {} from fragments file", contig));
            let mut read: Vec<u8> = Vec::new();
            let mut not_at_end = tbx_reader.read(&mut read).unwrap();
            let mut read_as_str = String::from_utf8(read.clone()).unwrap();
            while not_at_end {
                let read_cb = read_as_str.split('\t').nth(3).unwrap();
                if let Some(cell_types_to_add_read_to) = keys_containing_value(&cell_barcodes_to_cell_types, read_cb){
                    for cell_type in cell_types_to_add_read_to {
                        cell_type_to_fragments.get_mut(*cell_type).unwrap().push(read_as_str.clone());
                    }
                }
                not_at_end = tbx_reader.read(&mut read).unwrap();
                read_as_str = String::from_utf8(read.clone()).unwrap();
            }

        }
        return Ok(cell_type_to_fragments)
}

#[pymodule]
fn pseudobulk(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_fragments_for_cell_barcodes, m)?)?;
    Ok(())
}
