import gc
import gzip
import logging
import math
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyranges as pr
import re
import sys
from PIL import Image
from scipy import sparse


def region_names_to_coordinates(region_names):
    chrom = pd.DataFrame([i.split(':', 1)[0] for i in region_names if ':' in i])
    coor = [i.split(':', 1)[1] for i in region_names if ':' in i]
    start = pd.DataFrame([int(i.split('-', 1)[0]) for i in coor])
    end = pd.DataFrame([int(i.split('-', 1)[1]) for i in coor])
    regiondf = pd.concat([chrom, start, end], axis=1, sort=False)
    regiondf.index = [i for i in region_names if ':' in i]
    regiondf.columns = ['Chromosome', 'Start', 'End']
    return (regiondf)


def get_position_index(query_list, target_list):
    d = {k: v for v, k in enumerate(target_list)}
    index = (d[k] for k in query_list)
    return list(index)


def non_zero_rows(X):
    if isinstance(X, sparse.csr_matrix):
        # Remove all explicit zeros in sparse matrix.
        X.eliminate_zeros()
        # Get number of non zeros per row and get indices for each row which is
        # not completely zero.
        return np.nonzero(X.getnnz(axis=1))[0]
    else:
        # For non sparse matrices.
        return np.nonzero(np.count_nonzero(X, axis=1))[0]


def subset_list(target_list, index_list):
    X = list(map(target_list.__getitem__, index_list))
    return X


def loglikelihood(nzw, ndz, alpha, eta):
    D = ndz.shape[0]
    n_topics = ndz.shape[1]
    vocab_size = nzw.shape[1]

    const_prior = (n_topics * math.lgamma(alpha) -
                   math.lgamma(alpha * n_topics)) * D
    const_ll = (vocab_size * math.lgamma(eta) -
                math.lgamma(eta * vocab_size)) * n_topics

    # calculate log p(w|z)
    topic_ll = 0
    for k in range(n_topics):
        sum = eta * vocab_size
        for w in range(vocab_size):
            if nzw[k, w] > 0:
                topic_ll = math.lgamma(nzw[k, w] + eta)
                sum += nzw[k, w]
        topic_ll -= math.lgamma(sum)

    # calculate log p(z)
    doc_ll = 0
    for d in range(D):
        sum = alpha * n_topics
        for k in range(n_topics):
            if ndz[d, k] > 0:
                doc_ll = math.lgamma(ndz[d, k] + alpha)
                sum += ndz[d, k]
        doc_ll -= math.lgamma(sum)

    ll = doc_ll - const_prior + topic_ll - const_ll
    return ll


def sparse2bow(X):
    for indprev, indnow in zip(X.indptr, X.indptr[1:]):
        yield np.array(X.indices[indprev:indnow])


def chunks(l, n):
    return [l[x: x + n] for x in xrange(0, len(l), n)]


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten()  # all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative
    array = np.sort(array)  # values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements
    return ((np.sum((2 * index - n - 1) * array)) /
            (n * np.sum(array)))  # Gini coefficient


def regions_overlap(target, query):
    # Read input
    if isinstance(target, str):
        target_pr = pr.read_bed(target)
    if isinstance(target, list):
        target_pr = pr.PyRanges(regionNamesToCoordinates(target))
    if isinstance(target, pr.PyRanges):
        target_pr = target
    # Read input
    if isinstance(query, str):
        query_pr = pr.read_bed(query)
    if isinstance(query, list):
        query_pr = pr.PyRanges(regionNamesToCoordinates(query))
    if isinstance(query, pr.PyRanges):
        query_pr = query

    target_pr = target_pr.overlap(query_pr)
    selected_regions = (
        target_pr.Chromosome.astype(str) + ":" + target_pr.Start.astype(str) + "-" + target_pr.End.astype(str)
    ).to_list()
    return selected_regions


def format_input_regions(input_data):
    new_data = {}
    for key in input_data.keys():
        data = input_data[key]
        if isinstance(data, pd.DataFrame):
            regions = data.index.tolist()
            if len(regions) > 0:
                new_data[key] = pr.PyRanges(regionNamesToCoordinates(regions))
        else:
            new_data[key] = data
    return new_data


def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            return
    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        s = s.replace(old_string, new_string)
        f.write(s)


def load_cisTopic_model(path_to_cisTopic_model_matrices):
    metrics = None
    coherence = None
    marg_topic = None
    topic_ass = None
    cell_topic = pd.read_feather(
        path_to_cisTopic_model_matrices +
        'cell_topic.feather')
    cell_topic.index = ['Topic' + str(x)
                        for x in range(1, cell_topic.shape[0] + 1)]
    topic_region = pd.read_feather(
        path_to_cisTopic_model_matrices +
        'topic_region.feather')
    topic_region.index = [
        'Topic' +
        str(x) for x in range(
            1,
            topic_region.shape[0] +
            1)]
    topic_region = topic_region.T
    parameters = None
    model = cisTopicLDAModel(
        metrics,
        coherence,
        marg_topic,
        topic_ass,
        cell_topic,
        topic_region,
        parameters)
    return model


def prepare_tag_cells(cell_names, split_pattern = '___'):
    if split_pattern == '-':
        new_cell_names = [
            re.findall(
                r"^[ACGT]*-[0-9]+-",
                x)[0].rstrip('-') if len(
                re.findall(
                    r"^[ACGT]*-[0-9]+-",
                    x)) != 0 else x for x in cell_names]
        new_cell_names = [
            re.findall(
                r"^\w*-[0-9]*",
                new_cell_names[i])[0].rstrip('-') if (
                                                             len(
                                                                 re.findall(
                                                                     r"^\w*-[0-9]*",
                                                                     new_cell_names[i])) != 0) & (
                                                             new_cell_names[i] == cell_names[i]) else new_cell_names[i] for
            i in range(
                len(new_cell_names))]
    else:
        new_cell_names = [x.split(split_pattern)[0]
            for x in cell_names]
            
    return new_cell_names


def multiplot_from_generator(
        g,
        num_columns,
        n_plots,
        figsize=None,
        plot=True,
        save=None):
    if save is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)
    # call 'next(g)' to get past the first 'yield'
    next(g)
    # default to 15-inch rows, with square subplots
    if figsize is None:
        if num_columns == 1:
            figsize = (5, 5)
        else:
            num_rows = int(np.ceil(n_plots / num_columns))
            figsize = (6.4 * num_columns, 4.8 * num_rows)

    if num_columns > 1:
        fig = plt.figure(figsize=figsize)
        num_rows = int(np.ceil(n_plots / num_columns))
    plot = 0
    try:
        while True:
            # call plt.figure once per row
            if num_columns == 1:
                fig = plt.figure(figsize=figsize)
                ax = plt.subplot(1, 1, 1)
                if save is not None:
                    pdf.savefig(fig, bbox_inches='tight')
            if num_columns > 1:
                ax = plt.subplot(num_rows, num_columns, plot + 1)
                ax.autoscale(enable=True)
                plot = plot + 1
            next(g)
    except StopIteration:
        if num_columns == 1:
            if save is not None:
                pdf.savefig(fig, bbox_inches='tight')
        pass
    if num_columns > 1:
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if save is not None:
            pdf.savefig(fig, bbox_inches='tight')
    if save is not None:
        pdf.close()
    if not plot:
        plt.close()


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(
        buf,
        bbox_inches='tight',
        format='png',
        dpi=500,
        transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    return img


def collapse_duplicates(df):
    """
    Collapse duplicates from fragments df
    """
    a = df.values
    sidx = np.lexsort(a[:, :4].T)
    b = a[sidx, :4]
    m = np.concatenate(([True], (b[1:] != b[:-1]).any(1), [True]))
    out_ar = np.column_stack((b[m[:-1], :4], np.diff(np.flatnonzero(m) + 1)))
    return pd.DataFrame(out_ar, columns=['Chromosome', 'Start', 'End', 'Name', 'Score'])


def get_tss_matrix(fragments, flank_window, tss_space_annotation):
    """
    Get TSS matrix
    """
    overlap_with_TSS = fragments.join(tss_space_annotation, nb_cpu=1).df
    if len(overlap_with_TSS) == 0:
        return

    overlap_with_TSS['Strand'] = overlap_with_TSS['Strand'].astype(np.int32)
    overlap_with_TSS['start_pos'] = -(np.int32(overlap_with_TSS['Start_b'].values) + np.int32(flank_window) -
                                      np.int32(overlap_with_TSS['Start'].values)) * np.int32(
        overlap_with_TSS['Strand'].values)
    overlap_with_TSS['end_pos'] = -(np.int32(overlap_with_TSS['Start_b'].values) + np.int32(flank_window) -
                                    np.int32(overlap_with_TSS['End'].values)) * np.int32(
        overlap_with_TSS['Strand'].values)
    # We split them to also keep the start position of reads whose start is
    # in the space and their end not and viceversa
    overlap_with_TSS_start = overlap_with_TSS[(overlap_with_TSS['start_pos'].values <= flank_window) & (
            overlap_with_TSS['start_pos'].values >= -flank_window)]
    overlap_with_TSS_end = overlap_with_TSS[(overlap_with_TSS['end_pos'].values <= flank_window) & (
            overlap_with_TSS['end_pos'].values >= -flank_window)]
    overlap_with_TSS_start['rel_start_pos'] = overlap_with_TSS_start['start_pos'].values + flank_window
    overlap_with_TSS_end['rel_end_pos'] = overlap_with_TSS_end['end_pos'].values + flank_window
    cut_sites_TSS = pd.concat([overlap_with_TSS_start[['Name',
                                                       'rel_start_pos']].rename(columns={'Name': 'Barcode',
                                                                                         'rel_start_pos': 'Position'}),
                               overlap_with_TSS_end[['Name',
                                                     'rel_end_pos']].rename(columns={'Name': 'Barcode',
                                                                                     'rel_end_pos': 'Position'})],
                              axis=0)

    cut_sites_TSS['Barcode'] = cut_sites_TSS["Barcode"].astype("category")
    cut_sites_TSS['Position'] = cut_sites_TSS["Position"].astype("category")
    TSS_matrix = cut_sites_TSS.groupby(
        ["Position", "Barcode"], observed=True, sort=False).size().unstack(level="Position", fill_value=0).astype(
        np.int32)
    del cut_sites_TSS
    gc.collect()

    return TSS_matrix


def read_fragments_from_file(fragments_bed_filename, use_polars: bool = False) -> pr.PyRanges:
    """
    Read fragments BED file to PyRanges object.

    Parameters
    ----------
    fragments_bed_filename: Fragments BED filename.
    use_polars: Use polars instead of pandas for reading the fragments BED file.

    Returns
    -------
    PyRanges object of fragments.
    """

    bed_column_names = (
        "Chromosome", "Start", "End", "Name", "Score", "Strand", "ThickStart", "ThickEnd", "ItemRGB", "BlockCount",
        "BlockSizes", "BlockStarts"
    )

    # Set the correct open function depending if the fragments BED file is gzip compressed or not.
    open_fn = gzip.open if fragments_bed_filename.endswith('.gz') else open

    skip_rows = 0
    nbr_columns = 0
    with open_fn(fragments_bed_filename, 'rt') as fragments_bed_fh:
        for line in fragments_bed_fh:
            # Remove newlines and spaces.
            line = line.strip()

            if not line or line.startswith('#'):
                # Count number of empty lines and lines which start with a comment before the actual data.
                skip_rows += 1
            else:
                # Get number of columns from the first real BED entry.
                nbr_columns = len(line.split('\t'))

                # Stop reading the BED file.
                break

    if nbr_columns < 4:
        raise ValueError(
            f'Fragments BED file needs to have at least 4 columns. "{fragments_bed_filename}" contains only '
            f'{nbr_columns} columns.'
        )

    if use_polars:
        import polars as pl

        # Read fragments BED file with polars.
        df = pl.read_csv(
            fragments_bed_filename,
            has_headers=False,
            skip_rows=skip_rows,
            sep='\t',
            use_pyarrow=True,
            new_columns=bed_column_names[:nbr_columns]
        ).with_columns([
            pl.col('Chromosome').cast(pl.Utf8), pl.col('Start').cast(pl.Int32), pl.col('End').cast(pl.Int32),
            pl.col('Name').cast(pl.Utf8)
        ]).to_pandas()

        # Convert "Name" column to pd.Categorical as groupby operations will be done on it later.
        df["Name"] = df["Name"].astype('category')
    else:
        # Read fragments BED file with pandas.
        df = pd.read_table(
            fragments_bed_filename,
            sep='\t',
            skiprows=skip_rows,
            header=None,
            names=bed_column_names[:nbr_columns],
            doublequote=False,
            engine='c',
            dtype={"Chromosome": str, "Start'": np.int32, "End": np.int32, "Name": "category", "Strand": str}
        )

    # Convert pandas dataframe to PyRanges dataframe.
    # This will convert "Chromosome" and "Strand" columns to pd.Categorical.
    return pr.PyRanges(df)
    
def coord_to_region_names(coord):
    """
    PyRanges to region names
    """
    if isinstance(coord, pr.PyRanges):
        coord = coord.as_df()
        return list(coord['Chromosome'].astype(str) + ':' + coord['Start'].astype(str) + '-' + coord['End'].astype(str))
