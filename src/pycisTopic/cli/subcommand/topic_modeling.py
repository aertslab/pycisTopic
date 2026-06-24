from __future__ import annotations

import logging
import os
import sys
from argparse import ArgumentTypeError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction


def run_topic_modeling_with_mallet(args):
    from pycisTopic.topic_modeling.mallet_models import LDAMallet

    check_java_exists()

    mallet_corpus_filename = args.mallet_corpus_filename
    output_prefix = args.output_prefix
    n_topics_list = [args.topics] if isinstance(args.topics, int) else args.topics
    alpha = args.alpha
    alpha_by_topic = args.alpha_by_topic
    eta = args.eta
    eta_by_topic = args.eta_by_topic
    n_iter = args.iterations
    optimize_interval = args.optimize_interval
    optimize_burn_in = args.optimize_burn_in
    output_model_interval = args.output_model_interval
    n_threads = args.parallel
    random_seed = args.seed
    memory_in_gb = args.memory_in_gb
    mallet_path = args.mallet_path

    if args.verbose:
        level = logging.INFO
        log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level=level, format=log_format, handlers=handlers)

    print("Run topic modeling with Mallet with the following settings:")
    print(f"  - Mallet corpus filename:                     {mallet_corpus_filename}")
    print(f"  - Output prefix:                              {output_prefix}")
    print(f"  - Number of topics to run topic modeling for: {n_topics_list}")
    print(f"  - Alpha:                                      {alpha}")
    print(f"  - Divide alpha by the number of topics:       {alpha_by_topic}")
    print(f"  - Eta:                                        {eta}")
    print(f"  - Divide eta by the number of topics:         {eta_by_topic}")
    print(f"  - Number of iterations of Gibbs sampling:     {n_iter}")
    print(f"  - Optimize interval for hyperparameters:      {optimize_interval}")
    print(f"  - Number of burn-in iterations:               {optimize_burn_in}")
    print(f"  - Save model every N iterations (0=end only): {output_model_interval}")
    print(f"  - Number threads Mallet is allowed to use:    {n_threads}")
    print(f"  - Seed:                                       {random_seed}")
    print(f"  - Amount of memory Mallet is allowed to use:  {memory_in_gb}G")
    print(f"  - Mallet binary:                              {mallet_path}")

    for n_topics in n_topics_list:
        # Run models
        print(f"\nRunning Mallet topic modeling for {n_topics} topics.")
        print(f"----------------------------------{'-' * len(str(n_topics))}--------")

        LDAMallet.run_mallet_topic_modeling(
            mallet_corpus_filename=mallet_corpus_filename,
            output_prefix=output_prefix,
            n_topics=n_topics,
            alpha=alpha,
            alpha_by_topic=alpha_by_topic,
            eta=eta,
            eta_by_topic=eta_by_topic,
            n_threads=n_threads,
            iterations=n_iter,
            optimize_interval=optimize_interval,
            optimize_burn_in=optimize_burn_in,
            output_model_interval=output_model_interval,
            topic_threshold=0.0,
            random_seed=random_seed,
            memory_in_gb=memory_in_gb,
            mallet_path=mallet_path,
        )

        print(
            f'\nWriting Mallet topic modeling output files to "{output_prefix}.{n_topics}_topics.*"...'
        )


def run_convert_binary_matrix_to_mallet_corpus_file_with_mallet(args):
    import scipy

    from pycisTopic.topic_modeling.mallet_models import LDAMallet

    check_java_exists()

    binary_accessibility_matrix_filename = args.binary_accessibility_matrix_filename
    mallet_corpus_filename = args.mallet_corpus_filename
    mallet_path = args.mallet_path
    memory_in_gb = args.memory_in_gb

    if args.verbose:
        level = logging.INFO
        log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level=level, format=log_format, handlers=handlers)

    print(
        f'Read binary accessibility matrix from "{binary_accessibility_matrix_filename}" Matrix Market file.'
    )
    binary_accessibility_matrix = scipy.io.mmread(binary_accessibility_matrix_filename)

    print(
        f'Convert binary accessibility matrix to Mallet serialized corpus file "{mallet_corpus_filename}".'
    )
    LDAMallet.convert_binary_matrix_to_mallet_corpus_file_with_mallet(
        binary_accessibility_matrix=binary_accessibility_matrix,
        mallet_corpus_filename=mallet_corpus_filename,
        memory_in_gb=memory_in_gb,
        mallet_path=mallet_path,
    )


def run_convert_binary_matrix_to_mallet_corpus_file_with_malletjson(args):
    import scipy

    from pycisTopic.topic_modeling.mallet_models import LDAMallet

    binary_accessibility_matrix_filename = args.binary_accessibility_matrix_filename
    mallet_corpus_filename = args.mallet_corpus_filename
    memory_in_gb = args.memory_in_gb
    malletjson_jar = args.malletjson_jar

    if args.verbose:
        level = logging.INFO
        log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level=level, format=log_format, handlers=handlers)

    print(
        f'Read binary accessibility matrix from "{binary_accessibility_matrix_filename}" Matrix Market file.'
    )
    binary_accessibility_matrix = scipy.io.mmread(binary_accessibility_matrix_filename)

    print(
        f'Convert binary accessibility matrix to Mallet serialized corpus file "{mallet_corpus_filename}" with MalletJSON.'
    )
    LDAMallet.convert_binary_matrix_to_mallet_corpus_file_with_malletjson(
        binary_accessibility_matrix=binary_accessibility_matrix,
        mallet_corpus_filename=mallet_corpus_filename,
        memory_in_gb=memory_in_gb,
        malletjson_jar=malletjson_jar,
    )


def run_mallet_calculate_model_evaluation_stats(args):
    import scipy

    from pycisTopic.topic_modeling.stats import calculate_model_evaluation_stats

    binary_accessibility_matrix_filename = args.binary_accessibility_matrix_filename
    output_prefix = args.output_prefix
    n_topics_list = [args.topics] if isinstance(args.topics, int) else args.topics

    print(
        f'Read binary accessibility matrix from "{binary_accessibility_matrix_filename}" Matrix Market file.'
    )
    binary_accessibility_matrix = scipy.io.mmread(binary_accessibility_matrix_filename)

    for n_topics in n_topics_list:
        print(
            f'Calculate model evaluation statistics for {n_topics} topics from "{output_prefix}.{n_topics}_topics.*"...'
        )
        calculate_model_evaluation_stats(
            binary_accessibility_matrix=binary_accessibility_matrix,
            output_prefix=output_prefix,
            n_topics=n_topics,
            top_topics_coh=5,
        )


def run_mallet_plot_model_evaluation_stats(args):
    from pycisTopic.topic_modeling.plot_stats import plot_stats

    n_topics_list = [args.n_topics] if isinstance(args.n_topics, int) else args.n_topics
    plot_stats(
        output_prefix=args.output_prefix,
        n_topics=n_topics_list,
        plot_file_format=args.plot_file_format,
    )


def binarize_cell_or_region_topic(args):
    """Binarize cell-topics or region-topics."""
    target = args.target
    method = args.method
    ntop = args.ntop
    smooth_topics = args.smooth_topics
    nbins = args.nbins
    cell_barcodes_filename = args.cell_barcodes_filename
    region_ids_filename = args.region_ids_filename
    output_prefix = args.output_prefix
    n_topics = args.n_topics
    out_dir = args.out_dir

    # input validation
    if target == "cell" and cell_barcodes_filename is None:
        raise ValueError(
            "`cell_barcodes_filename` using `--cb` should be provided when target is `cell`"
        )
    if target == "region" and region_ids_filename is None:
        raise ValueError(
            "`region_ids_filename` using `--regions` should be provided when target is `region`"
        )

    import os

    if not os.path.exists(out_dir):
        print(f"Making directory: {out_dir}")
        os.makedirs(out_dir)

    from pycisTopic.fragments import read_barcodes_file_to_polars_series
    from pycisTopic.topic_binarization import binarize_topics
    from pycisTopic.topic_modeling.mallet_models import LDAMallet, LDAMalletFilenames

    lda_mallet_filenames = LDAMalletFilenames(
        output_prefix=output_prefix, n_topics=n_topics
    )

    if target == "cell":
        print(f'Read cell barcodes filename "{cell_barcodes_filename}".')
        cell_or_region_names = read_barcodes_file_to_polars_series(
            barcodes_tsv_filename=cell_barcodes_filename,
            sample_id=None,
            cb_end_to_remove=None,
            cb_sample_separator=None,
        ).to_list()
        print(
            f'Read cell-topic probabilities filename "{lda_mallet_filenames.cell_topic_probabilities_parquet_filename}".'
        )
        cell_or_region_topic_prob = LDAMallet.read_cell_topic_probabilities_parquet_file(
            mallet_cell_topic_probabilities_parquet_filename=lda_mallet_filenames.cell_topic_probabilities_parquet_filename
        )

    if target == "region":
        print(f'Read region IDs filename "{region_ids_filename}".')
        cell_or_region_names = read_barcodes_file_to_polars_series(
            barcodes_tsv_filename=region_ids_filename,
            sample_id=None,
            cb_end_to_remove=None,
            cb_sample_separator=None,
        ).to_list()
        print(
            f'Read region-topic probabilities filename "{lda_mallet_filenames.region_topic_counts_parquet_filename}".'
        )
        cell_or_region_topic_prob = LDAMallet.read_region_topic_counts_parquet_file_to_region_topic_probabilities(
            mallet_region_topic_counts_parquet_filename=lda_mallet_filenames.region_topic_counts_parquet_filename
        ).T

    print("Binarizing topics ...")
    cell_or_region_names_per_topic, scores_per_topic, thresholds = binarize_topics(
        cell_or_region_topic_prob=cell_or_region_topic_prob,
        cell_or_region_names=cell_or_region_names,
        method=method,
        smooth_topics=smooth_topics,
        ntop=ntop,
        nbins=nbins,
    )

    print(f'Saving results to "{out_dir}".')

    with open(os.path.join(out_dir, f"{target}_thresholds.tsv"), "w") as f:
        for topic, thr in enumerate(thresholds):
            f.write(f"{topic + 1}\t{thr}\n")

    if target == "cell":
        for topic, (cells, scores) in enumerate(
            zip(cell_or_region_names_per_topic, scores_per_topic)
        ):
            with open(
                os.path.join(out_dir, f"{target}_Topic_{topic + 1}_binarized.txt"), "w"
            ) as f:
                for cell, score in zip(cells, scores):
                    f.write(f"{cell}\t{score}\n")

    elif target == "region":
        for topic, (regions, scores) in enumerate(
            zip(cell_or_region_names_per_topic, scores_per_topic)
        ):
            with open(
                os.path.join(out_dir, f"{target}_Topic_{topic + 1}_binarized.bed"), "w"
            ) as f:
                for region, score in zip(regions, scores):
                    chrom, start, end = region.replace(":", "-").split("-")
                    f.write(f"{chrom}\t{start}\t{end}\tTopic_{topic + 1}\t{score}\n")


def run_create_anndata_from_mallet(args):
    from pycisTopic.topic_modeling.create_anndata import create_anndata_from_mallet

    cell_barcodes: list[str] = []
    with open(args.cell_barcodes) as f:
        for line in f:
            cell_barcodes.append(line.strip())

    region_ids: list[str] = []
    with open(args.region_ids) as f:
        for line in f:
            region_ids.append(line.strip())

    create_anndata_from_mallet(
        output_prefix=args.output_prefix,
        n_topics=args.n_topics,
        cell_barcodes=cell_barcodes,
        region_ids=region_ids,
    )


def str_to_bool(v: str) -> bool:
    """
    Convert string representation of a boolean value to a boolean.

    Parameters
    ----------
    v
        String representation of a boolean value.
        After conversion to lowercase, the following string values can be converted:
          - "yes", "true", "t", "y", "1" -> True
          - "no", "false", "f", "n", "0" -> False

    Returns
    -------
    True or False

    """
    if isinstance(v, str):
        v = v.lower()
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
    raise ArgumentTypeError("Boolean value expected.")


def check_java_exists():
    import subprocess

    print("Checking whether Java exists.")
    subprocess.run(["java", "--version"], shell=False, stdout=subprocess.DEVNULL)


def add_parser_topic_modeling(subparsers: _SubParsersAction[ArgumentParser]):
    """Creates an ArgumentParser to read the options for this script."""
    parser_topic_modeling = subparsers.add_parser(
        "topic_modeling",
        help="Run LDA topic modeling.",
    )

    subparser_topic_modeling = parser_topic_modeling.add_subparsers(
        title="Topic modeling",
        dest="topic_modeling",
        help="List of topic modeling subcommands.",
    )
    subparser_topic_modeling.required = True

    parser_topic_modeling_mallet = subparser_topic_modeling.add_parser(
        "mallet", help="Run LDA topic modeling with Mallet."
    )

    subparser_topic_modeling_mallet = parser_topic_modeling_mallet.add_subparsers(
        title="Topic modeling with Mallet",
        dest="mallet",
        help="List of Mallet topic modeling subcommands.",
    )
    subparser_topic_modeling_mallet.required = True

    parser_topic_modeling_mallet_create_corpus_with_mallet = subparser_topic_modeling_mallet.add_parser(
        "create_corpus_with_mallet",
        help="Convert binary accessibility matrix to Mallet serialized corpus file using Mallet `import-file`.",
        description="Convert binary accessibility matrix to Mallet serialized corpus file using Mallet `import-file`. Slower than `create_corpus_with_malletjson`.",
    )
    parser_topic_modeling_mallet_create_corpus_with_mallet.set_defaults(
        func=run_convert_binary_matrix_to_mallet_corpus_file_with_mallet
    )

    parser_topic_modeling_mallet_create_corpus_with_mallet.add_argument(
        "-i",
        "--input",
        dest="binary_accessibility_matrix_filename",
        action="store",
        type=str,
        required=True,
        help="Binary accessibility matrix (region IDs vs cell barcodes) in Matrix Market format.",
    )
    parser_topic_modeling_mallet_create_corpus_with_mallet.add_argument(
        "-o",
        "--output",
        dest="mallet_corpus_filename",
        action="store",
        type=str,
        required=True,
        help="Mallet serialized corpus filename.",
    )
    parser_topic_modeling_mallet_create_corpus_with_mallet.add_argument(
        "-m",
        "--memory",
        dest="memory_in_gb",
        type=int,
        required=False,
        default=10,
        help='Amount of memory (in GB) Mallet is allowed to use. Default: "10".',
    )
    parser_topic_modeling_mallet_create_corpus_with_mallet.add_argument(
        "-b",
        "--mallet_path",
        dest="mallet_path",
        type=str,
        required=False,
        default="mallet",
        help='Path to Mallet binary (e.g. "/xxx/Mallet/bin/mallet"). Default: "mallet".',
    )
    parser_topic_modeling_mallet_create_corpus_with_mallet.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help="Enable verbose mode.",
    )

    parser_topic_modeling_mallet_create_corpus_with_malletjson = subparser_topic_modeling_mallet.add_parser(
        "create_corpus_with_malletjson",
        help="Convert binary accessibility matrix to Mallet serialized corpus file using MalletJSON.",
        description="Convert binary accessibility matrix to Mallet serialized corpus file using MalletJSON "
        "(https://github.com/mimno/MalletJSON/). Faster than `create_corpus_with_mallet`.",
    )
    parser_topic_modeling_mallet_create_corpus_with_malletjson.set_defaults(
        func=run_convert_binary_matrix_to_mallet_corpus_file_with_malletjson
    )

    parser_topic_modeling_mallet_create_corpus_with_malletjson.add_argument(
        "-i",
        "--input",
        dest="binary_accessibility_matrix_filename",
        action="store",
        type=str,
        required=True,
        help="Binary accessibility matrix (region IDs vs cell barcodes) in Matrix Market format.",
    )
    parser_topic_modeling_mallet_create_corpus_with_malletjson.add_argument(
        "-o",
        "--output",
        dest="mallet_corpus_filename",
        action="store",
        type=str,
        required=True,
        help="Mallet serialized corpus filename.",
    )
    parser_topic_modeling_mallet_create_corpus_with_malletjson.add_argument(
        "-m",
        "--memory",
        dest="memory_in_gb",
        type=int,
        required=False,
        default=10,
        help='Amount of memory (in GB) MalletJSON is allowed to use. Default: "10".',
    )
    parser_topic_modeling_mallet_create_corpus_with_malletjson.add_argument(
        "-j",
        "--malletjson_jar",
        dest="malletjson_jar",
        type=str,
        required=False,
        default="mallet-json-1.0.0-fat-21.jar",
        help="Path to the MalletJSON fat JAR (https://github.com/mimno/MalletJSON/releases/latest). "
        'Default: "mallet-json-1.0.0-fat-21.jar".',
    )
    parser_topic_modeling_mallet_create_corpus_with_malletjson.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help="Enable verbose mode.",
    )

    parser_topic_modeling_mallet_run = subparser_topic_modeling_mallet.add_parser(
        "run",
        help="Run LDA topic modeling with Mallet `train-topics` command.",
        description="Run LDA topic modeling with Mallet `train-topics` command.",
    )
    parser_topic_modeling_mallet_run.set_defaults(func=run_topic_modeling_with_mallet)

    parser_topic_modeling_mallet_run.add_argument(
        "-i",
        "--input",
        dest="mallet_corpus_filename",
        action="store",
        type=str,
        required=True,
        help="Mallet corpus filename.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        action="store",
        type=str,
        required=True,
        help="Topic model output prefix.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "-t",
        "--topics",
        dest="topics",
        type=int,
        required=True,
        nargs="+",
        help="Number(s) of topics to create during topic modeling.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "-p",
        "--parallel",
        dest="parallel",
        type=int,
        required=True,
        help="Number of threads Mallet is allowed to use.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "-n",
        "--iterations",
        dest="iterations",
        type=int,
        required=False,
        default=150,
        help="Number of iterations of Gibbs sampling. Default: 150.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "--optimize-interval",
        dest="optimize_interval",
        type=int,
        required=False,
        default=0,
        help="Optimize hyperparameters every `optimize_interval` iterations. "
        "Only takes effect after running `optimize_burn_in` iterations. "
        "Disable optimizing hyperparameters by setting this option to 0. "
        "Default: 0.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "--optimize-burn-in",
        dest="optimize_burn_in",
        type=int,
        required=False,
        default=50,
        help="The number of iterations before starting hyperparameter optimization. "
        "Default: 50.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "--output-model-interval",
        dest="output_model_interval",
        type=int,
        required=False,
        default=0,
        help="Save the Mallet model every `output_model_interval` iterations of Gibbs "
        "sampling to `<output_prefix>.<n_topics>_topics.model.<iteration>`. "
        "When set to 0 (default), the model is saved only once at the very end of "
        "training as `<output_prefix>.<n_topics>_topics.model.<iterations>`. "
        "These Mallet output model files are used to resume an interrupted run "
        "from the saved state in the Mallet output model file with the highest "
        "iteration number. Training resumes from that checkpoint and the "
        "`iterations` / `optimize_burn_in` values that are actually passed to Mallet "
        "are adjusted (as Mallet sees them as the number of iterations to run "
        "from the loaded checkpoint, not as the total number of iterations to run). "
        "Default: 0.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "-a",
        "--alpha",
        dest="alpha",
        type=int,
        required=False,
        default=50,
        help="Alpha value. Default: 50.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "-A",
        "--alpha_by_topic",
        dest="alpha_by_topic",
        type=str_to_bool,
        choices=(True, False),
        required=False,
        default=True,
        help="Whether the alpha value should by divided by the number of topics. Default: True.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "-e",
        "--eta",
        dest="eta",
        type=float,
        required=False,
        default=0.1,
        help="Eta value. Default: 0.1.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "-E",
        "--eta_by_topic",
        dest="eta_by_topic",
        type=str_to_bool,
        choices=(True, False),
        required=False,
        default=False,
        help="Whether the eta value should by divided by the number of topics. Default: False.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "-s",
        "--seed",
        dest="seed",
        type=int,
        required=False,
        default=555,
        help="Seed for ensuring reproducibility. "
        "To get reproducible output, Mallet also has to be run with the same number of threads. "
        "Default: 555.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "-m",
        "--memory",
        dest="memory_in_gb",
        type=int,
        required=False,
        default=100,
        help='Amount of memory (in GB) Mallet is allowed to use. Default: "100".',
    )
    parser_topic_modeling_mallet_run.add_argument(
        "-b",
        "--mallet_path",
        dest="mallet_path",
        type=str,
        required=False,
        default="mallet",
        help='Path to Mallet binary (e.g. "/xxx/Mallet/bin/mallet"). Default: "mallet".',
    )
    parser_topic_modeling_mallet_run.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help="Enable verbose mode.",
    )

    parser_topic_modeling_mallet_calculate_stats = (
        subparser_topic_modeling_mallet.add_parser(
            "stats",
            help="Calculate model evaluation statistics.",
            description="Calculate model evaluation statistics.",
        )
    )
    parser_topic_modeling_mallet_calculate_stats.set_defaults(
        func=run_mallet_calculate_model_evaluation_stats
    )

    parser_topic_modeling_mallet_calculate_stats.add_argument(
        "-i",
        "--input",
        dest="binary_accessibility_matrix_filename",
        action="store",
        type=str,
        required=True,
        help="Binary accessibility matrix (region IDs vs cell barcodes) in Matrix Market format.",
    )
    parser_topic_modeling_mallet_calculate_stats.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        action="store",
        type=str,
        required=True,
        help="Topic model output prefix.",
    )
    parser_topic_modeling_mallet_calculate_stats.add_argument(
        "-t",
        "--topics",
        dest="topics",
        type=int,
        required=True,
        nargs="+",
        help="Topic number(s) to create the model evaluation statistics for.",
    )
    parser_topic_modeling_mallet_calculate_stats.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help="Enable verbose mode.",
    )

    parser_topic_modeling_mallet_plot_stats = (
        subparser_topic_modeling_mallet.add_parser(
            "plot_stats",
            help="Plot evaluation statistics.",
            description="Plot evaluation statistics.",
        )
    )
    parser_topic_modeling_mallet_plot_stats.set_defaults(
        func=run_mallet_plot_model_evaluation_stats
    )
    parser_topic_modeling_mallet_plot_stats.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        action="store",
        type=str,
        required=True,
        help="Topic model output prefix.",
    )
    parser_topic_modeling_mallet_plot_stats.add_argument(
        "-t",
        "--topics",
        dest="n_topics",
        type=int,
        required=True,
        nargs="+",
        help="Topic number(s) to create the model evaluation statistics for.",
    )
    parser_topic_modeling_mallet_plot_stats.add_argument(
        "-q",
        "--format",
        dest="plot_file_format",
        action="store",
        type=str,
        required=False,
        default="png",
        help="File format of the resulting plots. Default: png.",
    )

    parser_topic_modeling_mallet_binarize = subparser_topic_modeling_mallet.add_parser(
        "binarize",
        help="Binarize cell- or region-topic probabilities.",
        description="Binarize cell- or region-topic probabilities.",
    )
    parser_topic_modeling_mallet_binarize.set_defaults(
        func=binarize_cell_or_region_topic
    )
    parser_topic_modeling_mallet_binarize.add_argument(
        "-a",
        "--target",
        dest="target",
        action="store",
        type=str,
        choices=["region", "cell"],
        required=True,
        help='Choose between "region" or "cell" topic binarization.',
    )
    parser_topic_modeling_mallet_binarize.add_argument(
        "-m",
        "--method",
        dest="method",
        action="store",
        type=str,
        choices=("ntop", "otsu", "aucell", "li", "yen"),
        required=True,
        help='Binarization method. Choose between "ntop", "otsu", "aucell", "li" or "yen" for cell-or region-topic binarization.',
    )
    parser_topic_modeling_mallet_binarize.add_argument(
        "-n",
        "--ntop",
        dest="ntop",
        action="store",
        type=int,
        required=False,
        help="Number of top regions to select. Can only be used when `--method` is set to `ntop`.",
    )
    parser_topic_modeling_mallet_binarize.add_argument(
        "-s",
        "--smooth",
        dest="smooth_topics",
        action="store",
        type=str_to_bool,
        choices=(True, False),
        required=False,
        default=True,
        help="Whether to smooth the cell- or region-topic probabilities.",
    )
    parser_topic_modeling_mallet_binarize.add_argument(
        "-b",
        "--nbins",
        dest="nbins",
        action="store",
        type=int,
        required=False,
        default=100,
        help="Number of bins to use in the histogram used for `otsu`, `yen` and `li` thresholding.",
    )
    parser_topic_modeling_mallet_binarize.add_argument(
        "-c",
        "--cb",
        dest="cell_barcodes_filename",
        action="store",
        type=str,
        required=False,
        help="Filename with cell barcodes.",
    )
    parser_topic_modeling_mallet_binarize.add_argument(
        "-r",
        "--regions",
        dest="region_ids_filename",
        action="store",
        type=str,
        required=False,
        help="Filename with region IDs.",
    )
    parser_topic_modeling_mallet_binarize.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        action="store",
        type=str,
        required=True,
        help="Topic model output prefix.",
    )
    parser_topic_modeling_mallet_binarize.add_argument(
        "-t",
        "--n_topics",
        dest="n_topics",
        type=int,
        required=True,
        help="Model with `topic` number of topics to binarize.",
    )
    parser_topic_modeling_mallet_binarize.add_argument(
        "-p",
        "--output_dir",
        dest="out_dir",
        action="store",
        type=str,
        required=True,
        help="Directory to store results.",
    )

    parser_topic_modeling_mallet_anndata = subparser_topic_modeling_mallet.add_parser(
        "create_anndata",
        help="Generate AnnData h5ad file from Mallet result.",
        description="Generate AnnData h5ad file from Mallet result.",
    )
    parser_topic_modeling_mallet_anndata.set_defaults(
        func=run_create_anndata_from_mallet
    )

    parser_topic_modeling_mallet_anndata.add_argument(
        "-c",
        "--cb",
        dest="cell_barcodes",
        action="store",
        type=str,
        required=False,
        help="Filename with cell barcodes.",
    )
    parser_topic_modeling_mallet_anndata.add_argument(
        "-r",
        "--regions",
        dest="region_ids",
        action="store",
        type=str,
        required=False,
        help="Filename with region IDs.",
    )
    parser_topic_modeling_mallet_anndata.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        action="store",
        type=str,
        required=True,
        help="Topic model output prefix.",
    )
    parser_topic_modeling_mallet_anndata.add_argument(
        "-t",
        "--n_topics",
        dest="n_topics",
        type=int,
        required=True,
        help="Model with `topic` number of topics to generate AnnData from.",
    )
