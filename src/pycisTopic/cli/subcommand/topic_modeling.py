from __future__ import annotations

import gzip
import logging
import os
import sys
from argparse import ArgumentTypeError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction


def run_create_corpus(args) -> None:
    import scipy

    from pycisTopic.topic_modeling.mallet_models import LDAMallet

    check_java_exists()
    _configure_logging(verbose=args.verbose)

    print(
        f'Read binary accessibility matrix from "{args.binary_accessibility_matrix_filename}" Matrix Market file.'
    )
    binary_accessibility_matrix = scipy.io.mmread(
        args.binary_accessibility_matrix_filename
    )

    os.environ["MALLET_MEMORY"] = f"{args.memory_in_gb}G"
    print(
        f'Convert binary accessibility matrix to Mallet serialized corpus file "{args.mallet_corpus_filename}".'
    )
    LDAMallet.convert_binary_matrix_to_mallet_corpus_file(
        binary_accessibility_matrix=binary_accessibility_matrix,
        mallet_corpus_filename=args.mallet_corpus_filename,
        mallet_path=args.mallet_path,
    )


def run_topic_modeling(args) -> None:
    n_topics_list = args.topics if isinstance(args.topics, list) else [args.topics]
    _configure_logging(verbose=args.verbose)

    if args.backend == "mallet":
        from pycisTopic.topic_modeling.mallet_models import LDAMallet

        check_java_exists()
        os.environ["MALLET_MEMORY"] = f"{args.memory_in_gb}G"

        print("Run topic modeling with Mallet with the following settings:")
        print(f"  - Mallet corpus filename: {args.input_filename}")
        print(f"  - Output prefix:          {args.output_prefix}")
        print(f"  - Topics:                 {n_topics_list}")
        print(f"  - Threads:                {args.threads}")
        print(f"  - Alpha:                  {args.alpha}")
        print(f"  - Alpha by topic:         {args.alpha_by_topic}")
        print(f"  - Eta:                    {args.eta}")
        print(f"  - Eta by topic:           {args.eta_by_topic}")
        print(f"  - Iterations:             {args.iterations}")
        print(f"  - Optimize interval:      {args.optimize_interval}")
        print(f"  - Optimize burn-in:       {args.optimize_burn_in}")
        print(f"  - Seed:                   {args.seed}")
        print(f"  - Mallet memory:          {args.memory_in_gb}G")
        print(f"  - Mallet binary:          {args.mallet_path}")

        for n_topics in n_topics_list:
            print(f"\nRunning Mallet topic modeling for {n_topics} topics.")
            LDAMallet.run_topic_modeling(
                mallet_corpus_filename=args.input_filename,
                output_prefix=args.output_prefix,
                n_topics=n_topics,
                alpha=args.alpha,
                alpha_by_topic=args.alpha_by_topic,
                eta=args.eta,
                eta_by_topic=args.eta_by_topic,
                n_threads=args.threads,
                iterations=args.iterations,
                optimize_interval=args.optimize_interval,
                optimize_burn_in=args.optimize_burn_in,
                topic_threshold=0.0,
                random_seed=args.seed,
                mallet_path=args.mallet_path,
            )
        return

    import scipy

    from pycisTopic.topic_modeling.tomotopy_models import LDATomotopy

    if args.cell_barcodes_filename is None or args.region_ids_filename is None:
        raise ValueError(
            "`--cb` and `--regions` should be provided when backend is `tomotopy`."
        )

    print(
        f'Read binary accessibility matrix from "{args.input_filename}" Matrix Market file.'
    )
    binary_accessibility_matrix = scipy.io.mmread(args.input_filename)
    cell_names = _read_names_file(args.cell_barcodes_filename)
    region_names = _read_names_file(args.region_ids_filename)

    print("Run topic modeling with tomotopy with the following settings:")
    print(f"  - Binary accessibility matrix: {args.input_filename}")
    print(f"  - Output prefix:               {args.output_prefix}")
    print(f"  - Topics:                      {n_topics_list}")
    print(f"  - Threads:                     {args.threads}")
    print(f"  - Alpha:                       {args.alpha}")
    print(f"  - Alpha by topic:              {args.alpha_by_topic}")
    print(f"  - Eta:                         {args.eta}")
    print(f"  - Eta by topic:                {args.eta_by_topic}")
    print(f"  - Iterations:                  {args.iterations}")
    print(f"  - Optimize interval:           {args.optimize_interval}")
    print(f"  - Seed:                        {args.seed}")

    for n_topics in n_topics_list:
        print(f"\nRunning tomotopy topic modeling for {n_topics} topics.")
        LDATomotopy.run_topic_modeling(
            binary_accessibility_matrix=binary_accessibility_matrix,
            cell_names=cell_names,
            region_names=region_names,
            output_prefix=args.output_prefix,
            n_topics=n_topics,
            alpha=args.alpha,
            alpha_by_topic=args.alpha_by_topic,
            eta=args.eta,
            eta_by_topic=args.eta_by_topic,
            n_threads=args.threads,
            iterations=args.iterations,
            optimize_interval=args.optimize_interval,
            random_seed=args.seed,
        )


def run_calculate_model_evaluation_stats(args) -> None:
    import scipy

    from pycisTopic.topic_modeling.stats import calculate_model_evaluation_stats

    print(
        f'Read binary accessibility matrix from "{args.binary_accessibility_matrix_filename}" Matrix Market file.'
    )
    binary_accessibility_matrix = scipy.io.mmread(
        args.binary_accessibility_matrix_filename
    )

    n_topics_list = args.topics if isinstance(args.topics, list) else [args.topics]
    for n_topics in n_topics_list:
        print(
            f'Calculate model evaluation statistics for {n_topics} topics from "{args.output_prefix}.{n_topics}_topics.*"...'
        )
        calculate_model_evaluation_stats(
            binary_accessibility_matrix=binary_accessibility_matrix,
            output_prefix=args.output_prefix,
            n_topics=n_topics,
            top_topics_coh=5,
        )


def run_plot_model_evaluation_stats(args) -> None:
    from pycisTopic.topic_modeling.plot_stats import plot_stats

    n_topics_list = args.n_topics if isinstance(args.n_topics, list) else [args.n_topics]
    plot_stats(
        output_prefix=args.output_prefix,
        n_topics=n_topics_list,
        plot_file_format=args.plot_file_format,
    )


def binarize_cell_or_region_topic(args) -> None:
    """Binarize cell-topics or region-topics."""
    from pycisTopic.topic_binarization import binarize_topics
    from pycisTopic.topic_modeling.topic_models import TopicModelFilenames, load_topic_model_backend

    if args.target == "cell" and args.cell_barcodes_filename is None:
        raise ValueError(
            "`cell_barcodes_filename` using `--cb` should be provided when target is `cell`."
        )
    if args.target == "region" and args.region_ids_filename is None:
        raise ValueError(
            "`region_ids_filename` using `--regions` should be provided when target is `region`."
        )

    if not os.path.exists(args.out_dir):
        print(f"Making directory: {args.out_dir}")
        os.makedirs(args.out_dir)

    filenames = TopicModelFilenames(output_prefix=args.output_prefix, n_topics=args.n_topics)
    backend_cls = load_topic_model_backend(
        output_prefix=args.output_prefix,
        n_topics=args.n_topics,
    )

    if args.target == "cell":
        print(f'Read cell barcodes filename "{args.cell_barcodes_filename}".')
        cell_or_region_names = _read_names_file(args.cell_barcodes_filename)
        print(
            f'Read cell-topic probabilities filename "{filenames.cell_topic_probabilities_parquet_filename}".'
        )
        cell_or_region_topic_prob = backend_cls.read_cell_topic_probabilities_parquet_file(
            cell_topic_probabilities_parquet_filename=filenames.cell_topic_probabilities_parquet_filename
        )
    else:
        print(f'Read region IDs filename "{args.region_ids_filename}".')
        cell_or_region_names = _read_names_file(args.region_ids_filename)
        print(
            f'Read region-topic probabilities filename "{filenames.region_topic_counts_parquet_filename}".'
        )
        cell_or_region_topic_prob = (
            backend_cls.read_region_topic_counts_parquet_file_to_region_topic_probabilities(
                region_topic_counts_parquet_filename=filenames.region_topic_counts_parquet_filename
            ).T
        )

    print("Binarizing topics ...")
    cell_or_region_names_per_topic, scores_per_topic, thresholds = binarize_topics(
        cell_or_region_topic_prob=cell_or_region_topic_prob,
        cell_or_region_names=cell_or_region_names,
        method=args.method,
        smooth_topics=args.smooth_topics,
        ntop=args.ntop,
        nbins=args.nbins,
    )

    print(f'Saving results to "{args.out_dir}".')
    with open(os.path.join(args.out_dir, f"{args.target}_thresholds.tsv"), "w") as fh:
        for topic, threshold in enumerate(thresholds):
            fh.write(f"{topic + 1}\t{threshold}\n")

    if args.target == "cell":
        for topic, (cells, scores) in enumerate(
            zip(cell_or_region_names_per_topic, scores_per_topic)
        ):
            with open(
                os.path.join(args.out_dir, f"cell_Topic_{topic + 1}_binarized.txt"),
                "w",
            ) as fh:
                for cell, score in zip(cells, scores):
                    fh.write(f"{cell}\t{score}\n")
    else:
        for topic, (regions, scores) in enumerate(
            zip(cell_or_region_names_per_topic, scores_per_topic)
        ):
            with open(
                os.path.join(args.out_dir, f"region_Topic_{topic + 1}_binarized.bed"),
                "w",
            ) as fh:
                for region, score in zip(regions, scores):
                    chrom, start, end = region.replace(":", "-").split("-")
                    fh.write(f"{chrom}\t{start}\t{end}\tTopic_{topic + 1}\t{score}\n")


def run_create_anndata(args) -> None:
    from pycisTopic.topic_modeling.create_anndata import create_anndata_from_topic_model

    create_anndata_from_topic_model(
        output_prefix=args.output_prefix,
        n_topics=args.n_topics,
        cell_barcodes=_read_names_file(args.cell_barcodes),
        region_ids=_read_names_file(args.region_ids),
    )


def _configure_logging(verbose: bool) -> None:
    if not verbose:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
        force=True,
    )


def _read_names_file(filename: str) -> list[str]:
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(filename, "rt", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def str_to_bool(value: str) -> bool:
    """Convert a string representation of a boolean value to a boolean."""
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("yes", "true", "t", "y", "1"):
            return True
        if lowered in ("no", "false", "f", "n", "0"):
            return False
    raise ArgumentTypeError("Boolean value expected.")


def check_java_exists() -> None:
    import subprocess

    print("Checking whether Java exists.")
    subprocess.run(["java", "--version"], shell=False, stdout=subprocess.DEVNULL)


def add_parser_topic_modeling(subparsers: _SubParsersAction[ArgumentParser]) -> None:
    """Create the topic modeling CLI parser."""
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

    parser_create_corpus = subparser_topic_modeling.add_parser(
        "create_corpus",
        help="Convert a binary accessibility matrix to a Mallet serialized corpus file.",
    )
    parser_create_corpus.set_defaults(func=run_create_corpus)
    parser_create_corpus.add_argument(
        "-i",
        "--input",
        dest="binary_accessibility_matrix_filename",
        type=str,
        required=True,
        help="Binary accessibility matrix in Matrix Market format.",
    )
    parser_create_corpus.add_argument(
        "-o",
        "--output",
        dest="mallet_corpus_filename",
        type=str,
        required=True,
        help="Mallet serialized corpus filename.",
    )
    parser_create_corpus.add_argument(
        "-m",
        "--memory",
        dest="memory_in_gb",
        type=int,
        required=False,
        default=10,
        help='Amount of memory in GB that Mallet is allowed to use. Default: "10".',
    )
    parser_create_corpus.add_argument(
        "-b",
        "--mallet_path",
        dest="mallet_path",
        type=str,
        required=False,
        default="mallet",
        help='Path to the Mallet binary. Default: "mallet".',
    )
    parser_create_corpus.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help="Enable verbose mode.",
    )

    parser_run = subparser_topic_modeling.add_parser(
        "run",
        help="Run topic modeling with the selected backend.",
    )
    parser_run.set_defaults(func=run_topic_modeling)
    parser_run.add_argument(
        "--backend",
        dest="backend",
        type=str,
        choices=("mallet", "tomotopy"),
        required=True,
        help="Topic modeling backend to use.",
    )
    parser_run.add_argument(
        "-i",
        "--input",
        dest="input_filename",
        type=str,
        required=True,
        help="Input filename. Use a Mallet corpus for `mallet` or a Matrix Market matrix for `tomotopy`.",
    )
    parser_run.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        type=str,
        required=True,
        help="Topic model output prefix.",
    )
    parser_run.add_argument(
        "-t",
        "--topics",
        dest="topics",
        type=int,
        required=True,
        nargs="+",
        help="Number(s) of topics to create during topic modeling.",
    )
    parser_run.add_argument(
        "-p",
        "--threads",
        dest="threads",
        type=int,
        required=True,
        help="Number of threads the backend is allowed to use.",
    )
    parser_run.add_argument(
        "-n",
        "--iterations",
        dest="iterations",
        type=int,
        required=False,
        default=150,
        help="Number of iterations of Gibbs sampling. Default: 150.",
    )
    parser_run.add_argument(
        "--optimize-interval",
        dest="optimize_interval",
        type=int,
        required=False,
        default=0,
        help="Optimize hyperparameters every `optimize_interval` iterations. Default: 0.",
    )
    parser_run.add_argument(
        "--optimize-burn-in",
        dest="optimize_burn_in",
        type=int,
        required=False,
        default=50,
        help="Number of iterations before starting hyperparameter optimization. Default: 50.",
    )
    parser_run.add_argument(
        "-a",
        "--alpha",
        dest="alpha",
        type=float,
        required=False,
        default=50,
        help="Alpha value. Default: 50.",
    )
    parser_run.add_argument(
        "-A",
        "--alpha_by_topic",
        dest="alpha_by_topic",
        type=str_to_bool,
        choices=(True, False),
        required=False,
        default=True,
        help="Whether alpha should be divided by the number of topics. Default: True.",
    )
    parser_run.add_argument(
        "-e",
        "--eta",
        dest="eta",
        type=float,
        required=False,
        default=0.1,
        help="Eta value. Default: 0.1.",
    )
    parser_run.add_argument(
        "-E",
        "--eta_by_topic",
        dest="eta_by_topic",
        type=str_to_bool,
        choices=(True, False),
        required=False,
        default=False,
        help="Whether eta should be divided by the number of topics. Default: False.",
    )
    parser_run.add_argument(
        "-s",
        "--seed",
        dest="seed",
        type=int,
        required=False,
        default=555,
        help="Seed for ensuring reproducibility. Default: 555.",
    )
    parser_run.add_argument(
        "-m",
        "--memory",
        dest="memory_in_gb",
        type=int,
        required=False,
        default=100,
        help='Amount of memory in GB Mallet is allowed to use. Default: "100".',
    )
    parser_run.add_argument(
        "-b",
        "--mallet_path",
        dest="mallet_path",
        type=str,
        required=False,
        default="mallet",
        help='Path to the Mallet binary. Default: "mallet".',
    )
    parser_run.add_argument(
        "-c",
        "--cb",
        dest="cell_barcodes_filename",
        type=str,
        required=False,
        help="Filename with cell barcodes. Required for `tomotopy`.",
    )
    parser_run.add_argument(
        "-r",
        "--regions",
        dest="region_ids_filename",
        type=str,
        required=False,
        help="Filename with region IDs. Required for `tomotopy`.",
    )
    parser_run.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help="Enable verbose mode.",
    )

    parser_stats = subparser_topic_modeling.add_parser(
        "stats",
        help="Calculate model evaluation statistics.",
    )
    parser_stats.set_defaults(func=run_calculate_model_evaluation_stats)
    parser_stats.add_argument(
        "-i",
        "--input",
        dest="binary_accessibility_matrix_filename",
        type=str,
        required=True,
        help="Binary accessibility matrix in Matrix Market format.",
    )
    parser_stats.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        type=str,
        required=True,
        help="Topic model output prefix.",
    )
    parser_stats.add_argument(
        "-t",
        "--topics",
        dest="topics",
        type=int,
        required=True,
        nargs="+",
        help="Topic number(s) to calculate model evaluation statistics for.",
    )

    parser_plot_stats = subparser_topic_modeling.add_parser(
        "plot_stats",
        help="Plot evaluation statistics.",
    )
    parser_plot_stats.set_defaults(func=run_plot_model_evaluation_stats)
    parser_plot_stats.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        type=str,
        required=True,
        help="Topic model output prefix.",
    )
    parser_plot_stats.add_argument(
        "-t",
        "--topics",
        dest="n_topics",
        type=int,
        required=True,
        nargs="+",
        help="Topic number(s) to plot model evaluation statistics for.",
    )
    parser_plot_stats.add_argument(
        "-q",
        "--format",
        dest="plot_file_format",
        type=str,
        required=False,
        default="png",
        help="File format of the resulting plot. Default: png.",
    )

    parser_binarize = subparser_topic_modeling.add_parser(
        "binarize",
        help="Binarize cell- or region-topic probabilities.",
    )
    parser_binarize.set_defaults(func=binarize_cell_or_region_topic)
    parser_binarize.add_argument(
        "-a",
        "--target",
        dest="target",
        type=str,
        choices=["region", "cell"],
        required=True,
        help='Choose between "region" or "cell" topic binarization.',
    )
    parser_binarize.add_argument(
        "-m",
        "--method",
        dest="method",
        type=str,
        choices=("ntop", "otsu", "aucell", "li", "yen"),
        required=True,
        help="Binarization method.",
    )
    parser_binarize.add_argument(
        "-n",
        "--ntop",
        dest="ntop",
        type=int,
        required=False,
        help="Number of top regions to select when `--method` is `ntop`.",
    )
    parser_binarize.add_argument(
        "-s",
        "--smooth",
        dest="smooth_topics",
        type=str_to_bool,
        choices=(True, False),
        required=False,
        default=True,
        help="Whether to smooth the cell- or region-topic probabilities.",
    )
    parser_binarize.add_argument(
        "-b",
        "--nbins",
        dest="nbins",
        type=int,
        required=False,
        default=100,
        help="Number of bins to use in thresholding histograms.",
    )
    parser_binarize.add_argument(
        "-c",
        "--cb",
        dest="cell_barcodes_filename",
        type=str,
        required=False,
        help="Filename with cell barcodes.",
    )
    parser_binarize.add_argument(
        "-r",
        "--regions",
        dest="region_ids_filename",
        type=str,
        required=False,
        help="Filename with region IDs.",
    )
    parser_binarize.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        type=str,
        required=True,
        help="Topic model output prefix.",
    )
    parser_binarize.add_argument(
        "-t",
        "--n_topics",
        dest="n_topics",
        type=int,
        required=True,
        help="Model topic count to binarize.",
    )
    parser_binarize.add_argument(
        "-p",
        "--output_dir",
        dest="out_dir",
        type=str,
        required=True,
        help="Directory to store results.",
    )

    parser_anndata = subparser_topic_modeling.add_parser(
        "create_anndata",
        help="Generate AnnData h5ad files from topic modeling results.",
    )
    parser_anndata.set_defaults(func=run_create_anndata)
    parser_anndata.add_argument(
        "-c",
        "--cb",
        dest="cell_barcodes",
        type=str,
        required=True,
        help="Filename with cell barcodes.",
    )
    parser_anndata.add_argument(
        "-r",
        "--regions",
        dest="region_ids",
        type=str,
        required=True,
        help="Filename with region IDs.",
    )
    parser_anndata.add_argument(
        "-o",
        "--output",
        dest="output_prefix",
        type=str,
        required=True,
        help="Topic model output prefix.",
    )
    parser_anndata.add_argument(
        "-t",
        "--n_topics",
        dest="n_topics",
        type=int,
        required=True,
        help="Model topic count to convert to AnnData.",
    )
