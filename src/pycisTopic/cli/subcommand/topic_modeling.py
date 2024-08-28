from __future__ import annotations

import logging
import os
import pickle
import sys
import tempfile
from argparse import ArgumentTypeError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction


def run_topic_modeling_with_lda(args):
    from pycisTopic.lda_models import run_cgs_models

    input_filename = args.input
    output_filename = args.output
    n_topics = args.topics
    alpha = args.alpha
    alpha_by_topic = args.alpha_by_topic
    eta = args.eta
    eta_by_topic = args.eta_by_topic
    n_iter = args.iterations
    n_cpu = args.parallel
    save_path = (
        (output_filename[:-4] if output_filename.endswith(".pkl") else output_filename)
        if args.keep_intermediate_topic_models
        else None
    )
    random_state = args.seed
    temp_dir = args.temp_dir

    if args.verbose:
        level = logging.INFO
        log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level=level, format=log_format, handlers=handlers)

    print("Run topic modeling with lda with the following settings:")
    print(f"  - Input cisTopic object filename:             {input_filename}")
    print(f"  - Topic modeling output filename:             {output_filename}")
    print(f"  - Number of topics to run topic modeling for: {n_topics}")
    print(f"  - Alpha:                                      {alpha}")
    print(f"  - Divide alpha by the number of topics:       {alpha_by_topic}")
    print(f"  - Eta:                                        {eta}")
    print(f"  - Divide eta by the number of topics:         {eta_by_topic}")
    print(f"  - Number of iterations:                       {n_iter}")
    print(f"  - Number of topic models to run in parallel:  {n_cpu}")
    print(f"  - Seed:                                       {random_state}")
    print(f"  - Save intermediate topic models in dir:      {save_path}")
    print(f"  - TMP dir:                                    {temp_dir}")

    print(f'\nLoading cisTopic object from "{input_filename}"...\n')
    with open(input_filename, "rb") as fh:
        cistopic_obj = pickle.load(fh)

    # Run models
    print("Running models")
    print("--------------")
    models = run_cgs_models(
        cistopic_obj,
        n_topics=n_topics,
        n_cpu=n_cpu,
        n_iter=n_iter,
        random_state=random_state,
        alpha=alpha,
        alpha_by_topic=alpha_by_topic,
        eta=eta,
        eta_by_topic=eta_by_topic,
        save_path=save_path,
        _temp_dir=temp_dir,
    )

    print(f'\nWriting topic modeling output to "{output_filename}"...')
    with open(output_filename, "wb") as fh:
        pickle.dump(models, fh)


def run_topic_modeling_with_mallet(args):
    from pycisTopic.lda_models import LDAMallet

    mallet_corpus_filename = args.mallet_corpus_filename
    output_prefix = args.output_prefix
    n_topics_list = [args.topics] if isinstance(args.topics, int) else args.topics
    alpha = args.alpha
    alpha_by_topic = args.alpha_by_topic
    eta = args.eta
    eta_by_topic = args.eta_by_topic
    n_iter = args.iterations
    n_cpu = args.parallel
    random_state = args.seed
    memory_in_gb = f"{args.memory_in_gb}G"
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
    print(f"  - Number of iterations:                       {n_iter}")
    print(f"  - Number threads Mallet is allowed to use:    {n_cpu}")
    print(f"  - Seed:                                       {random_state}")
    print(f"  - Amount of memory Mallet is allowed to use:  {memory_in_gb}")
    print(f"  - Mallet binary:                              {mallet_path}")

    os.environ["MALLET_MEMORY"] = memory_in_gb

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
            n_cpu=n_cpu,
            optimize_interval=0,
            iterations=n_iter,
            topic_threshold=0.0,
            random_seed=random_state,
            mallet_path=mallet_path,
        )

        print(
            f'\nWriting Mallet topic modeling output files to "{output_prefix}.{n_topics}_topics.*"...'
        )


def run_convert_binary_matrix_to_mallet_corpus_file(args):
    import scipy
    from pycisTopic.lda_models import LDAMallet

    binary_accessibility_matrix_filename = args.binary_accessibility_matrix_filename
    mallet_corpus_filename = args.mallet_corpus_filename
    mallet_path = args.mallet_path
    memory_in_gb = f"{args.memory_in_gb}G"

    if args.verbose:
        level = logging.INFO
        log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level=level, format=log_format, handlers=handlers)

    print(
        f'Read binary accessibility matrix from "{binary_accessibility_matrix_filename}" Matrix Market file.'
    )
    binary_accessibility_matrix = scipy.io.mmread(binary_accessibility_matrix_filename)

    os.environ["MALLET_MEMORY"] = memory_in_gb

    print(
        f'Convert binary accessibility matrix to Mallet serialized corpus file "{mallet_corpus_filename}".'
    )
    LDAMallet.convert_binary_matrix_to_mallet_corpus_file(
        binary_accessibility_matrix=binary_accessibility_matrix,
        mallet_corpus_filename=mallet_corpus_filename,
        mallet_path=mallet_path,
    )


def run_mallet_calculate_model_evaluation_stats(args):
    import scipy
    from pycisTopic.fragments import read_barcodes_file_to_polars_series
    from pycisTopic.lda_models import LDAMallet, calculate_model_evaluation_stats

    binary_accessibility_matrix_filename = args.binary_accessibility_matrix_filename
    cell_barcodes_filename = args.cell_barcodes_filename
    region_ids_filename = args.region_ids_filename
    output_prefix = args.output_prefix
    n_topics_list = [args.topics] if isinstance(args.topics, int) else args.topics

    print(
        f'Read binary accessibility matrix from "{binary_accessibility_matrix_filename}" Matrix Market file.'
    )
    binary_accessibility_matrix = scipy.io.mmread(binary_accessibility_matrix_filename)

    print(f'Read cell barcodes filename "{cell_barcodes_filename}".')
    cell_barcodes = read_barcodes_file_to_polars_series(
        barcodes_tsv_filename=cell_barcodes_filename,
        sample_id=None,
        cb_end_to_remove=None,
        cb_sample_separator=None,
    ).to_list()

    print(f'Read region IDs filename "{region_ids_filename}".')
    region_ids = read_barcodes_file_to_polars_series(
        barcodes_tsv_filename=region_ids_filename,
        sample_id=None,
        cb_end_to_remove=None,
        cb_sample_separator=None,
    ).to_list()

    for n_topics in n_topics_list:
        print(
            f'Calculate model evaluation statistics for {n_topics} topics from "{output_prefix}.{n_topics}_topics.*"...'
        )
        calculate_model_evaluation_stats(
            binary_accessibility_matrix=binary_accessibility_matrix,
            cell_barcodes=cell_barcodes,
            region_ids=region_ids,
            output_prefix=output_prefix,
            n_topics=n_topics,
            top_topics_coh=5,
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

    parser_topic_modeling_lda = subparser_topic_modeling.add_parser(
        "lda",
        help='Run LDA topic modeling with "lda" package.',
    )
    parser_topic_modeling_lda.set_defaults(func=run_topic_modeling_with_lda)

    parser_topic_modeling_lda.add_argument(
        "-i",
        "--input",
        dest="input",
        action="store",
        type=str,
        required=True,
        help="cisTopic object pickle input filename.",
    )
    parser_topic_modeling_lda.add_argument(
        "-o",
        "--output",
        dest="output",
        action="store",
        type=str,
        required=True,
        help="Topic model list pickle output filename.",
    )
    parser_topic_modeling_lda.add_argument(
        "-t",
        "--topics",
        dest="topics",
        type=int,
        required=True,
        nargs="+",
        help="Number(s) of topics to create during topic modeling.",
    )
    parser_topic_modeling_lda.add_argument(
        "-p",
        "--parallel",
        dest="parallel",
        type=int,
        required=True,
        help="Number of topic models to run in parallel.",
    )
    parser_topic_modeling_lda.add_argument(
        "-n",
        "--iterations",
        dest="iterations",
        type=int,
        required=False,
        default=150,
        help="Number of iterations. Default: 150.",
    )
    parser_topic_modeling_lda.add_argument(
        "-a",
        "--alpha",
        dest="alpha",
        type=int,
        required=False,
        default=50,
        help="Alpha value. Default: 50.",
    )
    parser_topic_modeling_lda.add_argument(
        "-A",
        "--alpha_by_topic",
        dest="alpha_by_topic",
        type=str_to_bool,
        choices=(True, False),
        required=False,
        default=True,
        help="Whether the alpha value should by divided by the number of topics. Default: True.",
    )
    parser_topic_modeling_lda.add_argument(
        "-e",
        "--eta",
        dest="eta",
        type=float,
        required=False,
        default=0.1,
        help="Eta value. Default: 0.1.",
    )
    parser_topic_modeling_lda.add_argument(
        "-E",
        "--eta_by_topic",
        dest="eta_by_topic",
        type=str_to_bool,
        choices=(True, False),
        required=False,
        default=False,
        help="Whether the eta value should by divided by the number of topics. Default: False.",
    )
    parser_topic_modeling_lda.add_argument(
        "-k",
        "--keep",
        dest="keep_intermediate_topic_models",
        type=str_to_bool,
        choices=(True, False),
        required=False,
        default=False,
        help="Whether intermediate topic models should be kept. "
        "Useful to enable if running with a lot of topic numbers, to not lose finished topic model runs. "
        "Default: False.",
    )
    parser_topic_modeling_lda.add_argument(
        "-s",
        "--seed",
        dest="seed",
        type=int,
        required=False,
        default=555,
        help="Seed for ensuring reproducibility. Default: 555.",
    )
    parser_topic_modeling_lda.add_argument(
        "-T",
        "--temp_dir",
        dest="temp_dir",
        type=str,
        required=False,
        default=None,
        help=f'TMP directory to use instead of the default ("{tempfile.gettempdir()}").',
    )
    parser_topic_modeling_lda.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help="Enable verbose mode.",
    )

    parser_topic_modeling_mallet = subparser_topic_modeling.add_parser(
        "mallet", help='Run LDA topic modeling with "Mallet".'
    )

    subparser_topic_modeling_mallet = parser_topic_modeling_mallet.add_subparsers(
        title='Topic modeling with "Mallet"',
        dest="mallet",
        help='List of "Mallet" topic modeling subcommands.',
    )
    subparser_topic_modeling_mallet.required = True

    parser_topic_modeling_mallet_create_corpus = subparser_topic_modeling_mallet.add_parser(
        "create_corpus",
        help="Convert binary accessibility matrix to Mallet serialized corpus file.",
    )
    parser_topic_modeling_mallet_create_corpus.set_defaults(
        func=run_convert_binary_matrix_to_mallet_corpus_file
    )

    parser_topic_modeling_mallet_create_corpus.add_argument(
        "-i",
        "--input",
        dest="binary_accessibility_matrix_filename",
        action="store",
        type=str,
        required=True,
        help="Binary accessibility matrix (region IDs vs cell barcodes) in Matrix Market format.",
    )
    parser_topic_modeling_mallet_create_corpus.add_argument(
        "-o",
        "--output",
        dest="mallet_corpus_filename",
        action="store",
        type=str,
        required=True,
        help="Mallet serialized corpus filename.",
    )
    parser_topic_modeling_mallet_create_corpus.add_argument(
        "-m",
        "--memory",
        dest="memory_in_gb",
        type=int,
        required=False,
        default=10,
        help='Amount of memory (in GB) Mallet is allowed to use. Default: "10"',
    )
    parser_topic_modeling_mallet_create_corpus.add_argument(
        "-b",
        "--mallet_path",
        dest="mallet_path",
        type=str,
        required=False,
        default="mallet",
        help='Path to Mallet binary (e.g. "/xxx/Mallet/bin/mallet"). Default: "mallet".',
    )
    parser_topic_modeling_mallet_create_corpus.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help="Enable verbose mode.",
    )

    parser_topic_modeling_mallet_run = subparser_topic_modeling_mallet.add_parser(
        "run",
        help='Run LDA topic modeling with "Mallet".',
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
        help="Number of iterations. Default: 150.",
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
        help="Seed for ensuring reproducibility. Default: 555.",
    )
    parser_topic_modeling_mallet_run.add_argument(
        "-m",
        "--memory",
        dest="memory_in_gb",
        type=int,
        required=False,
        default=100,
        help='Amount of memory (in GB) Mallet is allowed to use. Default: "100"',
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
        "-c",
        "--cb",
        dest="cell_barcodes_filename",
        action="store",
        type=str,
        required=True,
        help="Filename with cell barcodes.",
    )
    parser_topic_modeling_mallet_calculate_stats.add_argument(
        "-r",
        "--regions",
        dest="region_ids_filename",
        action="store",
        type=str,
        required=True,
        help="Filename with region IDs.",
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
