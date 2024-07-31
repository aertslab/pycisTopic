from __future__ import annotations

import os
import pickle
import tempfile
from argparse import ArgumentTypeError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser, _SubParsersAction


def run_topic_modeling_lda(args):
    from pycisTopic.lda_models import run_cgs_models

    input_filename = args.input
    output_filename = args.output
    topics = args.topics
    alpha = args.alpha
    alpha_by_topic = args.alpha_by_topic
    eta = args.eta
    eta_by_topic = args.eta_by_topic
    iterations = args.iterations
    parallel = args.parallel
    save_path = (
        (output_filename[:-4] if output_filename.endswith(".pkl") else output_filename)
        if args.keep_intermediate_topic_models
        else None
    )
    random_state = args.seed
    temp_dir = args.temp_dir

    print("Run topic modeling with lda with the following settings:")
    print(f"  - Input cisTopic object filename:             {input_filename}")
    print(f"  - Topic modeling output filename:             {output_filename}")
    print(f"  - Number of topics to run topic modeling for: {topics}")
    print(f"  - Alpha:                                      {alpha}")
    print(f"  - Divide alpha by the number of topics:       {alpha_by_topic}")
    print(f"  - Eta:                                        {eta}")
    print(f"  - Divide eta by the number of topics:         {eta_by_topic}")
    print(f"  - Number of iterations:                       {iterations}")
    print(f"  - Number of topic models to run in parallel:  {parallel}")
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
        n_topics=topics,
        n_cpu=parallel,
        n_iter=iterations,
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


def run_topic_modeling_mallet(args):
    from pycisTopic.lda_models import run_cgs_models_mallet

    input_filename = args.input
    output_filename = args.output
    topics = args.topics
    alpha = args.alpha
    alpha_by_topic = args.alpha_by_topic
    eta = args.eta
    eta_by_topic = args.eta_by_topic
    iterations = args.iterations
    parallel = args.parallel
    save_path = (
        (output_filename[:-4] if output_filename.endswith(".pkl") else output_filename)
        if args.keep_intermediate_topic_models
        else None
    )
    random_state = args.seed
    memory_in_gb = f"{args.memory_in_gb}G"
    temp_dir = args.temp_dir
    reuse_corpus = args.reuse_corpus
    mallet_path = args.mallet_path

    print("Run topic modeling with Mallet with the following settings:")
    print(f"  - Input cisTopic object filename:             {input_filename}")
    print(f"  - Topic modeling output filename:             {output_filename}")
    print(f"  - Number of topics to run topic modeling for: {topics}")
    print(f"  - Alpha:                                      {alpha}")
    print(f"  - Divide alpha by the number of topics:       {alpha_by_topic}")
    print(f"  - Eta:                                        {eta}")
    print(f"  - Divide eta by the number of topics:         {eta_by_topic}")
    print(f"  - Number of iterations:                       {iterations}")
    print(f"  - Number threads Mallet is allowed to use:    {parallel}")
    print(f"  - Seed:                                       {random_state}")
    print(f"  - Save intermediate topic models in dir:      {save_path}")
    print(f"  - TMP dir:                                    {temp_dir}")
    print(f"  - Reuse Mallet corpus:                        {reuse_corpus}")
    print(f"  - Amount of memory Mallet is allowed to use:  {memory_in_gb}")
    print(f"  - Mallet binary:                              {mallet_path}")

    print(f'\nLoading cisTopic object from "{input_filename}"...\n')
    with open(input_filename, "rb") as fh:
        cistopic_obj = pickle.load(fh)

    # Run models
    print("Running models")
    print("--------------")

    os.environ["MALLET_MEMORY"] = memory_in_gb

    models = run_cgs_models_mallet(
        cistopic_obj,
        n_topics=topics,
        n_cpu=parallel,
        n_iter=iterations,
        random_state=random_state,
        alpha=alpha,
        alpha_by_topic=alpha_by_topic,
        eta=eta,
        eta_by_topic=eta_by_topic,
        save_path=save_path,
        top_topics_coh=5,
        tmp_path=temp_dir,
        reuse_corpus=reuse_corpus,
        mallet_path=mallet_path,
    )

    print(f'\nWriting topic modeling output to "{output_filename}"...')
    with open(output_filename, "wb") as fh:
        pickle.dump(models, fh)

def snapatac_to_corpus(args):
    import snapatac2 as snap
    from tqdm import tqdm
    input_filename = args.input
    output_filename = args.output
    chunk_size = args.chunk_size

    print("Loading data")
    anndata: snap.AnnData = snap.read(
        input_filename,
        backed = "r"
    )

    chunk_iter = anndata.X.chunked(chunk_size)
    n_cells = anndata.X.shape[0]

    print("Writing to file ...")
    with open(output_filename, "w") as outf:
        for chunk, cell_idx_offset, _ in tqdm(
            chunk_iter, total = n_cells // chunk_size
        ):
            for cell_idx, (indprev, indnow) in enumerate(
                zip(
                    chunk.indptr, chunk.indptr[1:]
                )
            ):
                tokens = chunk.indices[indprev:indnow]
                doc_idx = cell_idx + cell_idx_offset
                _ = outf.write(
                    f'{doc_idx}\t0\t{" ".join(tokens)}\n'
                )
    
    print("Done!")
    anndata.close()


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
        help='"Run LDA topic modeling with "lda" package.',
    )
    parser_topic_modeling_lda.set_defaults(func=run_topic_modeling_lda)

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
        default=500,
        help="Number of iterations. Default: 500.",
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

    parser_topic_modeling_mallet = subparser_topic_modeling.add_parser(
        "mallet",
        help='"Run LDA topic modeling with "Mallet".',
    )
    parser_topic_modeling_mallet.set_defaults(func=run_topic_modeling_mallet)

    parser_topic_modeling_mallet.add_argument(
        "-i",
        "--input",
        dest="input",
        action="store",
        type=str,
        required=True,
        help="cisTopic object pickle input filename.",
    )
    parser_topic_modeling_mallet.add_argument(
        "-o",
        "--output",
        dest="output",
        action="store",
        type=str,
        required=True,
        help="Topic model list pickle output filename.",
    )
    parser_topic_modeling_mallet.add_argument(
        "-t",
        "--topics",
        dest="topics",
        type=int,
        required=True,
        nargs="+",
        help="Number(s) of topics to create during topic modeling.",
    )
    parser_topic_modeling_mallet.add_argument(
        "-p",
        "--parallel",
        dest="parallel",
        type=int,
        required=True,
        help="Number of threads Mallet is allowed to use.",
    )
    parser_topic_modeling_mallet.add_argument(
        "-n",
        "--iterations",
        dest="iterations",
        type=int,
        required=False,
        default=150,
        help="Number of iterations. Default: 500.",
    )
    parser_topic_modeling_mallet.add_argument(
        "-a",
        "--alpha",
        dest="alpha",
        type=int,
        required=False,
        default=50,
        help="Alpha value. Default: 50.",
    )
    parser_topic_modeling_mallet.add_argument(
        "-A",
        "--alpha_by_topic",
        dest="alpha_by_topic",
        type=str_to_bool,
        choices=(True, False),
        required=False,
        default=True,
        help="Whether the alpha value should by divided by the number of topics. Default: True.",
    )
    parser_topic_modeling_mallet.add_argument(
        "-e",
        "--eta",
        dest="eta",
        type=float,
        required=False,
        default=0.1,
        help="Eta value. Default: 0.1.",
    )
    parser_topic_modeling_mallet.add_argument(
        "-E",
        "--eta_by_topic",
        dest="eta_by_topic",
        type=str_to_bool,
        choices=(True, False),
        required=False,
        default=False,
        help="Whether the eta value should by divided by the number of topics. Default: False.",
    )
    parser_topic_modeling_mallet.add_argument(
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
    parser_topic_modeling_mallet.add_argument(
        "-s",
        "--seed",
        dest="seed",
        type=int,
        required=False,
        default=555,
        help="Seed for ensuring reproducibility. Default: 555.",
    )
    parser_topic_modeling_mallet.add_argument(
        "-T",
        "--temp_dir",
        dest="temp_dir",
        type=str,
        required=False,
        default=None,
        help=f'TMP directory to use instead of the default ("{tempfile.gettempdir()}").',
    )
    parser_topic_modeling_mallet.add_argument(
        "-m",
        "--memory",
        dest="memory_in_gb",
        type=int,
        required=False,
        default=100,
        help='Amount of memory (in GB) Mallet is allowed to use. Default: "100"',
    )
    parser_topic_modeling_mallet.add_argument(
        "-r",
        "--reuse_corpus",
        dest="reuse_corpus",
        type=str_to_bool,
        required=False,
        default=False,
        help="Whether to reuse the corpus from Mallet. Default: False.",
    )
    parser_topic_modeling_mallet.add_argument(
        "-b",
        "--mallet_path",
        dest="mallet_path",
        type=str,
        required=False,
        default="mallet",
        help='Path to Mallet binary (e.g. "/xxx/Mallet/bin/mallet"). Default: "mallet".',
    )

    parser_topic_modeling_create_corpus = subparser_topic_modeling.add_parser(
        "create_corpus_from_snapatac",
        help='"Create corpus from snapatac anndata".',
    )
    parser_topic_modeling_create_corpus.set_defaults(func=snapatac_to_corpus)

    parser_topic_modeling_create_corpus.add_argument(
        "-i",
        "--input",
        dest="input",
        action="store",
        type=str,
        required=True,
        help="Input snapatac anndata.",
    )
    parser_topic_modeling_create_corpus.add_argument(
        "-o",
        "--output",
        dest="output",
        action="store",
        type=str,
        required=True,
        help="Output text file.",
    )
    parser_topic_modeling_create_corpus.add_argument(
        "-c",
        "--chunk_size",
        dest="chunk_size",
        type=int,
        required=False,
        default=100,
        help="Chunk size for iterating over snapatac file (default 100).",
    )
