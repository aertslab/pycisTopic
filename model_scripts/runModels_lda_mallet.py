import argparse
import os
import pickle
import tempfile

from pycisTopic.lda_models import run_cgs_models_mallet


def make_argument_parser():
    """Creates an ArgumentParser to read the options for this script."""
    parser = argparse.ArgumentParser(
        description="Run LDA topic modeling with Mallet.",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        action="store",
        type=str,
        required=True,
        help="cisTopic object pickle input filename.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        action="store",
        type=str,
        required=True,
        help="Topic model list pickle output filename.",
    )
    parser.add_argument(
        "-t",
        "--topics",
        dest="topics",
        type=int,
        required=True,
        nargs="+",
        help="Number(s) of topics to create during topic modeling.",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        dest="parallel",
        type=int,
        required=True,
        help="Number of threads Mallet is allowed to use.",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        dest="iterations",
        type=int,
        required=False,
        default=150,
        help="Number of iterations. Default: 150.",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        dest="alpha",
        type=int,
        required=False,
        default=50,
        help="Alpha value. Default: 50.",
    )
    parser.add_argument(
        "-A",
        "--alpha_by_topic",
        dest="alpha_by_topic",
        type=str_to_bool,
        choices=(True, False),
        required=False,
        default=True,
        help="Whether the alpha value should by divided by the number of topics. Default: True.",
    )
    parser.add_argument(
        "-e",
        "--eta",
        dest="eta",
        type=float,
        required=False,
        default=0.1,
        help="Eta value. Default: 0.1.",
    )
    parser.add_argument(
        "-E",
        "--eta_by_topic",
        dest="eta_by_topic",
        type=str_to_bool,
        choices=(True, False),
        required=False,
        default=False,
        help="Whether the eta value should by divided by the number of topics. Default: False.",
    )
    parser.add_argument(
        "-k",
        "--keep",
        dest="keep_intermediate_topic_models",
        type=str_to_bool,
        required=False,
        default=False,
        help="Whether intermediate topic models should be kept. "
        "Useful to enable if running with a lot of topic numbers, to not loose finished topic model runs. "
        "Default: False.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        type=int,
        required=False,
        default=555,
        help="Seed for ensuring reproducibility. Default: 555.",
    )
    parser.add_argument(
        "-T",
        "--temp_dir",
        dest="temp_dir",
        type=str,
        required=False,
        default=None,
        help=f'TMP directory to use instead of the default ("{tempfile.gettempdir()}").',
    )
    parser.add_argument(
        "-m",
        "--memory",
        dest="memory_in_gb",
        type=int,
        required=False,
        default=100,
        help='Amount of memory (in GB) Mallet is allowed to use. Default: "100"',
    )
    parser.add_argument(
        "-r",
        "--reuse_corpus",
        dest="reuse_corpus",
        type=str_to_bool,
        required=False,
        default=False,
        help="Whether to reuse the corpus from Mallet. Default: False.",
    )
    return parser


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
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = make_argument_parser()
    args = parser.parse_args()

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

    print(f"Input cisTopic object filename:             {input_filename}")
    print(f"Topic modeling output filename:             {output_filename}")
    print(f"Number of topics to run topic modeling for: {topics}")
    print(f"Alpha:                                      {alpha}")
    print(f"Divide alpha by the number of topics:       {alpha_by_topic}")
    print(f"Eta:                                        {eta}")
    print(f"Divide eta by the number of topics:         {eta_by_topic}")
    print(f"Number of iterations:                       {iterations}")
    print(f"Number threads Mallet is allowed to use:    {parallel}")
    print(f"Seed:                                       {random_state}")
    print(f"Path to TMP dir:                            {temp_dir}")
    print(f"Amount of memory Mallet is allowed to use:  {memory_in_gb}")
    print(f"Reuse Mallet corpus:                        {reuse_corpus}")

    print(f'\nLoading cisTopic object from "{input_filename}"...\n')
    with open(input_filename, "rb") as fh:
        cistopic_obj = pickle.load(fh)

    # Run models
    print("Running models")
    print("--------------")

    os.environ["MALLET_MEMORY"] = memory_in_gb

    models = run_cgs_models_mallet(
        "mallet",
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
    )

    print(f'\nWriting topic modeling output to "{output_filename}"...')
    with open(output_filename, "wb") as fh:
        pickle.dump(models, fh)


if __name__ == "__main__":
    main()
