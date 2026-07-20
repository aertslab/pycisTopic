from argparse import _SubParsersAction

import polars as pl

OBSM_SIGN_ENRICHMENT_KEY="AUCell"
UNS_SIGN_ENRICHMENT_KEY="AUCell signature names"


def _region_names_to_polars_df(
    region_names: list[str],
    chrom_split: str = ":",
    start_end_split: str = "-",
) -> pl.DataFrame:
    chromosomes: list[str] = []
    starts:      list[int] = []
    ends:        list[int] = []
    for region in region_names:
        chrom, start, end = region.replace(
            chrom_split, start_end_split
        ).split(start_end_split)
        chromosomes.append(chrom)
        starts.append(int(start))
        ends.append(int(end))

    return pl.from_dict(
        {
            "Chromosome": chromosomes,
            "Start": starts,
            "End": ends
        }
    )

def run_signature_enrichment(args):
    import logging
    import sys

    import anndata as ad  # type: ignore
    import numpy as np

    from pycisTopic.fragments import read_bed_to_polars_df
    from pycisTopic.signature_enrichment import signature_enrichment

    if len(args.signature_names) != len(args.signatures):
        raise ValueError(
            f"Length of signature_names ({len(args.signature_names)}) does not "
            f"match the length of the signatures ({len(args.signatures)})!"
        )

    if args.verbose:
        level = logging.INFO
        log_format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        handlers = [logging.StreamHandler(stream=sys.stdout)]
        logging.basicConfig(level=level, format=log_format, handlers=handlers)
        log = logging.getLogger("cisTopic")
    else:
        log = None

    if log is not None:
        log.info("Reading data ...")

    ad_region_topic = ad.read_h5ad(args.region_topic)
    ad_cell_topic = ad.read_h5ad(args.cell_topic)

    assert isinstance(ad_region_topic.X, np.ndarray), \
        "`ad_region_topic.X` should be numpy array"

    assert isinstance(ad_cell_topic.X, np.ndarray), \
        "`ad_cell_topic.X` should be numpy array"

    region_topic_granges = _region_names_to_polars_df(
        ad_region_topic.obs_names # type: ignore
    )

    signatures: dict[str, pl.DataFrame] = {}
    for signature_name, signature_bed in zip(
        args.signature_names, args.signatures,
        strict=True
    ):
        signatures[signature_name] = read_bed_to_polars_df(signature_bed)

    auc_values = signature_enrichment(
        region_topic=ad_region_topic.X,
        cell_topic=ad_cell_topic.X.T,
        region_topic_granges=region_topic_granges,
        signatures=signatures,
        chunk_size=args.chunk_size,
        normalize=args.normalize,
        min_frac_consensus=args.min_frac_consensus,
        min_frac_signature=args.min_frac_signature,
        seed=args.seed,
        auc_threshold=args.auc_threshold,
        log=log
    )

    if OBSM_SIGN_ENRICHMENT_KEY not in ad_cell_topic.obsm:
        if log is not None:
            log.info(
                f"Storing AUCell matrix in .obsm[{OBSM_SIGN_ENRICHMENT_KEY}] "
                f"and signature names in .uns[{UNS_SIGN_ENRICHMENT_KEY}]."
            )
        ad_cell_topic.obsm[OBSM_SIGN_ENRICHMENT_KEY] = auc_values
        ad_cell_topic.uns[UNS_SIGN_ENRICHMENT_KEY] = list(signatures.keys())
    else:
        if log is not None:
            log.info(
                f"Concatenating new AUCell matrix with .obsm[{OBSM_SIGN_ENRICHMENT_KEY}] "
                f"and extending signature names in .uns[{UNS_SIGN_ENRICHMENT_KEY}]."
            )
        ad_cell_topic.obsm[OBSM_SIGN_ENRICHMENT_KEY] = np.concatenate(
            (ad_cell_topic.obsm[OBSM_SIGN_ENRICHMENT_KEY], auc_values),
            axis=1
        )
        assert isinstance(ad_cell_topic.uns[UNS_SIGN_ENRICHMENT_KEY], list), \
            "cell topic anndata contains AUCell values but no signature names in " + \
            f".uns[{UNS_SIGN_ENRICHMENT_KEY}]."
        ad_cell_topic.uns[UNS_SIGN_ENRICHMENT_KEY].extend(list(signatures.keys()))

    if log is not None:
        log.info(f"Saving cell topic anndata to: {args.out_file}")
    ad_cell_topic.write(args.out_file)

def add_parser_signature_enrichment(
    subparser: _SubParsersAction
):
    parser_signature_enrichment = subparser.add_parser(
        "signature_enrichment",
        help="Run signature enrichment using AUCell."
    )
    parser_signature_enrichment.set_defaults(func=run_signature_enrichment)

    parser_signature_enrichment.add_argument(
        "-r",
        "--region-topic",
        dest="region_topic",
        type=str,
        required=True,
        help="Path to region-topic AnnData (.h5ad) file."
    )
    parser_signature_enrichment.add_argument(
        "-c",
        "--cell-topic",
        dest="cell_topic",
        type=str,
        required=True,
        help="Path to cell-topic AnnData (.h5ad) file."
    )
    parser_signature_enrichment.add_argument(
        "-n",
        "--signature-names",
        dest="signature_names",
        nargs="+",
        type=str,
        required=True,
        help="List of signature names (space-separated, same order as --signatures)."
    )
    parser_signature_enrichment.add_argument(
        "-s",
        "--signatures",
        dest="signatures",
        nargs="+",
        type=str,
        required=True,
        help="List of BED files for signatures (space-separated, same order as --signature-names)."
    )
    parser_signature_enrichment.add_argument(
        "-o",
        "--out-file",
        dest="out_file",
        type=str,
        required=True,
        help="Output file for updated cell-topic AnnData (.h5ad)."
    )
    parser_signature_enrichment.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=1000,
        help="Number of cells to process at once. Default: 1000."
    )
    parser_signature_enrichment.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        help="Normalize AUC values to a maximum of 1.0 per regulon."
    )
    parser_signature_enrichment.add_argument(
        "--min-frac-consensus",
        dest="min_frac_consensus",
        type=float,
        default=0.5,
        help="Minimal fractional overlap of signature and consensus peak relative to consensus peak. Default: 0.5."
    )
    parser_signature_enrichment.add_argument(
        "--min-frac-signature",
        dest="min_frac_signature",
        type=float,
        default=0.5,
        help="Minimal fractional overlap of signature and consensus peak relative to signature. Default: 0.5."
    )
    parser_signature_enrichment.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=12345,
        help="Random seed for ranking. Default: 12345."
    )
    parser_signature_enrichment.add_argument(
        "--auc-threshold",
        dest="auc_threshold",
        type=float,
        default=0.05,
        help="Fraction of ranked genome for AUC calculation. Default: 0.05."
    )
    parser_signature_enrichment.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose logging."
    )
