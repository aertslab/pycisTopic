import polars as pl


class PycisTopicCategoricals:
    """Categorical categories used by pycisTopic."""

    CHROMOSOME = pl.Categories("chromosome", "pycistopic")
    NAME = pl.Categories("name", "pycistopic")
    CB = pl.Categories("cb", "pycistopic")
    STRAND = pl.Categories("strand", "pycistopic")
    REGION_ID = pl.Categories("region_id", "pycistopic")
