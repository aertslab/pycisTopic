import contextlib

from pkg_resources import DistributionNotFound, get_distribution

with contextlib.suppress(DistributionNotFound):
    __version__ = get_distribution("pycisTopic").version
