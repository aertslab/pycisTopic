
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution("pycisTopic").version
except DistributionNotFound:
    pass
