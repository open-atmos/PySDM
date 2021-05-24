"""
PySDM offers a set of building blocks for development of atmospheric cloud
simulation systems revolving around the particle-based microphysics modelling concept
and the Super-Droplet Method algorithm ([Shima et al. 2009](http://doi.org/10.1002/qj.441))
for numerically tackling the probabilistic representation of particle coagulation.

For an overview of PySDM, see [Bartman et al. 2021](https://arxiv.org/abs/2103.17238).

Basic usage examples in Python, Julia and Matlab are provided in the project
[README.md file](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/README.md).

A suite of more elaborate examples engineered in Python and accompanied with Jupyter
notebooks are maintained in the [PySDM-examples package](https://github.com/atmos-cloud-sim-uj/PySDM-examples).
"""

from .builder import Builder
from .core import Core

from pkg_resources import get_distribution, DistributionNotFound, VersionConflict
try:
    __version__ = get_distribution(__name__).version
except (DistributionNotFound, VersionConflict):
    # package is not installed
    pass
