"""
PySDM offers a set of building blocks for development of atmospheric cloud
simulation systems revolving around the particle-based microphysics modelling concept
and the Super-Droplet Method algorithm ([Shima et al. 2009](http://doi.org/10.1002/qj.441))
for numerically tackling the probabilistic representation of particle coagulation.

For an overview of PySDM, see [Bartman, Arabas et al. 2021](https://arxiv.org/abs/2103.17238).
PySDM is releases under the [GNU GPL v3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
PySDM development has been spearheaded at the Faculty of Mathematics and Computer Science,
[Jagiellonian University in Kraków](https://en.uj.edu.pl/en) (the copyright holder).

For details on PySDM dependencies and installation procedures, see project
[README.md file](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/README.md)
which also includes basic usage examples in **Python**, **Julia** and **Matlab**.

A set of more elaborate examples engineered in Python and accompanied with Jupyter
notebooks are maintained in the [PySDM-examples package](https://github.com/atmos-cloud-sim-uj/PySDM-examples).

PySDM tests suite built using [pytest](https://docs.pytest.org/) is located in the
[PySDM_tests package](https://github.com/atmos-cloud-sim-uj/PySDM/tree/master/PySDM_tests).
"""

from .builder import Builder
from .core import Core

from pkg_resources import get_distribution, DistributionNotFound, VersionConflict
try:
    __version__ = get_distribution(__name__).version
except (DistributionNotFound, VersionConflict):
    # package is not installed
    pass
