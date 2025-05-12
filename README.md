# <img src="https://raw.githubusercontent.com/open-atmos/PySDM/main/docs/logos/pysdm_logo.svg" width=100 height=146 alt="pysdm logo">

[![Python 3](https://img.shields.io/static/v1?label=Python&logo=Python&color=3776AB&message=3)](https://www.python.org/)
[![LLVM](https://img.shields.io/static/v1?label=LLVM&logo=LLVM&color=gold&message=Numba)](https://numba.pydata.org)
[![CUDA](https://img.shields.io/static/v1?label=CUDA&logo=nVidia&color=87ce3e&message=ThrustRTC)](https://pypi.org/project/ThrustRTC/)
[![Linux OK](https://img.shields.io/static/v1?label=Linux&logo=Linux&color=yellow&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Linux)
[![macOS OK](https://img.shields.io/static/v1?label=macOS&logo=Apple&color=silver&message=%E2%9C%93)](https://en.wikipedia.org/wiki/macOS)
[![Windows OK](https://img.shields.io/static/v1?label=Windows&logo=Windows&color=white&message=%E2%9C%93)](https://en.wikipedia.org/wiki/Windows)
[![Jupyter](https://img.shields.io/static/v1?label=Jupyter&logo=Jupyter&color=f37626&message=%E2%9C%93)](https://jupyter.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/open-atmos/PySDM/graphs/commit-activity)
[![OpenHub](https://www.openhub.net/p/atmos-cloud-sim-uj-PySDM/widgets/project_thin_badge?format=gif)](https://www.openhub.net/p/atmos-cloud-sim-uj-PySDM)
[![status](https://joss.theoj.org/papers/62cad07440b941f73f57d187df1aa6e9/status.svg)](https://joss.theoj.org/papers/62cad07440b941f73f57d187df1aa6e9)
[![DOI](https://zenodo.org/badge/199064632.svg)](https://zenodo.org/badge/latestdoi/199064632)    
[![EU Funding](https://img.shields.io/static/v1?label=EU%20Funding%20by&color=103069&message=FNP&logoWidth=25&logo=image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAeCAYAAABTwyyaAAAEzklEQVRYw9WYS2yUVRiGn3P5ZzozpZ3aUsrNgoKlKBINmkhpCCwwxIAhsDCpBBIWhmCMMYTEhSJ4i9EgBnSBEm81MRrFBhNXEuUSMCopiRWLQqEGLNgr085M5//POS46NNYFzHQ6qGc1i5nzP/P973m/9ztCrf7A8T9csiibCocUbvTzfxLcAcaM3cY3imXz25lT3Y34G7gQYAKV3+bFAHcATlBTPogJNADG92iY28FHW97kyPbnuW/W7xgzAhukQ9xe04PJeOT0HkQRwK0TlEeGWb/kOO9v3kdD3a8YK9GhDMfa6mg9fxunOm/lWPtcpDI4K7n/jnN8+uQbrFrUSiwU/DtSEUB/MsKKBT+zYslJqiYNgVE4JwhHkzy86wlWvrKVWDSZ/YFjZlU39yw4y/rGoyQGowWB67zl4QQue+jssMdXrQvZ/00jyeHwqCgDKwnsiJjSvkYAxsG5K9WsenYbJdqAtAjhCIxCSZt/4fK1w5A2WCvxrUAKCHwNVoA2aGmvq11jJQQapEXrgMBKqmJJugejKGWLIxXrBPFoigfv/omd675gRkU/xgqUDlAhH3UDaAAlLSqUQekAYyVTyhLs3tDMsvntlIYzOFcEcOcEGd9jx9oDbGs6QO0t/Tijxi9S4bhzxiWaVh5m94Zm0n7oui4ybo0raUlcncQnxx+g+WgDF/vLoYDmoqSl/dJUnt7XRCoTZjij0Z6Pc2LiNS4EBBkNvoeOJXN+yPWWSZeANOhwJq/98nKVwNdoL8B5AROxBKBL0gjh8DMhdCh3eJnrA0yqhLpplwmyup6IajvAOIGfKGVx3VmCRGnOMpe5QAdG0bT8CAeeep0d6z6nqjSJnQiZWEllLMWrmz6k+fE9rGk8MVqYgsGv5ZH2i1Opr+9kajzB5d74hKQ+KS3d/WVMLhtgdu1lriRiOR/4nDVunaR24x7qp3UV5Cb/fJvC83nv26W81LIK58SYNFmwq4hsGx/5BwKlzYRma2NUthgOJSew4i7ru9nJYCQF5tApb2yvjiDQKJV/IfJKh0o6qssSLKv/jcAoRKHQQzE2Lj2OMV5OkWFc4MZIpsev8uXWXRx6ZicbGk8QZLxxgwe+x/rlR3h3816+f2E7lbEU+ZDn3vKVpePCdFovzCISHqbl5EIoQOteKMPB1rto65zNyfOz+KOrGl06lHPQyi/WOohH0/T0l1MZH6A3GUEKl7Pmr2la6wBrBWWRDP2DUcqjKVKBGom9RZmABAykwnglafpSJSPQvsfiOR0EQ7ExVmazA8cY6N4K1iw6RdAXRwi4mgrheT5Dvs4LeuS81a15Ll/3dQisFVSVpnj7sf1sX/sZvhAc+6UOrQyBVUQ8gx/orFmDsZqtaw/y1qZ9zKjp5vDpenyjcNe+cLNmTiUdf/bEOddVQ0VpgsOn54ET+EYxvWKALSu+5tGG76it7MNaiZKGQ23zCIcMfUMxBnrjN3fmHHvCAlp+vJcXWx6itqoXpAEnUNLx8iMfo5Xh1i17R3PJYCpC2cZ3qK3sQ8WGEDDuXlAQuFKGHzpmopXhTNfk0bmxs7uC1w6uJul79AxFkMIiBJy5UoUWjrZLU5DCFdTARDHuDqVw+OkSwI0MCEW4gtNF2BPrBCo8fKNbtILWX9aUDqFqHnn7AAAAAElFTkSuQmCC)](https://www.fnp.org.pl/en/)
[![PL Funding](https://img.shields.io/static/v1?label=PL%20Funding%20by&color=d21132&message=NCN&logoWidth=25&logo=image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAANCAYAAACpUE5eAAAABmJLR0QA/wD/AP+gvaeTAAAAKUlEQVQ4jWP8////fwYqAiZqGjZqIHUAy4dJS6lqIOMdEZvRZDPcDQQAb3cIaY1Sbi4AAAAASUVORK5CYII=)](https://www.ncn.gov.pl/?language=en)
[![US Funding](https://img.shields.io/static/v1?label=US%20DOE%20Funding%20by&color=267c32&message=ASR&logoWidth=25&logo=image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAQCAMAAAA25D/gAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAASFBMVEVOTXyyIjRDQnNZWINZWITtzdFUU4BVVIFVVYHWiZM9PG/KZnNXVoJaWYT67/FKSXhgX4hgX4lcW4VbWoX03uHQeIN2VXj///9pZChlAAAAAWJLR0QXC9aYjwAAAAd0SU1FB+EICRMGJV+KCCQAAABdSURBVBjThdBJDoAgEETRkkkZBBX0/kd11QTTpH1/STqpAAwWBkobSlkGbt0o5xmEfqxDZJB2Q6XMoBwnVSbTylWp0hi42rmbwTOYPDfR5Kc+07IIUQQvghX9THsBHcES8/SiF0kAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTctMDgtMDlUMTk6MDY6MzcrMDA6MDCX1tBgAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE3LTA4LTA5VDE5OjA2OjM3KzAwOjAw5oto3AAAAABJRU5ErkJggg==)](https://asr.science.energy.gov/)

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)

[![Github Actions Build Status](https://github.com/open-atmos/PySDM/workflows/tests/badge.svg?branch=main)](https://github.com/open-atmos/PySDM/actions)
[![Appveyor Build status](http://ci.appveyor.com/api/projects/status/github/open-atmos/PySDM?branch=main&svg=true)](https://ci.appveyor.com/project/slayoo/pysdm/branch/main)
[![Coverage Status](https://codecov.io/gh/open-atmos/PySDM/branch/main/graph/badge.svg)](https://app.codecov.io/gh/open-atmos/PySDM)    
[![PyPI version](https://badge.fury.io/py/PySDM.svg)](https://pypi.org/project/PySDM)
[![API docs](https://shields.mitmproxy.org/badge/docs-pdoc.dev-brightgreen.svg)](https://open-atmos.github.io/PySDM/)

PySDM is a package for simulating the dynamics of population of particles. 
It is intended to serve as a building block for simulation systems modelling
  fluid flows involving a dispersed phase,
  with PySDM being responsible for representation of the dispersed phase.
Currently, the development is focused on atmospheric cloud physics
  applications, in particular on modelling the dynamics of particles immersed in moist air 
  using the particle-based (a.k.a. super-droplet) approach 
  to represent aerosol/cloud/rain microphysics.
The package features a Pythonic high-performance implementation of the 
  Super-Droplet Method (SDM) Monte-Carlo algorithm for representing collisional growth 
  ([Shima et al. 2009](https://doi.org/10.1002/qj.441)), hence the name. 

PySDM documentation is maintained at: [https://open-atmos.github.io/PySDM](https://open-atmos.github.io/PySDM) 

There is a growing set of [example Jupyter notebooks](https://open-atmos.github.io/PySDM/PySDM_examples.html) exemplifying how to perform 
  various types of calculations and simulations using PySDM.
Most of the example notebooks reproduce results and plot from literature, see below for 
  a list of examples and links to the notebooks (which can be either executed or viewed 
  "in the cloud").

There are also a growing set of [tutorials](https://github.com/open-atmos/PySDM/tree/main/tutorials), also in the form of Jupyter notebooks.
These tutorials are intended for teaching purposes and include short explanations of cloud microphysical 
  concepts paired with widgets for running interactive simulations using PySDM.
Each tutorial also comes with a set of questions at the end that can be used as homework problems.
Like the examples, these tutorials can be executed or viewed "in the cloud" making it an especially 
  easy way for students to get started.

PySDM has two alternative parallel number-crunching backends 
  available: multi-threaded CPU backend based on [Numba](http://numba.pydata.org/) 
  and GPU-resident backend built on top of [ThrustRTC](https://pypi.org/project/ThrustRTC/).
The [`Numba`](https://open-atmos.github.io/PySDM/PySDM/backends/numba.html) backend (aliased ``CPU``) features multi-threaded parallelism for 
  multi-core CPUs, it uses the just-in-time compilation technique based on the LLVM infrastructure.
The [`ThrustRTC`](https://open-atmos.github.io/PySDM/PySDM/backends/thrust_rtc.html) backend (aliased ``GPU``) offers GPU-resident operation of PySDM
  leveraging the [SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads) 
  parallelisation model. 
Using the ``GPU`` backend requires nVidia hardware and [CUDA driver](https://developer.nvidia.com/cuda-downloads).

For an overview of PySDM features (and the preferred way to cite PySDM in papers), please refer to our JOSS papers:
- [Bartman et al. 2022](https://doi.org/10.21105/joss.03219) (PySDM v1).
- [de Jong, Singer et al. 2023](https://doi.org/10.21105/joss.04968) (PySDM v2).
  
PySDM includes an extension of the SDM scheme to represent collisional breakup described in [de Jong, Mackay et al. 2023](10.5194/gmd-16-4193-2023).   
For a list of talks and other materials on PySDM as well as a list of published papers featuring PySDM simulations, see the [project wiki](https://github.com/open-atmos/PySDM/wiki).

## Dependencies and Installation

PySDM dependencies are: [Numpy](https://numpy.org/), [Numba](http://numba.pydata.org/), [SciPy](https://scipy.org/), 
[Pint](https://pint.readthedocs.io/), [chempy](https://pypi.org/project/chempy/), 
[pyevtk](https://pypi.org/project/pyevtk/),
[ThrustRTC](https://fynv.github.io/ThrustRTC/) and [CURandRTC](https://github.com/fynv/CURandRTC).

To install PySDM using ``pip``, use: ``pip install PySDM`` 
(or ``pip install git+https://github.com/open-atmos/PySDM.git`` to get updates
beyond the latest release).

Conda users may use ``pip`` as well, see the [Installing non-conda packages](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages) section in the conda docs. Dependencies of PySDM are available at the following conda channels:
- numba: [numba](https://anaconda.org/numba/numba)
- conda-forge: [pyevtk](https://anaconda.org/conda-forge/pyevtk), [pint](https://anaconda.org/conda-forge/pint) and []()
- fyplus: [ThrustRTC](https://anaconda.org/fyplus/thrustrtc), [CURandRTC](https://anaconda.org/fyplus/curandrtc)
- bjodah: [chempy](https://anaconda.org/bjodah/chempy)
- nvidia: [cudatoolkit](https://anaconda.org/nvidia/cudatoolkit)

For development purposes, we suggest cloning the repository and installing it using ``pip -e``.
Test-time dependencies can be installed with ``pip -e .[tests]``.

PySDM examples constitute the [``PySDM-examples``](https://github.com/open-atmos/PySDM/blob/main/examples) package.
The examples have additional dependencies listed in [``PySDM_examples`` package ``setup.py``](https://github.com/open-atmos/PySDM/blob/main/examples/setup.py) file.
Running the example Jupyter notebooks requires the ``PySDM_examples`` package to be installed.
The suggested install and launch steps are:
```
git clone https://github.com/open-atmos/PySDM.git
pip install -e PySDM
pip install -e PySDM/examples
jupyter-notebook PySDM/examples/PySDM_examples
```
Alternatively, one can also install the examples package from pypi.org by 
using ``pip install PySDM-examples`` (note that this does not apply to notebooks itself,
only the supporting .py files).

## Contributing, reporting issues, seeking support 

#### Our technologicial stack:   
[![Python 3](https://img.shields.io/static/v1?label=+&logo=Python&color=darkred&message=Python)](https://www.python.org/)
[![Numba](https://img.shields.io/static/v1?label=+&logo=Numba&color=orange&message=Numba)](https://numba.pydata.org)
[![LLVM](https://img.shields.io/static/v1?label=+&logo=LLVM&color=gold&message=LLVM)](https://llvm.org)
[![CUDA](https://img.shields.io/static/v1?label=+&logo=nVidia&color=darkgreen&message=ThrustRTC/CUDA)](https://pypi.org/project/ThrustRTC/)
[![NumPy](https://img.shields.io/static/v1?label=+&logo=numpy&color=blue&message=NumPy)](https://numpy.org/)
[![pytest](https://img.shields.io/static/v1?label=+&logo=pytest&color=purple&message=pytest)](https://pytest.org/)   
[![Colab](https://img.shields.io/static/v1?label=+&logo=googlecolab&color=darkred&message=Colab)](https://colab.research.google.com/)
[![Codecov](https://img.shields.io/static/v1?label=+&logo=codecov&color=orange&message=Codecov)](https://codecov.io/)
[![PyPI](https://img.shields.io/static/v1?label=+&logo=pypi&color=gold&message=PyPI)](https://pypi.org/)
[![GithubActions](https://img.shields.io/static/v1?label=+&logo=github&color=darkgreen&message=GitHub&nbsp;Actions)](https://github.com/features/actions)
[![Jupyter](https://img.shields.io/static/v1?label=+&logo=Jupyter&color=blue&message=Jupyter)](https://jupyter.org/)
[![PyCharm](https://img.shields.io/static/v1?label=+&logo=pycharm&color=purple&message=PyCharm)](https:///)

Submitting new code to the project, please preferably use [GitHub pull requests](https://github.com/open-atmos/PySDM/pulls) - it helps to keep record of code authorship, 
track and archive the code review workflow and allows to benefit
from the continuous integration setup which automates execution of tests 
with the newly added code. 

Code contributions are assumed to imply transfer of copyright.
Should there be a need to make an exception, please indicate it when creating
a pull request or contributing code in any other way. In any case, 
the license of the contributed code must be compatible with GPL v3.

Developing the code, we follow [The Way of Python](https://www.python.org/dev/peps/pep-0020/) and 
the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle).
The codebase has greatly benefited from [PyCharm code inspections](https://www.jetbrains.com/help/pycharm/code-inspection.html)
and [Pylint](https://pylint.org), [Black](https://black.readthedocs.io/en/stable/) and [isort](https://pycqa.github.io/isort/)
code analysis (which are all part of the CI workflows).

We also use [pre-commit hooks](https://pre-commit.com). 
In our case, the hooks modify files and re-format them.
The pre-commit hooks can be run locally, and then the resultant changes need to be staged before committing.
To set up the hooks locally, install pre-commit via `pip install pre-commit` and
set up the git hooks via `pre-commit install` (this needs to be done every time you clone the project).
To run all pre-commit hooks, run `pre-commit run --all-files`.
The `.pre-commit-config.yaml` file can be modified in case new hooks are to be added or
  existing ones need to be altered.  

Further hints addressed at PySDM developers are maintained in the [open-atmos/python-dev-hints Wiki](https://github.com/open-atmos/python-dev-hints/wiki)
  and in [PySDM HOWTOs](https://github.com/open-atmos/PySDM/tree/main/examples/PySDM_examples/_HOWTOS).

Issues regarding any incorrect, unintuitive or undocumented bahaviour of
PySDM are best to be reported on the [GitHub issue tracker](https://github.com/open-atmos/PySDM/issues/new).
Feature requests are recorded in the "Ideas..." [PySDM wiki page](https://github.com/open-atmos/PySDM/wiki/Ideas-for-new-features-and-examples).

We encourage to use the [GitHub Discussions](https://github.com/open-atmos/PySDM/discussions) feature
(rather than the issue tracker) for seeking support in understanding, using and extending PySDM code.

We look forward to your contributions and feedback.

## Licensing:

copyright: [Jagiellonian University](https://en.uj.edu.pl/en) (2019-2023) & [AGH University of Krakow](https://agh.edu.pl/en) (2023-...)    
licence: [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html)

  
