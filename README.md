[![Build Status](https://travis-ci.org/atmos-cloud-sim-uj/PySDM.svg?branch=master)](https://travis-ci.org/atmos-cloud-sim-uj/PySDM)
[![Coverage Status](https://img.shields.io/codecov/c/github/atmos-cloud-sim-uj/PySDM/master.svg)](https://codecov.io/github/atmos-cloud-sim-uj/PySDM?branch=master)

# PySDM
PySDM is a package for simulating the dynamics of population of particles 
  immersed in moist air using the particle-based (a.k.a. super-droplet) approach 
  to represent aerosol/cloud/rain microphysics.
The package core is a Pythonic high-performance multi-threaded implementation of the 
  Super-Droplet Method (SDM) Monte-Carlo algorithm for representing collisinal growth 
  ([Shima et al. 2009](http://doi.org/10.1002/qj.441)), hence the name. 

## Dependencies and installation

It is worth here to distinguish the dependencies of the PySDM core subpackage 
(named simply ``PySDM``) vs. ``PySDM_examples`` and ``PySDM_tests`` subpackages.

PySDM core subpackage dependencies are all available through [PyPI](https://pypi.org), 
  the key dependencies are [Numba](http://numba.pydata.org/) and [Numpy](https://numpy.org/).
As of the time of writing, PySDM has three alternative number-crunching backends 
  implemented which are based on [Numba](http://numba.pydata.org/), 
  [Pythran](https://pythran.readthedocs.io/en/latest/) and 
  [ThrustRTC](https://pypi.org/project/ThrustRTC/).

The **Numba backend** is the default, and features multi-threaded parallelism for 
  multi-core CPUs. 
It uses the just-in-time compilation technique based on the LLVM infrastructure.

The **Pythran backend** uses the ahead-of-time compilation approach (also using LLVM) and
  offers an alternative implementation of the multi-threaded parallelism in PySDM.

The **ThrustRTC** backend offers GPU-resident operation of PySDM
  leveraging the [SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads) 
  parallelisation model. 
Note that, as of ThrustRTC v0.2.1, only Python 3.7 is supported by the ThrustRTC PyPI package
  (i.e., manual installation is needed for other versions of Python).

The dependencies of PySDM examples and test subpackages are summarised in
  the [requirements.txt](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/requirements.txt) 
  file.
Additionally, the [MPyDATA](https://github.com/atmos-cloud-sim-uj/MPyDATA) package
  is used in one of the examples (``ICMW_2012_case_1``), and is bundled 
  in the PySDM repository as a git submodule (``submodules/MPyDATA`` path).
Hints on the installation workflow can be sought in the [.travis.yml](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/.travis.yml) file
  used in the continuous integration workflow of PySDM for Linux, OSX and Windows.

## Demos:
- [Shima et al. 2009](http://doi.org/10.1002/qj.441) Fig. 2 
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_examples%2FShima_et_al_2009_Fig_2/demo.ipynb)
- [Arabas & Shima 2017](http://dx.doi.org/10.5194/npg-24-535-2017) Fig. 5
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_examples%2FArabas_and_Shima_2017_Fig_5/demo.ipynb)
- [Yang et al. 2018](http://doi.org/10.5194/acp-18-7313-2018) Fig. 2:
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_examples%2FYang_et_al_2018_Fig_2/demo.ipynb)
- ICMW 2012 case 1 (work in progress)
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_examples%2FICMW_2012_case_1/demo.ipynb)
  
## Tutorials:
- Introduction [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_tutorials%2F_intro.ipynb)
- Coalescence [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_tutorials%2Fcoalescence.ipynb)

## Package structure and API

The key element of the PySDM interface if the [``Particles``](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/simulation/particles.py) 
  class which instances are used to control the simulation.
Instantiation of the ``Particles`` class is handled by the ``ParticlesBuilder``.
  
.set_mesh()
.set_mesh_0d()
.set_condensation_parameters()
.set_terminal_velocity
.set_environment()
.create_state_0d()
.create_state_2d
.register_dynamic()
.get_particles()



## Credits:

Development of PySDM is supported by the EU through a grant of the Foundation for Polish Science (POIR.04.04.00-00-5E1C/18).

copyright: Jagiellonian University   
code licence: GPL v3   
tutorials licence: CC-BY

## Other open-source SDM implementations:
- SCALE-SDM (Fortran):    
  https://github.com/Shima-Lab/SCALE-SDM_BOMEX_Sato2018/blob/master/contrib/SDM/sdm_coalescence.f90
- Pencil Code (Fortran):    
  https://github.com/pencil-code/pencil-code/blob/master/src/particles_coagulation.f90
- PALM LES (Fortran):    
  https://palm.muk.uni-hannover.de/trac/browser/palm/trunk/SOURCE/lagrangian_particle_model_mod.f90
- libcloudph++ (C++):    
  https://github.com/igfuw/libcloudphxx/blob/master/src/impl/particles_impl_coal.ipp
- LCM1D (Python)
  https://github.com/SimonUnterstrasser/ColumnModel
