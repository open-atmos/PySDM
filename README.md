[![Build Status](https://travis-ci.org/atmos-cloud-sim-uj/PySDM.svg?branch=master)](https://travis-ci.org/atmos-cloud-sim-uj/PySDM)
[![Coverage Status](https://img.shields.io/codecov/c/github/atmos-cloud-sim-uj/PySDM/master.svg)](https://codecov.io/github/atmos-cloud-sim-uj/PySDM?branch=master)

# PySDM
PySDM is a package for simulating the dynamics of population of particles 
  immersed in moist air using the particle-based (a.k.a. super-droplet) approach 
  to represent aerosol/cloud/rain microphysics.
The package core is a Pythonic high-performance implementation of the 
  Super-Droplet Method (SDM) Monte-Carlo algorithm for representing collisional growth 
  ([Shima et al. 2009](http://doi.org/10.1002/qj.441)), hence the name. 
PySDM has three alternative number-crunching backends 
  available: multi-threaded CPU backends based on [Numba](http://numba.pydata.org/) 
  and [Pythran](https://pythran.readthedocs.io/en/latest/), and a GPU-resident
  backend built on top of [ThrustRTC](https://pypi.org/project/ThrustRTC/).

## Dependencies and installation

It is worth here to distinguish the dependencies of the PySDM core subpackage 
(named simply ``PySDM``) vs. ``PySDM_examples`` and ``PySDM_tests`` subpackages.

PySDM core subpackage dependencies are all available through [PyPI](https://pypi.org), 
  the key dependencies are [Numba](http://numba.pydata.org/) and [Numpy](https://numpy.org/).


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
  (Box model, coalescence only, test case employing Golovin analytical solution)
- [Arabas & Shima 2017](http://dx.doi.org/10.5194/npg-24-535-2017) Fig. 5
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_examples%2FArabas_and_Shima_2017_Fig_5/demo.ipynb)
  (Adiabatic parcel, monodisperse size spectrum activation/deactivation test case)
- [Yang et al. 2018](http://doi.org/10.5194/acp-18-7313-2018) Fig. 2:
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_examples%2FYang_et_al_2018_Fig_2/demo.ipynb)
  (Adiabatic parcel, polydisperse size spectrum activation/deactivation test case)
- ICMW 2012 case 1 (work in progress)
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_examples%2FICMW_2012_case_1/demo.ipynb)
  (2D prescripted flow stratocumulus-mimicking aerosol collisional processing test case)
  
## Tutorials:
- Introduction [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_tutorials%2F_intro.ipynb)
- Coalescence [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/atmos-cloud-sim-uj/PySDM.git/master?filepath=PySDM_tutorials%2Fcoalescence.ipynb)

## Usage example

In order to depict the PySDM API with a practical example, the following
  listings provide a sample code roughly reproducing the 
  Figure 2 from [Shima et al. 2009 paper](http://doi.org/10.1002/qj.441).
It is a coalescence-only set-up in which the initial particle size 
  spectrum is exponential and is deterministically sampled assuring
  that each super-droplet in the system initially has equal multiplicity:
```Python
from PySDM.physics import si
from PySDM.initialisation.spectral_sampling import constant_multiplicity
from PySDM.initialisation.spectra import Exponential
from PySDM.physics.formulae import volume

n_sd = 2**17
initial_spectrum = Exponential(norm_factor=8.39e12, scale=1.19e5 * si.um**3)
sampling_range = (volume(radius=10 * si.um), volume(radius=100 * si.um))
v, n = constant_multiplicity(n_sd=n_sd, spectrum=initial_spectrum, range=sampling_range)
```

The key element of the PySDM interface if the [``Particles``](https://github.com/atmos-cloud-sim-uj/PySDM/blob/master/PySDM/simulation/particles.py) 
  class which instances are used to control the simulation.
Instantiation of the ``Particles`` class is handled by the ``ParticlesBuilder``
  as exemplified below:
```Python
from PySDM.particles_builder import ParticlesBuilder
from PySDM.environments import Box
from PySDM.dynamics import Coalescence
from PySDM.dynamics.coalescence.kernels import Golovin
from PySDM.backends import Numba

particles_builder = ParticlesBuilder(n_sd=n_sd, backend=Numba)
particles_builder.set_environment(Box, {"dt": 1 * si.s, "dv": 1e6 * si.m**3})
particles_builder.set_attributes(n=n, extensive={'volume': v}, intensive={})
particles_builder.register_dynamic(Coalescence, {"kernel": Golovin(b=1.5e3 / si.s)})
particles = particles_builder.get_particles()
```
The ``backend`` argument may be set to either ``Numba`` or ``ThrustRTC``
  what translates to choosing multi-threaded or GPU-resident computation mode, respectively.
The employed ``Box`` environment corresponds to a zero-dimensional framework
  (particle positions are not considered).
The vectors of particle multiplicities ``n`` and particle volumes ``v`` are
  used to initialise super-droplet attributes.
The ``SDM`` Monte-Carlo coalescence algorithm is registered as the only
  dynamic in the system.
Finally, the ``get_particles()`` method is used to obtain an instance
  of ``Particles`` which can then be used to control time-stepping and
  access simulation state.

The ``run(nt)`` method advences the simulation by ``nt`` timesteps.
In the listing below, its usage is interleaved with plotting logic
  which displays a histogram of particle mass distribution 
  at selected timesteps:
```Python
from PySDM.physics.constants import rho_w
from matplotlib import pyplot
import numpy as np

radius_bins_edges = np.logspace(np.log10(10 * si.um), np.log10(5e3 * si.um), num=32)

for step in [0, 1200, 2400, 3600]:
    particles.run(step - particles.n_steps)
    pyplot.step(x=radius_bins_edges[:-1] / si.um,
                y=particles.products['dv/dlnr'].get(radius_bins_edges) * rho_w / si.g,
                where='post', label=f"t = {step}s")

pyplot.xscale('log')
pyplot.xlabel('particle radius [µm]')
pyplot.ylabel("dm/dlnr [g/m$^3$/(unit dr/r)]")
pyplot.legend()
pyplot.show()
```
The resultant plot looks as follows:

![plot](https://raw.githubusercontent.com/piotrbartman/PySDM/develop/PySDM_tutorials/pics/readme.png)

## Package structure and API

- backends:
    - [Numba](https://github.com/piotrbartman/PySDM/tree/master/PySDM/backends/numba): multi-threaded CPU backend using LLVM-powered just-in-time compilation
    - [ThrustRTC](https://github.com/piotrbartman/PySDM/tree/master/PySDM/backends/thrustRTC): GPU-resident backend using real-time compilation 
    - [Pythran](https://github.com/piotrbartman/PySDM/tree/master/PySDM/backends/pythran.py): multi-threaded CPU backend using LLVM-powered ahead-of-time compilation 
- initialisation:
    - multiplicities
        - integer values
    - r_wet_init
        - kappa_equilibrium
    - spatial_sampling
        - pseudorandom
    - spectra
        - Exponential
        - Lognormal
    - spectral_sampling
        - linear
        - logarithmic
        - constant_multiplicity
- physics:
    - constants: 
    - dimensional_analysis: tools for enabling dimensional analysis of the code for unit tests (based on [pint](https://pint.readthedocs.io/))
    - formulae: 
- Environments:
    - Box: bare zero-dimensional framework 
    - MoistLagrangianParcelAdiabatic: zero-dimensional adiabatic parcel framework
    - MoistEulerian2DKinematic: two-dimensional prescribed-flow-coupled framework with Eulerian advection handled by [MPyDATA](http://github.com/atmos-cloud-sim-uj/MPyDATA/)
- Dynamics:
    - Coalescence
        - coalescence.kernels
            - Golovin
            - Gravitational
    - Condensation
        - condensation.solvers
            - ImplicitInRadiusExplicitInThermodynamic (available: adaptive timestepping)
            - BDF
    - Displacement (includes advection with the flow & sedimentation)
    - EulerianAdvection
- Products (selected):
    - SuperDropletCount
    - ParticlesVolumeSpectrum
    - CondensationTimestep
    

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
