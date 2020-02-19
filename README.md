[![Build Status](https://travis-ci.org/atmos-cloud-sim-uj/PySDM.svg?branch=master)](https://travis-ci.org/atmos-cloud-sim-uj/PySDM)
[![Coverage Status](https://img.shields.io/codecov/c/github/atmos-cloud-sim-uj/PySDM/master.svg)](https://codecov.io/github/atmos-cloud-sim-uj/PySDM?branch=master)

# PySDM
PySDM simulates the dynamics of population of particles immersed in moist air using the particle-based (a.k.a. super-droplet) approach to represent aerosol/cloud/rain microphysics.
The package features a Pythonic implementation of the Super-Droplet Method (SDM) Monte-Carlo algorithm for representing collisinal growth ([Shima et al. 2009](http://doi.org/10.1002/qj.441)), hence the name. 

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
  https://zenodo.org/record/3547539
