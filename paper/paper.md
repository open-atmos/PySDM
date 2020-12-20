---
title: 'PySDM v1: Pythonic GPU-enabled particle-based cloud microphysics package'
tags:
  - Python
  - physics-simulation 
  - monte-carlo-simulation 
  - gpu-computing 
  - atmospheric-modelling 
  - particle-system 
  - numba 
  - thrust 
  - nvrtc 
  - pint 
  - atmospheric-physics
authors:
  - name: Piotr Bartman
    orcid: 0000-0003-0265-6428
    affiliation: "1"
  - name: Sylwester Arabas
    orcid: 0000-0003-2361-0082
    affiliation: "1"
  - name: Kamil Górski
    affiliation: "1"
  - name: Grzegorz Łazarski
    affiliation: "1"
  - name: Michael Olesik
    orcid: 0000-0002-6319-9358
    affiliation: "1"
  - name: Bartosz Piasecki
    affiliation: "1"
  - name: Aleksandra Talar
    affiliation: "1"
affiliations:
 - name: Jagiellonian University, Kraków, Poland
   index: 1
bibliography: paper.bib

---

# Summary

`PySDM` is an open-source Python package for simulating the dynamics of population of particles. 
It is intended to serve as a building block for simulation systems modelling fluid flows involving a dispersed phase, with `PySDM` being responsible for representation of the dispersed phase. 
So far, the development has been focused on atmospheric cloud physics applications, in particular on modelling the dynamics of particles immersed in moist air using the particle-based (a.k.a. super-droplet) approach to represent aerosol/cloud/rain microphysics. 

The package core is a Pythonic high-performance implementation of the Super-Droplet Method (SDM) Monte-Carlo algorithm for representing collisional growth [@Shima_et_al_2009], hence the name. 
`PySDM` has two alternative parallel number-crunching backends available: multi-threaded CPU backend based on `Numba` [@Numba] and GPU-resident backend built on top of `ThrustRTC` [@ThrustRTC].

PySDM together with a set of bundled usage examples (`PySDM_examples` subpackage) constitutes a tool for research on cloud microphysical processes, and for testing and development of novel modelling methods.
The usage examples were developed embracing the `Jupyter` interactive platform allowing control of the simulations via web browser.

All examples are ready for use in the cloud using the `mybinder.org` and the `Google Colab` platforms.
The packages ships with tutorial code depicting how `PySDM` can be used from `Matlab` and `Julia` (`PySDM_tutorials` subpackage).
Coninuous integration infrastructure used in the development of PySDM (`Travis`, `Github Actions` and `Appveyors`) has been used to reflect targetting full usability on Linux, macOS and Windows environments; Python versions 3.7 and 3.8; and 32- and 64-bit architectures.
Test coverage for PySDM is reported using the `codecov.io` platform.

PySDM essential dependencies (`numpy`, `numba`, `pint`, `molmass`, `scipy`) are free and open-sourse and are all available via the PyPI platform.
PySDM ships with a setup.py file allowing installation using the `pip` package manager (i.e., `pip install --pre git+https://github.com/atmos-cloud-sim-uj/PySDM.git`).
Selected examples have additional dependencies (listed in the `requitrements.txt` file). 
The optional GPU backend relies on proprietary vendor-specific CUDA technology and the accompanying non-free software libraries. 
The GPU backend is implemented using open-source `ThrustRTC` and `CURandRTC` packages released under the Anti-996 license.
PySDM is released under the GNU GPL v3 license.

# Physical processes represented

dynamics:
  - coalescence
  - condensation
  - 
  - 

# Selected relevant recent open-source developments

SDM patents?

  - SDM algorithm implementations are part of the following packages:
    - `SCALE-SDM` (`Fortran`, \url{https://github.com/Shima-Lab}) [@Sato_et_al_2018]
    - `superdroplet` (`Cython`, `Numba`, `C++11`, `Fortran 2008`, `Julia`, \url{https://github.com/darothen/superdroplet})
    - `Pencil Code` (`Fortran`, \url{https://github.com/pencil-code/pencil-code/blob/master/src/particles_coagulation.f90}) [@Li_et_al_2017]
    - `PALM LES` (`Fortran`, \url{https://palm.muk.uni-hannover.de/trac/browser/palm/trunk/SOURCE/lagrangian_particle_model_mod.f90}) [@Maronga_et_al_2020]
    - `libcloudph++` (`C++`, \url{https://github.com/igfuw/libcloudphxx/blob/master/src/impl/particles_impl_coal.ipp}) [@Arabas_et_al_2015]
    - `LCM1D` (`Python`, \url{https://github.com/SimonUnterstrasser/ColumnModel/blob/master/AON_Alg.gcc.py}) [@Unterstrasser_et_al_2020]
  - Python packages for solving dynamics of particles with sectional representation of the size spectrum:
 (all requireing the `Assimulo` package for solving ODEs, while PySDM offers a bespoke adaptive-timestep condensation solver):
    - `pyrcel` (\url{https://github.com/darothen/pyrcel}) [@Rothenberg_and_Wang_2017]
    - `py-cloud-parcel-model` (\rul{http://github.com/emmasimp/py-cloud-parcel-model}) [@]
    - `PyBox` (\url{https://github.com/loftytopping/PyBox}) [@Topping_et_al_2018]



# API in brief

`PySDM.backends`
`PySDM.Builder` and ``PySDM.Core
`PySDM.dynamics`
`PySDM.products`
`PySDM.attributes`
`PySDM.environments`

```python
for n in range(10):
    yield f(n)
```

# Examples and tutorials

examples (add one figure per each example): 
  - box: Shima [@Shima_et_al_2009], Berry [@Berry_1966]
  - parcel: AS [@Arabas_and_Shima_2017], Yang [@Yang_et_al_2018]
  - kinematic: ICMW [@Arabas_et_al_2015]

Figures can be included like this:
![Caption for example figure.\label{fig:example}](test.pdf)
and referenced from text using \autoref{fig:example}.

# Notable hacks

FakeThrust
FakeUnits

# Author contributions

PB has been the architect and lead developer of PySDM v1 with SA as the main co-developer.
PySDM 1.0 release accompanied PB's MSc thesis prepared under the mentorship of SA. 
MO contributed to the development of the condensation solver and led the development of relevant examples.
GŁ contributed the aqueous-chemistry extension (work in progress).
KG and BP contributed to the GPU backend.
AT contributed to the examples.
The paper was composed by SA and PB and is based on the content of the PySDM README file and PB's MSc thesis.

# Acknowledgements
We thank Shin-ichiro Shima (University of Hyogo, Japan) for his continuous help and support.
We thank Fei Yang (https://github.com/fynv/) for addressing several issues reported in ThrustRTC during the development of PySDM.
Development of PySDM has been supported by the EU through a grant of the Foundation for Polish Science (POIR.04.04.00-00-5E1C/18).

# References
