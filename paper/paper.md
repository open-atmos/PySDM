---
title: 'PySDM v1: particle-based warm-rain cloud microphysics package with box/parcel & 2D prescribed-flow examples'
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
As of major version 1 (v1), the development has been focused on atmospheric cloud physics applications, in particular on modelling the dynamics of particles immersed in moist air using the particle-based (a.k.a. super-droplet/moving-sectional/discrete-point Lagrangian) approach to represent the **evolution of size spectrum of aerosol/cloud/rain particles**. 

The package core is a Pythonic high-performance implementation of the Super-Droplet Method (SDM) Monte-Carlo algorithm for representing collisional growth [@Shima_et_al_2009], hence the name. 
`PySDM` has two alternative parallel number-crunching backends available: **multi-threaded CPU backend** based on `Numba` [@Numba] and **GPU-resident backend** built on top of `ThrustRTC` [@ThrustRTC].

PySDM together with a set of bundled usage examples (`PySDM_examples` subpackage) constitutes a tool for research on cloud microphysical processes, and for testing and development of novel modelling methods.
The usage examples were developed embracing the `Jupyter` interactive platform allowing control of the simulations via web browser.

All examples are ready for use in the cloud using the `mybinder.org` and the `Google Colab` platforms.
The packages ships with tutorial code depicting how **`PySDM` can be used from `Python`, `Julia` and `Matlab`** (`PySDM_tutorials` subpackage).
Coninuous integration infrastructure used in the development of PySDM (`Travis`, `Github Actions` and `Appveyor`) assures the targetted full usability on **Linux, macOS and Windows** environments (with Python versions 3.7 and above and on 32- and 64-bit architectures).
Test coverage for PySDM is reported using the `codecov.io` platform.

PySDM essential dependencies (`numpy`, `numba`, `pint`, `molmass`, `scipy`) are free and open-sourse and are all available via the PyPI platform.
PySDM ships with a setup.py file allowing **installation using the `pip` package manager** (i.e., `pip install --pre git+https://github.com/atmos-cloud-sim-uj/PySDM.git`).
Selected examples have additional dependencies (listed in the `requitrements.txt` file). 
The optional GPU backend relies on proprietary vendor-specific CUDA technology and the accompanying non-free software libraries. 
The GPU backend is implemented using open-source `ThrustRTC` and `CURandRTC` packages released under the Anti-996 license.
PySDM is released under the **GNU GPL v3 license**.

Subsequent sections of this paper provide: an outline of PySDM programming interface; an index of 
examples and tutorials included as of v1.1; an overview of selected relevant open-source software packages; 
and a description of a set of technical solutions introduced in PySDM that are of potential interest to
the wider research software engineering community. 

# API in brief

Its core is represented with the `PySDM.Core` class which instances are built using the `PySDM.Builder`.

 design is domain-driven.

In order to depict PySDM API with a practical example, the following listings provide sample code roughly reproducing the Figure 2 from [@Shima_et_al_2009]. 



It is a coalescence-only set-up in which the initial particle size spectrum is exponential and is deterministically sampled to match the condition of each super-droplet having equal initial multiplicity:

```python
from PySDM.physics import si
from PySDM.initialisation.spectral_sampling import constant_multiplicity
from PySDM.initialisation.spectra import Exponential
from PySDM.physics.formulae import volume

n_sd = 2 ** 17
initial_spectrum = Exponential(
    norm_factor=8.39e12, scale=1.19e5 * si.um ** 3)
sampling_range = (volume(radius=10 * si.um),
                  volume(radius=100 * si.um))
attributes = {}
attributes['volume'], attributes['n'] = constant_multiplicity(
    n_sd=n_sd, spectrum=initial_spectrum, range=sampling_range)
```

```python
from PySDM.builder import Builder
from PySDM.environments import Box
from PySDM.dynamics import Coalescence
from PySDM.dynamics.coalescence.kernels import Golovin
from PySDM.backends import Numba
from PySDM.state.products import ParticlesVolumeSpectrum

builder = Builder(n_sd=n_sd, backend=Numba)
builder.set_environment(Box(dt=1 * si.s, dv=1e6 * si.m ** 3))
builder.add_dynamic(Coalescence(kernel=Golovin(b=1.5e3 / si.s)))
products = [ParticlesVolumeSpectrum()]
core = builder.build(attributes, products)
```

The `backend` argument may be set to `CPU` or `GPU` what translates to choosing the multi-threaded `Numba`-based backend or the `ThrustRTC-based` GPU-resident computation mode, respectively. 
The employed `Box` environment corresponds to a zero-dimensional framework (particle positions are neglected).
The vectors of particle multiplicities `n` and particle volumes `v` are used to initialize super-droplet attributes.
The SDM Monte-Carlo coalescence algorithm is added as the only dynamic in the system (other dynamics available as of v1.1 represent condensational growth and particle displacement). 
Finally, the `build()` method is used to obtain an instance of `Core` which can then be used to control time-stepping and access simulation state.

```python
from PySDM.physics.constants import rho_w
from matplotlib import pyplot
import numpy as np

radius_bins_edges = np.logspace(
    np.log10(10 * si.um), np.log10(5e3 * si.um), num=32)

for step in [0, 1200, 2400, 3600]:
    core.run(step - core.n_steps)
    pyplot.step(
        x=radius_bins_edges[:-1] / si.um,
        y=core.products['dv/dlnr'].get(radius_bins_edges) * rho_w/si.g,
        where='post', label=f"t = {step}s")

pyplot.xscale('log')
pyplot.xlabel('particle radius [$\mu$ m]')
pyplot.ylabel("dm/dlnr [g/m$^3$/(unit dr/r)]")
pyplot.legend()
pyplot.show()
```

\begin{figure}[h]
    \centering
    \includegraphics[width=0.75\textwidth]{readme}
    \caption{Solution for size spectrum evolution with Golovin kernel.}
    \label{fig:readme_fig_1}
\end{figure}

# Examples and tutorials

`PySDM.backends`
`PySDM.dynamics`
`PySDM.products`
`PySDM.attributes`
`PySDM.environments`

examples (add one figure per each example): 
  - box: Shima [@Shima_et_al_2009], Berry [@Berry_1966]
  - parcel: AS [@Arabas_and_Shima_2017], Yang [@Yang_et_al_2018]
  - kinematic: ICMW [@Arabas_et_al_2015]

Figures can be included like this:

\begin{figure}[!htbp]
  \includegraphics[width=\linewidth]{test} 

  \caption{\label{fig:TODO}
    ...
  }
\end{figure}

and referenced from text using \autoref{fig:TODO}.

\begin{figure}[!htbp]
  \includegraphics[width=\linewidth]{spectrum} 

  \caption{\label{fig:TODO}
    ...
  }
\end{figure}

# Notable hacks

During development of PySDM, ...

FakeThrust
FakeUnits
nicethrust
closure pattern

# Selected relevant recent open-source developments

SDM patents?

  - SDM algorithm implementations are part of the following packages:
    - `SCALE-SDM` (`Fortran`, \url{https://github.com/Shima-Lab}) [@Sato_et_al_2018]
    - `superdroplet` (`Cython`, `Numba`, `C++11`, `Fortran 2008`, `Julia`, \url{https://github.com/darothen/superdroplet})
    - `Pencil Code` (`Fortran`, \url{https://github.com/pencil-code/pencil-code/blob/master/src/particles_coagulation.f90}) [@Li_et_al_2017]
    - `PALM LES` (`Fortran`, \url{https://palm.muk.uni-hannover.de/trac/browser/palm/trunk/SOURCE/lagrangian_particle_model_mod.f90}) [@Maronga_et_al_2020]
    - `libcloudph++` (C++ with Python bindings, \url{https://github.com/igfuw/libcloudphxx/blob/master/src/impl/particles_impl_coal.ipp}) [@Arabas_et_al_2015,@Jarecka_et_al_2015,@Jaruga_and_Pawlowska_2018]
    - `LCM1D` (`Python`, \url{https://github.com/SimonUnterstrasser/ColumnModel/blob/master/AON_Alg.gcc.py}) [@Unterstrasser_et_al_2020]
  - Python packages for solving dynamics of particles with sectional representation of the size spectrum:
 (all requireing the `Assimulo` package for solving ODEs, while PySDM offers a bespoke adaptive-timestep condensation solver):
    - `pyrcel` (\url{https://github.com/darothen/pyrcel}) [@Rothenberg_and_Wang_2017]
    - `py-cloud-parcel-model` (\url{http://github.com/emmasimp/py-cloud-parcel-model}) [@]
    - `PyBox` (\url{https://github.com/loftytopping/PyBox}) [@Topping_et_al_2018]

# Author contributions

PB has been the architect and lead developer of PySDM v1 with SA as the main co-developer.
PySDM 1.0 release accompanied PB's MSc thesis prepared under the mentorship of SA. 
MO contributed to the development of the condensation solver and led the development of relevant examples.
GŁ contributed the aqueous-chemistry extension (work in progress).
KG and BP contributed to the GPU backend.
AT contributed to the examples.
The paper was composed by SA and PB and is based on the content of the PySDM README file and PB's MSc thesis.

# Acknowledgements

We thank Shin-ichiro Shima (University of Hyogo, Japan) for his continuous help and support in implementing SDM.
We thank Fei Yang (https://github.com/fynv/) for addressing several issues reported in ThrustRTC during the development of PySDM.
Development of PySDM has been supported by the EU through a grant of the Foundation for Polish Science (POIR.04.04.00-00-5E1C/18).

# References
