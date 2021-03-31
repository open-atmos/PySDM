---
title: 'PySDM v1: Pythonic particle-based cloud modelling package for warm-rain microphysics and aqueous chemistry'
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
  - name: Anna Jaruga
    affiliation: "2"
    orcid: 0000-0003-3194-6440
  - name: Grzegorz Łazarski
    affiliation: "1, 3"
    orcid: 0000-0002-5595-371X
  - name: Michael Olesik
    orcid: 0000-0002-6319-9358
    affiliation: "4"
  - name: Bartosz Piasecki
    affiliation: "1"
  - name: Aleksandra Talar
    affiliation: "1"
affiliations:
 - name: Faculty of Mathematics and Computer Science, Jagiellonian University, Kraków, Poland  
   index: 1
 - name: Department of Environmental Science and Engineering, California Institute of Technology, Pasadena, CA, USA    
   index: 2
 - name: Faculty of Chemistry, Jagiellonian University, Kraków, Poland    
   index: 3
 - name: Faculty of Physics, Astronomy and Applied Computer Science, Jagiellonian University, Kraków, Poland    
   index: 4
bibliography: paper.bib

---

# Introduction

`PySDM` is an open-source Python package for simulating the dynamics of population of particles undergoing condensational and collisional growth,
  interacting with a fluid flow and subject to chemical composition changes. 
It is intended to serve as a building block for process-level as well as computational-fluid-dynamics simulation systems involving representation
  of a continuous phase (fluid) and a dispersed phase (aerosol/hydrosol), with `PySDM` being responsible for representation of the dispersed phase. 
As of the major version 1 (v1), the development has been focused on atmospheric cloud physics applications, in particular on 
  modelling the dynamics of particles immersed in moist air using the particle-based 
  approach to represent 
  the evolution of size spectrum of aerosol/cloud/rain particles. 
The particle-based approach contrasts the more commonly used bulk and bin methods
  in which atmospheric particles are segregated into multiple categories (aerosol, cloud, rain) 
  and their evolution is governed by deterministic dynamics solved on Eulerian grid. 
Particle-based methods employ discrete computational (super) particles, each carrying
   a set of continuously-valued attributes evolving in Lagrangian manner. 
Such approach is particularly well suited for using probabilistic representation of 
  particle collisional growth (coagulation) and for representing processes dependent 
  on numerous particle attributes which helps to overcome the limitations of bulk and bin methods
  [@Morrison_et_al_2020].

The `PySDM` package core is a Pythonic high-performance implementation of the Super-Droplet Method (SDM) Monte-Carlo algorithm for representing collisional growth [@Shima_et_al_2009], hence the name. 
The SDM is a probabilistic alternative to the mean-field approach embodied by the Smoluchowski equation (for a comparative outline of 
  both approaches see @Bartman_and_Arabas_2021).
In atmospheric aerosol-cloud interactions, particle collisional growth is responsible for  
  formation of rain drops through collisions of smaller cloud droplets (warm-rain process)
  as well as for aerosol washout. 

Besides collisional growth, `PySDM` includes representation of condensation/evaporation of
  water vapour on/from the particles.
Furthermore, representation of dissolution and, if applicable, dissociation 
  of trace gases (sulfur dioxide, ozone, hydrogen peroxide, carbon dioxide, nitric acid and ammonia)
  is included to model the subsequent aqueous-phase oxidation of the dissolved sulfur dioxide
  (representation following the work of @Jaruga_and_Pawlowska_2018).

The usage examples are built on four different `environment` classes included in `PySDM` v1
  and implementing common simple atmospheric cloud modelling frameworks: box, adiabatic
  parcel, single-column and 2D prescribed flow kinematic models.

In addition, the package ships with tutorial code depicting how `PySDM` can be used from `Julia` and `Matlab`.

# Dependencies and supported platforms 

PySDM essential dependencies are: `NumPy`, `SciPy`, `Numba`, `Pint` and `ChemPy` which are all free and open-source software available via the PyPI platform.
`PySDM` ships with a setup.py file allowing installation using the `pip` package manager (i.e., `pip install git+https://github.com/atmos-cloud-sim-uj/PySDM.git`).

`PySDM` has two alternative parallel number-crunching backends available: multi-threaded CPU backend based on `Numba` [@Numba] and GPU-resident backend built on top of `ThrustRTC` [@ThrustRTC].
The optional GPU backend relies on proprietary vendor-specific CUDA technology, the accompanying non-free software and drivers; `ThrustRTC` and `CURandRTC` packages released under the Anti-996 license.

The usage examples for `Python` were developed embracing the `Jupyter` interactive platform allowing control of the simulations via web browser.
All Python examples are ready for use in the cloud using the `mybinder.org` and the `Google Colab` platforms.

Continuous integration infrastructure used in the development of PySDM assures the targetted full usability on Linux, macOS and Windows environments 
  and as of the time of writing full compatibility with Python versions 3.7 through 3.9 is maintained.
Test coverage for PySDM is reported using the `codecov.io` platform.
Coverage analysis of the backend code requires execution with JIT-compilation disabled for the CPU backend 
  (e.g., using the `NUMBA_DISABLE_JIT=1` environment variable setting).
For the GPU backend, a purpose-built `FakeThrust` package is shipped with `PySDM` which implements a subset of the `ThrustRTC` API 
  and translates C++ kernels into equivalent `Numba` parallel Python code for debugging and coverage analysis. 

The `Pint` dimensional analysis package is used for unit testing.
It allows to assert on the dimensionality of arithmetic expressions representing physical formulae.
In order to enable JIT compilation of the formulae, a `FakeUnitRegistry` class is shipped that
  mocks the `Pint` API reducing its functionality to SI prefix handling for simulation runs.

# API in brief

In order to depict PySDM API with a practical example, the following listings provide sample code roughly reproducing the Figure 2 from the 
  @Shima_et_al_2009 paper in which the SDM algorithm was introduced. 

It is a coalescence-only set-up in which the initial particle size spectrum is exponential and is deterministically sampled to match the 
  condition of each super-droplet having equal initial multiplicity, with the multiplicity denoting the number of real particles
  represented by a single computational particle referred to as a super-droplet:

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

In the above snippet, the exponential distribution of particle volumes is sampled at $2^{17}$ points 
  in order to initialise two key attributes of the super-droplets, namely their volume and multiplicity. 
Subsequently, a `Builder` object is created to orchestrate dependency injection while instantiating
  the `Core` class of `PySDM`:

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

The `backend` argument may be set to either `CPU` or `GPU` what translates to choosing the multi-threaded `Numba`-based backend or the `ThrustRTC-based` GPU-resident computation mode, respectively. 
The employed `Box` environment corresponds to a zero-dimensional framework (particle positions are neglected).
The SDM Monte-Carlo coalescence algorithm is added as the only dynamic in the system (other dynamics available as of v1.3 represent condensational growth, particle displacement, aqueous chemistry, ambient thermodynamics and Eulerian advection). 
Finally, the `build()` method is used to obtain an instance of the `Core` class which can then be used to control time-stepping and access simulation state
  through the products registered with the builder.
A minimal simulation example is depicted with the code snippet and a resultant plot below:

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

# Usage examples

The PySDM examples are shipped in a separate package
  that can be instaled with `pip install git+https://github.com/atmos-cloud-sim-uj/PySDM-examples.git` or
  conveniently experimented with using Colab or mybinder.org platforms (single-click launching badges included in the 
  `PySDM` README file).
The examples are based on setups from literature and the package is structured using bibliographic labels (e.g., 
  `PySDM_examples.Shima_et_al_2009`).

All examples feature a `settings.py` file with simulation parameters, a `simulation.py` file including logic
  analogous to the one presented in the code snippets above for handling composition of `PySDM` components
  using the `Builder` class, and a `demo.ipynb` Jupyter notebook file with simulation launching code and
  basic result visualisation.

### Box environment examples

The `Box` environment is the simplest one available in PySDM and the `PySDM_examples` package ships with two examples based on it.
The first, is an extension of the code presented in the snippets in the preceding section
  and reproduces Fig. 2 from the seminal paper of @Shima_et_al_2009.
Coalescence is the only process (`Coalescence` dynamic) considered, and the probabilities of collisions of particles
  are evaluated using the Golovin additive kernel, which allows to compare the results with
  analytical solution of the Smoluchowski equation (included in the resultant plots).

The second example based on the `Box` environment, also featuring collision-only setup 
  reproduces several figures from the work of @Berry_1966 involving more sophisticated 
  collision kernels representing such phenomena as geometric sweep-out and influence of electric field of collision probability.

### Adiabatic parcel examples

The `Parcel` environment share the zero-dimensionality of `Box` (i.e., no particle physical coordinates considered), yet
  provides a thermodynamic evolution of the ambient air mimicking adiabatic displacement of an air parcel in 
  hydrostatically stratified atmosphere.
Adiabatic cooling during the ascent results in reaching supersaturation with water vapour what triggers activation of
  aerosol particles (condensation nuclei) into cloud droplets through condensation.
All examples based on the `Parcel` environment thus utilise the `Condensation` and `AmbientThermodynamics` dynamics.

The simplest example uses a monodisperse particle spectrum represented with a single super-droplet
  and reproduces simulations described in @Arabas_and_Shima_2017 where an ascent-descent scenario is employed to
  depict hysteretic behaviour of the activation/deactivation phenomena.

A polydisperse lognormal spectrum represented with multiple super-droplets is used in the example
  based on the work of @Yang_et_al_2018.
Presented simulations involve repeated ascent-descent cycles and depict the evolution of partitioning between
  activated and unactivated particles.
  
Finally, there are two examples featuring adiabatic
  parcel simulations involving representation of the dynamics of chemical composition of both ambient air and
  the droplet-dissolved substances, in particular focusing on the oxidation of aqueous-phase sulphur.
The examples reproduce the simulations discussed in @Kreidenweis_et_al_2003 and in @Jaruga_and_Pawlowska_2018.

### Kinematic (prescribed-flow) examples

Coupling of `PySDM` with fluid-flow simulation is depicted with both 1D and 2D prescribed-flow simulations,
  both dependent on the `PyMPDATA` package [@Arabas_et_al_2021] implementing the MPDATA advection 
  algorithm (for a review, see e.g., @Smolarkiewicz_2006).

Usage of the `kinematic_1d` environment is depicted in an example based on the work of @Shipway_and_Hill_2012,
  while the `kinematic_2d` environment is showcased with a Jupyter notebook featuring an interactive user interface 
  and allowing studying aerosol-cloud interactions in drizzling stratocumulus setup based on the work of @Arabas_et_al_2015.

The figure \autoref{fig:virga} presents a snapshot from the 2D simulation described in detail in @Arabas_et_al_2015 and works cited therein.
Each plot depicts a 1.5x1.5 km vertical slab of an idealised atmosphere in which a prescribed single-eddy non-divergent flow
  is forced (updraft in the left-hand part of the domain, downdraft in the right-hand part). 
The left plot shows the distribution of aerosol particles in the air. 
The upper part of the domain is covered with a stratocumulus-like cloud which formed on the aerosol particles
  above the flat cloud base at the level where relative humidity goes above 100%.
The middle plot depicts the sizes of particles. 
Particles larger than ca. 1 micrometre in radius are considered cloud droplets, particles larger than 
  25 micrometres are considered drizzle.
Within the cloud, the aerosol concentration is reduced. 
Concentration of drizzle particles (larger than 25 $\mu\text{m}$) forming through collisions at the top of the is depicted in the right panel.
A rain shaft forms in the right part of the domain where the downwards flow direction amplifies particle sedimentation.
Precipitating drizzle drops collide with aerosol particles washing out the sub-cloud aerosol.
Most of the drizzle drops evaporate before reaching the bottom of the domain depicting the virga phenomenon (and aerosol resuspension).

![Sample results from a 2D prescribed-flow simulation using the @Arabas_et_al_2015 example.\label{fig:virga}](test.pdf)

# Selected relevant recent open-source developments

The SDM algorithm implementations are part of the following packages (of otherwise largely differing functionality):

   - `SCALE-SDM` in Fortran, [@Sato_et_al_2018]
   - `superdroplet` in Python (`Cython` and `Numba`), C++, Fortran and Julia, (\url{https://github.com/darothen/superdroplet})
   - `Pencil Code` in Fortran, [@Pencil_2021]
   - `PALM LES` in Fortran, [@Maronga_et_al_2020]
   - `libcloudph++` in C++ [@Arabas_et_al_2015;@Jaruga_and_Pawlowska_2018] with Python bindings [@Jarecka_et_al_2015]
   - `LCM1D` in Python/C, [@Unterstrasser_et_al_2020]
   - `NTLP` in Fortran, [@Richter_et_al_2021]
List of links directing to SDM-related files within the above projects' repositories
  is included in the `PySDM` README file.

Python packages for solving dynamics of particles with moving-sectional representation of the size spectrum include:

    - `pyrcel`, [@Rothenberg_and_Wang_2017]
    - `PyBox`, [@Topping_et_al_2018]
Both of the above packages depend on `Assimulo` for solving ODEs.
   
# Summary

The key goal of the reported endeavour was to equip the cloud modelling community with 
  a solution enabling rapid development and paper-review-level reproducibility of simulations
  (i.e., technically feasible without contacting the authors and able to be set up within minutes)
  while being free from the two-language barrier commonly separating prototype and high-performance research code.
Thus, the key advantages of PySDM stem from the characteristics of the employed Python
  language which enables high performance computational
  modelling without trading off such features as:
\begin{description}
    \item[succinct syntax]{ -- the snippets presented in the paper are arguably close to pseudo-code;}
    \item[portability]{depicted in PySDM with continuous integration workflows for Linux, macOS and Windows, including 32-bit and 64-bit runs};
    \item[interoperability]{depicted in PySDM with Matlab and Julia usage examples which do not require any additional biding logic within PySDM;}
    \item[multifaceted ecosystem]{depicted in PySDM with one-click execution of Jupyter notebooks on mybinder.org and colab.research.google.com platforms};
    \item[availability of tools for modern hardware]{depicted in PySDM with the GPU backend}.
\end{description}

PySDM together with a set of developed usage examples constitutes a tool for research on cloud microphysical processes, and for testing and development of novel modelling methods.
PySDM is released under the GNU GPL v3 license.

# Author contributions

PB has been the architect and lead developer of PySDM v1 with SA as the main co-developer.
PySDM 1.0 release accompanied PB's MSc thesis prepared under the mentorship of SA. 
MO contributed to the development of the condensation solver and led the development of relevant examples.
GŁ contributed the initial draft of the aqueous-chemistry extension which was refactored and incorporated into PySDM under guidance from AJ.
KG and BP contributed to the GPU backend.
AT contributed to the examples.
The paper was composed by SA and PB and is partially based on the content of the PySDM README file and PB's MSc thesis.

# Acknowledgements

We thank Shin-ichiro Shima (University of Hyogo, Japan) for his continuous help and support in implementing SDM.
We thank Fei Yang (https://github.com/fynv/) for addressing several issues reported in ThrustRTC during the development of PySDM.
Development of PySDM has been supported by the EU through a grant of the Foundation for Polish Science (POIR.04.04.00-00-5E1C/18).

# References
