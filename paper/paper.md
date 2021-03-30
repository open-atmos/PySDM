---
title: 'PySDM v1: particle-based warm-rain/aqueous-chemistry cloud microphysics package with box, parcel & 1D/2D prescribed-flow examples'
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
  - name: Grzegorz Łazarski
    affiliation: "3"
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
 - name: Division of Geological and Planetary Sciences, California Institute of Technology, Pasadena, California, USA
   index: 2
 - name: Faculty of Chemistry, Jagiellonian University, Kraków, Poland
   index: 3
 - name: Faculty of Physics, Astronomy and Applied Computer Science, Jagiellonian University, Kraków, Poland
   index: 4
bibliography: paper.bib

---

# Summary

`PySDM` is an open-source Python package for simulating the dynamics of population of particles undergoing condensational and collisional growth,
  interacting with a fluid flow and subject to chemical composition changes. 
It is intended to serve as a building block for process-level as well as computational-fluid-dynamics simulation systems involving representation
  of a continuous (fluid) and a dispersed (aerosol/hydrosol) phases, with `PySDM` being responsible for representation of the dispersed phase. 
As of major version 1 (v1), the development has been focused on atmospheric cloud physics applications, in particular on modelling the dynamics of particles immersed in moist air using the particle-based (a.k.a. super-droplet/moving-sectional/discrete-point Lagrangian) approach to represent the **evolution of size spectrum of aerosol/cloud/rain particles**. 

The package core is a Pythonic high-performance implementation of the Super-Droplet Method (SDM) Monte-Carlo algorithm for representing collisional growth [@Shima_et_al_2009], hence the name. 
The SDM is a probabilistic alternative to the mean-field approach emodied in the Smoluchowski equation, for a comparative outline of 
  both approaches see [@Bartman_and_Arabas_2021].
`PySDM` has two alternative parallel number-crunching backends available: **multi-threaded CPU backend** based on `Numba` [@Numba] and **GPU-resident backend** built on top of `ThrustRTC` [@ThrustRTC].

PySDM together with a set of developed usage examples constitutes a tool for research on cloud microphysical processes, and for testing and development of novel modelling methods.
The usage examples for `Python` were developed embracing the `Jupyter` interactive platform allowing control of the simulations via web browser.
In addition, the package ships with tutorial code depicting how **`PySDM` can be used from `Julia` and `Matlab`**.

All Python examples are ready for use in the cloud using the `mybinder.org` and the `Google Colab` platforms.
Continuous integration infrastructure used in the development of PySDM assures the targetted full usability on **Linux, macOS and Windows** environments 
  and as of the time of writing full compatibility with Python versions 3.7 through 3.9 is maintained.
Test coverage for PySDM is reported using the `codecov.io` platform.

PySDM essential dependencies (`numpy`, `scipy`, `numba`, `pint` and `chempy`) are free and open-source and are all available via the PyPI platform.
PySDM ships with a setup.py file allowing **installation using the `pip` package manager** (i.e., `pip install git+https://github.com/atmos-cloud-sim-uj/PySDM.git`).
The optional GPU backend relies on proprietary vendor-specific CUDA technology and the accompanying non-free software libraries. 
The GPU backend is implemented using open-source `ThrustRTC` and `CURandRTC` packages released under the Anti-996 license.
PySDM is released under the **GNU GPL v3 license**.

Subsequent sections of this paper provide: an outline of PySDM programming interface; an index of 
examples and tutorials included as of v1.3; an overview of other open-source software packages implementing the SDM algorithm; 
and a brief summary highlighting the key virtues of PySDM and its design. 

# API in brief

In order to depict PySDM API with a practical example, the following listings provide sample code roughly reproducing the Figure 2 from the 
  [@Shima_et_al_2009] paper in which the SDM algorithm was introduced. 

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

# Boundled examples

The PySDM examples are hosted in a separate repository (\url{https://github.com/atmos-cloud-sim-uj/PySDM-examples})
  and are all based on setups from literature.
All examples feature a `settings.py` file with simulation parameters, a `simulation.py` file including logic
  analogous to the one presented in the code snippets above for handling composition of `PySDM` components
  using the `Builder` class, and a `demo.ipynb` Jupyter notebook file with simulation launching code and
  basic result visualisation.

### Box environment examples

The `Box` environment is the simplest one available in PySDM and the `PySDM_examples` package ships with two examples based on it.
The first, labelled `Shima_et_al_2009` is an extension of the code presented in the snippets in the preceding section
  and reproduces Fig. 2 from the seminal paper of [@Shima_et_al_2009].
Coalescence is the only process (`Coalescence` dynamic) considered, and the probabilities of collisions of particles
  are evaluated using the Golovin additive kernel, which is unphysical but allows to compare the results with
  analytical solution of the Smoluchowski equation (included in the resultant plots).

The second example based on the `Box` environment and also featuring collision-only setup is labelled `Berry_1966`
  and reproduces several figures from the seminal work of [@Berry_1966] involving more sophisticated 
  collision kernels representing such phenomena as geometric sweep-out and influence of electric field of collision probability.

### Adiabatic parcel examples

The `Parcel` environment share the zero-dimensional setting of `Box` (i.e., no particle physical coordinates considered), yet
  provides a thermodynamic evolution of the ambient air mimicking adiabatic displacement of an air parcel in 
  hydrostatically stratified atmosphere.
The adiabatic cooling during ascent results in reaching supersaturation with water vapour what triggers activation of
  aerosol particles (condensation nuclei) into cloud droplets through condensation.
All examples based on the `Parcel` environment thus utilise the `Condensation` and `AmbientThermodynamics` dynamics.

The simplest example labelled `Arabas_and_Shima_2017` uses a monodisperse particle spectrum represented with a single super-droplet
  and reproduces simulations described in [@Arabas_and_Shima_2017] where an ascent-descent scenario is employed to
  depict hysteretic behaviour of the activation/deactivation phenomena.

A polydisperse lognormal spectrum represented with multiple super-droplets is used in the `Yang_et_al_2018` example
  based on the work of [@Yang_et_al_2018].
Presented simulations involve repeated ascent-descent cycles and depict the evolution of partitioning between
  activated and unactivated particles.
  
Finally, there are two examples labelled `Kreidenweis_et_al_2003` and `Jaruga_and_Pawlowska_2018` featuring adiabatic
  parcel simulations involving representation of the dynamics of chemical composition of both ambient air and
  the droplet-dissolved substances, in particular focusing on the oxidation of aqueous-phase sulphur.
The examples reproduce the simulations discussed in [@Kreidenweis_et_al_2003] and in [@Jaruga_and_Pawlowska_2018].

### Kinematic (prescribed-flow) examples

Coupling of `PySDM` with fluid-flow simulation is depicted with both 1D and 2D prescribed-flow simulations.

TODO: kinematic 1D warm-rain: [@Shipway_and_Hill_2012]

TODO: kinematic 2D warm-rain: ICMW [@Arabas_et_al_2015]

\begin{figure}[!htbp]
  \includegraphics[width=\linewidth]{test} 

  \caption{\label{fig:TODO}
    ...
  }
\end{figure}

# Selected relevant recent open-source developments


  - SDM algorithm implementations are part of the following packages:
    - `SCALE-SDM` (`Fortran`, \url{https://github.com/Shima-Lab}) [@Sato_et_al_2018]
    - `superdroplet` (`Cython`, `Numba`, `C++11`, `Fortran 2008`, `Julia`, \url{https://github.com/darothen/superdroplet})
    - `Pencil Code` (`Fortran`, \url{https://github.com/pencil-code/pencil-code/blob/master/src/particles_coagulation.f90}) [@Li_et_al_2017]
    - `PALM LES` (`Fortran`, \url{https://palm.muk.uni-hannover.de/trac/browser/palm/trunk/SOURCE/lagrangian_particle_model_mod.f90}) [@Maronga_et_al_2020]
    - `libcloudph++` (C++ with Python bindings, \url{https://github.com/igfuw/libcloudphxx/blob/master/src/impl/particles_impl_coal.ipp}) [@Arabas_et_al_2015;@Jarecka_et_al_2015;@Jaruga_and_Pawlowska_2018]
    - `LCM1D` (`Python`, `C` preprocessor \url{https://github.com/SimonUnterstrasser/ColumnModel/blob/master/AON_Alg.gcc.py}) [@Unterstrasser_et_al_2020]
    - `NTLP` (`Fortran`, \url{https://github.com/Folca/NTLP/blob/SuperDroplet/les.F}) [@Richter_et_al_2021]
  - Python packages for solving dynamics of particles with sectional representation of the size spectrum:
 (all requireing the `Assimulo` package for solving ODEs, while PySDM offers a bespoke adaptive-timestep condensation solver):
    - `pyrcel` (\url{https://github.com/darothen/pyrcel}) [@Rothenberg_and_Wang_2017]
    - `PyBox` (\url{https://github.com/loftytopping/PyBox}) [@Topping_et_al_2018]

Note on the SDM patents: TODO

# Summary

The key virtues of PySDM stem from the characteristics of the employed Python language which enables high performance computational
  modelling without trading off such coveted features as:
\begin{description}
    \item[succinct syntax]{ -- the example snippets presented in the paper are arguably close to pseudo-code;}
    \item[seamless portability]{depicted in PySDM with continuous integration workflows for Linux, macOS and Windows, including 32-bit and 64-bit runs};
    \item[unrivalled interoperability]{depicted in PySDM with Matlab and Julia usage examples which do not require any additional biding logic within PySDM;}
    \item[multifaceted ecosystem]{including vibrant market of cloud-computing solutions and depicted in PySDM with one-click execution of Jupyter notebooks on mybinder.org and colab.research.google.com platforms};
    \item[availability of tools for modern hybrid hardware]{depicted in PySDM with the GPU backend}.
\end{description}
It is worth noting that the above features play well together.
For instance, the GPU backend of PySDM featuring the pseudo-code-like API 
  can be leveraged through Jupyter notebooks in the cloud, as well as from within 
  Matlab code on different operating systems.

The key goal of the reported endeavour was to equip the cloud modelling community with 
  a solution enabling rapid development and paper-review-level reproducibility of simulations
  (i.e., technically feasible without contacting the authors and able to be set up within minutes).
  while being free from the two-language barrier commonly separating prototype and high-performance research code.
  
TODO ...
  
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
