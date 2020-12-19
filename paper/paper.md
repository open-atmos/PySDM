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

PySDM together with a set of bundled usage examples constitutes a tool for research on cloud microphysical processes, and for testing and development of novel modelling methods.
The usage examples were developed embracing the `Jupyter` interactive platform allowing control of the simulations via web browser.

PySDM is released under the GNU GPL v3 license.

TODO
license
Linux, OSX, Windows
test coverage
dependencies (separate for core and examples)
setup.py
Travis, GA, Appveyor
Matlab, Julia
dynamics:
  - coalescence
  - condensation
  - 
  - 
other packages
SDM patents?
colab, Jupyter

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.


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

# Examples

examples (add one figure per each example): 
  - box: Shima [@Shima_et_al_2009], Berry [@Berry_1966]
  - parcel: AS [@Arabas_and_Shima_2017], Yang [@Yang_et_al_2018]
  - kinematic: ICMW [@Arabas_et_al_2015]

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
We thank Fei Yang for addressing several issues reported in ThrustRTC during the development of PySDM.
Development of PySDM has been supported by the EU through a grant of the Foundation for Polish Science (POIR.04.04.00-00-5E1C/18).

# References
