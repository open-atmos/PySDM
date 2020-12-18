---
title: 'PySDM: Pythonic GPU-enabled particle-based cloud microphysics package'
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
 - name: Jagiellonian University
   index: 1
date: 30 December 2020
bibliography: paper.bib

---

# Summary

`PySDM` is a package for simulating the dynamics of population of particles. 
It is intended to serve as a building block for simulation systems modelling fluid flows involving a dispersed phase, with `PySDM` being responsible for representation of the dispersed phase. 
Currently, the development is focused on atmospheric cloud physics applications, in particular on modelling the dynamics of particles immersed in moist air using the particle-based (a.k.a. super-droplet) approach to represent aerosol/cloud/rain microphysics. 

The package core is a Pythonic high-performance implementation of the Super-Droplet Method (SDM) Monte-Carlo algorithm for representing collisional growth [@Shima_et_al_2009], hence the name. 
`PySDM` has two alternative parallel number-crunching backends available: multi-threaded CPU backend based on `Numba` [@Numba] and GPU-resident backend built on top of `ThrustRTC` [@ThrustRTC].

TODO
Linux, OSX, Windows
Travis

# Statement of need 

TODO

# Mathematics

TODO?

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

@Shima_et_al_2009

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Fenced code blocks are rendered with syntax highlighting:
```python
for n in range(10):
    yield f(n)
```	

# Acknowledgements
We thank Shin-ichiro Shima (University of Hyogo, Japan) for his continuous help and support.
Development of PySDM has been supported by the EU through a grant of the Foundation for Polish Science (POIR.04.04.00-00-5E1C/18).

# References
