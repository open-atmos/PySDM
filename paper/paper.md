---
title: 'PySDM v2: particle-based cloud microphysics in Python -- collisional breakup, immersion freezing and time-step adaptivity for condensation and collisions'
date: 10 February 2022
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
  - atmospheric-chemistry
authors:
  - name: Emily&nbsp;de Jong
    affiliation: "1"
    orcid: 0000-0002-5310-4554
  - name: Piotr&nbsp;Bartman
    orcid: 0000-0003-0265-6428
    affiliation: "2"
  - name: Kacper&nbsp;Derlatka
    affiliation: "2"
  - name: Isabella&nbsp;Dula
    affiliation: "3"
  - name: Anna&nbsp;Jaruga
    affiliation: "3"
    orcid: 0000-0003-3194-6440
  - name: John&nbsp;Ben&nbsp;Mackay
    affiliation: "3"
    orcid: 0000-0001-8677-3562
  - name: Clare&nbsp;E.&nbsp;Singer
    orcid: 0000-0002-1708-0997
    affiliation: "3"
  - name: Ryan&nbsp;Ward
    affiliation: "3"
  - name: Sylwester&nbsp;Arabas
    orcid: 0000-0003-2361-0082
    affiliation: "4,2"
affiliations:
 - name: Department of Mechanical and Civil Engineering, California Institute of Technology, Pasadena,&nbsp;CA,&nbsp;USA
   index: 1
 - name: Faculty of Mathematics and Computer Science, Jagiellonian University, Kraków,&nbsp;Poland
   index: 2
 - name: Department of Environmental Science and Engineering, California Institute of Technology, Pasadena,&nbsp;CA,&nbsp;USA
   index: 3
 - name: University of Illinois at Urbana-Champaign, Urbana, IL, USA
   index: 4
bibliography: paper.bib

---

# Introduction

... an expanded suite of processes and tests for particle-based microphysics ...

@Bartman_et_al_2022_JOSS

@Bieli_et_al_2022

@Jokulsdottir_and_Archer_2016

@Alpert_and_Knopf_2016

@Shima_et_al_2020

@Rothenberg_and_Wang_2017

@Abdul_Razzak_and_Ghan_2000

@DeJong_et_al_2022

@Ruehl_et_al_2016

# Summary

# Author contributions

EDJ led the formulation and implementation of the collisional breakup scheme with contributions from JBM.
PB led the formulation and implementation of the adaptive time-stepping schemes for diffusional and collisional growth.
KD contributed to setting up continuous integration workflows for the GPU backend.
ID and AJ contributed to the examples (CCN activation).
CES and RW contributed the treatment of organics in surface-tension models and the relevant examples.
The immersion freezing representation code was developed by SA who also carried out the maintenance of the project.

# Acknowledgements

We thank Shin-ichiro Shima (University of Hyogo, Japan) for his continuous help and support in implementing SDM.
Part of the outlined developments was supported by the generosity of Eric and Wendy Schmidt (by recommendation of Schmidt Futures).
Development of ice-phase microphysics representation has been supported through 
grant no. DE-SC0021034 by the Atmospheric System Research Program and 
Atmospheric Radiation Measurement Program sponsored by the U.S. Department of Energy (DOE).

# References
