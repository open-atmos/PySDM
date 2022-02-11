---
title: 'PySDM v2'
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
  - name: Kacper&nbsp;Derlatka
    affiliation: "4"
  - name: Isabella&nbsp;Dula
    affiliation: "2"
  - name: Anna&nbsp;Jaruga
    affiliation: "2"
    orcid: 0000-0003-3194-6440
  - name: John&nbsp;Ben&nbsp;Mackay
    affiliation: "2"
    orcid: 0000-0001-8677-3562
  - name: Clare&nbsp;E.&nbsp;Singer
    orcid: 0000-0002-1708-0997
    affiliation: "2"
  - name: Ryan&nbsp;Ward
    affiliation: "2"
  - name: Sylwester&nbsp;Arabas
    orcid: 0000-0003-2361-0082
    affiliation: "3,4"
affiliations:
 - name: Department of Mechanical and Civil Engineering, California Institute of Technology, Pasadena,&nbsp;CA,&nbsp;USA
   index: 1
 - name: Department of Environmental Science and Engineering, California Institute of Technology, Pasadena,&nbsp;CA,&nbsp;USA
   index: 2
 - name: University of Illinois at Urbana-Champaign, Urbana, IL, USA
   index: 3
 - name: Faculty of Mathematics and Computer Science, Jagiellonian University, Kraków,&nbsp;Poland &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
   index: 4
bibliography: paper.bib

---

# Introduction

# Dependencies and supported platforms 

# API in brief

# Selected relevant recent open-source developments

The SDM algorithm implementations are part of the following open-source packages (of otherwise largely differing functionality):

   - `libcloudph++` in C++ [@Arabas_et_al_2015;@Jaruga_and_Pawlowska_2018] with Python bindings [@Jarecka_et_al_2015];
   - `SCALE-SDM` in Fortran, [@Sato_et_al_2018];
   - `PALM LES` in Fortran, [@Maronga_et_al_2020];
   - `LCM1D` in Python/C, [@Unterstrasser_et_al_2020];
   - `Pencil Code` in Fortran, [@Pencil_2021];
   - `NTLP` in Fortran, [@Richter_et_al_2021].
   - `superdroplet` in Python (`Cython` and `Numba`), C++, Fortran and Julia    
      (\url{https://github.com/darothen/superdroplet});

List of links directing to SDM-related files within the above projects' repositories
  is included in the `PySDM` README file.

Python packages for solving the dynamics of aerosol particles with discrete-particle (moving-sectional) representation of the size spectrum include (both depend on the `Assimulo` package for solving ODEs):

   - `pyrcel`, [@Rothenberg_and_Wang_2017];
   - `PyBox`, [@Topping_et_al_2018].
   
# Summary

# Author contributions

TODO

# Acknowledgements

We thank Shin-ichiro Shima (University of Hyogo, Japan) for his continuous help and support in implementing SDM.
TODO

# References
