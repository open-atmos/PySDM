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

`PySDM` and the accompanying `PySDM-examples` open-source packages serve as tools for
  computational studies of atmospheric clouds and precipitation.
Particle-based modelling approach and Pythonic design are the cruces of the project.

The eponymous `SDM` stands for Super Droplet Method -- a Monte-Carlo algorithm
  introduced in @Shima_et_al_2009 for representing coagulation of particles in 
  modelling frameworks such as Large-Eddy Simulations (LES) of atmospheric
  flows featuring formation of cloud particles 
  and precipitation of hydrometeors.

`PySDM` is implemented in Python with two alternative backends: multi-threaded
  CPU backend implemented using the Numba Just-In-Time (JIT) compiler, and a 
  GPU backend implemented using the `ThrustRTC` Python interface to the NVRTC 
  runtime compilation library for CUDA.
Entirety of `PySDM` code base can be run without employing Numba or GPU,
  for instance for debugging or code-coverage analysis purposes.
 
The initial "v1" releases of `PySDM` outlined in a preceding JOSS paper
  (@Bartman_et_al_2022_JOSS) featured representation of the following 
  processes: condensational growth/evaporation, collisional growth,
  aqueous sulphur chemistry, as well as coupling of particle transport
  and vapour/heat budget with grid-discretised fluid flow.
This paper outlines following developments in the "v2" releases of `PySDM`
  including two new processes -- collisional breakup and immersion freezing -- 
  and a set of examples depicting the new functionalities
  using simulation setups described in literature.
Additionally, "v2" comes with enhanced support for adaptive timestepping.

# Background and statement of need

Processes occurring in atmospheric clouds pertain to the interplay of the dispersed-
  and continuous-phases of atmospheric flows.
The dispersed phase here refers to the aerosol particles ranging in size from
  nanometers to micrometers, the cloud droplets forming on these aerosol particles,
  the ice particles, and various types of hydrometeors.
The continuous phase denotes moist air which exchanges heat, moisture and momentum 
  with the particles.

One way of representing clouds in numerical fluid-dynamics simulations is to 
  model liquid and ice water content as continuous fields in space, despite their
  inherent particulate nature.
This however hinders representation of the diverse 
  physical and compositional characteristics of the particles which determine
  the initial stages of formation of droplets and ice particles.
Detailed information on the density and shape of particles is also essential
  to model particle collisions and aerodynamic interactions.
Thus, an apt modelling choice is to model the dispersed phase using 
  a particle-based approach.

In process-level models lacking any spatial dimensions, the particle-based approach
  is congruent with the so-called moving-sectional discretisation of the
  particle attribute space.
The attribute space spans such dimensions as droplet water content, soluble species mass and hygroscopicity, insoluble 
  material surface, etc.
The moving-sectional models can be traced back to the very beginnings of computational 
  studies of cloud microphysics (@Howell_1949).
Coupling particle-based/moving-sectional (Lagrangian) representation of the attribute space
  with gridded (Eulerian) fluid-flow dynamics, as in the SDM, can be traced back to early
  systems for simulating dispersal of atmospheric pollutants (@Lange_1978).
Such coupling implies inclusion of spatial location among 
  particle attributes.

One of notable traits of the particle-based representation is its suitability 
  for Monte-Carlo techniques in which each simulation represents
  just one possible realisation of the system evolution.
To this end, the attribute space can be randomly sampled at initialisation.
One of the constituting assumptions here is that one computational
  particle represents a (significant) multiplicity of modelled particles,
  hence the term super-particle (e.g., @Zannetti_1983) or super-droplet.

Despite numerous benefits of employing particle-based/moving-sectional representation
  (as opposed to continuous-field/fixed-bin approaches) for
  modelling atmospheric aerosols, clouds and precipitation, this technique 
  has long been considered inapplicable in three-dimensional atmospheric
  models (@Jacobson_2005, sect.~13.5).
This was due to limitations in representation of processes such as nucleation,
  and collisions which lead to
  appearance in the system of particles of sizes not representable without
  dynamically enlarging the particle state vector.
This has been solved by devising super-particle-number-conserving 
  Monte-Carlo schemes such as SDM (@Shima_et_al_2009).

`PySDM` features implementation of the original SDM scheme as formulated in 
  @Shima_et_al_2009 as well as several extension to it outlined in subsequent section.
The key motivation behind development of `PySDM` has been to offer the community a set of
  readily reusable building blocks for development and community dissemination 
  of extensions to SDM.
To this end, we strive to maintain strict separation of concerns and extensive unit
  test coverage in the project.
We continue to expand and maintain a set of examples demonstrating project features 
  through reproduction of results from literature.

# Summary of new features and examples in v2

@Bartman_et_al_2022_JOSS

@Bieli_et_al_2022

@Jokulsdottir_and_Archer_2016

@Alpert_and_Knopf_2016

@Shima_et_al_2020

@Rothenberg_and_Wang_2017

@Abdul_Razzak_and_Ghan_2000

@DeJong_et_al_2022

@Ruehl_et_al_2016

# Author contributions

EDJ led the formulation and implementation of the collisional breakup scheme with contributions from JBM.
PB led the formulation and implementation of the adaptive time-stepping schemes for diffusional and collisional growth.
KD contributed to setting up continuous integration workflows for the GPU backend.
ID, CES and AJ contributed to the CCN activation examples.
CES and RW contributed the representation of organics in surface-tension models and the relevant examples.
The immersion freezing representation code was developed by SA who also carried out the maintenance of the project.

# Acknowledgements

We thank Shin-ichiro Shima (University of Hyogo, Japan) for his continuous help and support in implementing SDM.
Part of the outlined developments was supported by the generosity of Eric and Wendy Schmidt (by recommendation of Schmidt Futures).
Development of ice-phase microphysics representation has been supported through 
grant no. DE-SC0021034 by the Atmospheric System Research Program and 
Atmospheric Radiation Measurement Program sponsored by the U.S. Department of Energy (DOE).

# References
