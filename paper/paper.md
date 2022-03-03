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
`PySDM` and the accompanying `PySDM-examples` packages are open-source modeling tools
  for computational studies of atmospheric clouds, aerosols, and precipitation. The
  project hinges on a particle-based modeling approach and Pythonic design and
  implementation. 
The eponymous `SDM` refers to the Super Droplet Method -- a 
  Monte-Carlo algorithm introduced in @Shima_et_al_2009 to represent the coagulation
  of droplets in modelling frameworks such as Large-Eddy Simulations (LES) of atmospheric
  flows. 
While the SDM has been applied to additional systems such as oceanic particles
  as in @Jokulsdottir_and_Archer_2016, `PySDM` supports atmospheric particle 
  processes relevant to cloud particles and precipitation of hydrometeors.

`PySDM` is implemented modularly in Python with two alternative parallel backends: 
  a multi-threaded CPU implementation using the Numba Just-In-Time (JIT) compiler, 
  and a GPU implementation using the `ThrustRTC` Python interface to the NVRTC 
  runtime compilation library for CUDA. 
The entire `PySDM` codebase can also be 
  run without the Numba or GPU parallel backends (for debugging or code-coverage 
  analysis, for instance).
 
The initial "v1" releases of `PySDM` outlined in a preceding JOSS paper
  (@Bartman_et_al_2022_JOSS) featured representation of the following 
  processes: condensational growth/evaporation, collisional growth,
  aqueous sulphur chemistry, as well as coupling of particle transport
  and vapour/heat budget with grid-discretised fluid flow.
This paper outlines subsequent developments in the "v2" releases of `PySDM`
  including two new processes (collisional breakup and immersion freezing), 
  enhanced support for adaptive timestepping, and examples which illustrate the 
  new functionalities using simulation frameworks described in the scientific 
  literature.

# Background and statement of need

TODO: include information about the new processes (breakup and immersion freezing)
  and conservation of superparticle number -- reference to Jacobson statement
  on why NOT to use superparticles

STORY: moving sectional/motivation --> importance of superparticle conservation -->
  new processes

Atmospheric cloud processes involve a complex interplay of dispersed-phase and 
  continuous-phase flows. 
In the dispersed phase, microphysical particles range
  range in size from nanometer-sized aerosols, to micron-scale cloud droplets and
  ice particles that form on these aerosols, to millimeter and larger sized 
  hydrometeors. 
These particles interact with each other as well as with the 
  continuous phase moist air environment through exchange of heat, moisture,
  and momentum.

Traditional methods of representing clouds in numerical fluid-dynamics simulations
  model liquid and ice water content as continuous fields in space, using a mean
  field approximation for the particle populations.
This reductionist representation comes at the cost of the diverse physical 
  and compositional characteristics of the particles, which frequently determine
  the initial stages of formation of droplets and ice particles.
Detailed information regarding the density and shape of particles is also essential
  for modeling particle collisions and aerodynamic interactions.
A particle-based approach has the benefit of retaining the diverse characteristics
  of the diverse phase, making it an ideal choice to capture these physics.

In process-level models lacking any spatial dimensions (a "box" model), 
  the particle-based approach is congruent with the so-called moving-sectional 
  discretisation of the particle attribute space.
The attribute space includes properties such as droplet size or water content, 
  soluble species mass and aerosol hygroscopicity, insoluble material surface, etc.
Moving-sectional models can be traced back to the very beginnings of computational 
  studies of cloud microphysics (@Howell_1949).
Coupling particle-based/moving-sectional (Lagrangian) representation of the attribute space
  with gridded (Eulerian) fluid-flow dynamics with spatial dimensions, as in the SDM,
  can be traced back to early
  systems for simulating dispersal of atmospheric pollutants (@Lange_1978).
Such coupling implies inclusion of spatial location among 
  particle attributes.

A notable traits of the particle-based representation is its suitability 
  for Monte-Carlo techniques, in which each simulation represents
  just one possible realisation of the system evolution.
This Monte-Carlo approach is ideal for representing processes such as particle
  collisions and breakup, which are inherently stochastic at atmospherically
  relevant scales.
Upon initialisation, the attribute space can be randomly sampled to generate a
  representative populuation of computational particles. 
In the SDM, a core
  assumption is that one computational particle represents a (significant) multiplicity 
  of modelled particles in order to make the modeling of a physical system attainable,
  hence the term super-particle (e.g., @Zannetti_1983) or super-droplet.

Despite the numerous benefits of particle-based/moving-sectional representations
  (as opposed to continuous-field/fixed-bin approaches) for
  modelling atmospheric aerosols, clouds and precipitation, these techniques
  were long considered incomplete for three-dimensional atmospheric
  models (@Jacobson_2005, sect.~13.5).
Limitations include processes such as nucleation and collisions 
  which lead to appearance in the system of particles of sizes not representable without
  dynamically enlarging the particle state vector.
This has been solved by devising super-particle-number-conserving 
  Monte-Carlo schemes such as SDM (@Shima_et_al_2009), as well as a new conservative
  scheme for particle based breakup in the SDM (@DeJong_et_al_2022).

`PySDM` features implementation of the original SDM scheme as formulated in 
  @Shima_et_al_2009 as well as several extensions, which are outlined in subsequent sections.
The key motivation behind development of `PySDM` has been to offer the community a set of
  readily reusable building blocks for development and community dissemination 
  of extensions to SDM.
To this end, we strive to maintain strict modularity of the SDM building blocks, separation of
  functionality and examples, and extensive unit test coverage in the project.
The separation of physics information from backend engineering for GPU and CPU applications
  is intended to enhance and accelerate continued scientific development of the SDM and examples.
(EDJ: clean up this statement to better communicate; maybe use an example)
We continue to expand and maintain a set of examples demonstrating project features 
  through reproduction of results from literature.

# Summary of new features and examples in v2

## New PySDM Features: API in Brief

Breakup `lines of code for add_dynamic` and description of necessary physics specifications

Immersion freezing `lines of code for add_dynamic` and description of necessary physics specifications

Adaptive time-stepping `code for how to specify adaptivity` and mention external 
  scipy solver for condensation

Reference that the above can be run in the same framework (specify environment, etc.) as in v1

## Additional PySDM-examples
Write 1 paragraph on each example, maybe some figures. Main goals:
(1) Link back to the original JOSS paper
(2) Give a clear overview of what user can expect from playing with existing examples,
which are aimed at reproducing literature examples

BREAKUP
@Bieli_et_al_2022 - breakup
@DeJong_et_al_2022 - breakup **maybe** (skip for now)

IMMERSION FREEZING
@Alpert_and_Knopf_2016 - immersion freezing with time-dependent model
@Shima_et_al_2020 - immersion freezing with singular model

ACTIVATION
@Rothenberg_and_Wang_2017 - pyrcel reproduction
@Abdul_Razzak_and_Ghan_2000 - activation compared to parameterization
@Ruehl_et_al_2016 - organics and influence on surface tension

ADAPTIVITY (slayoo todo) -- **maybe** (skip for now)
@Bartmann_TBD - adaptive vs. nonadaptive for condensation

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
EDJ's contributions were made possible by support from the Department of Energy Computational Sciences Graduate Research Fellowship.

# References
