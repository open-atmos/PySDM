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
  as in @Jokulsdottir_and_Archer_2016, `PySDM` primarily supports atmospheric particle 
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

Atmospheric cloud processes involve a complex interplay of dispersed-phase particle
  processes and continuous-phase environmental flows. 
Microphysical particles range
  in size from nanometer-sized aerosols, to micron-scale cloud droplets and
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
  of the dispersed phase, making it an ideal choice to capture these physics.

The particle-based approach is congruent with the so-called moving-sectional 
  discretisation of the particle attribute space, which includes properties such
  as droplet size or water content, soluble species mass and aerosol hygroscopicity, 
  insoluble material surface, etc.
The approach is well-suited to Monte-Carlo techniques, which are themselves ideal for 
  representing inherently stochastic processes such as particle nulcation, collisions and breakup.
In the SDM, a core assumption is that one computational particle represents a 
   (significant) multiplicity 
  of modelled particles in order to make the modeling of a physical system attainable,
  hence the term super-particle (e.g., @Zannetti_1983) or super-droplet (@Shima_et_al_2009).

Equally important, the method's computational application hinges on the assumption that 
  the number of superparticles is conserved throughout the simulation.
The moving-sectional (or Lagrangian in attribute space) methods were long considered incomplete for three-dimensional atmospheric
  models (@Jacobson_2005, sect.~13.5), as certain processes such as nucleation and collisions 
  lead to appearance in the system of particles of sizes not representable without
  dynamically enlarging the particle state vector.
This challenge was solved by devising super-particle-number-conserving 
  Monte-Carlo schemes such as the SDM for collisions (@Shima_et_al_2009).
Enhancements included in v2 of `PySDM` address additional tracer-conserving representations
  of the droplet breakup process as described in (@deJong_et_al_2022), and the immersion
  freezing process.
In addition, we include enhanced support for adaptive time-stepping.
We continue to expand and maintain a set of examples demonstrating project features 
  through reproduction of results from literature.

The key motivation behind development of `PySDM` has been to offer the community a set of
  readily reusable building blocks for development and community dissemination 
  of extensions to particle-based microphysics models.
To this end, we strive to maintain strict modularity of the PySDM building blocks, separation of
  functionality and examples, and extensive unit test coverage in the project.
A user of the package might select from top-level physics options such as the simulation
  environment, particle processes, and output attributes without requiring a detailed understanding
  of the CPU and GPU underlying implementations at the superparticle level.
The separation of physics information from backend engineering is intended to make the
  software more approachable for both users and developers who wish to contribute to the
  scientific progress of particle-based methods for simulating atmospheric clouds.


# Summary of new features and examples in v2

## New PySDM Features: API in Brief
`PySDM` v2 includes support for three major enhancements. For a detailed example of running
  a SDM simulation, we refer to @Bartman_et_al_2022_JOSS. The following API examples
  can be added or substituted into the v1 API description to run a zero-dimensional box
  simulation using the new features.

### Collisional Breakup
The collisional breakup process represents the splitting of two colliding superdroplets
  into multiple fragments. 
It can be specified as an individual dynamic, as for coalescence in v1, or as a unified
  `collision` dynamic, in which the probability of breakup versus coalescence is sampled.

```python
from PySDM.dynamics.collisions import Collision
from PySDM.dynamics.collisions.collision_kernels import Golovin
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import ExpFrag
```
The rate of superdroplet collisions are specified by a collision kernel as in v1, and the
  breakup process requires two additional `dynamics` specifications: from `coalescence_efficiencies`
  (probability of coalescence occuring), `breakup_efficiencies` (probability of breakup occuring
  if not coalescence), and `breakup_fragmentations` (the number
  of fragments formed in the case of a breakup event). 
Specifying a breakup-only event requires only a collision kernel, fragmentation function, 
  and optional breakup efficiency.

```python
builder.add_dynamic(Collision(kernel=Golovin(b=1.5e3 / si.s), coalescence_efficiency=ConstEc(Ec=0.9),
                     breakup_efficiency=ConstEb(Eb=1.0), fragmentation_function=ExpFrag(scale=100*si.um**3)))
```

### Immersion Freezing
`lines of code for add_dynamic` and description of necessary physics specifications

### Adaptive time-stepping
The condensation and collision backends both support an adaptive time-stepping feature,
  which overwrites the user-specified environment time step. Adaptivity is specified as an additional
  keyword to the given dynamic: `builder.add_dynamic(Dynamic(**kwargs, adaptive=True))` and has
  a default value of `True`.

## Additional PySDM-examples
Write 1 paragraph on each example, maybe some figures. Main goals:
(1) Link back to the original JOSS paper
(2) Give a clear overview of what user can expect from playing with existing examples,
which are aimed at reproducing literature examples

BREAKUP
@Bieli_et_al_2022 - breakup
@DeJong_et_al_2022 - breakup **maybe** (skip for now)

### Immersion freezing 

This release of PySDM introduces representation of immersion freezing, 
  i.e. freezing contingent on the presence of insoluble ice nuclei immersed 
  in supercooled water droplets.
There are two alternative models implemented, in both cases the formulation
  is probabilistic and based on Poissonian model of heterogeneous freezing.
The two models embrace, so-called, singular and time-dependent approaches and
  are based on the formulation presented in @Shima_et_al_2020 and
  @Alpert_and_Knopf_2016, respectively.
In the singular model, the relevant introduced particle attribute is the freezing temperature
  which is randomly sampled at initialisation from an ice nucleation active sites (INAS) model;
  subsequently freezing occurs in a deterministic way upon encountering ambient 
  temperature that is lower than the particle's freezing temperature.
In the time-dependent model, the relevant introduced particle attribute is the insoluble
  material surface which is randomly sampled at initialisation; 
  freezing is triggered by evaluating probability of freezing at instantaneous
  ambient conditions and comparing it with a random number.
For the time-dependent model, the water Activity Based Immersion Freezing Model (ABIFM)
  of @Knopf_and_Alpert_2013 is used.
  
For validation of the the newly introduced immersion freezing models, a set of
  notebooks reproducing box-model simulations from @Alpert_and_Knopf_2016 was introduced
  to the PySDM-examples package.
A comparison of the time-dependent and singular models using the kinematic
  prescribed-flow environment introduced in PySDM v1 has been developed
  and is the focus of @Arabas_et_al_2022.

### ACTIVATION
@Rothenberg_and_Wang_2017 - pyrcel reproduction
@Abdul_Razzak_and_Ghan_2000 - activation compared to parameterization
@Ruehl_et_al_2016 - organics and influence on surface tension

### Adaptive timestepping (maybe)

Already in PySDM v1, adaptive time-stepping was gradualy introduced for two 
  of the represented microphysical processes: collisional and diffusional growth.
In both cases, the adaptivity controls bespoke developments.

Noteworthy, due to different ... GPU vs. CPU

For diffusional growth (condensation and evaporation of water), ..
- semi-implicit solution
- adaptivity control (theta)
- aim: improve performance
- the two tolerances 
- load balancing (sorting cells by ...)
- validation against SciPy solver which is available for CPU ...

For collisional growth (coalescence and breakup) ...
- adaptivity control (gamma, multiplicities)
- aim: reduce error
- load balancing
- GPU vs. CPU
- validation against analytic

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
