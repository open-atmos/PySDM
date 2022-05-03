---
title: 'PySDM v2: particle-based cloud microphysics in Python -- collisional breakup, immersion freezing and adaptive time-stepping'
date: 3 May 2022
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
  - name: Ryan&nbsp;X.&nbsp;Ward
    affiliation: "3"
    orcid: 0000-0003-2317-3310
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
  of the diverse phase, making it an ideal choice to capture these physics.
Moreover, the approach is well-suited to Monte-Carlo techniques, which are themselves ideal for 
  representing inherently stochastic processes such as particle collisions and breakup.
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
The additional necessary information can be imported via:
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
```python
bulder.add_dynamic(Freezing(singular=False))
```
TODO (Sylwester): attribute initialisation: freezing temperature for singular, immersed surface for time-dep
TODO (Sylwester): explain how to pass INAS or ABIFM constants

### Initialisation framework for aerosol composition
TODO (Clare)--will this be included in the next JOSS paper? How close is it?
Internal versus external mixture
Also include a brief note of why we use kappa * dry volume for coagulation logic

### Adaptive time-stepping
The condensation, collision, and displacement dynamics all support an adaptive time-stepping feature,
  which involves substepping within the user-specified environment time step.
Adaptivity is enabled by default and can be disabled by passing `False` as the value of optional `adaptive`
  keyword to the given dynamic, e.g. `builder.add_dynamic(Dynamic(**kwargs, adaptive=False))`.
The adaptive time-step controls are described in @Bartman_et_al_2022_adaptive and  
  are bespoke developments introduced in PySDM (partly already in version 1).
In the case of multi-dimensional environments, the adaptive time-stepping is aimed
  at adjusting the time-steps separately in each grid box (e.g., based
  on ambient supersaturation for condensation).
For CPU backend and the condensation dynamic, the adaptivity scheme features a load-balancing 
  logic ensuring that 
  in multi-threaded operation, grid cells with comparable substep count are handled
  simultaneously avoiding idle threads.
The dynamic load-balancing across threads can be switched off by setting the `schedule` 
  keyword parameter to a value of `"static"` when instantiating the `Condensation` dynamic
  (the default value is `"dynamic"`).
 
## Additional PySDM-examples
This release of `PySDM` is complemented by an additional suite of test cases in `PySDM-examples`
  which make use of the augmented functionality to reproduce results in the microphysics literature.

Write 1 paragraph on each example group, maybe some figures. Main goals:
(1) Link back to the original JOSS paper
(2) Give a clear overview of what user can expect from playing with existing examples,
which are aimed at reproducing literature examples

### Collisional Breakup
`PySDM` was recently used as a calibration tool to generate data for learning microphysics rate
  parameters in @Bieli_et_al_2022 (in review). 
Particles in a box environment undergo coalescence and breakup with a fixed coalescence 
  efficiency, and the moments of the distribution are used as training data. 
In addition, two figures from a forthcoming publication @DeJong_et_al_2022 that describes the
  physics and algorithm for superdroplet breakup are included. 
The first example (reproduced in \autoref{fig:readme_fig_1}), demonstrates the impact of including
  the breakup process on the particle size distribution, versus a coalescence-only case. 
The second similarly demonstrates the impact of the breakup process in a one-dimensional setup 
  based on the example of @Shipway_and_Hill_2012.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.6\textwidth]{deJong_fig1}
    \caption{Sample plot generated with the code snippets included in the paper.}
    \label{fig:dJ_fig_1}
\end{figure}


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

### Initialisation of multi-component internally or externally mixed aerosols 
The new aerosol initialisation framework allows flexible specification of multi-modal, multi-component
aerosol with arbitrary composition.
The `DryAerosolMixture` class takes a list of compounds and dictionaries specifying their molar masses,
densities, solubilities, and ionic dissociation numbers.
The must then specify the aerosol `modes` which are comprised of a `kappa` hygroscopicity value
and a `spectrum`.
The `kappa` is calculated by `PySDM` from the aerosol properties specified above in association with 
some specified `mass_fractions` dictionary.
A code snippet showing the creation of the aerosol for the `ARG2000` example is shown below.

```python
from PySDM.initialisation.aerosol_composition import DryAerosolMixture

class AerosolARG(DryAerosolMixture):
    def __init__(
        self,
        M2_sol: float = 0,
        M2_N: float = 100 / si.cm**3,
        M2_rad: float = 50 * si.nm,
    ):
        super().__init__(
            compounds=("(NH4)2SO4", "insoluble"),
            molar_masses={
                "(NH4)2SO4": 132.14 * si.g / si.mole,
                "insoluble": 44 * si.g / si.mole,
            },
            densities={
                "(NH4)2SO4": 1.77 * si.g / si.cm**3,
                "insoluble": 1.77 * si.g / si.cm**3,
            },
            is_soluble={"(NH4)2SO4": True, "insoluble": False},
            ionic_dissociation_phi={"(NH4)2SO4": 3, "insoluble": 0},
        )
        mass_fractions_mode1 = {"(NH4)2SO4": 1.0, "insoluble": 0.0}
        mass_fractions_mode2 = {"(NH4)2SO4": M2_sol, "insoluble": (1 - M2_sol)}
        self.modes = (
            {
                "kappa": self.kappa(mass_fractions_mode1),
                "spectrum": spectra.Lognormal(
                    norm_factor=100.0 / si.cm**3, m_mode=50.0 * si.nm, s_geom=2.0
                ),
            },
            {
                "kappa": self.kappa(mass_fractions_mode2),
                "spectrum": spectra.Lognormal(
                    norm_factor=M2_N, m_mode=M2_rad, s_geom=2.0
                ),
            },
        )
```

Code snippet to use the aerosol class. First create an instance of the aerosol, use it to calculate the total number of superdroplets given a prescribed number per mode and then create the builder object. The aerosol modes are iterated through to extract `kappa` and define the `kappa times dry volume` attribute. This `kappa times dry volume` is used because it is an extensive attribute of the superdroplets; or that the hygroscopicity of a particle is the volume-weighted average of the hygroscopicity of its individual components. Finally, before the simulation is run, the wet radii must be equilibrated based on the `kappa times dry volume`. 

```python
from PySDM_examples.Abdul_Razzak_Ghan_2000.aerosol import AerosolARG

aerosol = AerosolARG(M2_sol=sol2, M2_N=N2, M2_rad=rad2)
n_sd = n_sd_per_mode * len(aerosol.modes)

builder = Builder(backend=CPU(), n_sd=n_sd)

... (add dynamics) ...

for i, mode in enumerate(aerosol.modes):
    kappa, spectrum = mode["kappa"]["CompressedFilmOvadnevaite"], mode["spectrum"]

    ... (add other atributes) ...

    attributes["kappa times dry volume"] = np.append(
        attributes["kappa times dry volume"], v_dry * kappa
    )
r_wet = equilibrate_wet_radii(
    r_dry=builder.formulae.trivia.radius(volume=attributes["dry volume"]),
    environment=env,
    kappa_times_dry_volume=attributes["kappa times dry volume"],
)

... (run simulation) ...
```

Note: For the Abdul-Razzak and Ghan 2000 example we use the `CompressedFilmOvadnevaite` version of calculated `kappa` to indicate that only the soluble components of the aerosol contribute to the hygroscopicity, but the surface tension of the droplets is assumed still to be constant (that of pure water) and the `Constant` surface tension model is used there.

@Abdul_Razzak_and_Ghan_2000 - activation compared to parameterization
![ARG2000.](ARG_fig1.pdf){#fig:ARG2000 width="60%"}

### Surface-partitioning of organics to modify surface tension of droplets
In addition to the standard case of an assumed constant surface tension of water, three thermodynamic frameworks describing the surface-partitioning of organic species have been included in PySDM. These models describe the surface tension of a droplet as a function of the dry aerosol composition and the wet radius. An example of how to specify the surface tension formulation is shown below. The three additional thermodynamic frameworks have been implemented following @Ovad @Ruehl_et_al_2016, and Szyszkowski-Langmuir.

Code demonstrating how to create `formulae` objects using the different surface tension models.
```python
A = AerosolBetaCaryophylleneDark()
formulae_bulk = Formulae(
    surface_tension='Constant'
)
formulae_ovad = Formulae(
    surface_tension='CompressedFilmOvadnevaite',
    constants={
        'sgm_org': 35 * si.mN / si.m,
        'delta_min': 1.75 * si.nm
    }
)
formulae_ruehl = Formulae(
    surface_tension='CompressedFilmRuehl',
    constants={
        'RUEHL_nu_org': A.modes[0]['nu_org'],
        'RUEHL_A0': 115e-20 * si.m * si.m,
        'RUEHL_C0': 6e-7,
        'RUEHL_m_sigma': 0.3e17 * si.J / si.m**2,
        'RUEHL_sgm_min': 35 * si.mN / si.m
    }
)
formulae_sl = Formulae(
    surface_tension='SzyszkowskiLangmuir',
    constants={
        'RUEHL_nu_org': A.modes[0]['nu_org'],
        'RUEHL_A0': 115e-20 * si.m * si.m,
        'RUEHL_C0': 6e-7,
        'RUEHL_sgm_min': 35 * si.mN / si.m
    }
)
```

Using these different models for the surface-partitioning, we can demonstrate the effect variable surface tension has on the activation of aerosol with some organic fraction. The presence of the orgnaics both modifies the surface tension and the hygroscopicity, resulting sometimes in a Köhler curve with local minima features. Below is (psuedo-)code used to generate four Köhler curves for the same partially organic aerosol particle, just under different assumptions of surface-partitioning by the insoluble organic species.
```python
model = formulae.surface_tension.__name__
sigma = formulae.surface_tension.sigma(T, v_wet, v_dry, A.modes[0]['f_org'])
RH_eq = formulae.hygroscopicity.RH_eq(r_wet, T, A.modes[0]['kappa'][model], rd3, sigma)
plot(r_wet, (RH_eq - 1)*100)
```
![Köhler curves for aerosol under 4 assumptions of thermodynamic surface-partitioning of organic species.](surf_fig_kohler.pdf){#fig:kohler width="60%"}


### Adaptivity
TODO (Sylwester): @Bartman_et_al_2022_adaptive

# Author contributions

EDJ led the formulation and implementation of the collisional breakup scheme with contributions from JBM.
PB led the formulation and implementation of the adaptive time-stepping schemes for diffusional and collisional growth.
KD contributed to setting up continuous integration workflows for the GPU backend.
CES contributed the aerosol initialisation framework.
ID, CES, and AJ contributed to the CCN activation examples.
CES contributed the representation of surface-partitioning by organic aerosol and the relevant examples with contributions from RXW.
The immersion freezing representation code was developed by SA who also carried out the maintenance of the project.

# Acknowledgements

We thank Shin-ichiro Shima (University of Hyogo, Japan) for his continuous help and support in implementing SDM.
Part of the outlined developments was supported by the generosity of Eric and Wendy Schmidt (by recommendation of Schmidt Futures).
Development of ice-phase microphysics representation has been supported through 
grant no. DE-SC0021034 by the Atmospheric System Research Program and 
Atmospheric Radiation Measurement Program sponsored by the U.S. Department of Energy (DOE).
EDJ's contributions were made possible by support from the Department of Energy Computational Sciences Graduate Research Fellowship.

# References

