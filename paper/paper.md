---
title: 'New developments in PySDM and PySDM-examples v2: collisional breakup, immersion freezing, dry aerosol composition initialisation, and adaptive time-stepping'
date: 16 May 2022
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
  - name: Emily de Jong
    affiliation: "1"
    orcid: 0000-0002-5310-4554
  - name: Piotr Bartman
    orcid: 0000-0003-0265-6428
    affiliation: "2"
  - name: Kacper Derlatka
    affiliation: "2"
  - name: Isabella Dula
    affiliation: "3"
  - name: Anna Jaruga
    affiliation: "3"
    orcid: 0000-0003-3194-6440
  - name: J. Ben Mackay
    affiliation: "3"
    orcid: 0000-0001-8677-3562
  - name: Clare E. Singer
    orcid: 0000-0002-1708-0997
    affiliation: "3"
  - name: Ryan X. Ward
    affiliation: "3"
    orcid: 0000-0003-2317-3310
  - name: Sylwester Arabas
    orcid: 0000-0003-2361-0082
    affiliation: "4,2"
affiliations:
 - name: Department of Mechanical and Civil Engineering, California Institute of Technology, Pasadena, CA, USA
   index: 1
 - name: Faculty of Mathematics and Computer Science, Jagiellonian University, Kraków, Poland
   index: 2
 - name: Department of Environmental Science and Engineering, California Institute of Technology, Pasadena, CA, USA
   index: 3
 - name: University of Illinois at Urbana-Champaign, Urbana, IL, USA
   index: 4
bibliography: paper.bib

---

# Background and Statement of Need
`PySDM` and the accompanying `PySDM-examples` packages are open-source modeling tools
  for computational studies of atmospheric clouds, aerosols, and precipitation. The
  project hinges on a particle-based modeling approach and Pythonic design and
  implementation. 
The eponymous `SDM` refers to the Super Droplet Method -- a 
  Monte-Carlo algorithm introduced in @Shima_et_al_2009 to represent the coagulation
  of droplets in modelling frameworks such as Large-Eddy Simulations (LES) of atmospheric
  flows. 
The key motivation behind development of `PySDM` has been to offer the community an approachable
  readily reusable software for users and developers who wish to contribute to the
  scientific progress of particle-based methods for simulating atmospheric clouds.
To this end, we strive to maintain strict modularity of the PySDM building blocks, separation of
  functionality and examples, and extensive unit test coverage in the project.
A user of the package might select from top-level physics options such as the simulation
  environment, particle processes, and output attributes without requiring a detailed understanding
  of the CPU and GPU underlying implementations at the superparticle level.

`PySDM` v1 featured representation of the following 
  processes: condensational growth/evaporation, collisional growth,
  aqueous sulphur chemistry, as well as coupling of particle transport
  and vapour/heat budget with grid-discretised fluid flow.
Recent efforts and expanded collaboration with the scientific user base of `PySDM` have culminated
  in a second release, which includes a variety of new processes for both warm and ice-phase particles,
  performance enhancements such as adaptive time-stepping, as well as a broadened suite of 
  examples which demonstrate, test, and motivate the use of the SDM for cloud modeling research.
This paper outlines these subsequent developments in the "v2" releases of `PySDM`
  including two new processes (collisional breakup and immersion freezing), 
  initialisation framework for aerosol size and composition,
  enhanced support for adaptive timestepping, and examples which illustrate the 
  new functionalities using simulation frameworks described in the scientific 
  literature.

In v2 of the companion `PySDM-examples` package, we continue to expand and maintain 
  a set of examples demonstrating project features 
  through reproduction of results from literature.
The examples package has a fourfold role in the project.
First, it serves to guide the users and the developers through the package features.
Second, `PySDM-examples` has been used as educational material, offering
  interactive Jupyter notebooks suitable for hands-on demonstrations of basic cloud-physics
  simulations without exposing students to the technicalities of scientific coding.
Third, inclusion within `PySDM-examples` of simulation scripts/notebooks pertaining to
  newly submitted research papers is intended to streamline assessment of the
  results by reviewers. Running simulations described in a paper can be done independently, 
  anonymously and without technical or legal obstacles--in many cases just with a 
  single-click on a link to a cloud-computing platform such as Google Colab.
Last but not least, we encourage developers of new examples
  to include set of so-called ``smoke tests'' in `PySDM`,
  which assert results against reference data to ensure that published results remain 
  reproducible despite ongoing developments.



# Summary of new features and examples in v2

For an example of running basic zero-dimensional
  simulations with `PySDM`, we refer to the project README.md file and the
  precedig @Bartman_et_al_2022_JOSS JOSS paper.
The following code snippets demonstrating new elements of `PySDM` API 
  can be added or substituted into the v1 API description to run 
  simulation using the new features.

## Collisional Breakup
The collisional breakup process represents the splitting of two colliding superdroplets
  into multiple fragments.
It can be specified as an individual dynamic, as for coalescence in v1, or as a unified
  `collision` dynamic, in which the probability of breakup versus coalescence is sampled.
The additional `PySDM` components used in the example below can be imported via:
```python
from PySDM.dynamics.collisions import Collision
from PySDM.dynamics.collisions.collision_kernels import Golovin
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import ExponFrag
```
The rate of superdroplet collisions are specified by a collision kernel, and the
  breakup process requires two additional `dynamics` specifications: `coalescence_efficiencies`
  (probability of coalescence occuring), `breakup_efficiencies` (probability of breakup occuring
  if not coalescence), and `breakup_fragmentations` (the number
  of fragments formed in the case of a breakup event). 

```python
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.physics import si

builder = Builder(backend=CPU(), n_sd=100)
builder.set_environment(Box(dv=1 * si.m**3, dt=1 * si.s))
builder.add_dynamic(Collision(
  collision_kernel=Golovin(b=1.5e3 / si.s),
  coalescence_efficiency=ConstEc(Ec=0.9),
  breakup_efficiency=ConstEb(Eb=1.0),
  fragmentation_function=ExponFrag(scale=100*si.um**3)
  ))
```

In `PySDM-examples`, we reproduce results from two forthcoming publications.
In @Bieli_et_al_2022 (in review), `PySDM` results from collisional coalescence and breakup 
  were used as a calibration tool 
  for learning microphysics rate parameters.
In @DeJong_et_al_2022, the physics and algorithm for superdroplet breakup are described,
  and results demonstrating the impact of breakup on cloud properties in a box and 1D
  environment (based on @Shipway_and_Hill_2012) are reproduced, as in \autoref{fig:dJ_fig_1}).

![Particle size distribution using collisions, with and without breakup process, as is the focus of @DeJong_et_al_2022](deJong_fig1.pdf){#fig:dJ_fig_1 width="100%"}

### Immersion Freezing
This release of `PySDM` introduces representation of immersion freezing, 
  i.e. freezing contingent on the presence of insoluble ice nuclei immersed 
  in supercooled water droplets.
There are two alternative models implemented: the singular approach presented in 
  @Shima_et_al_2020, and the time-dependent approach of @Alpert_and_Knopf_2016.
For the time-dependent model, the water Activity Based Immersion Freezing Model (ABIFM)
  of @Knopf_and_Alpert_2013 is used.
The dynamic is introduced by specifying whether a singular model is used, and additional particle
  attributes must be initialised accordingly.
```python
from PySDM.dynamics import Freezing
builder.add_dynamic(Freezing(singular=False))
```

For validation of the the newly introduced immersion freezing models, a set of
  notebooks reproducing box-model simulations from @Alpert_and_Knopf_2016 was introduced
  to the `PySDM-examples` package using the kinematic prescribed-flow environment 
  introduced in `PySDM` v1.
A comparison of the time-dependent and singular models using this setup is the focus of @Arabas_et_al_2022.

## Initialisation of multi-component internally or externally mixed aerosols 
The new aerosol initialisation framework allows flexible specification of multi-modal, multi-component
  aerosol with arbitrary composition.
The `DryAerosolMixture` class takes a list of compounds and dictionaries specifying their molar masses,
  densities, solubilities, and ionic dissociation numbers.
The user must then specify the aerosol `modes` which are comprised of a `kappa` hygroscopicity value, 
  calculated from the molecular components and their associated `mass_fractions`,
  and a dry aerosol size `spectrum`.
For example, the aerosol for the @Abdul_Razzak_and_Ghan_2000 example,
  reproduced in `PySDM-examples`, can be specified as follows:

```python
from PySDM.initialisation import spectra
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
        self.modes = (
            {
                "kappa": self.kappa(mass_fractions={
                    "(NH4)2SO4": 1.0,
                    "insoluble": 0.0
                }),
                "spectrum": spectra.Lognormal(
                    norm_factor=100.0 / si.cm**3,
                    m_mode=50.0 * si.nm, s_geom=2.0
                ),
            },
            {
                "kappa": self.kappa(mass_fractions={
                    "(NH4)2SO4": M2_sol,
                    "insoluble": (1 - M2_sol)
                }),
                "spectrum": spectra.Lognormal(
                    norm_factor=M2_N, m_mode=M2_rad, s_geom=2.0
                ),
            },
        )
```

The `aerosol` object is then used in initialisation to calculate the total number of superdroplets 
  given a prescribed number per mode and then create the builder object.
The aerosol modes are iterated through to extract `kappa` and define the `kappa times dry volume` attribute.
The choice of `kappa times dry volume` as particle extensive attribute ensures that, upon coalescence,
  the hygroscopicity parameter kappa of a resultant super-particle is the volume-weighted average of the hygroscopicity 
  of the coalescing super-particles.
Finally, before a simulation is run, the wet radii must be equilibrated with ambient water vapour saturation 
  based on the `kappa times dry volume`.
Below is a code demonstrating how to initialize an aerosol population using the `AerosolARG` class
  defined above, as employed in the corresponding `PySDM-examples` module for @Abdul_Razzak_and_Ghan_2000.

```python
import numpy as np
from PySDM.initialisation.sampling import spectral_sampling

aerosol = AerosolARG(M2_sol=0.5, M2_N=1000 / si.cm**3, M2_rad=50 * si.nm)
n_sd_per_mode = 20
builder = Builder(backend=CPU(), n_sd=n_sd_per_mode * len(aerosol.modes))
attributes = {
    k: np.empty(0)
    for k in ("dry volume", "kappa times dry volume", "n")
}
for i, mode in enumerate(aerosol.modes):
    kappa = mode["kappa"]["CompressedFilmOvadnevaite"]
    sampler = spectral_sampling.ConstantMultiplicity(mode["spectrum"])
    r_dry, concentration = sampler.sample(n_sd_per_mode)
    v_dry = builder.formulae.trivia.volume(radius=r_dry)
    attributes["kappa times dry volume"] = np.append(
        attributes["kappa times dry volume"], v_dry * kappa
    )
```
![Activated aerosol fraction using various surface tension models, reproducing results from @Abdul_Razzak_and_Ghan_2000.](ARG_fig1.pdf){#fig:ARG width="100%"}

### Surface-partitioning of organics to modify surface tension of droplets

The new release of `PySDM` additionally includes a new suite of examples demonstrating three additional thermodynamic 
  frameworks for the surface-partitioning of organic species, based on a forthcoming publication
  (@Singer_Ward_2022).
The three additional thermodynamic frameworks have been implemented following 
  @Ovadnevaite_et_al_2017, @Ruehl_et_al_2016, and using the Szyszkowski-Langmuir equation.
Surface tension is as a function of the dry aerosol composition and the wet radius, and each
  framework takes a set of thermodynamic parameters, as demonstrated below.
The `aerosol` object is an instance of a class inheriting from the `DryAerosolMixture` base class.

```python
from PySDM import Formulae
from PySDM_examples.Singer_Ward.aerosol import AerosolBetaCaryophylleneDark
aerosol = AerosolBetaCaryophylleneDark()

models = {
    Formulae(surface_tension='Constant'),
    Formulae(
        surface_tension='CompressedFilmOvadnevaite',
        constants={
            'sgm_org': 35 * si.mN / si.m,
            'delta_min': 1.75 * si.nm
        }
    ),
    Formulae(
        surface_tension='CompressedFilmRuehl',
        constants={
            'RUEHL_nu_org': aerosol.modes[0]['nu_org'],
            'RUEHL_A0': 115e-20 * si.m * si.m,
            'RUEHL_C0': 6e-7,
            'RUEHL_m_sigma': 0.3e17 * si.J / si.m**2,
            'RUEHL_sgm_min': 35 * si.mN / si.m
        }
    ),
    Formulae(
        surface_tension='SzyszkowskiLangmuir',
        constants={
            'RUEHL_nu_org': aerosol.modes[0]['nu_org'],
            'RUEHL_A0': 115e-20 * si.m * si.m,
            'RUEHL_C0': 6e-7,
            'RUEHL_sgm_min': 35 * si.mN / si.m
        }
    )
}
```

In @Singer_Ward_2022, whose results are included in `PySDM-examples`, these different models of surface-partitioning
  are compared to demonstrate the effect of variable surface tension on the activation of aerosol with some organic fraction.

![Köhler curves for aerosol under 4 assumptions of thermodynamic surface-partitioning of organic species.](Singer_fig1_kohler.pdf){#fig:kohler width="100%"}

### Adaptive time-stepping

In `PySDM` v2, the condensation, collision, and displacement dynamics 
  all support adaptive time-stepping logic,
  which involves substepping within the user-specified time step used for coupling
  with the environmental coupled-flow framework.
Adaptivity is enabled by default and can be disabled by passing `False` as the value of optional `adaptive`
  keyword to the given dynamic, i.e. `builder.add_dynamic(Dynamic(**kwargs, adaptive=False))`.
The adaptive time-step controls are described in a forthcomming @Bartman_et_al_2022_adaptive 
  publication and are bespoke developments introduced in PySDM (partly already in version 1).
In the case of collisions, the time-step adaptivity is aimed at eliminating errors
  associated with multiple coalescence events within a timestep.
In the case of condensation, the time-step adaptivity is aimed at reducing computational
  load by coupling the time-step length choice with ambient supersaturation leading
  to using longer time-steps in cloud-free regions and shorter time-steps in regions
  where drople [de]activation or rain evaporation occurs.
In the case of displacement, the time-step adaptivity is aimed at obeying a given tolerance
  in integration of the super-particle trajectories, and the error measure is constructed
  by comparing implicit- and explicit-Euler solutions.

In multi-dimensional environments, adaptive time-stepping is aimed
  at adjusting the time-steps separately in each grid box (e.g., based
  on ambient supersaturation for condensation).
For CPU backend and the condensation dynamic, the adaptivity scheme features a load-balancing 
  logic which ensures that 
  in multi-threaded operation, grid cells with comparable substep count are handled
  simultaneously avoiding idle threads.
The dynamic load-balancing across threads can be switched off by setting the `schedule` 
  keyword parameter to a value of `"static"` when instantiating the `Condensation` dynamic
  (the default value is `"dynamic"`).


# Author contributions

EDJ led the formulation and implementation of the collisional breakup scheme with contributions from JBM.
PB led the formulation and implementation of the adaptive time-stepping schemes for diffusional and collisional growth.
KD contributed to setting up continuous integration workflows for the GPU backend. 
CES contributed the aerosol initialisation framework.
ID, CES, and AJ contributed to the CCN activation examples.
CES contributed the representation of surface-partitioning by organic aerosol and the relevant examples in consultation with RXW.
The immersion freezing representation code was developed by SA who also carried out the maintenance of the project.

# Acknowledgements

We thank Shin-ichiro Shima (University of Hyogo, Japan) for his continuous help and support in implementing SDM.
Part of the outlined developments was supported by the generosity of Eric and Wendy Schmidt (by recommendation of Schmidt Futures).
Development of ice-phase microphysics representation has been supported through 
grant no. DE-SC0021034 by the Atmospheric System Research Program and 
Atmospheric Radiation Measurement Program sponsored by the U.S. Department of Energy (DOE).
EDJ's contributions were made possible by support from the Department of Energy Computational Sciences Graduate Research Fellowship.

# References

