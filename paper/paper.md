---
title: 'New developments in PySDM and PySDM-examples v2: collisional breakup, immersion freezing, dry aerosol composition initialisation, and adaptive time-stepping'
date: 17 May 2022
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
  - name: Sajjad Azimi
    affiliation: "3"
    orcid: 0000-0002-6329-7775
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
# Summary

`PySDM` and the accompanying `PySDM-examples` packages are open-source modeling tools
  for computational studies of atmospheric clouds, aerosols, and precipitation. The
  project hinges on a particle-based modeling approach and Pythonic design and
  implementation. 
The eponymous `SDM` refers to the Super Droplet Method -- a 
  Monte-Carlo algorithm introduced in @Shima_et_al_2009 to represent the coagulation
  of droplets in modelling frameworks such as Large-Eddy Simulations (LES) of atmospheric
  flows. 
Recent efforts have culminated
  in a second release, which includes a variety of new processes for both liquid and ice-phase particles,
  performance enhancements such as adaptive time-stepping, as well as a broadened suite of 
  examples which demonstrate, test, and motivate the use of the SDM for cloud modeling research.


# Background and Statement of Need
The key motivation behind development of `PySDM` has been to offer the community an approachable
  readily reusable software for users and developers who wish to contribute to the
  scientific progress of particle-based methods for simulating atmospheric clouds.
To this end, we strive to maintain strict modularity of the PySDM building blocks, separation of
  functionality and examples, and extensive unit test coverage in the project.
A user of the package can select top-level options such as the simulation
  environment, particle processes, and output attributes without a detailed understanding
  of the CPU and GPU implementations at the superparticle level.

`PySDM` v1 featured representation of the following 
  processes: condensational growth/evaporation, collisional growth,
  aqueous sulphur chemistry, as well as coupling of particle transport
  and vapour/heat budget with grid-discretised fluid flow.
This paper outlines these subsequent developments in the "v2" releases of `PySDM`
  including three new processes (collisional breakup, immersion freezing, and surface-partitioning of organic aerosol components), 
  initialisation framework for aerosol size and composition,
  enhanced support for adaptive timestepping, and additional illustrative examples.

In v2 of the companion `PySDM-examples` package, we continue to expand and maintain 
  a set of examples demonstrating project features 
  through reproduction of results from literature.
The examples package has a fourfold role in the project.
First, it serves to guide users and developers through the package features.
Second, `PySDM-examples` has been used as educational material, offering
  interactive Jupyter notebooks suitable for hands-on demonstrations of basic cloud-physics
  simulations.
Third, inclusion of simulation scripts/notebooks pertaining to
  new research papers is intended to streamline assessment of the
  results by reviewers. Running simulations described in a paper can be done independently on a cloud-computing platform such as Google Colab.
Finally, we require new examples include a set of "smoke tests" in `PySDM`,
  which assert results against reference data to ensure that published results remain 
  reproducible with future developments.



# Summary of new features and examples in v2

For an example of running basic zero-dimensional
  simulations with `PySDM`, we refer to the project README.md file and the
  preceeding @Bartman_et_al_2022_JOSS JOSS paper.
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
  environment (based on @Shipway_and_Hill_2012) are reproduced.

## Immersion Freezing
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
For example, a single-mode aerosol class (`SimpleAerosol`) can be defined as follows.
```python
from PySDM.initialisation import spectra
from PySDM.initialisation.aerosol_composition import DryAerosolMixture
class SimpleAerosol(DryAerosolMixture):
    def __init__(self):
        super().__init__(
            compounds=("(NH4)2SO4", "NaCl"),
            molar_masses={"(NH4)2SO4": 132.14 * si.g / si.mole,
              "NaCl": 58.44 * si.g / si.mole},
            densities={"(NH4)2SO4": 1.77 * si.g / si.cm**3,
              "NaCl": 2.16 * si.g / si.cm**3},
            is_soluble={"(NH4)2SO4": True, "NaCl": True},
            ionic_dissociation_phi={"(NH4)2SO4": 3, "NaCl": 2},
        )
        self.modes = (
            {
                "kappa": self.kappa(
                  mass_fractions={"(NH4)2SO4": 0.7, "NaCl": 0.3}),
                "spectrum": spectra.Lognormal(
                    norm_factor=100.0 / si.cm**3,
                    m_mode=50.0 * si.nm, s_geom=2.0
                ),
            },
        )
```
The `aerosol` object can be used during initialisation to calculate the total number of 
  superdroplets given a prescribed number per mode, sample the size spectrum from the aerosol 
  `spectrum` property, and initialise the `kappa times dry volume` attribute using the `aerosol` 
  hygroscopicity property `kappa`.
The choice of `kappa times dry volume` as an extensive attribute ensures that, upon coalescence,
  the hygroscopicity of a resultant super-particle is the volume-weighted average of the hygroscopicity 
  of the coalescing super-particles.
This new aerosol initialisation framework is used in the new example that reproduces results from 
  @Abdul_Razzak_and_Ghan_2000, comparing these SDM results against the original bin implementation and a new 
  cloud microphysics method, as shown in \autoref{fig:ARG}).

![Activated aerosol fraction in Mode 1 as a function of aerosol number concentration in Mode 2, reproducing results from @Abdul_Razzak_and_Ghan_2000. The figure shows the results from `PySDM` in color with two definitions of activated fraction based on the critical supersaturation threshold (Scrit) or the critical volume threshold (Vcrit) compared against the parameterization developed in @Abdul_Razzak_and_Ghan_2000, as implemented in their paper (solid line) and as implemented in a new Julia model (CloudMicrophysics.jl, dashed line), as well as the results from a bin scheme employed in @Abdul_Razzak_and_Ghan_2000 (black dots).](ARG_fig1.pdf){#fig:ARG width="100%"}

## Surface-partitioning of organics to modify surface tension of droplets
`PySDM` v2 includes a new example demonstrating three new models for droplet surface tension.
The four surface tension options included in `PySDM`, which define the droplet surface tension as a function of dry aerosol composition and wet radius, are `'Constant'`, `'CompressedFilmOvadnevaite'` (@Ovadnevaite_et_al_2017), `'CompressedFilmRuehl'` (@Ruehl_et_al_2016), and `'SzyszkowskiLangmuir'` following the Szyszkowski-Langmuir equation.
Parameters for the three surface-partitioning models must be specified as shown in the example below, and a full comparison
  of surface-partitioning options can be found in the `Singer_Ward` example.
```python
from PySDM import Formulae
f = Formulae(
    surface_tension='CompressedFilmOvadnevaite',
    constants={
        'sgm_org': 35 * si.mN / si.m,
        'delta_min': 1.75 * si.nm
    }
)
```

## Adaptive time-stepping
In `PySDM` v2, the condensation, collision, and displacement dynamics 
  all support adaptive time-stepping logic,
  which involves substepping within the user-specified time step used for coupling
  with the environmental coupled-flow framework.
Adaptivity is enabled by default and can be disabled by passing `False` as the value of optional `adaptive`
  keyword to the given dynamic, i.e. `builder.add_dynamic(Dynamic(**kwargs, adaptive=False))`.
The adaptive time-step controls are described in a forthcomming @Bartman_et_al_2022_adaptive 
  publication and are bespoke developments introduced partialy in `PySDM` v1.
The time-step adaptivity aims both to reduce computational errors where the specified time step is
  longer than the timescale of the dynamic, as well as to reduce computational load by dynamically
  changing the time-step. 
This adaptive time-stepping applies separately in each grid box of a multidimensional environment,
  and includes a load-balancing logic for the CPU backend and condensation example to simultaneously
  handle grid cells with comparable substep count.
The dynamic load-balancing across threads can be switched off by setting the `schedule` 
  keyword parameter to a value of `"static"` when instantiating the `Condensation` dynamic
  (the default value is `"dynamic"`).


# Author contributions

EDJ led the formulation and implementation of the collisional breakup scheme with contributions from JBM.
PB led the formulation and implementation of the adaptive time-stepping.
KD contributed to setting up continuous integration workflows for the GPU backend. 
CES contributed the aerosol initialisation framework.
ID, CES, and AJ contributed to the CCN activation examples.
CES contributed the representation of surface-partitioning by organic aerosol and the relevant examples in consultation with RXW.
SA contributed to extensions and enhancement of the one-dimensional kinematic framework environment.
The immersion freezing representation code was developed by SA who also carried out the maintenance of the project.

# Acknowledgements

We thank Shin-ichiro Shima (University of Hyogo, Japan) for his continuous help and support in implementing SDM.
Part of the outlined developments was supported by the generosity of Eric and Wendy Schmidt (by recommendation of Schmidt Futures).
Development of ice-phase microphysics representation has been supported through 
grant no. DE-SC0021034 by the Atmospheric System Research Program and 
Atmospheric Radiation Measurement Program sponsored by the U.S. Department of Energy (DOE).
EDJ's contributions were made possible by support from the Department of Energy Computational Sciences Graduate Research Fellowship.

# References

