# Introduction

<img align="left" src="https://raw.githubusercontent.com/open-atmos/PySDM/main/docs/logos/pysdm_logo.svg" width=150 height=219 alt="pysdm logo" style="padding-right:1em">

PySDM examples are engineered as Jupyter Python notebooks supported by auxiliary Python commons
  that constitute a separate <mark>PySDM-examples</mark> Python package which is also
  <a href="https://pypi.org/p/PySDM-examples">available at PyPI</a>.
The examples have additional dependencies listed in
  <a href="https://github.com/open-atmos/PySDM/blob/main/examples/setup.py">PySDM-examples package setup.py file</a>.
Running the example Jupyter notebooks requires the <mark>PySDM-examples</mark> package to be pip-installed.
For installation instructions see [project docs homepage](https://open-atmos.github.io/PySDM).
Note that the Jupyter notebooks themselves are not included in the package wheels, but are included
  in the source .tar.gz file on PyPI, and are conveninently browsable on GitHub.
All notebooks feature header cells with badges enabling single-click execution on
  <a href="https://colab.research.google.com/">Google Colab</a> and on
  <a href="https://mybinder.org/">mybinder.org</a>.
The examples package is also used in the PySDM test suite.

# Example gallery

The examples are named referring to the journal paper they aim to reproduce simulations from
(or sometimes just where the inspiration originated from).
The list below groups all examples by the dimensionality and type of the employed modelling framework
("environment" in PySDM nomenclature, which can be: <mark>box</mark>, <mark>parcel</mark>,
<mark>single-column</mark>, <mark>2D prescribed flow</mark>), and by the set of physical processes simulated
(<mark>condensation</mark>, collisional <mark>coagulation</mark> and <mark>breakup</mark>,
<mark>drop freezing</mark>, <mark>isotopic fractionation</mark>, <mark>aqueous chemistry</mark>,
<mark>seeding</mark>, ...).

## 2D kinematic environment (prescribed-flow) mimicking Sc deck

The 2D prescribed-flow framework used here can be traced back to the work of
  <a href="https://doi.org/10.1007/978-1-935704-36-2_1">Kessler 1969 (section 3C)</a>.
  The setup employed in PySDM-examples, which mimics a stratiform cloud deck and features periodic horizontal boundary condition
    and vanishing flow at vertical boundaries, was introduced in <a href="https://doi.org/10.1175/JAS3980">Morrison and Grabowski (2007)</a>
    and later adopted for particle-based simulations in <a href="https://doi.org/10.5194/gmd-8-1677-2015">Arabas et al. (2015)</a>.
  It uses a non-devergent single-eddy flow field resulting in an updraft-downdraft pair in the domain.
  The flow field advects two scalar fields in an Eulerian way: water vapour mixing ratio
    and dry-air potential temperature.
  In PySDM-examples, the Eulerian advection is handled using the <a href="https://doi.org/10.21105/joss.03896">PyMPDATA</a> Numba-based
    implementation of the <a href="https://doi.org/10.1002/fld.1071">MPDATA numerical scheme of Smolarkiewicz (e.g., 2006)</a>.
  An animation depicting PySDM simulation capturing <mark>aerosol collisional processing</mark> by warm rain is shown below:

![animation](https://github.com/open-atmos/PySDM/wiki/files/kinematic_2D_example.gif)

Example notebooks:
- `PySDM_examples.Arabas_et_al_2015`
  - in-notebook GUI for setting up, running and interactively visualising the 2D kinematic simulations (with an option to export raw data to <mark>VTK</mark> and <mark>netCDF</mark> files, as well as to save plots to SVG or PDF):
  - "hello world" notebook depicting how to automate using Python the process of loading data and creating animations in <mark>Paraview</mark>
- `PySDM_examples.Arabas_et_al_2025`: adaptation of the 2D kinematic setup for studying <mark>glaciation</mark> of the cloud deck by <mark>immersion freezing</mark>

## 1D kinematic environment (prescribed-flow, single-column)

The single-column PySDM environment is a reimplementation of the <mark>Met Office KiD framework</mark>
  introduced in <a href="https://doi.org/10.1002/qj.1913">Shipway & Hill 2012</a>.
The framework features a single Eulerian-transported field of water vapour mixing ratio
  (vertical profile of potential temperature is fixed).
As in the 2D kinematic framework above, the Eulerian advection is handled by
  <a href="https://open-atmos.github.io/PyMPDATA/">PyMPDATA</a>.

Example notebooks:
- `PySDM_examples.Shipway_and_Hill_2012`: reproducing figures from the <a href="https://doi.org/10.1002/qj.1913">Shipway & Hill 2012</a> paper;
- `PySDM_examples.deJong_Mackay_et_al_2023`: reproducing figures from the <a href="https://doi.org/10.5194/gmd-16-4193-2023">de Jong et al. 2023</a> paper where the single-column
   framework was used to exemplify operation of the <mark>Monte-Carlo collisional breakup scheme</mark> in PySDM (scheme introduced in that paper).

## OD/1D iterative parcel/column environment mimicking removal of precipitation

This framework uses a parcel model with removal of precipitation for analysis,
iterative equilibration, the isotopic composition of the water vapour and
rain water in a column of air (no Eulerian transport, only iterative passage of a parcel through the column).

`PySDM_examples.Rozanski_and_Sonntag_1982`: bulk microphysics example (i.e. single super droplet) with
deuterium and heavy-oxygen <mark>water isotopologues</mark> featured.

## 0D parcel environment

The parcel framework implemented in PySDM uses a hydrostatic profile and adiabatic mass and energy conservation
  to drive evolution of thermodynamic state and microphysical properties of particles.

Example notebooks include:
- condensation only
  - `PySDM_examples.Arabas_and_Shima_2017`: monodisperse particle spectrum, activation/deactivation cycle
  - `PySDM_examples.Yang_et_al_2018`: polydisperse particle spectrum, activation/deactivation cycles
  - `PySDM_examples.Abdul_Razzak_Ghan_2000`: polydisperse activation, comparison against <mark>GCM parameterisation</mark>
  - `PySDM_examples.Pyrcel`: polydisperse activation, mimicking example test case from <mark>Pyrcel</mark> documentation
  - `PySDM_examples.Lowe_et_al_2019`: externally mixed polydisperse size spectrum with <mark>surface-active organics</mark> case
  - `PySDM_examples.Grabowski_and_Pawlowska_2023`: polydisperse activation, focus on <mark>ripening</mark>
  - `PySDM_examples.Jensen_and_Nugent_2017`: polydisperse activation featuring <mark>giant CCN</mark>
- condensation and aqueous-chemistry
  - `PySDM_examples.Kreidenweis_et_al_2003`: <mark>Hoppel gap</mark> simulation setup (i.e. depiction of evolution of aerosol mass spectrum from a monomodal to bimodal due to aqueous‐phase SO2 oxidation)
  - `PySDM_examples.Jaruga_and_Pawlowska_2018`: exploration of numerical convergence using the above Hoppel-gap simulation setup

The parcel environment is also featured in the <a href="https://open-atmos.github.io/PySDM/PySDM.html#tutorials">PySDM tutorials</a>.

## 0D box environment

The box environment is void of any spatial or thermodynamic context, it constitutes the most basic framework.

Example notebooks include:

- coalescence only:
  - `PySDM_examples.Shima_et_al_2009`: using <mark>Golovin additive kernel</mark> for comparison against analytic solution, featuring interactive in-notebook interface for selecting simulation parameters
  - `PySDM_examples.Berry_1967`: examples using geometric, hydrodynamic and electric-field collision kernels
- coalescence and breakup:
  - `PySDM_examples.Bieli_et_al_2022`: evolution of moments under collisional growth and breakage
  - `PySDM_examples.deJong_Mackay_et_al_2023`: validation of the breakup scheme against analytical solutions from <a href="https://doi.org/10.1175/1520-0469(1982)039%3C1317:ASMOPC%3E2.0.CO;2">Srivastava 1982</a>
- immersion freezing only:
  - `PySDM_examples.Alpert_and_Knopf_2016`: stochastic immersion freezing with monodisperse vs. lognormal immersed surface areas
  - `PySDM_examples.Arabas_et_al_2025`: comparison of time-dependent and singular immersion freezing schemes

The box environment is also featured in the <a href="https://open-atmos.github.io/PySDM/PySDM.html#tutorials">PySDM tutorials</a>.

## examples depicting isotope-related formulae (without any simulation context)
- <mark>equilibrium isotopic fractionation</mark> formulae:
  - `PySDM_examples.Lamb_et_al_2017`
  - `PySDM_examples.Bolot_et_al_2013`
  - `PySDM_examples.Merlivat_and_Nief_1967`
  - `PySDM_examples.Van_Hook_1968`
  - `PySDM_examples.Graf_et_al_2019`
- <mark>Rayleigh fractionation</mark>:
  - `PySDM_examples.Pierchala_et_al_2022`: reproducing model plots for a <mark>triple-isotope</mark> lab study, including <mark>kinetic fractionation</mark>
  - `PySDM_examples.Gonfiantini_1986`: flat-surface evaporation at different humidities for D and <sup>18</sup>O
- isotopic relaxation timescale:
  - `PySDM_examples.Miyake_et_al_1968`: incl. comparison of different <mark>ventilation</mark> parameterisations
  - `PySDM_examples.Bolin_1958`
- below-cloud <mark>kinetic fractionation</mark>:
  - `PySDM_examples.Gedzelman_and_Arnold_1994`

## examples depicting extraterrestrial clouds (formulae-only, no simulations yet)
- Titan (methane clouds):
  - `PySDM_examples.Toon_et_al_1980`
