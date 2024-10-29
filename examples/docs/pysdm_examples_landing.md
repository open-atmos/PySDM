# Introduction
PySDM examples are bundled with PySDM and located in the examples subfolder.
They constitute a separate PySDM_examples Python package which is also available at PyPI.
The examples have additional dependencies listed in PySDM_examples package setup.py file.
Running the examples requires the PySDM_examples package to be installed.

We recommend you look through the example gallery below to see the examples in action.

# Example gallery
Unless stated otherwise the following examples solve the <mark>TODO</mark>:
$$ TODO $$

The examples are grouped by TODO.

## no-env examples depicting isotope-related formulae
| tags              | link                                       |
|:------------------|:-------------------------------------------|
| <mark>TODO</mark> | `PySDM_examples.Bolot_et_al_2013`          |
| TODO $$ TODO $$   | `PySDM_examples.Merlivat_and_Nief_1967`    |
| TODO              | `PySDM_examples.Miyake_et_al_1968`         |
| TODO              | `PySDM_examples.Van_Hook_1968 `            |
| TODO              | `PySDM_examples.Gedzelman_and_Arnold_1994` |
| TODO              | `PySDM_examples.Graf_et_al_2019`           |
| TODO              | `PySDM_examples.Lamb_et_al_2017`           |
| TODO              | `PySDM_examples.Pierchala_et_al_2022 `     |

## 0D box-environment coalescence and breakup
| tags | link                                       |
|:-----|:-------------------------------------------|
| TODO | `PySDM_examples.Shima_et_al_2009 `         |
| TODO | `PySDM_examples.Berry_1967 `               |
| TODO | `PySDM_examples.Bieli_et_al_2022 `         |
| TODO | `PySDM_examples.deJong_Mackay_et_al_2023 ` |
| TODO | `PySDM_examples.deJong_Azimi`              |

## 0D box-environment immersion freezing-only
| tags | link                                   |
|:-----|:---------------------------------------|
| TODO | `PySDM_examples.Alpert_and_Knopf_2016` |
| TODO | `PySDM_examples.Arabas_et_al_2023 `    |

## 0D parcel-environment condensation only
| tags | link                                          |
|:-----|:----------------------------------------------|
| TODO | `PySDM_examples.Arabas_and_Shima_2017 `       |
| TODO | `PySDM_examples.Yang_et_al_2018 `             |
| TODO | `PySDM_examples.Abdul_Razzak_Ghan_2000 `      |
| TODO | `PySDM_examples.Pyrcel `                      |
| TODO | `PySDM_examples.Lowe_et_al_2019`              |
| TODO | `PySDM_examples.Grabowski_and_Pawlowska_2023` |
| TODO | `PySDM_examples.Jensen_and_Nugent_2017 `      |

## 0D parcel-environment condensation/aqueous-chemistry
| tags | link                                        |
|:-----|:--------------------------------------------|
| TODO | `PySDM_examples.Kreidenweis_et_al_2003 `    |
| TODO | `PySDM_examples.Jaruga_and_Pawlowska_2018 ` |

## 0D parcel-environment condensation/freezing
| tags | link                                         |
|:-----|:---------------------------------------------|
| TODO | `PySDM_examples.Abade_and_Albuquerque_2024 ` |

## OD parcel-environment iterative framework mimicking removal of precipitation
| tags | link                                         |
|:-----|:---------------------------------------------|
| TODO | `PySDM_examples.Rozanski_and_Sonntag_1982  ` |

## 1D kinematic environment (prescribed-flow, single-column)
| tags | link                                       |
|:-----|:-------------------------------------------|
| TODO | `PySDM_examples.Shipway_and_Hill_2012 `    |
| TODO | `PySDM_examples.deJong_Azimi`              |
| TODO | `PySDM_examples.deJong_Mackay_et_al_2023 ` |

## 2D kinematic environment (prescribed-flow) Sc-mimicking aerosol collisional processing (warm-rain) examples
| tags | link                                 |
|:-----|:-------------------------------------|
| TODO | `PySDM_examples.Arabas_et_al_2015 `  |
| TODO | `PySDM_examples.Arabas_et_al_2023 `  |
| TODO | `PySDM_examples.Bartman_et_al_2021 ` |

\* - with comparison against analytic solution

# Installation
Since the examples package includes Jupyter notebooks (and their execution requires write access), the suggested install and launch steps are:

```
git clone https://github.com/open-atmos/PySDM.git
pip install -e PySDM
pip install -e PySDM/examples
jupyter-notebook PySDM/examples/PySDM_examples
```

Alternatively, one can also install the examples package from pypi.org by using
```
pip install PySDM-examples
```
