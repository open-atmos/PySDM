"""
Homogeneous freezing example

hom_freezing.ipynb:
.. include:: ./hom_freezing.ipynb
"""

from .settings import Settings
from .simulation import Simulation
from .plot import (
    plot_thermodynamics_and_bulk,
    plot_freezing_temperatures_histogram,
    plot_freezing_temperatures_2d_histogram,
    plot_freezing_temperatures_2d_histogram_seaborn,
)
