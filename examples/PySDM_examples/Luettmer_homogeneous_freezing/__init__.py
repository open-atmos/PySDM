"""
Homogeneous freezing example

fig_1_2_3.ipynb:
.. include:: ./fig_1_2_3.ipynb

fig_4_5_6_S3_S4.ipynb:
.. include:: ./fig_4_5_6_S3_S4.ipynb

fig_S1_S2.ipynb:
.. include:: ./fig_S1_S2.ipynb

simple_homogenous_freezing_example.ipynb:
.. include:: ./simple_homogenous_freezing_example.ipynb
"""

from .settings import Settings
from .simulation import Simulation
from .plot import (
    plot_thermodynamics_and_bulk,
    plot_freezing_temperatures_histogram,
    plot_freezing_temperatures_histogram_allinone,
    plot_freezing_temperatures_2d_histogram_seaborn,
)
