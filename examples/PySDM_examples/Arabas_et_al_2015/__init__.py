# pylint: disable=invalid-name
"""
2D prescribed-flow case extended with Paraview visualisation with spin-up logic from
[Arabas et al. 2015](http://doi.org/10.5194/gmd-8-1677-2015)

gui.ipynb:
.. include:: ./gui.ipynb.badges.md

paraview_hello_world.ipynb:
.. include:: ./paraview_hello_world.ipynb.badges.md
"""
from .settings import Settings
from .spin_up import SpinUp
