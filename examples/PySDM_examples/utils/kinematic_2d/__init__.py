# pylint: disable=invalid-name
"""
The 2D prescribed-flow framework used here can be traced back to the work
of [Kessler 1969 (section 3C)](https://doi.org/10.1007/978-1-935704-36-2_1), can be found also in
[Szumowski et al. 1998 (Atmos. Res.)](https://doi.org/10.1016/S0169-8095(97)00082-3).

The setup mimics a stratiform cloud deck and features periodic horizontal boundary condition
and vanishing flow at vertical boundaries.
It was introduced in [Morrison & Grabowski 2007](https://doi.org/10.1175/JAS3980) and later adopted
for particle-based simulations in [Arabas et al. 2015](https://doi.org/10.5194/gmd-8-1677-2015)
It uses a non-devergent single-eddy flow field resulting in an updraft-downdraft pair in the domain.
The flow field advects two scalar fields in an Eulerian way: water vapour mixing ratio
and dry-air potential temperature.
"""

from .gui_settings import GUISettings
from .mpdata_2d import MPDATA_2D
from .simulation import Simulation
from .storage import Storage
