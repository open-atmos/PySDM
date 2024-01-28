"""
box-model and 2D prescribed-flow immersion-freezing examples based on
[Arabas et al. 2023](https://arxiv.org/abs/2308.05015)
"""

from .make_particulator import make_particulator
from .plots import make_freezing_spec_plot, make_pdf_plot, make_temperature_plot
from .run_simulation import run_simulation
